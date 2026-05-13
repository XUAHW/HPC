import argparse
import datetime
import os
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from utils.train_utils import cal_eta, get_cls_dataset
from model.cls_network.model import ClsNetwork
from utils.optimizer import PolyWarmupAdamW
from utils.pyutils import set_seed, setup_logger
from utils.fgbg_feature import FeatureExtractor, MaskAdapter_DynamicThreshold
from utils.hierarchical_utils import pair_features
from utils.contrast_loss import InfoNCELossFG, InfoNCELossBG
from utils.validate import validate, generate_cam
import logging
from utils.hc_loss import HCLoss
from utils.prototype_update import update_prototypes_from_folders
from model.cls_network.wss_model import WSSModel


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 开始时间
start_time = datetime.datetime.now()

# 创建解析器对象
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--gpu", type=int, default=0, help="gpu id")
parser.add_argument("--l6", type=float)
parser.add_argument("--warmup_epoch", type=int)
args = parser.parse_args()

def compute_multitask_ce_loss(logits, labels, loss_fn):
    # 多任务二分类交叉熵
    # logits: [B, 4, 2]，4个父类分别做二分类；labels: [B, 4]，每个父类取0/1
    labels = labels.long()
    loss = 0.0
    # 逐类计算 CE 后累加，避免把4个父类混成一个互斥多分类问题
    for class_idx in range(logits.shape[1]):
        loss = loss + loss_fn(logits[:, class_idx, :], labels[:, class_idx])
    return loss

# 训练函数
def train(cfg):
    # 将num_workers设为 CPU核心数 和 10 之间的较小值，防止 worker 过多导致内存和 I/O 压力过大
    num_workers = min(10, os.cpu_count())
    # 根据参数选择gpu
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    logging.info(f"Using device: {device}")

    # 设置随机数种子
    set_seed(3407)
    
    # 加载MedClip图像编码器，训练阶段只用于提取前景/背景区域特征，不更新其参数
    clip_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    clip_model.from_pretrained()
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    # 记录此时的时间
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    # 不同数据集配置不同，这里统一整理给 WSSModel 和 HC 使用
    if cfg.dataset.name == 'luad':
        # LUAD: TE/NEC/TAS 作为 Head 类，LYM 作为 Tail 类
        class_names = ["Tumor Epithelium", "Necrosis", "Lymphocyte", "Tumor-Associated Stroma"]
        hc_config = {
            'head_ids': [0, 1, 3],
            'tail_ids': [2],
            'class_map': {'TE': 0, 'NEC': 1, 'LYM': 2, 'TAS': 3},
            'proto_path': '/data/users/huming/Datasets/LUAD-HistoSeg/train/proto/LUAD/'
        }
    elif cfg.dataset.name == 'bcss':
        # BCSS: 4个类别样本相对均衡，这里全部作为 Head 类参与原型更新
        class_names = ["Tumor", "Stroma", "Lymphocytic infiltrate", "Necrosis"]
        hc_config = {
            'head_ids': [0, 1, 2, 3],
            'tail_ids': [],
            'class_map': {'TUM': 0, 'STR': 1, 'LYM': 2, 'NEC': 3},
            'proto_path': '/data/users/huming/Datasets/BCSS-WSSS/train/proto/BCSS/'
        }


    # ================================================ Prepare Model ================================================
    # 基础分类网络，输出多尺度分类结果和对应 CAM
    model = ClsNetwork(backbone=cfg.model.backbone.config,
                    cls_num_classes=cfg.dataset.cls_num_classes,
                    stride=cfg.model.backbone.stride,
                    pretrained=cfg.train.pretrained,
                    n_ratio=cfg.model.n_ratio,
                    l_fea_path=cfg.model.label_feature_path)
    
    # WSSModel 将分类网络、MedCLIP和可学习文本提示封装在一起
    # class_names 用于构造文本提示，n_ctx 表示可学习上下文 token 数量，ctx_init 是初始提示词
    wss_model = WSSModel(visual_net=model, medclip_full=clip_model, class_names=class_names, n_ctx=16, ctx_init="a photo of a")
    wss_model.to(device)
    wss_model.train()

    # ================================================ Prepare Data ================================================
    logging.info("Preparing Datasets")
    # split="valid" 返回训练集和验证集，训练过程中每个 epoch 后在验证集上选最优 CAM 权重
    train_dataset, val_dataset = get_cls_dataset(cfg, split="valid")
    
    # 训练集 DataLoader
    # shuffle=True 用于打乱训练样本；pin_memory/prefetch/persistent_workers 用于减少 GPU 等数据的时间
    train_loader = DataLoader(train_dataset,
                            batch_size=cfg.train.samples_per_gpu,
                            num_workers=num_workers,
                            pin_memory=True,
                            shuffle=True,
                            prefetch_factor=2,
                            persistent_workers=True)
    
    # 验证集 DataLoader
    # batch_size=1 是为了和 CAM/IoU 验证逻辑保持一致，验证时不打乱样本顺序
    val_loader = DataLoader(val_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True,
                          persistent_workers=True)
    
    # 混合精度和训练步数设置
    iters_per_epoch = len(train_loader) # 每个 epoch 的迭代步数（批次数）
    cfg.train.max_iters = cfg.train.epoch * iters_per_epoch # 总训练步数 = epoch 数 × 每epoch步数
    cfg.train.eval_iters = iters_per_epoch  # 每经过一整epoch（这么多步）做一次评估
    cfg.scheduler.warmup_iter = cfg.scheduler.warmup_iter * iters_per_epoch # warmup 由“epoch数”转为“步数”
    scaler = torch.cuda.amp.GradScaler() # AMP 的动态缩放器（混合精度防止下溢）
    

    # 优化器设置
    # PolyWarmupAdamW 会先 warmup，再按 poly 策略衰减学习率
    # 当 cfg.scheduler.warmup_iter = 0 时，一开始就进行学习率衰减
    optimizer = PolyWarmupAdamW(
        params=wss_model.parameters(),
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )

    # 损失函数和特征提取器设置
    # CrossEntropyLoss 用于4个父类的二分类监督
    loss_function = nn.CrossEntropyLoss().to(device)
    # 动态阈值mask处理器，根据 CAM 自适应生成区域
    mask_adapter = MaskAdapter_DynamicThreshold(alpha=cfg.train.mask_adapter_alpha,)

    # 前景背景提取器，负责把输入图像和多尺度 CAM 转成可送入 MedCLIP 的区域特征
    feature_extractor = FeatureExtractor(mask_adapter=mask_adapter)

    # HC loss
    # 使用类别原型约束图像特征分布，head/tail 类别由 hc_config 控制
    hc_criterion = HCLoss(
        num_classes=4,
        feat_dim=512,
        num_sub_prototypes=3,
        head_class_ids=hc_config['head_ids'],
        tail_class_ids=hc_config['tail_ids']
    ).to(device)

    # 对比损失函数
    # 前景特征靠近对应类别文本特征，背景特征远离前景类别文本特征
    fg_loss_fn = InfoNCELossFG(temperature=1).to(device)
    bg_loss_fn = InfoNCELossBG(temperature=1).to(device)
    # 记录验证集上 fuse234 的最佳 mIoU，用于保存最优 CAM 权重
    best_fuse234_dice = 0.0
    best_model_path = os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth")

    logging.info("Starting Training")
    # 创建迭代器
    train_loader_iter = iter(train_loader)
    

    # ================================================ Train Model ================================================
    # 开始训练
    for n_iter in range(cfg.train.max_iters):
        try:
            # 手动维护迭代器，按 max_iters 控制总步数，而不是直接按 epoch 外层循环
            _, inputs, cls_labels, _ = next(train_loader_iter)
        except StopIteration:
            # 一个 epoch 数据取完后重建迭代器，继续进入下一个 epoch
            train_loader_iter = iter(train_loader)
            _, inputs, cls_labels, _ = next(train_loader_iter)

        inputs = inputs.to(device).float()
        cls_labels = cls_labels.to(device).long()
        
        # autocast 开启 AMP 前向计算，降低显存占用并加速训练
        with torch.cuda.amp.autocast():
            cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, f_map = wss_model(inputs)

            # 提取前景和背景特征
            # process_batch 会根据 cls_labels 只处理图像中存在的类别，并用 CAM 生成区域 mask
            batch_info = feature_extractor.process_batch(inputs, cam1, cam2, cam3, cam4, cls_labels, clip_model)
            # 前景/背景经过图像编码器后的特征
            fg_features, bg_features = batch_info['fg_features'], batch_info['bg_features']

            # 将图像区域特征和文本嵌入配对
            # fg_text 是对应类别文本特征，bg_text 是其他类别文本特征集合
            set_info = pair_features(fg_features, bg_features, l_fea, cls_labels)

            # 前景特征, 背景特征, 前景文本嵌入, 背景文本嵌入
            fg_features, bg_features, fg_pro, bg_pro = set_info['fg_features'], set_info['bg_features'], set_info['fg_text'], set_info['bg_text']
                
            # 对比损失
            # fg_loss 约束前景区域与对应文本更接近；bg_loss 约束背景区域与前景文本区分开
            fg_loss = fg_loss_fn(fg_features, fg_pro, bg_pro)
            bg_loss = bg_loss_fn(bg_features, fg_pro, bg_pro)
            
            # 多尺度分类损失
            # 四个尺度分别监督，最后按配置文件中的 l1-l4 加权求和
            loss1 = compute_multitask_ce_loss(cls1, cls_labels, loss_function)
            loss2 = compute_multitask_ce_loss(cls2, cls_labels, loss_function)
            loss3 = compute_multitask_ce_loss(cls3, cls_labels, loss_function)
            loss4 = compute_multitask_ce_loss(cls4, cls_labels, loss_function)
            cls_loss = cfg.train.l1 * loss1 + cfg.train.l2 * loss2 + cfg.train.l3 * loss3 + cfg.train.l4 * loss4

            # 总损失分两阶段
            # 1) warmup_epoch 以内：只使用分类损失 + 前景/背景对比损失 + CAM 稀疏约束
            # 2) warmup_epoch 之后：额外加入 HC 原型约束，避免训练初期原型不稳定
            if (n_iter + 1) / cfg.train.eval_iters <= args.warmup_epoch:
                # CAM 均值项用于轻微抑制 CAM 过度激活
                loss = cls_loss + (fg_loss + bg_loss*0.5 + 0.0005*torch.mean(cam4)) * cfg.train.l5
            else:
                l_hc = hc_criterion(f_map, cls_labels)
                # cfg.train.l6 控制 HC loss 在总损失中的权重
                loss = cls_loss + (fg_loss + bg_loss*0.5 + 0.0005*torch.mean(cam4)) * cfg.train.l5 + cfg.train.l6 * l_hc

        # 混合精度反向传播
        # set_to_none=True 可以减少显存写入，GradScaler 用于缩放 loss 防止半精度梯度下溢
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (n_iter + 1) % 100 == 0:
            # 计算已用时间和剩余时间
            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            # 获取当前的学习率
            cur_lr = optimizer.param_groups[0]['lr']
            
            # Synchronize for accurate timing measurement
            # 把 GPU 异步队列里的计算全部执行完，确保随后的耗时统计准确
            torch.cuda.synchronize()
            
            # cls4 是最后一个尺度的分类输出，argmax 后得到每个父类的0/1预测
            cls_pred4 = torch.argmax(cls4, dim=-1)
            # “所有标签都预测对的样本”占总样本数的百分比
            all_cls_acc4 = (cls_pred4 == cls_labels).all(dim=1).float().mean() * 100
            # 先算出每个类别的平均准确率，然后再对这些准确率求平均。它衡量的是模型在所有类别上的综合平均表现
            avg_cls_acc4 = ((cls_pred4 == cls_labels).float().mean(dim=0)).mean() * 100
            
            # 打印: 当前迭代/总迭代数 消耗时间/剩余时间 学习率 损失值 完全匹配准确率 平均类别准确率
            logging.info(
                f"Iter: {n_iter + 1}/{cfg.train.max_iters}; "
                f"Elapsed: {delta}; ETA: {eta}; "
                f"LR: {cur_lr:.3e}; Loss: {loss.item():.4f}; "
                f"Acc4: {all_cls_acc4:.2f}/{avg_cls_acc4:.2f}"
            )

        # 定期验证和保存模型
        # 当到了eval_iters时或者训练完成时进行验证
        if (n_iter + 1) % cfg.train.eval_iters == 0 or (n_iter + 1) == cfg.train.max_iters:
            # validate 同时返回分类准确率和 fuse234 的 CAM/伪标签质量指标
            val_all_acc4, val_avg_acc4, fuse234_score, val_cls_loss = validate(model=wss_model, data_loader=val_loader, cfg=cfg, cls_loss_func=loss_function, type='valid')   
            logging.info("Validation results:")
            logging.info(f"Val all acc4: {val_all_acc4:.6f}")
            logging.info(f"Val avg acc4: {val_avg_acc4:.6f}")
            logging.info(f"Fuse234 score: {fuse234_score}, mIOU: {fuse234_score[:-1].mean():.4f}")
            
            # 保存最好的模型
            # fuse234_score 最后一位通常是背景/汇总项，这里按前景类别 mIoU 选择 checkpoint
            if fuse234_score[:-1].mean() > best_fuse234_dice:
                best_fuse234_dice = fuse234_score[:-1].mean()
                
                # 保存最优模型和优化器状态，后面测试和生成 CAM 都使用该 checkpoint
                torch.save({"cfg": cfg, "iter": n_iter, "model": wss_model.state_dict(), "optimizer": optimizer.state_dict()}, best_model_path, _use_new_zipfile_serialization=True)
                logging.info(f"Saved best model with mIOU: {best_fuse234_dice:.4f}")

            # 更新图像原型
            # 每次验证后用 proto 文件夹中的样本刷新 HC 原型，使后续训练阶段的原型约束保持最新
            update_prototypes_from_folders(
                model=wss_model,
                proto_root=hc_config['proto_path'],
                class_map=hc_config['class_map'],
                hc_criterion=hc_criterion,
                device=device,
                dataset_name=cfg.dataset.name
            )

    # 训练循环结束后释放未使用的 CUDA 缓存，并记录总训练时间
    torch.cuda.empty_cache()
    end_time = datetime.datetime.now()
    total_training_time = end_time - start_time
    logging.info(f'Total training time: {total_training_time}')

    # 训练结束后切换到验证集最优权重，再进行测试与CAM生成
    if os.path.exists(best_model_path):
        logging.info(f"Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        wss_model.load_state_dict(checkpoint["model"])
        best_iter = checkpoint.get("iter", "unknown")
        logging.info(f"Best model loaded successfully! (Saved at iteration: {best_iter})")
    else:
        logging.info("Warning: Best model checkpoint not found, using current model state")
        logging.info(f"Expected path: {best_model_path}")

    logging.info("Preparing Test Dataset")

    # enable_rotation 和 p 设置为 False 和 0 的目的是此时 train 要生成伪标签，不需要再做数据增强
    train_dataset, test_dataset = get_cls_dataset(cfg, split="test",enable_rotation=False,p=0.0)
    
    # 测试数据loader
    # 测试时 batch_size=1，方便逐图生成 CAM 并统计 per-class IoU
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    logging.info("Testing on Test Dataset")
    
    # 在测试集上评估最优 checkpoint 的分类准确率和 CAM 质量
    test_all_acc4, test_avg_acc4, fuse234_score, test_cls_loss = validate(model=wss_model, data_loader=test_loader, cfg=cfg, cls_loss_func=loss_function, type='test')   

    logging.info("Testing results:")
    logging.info(f"Test all acc4: {test_all_acc4:.6f}")
    logging.info(f"Test avg acc4: {test_avg_acc4:.6f}")
    logging.info(f"Fuse234 score: {fuse234_score}, mIOU: {fuse234_score[:-1].mean():.4f}")
    
    logging.info("Per-class IoU scores:")
    for i, score in enumerate(fuse234_score[:-1]):
        logging.info(f"  Class {i}: {score:.6f}")
    
    # 训练数据Loader
    # 使用未增强的训练集逐张生成 CAM，作为后续分割阶段的伪标签输入
    train_cam_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    logging.info(f"Generating CAMs for all {len(train_dataset)} training samples...")
    
    # 生成伪标签
    # generate_cam 会把训练集 CAM 可视化/伪标签写到 cfg.work_dir.pred_dir
    generate_cam(model=wss_model, data_loader=train_cam_loader, cfg=cfg)
    
    logging.info("Files generated:")
    logging.info(f"  • Training CAM visualizations: {cfg.work_dir.pred_dir}/*.png")
    logging.info(f"  • Model checkpoint: {cfg.work_dir.ckpt_dir}/best_cam.pth")
    logging.info("="*80)
    
    
if __name__ == "__main__":
    # 读取config文件
    cfg = OmegaConf.load(args.config)
    # 设置本次实验的主工作目录
    # 默认以 config 所在目录作为实验输出根目录
    cfg.work_dir.dir = os.path.dirname(args.config)
    # 时间戳
    # 用于区分不同训练运行，避免 checkpoint 被覆盖
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    # 权重目录,伪标签目录和训练日志目录
    # ckpt_dir 带时间戳，每次运行单独保存 best_cam.pth
    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    # 指定不同数据集的伪标签输出目录
    # BCSS/LUAD 的伪标签目录固定到数据集路径，其他数据集使用 config 中的相对目录
    if cfg.dataset.name == "bcss":
        cfg.work_dir.pred_dir = "/data/users/huming/Datasets/BCSS-WSSS/train/pseudo_label"
    elif cfg.dataset.name == "luad":
        cfg.work_dir.pred_dir = "/data/users/huming/Datasets/LUAD-HistoSeg/train/pseudo_label"
    else:
        cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.train_log_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.train_log_dir)
    # 命令行传入的 l6 覆盖配置文件，便于批量实验时只改 DGPC 权重
    cfg.train.l6 = args.l6

    # 创建目录
    # exist_ok=True 保证重复运行同一实验配置时不会因为目录已存在而报错
    os.makedirs(cfg.work_dir.dir, exist_ok=True)
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.train_log_dir, exist_ok=True)

    # 日志文件写入 train_log_dir，同时记录命令行参数和完整配置，方便复现实验
    setup_logger(filename=os.path.join(cfg.work_dir.train_log_dir, timestamp + '.log'))
    logging.info(f"args: {args}")
    logging.info(f"configs: {cfg}")

    # 开始训练
    train(cfg=cfg)
