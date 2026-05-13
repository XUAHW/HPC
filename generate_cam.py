from utils.validate import generate_cam
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils.train_utils import get_cls_dataset
import os
from model.cls_network.model import ClsNetwork
import torch
from model.cls_network.wss_model import WSSModel
from medclip import MedCLIPModel, MedCLIPVisionModelViT

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 创建解析器对象
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--gpu", type=int, default=0, help="gpu id")
args = parser.parse_args()


# 根据参数选择gpu
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# 读取config文件
cfg = OmegaConf.load(args.config)
cfg.work_dir.pred_dir = "/data/users/huming/Datasets/BCSS-WSSS/train/pseudo_label"
best_model_path = "/data/users/huming/project/wsss_medical/work_dirs/bcss/classification/checkpoints/2025-12-27-19-30/best_cam.pth"

if cfg.dataset.name == 'luad':
    class_names = ["Tumor Epithelium", "Necrosis", "Lymphocyte", "Tumor-Associated Stroma"]
    dgpc_config = {
        'head_ids': [0, 1, 3],
        'tail_ids': [2],
        'class_map': {'TE': 0, 'NEC': 1, 'LYM': 2, 'TAS': 3},
        'proto_path': '/data/users/huming/Datasets/LUAD-HistoSeg/train/proto/LUAD/'
    }
elif cfg.dataset.name == 'bcss':
    class_names = ["Tumor", "Stroma", "Lymphocytic infiltrate", "Necrosis"]
    # 全部 Head
    dgpc_config = {
        'head_ids': [0, 1, 2, 3],
        'tail_ids': [],
        'class_map': {'TUM': 0, 'STR': 1, 'LYM': 2, 'NEC': 3},
        'proto_path': '/data/users/huming/Datasets/BCSS-WSSS/train/proto/BCSS/'
    }

# 加载MedClip
clip_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
clip_model = clip_model.to(device)
clip_model.eval()

# 将num_workers设为 CPU核心数 和 10 之间的较小值
num_workers = min(10, os.cpu_count())  # Optimize worker count based on CPU cores

# enable_rotation 和 p 设置为False和0的目的是此时trian要生成伪标签，不需要再变换了
train_dataset, test_dataset = get_cls_dataset(cfg, split="test",enable_rotation=False,p=0.0)

# 训练数据Loader
train_cam_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

# 加载模型
model = ClsNetwork(backbone=cfg.model.backbone.config,
                cls_num_classes=cfg.dataset.cls_num_classes,
                stride=cfg.model.backbone.stride,
                pretrained=cfg.train.pretrained,
                n_ratio=cfg.model.n_ratio,
                l_fea_path=cfg.model.label_feature_path)

wss_model = WSSModel(visual_net=model, medclip_full=clip_model, class_names=class_names, n_ctx=16, ctx_init="a photo of a")
wss_model.to(device)
wss_model.train()

checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
wss_model.load_state_dict(checkpoint["model"])

# 生成伪标签
generate_cam(model=wss_model, data_loader=train_cam_loader, cfg=cfg)
