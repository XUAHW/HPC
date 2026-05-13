import os
import numpy as np
import torch
import torch.nn.functional as F
import ttach as tta
from tqdm import tqdm
from PIL import Image

from .evaluate import ConfusionMatrixAllClass
from .pyutils import AverageMeter
import cv2 as cv
from skimage import morphology


def compute_multitask_ce_loss(logits, labels, loss_fn):
    labels = labels.long()
    loss = 0.0
    for class_idx in range(logits.shape[1]):
        loss = loss + loss_fn(logits[:, class_idx, :], labels[:, class_idx])
    return loss

def get_seg_label(cams, inputs, label, cfg):
    with torch.no_grad():
        b, c, h, w = inputs.shape
        label = label.view(b, -1, 1, 1)
        # 将所有小于0的值置为0。CAM在计算过程中可能出现负值，但作为“激活热力图”，负值没有意义
        cams = torch.clamp(cams, min=0) 
        
        # Normalize CAMs to [0,1]
        # 对每个类别（channel）的CAM，在空间维度(H, W)上找到最大值和最小值。
        channel_max = cams.amax(dim=(2, 3), keepdims=True)
        channel_min = cams.amin(dim=(2, 3), keepdims=True)
        # 这是标准的min-max归一化，将每个CAM的数值范围缩放到 [0, 1] 之间
        cams = (cams - channel_min) / (channel_max - channel_min + 1e-6) 
        cams = cams * label 
        
        cams = F.interpolate(cams, size=(h, w), mode="bilinear", align_corners=True)
        # 在类别维度(dim=1)上取最大值。结果 cam_max 的维度是 [B, 1, 224, 224]。
        # 它代表了在每个像素点上，所有前景类别中最高的激活值。
        # 我们可以把它理解为这个像素属于“前景”的置信度。
        cam_max = cams.max(dim=1, keepdim=True)[0] 

        bg_cam = (1 - cam_max) ** 10 
        # 将4个前景类别的CAMs和1个背景类别的CAM在类别维度(dim=1)上拼接起来。
        cam_all = torch.cat([cams, bg_cam], dim=1) 

    # 是否加入背景IOU, luad时加入，bcss时不加入
    if cfg.dataset.name == "luad":
        return cam_all
    else:
        return cams



def validate(model=None, data_loader=None, cfg=None, cls_loss_func=None, type='valid'):
    model.eval()
    avg_meter = AverageMeter()
    fuse234_matrix = ConfusionMatrixAllClass(num_classes=cfg.dataset.cls_num_classes + 1)
    
    # Test-time augmentation setup
    # 生成6种图像: 原始图像, 稍暗的图像 (亮度 x 0.9), 稍亮的图像 (亮度 x 1.1), 水平翻转的图像, 水平翻转且稍暗的图像, 水平翻转且稍亮的图像
    tta_transform = tta.Compose([
        tta.HorizontalFlip(),
        tta.Multiply(factors=[0.9, 1.0, 1.1])
    ])

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader), ncols=100, ascii=" >="):
            # 图像、类别多标签、分割掩码
            name, inputs, cls_label, labels = data

            # 移动到gpu上
            inputs = inputs.to(next(model.parameters()).device).float()
            labels = labels.to(next(model.parameters()).device)
            cls_label = cls_label.to(next(model.parameters()).device).long()

            cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, f_map = model(inputs)

            # Multi-scale classification losses
            cls_loss1 = compute_multitask_ce_loss(cls1, cls_label, cls_loss_func)
            cls_loss2 = compute_multitask_ce_loss(cls2, cls_label, cls_loss_func)
            cls_loss3 = compute_multitask_ce_loss(cls3, cls_label, cls_loss_func)
            cls_loss4 = compute_multitask_ce_loss(cls4, cls_label, cls_loss_func)
            cls_loss = cfg.train.l1 * cls_loss1 + cfg.train.l2 * cls_loss2 + cfg.train.l3 * cls_loss3 + cfg.train.l4 * cls_loss4

            cls4 = torch.argmax(cls4, dim=-1) 
            all_cls_acc4 = (cls4 == cls_label).all(dim=1).float().sum() / cls4.shape[0] * 100
            # 先算出每个类别的平均准确率，然后再对这些准确率求平均。它衡量的是模型在所有类别上的综合平均表现
            avg_cls_acc4 = ((cls4 == cls_label).sum(dim=0) / cls4.shape[0]).mean() * 100
            # 使用 AverageMeter 记录数据
            avg_meter.add({"all_cls_acc4": all_cls_acc4, "avg_cls_acc4": avg_cls_acc4, "cls_loss": cls_loss})
            
            # Generate evaluation CAMs with TTA
            cams1 = []
            cams2 = []
            cams3 = []
            cams4 = []
            # 遍历所有预设的 TTA 变换方法
            for tta_trans in tta_transform:
                # 生成一张增强后的图像张量 augmented_tensor
                augmented_tensor = tta_trans.augment_image(inputs) 
        
                cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, f_map = model(augmented_tensor)


                # 优化并上采样
                cam1 = get_seg_label(cam1, augmented_tensor, cls_label, cfg).to(next(model.parameters()).device)
                # 逆变换
                cam1 = tta_trans.deaugment_mask(cam1).unsqueeze(dim=0) 
                cams1.append(cam1)
                # 优化并上采样
                cam2 = get_seg_label(cam2, augmented_tensor, cls_label, cfg).to(next(model.parameters()).device) 
                # 逆变换
                cam2 = tta_trans.deaugment_mask(cam2).unsqueeze(dim=0) 
                cams2.append(cam2)
                # 优化并上采样
                cam3 = get_seg_label(cam3, augmented_tensor, cls_label, cfg).to(next(model.parameters()).device) 
                # 逆变换
                cam3 = tta_trans.deaugment_mask(cam3).unsqueeze(dim=0) 
                cams3.append(cam3)
                # 优化并上采样
                cam4 = get_seg_label(cam4, augmented_tensor, cls_label, cfg).to(next(model.parameters()).device) 
                # 逆变换
                cam4 = tta_trans.deaugment_mask(cam4).unsqueeze(dim=0) 
                cams4.append(cam4)

            cams1 = torch.cat(cams1, dim=0).mean(dim=0) 
            cams2 = torch.cat(cams2, dim=0).mean(dim=0)
            cams3 = torch.cat(cams3, dim=0).mean(dim=0) 
            cams4 = torch.cat(cams4, dim=0).mean(dim=0) 

            # priori mask
            if cfg.dataset.name == "luad":
                img = cv.imread(os.path.join(cfg.dataset.val_root, type, 'img', name[0]), cv.IMREAD_UNCHANGED)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                ret, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
                binary = np.uint8(binary)
                dst = morphology.remove_small_objects(binary == 255, min_size=80, connectivity=1).astype(np.uint8)
                priori_bg_mask = (1 - dst).reshape(1, 1, img.shape[0], img.shape[1])
                priori_bg_mask = torch.from_numpy(priori_bg_mask).to(next(model.parameters()).device)
                cams1[:, :-1, :, :] *= priori_bg_mask
                cams2[:, :-1, :, :] *= priori_bg_mask
                cams3[:, :-1, :, :] *= priori_bg_mask
                cams4[:, :-1, :, :] *= priori_bg_mask

            # Fuse multi-scale predictions
            fuse234 = 0.2 * cams2 + 0.2 * cams3 + 0.6 * cams4 
            # 返回分数最高的那个类别的索引（0, 1, 2, 或 3）
            fuse_label234 = torch.argmax(fuse234, dim=1).to(next(model.parameters()).device) 
            
            # 每个batch叠加
            fuse234_matrix.update(labels.detach().clone(), fuse_label234.clone())

    # 取出分类指标并将平均指标器重置为0
    all_cls_acc4, avg_cls_acc4, cls_loss = avg_meter.pop('all_cls_acc4'), avg_meter.pop("avg_cls_acc4"), avg_meter.pop("cls_loss")
    # 取出交并比
    fuse234_score = fuse234_matrix.compute()[2] # 长度为5的list
    model.train()

    return all_cls_acc4, avg_cls_acc4, fuse234_score, cls_loss


# 创建伪标签
def generate_cam(model=None, data_loader=None, cfg=None, cls_loss_func=None):

    model.eval()

    # Test-time augmentation setup
    tta_transform = tta.Compose([
        tta.HorizontalFlip(),
        tta.Multiply(factors=[0.9, 1.0, 1.1])
    ])

    with torch.no_grad():
        for data in tqdm(data_loader,
                         total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data

            inputs = inputs.to(next(model.parameters()).device).float()
            labels = labels.to(next(model.parameters()).device)
            cls_label = cls_label.to(next(model.parameters()).device).long()

            # Generate evaluation CAMs with TTA
            cams1 = []
            cams2 = []
            cams3 = []
            cams4 = []
            # 遍历所有预设的 TTA 变换方法
            for tta_trans in tta_transform:
                # 生成一张增强后的图像张量 augmented_tensor
                augmented_tensor = tta_trans.augment_image(inputs) 
                cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, f_map = model(augmented_tensor)


                # 优化并上采样
                cam1 = get_seg_label(cam1, augmented_tensor, cls_label, cfg).to(next(model.parameters()).device) 
                # 逆变换
                cam1 = tta_trans.deaugment_mask(cam1).unsqueeze(dim=0) 
                cams1.append(cam1)
                # 优化并上采样
                cam2 = get_seg_label(cam2, augmented_tensor, cls_label, cfg).to(next(model.parameters()).device) 
                # 逆变换
                cam2 = tta_trans.deaugment_mask(cam2).unsqueeze(dim=0) 
                cams2.append(cam2)
                # 优化并上采样
                cam3 = get_seg_label(cam3, augmented_tensor, cls_label, cfg).to(next(model.parameters()).device) 
                # 逆变换
                cam3 = tta_trans.deaugment_mask(cam3).unsqueeze(dim=0) 
                cams3.append(cam3)
                # 优化并上采样
                cam4 = get_seg_label(cam4, augmented_tensor, cls_label, cfg).to(next(model.parameters()).device) 
                # 逆变换
                cam4 = tta_trans.deaugment_mask(cam4).unsqueeze(dim=0) 
                cams4.append(cam4)
            cams1 = torch.cat(cams1, dim=0).mean(dim=0) 
            cams2 = torch.cat(cams2, dim=0).mean(dim=0) 
            cams3 = torch.cat(cams3, dim=0).mean(dim=0) 
            cams4 = torch.cat(cams4, dim=0).mean(dim=0) 

            # priori mask
            if cfg.dataset.name == "luad":
                img = cv.imread(os.path.join(cfg.dataset.train_root, name[0]), cv.IMREAD_UNCHANGED)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                ret, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
                binary = np.uint8(binary)
                dst = morphology.remove_small_objects(binary == 255, min_size=80, connectivity=1).astype(np.uint8)
                priori_bg_mask = (1 - dst).reshape(1, 1, img.shape[0], img.shape[1])
                priori_bg_mask = torch.from_numpy(priori_bg_mask).to(next(model.parameters()).device)
                cams1[:, :-1, :, :] *= priori_bg_mask
                cams2[:, :-1, :, :] *= priori_bg_mask
                cams3[:, :-1, :, :] *= priori_bg_mask
                cams4[:, :-1, :, :] *= priori_bg_mask
            
            # bcss:0.2,0.2,0.6
            fuse234 = 0.2 * cams2 + 0.2 * cams3 + 0.6 * cams4
            output_fuse234 = torch.argmax(fuse234, dim=1).long()


            PALETTE = [
             [255, 0, 0] ,   # 红色
             [0, 255, 0],    # 绿色 
             [0,0,255],      # 蓝色
             [153, 0, 255],  # 紫色 
             [255, 255, 255],# 白色
             [0, 0, 0],]     # 黑色

            for i in range(len(output_fuse234)):
                pred_mask = Image.fromarray(output_fuse234[i].cpu().clone().squeeze().numpy().astype(np.uint8)).convert('P')
                flat_palette = [val for sublist in PALETTE for val in sublist]
                pred_mask.putpalette(flat_palette)
                pred_mask.save(os.path.join(cfg.work_dir.pred_dir, name[i]))
    model.train()
    return 
