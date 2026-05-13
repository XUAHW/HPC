import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskAdapter_DynamicThreshold(nn.Module):
    def __init__(self, alpha, mask_cam=False):
        super(MaskAdapter_DynamicThreshold, self).__init__()
        self.alpha = alpha
        self.mask_cam = mask_cam


    def forward(self, x):
        binary_mask = []
        for i in range(x.shape[0]):
            # 取第i个样本的最大值*比例作为阈值
            th = torch.max(x[i]) * self.alpha
            # 大于阈值的置为1, 小于阈值的置为0
            binary_mask.append(torch.where(x[i] >= th, torch.ones_like(x[0]), torch.zeros_like(x[0])))
        binary_mask = torch.stack(binary_mask, dim=0)

        if self.mask_cam:
            return x * binary_mask
        else:
            return binary_mask

class FeatureExtractor:
    def __init__(self, mask_adapter, clip_size=224):
        self.mask_adapter = mask_adapter
        self.clip_size = clip_size  
        
    @torch.cuda.amp.autocast()
    def extract_features(self, img_224, cam_224, cam_intersection_224_mask, cam_union_224_mask, label):
        batch_indices, class_indices = torch.where(label == 1)
        
        img_selected = img_224[batch_indices] 
        cam_selected = cam_224[batch_indices, class_indices] 

        mask_intersection_selected = cam_intersection_224_mask[batch_indices, class_indices]  # [N, 224, 224]
        mask_union_selected = cam_union_224_mask[batch_indices, class_indices]  # [N, 224, 224]
        
        # 增加维度
        cam_expanded = cam_selected.unsqueeze(1)  # [N, 1, 224, 224]
        mask_intersection_expanded = mask_intersection_selected.unsqueeze(1)  # [N, 1, 224, 224]
        mask_union_expanded = mask_union_selected.unsqueeze(1) # [N, 1, 224, 224]
        
        # 前景特征 乘cam的目的是在图片上对应的区域权重更高
        # CAM 中接近1的区域会保留图像像素，接近0的区域会变黑
        fg_features = cam_expanded * img_selected  # [N, 3, 224, 224]
        # 背景特征
        bg_features = (1 - cam_expanded) * img_selected  # [N, 3, 224, 224]
        
        # 前景mask
        fg_masks = mask_intersection_expanded   # [N, 1, 224, 224]
        # 背景mask
        bg_masks = 1 - mask_union_expanded  # [N, 1, 224, 224]
        
        return fg_features, bg_features, fg_masks, bg_masks
        
    @torch.cuda.amp.autocast()
    def get_masked_features(self, fg_features, bg_features, fg_masks, bg_masks, clip_model):

        # --- 前景图像预处理 ---
        # 找到每个前景图像、每个通道在空间维度(H, W)上的最小值, keepdim=True 保证输出形状是 [N, 3, 1, 1]，以便后续广播
        fg_min = fg_features.amin(dim=(2, 3), keepdim=True)
        # 找到每个前景图像、每个通道在空间维度(H, W)上的最大值, keepdim=True 保证输出形状是 [N, 3, 1, 1]，以便后续广播
        fg_max = fg_features.amax(dim=(2, 3), keepdim=True)
        # 对前景图像进行 Min-Max 归一化处理, 将每个通道的像素值缩放到 [0, 1] 区间，这是送入预训练模型的标准操作
        # 加上 1e-8 是为了防止 fg_max 和 fg_min 相等时出现除以零的错误
        normalized_fg_features = (fg_features - fg_min) / (fg_max - fg_min + 1e-8) # [N, 3, 224, 224]
        
        # --- 背景图像预处理 ---
        bg_min = bg_features.amin(dim=(2, 3), keepdim=True)
        bg_max = bg_features.amax(dim=(2, 3), keepdim=True)
        normalized_bg_features = (bg_features - bg_min) / (bg_max - bg_min + 1e-8)
        
        # 将 mask 与 加权后的图像 相乘
        fg_img_features = clip_model.vision_model(normalized_fg_features*fg_masks)
        bg_img_features = clip_model.vision_model(normalized_bg_features*bg_masks) 
        
        return fg_img_features, bg_img_features


    @torch.cuda.amp.autocast()
    def process_batch(self, inputs, cam1, cam2, cam3, cam4, label, clip_model):

        # 如果整个批次的标签中一个正样本(1)都没有, 则返回None
        if not torch.any(label == 1):
            return None
            
        # 上采样至[224, 224]
        cam1_224 = F.interpolate(cam1, (self.clip_size, self.clip_size), mode="bilinear", align_corners=True) 
        cam2_224 = F.interpolate(cam2, (self.clip_size, self.clip_size), mode="bilinear", align_corners=True) 
        cam3_224 = F.interpolate(cam3, (self.clip_size, self.clip_size), mode="bilinear", align_corners=True)
        cam4_224 = F.interpolate(cam4, (self.clip_size, self.clip_size), mode="bilinear", align_corners=True) 
        
        # 使用阈值处理, 将cam_224 -> 二值掩码
        cam1_224_mask = self.mask_adapter(cam1_224) 
        cam2_224_mask = self.mask_adapter(cam2_224) 
        cam3_224_mask = self.mask_adapter(cam3_224) 
        cam4_224_mask = self.mask_adapter(cam4_224) 

        # 通道取交集，找出置信的前景
        cam_intersection_224_mask = cam1_224_mask * cam2_224_mask * cam3_224_mask * cam4_224_mask 
        # 通道取并集，目的是1- cam_union_224_mask 可以得到最置信的背景
        # cam_union_224_mask = cam1_224_mask | cam2_224_mask | cam3_224_mask | cam4_224_mask 
        cam_union_224_mask = (cam1_224_mask.bool() | cam2_224_mask.bool() | cam3_224_mask.bool() | cam4_224_mask.bool()).float()
       

        fg_features, bg_features, fg_masks, bg_masks = self.extract_features(inputs, cam4_224, cam_intersection_224_mask, cam_union_224_mask, label)
        fg_features, bg_features = self.get_masked_features(fg_features, bg_features, fg_masks, bg_masks, clip_model)

        return {'fg_features': fg_features, 'bg_features': bg_features, 'fg_masks': fg_masks, 'bg_masks': bg_masks}
