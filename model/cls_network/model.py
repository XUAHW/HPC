import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from . import mix_transformer


class VariousReceptiveBranch(nn.Module):
    # 多感受野视觉增强分支。

    def __init__(self, in_channels, kernel_sizes=[3, 5, 7], reduction=4):
        super(VariousReceptiveBranch, self).__init__()

        self.multi_scale_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=k,
                padding=k // 2,
                groups=in_channels,
            )
            for k in kernel_sizes
        ])

        mid_channels = max(in_channels // reduction, 1)
        # 视觉门控分支：先压缩通道再恢复通道，输出范围为 [-1, 1]。
        self.visual_gate = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        spatial_maps = []
        for conv in self.multi_scale_convs:
            conv_out = conv(x)
            channel_mean = torch.mean(conv_out, dim=1, keepdim=True)
            spatial_maps.append(channel_mean)

        # 融合不同卷积核尺度的空间响应，并用 sigmoid 归一化为 [0, 1] 权重。
        sum_spatial_map = sum(spatial_maps)
        weight_map = torch.sigmoid(sum_spatial_map)
        enhanced_x = x * weight_map

        # 逐位置逐通道控制增强特征的回加比例。
        alpha = self.visual_gate(enhanced_x)
        out = x + alpha * enhanced_x
        return out


class AdaptiveLayer(nn.Module):

    def __init__(self, in_dim, n_ratio, out_dim, is_residual=False):
        super().__init__()
        hidden_dim = int(in_dim * n_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.is_residual = is_residual

        self.apply(self._init_weights)

        if self.is_residual:
            nn.init.constant_(self.fc2.weight, 0)
            if self.fc2.bias is not None:
                nn.init.constant_(self.fc2.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TextGuidedGatedFusion(nn.Module):

    def __init__(self, channels):
        super().__init__()
        # 1x1 卷积只做通道投影，不改变空间尺寸。
        self.text_gate_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.image_gate_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # 点积注意力缩放项，缓解通道维较大时 softmax 过尖锐。
        self.scale = channels ** -0.5

    def forward(self, image_feat, updated_feat, class_text_tokens, text_mask):
        b, c, h, w = image_feat.shape

        # 把图像特征展平为 HW 个 token，并归一化后与文本 token 做余弦相似度。
        query = image_feat.flatten(2).transpose(1, 2)
        query = F.normalize(query, dim=-1)

        # 只使用有效文本 token。若 mask 全无效，则回退到全部 token，避免空矩阵。
        valid_mask = text_mask.bool()
        key_value = class_text_tokens[valid_mask]
        if key_value.numel() == 0:
            key_value = class_text_tokens
        key_value = F.normalize(key_value, dim=-1)

        # 每个图像位置对当前类别文本 token 做 cross-attention。
        attn = torch.matmul(query, key_value.transpose(0, 1)) * self.scale 
        attn = F.softmax(attn, dim=-1)
        attended = torch.matmul(attn, key_value)  
        attended = attended.transpose(1, 2).reshape(b, c, h, w) 

        # 文本引导特征与原图像特征逐元素门控，保留同时被文本和图像激活的位置/通道。
        gated = self.text_gate_proj(attended) * self.image_gate_proj(image_feat)
        text_guided_feat = self.out_proj(gated) 
        fused_feat = updated_feat + text_guided_feat
        return fused_feat


class ClsNetwork(nn.Module):
    def __init__(
        self,
        backbone="mit_b1",
        cls_num_classes=4,
        stride=[4, 2, 2, 1],
        pretrained=True,
        n_ratio=0.5,
        l_fea_path=None,
    ):
        super().__init__()
        self.cls_num_classes = cls_num_classes
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        if pretrained:
            # 只加载 backbone 中存在的权重，并去掉 ImageNet 分类 head。
            state_dict = torch.load("./pretrained/" + backbone + ".pth", map_location="cpu")
            state_dict.pop("head.weight")
            state_dict.pop("head.bias")
            state_dict = {
                k: v for k, v in state_dict.items() if k in self.encoder.state_dict().keys()
            }
            self.encoder.load_state_dict(state_dict, strict=False)

        self.pooling = F.adaptive_avg_pool2d

        self.ms_branches = nn.ModuleList(
            [VariousReceptiveBranch(in_channels=c) for c in self.in_channels]
        )
        # stage_alpha 控制原始特征比例，stage_beta 控制增强分支比例。
        self.stage_alpha = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.0, dtype=torch.float32)) for _ in self.in_channels]
        )
        self.stage_beta = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1, dtype=torch.float32)) for _ in self.in_channels]
        )

        # 文本/原型适配器：512 维语义空间 -> 各视觉 stage 通道维。
        self.l_fc_anchor = nn.ModuleList(
            [AdaptiveLayer(512, n_ratio, c, is_residual=False) for c in self.in_channels]
        )
        self.l_fc_residual = nn.ModuleList(
            [AdaptiveLayer(512, n_ratio, c, is_residual=True) for c in self.in_channels]
        )

        # 每个 stage 一个文本引导融合模块；每个类别会复用同一个 stage 模块。
        self.text_fusion = nn.ModuleList(
            [TextGuidedGatedFusion(c) for c in self.in_channels]
        )
        # 每个 stage、每个类别各自一个 2 分类 head：
        # 输出维度 2 一般表示该类 absent/present。
        self.binary_heads = nn.ModuleList(
            [
                nn.ModuleList([nn.Linear(c, 2) for _ in range(self.cls_num_classes)])
                for c in self.in_channels
            ]
        )

        # 离线保存的 512 维硬原型/图像特征，按类别组织。
        with open("./features/image_features/{}.pkl".format(l_fea_path), "rb") as lf:
            info = pkl.load(lf)
            self.l_fea = info["features"].cpu()

        self.total_classes = cls_num_classes
        # CAM 相似度的可学习缩放系数，初始化为 CLIP 常用的 1/0.07。
        self.logit_scales = nn.ParameterList(
            [nn.Parameter(torch.ones([1]) * 1 / 0.07) for _ in self.in_channels]
        )

    def get_param_groups(self):

        regularized = []
        not_regularized = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]

    def _forward_stage(self, x, patch_embed, blocks, norm, stage_idx, attns):

        b = x.shape[0]
        x, h, w = patch_embed(x)
        for blk in blocks:
            x, attn = blk(x, h, w)
            attns.append(attn)
        x = norm(x)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        prev_feat = x
        vrb_out = self.ms_branches[stage_idx](prev_feat)  # [B, C, H, W]
        updated_feat = self.stage_alpha[stage_idx] * prev_feat + self.stage_beta[stage_idx] * vrb_out
        return prev_feat, updated_feat

    def _forward_backbone(self, x):

        prev_outs = []
        updated_outs = []
        attns = []

        prev_feat, x = self._forward_stage(
            x,
            self.encoder.patch_embed1,
            self.encoder.block1,
            self.encoder.norm1,
            0,
            attns,
        )
        prev_outs.append(prev_feat)
        updated_outs.append(x)

        prev_feat, x = self._forward_stage(
            x,
            self.encoder.patch_embed2,
            self.encoder.block2,
            self.encoder.norm2,
            1,
            attns,
        )
        prev_outs.append(prev_feat)
        updated_outs.append(x)

        prev_feat, x = self._forward_stage(
            x,
            self.encoder.patch_embed3,
            self.encoder.block3,
            self.encoder.norm3,
            2,
            attns,
        )
        prev_outs.append(prev_feat)
        updated_outs.append(x)

        prev_feat, x = self._forward_stage(
            x,
            self.encoder.patch_embed4,
            self.encoder.block4,
            self.encoder.norm4,
            3,
            attns,
        )
        prev_outs.append(prev_feat)
        updated_outs.append(x)

        return prev_outs, updated_outs, attns

    def _build_stage_text(self, stage_idx, soft_text_tokens, soft_text_pooled):

        hard_text = self.l_fea.to(soft_text_pooled.device)

        class_proto = self.l_fc_anchor[stage_idx](hard_text) + self.l_fc_residual[stage_idx](
            soft_text_pooled
        )  # [4, C]
        class_proto = F.normalize(class_proto, dim=-1)

        token_proj = self.l_fc_anchor[stage_idx](hard_text).unsqueeze(1) + self.l_fc_residual[
            stage_idx
        ](soft_text_tokens)  # [4, L, C]
        token_proj = F.normalize(token_proj, dim=-1)

        return hard_text, class_proto, token_proj

    def _stage_logits_and_cam(self, stage_idx, class_feature_maps, class_proto):
        # class_feature_maps: [B, 4, C, H, W]
        # class_proto: [4, C]
        b, num_classes, c, h, w = class_feature_maps.shape

        # 分类分支：每个类别的特征图先做全局平均池化，再进入该类别独立的二分类头。
        pooled = class_feature_maps.mean(dim=(-1, -2))  # [B, 4, C]
        logits = []
        for class_idx in range(num_classes):
            logits.append(self.binary_heads[stage_idx][class_idx](pooled[:, class_idx, :]))
        logits = torch.stack(logits, dim=1)  # [B, 4, 2]

        selected_cams = []
        logit_scale = self.logit_scales[stage_idx]
        for class_idx in range(num_classes):
            # CAM 分支：把该类别专属图像特征的每个空间位置与所有类别原型做相似度。
            class_feat = class_feature_maps[:, class_idx]  # [B, C, H, W]
            class_feat = class_feat.flatten(2).transpose(1, 2)  # [B, HW, C]
            class_feat = F.normalize(class_feat, dim=-1)
            sim = logit_scale * torch.matmul(class_feat, class_proto.t().float())  # [B, HW, 4]
            sim = sim.transpose(1, 2).reshape(b, num_classes, h, w)  # [B, 4, H, W]
            # 这里只保留“类别 i 专属特征图”对“类别 i 原型”的响应，
            # 最终拼成 [B, num_classes, H, W] 的多类 CAM。
            selected_cams.append(sim[:, class_idx : class_idx + 1, :, :])
        cams = torch.cat(selected_cams, dim=1)  # [B, 4, H, W]

        return logits, cams

    def _process_stage(
        self,
        stage_idx,
        prev_image_feat,
        updated_image_feat,
        soft_text_tokens,
        soft_text_pooled,
        text_mask,
    ):

        _, class_proto, token_proj = self._build_stage_text(
            stage_idx, soft_text_tokens, soft_text_pooled
        )

        class_features = []
        for class_idx in range(self.cls_num_classes):
            # 每个类别用自己的文本 token 与同一份图像特征融合，
            class_feature = self.text_fusion[stage_idx](
                image_feat=prev_image_feat,
                updated_feat=updated_image_feat,
                class_text_tokens=token_proj[class_idx],
                text_mask=text_mask[class_idx],
            )
            class_features.append(class_feature)

        class_feature_maps = torch.stack(class_features, dim=1)  # [B, 4, C, H, W]
        logits, cams = self._stage_logits_and_cam(stage_idx, class_feature_maps, class_proto)
        return logits, cams, class_feature_maps

    def forward(self, x, soft_text_tokens, soft_text_pooled, text_mask):
        prev_stage_features, updated_stage_features, _ = self._forward_backbone(x)

        cls_outputs = []
        cam_outputs = []
        class_feature_maps = []

        for stage_idx, (prev_stage_feat, updated_stage_feat) in enumerate(
            zip(prev_stage_features, updated_stage_features)
        ):
            cls_logits, cams, fused_feature_maps = self._process_stage(
                stage_idx,
                prev_stage_feat,
                updated_stage_feat,
                soft_text_tokens,
                soft_text_pooled,
                text_mask,
            )
            cls_outputs.append(cls_logits)  # [B, 4, 2]
            cam_outputs.append(cams)  # [B, 4, H_i, W_i]
            class_feature_maps.append(fused_feature_maps)

        l_fea_hard = self.l_fea.to(x.device)
        f_map = updated_stage_features[-1]  # [B, 512, 14, 14]

        return (
            cls_outputs[0],
            cam_outputs[0],
            cls_outputs[1],
            cam_outputs[1],
            cls_outputs[2],
            cam_outputs[2],
            cls_outputs[3],
            cam_outputs[3],
            l_fea_hard,
            f_map,
        )
