import torch
import torch.nn as nn
import torch.nn.functional as F

class HCLoss(nn.Module):
    def __init__(self, 
                 num_classes=4, 
                 feat_dim=512, 
                 num_sub_prototypes=3, 
                 head_class_ids=[],      # 动态传入: LUAD=[0,1,3], BCSS=[0,1,2,3]
                 tail_class_ids=[],      # 动态传入: LUAD=[2], BCSS=[]
                 tau_sub=0.07, 
                 tau_cohe=0.1, 
                 lambda_sub=1, 
                 lambda_cohe=0.2):
        super(HCLoss, self).__init__()
        
        self.num_classes = num_classes
        self.head_class_ids = set(head_class_ids)
        self.tail_class_ids = set(tail_class_ids)
        self.tau_sub = tau_sub
        self.tau_cohe = tau_cohe
        self.lambda_sub = lambda_sub
        self.lambda_cohe = lambda_cohe

        # 原型 Buffer [K, M, Dim]
        self.register_buffer("prototypes", torch.randn(num_classes, num_sub_prototypes, feat_dim))
        # 初始化后立即进行 L2 归一化，保证计算余弦相似度时数值稳定
        self.prototypes = F.normalize(self.prototypes, p=2, dim=2)

    def forward(self, feature_map, labels):
        # 自动池化
        batch_size = feature_map.size(0)
        features = F.adaptive_avg_pool2d(feature_map, (1, 1)).view(batch_size, -1)
        # 进行 L2 归一化，保证计算余弦相似度时数值稳定
        z_a = F.normalize(features, p=2, dim=1)
        
        # 初始归一化后此处仍然归一化的原因是因为每次更新原型后，都需要再归一化一次
        prototypes = F.normalize(self.prototypes, p=2, dim=2)

        total_loss_sub = 0.0
        total_loss_cohe = 0.0

        # 计数器
        valid_samples = 0

        device = feature_map.device

        # 对batch中每个样本逐一处理
        for i in range(batch_size):
            current_label_indices = torch.nonzero(labels[i]).squeeze(1).tolist()
            # 如果解析出来的列表是空的（即 labels[i] 全是 0），说明这张图没有任何标签，跳过
            if not current_label_indices: continue
            # 计数器++
            valid_samples += 1
            # 取出当前这张图的特征向量
            sample_z = z_a[i]
            # P_sub 存放该图片所属类别的最近子原型（最像它的那个原型）
            # P_sibling 存放该图片所属类别的其他子原型（同类但不那么像的）
            # N_set 存放所有其他类别的原型
            P_sub, P_sibling, N_set = [], [], []

            for class_idx in range(self.num_classes):
                # 取出该类别的所有子原型 class_protos，其形状为 [M, Dim]（M 为子原型数量）
                class_protos = prototypes[class_idx]
                # 依次检查每一个可能的类别
                if class_idx in current_label_indices:
                    # === 正样本类 ===
                    if class_idx in self.tail_class_ids:
                        P_sub.append(class_protos[0])
                    else:
                        # 计算样本与该类所有子原型的相似度
                        sims = torch.matmul(class_protos, sample_z)
                        # 找到相似度最高的那个子原型的索引
                        closest_idx = torch.argmax(sims)
                        # 将这个“最像的子原型”放入 P_sub
                        P_sub.append(class_protos[closest_idx])

                        indices = torch.arange(class_protos.size(0), device=device)
                        sib_indices = indices[indices != closest_idx]
                        if len(sib_indices) > 0:
                            P_sibling.extend([class_protos[idx] for idx in sib_indices])
                else:
                    # === 负样本类 ===
                    if class_idx in self.tail_class_ids:
                        N_set.append(class_protos[0])
                    # 遍历该类的所有子原型 把它们一个不漏地全部加入负样本集
                    else:
                        for p in class_protos:
                            N_set.append(p)

            # --- 计算 Sub-class Loss ---
            if P_sub:
                # w0个512的tensor -> [w0,dim]
                t_P_sub = torch.stack(P_sub)
                # 计算点积并除于温度系数 [得分_肿瘤1, 得分_间质1]
                logits_pos = torch.matmul(t_P_sub, sample_z) / self.tau_sub
                
                # 先把正样本放进去
                den_list = [logits_pos]
                if P_sibling: den_list.append(torch.matmul(torch.stack(P_sibling), sample_z) / self.tau_sub)
                # 如果有负样本，算一下相似度放进去
                if N_set: den_list.append(torch.matmul(torch.stack(N_set), sample_z) / self.tau_sub)
                
                # logsumexp(x) = log(sum(exp(x))) 对分母进行处理
                log_Z = torch.logsumexp(torch.cat(den_list), dim=0)
                # -log(exp(a)/exp(a+b+c))=-(a-log(exp(a+b+c))) 
                # [Loss_肿瘤1, Loss_间质1] -> mean -> [loss]
                total_loss_sub += -(logits_pos - log_Z).mean()

            if P_sibling:
                # w1个512的tensor -> [w1,dim]
                t_P_sib = torch.stack(P_sibling)
                # 计算点积并除于温度系数 [得分_肿瘤2, 得分_间质2]
                logits_sib = torch.matmul(t_P_sib, sample_z) / self.tau_cohe
                
                # 分母不包含 P_sub
                den_list_cohe = [logits_sib]
                # 如果有负样本，算一下相似度放进去
                if N_set: den_list_cohe.append(torch.matmul(torch.stack(N_set), sample_z) / self.tau_cohe)
                # logsumexp(x) = log(sum(exp(x))) 对分母进行处理
                log_Z_cohe = torch.logsumexp(torch.cat(den_list_cohe), dim=0)
                # -log(exp(a)/exp(a+b+c))=-(a-log(exp(a+b+c))) 
                # [Loss_肿瘤2, Loss_间质2] -> mean -> [loss]
                total_loss_cohe += -(logits_sib - log_Z_cohe).mean()

        # 计算batch内的平均损失
        if valid_samples > 0:
            total_loss_sub /= valid_samples
            total_loss_cohe /= valid_samples

        # 返回加权损失
        return self.lambda_sub * total_loss_sub + self.lambda_cohe * total_loss_cohe
