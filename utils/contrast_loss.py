import torch
import torch.nn as nn

class InfoNCELossFG(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, fg_img_feature, fg_pro_feature, bg_pro_feature):
        positive_sims = torch.tensor(0., requires_grad=True, device=fg_img_feature.device)
        negative_sims = torch.tensor(0., requires_grad=True, device=fg_img_feature.device)

        # 对前景图像特征进行 L2 归一化，使其成为单位向量, 归一化后，向量点积的结果就等于余弦相似度
        fg_img_feature = fg_img_feature / fg_img_feature.norm(dim=-1, keepdim=True) 
        
        batch_size = fg_img_feature.shape[0]

        # 循环遍历批次中的每一个样本来计算损失，而不是一次性进行矩阵运算
        for i in range(batch_size):
            # 提取当前样本的锚点特征。切片 [i:i+1] 是为了保持形状
            curr_fg_img = fg_img_feature[i:i+1]  
            # 提取当前样本的正样本原型特征
            curr_fg_pro = fg_pro_feature[i:i+1]  
            # 提取当前样本的所有负样本原型特征
            curr_bg_pro = bg_pro_feature[i]  
            
            # 计算锚点与正样本的相似度
            fg_img_fg_pro_logits = curr_fg_img @ curr_fg_pro.t() 
            # 计算锚点与所有负样本的相似度
            fg_img_bg_pro_logits = curr_fg_img @ curr_bg_pro.t() 
            
            # InfoNCE损失的分子是 e^(正样本相似度 / 温度), 进行累加
            positive_sims = positive_sims + torch.exp(fg_img_fg_pro_logits / self.temperature).sum()
            
            # InfoNCE损失的分母是 e^(正样本相似度 / 温度) + Σ e^(所有负样本相似度 / 温度), 进行累加
            negative_sims = negative_sims + torch.exp(fg_img_fg_pro_logits / self.temperature).sum() + torch.exp(fg_img_bg_pro_logits / self.temperature).sum()
        
        # 根据InfoNCE公式计算最终的损失值 公式为 -log(分子 / 分母)
        loss = -torch.log(positive_sims / negative_sims)

        return loss



class InfoNCELossBG(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, bg_img_feature, fg_pro_feature, bg_pro_feature):

        positive_sims = torch.tensor(0., requires_grad=True, device=bg_img_feature.device)
        negative_sims = torch.tensor(0., requires_grad=True, device=bg_img_feature.device)

        # 对背景图像特征进行 L2 归一化，使其成为单位向量, 归一化后，向量点积的结果就等于余弦相似度
        bg_img_feature = bg_img_feature / bg_img_feature.norm(dim=-1, keepdim=True)  
        
        batch_size = bg_img_feature.shape[0]
        for i in range(batch_size):
            # 提取当前样本的锚点特征。切片 [i:i+1] 是为了保持形状
            curr_bg_img = bg_img_feature[i:i+1]  
            # 提取当前样本的正样本原型特征
            curr_fg_pro = fg_pro_feature[i:i+1] 
            # 提取当前样本的所有负样本原型特征
            curr_bg_pro = bg_pro_feature[i]  

            # 计算锚点与负样本的相似度
            bg_img_bg_pro_logits = curr_bg_img @ curr_bg_pro.t()  
            # 计算锚点和前景原型的相似度
            bg_img_fg_pro_logits = curr_bg_img @ curr_fg_pro.t()  

            # InfoNCE损失的分子是 e^(锚点与负样本的相似度 / 温度), 进行累加
            positive_sims = positive_sims + torch.exp(bg_img_bg_pro_logits / self.temperature).mean()
            # InfoNCE损失的分母是 e^(锚点与负样本的相似度 / 温度) + Σ e^(锚点和前景原型的相似度 / 温度), 进行累加
            negative_sims = negative_sims + torch.exp(bg_img_bg_pro_logits / self.temperature).mean() + torch.exp(bg_img_fg_pro_logits / self.temperature).sum()

        # 根据InfoNCE公式计算最终的损失值 公式为 -log(分子 / 分母)
        loss = -torch.log(positive_sims / negative_sims)

        return loss
