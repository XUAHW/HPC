import torch

def pair_features(fg_features, bg_features, text_features, labels):

    batch_indices, class_indices = torch.where(labels == 1)
    
    paired_fg_features = [] 
    paired_bg_features = [] 
    paired_fg_text = [] 
    paired_bg_text = [] 
 
    for i in range(len(batch_indices)):
        # 获取当前正样本对应的类别索引
        curr_class = class_indices[i]
        
        # 获取当前样本对应的前景背景特征
        curr_fg = fg_features[i] 
        curr_bg = bg_features[i]  
        
        # 获取对应类的文本特征
        curr_fg_text = text_features[curr_class]  
        
        bg_text_indices = [j for j in range(text_features.shape[0]) if j != curr_class]

        # 从 text_features 中批量取出所有不相关的文本特征
        curr_bg_text = text_features[bg_text_indices] 
        
        paired_fg_features.append(curr_fg)
        paired_bg_features.append(curr_bg)
        paired_fg_text.append(curr_fg_text)
        paired_bg_text.append(curr_bg_text)
    
    paired_fg_features = torch.stack(paired_fg_features)  
    paired_bg_features = torch.stack(paired_bg_features) 
    paired_fg_text = torch.stack(paired_fg_text)        
    paired_bg_text = torch.stack(paired_bg_text)        
    
    # 前景特征， 背景特征， 前景文本嵌入， 背景文本嵌入
    return {'fg_features': paired_fg_features, 'bg_features': paired_bg_features, 'fg_text': paired_fg_text, 'bg_text': paired_bg_text}
