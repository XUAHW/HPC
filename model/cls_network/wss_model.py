import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

class TextEncoderWrapper(nn.Module):
    def __init__(self, medclip_text_model):
        super().__init__()
        self.text_model = medclip_text_model
        self.hf_model = medclip_text_model.model 
        self.projection_head = medclip_text_model.projection_head

    def forward(self, inputs_embeds, attention_mask):
        outputs = self.hf_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        all_hidden_states = outputs.hidden_states
        target_indices = [1, 2, -1]
        selected_layers = [all_hidden_states[i] for i in target_indices]
        stacked_states = torch.stack(selected_layers, dim=0)

        token_output = stacked_states.mean(dim=0) 
        projected_tokens = self.projection_head(token_output) 
        projected_tokens = F.normalize(projected_tokens, dim=-1)

        token_mask = attention_mask.unsqueeze(-1).float()  # [n_cls, L, 1]
        pooled_output = (token_output * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1e-9)
        projected_pooled = self.projection_head(pooled_output)  # [n_cls, 512]
        projected_pooled = F.normalize(projected_pooled, dim=-1)

        return projected_tokens, projected_pooled


class PromptLearner(nn.Module):
    def __init__(self, class_names, text_model, n_ctx=16, dim=768, ctx_init="a photo of a"):
        super().__init__()
        n_cls = len(class_names)

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

        # 从预训练好的 BERT 模型中把第一层取出来 这是一个查表层，它负责把整数 ID（如 345）变成一个 768 维的向量
        embed_layer = text_model.model.get_input_embeddings()
        target_device = text_model.model.embeddings.word_embeddings.weight.device

        # 先创建一个全随机的容器 (作为保底，或者用于填补 n_ctx > template_len 的部分)
        ctx_vectors = torch.empty(n_ctx, dim)
        nn.init.normal_(ctx_vectors, std=0.02) # 正态分布初始化

        if ctx_init:
            init_tokenized = tokenizer(ctx_init, add_special_tokens=False, return_tensors="pt")
            # 句子对应的数字 ID 矩阵 [1, n_word] n_word:类别名称的最大分词长度
            init_ids = init_tokenized["input_ids"].to(target_device)
            
            # 查表获取 Embeddings
            with torch.no_grad():
                init_embeddings = embed_layer(init_ids) # [1, 序列长度, 768]
            
            # 填充到 ctx_vectors
            # 如果 n_ctx (16) > len_init (shape[1])，我们只用模板初始化前 shape[1] 个，后面保持随机
            # 如果 n_ctx < len_init，截断模板
            len_init = init_embeddings.shape[1]
            n_fill = min(n_ctx, len_init)
            
            # 复制权重
            ctx_vectors[:n_fill] = init_embeddings[0, :n_fill].cpu() # 移回 CPU 初始化 Parameter

        # 注册为可训练参数
        self.ctx = nn.Parameter(ctx_vectors) # 变成可训练参数 [n_ctx, dim]
        
        # 调用 HuggingFace 的分词器处理 class_names class_tokens 是一个字典对象
        class_tokens = tokenizer(class_names, padding=True, return_tensors="pt")
        # 单词对应的数字 ID 矩阵 [n_cls, n_word] n_word:类别名称的最大分词长度
        class_input_ids = class_tokens["input_ids"] 
        # 标记哪些是真词，哪些是补位的 [n_cls, n_word]
        class_mask = class_tokens["attention_mask"] 
        
        # 移动到对应设备上
        class_input_ids = class_input_ids.to(target_device)
        class_mask = class_mask.to(target_device)
        
        with torch.no_grad():
            class_embeddings = embed_layer(class_input_ids)

        # 注册 Buffer
        self.register_buffer("class_embeddings", class_embeddings) # [n_cls, n_word, 768]
        self.register_buffer("class_mask", class_mask)             # [n_cls, n_word]
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        
    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prompts = torch.cat([ctx, self.class_embeddings], dim=1)
        
        # 产生mask [n_cls, n_ctx]
        ctx_mask = torch.ones(self.n_cls, self.n_ctx, dtype=self.class_mask.dtype, device=self.class_mask.device)
        
        # 最终 mask: [n_cls, n_ctx + n_word]
        full_mask = torch.cat([ctx_mask, self.class_mask], dim=1)
        
        # 返回embedding和mask
        return prompts, full_mask


class WSSModel(nn.Module):
    def __init__(self, visual_net, medclip_full, class_names, n_ctx=16, ctx_init="a photo of a"):
        super().__init__()
        # backbone
        self.visual_net = visual_net
        # 获取medclip的text_encoder
        text_model_part = medclip_full.text_model
        
        # 由TextEncoderWrapper进行处理
        self.text_encoder = TextEncoderWrapper(text_model_part)
        
        # 冻结Text_Encoder全部参数
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        # 固定在评估态以关闭 dropout，避免训练/推理分布漂移
        self.text_encoder.eval()
            
        # 初始化 Learner
        self.prompt_learner = PromptLearner(class_names, self.text_encoder.text_model, n_ctx=n_ctx, dim=768, ctx_init=ctx_init)

    def train(self, mode=True):
        super().train(mode)
        # 始终让文本编码器保持评估态
        self.text_encoder.eval()
        return self

    def forward(self, image):
        prompts, attention_mask = self.prompt_learner()
        soft_text_tokens, soft_text_pooled = self.text_encoder(prompts, attention_mask)
        return self.visual_net(image, soft_text_tokens, soft_text_pooled, attention_mask)
