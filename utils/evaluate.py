import torch


class ConfusionMatrixAllClass(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat1 = None

    def update(self, a, b):
        """
        :param a: ground truth [B, 224, 224]
        :param b: pred [B, 224, 224]
        :return:
        """
        n = self.num_classes
        if self.mat1 is None:
            self.mat1 = torch.zeros((n, n), dtype=torch.int64, device=a.device)

            
        with torch.no_grad():
            k = (a >= 0) & (a < n) # [B, 224, 224]
            inds = n * a[k].to(torch.int64) + b[k].to(torch.int64) # [B*224*224]
            
            self.mat1 += torch.bincount(inds, minlength=n**2).reshape(n, n)
            del a
            del b

    def reset(self):
        if self.mat1 is not None:
            self.mat1.zero_()

    def compute(self):
        # 转化为浮点数
        h = self.mat1.float()
        # 全局像素准确率
        acc_global = torch.diag(h).sum() / h.sum()
        # 每个类别 的分类准确率，即预测为该类别且实际为该类别像素数量/类别 i 的真实像素总数 TP/(TP + FN)
        acc = torch.diag(h) / h.sum(1) # [N]
        # 交并比
        # torch.diag(h): 交集 (Intersection)。也就是每个类别的正确预测数 (TP)
        # h.sum(1): 每个类别的 真实像素总数 (TP + FN)
        # h.sum(0): 被预测为类别 i 的总像素数 (TP + FP)
        # (h.sum(1) + h.sum(0) - torch.diag(h)): 这就是 并集 (Union) 的计算
        # Union = (真实区域) + (预测区域) - (交集区域) = (TP + FN) + (TP + FP) - TP = TP + FN + FP
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h)) # [N]
        # Dice系数 
        # 2 * torch.diag(h): 2 * 交集 (2 * TP)
        # h.sum(1) + h.sum(0): (真实区域) + (预测区域) = (TP + FN) + (TP + FP)
        dice = 2 * torch.diag(h) / (h.sum(1) + h.sum(0)) # [N]
        return acc_global, acc, iu, dice

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat1)

    def __str__(self):
        acc_global, acc, iu, dice = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}\n'
            'dice: {}\n'
            'mean dice: {}\n').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu[:-1].mean().item() * 100,
                ['{:.1f}'.format(i) for i in (dice * 100).tolist()],
                dice[:-1].mean().item() * 100
            )
    


