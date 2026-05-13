import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets.bcss import BCSSTrainingDataset, BCSSTestDataset
from datasets.luad_histoseg import LUADTrainingDataset, LUADTestDataset


def cal_eta(time0, cur_iter, total_iter):
    # 获取当前时间
    time_now = datetime.datetime.now()
    # 去掉当前时间的微秒部分，方便显示/比较
    time_now = time_now.replace(microsecond=0)
    # 估算剩余进度比例：剩余迭代 / 已完成迭代（注意：cur_iter 为 0 时会报错）
    scale = (total_iter - cur_iter) / float(cur_iter)
    # 已经消耗的时间（当前时间 - 起始时间）
    delta = (time_now - time0)
    # 预计剩余时间 = 已消耗时间 * 比例
    eta = (delta * scale)
    # 预计完成时间 = 当前时间 + 剩余时间
    time_fin = time_now + eta
    # 将预计完成时间去掉微秒，再减去当前时间，得到“整秒”的剩余时间
    eta = time_fin.replace(microsecond=0) - time_now
    # 返回字符串形式的“已用时间”和“剩余时间”
    return str(delta), str(eta)

def get_cls_dataset(cfg, split="valid", p=0.5, enable_rotation=True):
    # 根据数据集获取均值与标准差
    MEAN, STD = get_mean_std(cfg.dataset.name)
    
    # 构建训练时的变换列表
    train_transforms = [
        A.Normalize(MEAN, STD),
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
    ]
    
    # 根据参数决定是否添加旋转
    if enable_rotation:
        train_transforms.append(A.RandomRotate90())
    
    # 加入ToTensor
    train_transforms.append(ToTensorV2(transpose_mask=True))
    
    # 根据train/val选择不同的transform
    transform = {
        "train": A.Compose(train_transforms),
        "val": A.Compose([
            A.Normalize(MEAN, STD),
            ToTensorV2(transpose_mask=True),
        ]),
    }

    if cfg.dataset.name == "bcss":
        train_dataset = BCSSTrainingDataset(cfg.dataset.train_root, transform=transform["train"])
        val_dataset = BCSSTestDataset(cfg.dataset.val_root, split, transform=transform["val"])
    elif cfg.dataset.name == "luad":
        train_dataset = LUADTrainingDataset(cfg.dataset.train_root, transform=transform["train"])
        val_dataset = LUADTestDataset(cfg.dataset.val_root, split, transform=transform["val"])


    return train_dataset, val_dataset


# 返回不同数据集的均值与标准差，用于数据归一化
def get_mean_std(dataset):
    if dataset == "luad":
        norm = [[0.69164956, 0.50502165, 0.73429191], [0.15521939, 0.22774934, 0.18846453]]
    elif dataset == "bcss":
        norm = [[0.66791496, 0.47791372, 0.70623304], [0.1736589,  0.22564577, 0.19820057]]
    return norm[0], norm[1]
