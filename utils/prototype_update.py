import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import os
from torchvision import transforms
from PIL import Image
import cv2 as cv
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.train_utils import get_mean_std
import logging

class SingleClassDataset(data.Dataset):
    def __init__(self, root_dir, dataset_name):
        super().__init__()
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                            if f.lower().endswith(('.png', '.jpg'))]
        MEAN, STD = get_mean_std(dataset_name)
        # 根据train/val选择不同的transform
        transform = {
            "val": A.Compose([
                A.Normalize(MEAN, STD),
                ToTensorV2(transpose_mask=True),
            ]),
        }
        self.transform = transform["val"]
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        # 以不改变通道/位深的方式读取图像
        img = cv.imread(self.image_paths[idx], cv.IMREAD_UNCHANGED)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img

@torch.no_grad()
def update_prototypes_from_folders(model, proto_root, class_map, hc_criterion, device, dataset_name):
    model.eval()
    # 变量 num_sub 存储了每个类别有多少个子原型
    num_sub = hc_criterion.prototypes.size(1)
    # 创建一个全零张量 shape与prototypes相同
    new_protos = torch.zeros_like(hc_criterion.prototypes)
    
    logging.info(f"--> [HC] Updating Prototypes from {proto_root} ...")

    for name, idx in class_map.items():
        # 拼接路径  Datasets/LUAD-HistoSeg/train/proto/LUAD/name
        class_dir = os.path.join(proto_root, name)
        if not os.path.exists(class_dir): continue
        
        dataset = SingleClassDataset(class_dir, dataset_name)
        if len(dataset) == 0: continue
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, drop_last=False)

        features_list = []
        
        for imgs in loader:
            imgs = imgs.to(device)
            cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list, f_map = model(imgs)
            # 池化 + 归一化
            vec = torch.nn.functional.adaptive_avg_pool2d(f_map, (1, 1)).view(f_map.size(0), -1)
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            # 移动到cpu
            features_list.append(vec.cpu())
        
        # 聚合所有tensor 
        all_feats = torch.cat(features_list, dim=0)
        
        # 判定 Head vs Tail
        if idx in hc_criterion.head_class_ids:
            # 情况 A: 样本充足，正常聚类
            if all_feats.size(0) >= num_sub:
                kmeans = KMeans(n_clusters=num_sub, n_init=10).fit(all_feats.numpy())
                centroids = torch.from_numpy(kmeans.cluster_centers_)
            new_protos[idx] = centroids
        else:
            # Tail: Average
            centroid = all_feats.mean(dim=0, keepdim=True)
            new_protos[idx] = centroid.repeat(num_sub, 1) # 填满所有槽位

    new_protos = torch.nn.functional.normalize(new_protos, p=2, dim=2)
    hc_criterion.prototypes.data = new_protos.to(device)
    logging.info("--> [HC] Prototypes updated.")
    model.train()
