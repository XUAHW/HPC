import numpy as np
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import cv2 as cv
import re


class LUADTrainingDataset(Dataset):
    # TE-肿瘤上皮组织 NEC-坏死组织 LYN-淋巴细胞组织 TAS-肿瘤相关基质组织 BACK-背景
    CLASSES = ["TE", "NEC", "LYM", "TAS", "BACK"]
    def __init__(self, img_root="/data/users/huming/Datasets/LUAD-HistoSeg/train/img", transform=None):
        super(LUADTrainingDataset, self).__init__()
        self.get_images_and_labels(img_root)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        cls_label = self.cls_labels[index]
        assert os.path.exists(img_path), "img_path: {} does not exists".format(img_path)

        img = cv.imread(img_path, cv.IMREAD_UNCHANGED)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return os.path.basename(img_path), img, cls_label, 0

    def get_images_and_labels(self, img_root=None):
        self.img_paths = []
        self.cls_labels = []

        self.img_paths = glob.glob(os.path.join(img_root, "*.png"))
        for img_path in self.img_paths:
            term_split = re.split("\[|\]", img_path)
            clean_label_str = term_split[1].replace(" ", "")
            cls_label = np.array([int(x) for x in clean_label_str])
            self.cls_labels.append(cls_label)


class LUADTestDataset(Dataset):
    # TE-肿瘤上皮组织 NEC-坏死组织 LYN-淋巴细胞组织 TAS-肿瘤相关基质组织 BACK-背景
    CLASSES = ["TE", "NEC", "LYM", "TAS", "BACK"]
    def __init__(self, img_root="/data/users/huming/Datasets/LUAD-HistoSeg/", split="test", transform=None):
        assert split in ["test", "valid"], "split must be one of [test, valid]"
        super(LUADTestDataset, self).__init__()
        self.get_images_and_labels(img_root, split)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        assert os.path.exists(img_path), "img_path: {} does not exist".format(img_path)
        assert os.path.exists(mask_path), "mask_path: {} does not exist".format(mask_path)

        img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        mask = np.array(Image.open(mask_path))
        cls_label = np.array([0, 0, 0, 0])
        x = np.unique(mask) if np.unique(mask)[-1] != 4 else np.unique(mask)[:-1]
        cls_label[x] = 1

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        return os.path.basename(img_path), img, cls_label, mask

    def get_images_and_labels(self, img_root=None, split=None):
        self.img_paths = []
        self.mask_paths = []

        self.mask_paths = glob.glob(os.path.join(img_root, split, "mask", "*.png"))

        for mask_path in self.mask_paths:
            img_name = os.path.basename(mask_path)
            self.img_paths.append(os.path.join(img_root, split, "img", img_name))
