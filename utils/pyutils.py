import torch
import numpy as np
import random
import logging
import sys
import datetime
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            # 将传入的值 v 加到对应键的总和上（列表的第0个元素）
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            # 遍历所有传入的 keys，对每一个 k 计算其平均值，然后将所有结果收集到一个名为 v_list 的新列表中
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        # 如果是None，就遍历 __data 字典中的所有键, 将其重置
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        # 如果非None, 获取平均值后再重置
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v
        
def setup_logger(filename='test.log'):
    # 定义格式: 时间, 文件名, 日志级别, 日志消息
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    # 获取 Logger 对象
    logger = logging.getLogger()
    # 设置日志记录器的最低级别, DEBUG 级别的日志将会被忽略
    logger.setLevel(logging.INFO)

    # 创建一个“处理器”(Handler)，它的任务是将日志写入文件
    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    # 创建另一个处理器，它的任务是将日志输出到控制台（终端屏幕）
    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)
