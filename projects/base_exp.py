import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from threading import Lock

import numpy as np

from lib import WarmCosineLRScheduler
from utils import setup_logger, setup_tensorboard_logger


class BaseExp:
    """Basic class for experiment, store variable for exp need.
    
    Required:
        exp_name(str): Loading project name.

        batch_size(int):
        max_epoch(int):
        lr(float):

        model_config(dict): 用于构造模型所需要的参数

        train_transforms(list): 用于构造训练数据的数据增强
        test_transforms(list): 用于构造验证数据的数据增强
    Optional:
        checkpoint_file: 用于恢复训练
        save_folder_name(str): save folder for checkpoint, log, etc.

        warmup_epochs(int): 预热步数
        warmup_lr(float): 起始的预热学习率
        end_lr(float): 最终学习率(非预热阶段)
    """

    def __init__(self, exp_name, save_folder_name, batch_size, max_epoch, lr, args:dict):
        # ------------------------------------- 校验必备数据 ------------------------------ #
        try:
            self.exp_name = exp_name
            self.save_folder_name = save_folder_name

            self.batch_size = batch_size
            self.max_epoch = max_epoch
            self.lr = lr
            
            for k,v in args.items():
                setattr(self, k, v)
        except Exception as e:
            raise KeyError("init exp error, missing required value: {}".format(e))
        
        # ------------------------------------- 创建日志环境 ------------------------------ #
        os.makedirs("exp/{}".format(self.save_folder_name), exist_ok=True)
        os.makedirs("exp/{}/checkpoint/".format(self.save_folder_name), exist_ok=True)
        os.makedirs("exp/{}/log/".format(self.save_folder_name), exist_ok=True)
        os.makedirs("exp/{}/tensorboard/".format(self.save_folder_name), exist_ok=True)
        
        self.tb_writer = setup_tensorboard_logger("exp/{}/tensorboard/".format(self.save_folder_name))
        self.logger = setup_logger("exp/{}/log/".format(self.save_folder_name))

        # ------------------------------------- 恢复训练数据 ------------------------------ #
        if os.path.exists(self.checkpoint_file):
            self.logger.info("加载checkpoint...")
            self.checkpoint = torch.load(self.checkpoint_file, weights_only=True)
            self.get_model().cuda()
            self.get_optimizer()
            self.model.load_state_dict(self.checkpoint["model"])
            self.optimizer.load_state_dict(self.checkpoint["optimizer"])
            self.last_epoch = self.checkpoint["epoch"] + 1
        else:
            self.last_epoch = 0

    def get_model(self) -> nn.Module:
        raise NotImplementedError

    def calc_loss(self, model_output, target) -> torch.FloatTensor:
        raise NotImplementedError
    
    def run_eval(self, output, target) -> dict:
        """返回根据metric计算好的值, {metric_name: metric_value, ...}"""
        raise NotImplementedError

    def get_data_loader(self, train=True) -> DataLoader:
        raise NotImplementedError

    def get_optimizer(self) -> torch.optim.Optimizer:
        if "optimizer" not in self.__dict__:
            warmup_lr = getattr(self, "warmup_lr", 0)
            weight_decay = getattr(self, "weight_decay", 0)
            if warmup_lr > 0:
                lr = warmup_lr
            else:
                lr = self.lr

            self.optimizer = torch.optim.Adam(
                self.get_model().parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        return self.optimizer

    def get_lr_scheduler(self):
        warmup_epochs = getattr(self, "warmup_epochs", 0)
        warmup_lr = getattr(self, "warmup_lr", 0)
        end_lr = getattr(self, "end_lr", 1e-6)
        if "scheduler" not in self.__dict__:
            self.scheduler = WarmCosineLRScheduler(
                self.get_optimizer(),
                self.lr,
                len(self.get_data_loader()) * self.max_epoch,
                len(self.get_data_loader()) * warmup_epochs,
                warmup_lr,
                end_lr
            )
            self.scheduler.step(len(self.get_data_loader()) * self.last_epoch)
        return self.scheduler

    def data_preprocess(self, inputs, target):
        return inputs, target

    def save_checkpoint(self):
        checkpoint = {
            "epoch": self.last_epoch,
            "model": self.get_model().state_dict(),
            "optim": self.get_optimizer().state_dict(),
            "scheduler": self.get_lr_scheduler().state_dict(),
        }
        torch.save(checkpoint, "exp/{}/checkpoint/{}.pth".format(self.save_folder_name, self.last_epoch))