# -*- encoding: utf-8 -*-
'''
@File    :   classify_vit_tiny_cifar_moco_linear.py
@Time    :   2025/01/22 23:29:37
@Author  :   crab 
@Version :   1.0
@Desc    :   使用moco预训练的模型, 添加
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from base_exp import BaseExp
from models import vit_tiny_patch16
from datasets import CIFAR100
from utils import topkAcc

class Exp(BaseExp):
    def __init__(self, args:dict = {}):
        # ------------------------------- Basic Config ------------------------------ #
        self.checkpoint_file = ""
        self.exp_name = "classify_vit"
        self.save_folder_name = "classify_vit_tiny_imnet_moco_cifar_finetune"
        self.pretrain_ckpt = "exp/moco_vit_tiny_imnet_pretrain_ep100/checkpoint/99.pth" # moco预训练模型, 如果checkpoint_file为空就从这加载
        self.weights_prefix = "base_encoder."

        self.batch_size = 128
        self.max_epoch = 100
        self.lr = 1e-4
        
        # ------------------------------- Model Config ------------------------------ #
        self.img_size = 224

        # ------------------------------- Train Config ------------------------------ #
        self.warmup_lr = 1e-6
        self.warmup_epochs = 5
        self.end_lr = 1e-6
        self.weight_decay = 1e-3

        self.loss_func = nn.CrossEntropyLoss()
        super(Exp, self).__init__(self.exp_name, self.save_folder_name, 
                                       self.batch_size, self.max_epoch, self.lr, args)
        if self.checkpoint_file == "":
            self.set_model_weights(self.pretrain_ckpt)
    
    def get_model(self):
        if "model" not in self.__dict__:
            self.model = vit_tiny_patch16(num_classes=100)
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        return self.model
    
    def get_data_loader(self, train=True) -> DataLoader:
        if train:
            if "train_dataloader" not in self.__dict__:
                dataset = CIFAR100((self.img_size, self.img_size))
                train_sampler = DistributedSampler(dataset)
                self.train_dataloader = DataLoader(dataset, batch_size = self.batch_size, sampler=train_sampler, pin_memory=True)
            return self.train_dataloader
        else:
            if "test_dataloader" not in self.__dict__:
                dataset = CIFAR100((self.img_size, self.img_size), train=False)
                test_sampler = DistributedSampler(dataset)
                self.test_dataloader = DataLoader(dataset, batch_size = self.batch_size, sampler=test_sampler, pin_memory=True)
            return self.test_dataloader

    def calc_loss(self, model_output, target):
        loss = self.loss_func(model_output, target)
        return loss

    def data_preprocess(self, inputs, target):
        return inputs.cuda(), target.cuda()

    def run_eval(self, output:torch.FloatTensor, target:torch.LongTensor):
        """计算topk(acc1, acc5)"""
        acc_1, acc_5 = topkAcc(output, target, topk=(1,5))
        return {
            "acc_1": acc_1,
            "acc_5": acc_5
        }
    
    def set_model_weights(self, ckpt_path, map_location="cpu"):
        BLACK_LIST = ("head", )

        def _match(key):
            return any([k in key for k in BLACK_LIST])

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        weights_prefix = self.weights_prefix
        if not weights_prefix:
            state_dict = {k: v for k, v in ckpt["model"].items() if not _match(k)}
        else:
            if weights_prefix and not weights_prefix.endswith("."):
                weights_prefix += "."
            if all(key.startswith("module.") for key in ckpt["model"].keys()):
                weights_prefix = "module." + weights_prefix
            state_dict = {k.replace(weights_prefix, ""): v for k, v in ckpt["model"].items() if not _match(k)}
        msg = self.get_model().load_state_dict(state_dict, strict=False)
        return msg