import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from base_exp import BaseExp
from models import resnet18
from datasets import CIFAR100
from utils import topkAcc

class Exp(BaseExp):
    def __init__(self, args:dict={}):
        # ------------------------------- Basic Config ------------------------------ #
        self.checkpoint_file = ""
        self.exp_name = "resnet"
        self.save_folder_name = "classify_resnet18_cifar_offical_pretrain"

        self.batch_size = 64
        self.max_epoch = 100
        self.lr = 1e-3
        
        # ------------------------------- Model Config ------------------------------ #
        self.img_size = 224

        # ------------------------------- Train Config ------------------------------ #
        self.warmup_lr = 1e-5
        self.warmup_epochs = 5
        self.end_lr = 1e-6
        self.weight_decay = 1e-3

        self.loss_func = nn.CrossEntropyLoss()
        super(Exp, self).__init__(self.exp_name, self.save_folder_name, 
                                       self.batch_size, self.max_epoch, self.lr, args)
    
    def get_model(self):
        if "model" not in self.__dict__:
            self.model = resnet18(num_classes=100, pretrained=True)
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        return self.model
    
    def get_data_loader(self, train=True) -> DataLoader:
        if train:
            if "train_dataloader" not in self.__dict__:
                dataset = CIFAR100((self.img_size, self.img_size))
                train_sampler = DistributedSampler(dataset)
                self.train_dataloader = DataLoader(dataset, batch_size = self.batch_size, sampler=train_sampler, 
                                                   pin_memory=True)
            return self.train_dataloader
        else:
            if "test_dataloader" not in self.__dict__:
                dataset = CIFAR100((self.img_size, self.img_size), train=False)
                test_sampler = DistributedSampler(dataset)
                self.test_dataloader = DataLoader(dataset, batch_size = self.batch_size, sampler=test_sampler, 
                                                  pin_memory=True)
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