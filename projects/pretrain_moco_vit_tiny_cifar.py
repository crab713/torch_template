import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from base_exp import BaseExp

from models import MoCo_ViT, vit_tiny_patch16
from datasets import CIFAR100, ToRGB, Solarization, GaussianBlur

class Exp(BaseExp):
    def __init__(self, args={}):
        # ------------------------------- Basic Config ------------------------------ #
        self.checkpoint_file = ""
        self.exp_name = "moco"
        self.save_folder_name = "moco_vit_tiny_cifar_pretrain"

        self.batch_size = 128
        self.max_epoch = 50
        self.lr = 1e-3
        
        # ------------------------------- Model Config ------------------------------ #
        self.img_size = 224
        self.moco_dim = 256
        self.moco_mlp_dim = 4096

        # ------------------------------- Train Config ------------------------------ #
        self.warmup_lr = 1e-5
        self.warmup_epochs = 10
        self.end_lr = 1e-5
        self.weight_decay = 1e-3

        super(Exp, self).__init__(self.exp_name, self.save_folder_name, 
                                       self.batch_size, self.max_epoch, self.lr, args)
    
    def get_model(self):
        if "model" not in self.__dict__:
            backbone = vit_tiny_patch16
            self.model = MoCo_ViT(backbone, len(self.get_data_loader())*self.max_epoch, 
                                  dim=self.moco_dim, mlp_dim=self.moco_mlp_dim, T=0.2)
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        return self.model
    
    def get_data_loader(self, train=True) -> DataLoader:
        if train:
            if "train_dataloader" not in self.__dict__:
                transform_k = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1)),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([Solarization()], p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )
                dataset = CIFAR100((self.img_size, self.img_size), transform_k=transform_k)
                train_sampler = DistributedSampler(dataset)
                self.train_dataloader = DataLoader(dataset, batch_size = self.batch_size, 
                                                   sampler=train_sampler, pin_memory=True)
            return self.train_dataloader
        else:
            if "test_dataloader" not in self.__dict__:
                dataset = CIFAR100((self.img_size, self.img_size), train=False)
                test_sampler = DistributedSampler(dataset)
                self.test_dataloader = DataLoader(dataset, batch_size = self.batch_size, 
                                                  sampler=test_sampler, pin_memory=True)
            return self.test_dataloader

    def calc_loss(self, model_output, target):
        return model_output

    def data_preprocess(self, inputs, target):
        inputs = [img.cuda() for img in inputs]
        target = target.cuda()
        return inputs, target

    def run_eval(self, output, target):
        pass