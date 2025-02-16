import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from base_exp import BaseExp

from models.mae import mae_vit_tiny_patch16
from datasets import ImageNet, Solarization, GaussianBlur

class Exp(BaseExp):
    def __init__(self, args={}):
        # ------------------------------- Basic Config ------------------------------ #
        self.checkpoint_file = ""
        self.exp_name = "moco"
        self.save_folder_name = "mae_vit_tiny_imnet_pretrain"
        self.data_root = "data/imagenet-mini/train"
        self.memory_cache = False

        self.batch_size = 64
        self.max_epoch = 300
        self.lr = 1e-4
        
        # ------------------------------- Model Config ------------------------------ #
        self.img_size = 224

        # ------------------------------- Train Config ------------------------------ #
        self.warmup_lr = 1e-6
        self.warmup_epochs = 30
        self.end_lr = 1e-5
        self.weight_decay = 1e-3

        super(Exp, self).__init__(self.exp_name, self.save_folder_name, 
                                       self.batch_size, self.max_epoch, self.lr, args)
    
    def get_model(self):
        if "model" not in self.__dict__:
            self.model = mae_vit_tiny_patch16()
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        return self.model
    
    def get_data_loader(self, train=True) -> DataLoader:
        if train:
            if "train_dataloader" not in self.__dict__:
                dataset = ImageNet((self.img_size, self.img_size), root=self.data_root, memory_cache=self.memory_cache)
                train_sampler = DistributedSampler(dataset)
                self.train_dataloader = DataLoader(dataset, batch_size = self.batch_size, 
                                                   sampler=train_sampler, pin_memory=True)
            return self.train_dataloader

    def calc_loss(self, model_output, target):
        return model_output[0]

    def data_preprocess(self, inputs, target):
        inputs = inputs.cuda()
        # inputs = [img.cuda() for img in inputs]
        return inputs, target

    def run_eval(self, output, target):
        pass