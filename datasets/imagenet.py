# -*- encoding: utf-8 -*-
'''
@File    :   imagenet.py
@Time    :   2025/01/27 00:53:25
@Author  :   crab 
@Version :   1.0
@Desc    :   只加载用来做无监督预训练
'''
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import os
from io import BytesIO


class ImageNet(Dataset):
    def __init__(self, img_size, transform = None, transform_k = None, 
                 root = "", memory_cache = False):
        self.num_classes = 1000
        self.memory_cache = memory_cache
        if root == "":
            root = "data/imagenet/"
        self.root = root
        
        if transform is None:
            self.transform = torchvision.transforms.Compose(
                [
                    transforms.RandomResizedCrop(img_size, scale=(0.5, 1)),
                    transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        self.transform_k = transform_k

        def get_image_dirs(root_dir):
            image_paths = []
            for dirpath, _, filenames in os.walk(root_dir):
                for filename in filenames:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(dirpath, filename))
            return image_paths
        self.img_dirs = get_image_dirs(self.root)

        def load_compressed_image(path):
            with open(path, "rb") as f:
                return BytesIO(f.read())  # 返回压缩的字节流
        self.img_list: list[Image.Image] = []
        if self.memory_cache:
            for img_path in self.img_dirs:
                self.img_list.append(load_compressed_image(img_path))
            self.img_dirs = self.img_list

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, index: int):
        img_path = self.img_dirs[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img1 = self.transform(img.copy())
        if self.transform_k is not None:
            img2 = self.transform_k(img.copy())
            return [img1, img2], 0
        return img1, 0
