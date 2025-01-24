import torchvision
from torchvision import transforms

from typing import Tuple, Any
from PIL import Image

from .transforms import GaussianBlur

class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, img_size, train=True, download=False, transform = None, transform_k = None, root = ""):
        if root == "":
            root = "data/cifar_100/"
        self.root = root
        if train is True:
            if transform is None:
                transform = torchvision.transforms.Compose(
                    [
                        transforms.RandomResizedCrop(img_size, scale=(0.8, 1)),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=1.0),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                    ]
                )
        else:
            transform = torchvision.transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ]
            )
        self.transform_k = transform_k
        super(CIFAR100, self).__init__(root=self.root, train=train, 
                                       transform=transform, download=download)
        self.num_classes = 100

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform_k is not None:
            img2 = self.transform_k(img)
            return [img1, img2], target
        return img1, target

if __name__ == '__main__':
    train_dataset = CIFAR100(train=True, download=True)
    test_dataest = CIFAR100(train=False, download=True)