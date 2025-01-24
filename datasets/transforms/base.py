import random
import torchvision.transforms as transforms
from PIL import ImageFilter, ImageOps, Image


__all__ = ["ToRGB", "Solarization", "GaussianBlur"]

class ToRGB:
    def __call__(self, x: Image.Image) -> Image.Image:
        return x.convert("RGB")

class Solarization(object):
    def __call__(self, x: Image.Image) -> Image.Image:
        return ImageOps.solarize(x)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x: Image.Image) -> Image.Image:
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x