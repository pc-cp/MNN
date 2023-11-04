from torchvision import datasets, transforms
from PIL import ImageFilter, Image, ImageOps
import numpy as np
import os
import sys
import random
from torch.utils.data import Dataset

class TwoCrop:
    def __init__(self, strong, weak):
        self.strong = strong
        self.weak = weak

    def __call__(self, img):
        im_1 = self.strong(img)
        im_2 = self.weak(img)

        return im_1, im_2

class STL10Pair(datasets.STL10):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False, weak_aug=None):
        super().__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        self.weak_aug = weak_aug

    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)

            if self.weak_aug is not None:
                pos_2 = self.weak_aug(img)
            else:
                pos_2 = self.transform(img)

        return ((pos_1, pos_2), target)

class CIFAR10Pair(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, weak_aug=None):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.weak_aug = weak_aug

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            pos_1 = self.transform(img)
            if self.weak_aug is not None:
                pos_2 = self.weak_aug(img)
            elif self.target_transform is not None:
                pos_2 = self.target_transform(img)
            else:
                pos_2 = self.transform(img)
        return ((pos_1, pos_2), target)

class CIFAR100Pair(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, weak_aug=None):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.weak_aug = weak_aug

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            pos_1 = self.transform(img)
            if self.weak_aug is not None:
                pos_2 = self.weak_aug(img)
            else:
                pos_2 = self.transform(img)
        return ((pos_1, pos_2), target)





