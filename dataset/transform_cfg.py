from __future__ import print_function

import numpy as np
from PIL import Image
import torchvision.transforms as transforms


mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
normalize = transforms.Normalize(mean=mean, std=std)


transform_A = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize
    ])
]

transform_A_test = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize
    ])
]


transforms_list = ['A', 'D']


transforms_options = {
    'A': transform_A,
}

transforms_test_options = {
    'A': transform_A_test,
}