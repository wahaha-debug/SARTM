import os
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
import glob
import einops
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from semseg.augmentations_mm import get_train_augmentation

from matplotlib import pyplot as plt
import torchvision.transforms.functional as F


class FMB(Dataset):
    """
    num_classes: 15
    """
    CLASSES = ['background', 'Road', 'Sidewalk', 'Building',
               'Lamp', 'Sign', 'Vegetation', 'Sky', 'Person',
               'Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle', 'Pole']

    PALETTE = torch.tensor([        [0, 0, 0],  # background  0
        [228, 228, 179],  # Road 1
        [133, 57, 181],  # Sidewalk 2
        [177, 162, 67],  # Building 3
        [50, 178, 200],  # Lamp 4
        [199, 45, 132],  # Sign 5
        [84, 172, 66],  # Vegetation 6
        [79, 73, 179],  # Sky 7
        [166, 99, 76],  # Person 8
        [253, 121, 66],  # Car 9
        [91, 165, 137],  # Truck 10
        [152, 97, 155],  # Bus 11
        [140, 153, 105],  # Motorcycle 12
        [158, 215, 222],  # Bicycle 13
        [90, 113, 135],  # Pole 14
            ])

    def __init__(self, root: str = '../../data/FMB', split: str = 'train', transform=None, modals=['img', 'thermal'],
                 case=None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.root = root
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.files = self._get_file_names(split)

        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        item_name = str(self.files[index])

        rgb = os.path.join(*[self.root, 'Visible', item_name + '.png'])
        x1 = os.path.join(*[self.root, 'Infrared', item_name + '.png'])
        lbl_path = os.path.join(*[self.root, 'Label', item_name + '.png'])
        sample = {}
        sample['img'] = io.read_image(rgb)[:3, ...]
        if 'thermal' in self.modals:
            sample['thermal'] = self._open_img(x1)

        label = io.read_image(lbl_path)[0, ...].unsqueeze(0)
        sample['mask'] = label

        if self.transform:
            sample = self.transform(sample)

        label = sample['mask']
        del sample['mask']

        label = self.encode(label.squeeze().numpy()).long()
        sample = [sample[k] for k in self.modals]

        return sample, label

    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def encode(self, label: Tensor) -> Tensor:
        return torch.from_numpy(label)

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = os.path.join(self.root, 'test.txt') if split_name == 'val' else os.path.join(self.root, 'train.txt')
        file_names = []
        with open(source) as f:
            files = f.readlines()
        for item in files:
            file_name = item.strip()
            if ' ' in file_name:
                file_name = file_name.split(' ')[0]
            file_names.append(file_name)
        return file_names

if __name__ == '__main__':
    traintransform = get_train_augmentation((480, 640), seg_fill=255)

    trainset = FMB(transform=traintransform)
    trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=True, pin_memory=False)

    for i, (sample, lbl) in enumerate(trainloader):
        print(torch.unique(lbl))
