import os 

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np

class npTotensor():
    def __call__(self, x):
        return torch.from_numpy(x).to(torch.uint8)

class ImageNetDataset(Dataset):
    def __init__(self, data, transform=None):
        # N C H W
        super().__init__()
        self.data = data
        self.transform = transform
    
    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        return x, torch.tensor([0])
    
    def __len__(self):
        return len(self.data)

def load_imagenet(args, resolution=32):
    
    trainpath = os.path.join(args.data_path, f'train_{resolution}x{resolution}.npy')
    testpath = os.path.join(args.data_path, f'val_{resolution}x{resolution}.npy')

    data_transform = transforms.Compose([
        npTotensor()
    ])

    n = 1280000
    n_val = 50000
    x_train_val = np.load(trainpath)[:n]
    x_train = x_train_val[:-n_val]
    x_val = x_train_val[-n_val:]
    x_test = np.load(testpath)[-1000:]
    print(f'Imagenet, {len(x_train)} images for train, {len(x_val)} images for validation, {len(x_test)} images for test.')

    trainset = ImageNetDataset(x_train, data_transform)
    valset = ImageNetDataset(x_val, data_transform)
    testset = ImageNetDataset(x_test, data_transform)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True)

    return train_loader, val_loader, test_loader


def load_imagenet32(args):
    return load_imagenet(args, 32)

def load_imagenet64(args):
    return load_imagenet(args, 64)