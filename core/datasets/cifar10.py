import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import math

class ToINT():
    def __call__(x):
        x *= 255
        x = x.to(torch.uint8)
        return x 

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def load_cifar10(args):
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Pad(int(math.ceil(32 * 0.05)), padding_mode='edge'),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            ToINT,
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            ToINT
        ]
    )

    train_val_set = datasets.CIFAR10(
        args.data_path, 
        train=True, 
        download=True, 
        transform=None
    )

    train_set = Subset(train_val_set, list(range(0, len(train_val_set) - 10000)))
    val_set = Subset(train_val_set, list(range(len(train_val_set) - 10000, len(train_val_set))))
    
    train_set = DatasetFromSubset(train_set, transform_train)
    val_set = DatasetFromSubset(val_set, transform_test)


    test_set = datasets.CIFAR10(
        args.data_path,
        train=False,
        download=True,
        transform=transform_test
    )
    train_loader = DataLoader(
        train_set, 
        batch_size = args.batch_size,
        shuffle=True,
        num_workers = args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size = args.batch_size,
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

def get_data(args):
    return load_cifar10(args)