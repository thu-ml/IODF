import numpy as np 
from PIL import Image

from core.datasets.cifar10 import load_cifar10
from core.datasets.imagenet import load_imagenet32, load_imagenet64

def load_data_array(configs):
    if configs.dataset == 'cifar10':
        _, _, test_loader =  load_cifar10(configs)
    elif configs.dataset == 'imagenet32':
        _, _, test_loader = load_imagenet32(configs)
    elif configs.dataset == 'imagenet64':
        _, _, test_loader = load_imagenet64(configs)
    else:
        raise NotImplementedError
    
    test_x = []
    for x, _ in test_loader:
        test_x.append(x.numpy())
    
    return None, test_x
