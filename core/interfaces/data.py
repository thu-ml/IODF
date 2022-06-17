from core.datasets.cifar10 import load_cifar10
from core.datasets.imagenet import load_imagenet32, load_imagenet64

def load_data(configs):
    if configs.dataset == 'cifar10':
        return load_cifar10(configs)
    elif configs.dataset == 'imagenet32':
        return load_imagenet32(configs)
    elif configs.dataset == 'imagenet64':
        return load_imagenet64(configs)
    else:
        raise NotImplementedError