from argparse import Namespace
from .base import base

config = Namespace(
    **vars(base),
    
    dataset='imagenet64',
    n_train_samples=1230000,
    data_path='../DATASETS/imagenet64x64',
    input_size=(3,64,64),
    num_workers=8,

    n_epochs=100,
    evaluate_interval_epochs=1
)

config.n_levels = 4
config.n_channels_list = [512, 512, 512, 512]