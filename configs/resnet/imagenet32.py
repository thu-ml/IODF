from argparse import Namespace
from .base import base

config = Namespace(
    **vars(base),
    
    dataset='imagenet32',
    n_train_samples=1230000,
    data_path='../DATASETS/imagenet32x32',
    input_size=(3,32,32),
    num_workers=8,

    n_epochs=100,
    evaluate_interval_epochs=1
)
