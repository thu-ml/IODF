from argparse import Namespace
from .base import base

config = Namespace(
    **vars(base),

    dataset='cifar10',
    n_train_samples=40000,
    data_path='../DATASETS/cifar10',
    input_size=(3,32,32),
    num_workers=8,

    n_epochs=2000,
    evaluate_interval_epochs=10
)