from argparse import Namespace
import os
import importlib

from core.utils.logger import get_tag

def get_configs(
    dataset: str,
    nn_type: str,
    batch_size: int,
    quantize: bool,
    pruned: bool, 
    resume: str = None,
    **kwargs
):
    config_module = importlib.import_module(name=f"configs.{nn_type}.{dataset}")
    configs = config_module.config

    configs.batch_size = batch_size
    configs.pruned = pruned
    configs.quantize = quantize

    configs.log_interval = configs.n_train_samples // (batch_size * 5)

    tag = get_tag(configs)
    configs.resume = None if resume is None else \
        os.path.join(configs.out_dir, dataset.lower(), tag + '-' + resume, 'checkpoints')
    configs = Namespace(**vars(configs), **kwargs)
    return configs