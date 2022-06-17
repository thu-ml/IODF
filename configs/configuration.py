from argparse import Namespace
import os
import importlib

from configs.configs_wrapper import DPConfig, IODFConfig
from core.utils.logger import get_tag

def get_configs(
    dataset: str,
    nn_type: str,
    batch_size: int,
    quantize: bool = False,
    pruning: bool = False, 
    pruned: bool = False,
    resume: str = None,
    **kwargs
):
    config_module = importlib.import_module(name=f"configs.{nn_type}.{dataset}")
    configs = config_module.config

    configs.batch_size = batch_size
    configs.pruning = pruning
    configs.quantize = quantize
    configs.pruned = pruned

    configs.log_interval = configs.n_train_samples // (batch_size * 5)
    
    if 'out_dir' in kwargs.keys():
        configs.out_dir = kwargs['out_dir']
        kwargs.pop('out_dir')

    tag = get_tag(configs)
    configs.resume = None if resume is None else \
        os.path.join(configs.out_dir, dataset.lower(), tag + '-' + resume, 'checkpoints')
    configs = Namespace(**vars(configs), **kwargs)
    if not pruning:
        return configs
    else:
        return configs, get_dp_configs(configs)

def get_dp_configs(cfg:IODFConfig) -> DPConfig:
    
    iter_one_epoch = cfg.n_train_samples // cfg.batch_size
    mask_interval = cfg.log_interval

    dp_config = DPConfig(
        weight_decay=0.0001,
        weight_decay_bias = 0.,
        momentum=0.9,
        compactor_momentum=0.99,
        lasso_strength=1,
        mask_interval=mask_interval,
        before_mask_iters=iter_one_epoch,  # start to log filter information.
        no_l2_keywords=['compactor']
    )
    
    return dp_config

def disp_dp_config(dp_config):
    d = dp_config._asdict()
    return d.items()