import datetime
import os
import random

import torch
import numpy as np

import logging

def get_tag(cfg):
    q = 'nq' if not cfg.quantize else 'aq' if cfg.w_bits == 32 else 'awq'
    if cfg.pruned:
        p = 'p'
    elif cfg.pruning:
        p = 'dp'
    else:
        p = 'np'
    return f"{cfg.coupling_type}_{q}_{p}"

def get_time():
    return str(datetime.datetime.now())[0:19]

def get_log_path(workspace, dataset, tag):
    signature = get_time().replace(' ', '_').replace(':', '_').replace('-', '_')
    snapshots_path = f'{dataset.lower()}/{tag}-{signature}'
    snapshots_path = os.path.join(workspace, snapshots_path)
    checkpoint_path = os.path.join(snapshots_path, 'checkpoints')
    sample_path = os.path.join(snapshots_path, 'samples')
    
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(os.path.join(snapshots_path, 'metric_vectors'))
    
    return snapshots_path

def set_logger(log):
    level = getattr(logging, 'INFO', None)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(log)
    formatter = logging.Formatter('')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)


def save_states(model, optimizer=None, scheduler=None, cfg=None, directory=None):
    if directory is None:
        directory = os.path.join(cfg.snap_dir, 'checkpoints')
    torch.save(model.state_dict() if not cfg.multi_gpu else model.module.state_dict(), \
                os.path.join(directory, 'best.pth'))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(directory, 'optim.pth'))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(directory, 'lr_scheduler.pth'))

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)