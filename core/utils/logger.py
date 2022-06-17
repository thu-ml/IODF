import datetime
import os
import logging

def get_tag(cfg):
    q = 'nq' if not cfg.quantize else 'q'
    if cfg.pruned:
        p = 'p'
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
    
    os.makedirs(checkpoint_path)
    
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