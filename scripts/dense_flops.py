import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

import torch
import torch.nn as nn

from configs.configuration import get_configs
from core.interfaces.model import load_model
from core.interfaces.data import load_data

if __name__ == '__main__':
    args = argparse.Namespace(
        nn_type = 'densenet',
        dataset = 'imagenet64'
    )

    cfg = get_configs(
        dataset = args.dataset,
        nn_type=args.nn_type,
        batch_size=100,
        resume=None,
        pruning=False,
        quantize=False, 
        device=torch.device('cuda:0')
    )

    model = load_model(cfg)
    model.to(cfg.device)
    _, _, test_loader = load_data(cfg)

    with torch.no_grad():
        for (data, _) in test_loader:
            model(data.to(cfg.device))
            break 

    flops = 0
    for child_mo in model.modules():
        if isinstance(child_mo, nn.Conv2d):
            flops += child_mo.flops
    # print(N, n)
    
    print(f'{flops/1e9} GFLOPs')