# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os 

import torch
import numpy as np 

from configs.configuration import get_configs
from core.interfaces.model import load_model
from core.interfaces.data import load_data
from tools.evaluate import evaluate_analytic_bpd

def main(args):    

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    cfg = get_configs(
        dataset=args.dataset,
        nn_type=args.nn_type,
        batch_size=args.batchsize,
        resume=args.resume,
        quantize=args.quantize, 
        pruning=args.pruning,
        pruned=args.pruned,
        device=device,
        out_dir='assets'
    )

    _, _, test_loader = load_data(cfg)

    model = load_model(cfg)

    model.to(cfg.device)

    from core.interfaces.init import disable_lsq
    from core.utils.part_of_dataloader import part_of_dataloader
    test_loader = part_of_dataloader(test_loader, 5)
    disable_lsq(model)
    
    a_bpd = evaluate_analytic_bpd(model, test_loader, cfg)

    with open(os.path.join(cfg.resume, '..', 'evaluate_a_bpd.tx'), 'w') as f:
        f.write(f'Evaluate over {len(a_bpd)} samples in test dataset. Analytic bpd: {np.mean(a_bpd):.3f}')

if __name__ == '__main__':

    args = argparse.Namespace(
        dataset = 'imagenet32',
        nn_type = 'resnet',
        quantize = True,
        pruning = False,
        pruned = True,
        batchsize = 1000,
        resume = 'IODF'
    )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(args)