# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
from argparse import Namespace
import os 

import torch
import numpy as np 

from configs.configuration import get_configs
from core.interfaces.model import load_model
from core.interfaces.data import load_data
from core.interfaces.eval_coding import evaluate_coding

def main(args, f):    

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    cfg = get_configs(
        dataset=args.dataset,
        nn_type=args.nn_type,
        batch_size=args.batchsize,
        resume=args.resume,
        quantize=args.quantize, 
        pruning=False,
        pruned=args.pruned,
        device=device,
        no_decode=args.no_decode,
        out_dir='assets'
    )

    model = load_model(cfg)

    model.to(cfg.device)
    _, _, test_loader = load_data(cfg)

    from core.utils.part_of_dataloader import part_of_dataloader

    test_loader = part_of_dataloader(test_loader, 30)

    # Warm up GPU
    with torch.no_grad():
        for idx, (data, _) in enumerate(test_loader):
            _ = model(data.to(cfg.device))
            if idx == 5:
                break

    a_bpd, c_bpd, error, N, ts = evaluate_coding(model, test_loader, cfg)

    ts = [t * 1e3 for t in ts]

    print(f'== {cfg.resume.split("/")[2]} == batch size: {cfg.batch_size} == over {N} samples', file=f)
    print(f'   Latency [ms/sample]: encode {ts[0]:.2f}, inference {ts[1]:.2f}, rans {ts[2]:.2f}, decode {ts[3]:.2f}', file=f)

if __name__ == '__main__':

    bs = [4,8,16,32,64]


    args = Namespace(dataset='imagenet32', nn_type='densenet', quantize=False, pruned=False, resume='base'),

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    args.no_decode = True
    for bb in bs:
        args.batchsize = bb
        with open('assets/coding_latency.txt', 'a+') as f:
            main(args, f)