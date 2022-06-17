# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import logging

import torch

from configs.configuration import get_configs
from core.interfaces.model import load_model
from core.interfaces.eval_coding import evaluate_coding
from core.utils.logger import set_logger
from core.interfaces.data import load_data
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--nn_type', type=str)
    parser.add_argument('--quantize', action='store_true', default=False)
    parser.add_argument('--pruning', action='store_true', default=False)
    parser.add_argument('--pruned', action='store_true', default=False)
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no_decode', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:0') if not args.no_cuda else torch.device('cpu')
    log_dir = f"workspace/evaluation/{args.dataset}/{args.nn_type}-{args.resume}"
    os.makedirs(log_dir, exist_ok=True)
    set_logger(os.path.join(log_dir, 'evfaluate_coding.log'))

    cfg = get_configs(
        dataset=args.dataset,
        nn_type=args.nn_type,
        batch_size=args.batchsize,
        resume=args.resume,
        quantize=args.quantize, 
        pruning=args.pruning,
        pruned=args.pruned,
        no_decode=args.no_decode,
        device=device,
        log_dir=log_dir
    )
    
    logging.info(vars(cfg))

    train_loader, _, test_loader = load_data(cfg)

    model = load_model(cfg)
    model.to(cfg.device)

    # GPU warmup
    model.eval()
    with torch.no_grad():
        for x, _ in train_loader:
            _ = model(x.to(cfg.device))
            break
    print('GPU is warmed up.')
    
    a_bpd, c_bpd, error, N, ts = evaluate_coding(model, test_loader, cfg)
    ts = [t*1e3 for t in ts]
    logging.info(f"Coding test dataset {N} images, analytic bpd: {a_bpd:.3f}, coding bpd: {c_bpd:.3f}, sum of error: {error}")
    logging.info(f"Latency [ms/sample]: encode {ts[0]:.2f}, inference {ts[1]:.2f}, rans {ts[2]:.2f}, decode {ts[3]:.2f}")