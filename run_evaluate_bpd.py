import argparse
import logging
import os
import torch
import numpy as np 

from configs.configuration import get_configs
from core.interfaces.model import load_model
from core.interfaces.data import load_data
from core.utils.logger import set_logger, get_tag
from tools.evaluate import evaluate_analytic_bpd

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate IODF')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--nn_type', type=str)
    parser.add_argument('--quantize', action='store_true', default=False)
    parser.add_argument('--pruning', action='store_true', default=False)
    parser.add_argument('--pruned', action='store_true', default=False)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--batchsize', type=int)

    args = parser.parse_args()    

    cfg = get_configs(
        dataset=args.dataset, 
        nn_type=args.nn_type,
        batch_size=args.batchsize,
        quantize=args.quantize,
        pruning=args.pruning,
        pruned=args.pruned,
        resume=args.resume,
        device=torch.device('cuda:0')
    )

    if args.pruning:
        cfg, _ = cfg
    
    log_dir = f"{cfg.out_dir}/evaluation/{args.dataset}/{get_tag(cfg)}-{args.resume}"
    os.makedirs(log_dir, exist_ok=True)
    set_logger(os.path.join(log_dir, 'evaluate_bpd.log'))
    cfg.log_dir = log_dir

    logging.info('='*20+'Model Setting'+'='*20)
    logging.info(vars(cfg))
    
    _, _, test_loader = load_data(cfg)

    model = load_model(cfg)

    model.to(cfg.device)

    test_bpds = evaluate_analytic_bpd(model, test_loader, cfg)
    logging.info(f'Evaluate over {len(test_bpds)} samples:')
    logging.info(f"Analytic bpd: {np.mean(test_bpds):.3f} on test dataset")