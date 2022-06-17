import os 
import argparse
import logging

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

import torch

from configs.configuration import get_configs
from core.interfaces.model import load_model
from core.interfaces.data import load_data
from core.utils.logger import set_logger
from core.utils.dp_utils import * 
from core.utils.dp_scripts import *
from tools.evaluate import evaluate_analytic_bpd

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train Model')
    parser.add_argument('--dataset', type=str, default='imagenet32')
    parser.add_argument('--nn_type', type=str, default='resnet')
    parser.add_argument('--quantize', action='store_true', default=False)
    parser.add_argument('--batchsize', type=int, default=500)
    parser.add_argument('--pmode', default=None,type=str)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--epoch', default=None, type=int)

    args = parser.parse_args()

    cfg = get_configs(
        dataset=args.dataset,
        nn_type=args.nn_type,
        batch_size=args.batchsize,
        pruning=False,
        pruned=False,
        quantize=args.quantize,
        resume=None
    )

    pure_model = load_model(cfg)

    num_filters_origin = calculate_num_filters_of_model(pure_model)
    conv_flops_origin = calculate_conv_flops_of_model(pure_model, cfg.n_levels, cfg.n_flows, cfg.nn_depth, cfg.input_size[-1])

    cfg, dp_config = get_configs(
        dataset=args.dataset,
        nn_type=args.nn_type,
        batch_size=args.batchsize,
        pruning=True,
        pruned=False,
        quantize=args.quantize,
        resume=args.resume,
        device=torch.device('cuda:0')
    )   
    
    cfg.resume += f'/epoch{args.epoch}'

    if args.pmode is not None:
        cfg.prune_mode = args.pmode

    model = load_model(cfg)

    cfg.resume = cfg.resume.replace(f'/epoch{args.epoch}', '')

    remove_compactor = remove_fun(cfg.prune_mode)
    convert_model = convert_fun(cfg.prune_mode)

    params = remove_compactor(model, args.dataset)    
    save_path = cfg.resume.replace(args.resume, args.resume+f'_ep{args.epoch}_params').replace('q_dp', 'q_p')
    os.makedirs(save_path)
    torch.save(params, os.path.join(save_path, 'best.pth'))

    pure_model = convert_model(pure_model, os.path.join(save_path, 'best.pth'))

    num_filters_pruning = calculate_num_filters_of_model(pure_model)
    conv_flops_pruning = calculate_conv_flops_of_model(pure_model, cfg.n_levels, cfg.n_flows, cfg.nn_depth, cfg.input_size[-1])

    set_logger(os.path.join(save_path.replace('checkpoints', ''), 'pruning.log'))

    conv_cnt, sum_in_deps, sum_out_deps = 0, 0, 0
    for name, child_module in pure_model.named_modules():
        if isinstance(child_module, PadModule) or (isinstance(child_module, Identity) and 'shortcut' not in name):
            conv_cnt += 2
        if isinstance(child_module, nn.Conv2d):
            conv_cnt += 1 
            out_chn, in_chn, _, _ = child_module.weight.shape
            sum_in_deps += in_chn
            sum_out_deps += out_chn
            logging.info(f"Conv {conv_cnt:3d}: In {in_chn:3d}, Out {out_chn:3d}")

    logging.info(f"filters before and after pruning:  {num_filters_origin}, \t{num_filters_pruning}, " + \
        f"pruned out {(1. - num_filters_pruning / num_filters_origin) * 100.:.1f}%")

    logging.info(f"conv flops before and after pruning: {conv_flops_origin/1e9:.3f}G \t" +\
         f"{conv_flops_pruning/1e9:.3f}G, pruned out {(1. - conv_flops_pruning / conv_flops_origin) * 100.:.1f}%")

    train_loader, val_loader, test_loader = load_data(cfg)
    n_eval_samples = 50000
    from core.utils.part_of_dataloader import part_of_dataloader
    val_loader = part_of_dataloader(val_loader, 5)
    n_eval_samples = 5 * cfg.batch_size

    pure_model.to(cfg.device)
    model.to(cfg.device)

    val_bpd_masked = evaluate_analytic_bpd(model, val_loader, cfg)
    logging.info(f"Evaluating model with compactors over {n_eval_samples} samples: {val_bpd_masked:.4f}")

    val_bpd_pruned = evaluate_analytic_bpd(pure_model, val_loader, cfg)
    logging.info(f"Evaluating pruned model over {n_eval_samples} samples: {val_bpd_pruned:.4f}")

    pruned_init_path = save_path.replace(f'params', 'init')
    os.makedirs(pruned_init_path)
    torch.save(pure_model.state_dict(), os.path.join(pruned_init_path, 'best.pth'))
    with open(os.path.join(pruned_init_path, 'base_model_params.txt'), 'w') as f:
        f.write(os.path.join(save_path, 'best.pth'))