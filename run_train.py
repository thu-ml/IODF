import os 
import argparse
import random
import logging

import torch
from torch.utils.tensorboard import SummaryWriter

from configs.configuration import get_configs, disp_dp_config
from core.interfaces.model import load_model
from core.interfaces.data import load_data
from core.interfaces.optim import get_optimizer_and_lr_scheduler
from core.utils.logger import set_logger, set_seed, get_log_path, get_tag
from core.utils.reproducibility import backup_codes
from tools.train import train, train_pruning

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train IODF')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--nn_type', type=str)
    parser.add_argument('--quantize', action='store_true', default=False)
    parser.add_argument('--pruning', action='store_true', default=False)
    parser.add_argument('--pruned', action='store_true', default=False)
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--resume', default=None, type=str)

    args = parser.parse_args()

    cfg = get_configs(
        dataset=args.dataset,
        nn_type=args.nn_type,
        pruning=args.pruning,
        quantize=args.quantize,
        pruned=args.pruned,
        batch_size=args.batchsize,
        resume=args.resume,
        device=torch.device('cuda:0')
    )

    if args.pruning:
        cfg, dp_config = cfg
    else:
        dp_config = None

    # random seed 
    if cfg.manual_seed is None:
        cfg.manual_seed = random.randint(1, 100000)
    set_seed(cfg.manual_seed)

    # logger and reproduction
    tag = get_tag(cfg)
    cfg.snap_dir = get_log_path(cfg.out_dir, cfg.dataset, tag)
    set_logger(os.path.join(cfg.snap_dir, 'train.log'))
    backup_codes(os.path.join(cfg.snap_dir, 'reproducibility'))

    if args.pruned:
        import shutil
        shutil.copyfile(src = os.path.join(cfg.resume, 'base_model_params.txt'), dst = os.path.join(cfg.snap_dir, 'checkpoints', 'base_model_params.txt'))

    #  prepare for training
    train_loader, val_loader, test_loader = load_data(cfg)

    model = load_model(cfg)

    # multiple gpu 
    cfg.multi_gpu = False
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        cfg.multi_gpu = True
        model = torch.nn.DataParallel(model, dim=0)
    model.to(cfg.device)

    optimizer, lr_scheduler = get_optimizer_and_lr_scheduler(model, cfg, dp_config)

    start_epoch = 1 if cfg.resume is None else lr_scheduler.state_dict()['last_epoch']

    writer = SummaryWriter(cfg.snap_dir)

    logging.info('='*20+'Model Setting'+'='*20)
    logging.info(vars(cfg))

    if args.pruning:
        logging.info('='*20+'Pruning Setting'+'='*20)
        logging.info(disp_dp_config(dp_config))
    
    if not cfg.pruning:
        train(
            model, 
            optimizer,
            lr_scheduler,
            writer, 
            start_epoch,
            train_loader,
            val_loader,
            test_loader,
            cfg
        )
    else:
        train_pruning(
            model, 
            optimizer, 
            lr_scheduler, 
            writer,
            start_epoch,
            train_loader, 
            val_loader,
            test_loader,
            cfg,
            dp_config
        )