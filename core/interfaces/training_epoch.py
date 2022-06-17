import os
import logging
import numpy as np
import torch

from configs.configs_wrapper import IODFConfig, DPConfig
from core.utils.dp_scripts import *
from core.utils.dp_utils import get_filters_deactivated
from core.utils.logger import get_time, save_states
from core.loss.lasso import add_lasso_grad
from core.interfaces.evaluation import evaluate


def train_epoch(epoch, model, opt, train_loader, cfg: IODFConfig):
    model.train()
    train_loss = np.zeros(len(train_loader))
    train_bpd = np.zeros(len(train_loader))

    num_data = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, *cfg.input_size)  # integers in [0, 255]

        opt.zero_grad()
        loss, bpd, _, _, z, _, _ = model(data.to(cfg.device))

        loss = torch.mean(loss)
        bpd = torch.mean(bpd).item()

        z_max = z.max().item() * 256
        z_min = z.min().item() * 256

        loss.backward()
        loss = loss.item()

        train_loss[batch_idx] = loss
        train_bpd[batch_idx] = bpd

        opt.step()

        num_data += len(data)

        if batch_idx % (cfg.log_interval) == 0:
            perc = 100. * batch_idx / len(train_loader)

            logging.info(f'{get_time()} Epoch: {epoch:3d} [{num_data:5d}/{len(train_loader.sampler):5d} ({perc:2.0f}%)] \tLoss: {loss:11.6f}\tbpd: {bpd:8.6f} z_max: {z_max:.2f} z_min: {z_min:.2f}') 

    return train_loss.mean(), train_bpd.mean()


def train_epoch_pruning(epoch, iteration, model, opt, train_loader, cfg:IODFConfig, dp_config:DPConfig, compactor_params):
    model.train()
    train_loss = np.zeros(len(train_loader))
    train_bpd = np.zeros(len(train_loader))

    num_data = 0

    for batch_idx, (data, _) in enumerate(train_loader):

        data = data.view(-1, *cfg.input_size)  # integers in [0, 255]
        loss, bpd, _, _, _, _, _ = model(data.to(cfg.device))
        loss = torch.mean(loss)
        bpd = torch.mean(bpd).item()

        loss.backward()
        loss = loss.item()
        train_loss[batch_idx] = loss
        train_bpd[batch_idx] = bpd

        if iteration >= dp_config.before_mask_iters:

            total_iters_in_compactor_phase = iteration - dp_config.before_mask_iters

            if total_iters_in_compactor_phase % dp_config.mask_interval == 0:
                filters_deactivated, layer_metric_dict = get_filters_deactivated(model, cfg.prune_mode)
                s1, s2 = pruning_msg(filters_deactivated, layer_metric_dict, iteration, cfg.prune_mode, cfg.dataset)
                logging.info(s1)
                logging.info(s2)

        add_lasso_grad(compactor_params, cfg, dp_config)

        opt.step() 
        opt.zero_grad()

        num_data += len(data)

        if batch_idx % cfg.log_interval == 0:
            perc = 100. * batch_idx / len(train_loader)

            logging.info(f'{get_time()} Epoch: {epoch:2d} [{num_data:5d}/{len(train_loader.sampler):7d} ({perc:2.0f}%)] \tLoss: {loss:7.2f}\tbpd: {bpd:6.5f}')
        
        iteration += 1
        
    return train_loss.mean(), train_bpd.mean(), iteration