import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 

from configs.configs_wrapper import IODFConfig, DPConfig

def get_optimizer_and_lr_scheduler(model:nn.Module, configs:IODFConfig, dp_configs:DPConfig=None):
    if configs.pruning == False:
        optimizer = optim.Adamax(model.parameters(), lr=configs.learning_rate, eps=1.e-7)

        def lr_lambda(epoch):
            return min(1., (epoch+1)/configs.warmup) * np.power(configs.lr_decay, epoch)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

        # if configs.resume:
        #     optimizer.load_state_dict(torch.load(os.path.join(configs.resume, 'optim.pth')))
        #     scheduler.load_state_dict(torch.load(os.path.join(configs.resume, 'lr_scheduler.pth')))
            
        return optimizer, scheduler

    else:
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                print(f"{key} will not be put into optimizer.")
                continue
            lr = configs.learning_rate
            weight_decay = dp_configs.weight_decay

            if 'bias' in key:
                weight_decay = dp_configs.weight_decay_bias

            for kw in dp_configs.no_l2_keywords:
                if kw in key:
                    weight_decay=0
                    print('NOTICE! weight decay = 0 for ', key, 'because {} in {}'.format(kw, key))
                    break
            if 'bias' in key:
                apply_lr = 2 * lr
            else:
                apply_lr = lr
            
            if 'compactor' in key:
                use_momentum = dp_configs.compactor_momentum
                print('momentum {} for {}'.format(use_momentum, key))
            else:
                use_momentum = dp_configs.momentum
            params += [{"params": [value], "lr": apply_lr, "weight_decay": weight_decay, "beta1": use_momentum}]
        
        optimizer = optim.Adamax(params, lr)

        def lr_lambda(epoch):
            return min(1., (epoch+1)/configs.warmup) * np.power(configs.lr_decay, epoch)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

        # if configs.resume:
        #     optimizer.load_state_dict(torch.load(os.path.join(configs.resume, 'optim.pth')))
        #     scheduler.load_state_dict(torch.load(os.path.join(configs.resume, 'lr_scheduler.pth')))

        return optimizer, scheduler