import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

import torch.nn as nn

from configs.configuration import get_configs
from core.interfaces.model import load_model

def get_conv_depth():
    configs = get_configs(
        dataset='imagenet32',
        mode='eval',
        model='idf', 
        nn_type='resnet',
        batch_size=100,
        resume=None,
        pruning=False
    )

    model = load_model(configs)
    deps = []
    in_deps = []
    for mm in model.modules():
        if isinstance(mm, nn.Conv2d):
            deps.append(mm.out_channels)
            in_deps.append(mm.in_channels)
    # print(in_deps)
    # exit()
    with open('./snapshots/pruning_info.txt', 'a') as f:
        print('=' * 20 + '\n' + 'INFO: convolution layer depth', file=f)
        print(len(deps), file=f)
        print(str(deps) + '\n', file=f)

def get_target_layers(n_levels, n_flows, nn_depth):
    n_qnns = n_flows * n_levels + (n_levels - 1)  # two splitpriors
    conv_idx = 0
    target_layers = []
    for i in range(n_qnns):
        # in each QNN, one in-conv, one out-conv, nn_depth residualblock
        conv_idx += 1
        for k in range(nn_depth):
            conv_idx+=1
            target_layers.append(conv_idx)
            conv_idx+=1
        conv_idx+=1

            
    with open('./snapshots/pruning_info.txt', 'a') as f:
        print('=' * 20 + '\n' + 'INFO: pruning convolution layer index', file=f)
        print(len(target_layers), file=f)
        print(str(target_layers) + '\n', file=f)


def get_follow_dict(n_levels=3, n_flows=8, nn_depth=8):
    follow_dict = dict()
    n_qnns = n_flows * n_levels + (n_levels - 1)
    n_convs_in_one_qnn = nn_depth * 2 + 2
    for i in range(n_qnns):
        start_idx = n_convs_in_one_qnn * i + 1
        for j in range(nn_depth):
            follow_dict[start_idx + (2*j+1)] = start_idx + (2*j+2)
    with open('./snapshots/pruning_info.txt', 'a') as f:
        print('=' * 20 + '\n' + 'INFO: follow dict', file=f)
        print(str(follow_dict) + '\n', file=f)

def get_shortcut_layers(n_levels=3, n_flows=8, nn_depth=8):
    # keep out channels the same 
    keep_dict = dict()
    n_qnns = n_flows * n_levels + (n_levels - 1)
    n_convs_in_one_qnn = nn_depth * 2 + 2
    for i in range(n_qnns):
        start_idx = n_convs_in_one_qnn * i 
        for j in range(nn_depth):
            keep_dict[start_idx + (2*j+1)] = start_idx + (2*j+3)
    
    with open('./snapshots/pruning_info.txt', 'a') as f:
        print('=' * 20 + '\n' + 'INFO: keep the same out channels dict', file=f)
        print(str(keep_dict) + '\n', file=f)

# get_conv_depth()
get_target_layers(4, 8, 8)
# get_follow_dict()
# get_shortcut_layers()