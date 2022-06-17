import numpy as np
import torch
from torch import nn

from .constants import *
from .logger import get_time

def get_conv_flops(input_deps, output_deps, h, w=None, kernel_size=3, groups=1):
    if w is None:
        w = h
    return input_deps * output_deps * h * w * kernel_size * kernel_size // groups

def calculate_flops():
    return None

def get_feature_sizes(n_levels, n_flows, nn_depth, input_resolution=32):
    convidx_to_feature_size = dict()
    conv_cnt = 0
    input_resolution /= 2
    for i in range(n_levels):
        n_resblocks_this_level = n_flows+1 if i != n_levels-1 else n_flows

        for j in range(n_resblocks_this_level):
            conv_cnt += 1
            convidx_to_feature_size[conv_cnt] = input_resolution
            for k in range(nn_depth):
                conv_cnt += 1
                convidx_to_feature_size[conv_cnt] = input_resolution
                conv_cnt += 1
                convidx_to_feature_size[conv_cnt] = input_resolution
            conv_cnt += 1 
            convidx_to_feature_size[conv_cnt] = input_resolution
        input_resolution = input_resolution // 2

    return convidx_to_feature_size

def calculate_num_filters_of_model(model:nn.Module):
    n_filters = 0
    
    for child_module in model.modules():
        if isinstance(child_module, nn.Conv2d):

            out_chn, in_chn, _, _ = child_module.weight.shape
            n_filters += out_chn * in_chn


    return np.sum(n_filters)

def calculate_conv_flops_of_model(model:nn.Module, n_levels=3, n_flows=8, nn_depth=8, input_resolution=32):
    print(f"Flow architecture: n_levels: {n_levels}, n_flows: {n_flows}, nn_depth: {nn_depth}, input_resolution: {input_resolution}")
    convidx_to_feature_size = get_feature_sizes(n_levels, n_flows, nn_depth, input_resolution)
    
    conv_cnt = 0
    conv_flops = []

    for child_module in model.modules():
        if isinstance(child_module, nn.Conv2d):
            conv_cnt += 1
            h = convidx_to_feature_size[conv_cnt]
            out_chn, in_chn, _, _ = child_module.weight.shape
            conv_flops.append(get_conv_flops(in_chn, out_chn, h))

    return np.sum(conv_flops)

def calculate_parameters_of_model(model:nn.Module):
    num_float = 0
    for child_module in model.modules():
        if isinstance(child_module, nn.Conv2d):
            num_float += child_module.weight.numel()
            num_float += child_module.bias.numel()
    return num_float / 1e6 # MByte

def pruning_msg(filters_deactivated, layer_metric_dict, iteration, prune_mode='conv1', dataset = 'imagenet32'):
    if prune_mode == 'conv1':
        sorted_layer_metric = sorted(layer_metric_dict, key=layer_metric_dict.get)
        s = ''
        for k in sorted_layer_metric[:10]:
            s+= f"{layer_metric_dict[k]:5.4f} "
        s += '... '
        for k in sorted_layer_metric[-10:]:
            s+= f"{layer_metric_dict[k]:5.4f} "
        
        filters_deactivated_per_level = [0,0,0,0]
        for it in filters_deactivated:
            filters_deactivated_per_level[specify_resolution_for_each_conv(it[0])-1] += 1
        s1 = f"Metrics: {s}"
        s2 = f'{get_time()} Iter {iteration:<5d}\t {len(filters_deactivated):<5d} filters deactivated. ( {filters_deactivated_per_level} )'

    elif prune_mode == 'mask':
        sorted_layer_metric = sorted(layer_metric_dict, key=layer_metric_dict.get)
        s = ''
        for k in sorted_layer_metric[:10]:
            s+= f"{layer_metric_dict[k]:5.4f} "
        s += '... '
        for k in sorted_layer_metric[-10:]:
            s+= f"{layer_metric_dict[k]:5.4f} "
        if dataset == 'imagenet32':
            filters_deactivated_per_level = [[0,0,0],[0,0,0],[0,0,0]]
            from .constants import conv1_filters_per_level_32 as conv1_filters_per_level, conv2_filters_per_level_32 as conv2_filters_per_level
        else:  
            filters_deactivated_per_level = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
            from .constants import conv1_filters_per_level_64 as conv1_filters_per_level, conv2_filters_per_level_64 as conv2_filters_per_level

        for it in filters_deactivated:
            flow_level = specify_resolution_for_each_mask(it[0], dataset)
            pos = specify_pos_for_each_mask(it[0]) 
            filters_deactivated_per_level[flow_level-1][pos] += 1

        s1 = f"Metrics: {s}"
        msg = ""
        for l in range(len(filters_deactivated_per_level)):
            msg += f"\nLevel {l+1}. In: {filters_deactivated_per_level[l][0]} / {conv1_filters_per_level[l]} Mid: {filters_deactivated_per_level[l][1]} / {conv2_filters_per_level[l]} Out: {filters_deactivated_per_level[l][2]} / {conv2_filters_per_level[l]}"

        s2 = f'{get_time()} Iter {iteration:<5d}\t deactivated filters information: {msg}'

    return s1, s2