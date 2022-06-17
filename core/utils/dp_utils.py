import torch
import torch.nn as nn

from core.models.layers.modules.blocks import ResBlock
from core.models.layers.modules.modules import FeatureMapScatter, FeatureMapPruner, PadModule, QConv2d

def set_conv(conv:nn.Conv2d, weight, bias):
    conv.weight.data = weight
    conv.bias.data = bias
    conv.in_channels = weight.size(1)
    conv.out_channels = weight.size(0)

def convert_model_conv1_mode(model:nn.Module, load_path):

    params = dict(torch.load(load_path))
    conv_cnt = 0
    for name, mo in model.named_modules():
        if hasattr(mo, 'permutation'):
            mo.permutation = params[name+'.permutation']
            mo.permutation_inv = params[name+'.permutation_inv']
        if isinstance(mo, nn.Conv2d):
            conv_cnt += 1
            mo.weight.data = params[f'conv{conv_cnt}.weight']
            mo.bias.data = params[f'conv{conv_cnt}.bias']
            mo.in_channels = mo.weight.size(1)
            mo.out_channels = mo.weight.size(0)
            print(f'Conv {conv_cnt}: In [{mo.in_channels}]  Out [{mo.out_channels}]')
    model.prior.mu, model.prior.logs, model.prior.pi_logit = params['prior.mu'], params['prior.logs'], params['prior.pi_logit']
    return model

def convert_model_mask_mode(model:nn.Module, load_path):

    params = dict(torch.load(load_path))

    conv_cnt = 0
    for name, mo in model.named_modules():
        if hasattr(mo, 'permutation'):
            mo.permutation = params[name+'.permutation']
            mo.permutation_inv = params[name+'.permutation_inv']
        if isinstance(mo, nn.Conv2d):
            conv_cnt += 1
            if f'conv{conv_cnt}.weight' in params.keys():
                set_conv(mo, params[f'conv{conv_cnt}.weight'], params[f'conv{conv_cnt}.bias'])

    rb_cnt = 0
    for name, mo in model.named_modules():  
        if isinstance(mo, ResBlock):
            rb_cnt += 1
            if rb_cnt in params['removed_resblocks']:
                mo.residual_function = nn.Sequential(PadModule(params[f'resblock{rb_cnt}.output_scatter.h']))
            else:
                conv1, relu, conv2 = mo.residual_function
                mo.residual_function = nn.Sequential(
                    FeatureMapPruner(128, params[f'resblock{rb_cnt}.input_pruner.indices']),
                    conv1,
                    relu,
                    conv2,
                    FeatureMapScatter(128, params[f'resblock{rb_cnt}.output_scatter.indices'])
                )

    model.prior.mu, model.prior.logs, model.prior.pi_logit = params['prior.mu'], params['prior.logs'], params['prior.pi_logit']
    return model

def convert_fun(pruning_mode='conv1'):
    if pruning_mode == 'conv1':
        return convert_model_conv1_mode
    elif pruning_mode == 'mask':
        return convert_model_mask_mode