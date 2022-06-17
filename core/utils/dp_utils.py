import torch
import torch.nn as nn

from core.models.layers.modules.compactor import CompactorLayer
from core.models.layers.modules.blocks import ResBlock64, ResBlockWithMask, ResBlock64WithMask, ResBlock
from core.models.layers.modules.modules import FeatureMapScatter, FeatureMapPruner, Identity, PadModule

remain_0 = True

def set_mask_idx(model:nn.Module):
    mask_cnt = 0
    for child_mo in model.modules():
        if isinstance(child_mo, CompactorLayer):
            mask_cnt += 1 
            child_mo.mask_idx = mask_cnt

def set_conv(conv, weight, bias):
    conv.weight.data = weight
    conv.bias.data = bias
    conv.in_channels = weight.shape[1]
    conv.out_channels = weight.shape[0]

def get_filters_deactivated(model:nn.Module, prune_mode='conv1'):
    layer_metric_dict = dict()
    filters_deactivated = set()
    if prune_mode == 'conv1':
        for child_module in model.modules():
            if hasattr(child_module, 'conv_idx'):
                mask, pwc = child_module.get_mask(), child_module.pwc
                
                for i in range(len(mask)):
                    layer_metric_dict[(child_module.conv_idx, i)] = pwc[i]
                    if mask[i] == 0:
                        filters_deactivated.add((child_module.conv_idx, i))
    elif prune_mode == 'mask':
        for child_module in model.modules():
            if isinstance(child_module, CompactorLayer):
                mask, pwc = child_module.get_mask(), child_module.pwc
                
                for i in range(len(mask)):
                    layer_metric_dict[(child_module.get_idx(), i)] = pwc[i]
                    if mask[i] == 0:
                        filters_deactivated.add((child_module.get_idx(), i))
    return filters_deactivated, layer_metric_dict

def get_compactor_params(model:nn.Module, prune_mode='conv1'):
    if prune_mode == 'conv1':
        compactor_params = {}
        for name, child_module in model.named_modules():
            if hasattr(child_module, 'pwc'): # a compactor
                compactor_params[child_module.conv_idx] = child_module.pwc
    
    elif prune_mode == 'mask':
        compactor_params = {}
        for name, child_module in model.named_modules():
            if isinstance(child_module, CompactorLayer):
                compactor_params[child_module.get_idx()] = child_module.pwc

    return compactor_params

def get_target_layers(dataset):
    if dataset == 'imagenet32':
        from core.utils.constants import target_layers_32 as target_layers
    elif dataset == 'imagenet64':
        from core.utils.constants import target_layers_64 as target_layers
    return target_layers

def remove_compactor_conv1_mode(model: nn.Module, dataset='imagenet32'):

    if remain_0:
        name_to_in_idx = dict()
        target_layers = get_target_layers(dataset)
        params = dict()

        with torch.no_grad():

            conv_cnt = 0

            for name, child_module in model.named_modules():
                if hasattr(child_module, 'permutation'):
                    params[name+'.permutation'] = child_module.permutation
                    params[name+'.permutation_inv'] = child_module.permutation_inv

                if isinstance(child_module, nn.Conv2d):
                    conv_cnt += 1
                    if hasattr(child_module, 'compactor'):
                        assert conv_cnt == child_module.compactor.conv_idx,  f"{conv_cnt} {child_module.compactor.conv_idx}"
                        assert conv_cnt in target_layers, f"{conv_cnt} not in target layers"

                        channel_metric = child_module.compactor.get_metric_vector()
                        remained_idx = channel_metric > 0.5

                        if remained_idx.sum() == 0:
                            child_module.weight.data.zero_()
                            child_module.bias.data.zero_()
                            remained_idx[0] = True

                        weight, bias = child_module.weight.data[remained_idx, :, :, :], child_module.bias.data[remained_idx]
                        
                        follower = conv_cnt + 1
                        name_to_in_idx[follower] = remained_idx
                            
                    else:
                        weight, bias = child_module.weight.data, child_module.bias.data

                    if conv_cnt in name_to_in_idx.keys():
                        weight = weight[:, name_to_in_idx[conv_cnt], :, :]

                    params[f'conv{conv_cnt}.weight'] = weight
                    params[f'conv{conv_cnt}.bias'] = bias
                    
        
        params['prior.mu'] = model.prior.mu
        params['prior.logs'] = model.prior.logs
        params['prior.pi_logit'] = model.prior.pi_logit

        return params

    else:
        def pair_conv_resblock():
            if dataset == 'imagenet32':
                n_levels, n_flows, nn_depth = 3,8,8
            elif dataset == 'imagenet64':
                n_levels, n_flows, nn_depth = 4,8,8
            conv_cnt, resblock_cnt = 0, 0
            conv_to_resblock = {}
            for i in range(n_levels):
                num_nn = n_flows+1 if i < n_levels - 1 else n_flows
                for j in range(num_nn):
                    conv_cnt += 1
                    for k in range(nn_depth):
                        resblock_cnt += 1
                        conv_cnt += 1
                        conv_to_resblock[conv_cnt] = resblock_cnt
                        conv_cnt += 1
                        conv_to_resblock[conv_cnt] = resblock_cnt
                        
                    conv_cnt += 1

            return conv_to_resblock
        
        name_to_in_idx = dict()
        target_layers = get_target_layers(dataset)
        params = dict()
        conv_to_resblock = pair_conv_resblock()
        removed_rbs = []

        with torch.no_grad():

            conv_cnt = 0

            for name, child_module in model.named_modules():
                if hasattr(child_module, 'permutation'):
                    params[name+'.permutation'] = child_module.permutation
                    params[name+'.permutation_inv'] = child_module.permutation_inv

                if isinstance(child_module, nn.Conv2d):
                    conv_cnt += 1
                    if hasattr(child_module, 'compactor'):
                        assert conv_cnt == child_module.compactor.conv_idx,  f"{conv_cnt} {child_module.compactor.conv_idx}"
                        assert conv_cnt in target_layers, f"{conv_cnt} not in target layers"

                        if conv_to_resblock[conv_cnt] in removed_rbs:
                            continue

                        channel_metric = child_module.compactor.get_metric_vector()
                        remained_idx = channel_metric > 0.5

                        if remained_idx.sum() == 0:
                            removed_rbs.append(conv_to_resblock[conv_cnt])
                            print(conv_cnt)
                            continue

                        weight, bias = child_module.weight.data[remained_idx, :, :, :], child_module.bias.data[remained_idx]
                        
                        follower = conv_cnt + 1
                        name_to_in_idx[follower] = remained_idx
                            
                    else:
                        weight, bias = child_module.weight.data, child_module.bias.data

                    if conv_cnt in name_to_in_idx.keys():
                        weight = weight[:, name_to_in_idx[conv_cnt], :, :]

                    params[f'conv{conv_cnt}.weight'] = weight
                    params[f'conv{conv_cnt}.bias'] = bias
                    
        
        params['prior.mu'] = model.prior.mu
        params['prior.logs'] = model.prior.logs
        params['prior.pi_logit'] = model.prior.pi_logit
        params['removed_rbs'] = removed_rbs

        return params


def convert_model_conv1_mode(model:nn.Module, load_path):
    ''''
        Model is a normal model with no compactors.

        Convert original model to a pruned thinner model. (codes defining the model do not need changing.) 
    '''
    if remain_0:
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
    else:
        def pair_conv_resblock(dataset):
            if dataset == 'imagenet32':
                n_levels, n_flows, nn_depth = 3,8,8
            elif dataset == 'imagenet64':
                n_levels, n_flows, nn_depth = 4,8,8
            conv_cnt, resblock_cnt = 0, 0
            conv_to_resblock = {}
            for i in range(n_levels):
                num_nn = n_flows+1 if i < n_levels - 1 else n_flows
                for j in range(num_nn):
                    conv_cnt += 1
                    for k in range(nn_depth):
                        resblock_cnt += 1
                        conv_cnt += 1
                        conv_to_resblock[conv_cnt] = resblock_cnt
                        conv_cnt += 1
                        conv_to_resblock[conv_cnt] = resblock_cnt
                        
                    conv_cnt += 1

            return conv_to_resblock

        params = dict(torch.load(load_path))
        removed_rbs = params['removed_rbs']
        conv_cnt, rb_cnt = 0, 0
        conv_to_resblocks = pair_conv_resblock('imagenet32')

        # for mo in model.modules():
        #     if isinstance(mo, ResBlock):
        #         rb_cnt += 1
        #         if rb_cnt in removed_rbs:
        #             mo = Identity()

        for name, mo in model.named_modules():
            if hasattr(mo, 'permutation'):
                mo.permutation = params[name+'.permutation']
                mo.permutation_inv = params[name+'.permutation_inv']
            if isinstance(mo, nn.Conv2d):
                conv_cnt += 1
                if conv_cnt in conv_to_resblocks.keys() and conv_to_resblocks[conv_cnt] in removed_rbs:
                    continue
                mo.weight.data = params[f'conv{conv_cnt}.weight']
                mo.bias.data = params[f'conv{conv_cnt}.bias']
                mo.in_channels = mo.weight.size(1)
                mo.out_channels = mo.weight.size(0)
                print(f'Conv {conv_cnt}: In [{mo.in_channels}]  Out [{mo.out_channels}]')

        rb_cnt = 0
        from core.models.layers.networks import NN
        for name, mo in model.named_modules():
            if isinstance(mo, NN):
                for i, module in enumerate(mo.nn):
                    if isinstance(module, ResBlock):
                        rb_cnt += 1
                        if rb_cnt in removed_rbs:
                            mo.nn[i] = Identity()

        model.prior.mu, model.prior.logs, model.prior.pi_logit = params['prior.mu'], params['prior.logs'], params['prior.pi_logit']
        return model

def remove_compactor_mask_mode(model:nn.Module, dataset='imagenet32'):

    n_channels = model.configs.n_channels

    def pair_conv_mask():
        if dataset == 'imagenet32':
            n_levels, n_flows, nn_depth = 3,8,8
        elif dataset == 'imagenet64':
            n_levels, n_flows, nn_depth = 4,8,8
        conv_cnt, mask_cnt, resblock_cnt = 0, 0, 0
        conv_to_in_out_mask = {}
        conv_to_resblock = {}
        for i in range(n_levels):
            num_nn = n_flows+1 if i < n_levels - 1 else n_flows
            for j in range(num_nn):
                conv_cnt += 1
                for k in range(nn_depth):
                    resblock_cnt += 1
                    conv_cnt += 1
                    conv_to_in_out_mask[conv_cnt] = (mask_cnt+2, mask_cnt+1)
                    conv_to_resblock[conv_cnt] = resblock_cnt
                    conv_cnt += 1
                    conv_to_in_out_mask[conv_cnt] = (mask_cnt+3, mask_cnt+2)
                    conv_to_resblock[conv_cnt] = resblock_cnt
                    mask_cnt += 3

                conv_cnt += 1
        return conv_to_in_out_mask, conv_to_resblock
    
    def extract_masked_model():
        conv_cnt, mask_cnt, rb_cnt = 0,0,0
        convs, masks, resblocks = {},{},{}
        for name, child_module in model.named_modules():
            if isinstance(child_module, nn.Conv2d):
                conv_cnt += 1
                convs[conv_cnt] = (name, child_module)
            elif isinstance(child_module, CompactorLayer):
                mask_cnt += 1
                masks[mask_cnt] = child_module
            elif isinstance(child_module, ResBlockWithMask) or isinstance(child_module, ResBlock64WithMask):
                rb_cnt += 1
                resblocks[rb_cnt] = child_module
        return convs, masks, resblocks

    params = {}

    for name, child_module in model.named_modules():
        if hasattr(child_module, 'permutation'):
            params[name+'.permutation'] = child_module.permutation
            params[name+'.permutation_inv'] = child_module.permutation_inv
    
    params['prior.mu'], params['prior.logs'], params['prior.pi_logit'] = model.prior.mu, model.prior.logs, model.prior.pi_logit

    conv_to_in_out_mask, conv_to_resblock = pair_conv_mask()
    convs, masks, resblocks = extract_masked_model()
        
    removed_resblocks = set()

    for conv_idx, (name, conv_layer) in convs.items():
        if conv_idx in conv_to_in_out_mask.keys():

            out_mask, in_mask = conv_to_in_out_mask[conv_idx]

            remain_in = masks[in_mask].get_mask().bool()
            remain_out = masks[out_mask].get_mask().bool()

            if remain_in.sum() == 0 or remain_out.sum() == 0:
                removed_resblocks.add(conv_to_resblock[conv_idx])
                continue
            else:
                weight, bias = conv_layer.weight.data, conv_layer.bias.data
                weight = weight[:, remain_in, :, :]
                weight = weight[remain_out, :, :, :]
                bias = bias[remain_out]

        else:
            weight, bias = conv_layer.weight.data, conv_layer.bias.data

        params[f'conv{conv_idx}.weight'] = weight
        params[f'conv{conv_idx}.bias'] = bias

    params['removed_resblocks'] = removed_resblocks
 
    for rb_idx in removed_resblocks: 
        r2 = resblocks[rb_idx]
        x=torch.zeros([1, n_channels, 2, 2])
        with torch.no_grad():
            h = r2.in_mask(x)
            h = r2.mid_mask(r2.residual_function[0](h))
            h = r2.out_mask(r2.residual_function[2](r2.residual_function[1](h)))
            if dataset == 'imagenet64':
                h = r2.residual_function[3](h)
            h = h[0,:,0,0].view(1, n_channels, 1, 1)
        
        params[f'resblock{rb_idx}.output_scatter.h'] = h

    for mask_idx, mask in masks.items():
        resblock_idx, r = (mask_idx - 1) // 3 + 1, (mask_idx - 1) % 3
        if resblock_idx in removed_resblocks:
            continue
        else:
            remain_idx = mask.get_mask().bool()
            if r == 0:
                params[f'resblock{resblock_idx}.input_pruner.indices'] = torch.arange(n_channels)[remain_idx]
            if r == 2:
                params[f'resblock{resblock_idx}.output_scatter.indices'] = FeatureMapScatter.set_indices(n_channels, remain_idx)

    return params

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
        if isinstance(mo, ResBlock) or isinstance(mo, ResBlock64):
            rb_cnt += 1
            if rb_cnt in params['removed_resblocks']:
                mo.residual_function = nn.Sequential(PadModule(params[f'resblock{rb_cnt}.output_scatter.h']))
            else:
                mlist = list(mo.residual_function)
                # conv1, relu, conv2 = mo.residual_function
                # mlist = [conv1, relu, conv2] if model.configs.dataset == 'imagenet32' else [conv1, relu, conv2, relu]
                mo.residual_function = nn.Sequential(
                    FeatureMapPruner(128, params[f'resblock{rb_cnt}.input_pruner.indices']),
                    *mlist,
                    FeatureMapScatter(128, params[f'resblock{rb_cnt}.output_scatter.indices'])
                )

    model.prior.mu, model.prior.logs, model.prior.pi_logit = params['prior.mu'], params['prior.logs'], params['prior.pi_logit']
    return model

def remove_fun(pruning_mode='conv1'):
    if pruning_mode == 'conv1':
        return remove_compactor_conv1_mode
    elif pruning_mode == 'mask':
        return remove_compactor_mask_mode

def convert_fun(pruning_mode='conv1'):
    if pruning_mode == 'conv1':
        return convert_model_conv1_mode
    elif pruning_mode == 'mask':
        return convert_model_mask_mode