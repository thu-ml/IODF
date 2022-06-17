import torch

from core.utils.constants import resolution_scale, specify_resolution_for_each_conv, specify_resolution_for_each_mask

def add_lasso_grad(compactor_params, cfg, dp_config):
    if cfg.prune_mode == 'conv1':
        for conv_idx, compactor_param in compactor_params.items():
            if not compactor_param.requires_grad:
                continue
            lasso_mask = torch.logical_and(compactor_param.data >= 0, compactor_param.data <= 1.)
            lasso_grad = torch.ones_like(compactor_param.data, device=cfg.device)
            scale = resolution_scale(cfg.dataset)[specify_resolution_for_each_conv(conv_idx)]
            compactor_param.grad.data.mul_(scale).add_(dp_config.lasso_strength, lasso_grad)
            compactor_param.grad.data = compactor_param.grad.data * lasso_mask
    
    elif cfg.prune_mode == 'mask':
        for mask_idx, compactor_param in compactor_params.items():
            if not compactor_param.requires_grad:
                continue
            lasso_mask = torch.logical_and(compactor_param.data >= 0, compactor_param.data <= 1.)
            lasso_grad = torch.ones_like(compactor_param.data, device=cfg.device)
            scale = resolution_scale(cfg.dataset)[specify_resolution_for_each_mask(mask_idx, cfg.dataset)]
            compactor_param.grad.data.mul_(scale).add_(dp_config.lasso_strength, lasso_grad)
            compactor_param.grad.data = lasso_mask * compactor_param.grad.data