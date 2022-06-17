import torch 
import torch.nn as nn

from core.models.layers.modules.lsq import LSQ, LSQPerChannel

def set_lsq_initialized(model:nn.Module):
    for m in model.modules():
        if isinstance(m, LSQ) or isinstance(m, LSQPerChannel):
            m.initialized = True

def set_lsq_un_initialized(model:nn.Module):
    for m in model.modules():
        if isinstance(m, LSQ) or isinstance(m, LSQPerChannel):
            m.initialized = False

def init_lsq_with_data(model:nn.Module, data_loader, device):
    model.eval()
    with torch.no_grad():
        for x, _ in data_loader:
            model(x.to(device))
            break

def disable_lsq(model:nn.Module):
    for child_module in model.modules():
        if isinstance(child_module, LSQ) or isinstance(child_module, LSQPerChannel):
            child_module.enable_lsq = False

def disable_lsq_weights(model:nn.Module):
    for name, child_module in model.named_modules():
        if (isinstance(child_module, LSQ) or isinstance(child_module, LSQPerChannel)) and 'weight_quantizer' in name:
            child_module.enable_lsq = False

def enable_lsq(model:nn.Module):
    for child_module in model.modules():
        if isinstance(child_module, LSQ) or isinstance(child_module, LSQPerChannel):
            child_module.enable_lsq = True