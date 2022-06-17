import torch 
import torch.nn as nn
import torch.quantization as quantization

def set_fakequantization_states(model:nn.Module):
    model.apply(quantization.disable_observer)
    model.apply(quantization.enable_fake_quant)