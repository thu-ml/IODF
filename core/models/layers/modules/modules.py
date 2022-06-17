import torch
import torch.nn as nn 
import torch.nn.functional as F

from .lsq import LSQ, LSQPerChannel
from .compactor import CompactorLayer

def get_conv(in_chn, out_chn, kernel_size, _quantize, _pruning, **kwargs):
    kernel_to_padding = {1:0, 3:1, 5:2}
    padding = kernel_to_padding[kernel_size]
    if not _quantize:
        return ConvBuilder.Conv(
            in_channels=in_chn,
            out_channels=out_chn, 
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
            _pruning=_pruning
        )
    else:
        return ConvBuilder.LSQConv(
            in_channels=in_chn, 
            out_channels=out_chn, 
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
            w_bits=kwargs['w_bits'],
            a_bits=kwargs['a_bits'],
            q_level=kwargs['wq_level'],
            _pruning=_pruning
        )
        

class ConvBuilder:
    conv_idx = 0 # index pruned conv, this is a static variable
    
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def Conv(in_channels, out_channels, kernel_size, stride, padding, bias, _pruning):
        ConvBuilder.conv_idx += 1
        if not _pruning:
            return Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias, pruning=False)
        else: 
            return Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias, pruning=True, conv_idx=ConvBuilder.conv_idx)


    @staticmethod
    def LSQConv(in_channels, out_channels, kernel_size, stride, padding, bias, w_bits, a_bits, q_level, _pruning):
        ConvBuilder.conv_idx += 1
        if not _pruning:
            return LSQConv2d(in_channels, out_channels, kernel_size, stride, padding, \
                    bias=bias, w_bits=w_bits, a_bits=a_bits, q_level = q_level, \
                    pruning=False)
        else:
            return LSQConv2d(in_channels, out_channels, kernel_size, stride, padding, \
                    bias=bias, w_bits=w_bits, a_bits=a_bits, q_level = q_level, \
                    pruning=True, conv_idx=ConvBuilder.conv_idx)
    
    @staticmethod
    def reset_conv_cnt():
        ConvBuilder.conv_idx = 0

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, pruning, conv_idx=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.pruning = pruning
        if pruning:
            assert conv_idx is not None
            self.compactor = CompactorLayer(num_features=out_channels, conv_idx=conv_idx)
        
    def forward(self, x):
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        self.shape = (x.shape[-2], x.shape[-1])
        self.set_flops()
        
        if not self.pruning:
            return x
        else:
            return self.compactor(x)
    
    def set_zero(self):
        assert self.pruning
        pwc = self.compactor.get_metric_vector()
        self.weight.data[pwc <= 0.5, :, :, :] = 0.
        self.bias.data[pwc <= 0.5] = 0.
    
    def set_flops(self):
        import numpy as np 
        self.flops = np.prod(self.shape) * np.prod(self.kernel_size) * self.in_channels * self.out_channels


class LSQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, w_bits, a_bits, q_level, pruning, conv_idx=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if w_bits == 32:
            self.weight_quantizer = None
        else:
            if q_level == 'L':
                self.weight_quantizer = LSQ(w_bits, is_act=False) 
            elif q_level == 'C':
                self.weight_quantizer = LSQPerChannel(out_channels, w_bits)
        self.input_quantizer = LSQ(a_bits, is_act=True)

    def forward(self, x):
        x = self.input_quantizer(x)
        if self.weight_quantizer is not None:
            qweight = self.weight_quantizer(self.weight)
        else:
            qweight = self.weight
    
        x = F.conv2d(x, qweight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x 

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


class PadModule(nn.Module):
    def __init__(self, h=None):
        super().__init__()
        if h is None:
            self.register_buffer('h', torch.tensor([0]))
        self.register_buffer('h', h)
    
    def forward(self, x):
        return self.h


class FeatureMapPruner(nn.Module):
    def __init__(self, num_features, indices=None):
        super().__init__()
        if indices is None:
            indices = torch.arange(num_features)
        self.register_buffer('indices', indices) # remained channels indices
    
    def forward(self, x):
        return x[:, self.indices, :, :]

class FeatureMapScatter(nn.Module):
    def __init__(self, num_features, indices = None):
        super().__init__()
        self.out_chn = num_features
        if indices is None:
            indices = torch.arange(num_features) 
        self.register_buffer('indices', indices)
    
    @staticmethod
    def set_indices(n_channels, remain_idx):
        # remain_idx is bool tensor
        indices = torch.arange(n_channels)
        inv_indices = torch.cat([torch.arange(n_channels)[remain_idx], torch.arange(n_channels)[torch.logical_not(remain_idx)]])
        indices[inv_indices] = torch.arange(n_channels)
        return indices
    
    def forward(self, x):
        x = torch.cat([x, torch.zeros([x.size(0), self.out_chn - x.size(1), x.size(2), x.size(3)], device=x.device)], dim=1)
        return x[:, self.indices, :, :]
