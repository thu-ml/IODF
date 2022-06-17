import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.quantization as quantization

def get_conv(in_chn, out_chn, kernel_size, _quantize, wq_level):
    kernel_to_padding = {1:0, 3:1, 5:2}
    padding = kernel_to_padding[kernel_size]
    return QConv2d(
        in_channels=in_chn, 
        out_channels=out_chn, 
        kernel_size=kernel_size, 
        stride=1,
        padding=padding,
        bias=True,
        _quantize=_quantize,
        wq_level=wq_level
    )

def print_t(t):
    print(t.detach().cpu().numpy())

class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, _quantize, wq_level):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self._quantize = _quantize
        if self._quantize:
            if wq_level == 'L':
                self.weight_quantizer = quantization.FakeQuantize(observer=quantization.MovingAverageMinMaxObserver, \
                    quant_min=-128, quant_max=127, dtype=torch.qint8)
            elif wq_level == 'C':
                self.weight_quantizer = quantization.FakeQuantize(observer=quantization.MovingAveragePerChannelMinMaxObserver, \
                    quant_min=-128, quant_max=127, dtype=torch.qint8)
            
            # self.input_quantizer = quantization.FakeQuantize(observer=quantization.MovingAverageMinMaxObserver, \
            #     quant_min=0, quant_max=255, dtype=torch.quint8)
            self.input_quantizer = quantization.FakeQuantize(observer=quantization.MovingAverageMinMaxObserver, \
                quant_min=-128, quant_max=127, dtype=torch.qint8)
            # else:
            #     self.input_quantizer = quantization.FakeQuantize(observer=quantization.MovingAverageMinMaxObserver, \
            #         quant_min=-128, quant_max=127, dtype=torch.qint8)
        
    def forward(self, x):
        if not self._quantize:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            x = self.input_quantizer(x)
            qweight = self.weight_quantizer(self.weight)
            return F.conv2d(x, qweight, self.bias, self.stride, self.padding, self.dilation, self.groups)


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
        else:
            self.register_buffer('h', h)
    
    def forward(self, x):
        return self.h


class FeatureMapPruner(nn.Module):
    def __init__(self, num_features, indices=None):
        super().__init__()
        if indices is None:
            indices = torch.arange(num_features)
        self.register_buffer('indices', torch.ones(num_features))
    
    def forward(self, x):
        return x[:, self.indices, :, :]

class FeatureMapScatter(nn.Module):
    def __init__(self, num_features, indices = None):
        super().__init__()
        self.out_chn = num_features
        if indices is None:
            indices = torch.arange(num_features) 
        self.register_buffer('indices', indices)
        
    def set_indices(self, remain_idx):
        # remain_idx is bool tensor
        inv_indices = torch.cat([torch.arange(self.out_chn)[remain_idx], torch.arange(self.out_chn)[torch.logical_not(remain_idx)]])
        self.indices[inv_indices] = torch.arange(self.out_chn, dtype=torch.int32)
    
    def forward(self, x):
        x = torch.cat([x, torch.zeros([x.size(0), self.out_chn - x.size(1), x.size(2), x.size(3)], device=x.device)], dim=1)
        return x[:, self.indices, :, :]
