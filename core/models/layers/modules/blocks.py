import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU

from .compactor import CompactorLayer
from .modules import get_conv, Identity
from .lsq import LSQ

class ResBlock(nn.Module):
    def __init__(self, n_chn, _quantize, _pruning, **quant_args):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            get_conv(n_chn, n_chn, 3, _quantize, _pruning, **quant_args),
            nn.ReLU(inplace=True),
            get_conv(n_chn, n_chn, 3, _quantize, False, **quant_args)
        )

        if _quantize:
            self.shortcut = LSQ(quant_args['a_bits'], is_act=True)
        else:
            self.shortcut = Identity()

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResBlock64(nn.Module):
    def __init__(self, n_chn, _quantize, _pruning, **quant_args):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            get_conv(n_chn, n_chn, 3, _quantize, _pruning, **quant_args),
            nn.ReLU(inplace=True),
            get_conv(n_chn, n_chn, 3, _quantize, False, **quant_args),
            nn.ReLU(inplace=True)
        )

        if _quantize:
            self.shortcut = LSQ(quant_args['a_bits'], is_act=True)
        else:
            self.shortcut = Identity()

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResBlockWithMask(nn.Module):
    def __init__(self, n_chn, _quantize, _pruning, **quant_args):
        super().__init__()

        assert _pruning

        self.in_mask = CompactorLayer(n_chn)
        self.mid_mask = CompactorLayer(n_chn)
        self.out_mask = CompactorLayer(n_chn)

        #residual function
        self.residual_function = nn.ModuleList([
            get_conv(n_chn, n_chn, 3, _quantize, _pruning=False, **quant_args),
            nn.ReLU(inplace=True),
            get_conv(n_chn, n_chn, 3, _quantize, _pruning=False, **quant_args),
        ])
    
        if _quantize:
            self.shortcut = LSQ(quant_args['a_bits'], is_act=True)
        else:
            self.shortcut = Identity()

    def forward(self, x):
        h = self.in_mask(x)
        h = self.mid_mask(self.residual_function[0](h))
        h = self.residual_function[2](self.residual_function[1](h))
        return nn.ReLU(inplace=True)(self.out_mask(h) + self.shortcut(x))


class ResBlock64WithMask(nn.Module):
    def __init__(self, n_chn, _quantize, _pruning, **quant_args):
        super().__init__()

        assert _pruning

        self.in_mask = CompactorLayer(n_chn)
        self.mid_mask = CompactorLayer(n_chn)
        self.out_mask = CompactorLayer(n_chn)

        #residual function
        self.residual_function = nn.ModuleList([
            get_conv(n_chn, n_chn, 3, _quantize, _pruning=False, **quant_args),
            nn.ReLU(inplace=True),
            get_conv(n_chn, n_chn, 3, _quantize, _pruning=False, **quant_args),
            nn.ReLU(inplace=True)
        ])
    
        if _quantize:
            self.shortcut = LSQ(quant_args['a_bits'], is_act=True)
        else:
            self.shortcut = Identity()

    def forward(self, x):
        h = self.in_mask(x)
        h = self.mid_mask(self.residual_function[0](h))
        h = self.out_mask(self.residual_function[2](self.residual_function[1](h)))
        h = self.residual_function[3](h)
        return nn.ReLU(inplace=True)(h + self.shortcut(x))

def get_resblock(cfg):
    if cfg.dataset == 'imagenet32':
        if cfg.pruning:
            if cfg.prune_mode == 'conv1':
                Block = ResBlock
            elif cfg.prune_mode == 'mask':
                Block = ResBlockWithMask
            else:
                print('wrong prune mode: ', cfg.prune_mode)
                raise NotImplementedError
        else:
            Block = ResBlock
    elif cfg.dataset == 'imagenet64':
        if cfg.pruning:
            if cfg.prune_mode == 'conv1':
                Block = ResBlock64
            if cfg.prune_mode == 'mask':
                Block = ResBlock64WithMask
        else:
            Block = ResBlock64
    return Block

class Conv2dReLU(nn.Module):
    def __init__(
            self, n_inputs, n_outputs, kernel_size=3, _quantize = False, _pruning = False, **kwargs):
        super().__init__()
        self.nn = get_conv(n_inputs, n_outputs, kernel_size, _quantize, _pruning, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.nn(x)
        y = self.relu(h)
        return y

class DenseLayer(nn.Module):
    def __init__(self, n_inputs, growth, Conv2dAct, _quantize, _pruning, **kwargs):
        super().__init__()

        conv1x1 = Conv2dAct(
                n_inputs, n_inputs, 1, _quantize, _pruning, **kwargs)

        self.nn = nn.Sequential(
            conv1x1,
            Conv2dAct(
                n_inputs, growth, 3, _quantize, _pruning, **kwargs)
            )

    def forward(self, x):
        h = self.nn(x)
        h = torch.cat([x, h], dim=1)
        return h

class DenseBlock(nn.Module):
    def __init__(
            self, args, n_inputs, n_outputs, Conv2dAct=Conv2dReLU):
        super().__init__()
        depth = args.densenet_depth

        future_growth = n_outputs - n_inputs

        layers = []

        for d in range(depth):
            growth = future_growth // (depth - d)

            layers.append(DenseLayer(n_inputs, growth, Conv2dAct, args.quantize, args.pruning, w_bits = args.w_bits, a_bits = args.a_bits, wq_level = args.wq_level))
            n_inputs += growth
            future_growth -= growth

        self.nn = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)
