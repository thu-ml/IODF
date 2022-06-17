import torch
import torch.nn as nn
import torch.quantization as quantization

from .modules import get_conv, Identity

class ResBlock(nn.Module):
    def __init__(self, n_chn, _quantize, wq_level):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            get_conv(n_chn, n_chn, 3, _quantize, wq_level),
            nn.ReLU(inplace=True),
            get_conv(n_chn, n_chn, 3, _quantize, wq_level),
        )

        if _quantize:
            self.shortcut = quantization.FakeQuantize(observer=quantization.MovingAverageMinMaxObserver, \
                quant_min=-128, quant_max=127, dtype=torch.qint8)
        else:
            self.shortcut = Identity()

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResBlock64(nn.Module):
    def __init__(self, n_chn, _quantize, wq_level):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            get_conv(n_chn, n_chn, 3, _quantize, wq_level),
            nn.ReLU(inplace=True),
            get_conv(n_chn, n_chn, 3, _quantize, wq_level)
            # nn.ReLU(inplace=True)
        )

        if _quantize:
            self.shortcut = quantization.FakeQuantize(observer=quantization.MovingAverageMinMaxObserver, \
                quant_min=-128, quant_max=127, dtype=torch.qint8)
        else:
            self.shortcut = Identity()

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

def get_resblock(cfg):
    if cfg.dataset == 'imagenet32':
        Block = ResBlock
    elif cfg.dataset == 'imagenet64':
        Block = ResBlock64
    return Block


class Conv2dReLU(nn.Module):
    def __init__(
            self, n_inputs, n_outputs, kernel_size=3, _quantize = False, wq_level = 'C'):
        super().__init__()
        self.nn = get_conv(n_inputs, n_outputs, kernel_size, _quantize, wq_level)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.nn(x)
        y = self.relu(h)
        return y

class DenseLayer(nn.Module):
    def __init__(self, n_inputs, growth, Conv2dAct, _quantize, wq_level):
        super().__init__()

        conv1x1 = Conv2dAct(
                n_inputs, n_inputs, 1, _quantize, wq_level)

        self.nn = nn.Sequential(
            conv1x1,
            Conv2dAct(
                n_inputs, growth, 3, _quantize, wq_level)
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

            layers.append(DenseLayer(n_inputs, growth, Conv2dAct, args.quantize, args.wq_level))
            n_inputs += growth
            future_growth -= growth

        self.nn = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)
