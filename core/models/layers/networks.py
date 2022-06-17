import torch
import torch.nn as nn
import torch.quantization as quantization

from .modules.modules import get_conv
from .modules.blocks import get_resblock, DenseBlock

from configs.configs_wrapper import IODFConfig

class NN(nn.Module):
    def __init__(
            self, configs:IODFConfig, c_in, c_out, nn_type):
        super().__init__()

        n_channels = configs.n_channels
        
        if nn_type == 'resnet':

            Block = get_resblock(configs)

            layers = [
                get_conv(c_in, n_channels, 3, False, wq_level = configs.wq_level)
            ]

            for _ in range(configs.nn_depth):
                layers.append(
                    Block(n_channels, configs.quantize, wq_level = configs.wq_level)
                )

            layers.append(
                get_conv(n_channels, c_out, 3, _quantize = configs.quantize, wq_level = configs.wq_level)
            )

            if configs.quantize:
                # print(layers[1].residual_function[0].input_quantizer)
                layers[1].residual_function[0].input_quantizer = quantization.FakeQuantize(observer=quantization.MovingAverageMinMaxObserver, \
                    quant_min=-128, quant_max=127, dtype=torch.qint8)
        
        elif nn_type == 'densenet':
            configs.densenet_depth = configs.nn_depth

            layers = [
                DenseBlock(
                    args=configs,
                    n_inputs=c_in,
                    n_outputs=n_channels + c_in)
            ]

            layers += [
                nn.Conv2d(n_channels + c_in, c_out, kernel_size=3, padding=1)
                ]
            
        else:
            raise ValueError

        self.nn = nn.Sequential(*layers)

        self.nn[-1].weight.data.zero_() 
        self.nn[-1].bias.data.zero_()
 
    def forward(self, x):
        return self.nn(x)