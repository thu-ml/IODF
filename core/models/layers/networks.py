import torch
import torch.nn as nn

from .modules.modules import get_conv
from .modules.lsq import LSQ
from .modules.blocks import get_resblock, DenseBlock
from .base import Base

from configs.configs_wrapper import IODFConfig

class NN(Base):
    def __init__(
            self, configs:IODFConfig, c_in, c_out, nn_type):
        super().__init__()

        n_channels = configs.n_channels
        
        if nn_type == 'resnet':

            Block = get_resblock(configs)

            layers = [
                get_conv(c_in, n_channels, 3, _quantize = False, _pruning = False, \
                        w_bits = configs.w_bits, a_bits = configs.a_bits, wq_level = configs.wq_level)
                ]

            for _ in range(configs.nn_depth):
                layers.append(
                    Block(n_channels, configs.quantize, configs.pruning, \
                        w_bits = configs.w_bits, a_bits = configs.a_bits, wq_level = configs.wq_level)
                )

            layers.append(
                get_conv(n_channels, c_out, 3, _quantize = configs.quantize, _pruning = False, \
                        w_bits = configs.w_bits, a_bits = configs.a_bits, wq_level = configs.wq_level)
            )

            if configs.quantize:
                layers[1].residual_function[0].input_quantizer = LSQ(configs.a_bits, is_act = False)
                layers[1].shortcut = LSQ(configs.a_bits, is_act = False)
        
        elif nn_type == 'densenet':
            configs.densenet_depth = configs.nn_depth

            layers = [
                DenseBlock(
                    args=configs,
                    n_inputs=c_in,
                    n_outputs=n_channels + c_in)
            ]

            layers += [
                get_conv(n_channels + c_in, c_out, 3, configs.quantize, False, w_bits = configs.w_bits, a_bits = configs.a_bits, wq_level = configs.wq_level)
            ]
            
        else:
            raise ValueError

        self.nn = nn.Sequential(*layers)

        self.nn[-1].weight.data.zero_() 
        self.nn[-1].bias.data.zero_()
 
    def forward(self, x):
        return self.nn(x)