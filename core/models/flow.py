from __future__ import print_function

import torch
import torch.nn as nn

from .layers.base import Base
from .layers.priors import SplitPrior
from .layers.coupling import Coupling
from .layers.transforms import Squeeze, Permute
from configs.configs_wrapper import IODFConfig

class Flow(Base):
    def __init__(self, n_channels, height, width, configs:IODFConfig):
        super().__init__()
        layers = []
        layers.append(Squeeze())
        n_channels *= 4
        height //= 2
        width //= 2

        for level in range(configs.n_levels):
            configs.n_channels = configs.n_channels_list[level]
            for i in range(configs.n_flows):
                perm_layer = Permute(n_channels)
                layers.append(perm_layer)

                layers.append(
                    Coupling(n_channels, height, width, configs))

            if level < configs.n_levels - 1:
                if configs.splitprior_type != 'none':
                    # Standard splitprior
                    factor_out = n_channels // 2
                    layers.append(SplitPrior(n_channels, factor_out, height, width, configs))
                    n_channels = n_channels - factor_out

                layers.append(Squeeze())
                n_channels *= 4
                height //= 2
                width //= 2

        self.layers = nn.ModuleList(layers)
        self.z_size = (n_channels, height, width)

    def forward(self, z, pys=(), ys=(), reverse=False):
        mid_z = []
        if not reverse:
            for l, layer in enumerate(self.layers):
                if isinstance(layer, (SplitPrior)):
                    py, y, z = layer(z)
                    pys += (py,)   # py is [mu(z), logs(z)] which is used to compute log p(y), z is passed to next layer while y is dropped. 
                    ys += (y,)
                    mid_z.append(z)
                else:
                    z = layer(z)

        else:
            for l, layer in reversed(list(enumerate(self.layers))):
                if isinstance(layer, (SplitPrior)):
                    if len(ys) > 0:
                        z = layer.inverse(z, y=ys[-1])
                        # Pop last element
                        ys = ys[:-1]
                    else:
                        z = layer.inverse(z, y=None)

                else:
                    z = layer(z, reverse=True)

        return z, pys, ys, mid_z

    def decode(self, z, state, decode_fn):

        for l, layer in reversed(list(enumerate(self.layers))):
            if isinstance(layer, SplitPrior):
                z, state = layer.decode(z, state, decode_fn)

            else:
                z = layer(z, reverse=True)

        return z
