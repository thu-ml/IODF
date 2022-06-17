"""
Collection of flow strategies
"""

from __future__ import print_function

import torch
import numpy as np

from .base import Base
from .networks import NN


UNIT_TESTING = False

class SplitFactorCoupling(Base):
    def __init__(self, c_in, factor, height, width, configs):
        super().__init__()
        self.n_channels = configs.n_channels
        self.kernel = 3
        self.input_channel = c_in
        self.round_approx = configs.round_approx

        self.split_idx = c_in - (c_in // factor)
        
        self.nn = NN(
            configs, 
            c_in=self.split_idx, 
            c_out=c_in - self.split_idx,
            nn_type=configs.coupling_type
        )

    def forward(self, z, reverse=False):
        z1 = z[:, :self.split_idx, :, :]
        z2 = z[:, self.split_idx:, :, :]

        t = self.nn(z1)

        t = torch.floor(t*256 + 0.5)
        t /= 256.

        if not reverse:
            z2 = z2 + t
        else:
            z2 = z2 - t

        z = torch.cat([z1, z2], dim=1)

        return z


class Coupling(Base):
    def __init__(self, c_in, height, width, configs):
        super().__init__()

        if configs.split_quarter:
            factor = 4
        elif configs.splitfactor > 1:
            factor = configs.splitfactor
        else:
            factor = 2

        self.coupling = SplitFactorCoupling(
            c_in, factor, height, width, configs=configs)

    def forward(self, z, reverse=False):
        return self.coupling(z, reverse)

