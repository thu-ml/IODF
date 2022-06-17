import torch
import numpy as np

from .flow import Flow 
from .layers.base import Base
from .layers.priors import Prior
from core.loss.loss import compute_loss_array
from configs.configs_wrapper import IODFConfig

class Normalize(Base):
    
    def __init__(self, configs):
        super().__init__()
        self.n_bits = configs.n_bits
        self.domain = configs.inverse_bin_width
        self.variable_type = configs.variable_type
        self.input_size = configs.input_size

    def forward(self, x, reverse=False):
        domain = 2.**self.n_bits

        if not reverse:
            x = (x - domain // 2) / domain 
        else:
            x = x * domain + domain // 2

        return x

class Model(Base):

    def __init__(self, configs:IODFConfig):
        super().__init__()
        self.configs = configs
        self.variable_type = configs.variable_type
        self.distribution_type = configs.distribution_type

        n_channels, height, width = configs.input_size

        self.normalize = Normalize(configs)

        self.flow = Flow(
            n_channels, height, width, configs)

        self.n_bits = configs.n_bits

        self.z_size = self.flow.z_size

        self.prior = Prior(self.z_size, configs) 

    def loss(self, pz, z, pys, ys):
        batchsize = z.size(0)
        loss, bpd, bpd_per_prior = \
            compute_loss_array(pz, z, pys, ys, self.configs)

        for module in self.modules():
            if hasattr(module, 'auxillary_loss'):
                loss += module.auxillary_loss() / batchsize

        return loss, bpd, bpd_per_prior

    def forward(self, x):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log
         det jacobian is zero for a plain VAE (without flows), and z_0 = z_k.
        """
        # Decode z to x.

        assert x.dtype == torch.uint8

        x = x.float()

        assert self.variable_type == 'discrete'

        x = self.normalize(x)

        z, pys, ys = self.flow(x, pys=(), ys=())

        pz, z = self.prior(z)  # pz is mu, logs, pi. refer to original paper. 

        loss, bpd, bpd_per_prior = self.loss(pz, z, pys, ys)

        return loss, bpd, bpd_per_prior, pz, z, pys, ys

    def inverse(self, z, ys):
        x, pys, py = \
            self.flow(z, pys=[], ys=ys, reverse=True)

        x = self.normalize(x, reverse=True)

        x_uint8 = torch.clamp(x, min=0, max=255).to(
                torch.uint8)

        return x_uint8

    def sample(self, n):
        z_sample = self.prior.sample(n)

        x_sample, pys, py = \
            self.flow(z_sample, pys=[], ys=[], reverse=True)

        x_sample = self.normalize(x_sample, reverse=True)

        x_sample_uint8 = torch.clamp(x_sample, min=0, max=255).to(
                torch.uint8)

        return x_sample_uint8
