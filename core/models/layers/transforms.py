import numpy as np 
import torch

from .base import Base

def space_to_depth(x):
    xs = x.size()
    # Pick off every second element
    x = x.view(xs[0], xs[1], xs[2] // 2, 2, xs[3] // 2, 2)
    # Transpose picked elements next to channels.
    x = x.permute((0, 1, 3, 5, 2, 4)).contiguous()
    # Combine with channels.
    x = x.view(xs[0], xs[1] * 4, xs[2] // 2, xs[3] // 2)
    return x


def depth_to_space(x):
    xs = x.size()
    # Pick off elements from channels
    x = x.view(xs[0], xs[1] // 4, 2, 2, xs[2], xs[3])
    # Transpose picked elements next to HW dimensions.
    x = x.permute((0, 1, 4, 2, 5, 3)).contiguous()
    # Combine with HW dimensions.
    x = x.view(xs[0], xs[1] // 4, xs[2] * 2, xs[3] * 2)
    return x


def int_shape(x):
    return list(map(int, x.size()))


class Flatten(Base):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Reshape(Base):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class Reverse(Base):
    def __init__(self):
        super().__init__()

    def forward(self, z, reverse=False):
        flip_idx = torch.arange(z.size(1) - 1, -1, -1).long()
        z = z[:, flip_idx, :, :]
        return z


class Permute(Base):
    def __init__(self, n_channels):
        super().__init__()

        permutation = np.arange(n_channels, dtype='int')
        np.random.shuffle(permutation)

        permutation_inv = np.zeros(n_channels, dtype='int')
        permutation_inv[permutation] = np.arange(n_channels, dtype='int')

        self.register_buffer('permutation', torch.tensor(permutation))
        self.register_buffer('permutation_inv', torch.tensor(permutation_inv))

    def forward(self, z, reverse=False):
        if not reverse:
            z = z[:, self.permutation, :, :]
        else:
            z = z[:, self.permutation_inv, :, :]

        return z

    def InversePermute(self):
        inv_permute = Permute(len(self.permutation))
        inv_permute.permutation = self.permutation_inv
        inv_permute.permutation_inv = self.permutation
        return inv_permute


class Squeeze(Base):
    def __init__(self):
        super().__init__()

    def forward(self, z, reverse=False):
        if not reverse:
            z = space_to_depth(z)
        else:
            z = depth_to_space(z)
        return z