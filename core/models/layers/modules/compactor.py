import torch
import torch.nn as nn
import numpy as np

class binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input): 
        num_features = len(input)
        grad_mask = torch.logical_and(input>=0, input<=1.).view(1, num_features, 1, 1).float()
        ctx.save_for_backward(grad_mask)
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_mask, = ctx.saved_tensors
        return grad_output * grad_mask 

class CompactorLayer(nn.Module): 

    def __init__(self, num_features, conv_idx=None):
        super(CompactorLayer, self).__init__()
        self.conv_idx = conv_idx
        self.pwc = nn.Parameter(torch.ones(num_features) * 0.9, requires_grad=True)
        self.num_features = num_features
        self.mask_idx = 0
    
    def get_num_zeros_ones_mask(self):
        return (self.mask == 0).float().sum(), (self.mask == 1).float().sum()
    
    def get_mask(self):
        return binarize.apply(self.pwc.detach().cpu())

    def forward(self, x):
        mask = binarize.apply(self.pwc)
        return mask.view(1, self.num_features, 1, 1) * x
    
    def get_metric_vector(self):
        return self.pwc.detach()

    def get_idx(self):
        return self.mask_idx 