import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 

class lsq_weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, bits):
        num_bins = 2**bits - 1  # 255
        bias = - (num_bins // 2) - 1 # Q_N = 128, Q_P = 127, bias = -Q_N
        num_features = input.numel()
        grad_scale = 1.0 / np.sqrt(num_features)

        # Forward
        eps = 1e-7
        scale = scale + eps
        transformed = input / scale - bias
        vbar = torch.clamp(transformed, 0.0, num_bins).round()
        quantized = (vbar + bias) * scale 

        # step size gradient 
        error = vbar - transformed
        mask = torch.logical_and(transformed >= 0, transformed <= num_bins)
        case1 = (transformed < 0).float() * bias   
        case2 = mask.float() * error
        case3 = (transformed > num_bins).float() * (bias + num_bins) 
        ss_gradient = (case1 + case2 + case3) * grad_scale
        ctx.save_for_backward(mask, ss_gradient)
        
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        mask, ss_gradient = ctx.saved_tensors
        return grad_output * mask.float(), (grad_output * ss_gradient).sum(), None

class lsq_act(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, bits):
        num_bins = 2**bits - 1
        num_features = input.numel()
        grad_scale = 1.0 / np.sqrt(num_features)

        # Forward
        eps = 1e-7
        scale = scale + eps
        transformed = input / scale
        vbar = torch.clamp(transformed, 0.0, num_bins).round()
        quantized = vbar * scale 

        # step size gradient 
        error = vbar - transformed
        mask = torch.logical_and(transformed >= 0, transformed <= num_bins)
        case2 = mask.float() * error
        case3 = (transformed > num_bins).float() * num_bins
        ss_gradient = (case2 + case3) * grad_scale
        ctx.save_for_backward(mask, ss_gradient)
        
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        mask, ss_gradient = ctx.saved_tensors
        return grad_output * mask.float(), (grad_output * ss_gradient).sum(), None

class LSQ(nn.Module):
    def __init__(self, bits, is_act = True):
        super().__init__()
        self.bits = bits
        self.is_act = is_act
        self.step_size = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.initialized = False
        self.enable_lsq = True
    
    def forward(self, x):
        if not self.enable_lsq:
            return x 

        if not self.initialized:
            with torch.no_grad():
                num_bins = 2**self.bits-1
                self.step_size.data.copy_(2 * x.abs().mean() / np.sqrt(num_bins))
                self.initialized = True
                print(f'Initializing step size to {self.step_size.item():.6f}')
        if self.is_act:
            return lsq_act().apply(x, self.step_size, self.bits)
        else:
            return lsq_weight().apply(x, self.step_size, self.bits)

class lsq_quantize_perchannel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, bits):
        scale = scale.view(-1, 1, 1, 1)
        num_bins = 2 ** bits - 1
        bias = - (num_bins // 2) - 1
        num_features = input.numel() / input.shape[0]
        grad_scale = 1.0 / np.sqrt(num_features * num_bins)

        # Forward
        eps = 1e-7
        scale = scale + eps
        transformed = input / scale - bias
        vbar = torch.clamp(transformed, 0.0, num_bins).round()
        quantized = (vbar + bias) * scale

        # Step size gradient
        error = vbar - transformed
        mask = torch.logical_and(transformed >= 0, transformed <= num_bins)
        case1 = (transformed < 0).float() * bias
        case2 = mask.float() * error
        case3 = (transformed > num_bins).float() * (bias + num_bins)
        ss_gradient = (case1 + case2 + case3) * grad_scale #* 100 * scale
        ctx.save_for_backward(mask, ss_gradient)
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        mask, ss_gradient = ctx.saved_tensors
        return grad_output * mask.float(), \
               (grad_output * ss_gradient).sum([1, 2, 3]), None

class LSQPerChannel(nn.Module):
    def __init__(self, num_channels, bits):
        super(LSQPerChannel, self).__init__()
        self.bits = bits
        self.step_size = nn.Parameter(torch.ones(num_channels), requires_grad=True)
        self.initialized = False
        self.enable_lsq = True

    def forward(self, x):
        if not self.enable_lsq:
            return x 
        if not self.initialized:
            with torch.no_grad():
                num_bins = 2 ** self.bits - 1
                self.step_size.copy_(2 * x.abs().mean([1,2,3]) / np.sqrt(num_bins))
                self.step_size.clamp_(1e-5, 1)
                self.initialized = True
                print(f'Initializing step size to {self.step_size.mean().item()}')

        return lsq_quantize_perchannel().apply(x, self.step_size.abs(), self.bits)
