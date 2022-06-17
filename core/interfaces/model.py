import os
import torch
from torch import nn
from core.models.layers.modules.lsq import LSQPerChannel

from core.models.model import Model 
from core.models.layers.modules.modules import ConvBuilder
from core.interfaces.init import set_lsq_initialized

def load_model(configs):
    ConvBuilder.reset_conv_cnt()
    model = Model(configs)
    
    model.set_temperature(configs.temperature)
    model.enable_hard_round(configs.hard_round)
    
    if configs.resume is None:
        ## init dependent on data
        pass
    else:
        if configs.pruned: 
            from core.utils.dp_utils import convert_fun
            convert_model = convert_fun(configs.prune_mode)
            with open(os.path.join(configs.resume, 'base_model_params.txt'), 'r') as f:
                base_model_params = f.read()
            model = convert_model(model, base_model_params)

            if configs.quantize and configs.w_bits == 8:
                for child_mo in model.modules():
                    if hasattr(child_mo, 'weight_quantizer') and isinstance(child_mo.weight_quantizer, LSQPerChannel):
                        child_mo.weight_quantizer.step_size = nn.Parameter(torch.ones(child_mo.out_channels), requires_grad=True)

            model.load_state_dict(torch.load(os.path.join(configs.resume, 'best.pth')))
        else:
            model.load_state_dict(torch.load(os.path.join(configs.resume, 'best.pth')))
            
        if configs.quantize:
            ## no need to initialize stepsize in LSQ if checkpoint is loaded.
            set_lsq_initialized(model)

    return model