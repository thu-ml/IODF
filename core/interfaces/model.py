import os
import torch

from core.models.model import Model 

def load_model(configs):
    model = Model(configs)
    
    model.set_temperature(configs.temperature)
    model.enable_hard_round(configs.hard_round)
    
    if configs.resume is None:
        pass
    else:
        if configs.pruned: 
            from core.utils.dp_utils import convert_fun
            convert_model = convert_fun(configs.prune_mode)
            with open(os.path.join(configs.resume, 'base_model_params.txt'), 'r') as f:
                base_model_params = f.read()
            model = convert_model(model, load_path=base_model_params)
            model.load_state_dict(torch.load(os.path.join(configs.resume, 'best.pth')))
        else:
            model.load_state_dict(torch.load(os.path.join(configs.resume, 'best.pth')))

    return model