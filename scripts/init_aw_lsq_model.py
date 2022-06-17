import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

import torch
from torch import nn
import numpy as np 

from configs.configuration import get_configs
from core.interfaces.model import load_model
from core.interfaces.init import disable_lsq_weights, enable_lsq, init_lsq_with_data, set_lsq_initialized
from core.interfaces.data import load_data
from core.models.layers.modules.modules import Identity, LSQConv2d
from tools.evaluate import evaluate_analytic_bpd

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--nn_type', type=str)
    parser.add_argument('--pruned', action='store_true', default=False)
    parser.add_argument('--resume', default=None, type=str)

    args = parser.parse_args()

    cfg = get_configs(
        dataset = args.dataset,
        nn_type=args.nn_type,
        quantize=True,
        pruning=False,
        pruned=args.pruned,
        resume=args.resume,
        batch_size=500,
        device=torch.device('cuda:0')
    )

    cfg.w_bits = 8
    assert cfg.w_bits == 8

    tmp = cfg.resume.replace('_awq_', '_aq_')
    cfg.resume = None
    lsq_model = load_model(cfg)
    cfg.resume = tmp

    if args.pruned:
        from core.utils.dp_utils import convert_fun
        convert_model = convert_fun(cfg.prune_mode)
        with open(os.path.join(cfg.resume, 'base_model_params.txt'), 'r') as f:
            base_model_params = f.read()
        convert_model(lsq_model, load_path=base_model_params)
        lsq_model.load_state_dict(torch.load(os.path.join(cfg.resume, 'best.pth')), strict = False)
    else:
        lsq_model.load_state_dict(torch.load(os.path.join(cfg.resume, 'best.pth')), strict=False)

    train_loader, val_loader,_ = load_data(cfg)
    
    from core.utils.part_of_dataloader import part_of_dataloader
    val_loader = part_of_dataloader(val_loader, 5)
    
    set_lsq_initialized(lsq_model)

    ## check 
    disable_lsq_weights(lsq_model)
    lsq_model.to(cfg.device)
    bpd = evaluate_analytic_bpd(lsq_model, val_loader, cfg)
    print(f'Evaluating bpd: {np.mean(bpd):.3f} (lsq disabled)')
    
    enable_lsq(lsq_model)

    ## remove quantization in sensitive layers. 
    # from core.utils.constants import no_lsq_layers
    # conv_cnt = 0
    # for child_module in lsq_model.modules():
    #     if isinstance(child_module, LSQConv2d):
    #         conv_cnt += 1
    #         if conv_cnt in no_lsq_layers:
    #             child_module.input_quantizer = Identity()
    #             print(f'remove lsq in conv {conv_cnt}')
    
    # exit()

    for child_mo in lsq_model.modules():
        if hasattr(child_mo, 'weight_quantizer'):
            if child_mo.weight_quantizer is not None:
                child_mo.weight_quantizer.initialized = False
                if args.pruned:
                    child_mo.weight_quantizer.step_size = nn.Parameter(torch.ones(child_mo.out_channels), requires_grad=True)
    
    lsq_model.to(cfg.device)

    print('Initializing LSQ')
    init_lsq_with_data(lsq_model, train_loader, cfg.device)
        
    bpd = evaluate_analytic_bpd(lsq_model, val_loader, cfg)
    print(f"evaluate lsq model, bpd: {np.mean(bpd):.3f}")

    save_path = cfg.resume.replace('_aq_', '_awq_').replace(f'{args.resume}', f'{args.resume}_init_{cfg.a_bits}bit')
    os.makedirs(save_path)
    if args.pruned:
        import shutil 
        shutil.copyfile(os.path.join(cfg.resume, 'base_model_params.txt'), os.path.join(save_path, 'base_model_params.txt'))
    torch.save(lsq_model.state_dict(),  os.path.join(save_path, 'best.pth'))