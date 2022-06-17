import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

import torch

from configs.configuration import get_configs
from core.models.layers.modules.compactor import CompactorLayer
from core.interfaces.model import load_model
from core.interfaces.data import load_data
from core.utils.dp_utils import set_mask_idx
from core.utils.constants import specify_resolution_for_each_mask
from tools.evaluate import evaluate_analytic_bpd

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--nn_type', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--q', action='store_true', default=False)

    args = parser.parse_args()

    configs, dp_configs = get_configs(
        dataset = args.dataset,
        nn_type=args.nn_type,
        batch_size=500,
        resume=None,
        pruning=True,
        quantize=args.q
    )
    
    pruning_model = load_model(configs)

    configs = get_configs(
        dataset = args.dataset,
        nn_type=args.nn_type,
        batch_size=500,
        resume=args.resume,
        pruning=False,
        quantize=args.q
    )

    state_dict = torch.load(configs.resume + '/best.pth', map_location='cpu')

    # check
    pruning_state_dict = pruning_model.state_dict()
    ks1 = state_dict.keys()
    ks2 = pruning_state_dict.keys()
    for k in ks2:
        if k not in ks1:
            assert 'mask' in k, f'{k} not exist in keys.'
    print('check well.')

    pruning_model.load_state_dict(state_dict, strict=False)

    set_mask_idx(pruning_model)

    num_conv = 0
    num_mask = [0,0,0]
    for name, child_module in pruning_model.named_modules():
        if isinstance(child_module, torch.nn.Conv2d):
            num_conv += 1
        if isinstance(child_module, CompactorLayer):
            print(child_module.get_idx())
            flow_level = specify_resolution_for_each_mask(child_module.get_idx(), configs.dataset)
            if flow_level == 1:
                num_mask[0]+=1
            elif flow_level == 2:
                num_mask[1]+=1
            else:
                num_mask[2]+=1

    print(f"{num_conv} convs, {num_mask} masks.")

    train_loader, val_loader,_ = load_data(configs)
    from core.utils.part_of_dataloader import part_of_dataloader
    val_loader = part_of_dataloader(val_loader, 5)

    configs.device = torch.device('cuda:0')
    pruning_model.to(configs.device)

    bpd = evaluate_analytic_bpd(pruning_model, val_loader, configs)
    print(f"evaluate init-pruning model, bpd: {bpd:.3f}")

    d = configs.resume.replace('q_np', 'q_dp').replace(args.resume, args.resume + '_init_mask')
    os.makedirs(d)
    torch.save(pruning_model.state_dict(),  d+'/best.pth')