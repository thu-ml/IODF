import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

import torch

from configs.configuration import get_configs
from core.interfaces.model import load_model
from core.interfaces.data import load_data
from tools.evaluate import evaluate_analytic_bpd

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--nn_type', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--resume', default=None, type=str)

    args = parser.parse_args()

    configs, dp_configs = get_configs(
        dataset = args.dataset,
        nn_type=args.nn_type,
        batch_size=500,
        resume=None,
        pruning=True,
        quantize=False
    )
    
    pruning_model = load_model(configs)

    configs = get_configs(
        dataset = args.dataset,
        nn_type=args.nn_type,
        batch_size=500,
        resume=args.resume,
        pruning=False,
        quantize=False
    )

    state_dict = torch.load(configs.resume + '/best.pth', map_location='cpu')

    pruning_model.load_state_dict(state_dict, strict=False)

    conv_cnt = 0
    pruned_conv_cnt = 0
    for name, child_module in pruning_model.named_modules():
        if isinstance(child_module, torch.nn.Conv2d):
            conv_cnt += 1
        if hasattr(child_module, 'compactor'):
            pruned_conv_cnt += 1
            print(child_module.compactor.conv_idx)

    print(f"{conv_cnt} convs in total, prune {pruned_conv_cnt} of them.")

    train_loader, val_loader,_ = load_data(configs)
    from core.utils.part_of_dataloader import part_of_dataloader
    val_loader = part_of_dataloader(val_loader, 5)

    configs.device = torch.device('cuda:0')
    pruning_model.to(configs.device)

    bpds = evaluate_analytic_bpd(pruning_model, val_loader, configs)
    print(f"evaluate init-pruning model, bpd: {bpds.mean():.3f}")

    d = configs.resume.replace(args.resume, args.resume+'_init').replace('q_np', 'q_dp')
    os.makedirs(d)
    torch.save(pruning_model.state_dict(),  d+'/best.pth')