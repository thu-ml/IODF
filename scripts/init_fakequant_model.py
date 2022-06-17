import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

import torch
from torch import quantization

from configs.configuration import get_configs
from core.interfaces.data import load_data
from core.interfaces.model import load_model
from tools.evaluate import evaluate_analytic_bpd

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate IODF')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--nn_type', type=str)
    parser.add_argument('--pruned', action='store_true', default=False)
    parser.add_argument('--from_lsq', action='store_true', default=False)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--batchsize', type=int)

    args = parser.parse_args()    

    cfg = get_configs(
        dataset=args.dataset, 
        nn_type=args.nn_type,
        batch_size=args.batchsize,
        quantize=False,
        pruned=args.pruned,
        resume=args.resume,
    )

    resume = cfg.resume 

    cfg = get_configs(
        dataset=args.dataset, 
        nn_type=args.nn_type,
        batch_size=args.batchsize,
        quantize=True,
        pruned=args.pruned,
        resume=None,
        device=torch.device('cuda:0'),
    )

    model = load_model(cfg)

    if args.pruned:
        from core.utils.dp_utils import convert_fun
        convert_model = convert_fun(cfg.prune_mode)
        with open(os.path.join(resume, 'base_model_params.txt'), 'r') as f:
            base_model_params = f.read()
        model = convert_model(model, load_path=base_model_params)
    
    model.load_state_dict(torch.load(os.path.join(resume, 'best.pth')), strict=False)

    _, _, tl= load_data(cfg)

    from core.utils.part_of_dataloader import part_of_dataloader
    tl = part_of_dataloader(tl, 5)

    model.to(cfg.device)

    model.apply(quantization.disable_fake_quant)
    bpd = evaluate_analytic_bpd(model, tl, cfg)
    print('quantization disabled: ', bpd)

    if args.from_lsq:
        # load lsq scale
        # model.load(torch.load(os.path.join(cfg.resume, 'best.pth')))
        pass
    
    else:
        # use observer to initialize fakequantize    
        model.apply(quantization.enable_fake_quant)
        model.apply(quantization.enable_observer)

        model.train()
        with torch.no_grad():
            for data, _ in tl:
                model(data.to(cfg.device))
                break

    for n, module in model.named_modules():
        if isinstance(module, quantization.FakeQuantize):
            module.scale.data.copy_(torch.abs(module.scale.data)+1e-7)
            module.zero_point.data.zero_()

    model.apply(quantization.disable_observer)

    model.eval()
    
    bpd = evaluate_analytic_bpd(model, tl, cfg)
    print(bpd)

    save_path = resume.replace('_nq_', '_q_').replace(args.resume, args.resume + '_8bit')
    os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))
    if args.pruned:
        import shutil
        shutil.copyfile(os.path.join(resume, 'base_model_params.txt'), os.path.join(save_path, 'base_model_params.txt'))