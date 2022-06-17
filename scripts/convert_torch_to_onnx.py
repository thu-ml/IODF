import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

import torch
from torch import quantization
import numpy as np 

from configs.configuration import get_configs
from core.interfaces.model import load_model
from core.interfaces.data import load_data
from core.interfaces.init import set_fakequantization_states
from tools.onnx.evaluate_onnx import evaluate_onnx


def torch_model_to_onnx(model, onnx_path, batch_size, device):
    dummy_input = torch.randn(batch_size,3,32,32).float().to(device)
    input_names = ['x']
    output_names = ['output']
    import traceback
    try:
        torch.onnx.export(model, dummy_input, onnx_path, verbose=False, \
            input_names=input_names, output_names=output_names, opset_version=13)
        
        print(f"Onnx model saved at {onnx_path}")
    except: 
        with open('torch_to_onnx_error.txt', 'w') as f:
            print(traceback.format_exc(), file=f)
        print('failed to generate onnx file.')
        exit()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Evaluate IODF')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--nn_type', type=str)
    parser.add_argument('--quantize', action='store_true', default=False)
    parser.add_argument('--pruned', action='store_true', default=False)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--eval', action='store_true', default=False)

    args = parser.parse_args()    

    cfg = get_configs(
        dataset=args.dataset, 
        nn_type=args.nn_type,
        batch_size=args.batchsize,
        quantize=args.quantize,
        pruned=args.pruned,
        resume=args.resume,
        device=torch.device('cuda:0'),
    )

    model = load_model(cfg)
    device = torch.device('cuda:0')
    model.to(device)

    if args.quantize:
        set_fakequantization_states(model)

    for module in model.modules():
        if isinstance(module, quantization.FakeQuantize):
            module.scale.data = torch.abs(module.scale.data)
            assert module.zero_point.data == 0, 'zero point must be 0!'

    if not args.eval:
        torch_model_to_onnx(
            model, 
            os.path.join(cfg.resume, f'model_bs{args.batchsize}.onnx'),
            args.batchsize,
            device
        )
    
    # import onnx
    # m = onnx.load(os.path.join(cfg.resume, f'model_bs{args.batchsize}.onnx'))
    # for i, no in enumerate(m.graph.node):
    #     print(i)
    #     print(no)
    # exit()
    
    # import pdb
    # pdb.set_trace()

    _, _, test_loader = load_data(cfg)

    bpds = evaluate_onnx(
        os.path.join(cfg.resume, f'model_bs{args.batchsize}.onnx'),
        data_loader=test_loader,
        z_idx=0
    )

    print(bpds)
    
    # print(f"Evaluate {len(bpds)} samples, average bpd: {np.mean(bpds):.3f}")