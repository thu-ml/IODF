import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

import torch
from torch import quantization
import numpy as np 

from configs.configuration import get_configs
from core.interfaces.model import load_model
from core.interfaces.init import set_fakequantization_states
from core.utils.logger import get_tag
from tools.tensorrt.engine import build_engine, save_engine


def onnx_to_trt(onnx_path, quantize, bs):
    engine = build_engine(
        max_batch_size=bs,
        fp16_mode=False,
        int8_mode=quantize,
        onnx_file_path=onnx_path,
        calib=None
    )
    return engine

def torch_model_to_onnx(model, onnx_path, batch_size, device):
    dummy_input = torch.randn(batch_size,*model.configs.input_size).float().to(device)
    input_names = ['x']
    output_names = ['output']
    with torch.no_grad():
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

def run_build_tensorrt_engines(args):

    args.coupling_type = args.nn_type
    args.pruning = False
    tag = get_tag(args)

    resolution = 32 if args.dataset == 'imagenet32' else 64

    if os.path.exists(os.path.join(f'assets/trt_pmf_{resolution}', f'{tag}-{args.resume}', f'engine_pmf_bs{args.batchsize}.trt')):
        print('Exist. Next.')
        return 

    cfg = get_configs(
        dataset=args.dataset, 
        nn_type=args.nn_type,
        batch_size=args.batchsize,
        quantize=args.quantize,
        pruned=args.pruned,
        resume=args.resume,
        device=torch.device('cuda:0'),
    )

    cfg.build_engine = True

    model = load_model(cfg)
    model.to(cfg.device)

    if args.quantize:
        set_fakequantization_states(model)

        for module in model.modules():
            if isinstance(module, quantization.FakeQuantize):
                module.scale.data = torch.abs(module.scale.data)
                assert torch.sum(module.zero_point.data) == 0, 'zero point must be 0!'

    onnx_path = os.path.join(cfg.resume, f'model_pmf_bs{args.batchsize}.onnx')

    if not os.path.exists(onnx_path):
        torch_model_to_onnx(
            model, 
            onnx_path,
            args.batchsize,
            cfg.device
        )

    engine_dir = os.path.join(f'assets/trt_pmf_{resolution}', f'{tag}-{args.resume}')
    
    engine = onnx_to_trt(onnx_path, args.quantize, args.batchsize)

    os.makedirs(engine_dir, exist_ok=True)
    with open(os.path.join(engine_dir, 'info.txt'), 'w') as f:
        f.write(onnx_path)

    save_engine(engine, os.path.join(engine_dir, f'engine_pmf_bs{args.batchsize}.trt'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate IODF')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--nn_type', type=str)
    parser.add_argument('--quantize', action='store_true', default=False)
    parser.add_argument('--pruned', action='store_true', default=False)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--eval_onnx', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)

    args = parser.parse_args()

    run_build_tensorrt_engines(args)