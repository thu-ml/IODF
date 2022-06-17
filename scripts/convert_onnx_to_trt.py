import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

import numpy as np 

from configs.configuration import get_configs
from tools.tensorrt.engine import build_engine, save_engine
from tools.evaluate_trt import evaluate_trt
from tools.tensorrt.utils.load_data_array import load_data_array

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--nn_type', type=str)
    parser.add_argument('--quantize', action='store_true', default=False)
    parser.add_argument('--pruned', action='store_true', default=False)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--eval', default=False, action='store_true')

    args = parser.parse_args()

    configs = get_configs(
        dataset=args.dataset,
        nn_type=args.nn_type,
        quantize=args.quantize,
        pruned=args.pruned,
        batch_size=args.batchsize,
        resume=args.resume
    )

    onnx_path = os.path.join(configs.resume, f'model_bs{args.batchsize}.onnx')

    path_to_engine = configs.resume.replace('imagenet32', 'trt_engines_imagenet32').replace('/checkpoints', '')
    os.makedirs(path_to_engine, exist_ok=True)
    trt_path = os.path.join(path_to_engine, f'engine_bs{args.batchsize}.trt')

    int8_mode = True if args.quantize else False

    if not args.eval:
        engine = build_engine(
            max_batch_size=args.batchsize,
            fp16_mode=False,
            int8_mode=int8_mode,
            onnx_file_path=onnx_path,
            calib = None
        )

        save_engine(engine, trt_path)
        print('TensorRT model saved at {}'.format(trt_path))

    _, test_data = load_data_array(configs)
    test_data = test_data[:10]

    bpds = evaluate_trt(trt_path, args.batchsize, test_data)
    print(f"Evaluate {len(bpds)} samples, average bpd: {np.mean(bpds):.3f}")