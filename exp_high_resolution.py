import time
import os
from argparse import Namespace

import torch
import numpy as np 

from core.datasets.wild import load_wild
from tools.code_trt import code_with_trt_engine 
from tools.tensorrt.utils import load_data_array

def encode_wild(engine_dir, size, reso = '2k', no_decode=True):

    test_data = load_wild(reso, size)
    bs = test_data.shape[0]
    test_data = np.ascontiguousarray(test_data.reshape(1, *test_data.shape)).astype(np.float)

    a_bpd, c_bpd, t_infer, t_rans = code_with_trt_engine(os.path.join(engine_dir, f'engine_pmf_bs{bs}.trt'), test_data)
    t_encode = t_infer + t_rans 

    bandwidth = np.prod(test_data.shape) / t_encode / 1e6

    return a_bpd, c_bpd, t_encode, t_infer, t_rans, bandwidth 


if __name__ == '__main__':

    engine_dir = 'assets/trt_pmf_32/resnet_q_p-IODF/'

    input_size = (3,32,32)
    
    a_bpd, c_bpd, t_encode, t_inference, t_rans, bandwidth = encode_wild(engine_dir, 32, '4k')

    import os
    with open(os.path.join(engine_dir, '..', 'encode_high_resolution.txt'), 'w') as f:
        print(f'coding bpd: {a_bpd:.3f}', file = f)
        print(f'coding bpd: {c_bpd:.3f}, compression ratio: {8 / c_bpd :2f}', file=f)
        print(f'Latency: {t_encode:.3f}s, inference: {t_inference:.3f}, rans: {t_rans:.3f}', file=f)
        print(f'Bandwidth: {bandwidth:.2f} MB/s', file=f)