from argparse import Namespace
import os
import logging

import numpy as np 

from tools.tensorrt.utils import load_data_array
from tools.code_trt import code_with_trt_engine 
from core.utils.logger import set_logger

def code_trt(args):

    _, test_data = load_data_array(args)
    test_data = test_data[:100]

    resolution = 32 if args.dataset == 'imagenet32' else 64

    engine_dir = os.path.join('assets', f'trt_pmf_{resolution}', f'{args.engine_name}')

    a_bpd, c_bpd, t_infer, t_rans = code_with_trt_engine(os.path.join(engine_dir, f'engine_pmf_bs{args.batch_size}.trt'), test_data)
    t_encode = t_infer + t_rans

    bandwidth = (3 * resolution ** 2) / 1e6 / t_encode

    # return a_bpd, c_bpd, t_encode, t_infer, t_rans, bandwidth
    logging.info('=' * 40)
    logging.info(f'Batch size: {args.batch_size}. {len(test_data) * args.batch_size} samples, analytic bpd: {np.mean(a_bpd):.3f}, coding bpd: {np.mean(c_bpd):.3f}')
    logging.info(f'total: [{t_encode * 1e3:.3f}] inference: [{t_infer * 1e3:.3f}], rans: [{t_rans * 1e3:.3f}] ms / sample')
    logging.info(f'Encode a batch. total: [{t_encode * args.batch_size:.3f}] inference: [{t_infer * args.batch_size:.3f}], rans: [{t_rans * args.batch_size:.3f}] s.')
    logging.info(f'Bandwidth: {bandwidth:.4f} MB/s')

if __name__ == '__main__':

    datapaths = {'imagenet32': '../DATASETS/imagenet32x32', 'imagenet64': '../DATASETS/imagenet64x64'}
    dataset = 'imagenet32'
    bs = 32
    engine_name = 'resnet_nq_np-base',

    resolution = 32 if dataset == 'imagenet32' else 64

    set_logger(f'assets/trt_pmf_{resolution}/compression_latency.log')
    logging.info('=' * 20)

    args = Namespace(
        dataset = dataset,
        batch_size = bs,
        engine_name = engine_name,
        num_workers = 4,
        data_path = datapaths[dataset]
    )

    code_trt(args)