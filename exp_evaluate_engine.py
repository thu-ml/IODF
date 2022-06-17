from argparse import Namespace
import os
import numpy as np 

from tools.tensorrt.utils import load_data_array
from tools.evaluate_trt import evaluate_trt_noloss
from core.utils.logger import set_logger
from core.datasets.wild import load_wild

def eval_trt(args):

    _, test_data = load_data_array(args)
    test_data = test_data[:100]

    resolution = 32 if args.dataset == 'imagenet32' else 64

    engine_dir = os.path.join('assets', f'trt_pmf_{resolution}', f'{args.engine_name}')

    set_logger(os.path.join(engine_dir, 'bpd.log'))

    bpds = evaluate_trt_noloss(os.path.join(engine_dir, f'engine_pmf_bs{args.batch_size}.trt'), args.batch_size, test_data)

    with open(os.path.join(engine_dir, 'high_resolution_a_bpd.txt'), 'w') as f:
        f.write(f'Evaluae engine_pmf_bs{args.batch_size}.trt, average analytic bpd: {np.mean(bpds):.3f}')


if __name__ == '__main__':

    datapaths = {'imagenet32': '../DATASETS/imagenet32x32', 'imagenet64': '../DATASETS/imagenet64x64'}
    dataset = 'imagenet32'
    engine_name = 'resnet_q_p-conv1_ep24_ft_asym'

    args = Namespace(
        dataset = dataset,
        batch_size = 2025,
        engine_name = engine_name,
        num_workers = 4,
        data_path = datapaths[dataset]
    )

    eval_trt(args)