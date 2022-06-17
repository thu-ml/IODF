from argparse import Namespace
import os
import logging

from tools.tensorrt.engine import load_engine
from tools.tensorrt.utils import do_inference, allocate_buffers
from tools.tensorrt.utils import load_data_array
from tools.tensorrt.utils.inference import do_inference_async
from tools.tensorrt.utils.profiler import MyProfiler
from core.utils.logger import set_logger

profiling = False

def test_inference_latency(args):

    _, test_data = load_data_array(args)
    test_data = test_data[:100]

    resolution = 32 if args.dataset == 'imagenet32' else 64
    path_to_engine = os.path.join('assets', f'trt_pmf_{resolution}', args.engine_path)

    engine = load_engine(os.path.join(path_to_engine, f'engine_pmf_bs{args.batch_size}.trt'))
    assert engine, 'Broken Engine'
    context = engine.create_execution_context()

    if profiling:
        context.profiler = MyProfiler(None)

    inputs, outputs, bindings, stream = allocate_buffers(context.engine)

    ## warmup GPU
    for i in range(len(test_data[:10])):
        x = test_data[i].astype('float32')
        bs = len(x)
        inputs[0].host = x
        _ = do_inference(context, bindings, inputs, outputs, bs, timing=False)

    import pycuda.driver as cuda
    start, end = cuda.Event(), cuda.Event()

    if profiling:
        context.profiler.time = 0.

    N = 0.
    
    start.record()
    for i in range(len(test_data)):
        x = test_data[i].astype('float32')
        bs = len(x)
        inputs[0].host = x
        # res = do_inference_async(context, bindings, inputs, outputs, stream, bs)
        res = do_inference(context, bindings, inputs, outputs, bs, timing=False)
        N += bs
    end.record()
    end.synchronize()

    t = 1e-3 * start.time_till(end)
    
    t_profile = 0
    if profiling:
        t_profile = context.profiler.time

    logging.info(f'Engine: {path_to_engine}')
    logging.info(f"batch size: {args.batch_size}, {N} samples:\n{t*1000./N:.3f} ms per sample, execute {t_profile * 1e3 / N:.3f} ms per sample.")
    logging.info('='*20)

if __name__ == '__main__':
    datapaths = {'imagenet32': '../DATASETS/imagenet32x32', 'imagenet64': '../DATASETS/imagenet64x64'}
    dataset = 'imagenet32'
    bs = 32
    engine_path = 'resnet_nq_np-base',
    
    resolution = 32 if dataset == 'imagenet32' else 64
    set_logger(f'./assets/trt_pmf_{resolution}/latency.log')
    logging.info('\n' + '='*10 + 'A set of Experiments' + '='*10)
    logging.info('Use Async inference.')

    args = Namespace(
        dataset = dataset,
        batch_size = bs,
        engine_path = engine_path,
        num_workers = 4,
        data_path = datapaths[dataset]
    )
    try:
        test_inference_latency(args)
    except:
        print(f'Something is wrong with {str(args)}')