from tqdm import tqdm
import numpy as np 
import torch

from core.loss.loss import compute_loss_array
from tools.tensorrt.engine import load_engine
from tools.tensorrt.utils import allocate_buffers, do_inference


def evaluate_trt(trt_path, batch_size, test_data, z_idx = -2):
    '''
        test_data: numpy.ndarray, [n_batch, batch_size, 3, 32, 32]
    '''
    
    zs = []
    bpds = []

    print('Evaluating tensorrt engine {}'.format(trt_path))
    engine = load_engine(trt_path)
    assert engine, 'Broken Engine'
    # for binding in engine:
    #     print(binding, engine.get_binding_shape(binding))
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(context.engine)
    
    for i in tqdm(range(len(test_data))):
        x = test_data[i].astype('float32')
        efficient_bs = len(x)
        inputs[0].host = x
        res = do_inference(context, bindings, inputs, outputs, efficient_bs)
        z = res[z_idx].reshape(batch_size, -1)
        bpd = res[-1].reshape(batch_size, -1)
        zs.append(np.copy(z))
        bpds.append(np.copy(bpd))

    bpds = np.concatenate(bpds, axis=0)

    print(bpds)
    
    return bpds


def postprocess(res, bs):

    mid_z1, y1, py1_mu, py1_logs, mid_z2, y2, py2_mu, py2_logs, z, pz_mu, pz_logs, pz_pi = res
    ys = (torch.from_numpy(y1.reshape(bs, 6, 16, 16)), torch.from_numpy(y2.reshape(bs, 12, 8, 8)))
    pys = ( 
            [torch.from_numpy(py1_mu.reshape(bs, 6, 16, 16)), torch.from_numpy(py1_logs.reshape(bs, 6, 16, 16))],  
            [torch.from_numpy(py2_mu.reshape(bs, 12, 8, 8)), torch.from_numpy(py2_logs.reshape(bs, 12, 8, 8))]
        )
    z = torch.from_numpy(z.reshape(bs, 48, 4, 4))
    pz = [
        torch.from_numpy(pz_mu.reshape(bs, 48, 4, 4, 5)),
        torch.from_numpy(pz_logs.reshape(bs, 48, 4, 4, 5)),
        torch.from_numpy(pz_pi.reshape(bs, 48, 4, 4, 5))
    ]

    return ys, pys, z, pz

def evaluate_trt_noloss(trt_path, batch_size, test_data):
    
    bpds = []
    from argparse import Namespace
    loss_args = Namespace(
        variable_type = 'discrete', 
        input_size = (3, 32, 32),
        distribution_type='logistic',
        inverse_bin_width=256,
        n_mixtures=5
    )

    print('Evaluating tensorrt engine {}'.format(trt_path))
    engine = load_engine(trt_path)
    assert engine, 'Broken Engine'
    # for binding in engine:
    #     print(binding, engine.get_binding_shape(binding))
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(context.engine)
    
    for i in tqdm(range(len(test_data))):
        x = test_data[i].astype('float32')
        efficient_bs = len(x)
        inputs[0].host = x
        res = do_inference(context, bindings, inputs, outputs, efficient_bs)
        ys, pys, z, pz = postprocess(res, efficient_bs)
        with torch.no_grad():
            _, bpd, _ = compute_loss_array(pz, z, pys, ys, loss_args)
        bpds.append(bpd)

    bpds = np.concatenate(bpds, axis=0)
    
    return bpds