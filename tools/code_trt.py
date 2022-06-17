from tabnanny import process_tokens
from tqdm import tqdm
import numpy as np 
import torch
import time

from core.loss.loss import compute_loss_array
from core.coder.apis import encode
from tools.tensorrt.engine import load_engine
from tools.tensorrt.utils import allocate_buffers, do_inference

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

def postprocess_numpy(res, bs):
    mid_z1, y1, py1_mu, py1_logs, mid_z2, y2, py2_mu, py2_logs, z, pz_mu, pz_logs, pz_pi = res
    ys = (y1.reshape(bs, 6, 16, 16), y2.reshape(bs, 12, 8, 8))
    pys = ( 
            [py1_mu.reshape(bs, 6, 16, 16), py1_logs.reshape(bs, 6, 16, 16)],  
            [py2_mu.reshape(bs, 12, 8, 8), py2_logs.reshape(bs, 12, 8, 8)]
        )
    z = z.reshape(bs, 48, 4, 4)
    pz = [
        pz_mu.reshape(bs, 48, 4, 4, 5),
        pz_logs.reshape(bs, 48, 4, 4, 5),
        pz_pi.reshape(bs, 48, 4, 4, 5)
    ]

    return ys, pys, z, pz

def postprocess_pmf_cdf(res):
    # [y1, y2, z]
    cdfs = [res[-6], res[-4], res[-2]]
    pmfs = [res[-5], res[-3], res[-1]]

    # cdfs = [j.astype(np.int) for j in cdfs]
    # pmfs = [j.astype(np.int) for j in pmfs]

    return pmfs, cdfs

def code_with_trt_engine(trt_path, test_data):
    '''
        test_data: numpy.ndarray, [n_batch, batch_size, 3, 32, 32]
    '''
    
    bpds = []
    state_sizes  = []
    N = 0

    from argparse import Namespace
    loss_args = Namespace(
        variable_type = 'discrete', 
        input_size = (3, 32, 32),
        distribution_type='logistic',
        inverse_bin_width=256,
        n_mixtures=5
    )

    engine = load_engine(trt_path)
    assert engine, 'Broken Engine'
    # for binding in engine:
    #     print(binding, engine.get_binding_shape(binding))
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(context.engine)

    infer_ts, rans_ts = [], []

    # warmup
    for x in test_data[:10]:
        inputs[0].host = x
        _ = do_inference(context, bindings, inputs, outputs, len(x))
    
    for i in tqdm(range(len(test_data))):
        x = test_data[i].astype('float32')
        efficient_bs = len(x)

        tic = time.time()
        inputs[0].host = x
        res = do_inference(context, bindings, inputs, outputs, efficient_bs)
    
        infer_ts.append(time.time() - tic)

        pmfs, cdfs = postprocess_pmf_cdf(res)
        
        # with torch.no_grad():
        #     _, bpd, _ = compute_loss_array(pz, z, pys, ys, loss_args)
        
        # bpds.append(bpd)

        tic = time.time()
        ans_coder = encode(pmfs, cdfs, efficient_bs)
        rans_ts.append(time.time() - tic)

        state_sizes += [ans_coder.stream_length()]

        N += efficient_bs

    # analytic_bpd = np.mean(np.concatenate(bpds, axis=0))
    analytic_bpd = 0.
    code_bpd = np.sum(state_sizes) / (np.prod(loss_args.input_size) * N)
    infer_t, rans_t = np.sum(infer_ts)/N, np.sum(rans_ts)/N
    
    return analytic_bpd, code_bpd, infer_t, rans_t