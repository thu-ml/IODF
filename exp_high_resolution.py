import time
from argparse import Namespace
from tqdm import tqdm

import torch
import numpy as np 

from core.datasets.wild import load_wild
from core.datasets.clic import load_clic
from core.coder.apis import encode, decode
from core.interfaces.model import load_model
from configs.configuration import get_configs

def encode_high_resolution(model, input_size, dataset, no_decode=True):
    reso = input_size[-1]
    if dataset == 'wild':
        img_array = load_wild('2k', reso)
        img_array = np.array([img_array])
    elif dataset == 'clic':
        img_array = load_clic(reso)

    device = torch.device('cuda:0')
    model.to(device)
    model.eval()

    a_bpds = []
    c_bpds = []
    bandwidths = []

    with torch.no_grad():
        for data in tqdm(img_array):
            data = torch.from_numpy(data).to(torch.uint8).to(device)

            tic_encode = time.time()

            _, bpd, _, pz, z, pys, ys= model(data)

            t_inference = time.time() - tic_encode

            a_bpds.append(bpd.mean().item())
            
            tic = time.time()
            ans_coder = encode(pz, z, pys, ys)
            
            t_rans = time.time() - tic

            t_encode = time.time() - tic_encode

            state_size = ans_coder.stream_length()

            c_bpd = state_size / (np.prod(input_size) * len(data))

            c_bpds.append(c_bpd)

            bandwidth = np.prod(data.shape) / 1e6 / t_encode

            bandwidths.append(bandwidth)

            if not no_decode:

                tic = time.time()
                
                x_recon = decode(model, ans_coder).cpu()
                
                t_decode = time.time() - tic

                error = torch.sum(torch.abs(x_recon.int() - data.cpu().int())) .item()
        
                print(error)
                print(t_decode)

        return a_bpds, c_bpds, bandwidths

def encode_wild(model, input_size, reso = '4k', no_decode=True):
    
    img_array = load_wild(reso = reso, size = input_size[-1])

    print(img_array.shape[0])

    img_tensor = torch.from_numpy(img_array).to(torch.uint8)

    device = torch.device('cuda:0')
    model.to(device)

    model.eval()
    
    with torch.no_grad():

        data = img_tensor.to(device)

        tic_encode = time.time()

        _, bpd, _, pz, z, pys, ys= model(data)

        t_inference = time.time() - tic_encode
        
        tic = time.time()
        ans_coder = encode(pz, z, pys, ys)
        
        t_rans = time.time() - tic

        t_encode = time.time() - tic_encode

        state_size = ans_coder.stream_length()


        if not no_decode:

            tic = time.time()
            
            x_recon = decode(model, ans_coder).cpu()
            
            t_decode = time.time() - tic

            error = torch.sum(torch.abs(x_recon.int() - data.cpu().int())) .item()
    
            print(error)
            print(t_decode)
    
    a_bpd = np.mean(bpd.detach().cpu().numpy())
    
    c_bpd = state_size / (np.prod(input_size) * len(img_array))

    bandwidth = np.prod(img_array.shape) / 1e6 / t_encode
    
    return a_bpd, c_bpd, t_encode, t_inference, t_rans, bandwidth


if __name__ == '__main__':
    args = Namespace(
        dataset = 'imagenet64',
        nn_type = 'densenet',
        quantize = True,
        pruning = False,
        pruned = False,
        batchsize = 1000,
        resume = '8bit',
    )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = get_configs(
        dataset=args.dataset,
        nn_type=args.nn_type,
        batch_size=args.batchsize,
        resume=args.resume,
        quantize=args.quantize, 
        pruning=args.pruning,
        pruned=args.pruned,
        out_dir = 'assets'
    )
    
    model = load_model(cfg)

    input_size = (3,32,32) if args.dataset == 'imagenet32' else (3, 64, 64)

    dataset = 'clic'

    abpds, cbpds, bandwidths = encode_high_resolution(model, input_size, dataset, True)

    import os
    with open(os.path.join(cfg.resume, '..', 'encode_high_resolution.txt'), 'a+') as f:
        msg = '=' * 20 + '\n' + \
            f'Encode {dataset}, {len(abpds)} images\n' + \
            f'analytic bpd: {np.mean(abpds):.3f}\n' + \
            f'coding bpd: {np.mean(cbpds):.3f}, compression ratio: {8 / np.mean(cbpds):.3f}\n' + \
            f'bandwidth: {np.mean(bandwidths):.3f} MB/s\n'
        
        print(msg, file=f)
        print(msg)