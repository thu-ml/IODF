import time
import tqdm
import torch
import numpy as np 

from core.coder.apis import encode, decode

def evaluate_coding(model, data_loader, cfg):
    state_sizes = []
    bpds = []

    N, t_encode, t_inference, t_rans, t_decode, error = [0. for _ in range(6)]

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for idx, (data, _) in enumerate(tqdm.tqdm(data_loader, desc='encoding')):
            batchsize = data.size(0)
            data = data.to(cfg.device)
            tic_encode = time.time()

            starter.record()
            _, bpd, _, pz, z, pys, ys= model(data)
            ender.record()
            torch.cuda.synchronize()
            t_inference += starter.elapsed_time(ender) # milliseconds
            
            tic = time.time()
            ans_coder = encode(pz, z, pys, ys)
            
            t_rans += (time.time() - tic)

            t_encode += (time.time() - tic_encode)

            state_sizes += [ans_coder.stream_length()]

            if not cfg.no_decode:
                x_recon = []

                tic = time.time()
                
                img = decode(model, ans_coder)
                x_recon.append(img)
                
                t_decode += (time.time() - tic)
                x_recon = torch.stack(x_recon, dim=0).cpu()
                error += torch.sum(torch.abs(x_recon.int() - data.cpu().int())) .item()
                
            N += len(data)
            bpds.append(bpd.cpu().numpy())
                
    analytic_bpd = np.mean(np.concatenate(bpds, axis=0))
    code_bpd = np.sum(state_sizes) / (np.prod(cfg.input_size) * N)
    ts = [t_encode/N, t_inference/N/1e3, t_rans/N, t_decode/N]

    return analytic_bpd, code_bpd, error, N, ts