import tqdm
import torch
import numpy as np 

from configs.configs_wrapper import IODFConfig

def evaluate_analytic_bpd(model, data_loader, cfg:IODFConfig):
    model.eval()
    bpds = []
    with torch.no_grad():
        for idx, (data, _) in enumerate(tqdm.tqdm(data_loader, desc='evaluating bpd')):
            
            data = data.to(cfg.device)
            data = data.view(-1, *cfg.input_size)    

            _, batch_bpd, _,_,_,_,_, = model(data)

            batch_bpd = batch_bpd.cpu().numpy()
            bpds.append(batch_bpd)
            
    bpds = np.concatenate(bpds, axis=0)

    
    return bpds
