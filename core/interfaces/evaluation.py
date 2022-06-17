from tqdm import tqdm

import torch 
import numpy as np 
from core.utils.visualization import plot_reconstructions

def evaluate(model, data_loader, args, epoch=0, plot=True):
    model.eval()
    loss_type = 'bpd'

    def analyse(dl, plot=False):
        bpds = []
        batch_idx = 0
        with torch.no_grad():
            for data, _ in tqdm(dl, desc='Evaluating during training...'):
                batch_idx += 1

                data = data.view(-1, *args.input_size)

                loss, batch_bpd, _,_,_,_,_ = \
                    model(data.to(args.device))
                loss = torch.mean(loss).item()
                batch_bpd = torch.mean(batch_bpd).item()

                bpds.append(batch_bpd)

        bpd = np.mean(bpds)

        with torch.no_grad():
            if plot:
                from torch.nn.parallel.data_parallel import DataParallel
                if isinstance(model, DataParallel):
                    x_sample = model.module.sample(n=100)
                else:
                    x_sample = model.sample(n=100)

                try:
                    plot_reconstructions(
                        x_sample, bpd, loss_type, epoch, args)
                except:
                    print('Not plotting')

        return bpd

    bpd_val = analyse(data_loader, plot=plot)
    loss = bpd_val * np.prod(args.input_size) * np.log(2.)
    bpd = bpd_val
    return loss, bpd