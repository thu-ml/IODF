import math
import numpy as np 
import time

from configs.configs_wrapper import IODFConfig, DPConfig
from core.interfaces.training_epoch import train_epoch, train_epoch_pruning
from core.interfaces.evaluation import evaluate
from core.utils.logger import save_states
from core.utils.dp_utils import * 
import logging

def train(
        model, optimizer, scheduler, writer, start_epoch,
        train_loader, val_loader, test_loader, 
        cfg
    ):

    best_val_bpd = np.inf

    train_times = []  

    model.eval()
    model.train()

    for epoch in range(start_epoch, cfg.n_epochs + 1):
        t_start = time.time()

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch}, Learning rate {lr}')

        tr_loss, tr_bpd = train_epoch(epoch, model, optimizer, train_loader, cfg)

        train_times.append(time.time() - t_start)
        
        logging.info(f'One training epoch took {time.time() - t_start:.2f} seconds')
        logging.info(f'====> Epoch: {epoch:3d} Average train loss: {tr_loss:.4f}, average bpd: {tr_bpd:.4f}')
        
        writer.add_scalars('bpd', {'train': tr_bpd})

        if epoch < 25 or epoch % cfg.evaluate_interval_epochs == 0:

            v_loss, v_bpd = evaluate(model, val_loader, cfg, epoch=epoch)
            
            logging.info(f"Epoch {epoch}, val bpd: {v_bpd:.3f}")
            
            writer.add_scalars('bpd', {'val': v_bpd}, epoch)

            if v_bpd < best_val_bpd:
                best_val_bpd = v_bpd
                save_states(model, optimizer, scheduler, cfg)
                logging.info('->model saved<-')

            logging.info(f'(BEST: validation bpd {best_val_bpd:.4f})\n')
                 
            if math.isnan(v_loss):
                raise ValueError('NaN encountered!')

    train_times = np.array(train_times)
    mean_train_time = np.mean(train_times)
    std_train_time = np.std(train_times, ddof=1)
    logging.info('Average train time per epoch: %.2f +/- %.2f' % (mean_train_time, std_train_time))

    test_loss, test_bpd = evaluate(model, test_loader, cfg, plot=False)
    logging.info('Test loss / bpd: %.2f / %.2f' % (test_loss, test_bpd))


def train_pruning(
        model, optimizer, scheduler, writer, start_epoch,
        train_loader, val_loader, test_loader, 
        cfg:IODFConfig, dp_config:DPConfig
    ):
    assert cfg.pruning is True

    best_val_bpd = np.inf
    train_times = []  
    iteration = 0

    compactor_params = get_compactor_params(model, cfg.prune_mode)
    
    model.train()

    for epoch in range(start_epoch, cfg.n_epochs + 1):
        t_start = time.time()

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch}, base learning rate {lr}')

        tr_loss, tr_bpd, iteration = train_epoch_pruning(epoch, iteration, model, optimizer, train_loader,\
                    cfg, dp_config, compactor_params=compactor_params)

        train_times.append(time.time()-t_start)
        
        logging.info(f'One training epoch took {time.time() - t_start:.2f} seconds')
        logging.info(f'====> Epoch: {epoch:3d} Average train loss: {tr_loss:.4f}, average bpd: {tr_bpd:.4f}')
        
        writer.add_scalars('bpd', {'train': tr_bpd})

        if epoch < 25 or epoch % cfg.evaluate_interval_epochs == 0:

            v_loss, v_bpd = evaluate(model, val_loader, cfg, epoch=epoch)
            
            logging.info(f"Epoch {epoch}, val bpd: {v_bpd:.3f}")
            
            writer.add_scalars('bpd', {'val': v_bpd}, epoch)
            
            import os
            directory = os.path.join(cfg.snap_dir, 'checkpoints', f'epoch{epoch}')
            os.makedirs(directory)
            save_states(model, optimizer, scheduler, cfg, directory)

            if v_bpd < best_val_bpd:
                best_val_bpd = v_bpd
                save_states(model, optimizer, scheduler, cfg, None)
                logging.info('->model saved<-')

            logging.info(f'(BEST: validation bpd {best_val_bpd:.4f})\n')
            if math.isnan(v_loss):
                raise ValueError('NaN encountered!')

    train_times = np.array(train_times)
    mean_train_time = np.mean(train_times)
    std_train_time = np.std(train_times, ddof=1)
    logging.info('Average train time per epoch: %.2f +/- %.2f' % (mean_train_time, std_train_time))

    test_loss, test_bpd = evaluate(model, test_loader, cfg, plot=False)
    logging.info('Test loss / bpd: %.2f / %.2f' % (test_loss, test_bpd))
