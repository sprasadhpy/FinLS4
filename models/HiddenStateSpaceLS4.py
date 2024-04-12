import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import random

from torch.utils.data import Dataset, DataLoader

import os
import argparse
import yaml

from itertools import product


import sys
sys.path.append('/Users/saurabh/Downloads/ucl_courses/multi_agent_ai/FinLS4/models/ls4_base/')

from models.ls4 import VAE
from tqdm.auto import tqdm

from pathlib import Path
from omegaconf import OmegaConf
# from datasets import parse_datasets
import wandb
import sklearn

import matplotlib.pyplot as plt
# from metrics import compute_all_metrics

class HiddenStateSpaceLS4:
    def __init__(self, cfg):
        super(HiddenStateSpaceLS4, self).__init__()
        print('ls4 hidden state space ')
        yaml_file_path = './models/ls4_base/configs/monash/vae_fred_md.yaml'
        # Load the YAML file
        with open(yaml_file_path, 'r') as file:
            vae_config = yaml.safe_load(file)
        
        config = OmegaConf.create(vae_config)
        #n_labels=1 is it for regression?
        config.model.n_labels = 1

        config.optim.epochs = cfg.n_epochs

        config.model.z_dim = cfg.hid_g
        config.model.decoder.prior.d_input = cfg.hid_g
        config.model.decoder.prior.d_output = cfg.hid_g
        config.model.decoder.decoder.d_input = cfg.hid_g
        config.model.encoder.posterior.d_output = cfg.hid_g

        #print('Encoder dimns :: ')
        #print(type(config))
        #print(config)
        # Model
        print('==> Building model..')
        model = VAE(config.model)
        model = model.to(cfg.device)

        self.model = model
        self.config = config

        #cfg.forgan_part_type

    def parameters(self):
        return self.model.parameters()

    def buffers(self):
        return self.model.buffers()

###############################

def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.
    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.
    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

###############################


# Training
def train(model, ema_model, ema_start_step, device, 
          trainloader, optimizer, scheduler, step, ticker='', wandb_info = False, forgan_part_type = 'generator'):
    #discriminator or generator
    if wandb_info:
        wandb.init(    
            project="for_ls4",
            train_config={
            "test" : f'ls4_{ticker}',
            "learning_rate": 0.02,
            }
        )
    model.train()

    train_loss = 0
    mse = 0
    kld = 0
    nll = 0
    ce = 0
    print()
    pbar = tqdm(enumerate(trainloader))
    #for batch_idx, (data, masks) in pbar:
    for batch_idx, data in pbar:
        step += 1
        masks = data[:,-1].unsqueeze(1).unsqueeze(2)
        if forgan_part_type == 'generator':
            data = data[:, :-1]
        data = data.unsqueeze(2)

        #print(masks.shape, data.shape)
        data = data.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        loss , log_info = model(data, None, masks,
                                plot=batch_idx==0, sum=False)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        mse += log_info['mse_loss']
        kld += log_info['kld_loss']
        nll += log_info['nll_loss']
        ce += log_info.get('ce_loss', 0)

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f | KLD: %.3f | NLL: %.3f | MSE: %.7f | CE: %.3f' %
            (batch_idx, len(trainloader), train_loss / (batch_idx + 1), kld / (batch_idx + 1),
             nll / (batch_idx + 1), mse / (batch_idx + 1), ce / (batch_idx + 1))
        )
        if wandb_info:
            wandb.log({f'train_{k}':v / (batch_idx + 1) for k, v in dict(log_info).items()})
        
        if ema_model and step > ema_start_step:
            ema_model.update_parameters(model)
            

    scheduler.step(train_loss / (batch_idx + 1))
    return step


###############################


@torch.no_grad()
def eval(model, device, epoch, dataloader, decode_func, log_dir, savenpy=False, return_preds=False, wandb_info = False, forgan_part_type = 'generator'):

    img = os.path.join(log_dir, 'images')
    epoch_dir = os.path.join(img, f'{epoch:03d}')

    if not os.path.isdir(img):
        os.mkdir(img)
        

    if not os.path.isdir(epoch_dir):
        os.mkdir(epoch_dir)

    model.eval()
    mse_loss = 0
    mse_noscale_loss = 0
    nll_loss = 0
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.optim.swa_utils.AveragedModel):
        model.module.setup_rnn()
    else:
        model.setup_rnn()

    all_data = []
    all_recon_data = []
    hidden_state_space_info = []
    cnt = 0
    pbar = tqdm(enumerate(dataloader))
    #for batch_idx, (data, masks)  in pbar:
    for batch_idx, data  in pbar:
        masks = data[:,-1].unsqueeze(1).unsqueeze(2)
        if forgan_part_type == 'generator':
            data = data[:, :-1]
        data = data.unsqueeze(2)
        data = data.to(device)
        masks = masks.to(device)
        #print(data[0,:,0])
        #print(masks[0,:,0])

        if isinstance(model, torch.nn.DataParallel)  or isinstance(model, torch.optim.swa_utils.AveragedModel):
            recon  = model.module.reconstruct(data, None, masks=masks,
                                                       get_full_nll=False)
        else:
            recon = model.reconstruct(data, None,  masks=masks,
                                                get_full_nll=False)
            
            z, z_mean, z_std = model.encode(data, None)
        hidden_state_space_info.append([z, z_mean, z_std])
        '''
        as written in  ls4 file
        
        def reconstruct(self, x, t_vec, t_vec_pred=None, x_full=None, masks=None, get_full_nll=False):
            if t_vec_pred is None: recon  x, if not, predict t_vec_pred's x
    
            z, z_mean, z_std = self.encoder.encode(x, t_vec, use_forward=True)
        '''

        

        print(recon.shape)
        
        recon_scaled = decode_func(recon)
        data_scaled = decode_func(data)
        all_data.append(data_scaled)
        all_recon_data.append(recon_scaled)
        cnt+=(recon_scaled.shape[0])
        
        mask_sum = masks.sum(1)
        mask_sum[mask_sum==0] = 1
        mse = (
            ((recon_scaled - data_scaled).pow(2) * masks).sum(1) / mask_sum
        ).mean()
        
        mse_noscale = (
            ((recon - data).pow(2) * masks).sum(1) / mask_sum
        ).mean()

        ## likelihood
        
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.optim.swa_utils.AveragedModel):
            full_nll = model.module.full_nll(data, None, torch.ones_like(data),)
        else:
            full_nll = model.full_nll(data, None, torch.ones_like(data),)

        mse_loss += mse.detach().cpu().item()
        nll_loss += full_nll.detach().cpu().item()
        mse_noscale_loss+= mse_noscale.detach().cpu().item()

    print(f'Total datapoints on eval :: {cnt}')

    #gen
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.optim.swa_utils.AveragedModel):
        gen = model.module.generate(16, data.shape[1], device=data.device)
    else:
        gen =  model.generate(16, data.shape[1], device=data.device)

    gen = (gen).cpu().numpy()
    for k in range(min(gen.shape[0], 8)): 
        filename = f'{epoch_dir}/{k:02d}_gen.png'
        plt.plot(gen[k], c='black', alpha=0.6)
        plt.savefig(filename)
        plt.close()
        
    if savenpy:
        np.save(f'{epoch_dir}/gen.npy', gen)
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.optim.swa_utils.AveragedModel):
        recon  = model.module.reconstruct(data, None, masks=masks,
                                                    get_full_nll=False)
    else:
        recon = model.reconstruct(data, None,  masks=masks,
                                            get_full_nll=False)
    
    recon = (recon).cpu().numpy()
    data = (data).cpu().numpy()
    for k in range(min(recon.shape[0], 8)): 
        plt.plot(recon[k], c='black', alpha=0.6)
        
        plt.savefig(f'{epoch_dir}/{k:02d}_recon.png')
        plt.close()
        plt.plot(data[k], c='black', alpha=0.6)
        plt.savefig(f'{epoch_dir}/{k:02d}_x.png')
        plt.close()
    
    if savenpy:
        np.save(f'{epoch_dir}/data.npy', data)
    p = 'TEST MSE: %.8f TEST MSE No Scale: %.8f  FULL NLL: %.8f' % \
        (mse_loss / (batch_idx + 1), mse_noscale_loss/(batch_idx + 1), nll_loss / (batch_idx + 1))
    print(p)
    if wandb_info:
        wandb.log({'eval_mse': mse_loss / (batch_idx + 1),
                   'eval_mse_noscale':  mse_noscale_loss/(batch_idx + 1),
                   'eval_full_nll': nll_loss / (batch_idx + 1)})
    if return_preds:
        return all_data, all_recon_data, hidden_state_space_info, mse_loss / (batch_idx + 1)

    return mse_loss / (batch_idx + 1)


###############################


@torch.no_grad()
def get_hidden_states(model, device, epoch, dataloader, decode_func, log_dir, savenpy=False, return_preds=False):
    
    model.eval()

    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.optim.swa_utils.AveragedModel):
        model.module.setup_rnn()
    else:
        model.setup_rnn()

    all_data = []
    all_recon_data = []
    hidden_state_space_info = []
    hidden_state_space = []
    cnt = 0

    pbar = tqdm(enumerate(dataloader))
    for batch_idx, (data, masks)  in pbar:
        data = data.to(device)
        masks = masks.to(device)
        #print(data[0,:,0])
        #print(masks[0,:,0])

        if isinstance(model, torch.nn.DataParallel)  or isinstance(model, torch.optim.swa_utils.AveragedModel):
            recon  = model.module.reconstruct(data, None, masks=masks,
                                                       get_full_nll=False)
        else:
            recon = model.reconstruct(data, None,  masks=masks,
                                                get_full_nll=False)
            
            z, z_mean, z_std = model.encode(data, None)
        hidden_state_space_info.append([z, z_mean, z_std])
        hidden_state_space.append(z[:,-1,:])  #dimns (batch_size,  seq_len, hid_state)
        #print('Single dat point :: ',z[0].shape, type(z))
        '''
        as written in  ls4 file
        
        def reconstruct(self, x, t_vec, t_vec_pred=None, x_full=None, masks=None, get_full_nll=False):
            if t_vec_pred is None: recon  x, if not, predict t_vec_pred's x
    
            z, z_mean, z_std = self.encoder.encode(x, t_vec, use_forward=True)
        '''
        recon_scaled = decode_func(recon)
        data_scaled = decode_func(data)
        all_data.append(data_scaled)
        all_recon_data.append(recon_scaled)
        cnt+=(recon_scaled.shape[0])
        
        mask_sum = masks.sum(1)
        mask_sum[mask_sum==0] = 1
        mse = (
            ((recon_scaled - data_scaled).pow(2) * masks).sum(1) / mask_sum
        ).mean()
        
        mse_noscale = (
            ((recon - data).pow(2) * masks).sum(1) / mask_sum
        ).mean()

    hidden_state_space = torch.vstack(hidden_state_space)
    return hidden_state_space#hidden_state_space_info #, all_data, all_recon_data, 


