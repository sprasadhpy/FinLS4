import torch
import pandas as pd
from models.Generator import Generator
from models.Discriminator import Discriminator
from Utils.data_factory import data_provider
from Utils.gradient_check import gradient_check

def execute_for_gan_lstm(cfg):


    device = cfg.device

    cfg.data = "custom"

    print("\nCalculating excess returns and splitting data into train, validation and test sets...\n")

    train_data, train_loader = data_provider(cfg, 'train')
    val_data, val_loader = data_provider(cfg, 'val')
    test_data, test_loader = data_provider(cfg, 'test')

    print("\nData Splitting Completed...\n")

    first_batch_inputs = next(iter(train_loader))
    print(f"first_batch_inputs shape: {first_batch_inputs.shape}")
    ref_mean = torch.mean(first_batch_inputs)
    ref_std = torch.std(first_batch_inputs)



    gen = Generator(cfg.z_dim, cfg.l, cfg.hid_g, cfg.pred, ref_mean, ref_std).to(device)
    disc = Discriminator(cfg.l+cfg.pred, cfg.hid_d, ref_mean, ref_std).to(device)

    print("\nGenerator and Discriminator Initialized...\n")

    gen_opt = torch.optim.RMSprop(gen.parameters(), lr=cfg.lrg_s)
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=cfg.lrd_s)
    criterion = torch.nn.BCELoss()

    gen, disc, gen_opt, disc_opt = gradient_check(gen, disc, gen_opt, disc_opt, criterion, cfg.n_epochs, train_loader, cfg.batch_size, cfg.hid_d, cfg.hid_g,
                   cfg.z_dim, l=cfg.l, pred=1, diter=1, tanh_coeff=100, device=device, cfg=None)









