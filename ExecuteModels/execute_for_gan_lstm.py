import torch
import pandas as pd
from models.Generator import Generator
from models.Discriminator import Discriminator
from Utils.data_factory import data_provider
from Utils.gradient_check import gradient_check
from Utils.train_loop import train_loop
from Utils.evaluate import evaluate
import os
import json

def execute_for_gan_lstm(cfg):

    trained_model_dir = os.path.join(cfg.trained_model_loc, cfg.model, cfg.current_ticker)
    if not os.path.exists(trained_model_dir):
        os.makedirs(trained_model_dir)

    results_dir = os.path.join(cfg.results_loc, cfg.model, cfg.current_ticker)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)




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



    gen = Generator(cfg,ref_mean, ref_std).to(device)
    disc = Discriminator(cfg, ref_mean, ref_std).to(device)

    print("\nGenerator and Discriminator Initialized...\n")

    gen_opt = torch.optim.RMSprop(gen.parameters(), lr=cfg.lrg_s)
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=cfg.lrd_s)
    criterion = torch.nn.BCELoss()

    gen, disc, gen_opt, disc_opt,alpha, beta, gamma, delta, gradients = gradient_check(gen, disc, gen_opt, disc_opt, criterion, train_loader, cfg=cfg)

    # convert the gradients dict to csv and save it
    gradients_df = pd.DataFrame(gradients)
    gradients_df.to_csv(os.path.join(results_dir, 'gradients.csv'))

    genfg, discfg, gen_optfg, disc_optfg = train_loop(gen, disc, gen_opt, disc_opt, criterion, train_loader, alpha, beta, gamma, delta, cfg=cfg)

    torch.save(genfg.state_dict(), os.path.join(trained_model_dir, 'generator_model.pth'))
    torch.save(discfg.state_dict(), os.path.join(trained_model_dir, 'discriminator_model.pth'))

    df_temp, PnLs = evaluate(genfg, test_loader, val_loader, cfg=cfg)

    df_temp.to_csv(os.path.join(results_dir, 'result.csv'))

    # Use json.dump to write the dictionary to a file
    with open(os.path.join(results_dir, 'PnLs.json'), 'w') as f:
        json.dump(PnLs, f)


