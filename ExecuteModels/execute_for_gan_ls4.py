import torch
import pandas as pd
from models.GeneratorWithLS4 import GeneratorWithLS4 as Generator
from models.DiscriminatorWithLS4 import DiscriminatorWithLS4 as Discriminator
from models.HiddenStateSpaceLS4 import HiddenStateSpaceLS4
from models.HiddenStateSpaceLS4 import get_hidden_states, eval, train, setup_optimizer
from Utils.data_factory import data_provider
from Utils.gradient_check import gradient_check
from Utils.train_loop import train_loop
from Utils.evaluate import evaluate
import os
import json
from tqdm.auto import tqdm


def execute_for_gan_ls4(cfg):

    trained_model_dir = os.path.join(cfg.trained_model_loc, cfg.model, cfg.current_ticker)
    if not os.path.exists(trained_model_dir):
        os.makedirs(trained_model_dir)

    results_dir = os.path.join(cfg.results_loc, cfg.model, cfg.current_ticker)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


    device = cfg.device

    cfg.data = "custom"
    #TODO : fails on saurabh's mac if cfg.num_workers > 1: try to find out later(0 works)
    if device == 'cpu':
        cfg.num_workers = 0

    print("\nCalculating excess returns and splitting data into train, validation and test sets...\n")

    cfg.batch_size = 1024
    train_data, train_loader = data_provider(cfg, 'train')
    val_data, val_loader = data_provider(cfg, 'val')
    test_data, test_loader = data_provider(cfg, 'test')

    print("\nData Splitting Completed...\n")

    print(device)
    first_batch_inputs = next(iter(train_loader)).to(device)
    ref_mean = torch.mean(first_batch_inputs)
    ref_std = torch.std(first_batch_inputs)
    print(ref_mean, ref_std)
    #print(first_batch_inputs.shape)
    #print(type(train_data))
    #print(train_data.shape, val_data.shape)
    for data in next(iter(train_loader)):
        print('with batch')
        print(data.shape)
        break
    
    model_ls4_gen_name = 'hidden_state_space_ls4_generator.pth'
    model_ls4_dis_name = 'hidden_state_space_ls4_discriminator.pth'
    cfg.hidden_state_space_ls4_generator_path = os.path.join(trained_model_dir, model_ls4_gen_name)
    cfg.hidden_state_space_ls4_discriminator_path = os.path.join(trained_model_dir, model_ls4_dis_name)
    
    #create hidden state space model for Generator
    print('Training LS4 Generator hidden state space')
    cfg.forgan_part_type = 'generator'
    
    model_ls4_gen = HiddenStateSpaceLS4(cfg)
    print(model_ls4_gen.config)

    if model_ls4_gen_name not in os.listdir(trained_model_dir):

        use_ema = model_ls4_gen.config.optim.get('use_ema', False)
        model_ls4_gen.config.optim.batch_size = cfg.batch_size
        
        if use_ema:
            ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
                            model_ls4_gen.config.optim.lamb * averaged_model_parameter + (1 - model_ls4_gen.config.optim.lamb) * model_parameter
            ema_model = torch.optim.swa_utils.AveragedModel(model_ls4_gen, avg_fn=ema_avg)
            ema_start_step =  model_ls4_gen.config.optim.start_step
        else:
            ema_model = None
            ema_start_step = 0
        
        optimizer, scheduler = setup_optimizer(
            model_ls4_gen, \
            lr=model_ls4_gen.config.optim.lr, \
            weight_decay=model_ls4_gen.config.optim.weight_decay, \
            epochs=model_ls4_gen.config.optim.epochs)
    
        best_marginal = float('inf')  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        step = 0
        metrics_log = {
                'mse': [],
                'clf_score': [],
                'marginal_score': [],
                'predictive_score': []
        }
        
        if device == 'cuda':
            model = torch.nn.DataParallel(model_ls4_gen)
            cudnn.benchmark = True

        debug = False
        resume = False
        model_ls4_gen.config.optim.epochs = 3 #TODO
        pbar = tqdm(range(start_epoch, start_epoch + model_ls4_gen.config.optim.epochs), disable=not debug)
        print(model_ls4_gen.config)
        for epoch in pbar:
            if epoch == 0 or resume:
                p = 'Epoch: %d' % (epoch)
                if not debug:
                    print(p)
                else:
                    pbar.set_description(p)

            step = train(model_ls4_gen.model, ema_model, ema_start_step, device, train_loader, optimizer, scheduler, step)
            # [optional] finish the wandb run, necessary in notebooks
            #wandb.finish()

        print(f'Gen ls4 saved at {cfg.hidden_state_space_ls4_generator_path}')
        torch.save(model_ls4_gen.model.state_dict(), cfg.hidden_state_space_ls4_generator_path)

    else:
        print('Loading saved model')
        #loaded_model = VAE(config.model)
        #loaded_model = loaded_model.to(device)
        model_ls4_gen.model.load_state_dict(torch.load(cfg.hidden_state_space_ls4_generator_path))
    #training

    
    #create hidden state space model for Discriminator
    print('Training LS4 Discriminator hidden state space')
    cfg.forgan_part_type = 'discriminator'
    model_ls4_disc = HiddenStateSpaceLS4(cfg)

    if model_ls4_dis_name not in os.listdir(trained_model_dir):

        use_ema = model_ls4_disc.config.optim.get('use_ema', False)
        model_ls4_disc.config.optim.batch_size = cfg.batch_size
        
        if use_ema:
            ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
                            model_ls4_disc.config.optim.lamb * averaged_model_parameter + (1 - model_ls4_disc.config.optim.lamb) * model_parameter
            ema_model = torch.optim.swa_utils.AveragedModel(model_ls4_disc, avg_fn=ema_avg)
            ema_start_step =  model_ls4_disc.config.optim.start_step
        else:
            ema_model = None
            ema_start_step = 0
        
        optimizer, scheduler = setup_optimizer(
            model_ls4_disc, \
            lr=model_ls4_disc.config.optim.lr, \
            weight_decay=model_ls4_disc.config.optim.weight_decay, \
            epochs=model_ls4_disc.config.optim.epochs)
    
        best_marginal = float('inf')  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        step = 0
        metrics_log = {
                'mse': [],
                'clf_score': [],
                'marginal_score': [],
                'predictive_score': []
        }
        
        if device == 'cuda':
            model = torch.nn.DataParallel(model_ls4_disc)
            cudnn.benchmark = True

        debug = False
        resume = False
        model_ls4_disc.config.optim.epochs = 3 #TODO
        pbar = tqdm(range(start_epoch, start_epoch + model_ls4_disc.config.optim.epochs), disable=not debug)
        print(model_ls4_disc.config)
        for epoch in pbar:
            if epoch == 0 or resume:
                p = 'Epoch: %d' % (epoch)
                if not debug:
                    print(p)
                else:
                    pbar.set_description(p)

            step = train(model_ls4_disc.model, ema_model, ema_start_step, device, train_loader, optimizer, scheduler, step, forgan_part_type = cfg.forgan_part_type)
            # [optional] finish the wandb run, necessary in notebooks
            #wandb.finish()

        print(f'Dis ls4 saved at {cfg.hidden_state_space_ls4_discriminator_path}')
        torch.save(model_ls4_disc.model.state_dict(), cfg.hidden_state_space_ls4_discriminator_path)

    else:
        print('Loading saved model')
        #loaded_model = VAE(config.model)
        #loaded_model = loaded_model.to(device)
        model_ls4_disc.model.load_state_dict(torch.load(cfg.hidden_state_space_ls4_discriminator_path))



    

    gen = Generator(cfg, ref_mean, ref_std, model_ls4_gen).to(device)
    #print('Gen model loaded.')
    #from torch.distributions import Normal
    #normal_dist = Normal(0, 1)
    # Sample from the distribution
    #print(first_batch_inputs.shape)
    #noise_tmp = normal_dist.sample((first_batch_inputs.shape[0], cfg.z_dim))
    #xx = gen(noise_tmp, first_batch_inputs, 0, 0)
    #print(xx.shape)
    #return
    
    disc = Discriminator(cfg, ref_mean, ref_std, model_ls4_disc).to(device)

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


