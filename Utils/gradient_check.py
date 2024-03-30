import torch
from tqdm import tqdm
from Utils.combine_vectors import combine_vectors

def gradient_check(gen, disc, gen_opt, disc_opt, criterion, train_loader, cfg=None):
    """
    Gradient norm check
    """

    batch_size = cfg.batch_size
    n_epochs = cfg.n_grad
    hid_d = cfg.hid_d
    hid_g = cfg.hid_g
    z_dim = cfg.z_dim
    l = cfg.l
    pred = cfg.pred
    diter = cfg.diter
    tanh_coeff = cfg.tanh_coeff
    device = cfg.device

    ntrain = len(train_loader.dataset)
    nbatches = ntrain // batch_size + 1
    BCE_norm = torch.empty(nbatches * n_epochs -1, device=device)
    PnL_norm = torch.empty(nbatches * n_epochs -1, device=device)
    MSE_norm = torch.empty(nbatches * n_epochs -1, device=device)
    SR_norm = torch.empty(nbatches * n_epochs -1, device=device)
    STD_norm = torch.empty(nbatches * n_epochs -1, device=device)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    # totlen = train_data.shape[0]

    # currstep = 0
    # train the discriminator more

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        # perm = torch.randperm(ntrain)
        # train_data = train_data[perm, :]
        # shuffle the dataset for the optimisation to work
        for i, train_data in enumerate(train_loader):
            train_data = train_data.to(device)
            curr_batch_size = train_data.size(0)
            # print(f"curr_batch_size: {curr_batch_size}")
            h_0d = torch.zeros((1, curr_batch_size, hid_d), device=device, dtype=torch.float)
            c_0d = torch.zeros((1, curr_batch_size, hid_d), device=device, dtype=torch.float)
            h_0g = torch.zeros((1, curr_batch_size, hid_g), device=device, dtype=torch.float)
            c_0g = torch.zeros((1, curr_batch_size, hid_g), device=device, dtype=torch.float)
            condition = train_data[:, 0:l]
            condition = condition.unsqueeze(0)
            real = train_data[:, l:(l + pred)]
            real = real.unsqueeze(0)

            ### Update discriminator ###
            # Zero out the discriminator gradients
            for j in range(diter):
                disc_opt.zero_grad()
                # Get noise corresponding to the current batch_size
                noise = torch.randn(1, curr_batch_size, z_dim, device=device, dtype=torch.float)
                # Get outputs from the generator
                # print('condition shape:', condition.shape)
                fake = gen(noise, condition, h_0g, c_0g)
                # fake = fake.unsqueeze(0)

                if cfg.model == "ForGAN-LSTM":
                    fake_and_condition = combine_vectors(condition, fake, dim=-1)
                    fake_and_condition.to(torch.float)
                    real_and_condition = combine_vectors(condition, real, dim=-1)
                    disc_fake_pred = disc(fake_and_condition.detach(), h_0d, c_0d)
                    disc_real_pred = disc(real_and_condition, h_0d, c_0d)
                elif cfg.model == "ForGAN-SegRNN":
                    fake_and_condition = combine_vectors(condition[:, :, 1:], fake, dim=-1)
                    fake_and_condition.to(torch.float)
                    real_and_condition = combine_vectors(condition[:, :, 1:], real, dim=-1)
                    disc_fake_pred = disc(fake_and_condition.detach(), h_0d, c_0d)
                    disc_real_pred = disc(real_and_condition, h_0d, c_0d)

                # Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                # disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            # Update generator
            # Zero out the generator gradients

            noise = torch.randn(1, curr_batch_size, z_dim, device=device, dtype=torch.float)
            fake = gen(noise, condition, h_0g, c_0g)

            # fake1 = fake1.unsqueeze(0).unsqueeze(2)

            if cfg.model == "ForGAN-LSTM":
                fake_and_condition = combine_vectors(condition, fake, dim=-1)
            elif cfg.model == "ForGAN-SegRNN":
                fake_and_condition = combine_vectors(condition[:, :, 1:], fake, dim=-1)

            disc_fake_pred = disc(fake_and_condition, h_0d, c_0d)

            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)

            # print('ft',ft)

            sign_approx = torch.tanh(tanh_coeff * ft)
            # print('sign_approx',sign_approx)

            PnL_s = sign_approx * rl
            PnL = torch.mean(PnL_s)
            MSE = (torch.norm(ft - rl) ** 2) / curr_batch_size
            SR = (torch.mean(PnL_s)) / (torch.std(PnL_s))
            STD = torch.std(PnL_s)
            gen_opt.zero_grad()
            # print('sr before',SR)
            SR.backward(retain_graph=True)
            # print('sr after',SR)
            total_norm = 0
            for p in gen.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            # list of gradient norms
            SR_norm[epoch * nbatches + i] = total_norm

            gen_opt.zero_grad()
            PnL.backward(retain_graph=True)
            total_norm = 0
            for p in gen.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
            PnL_norm[epoch * nbatches + i] = total_norm

            gen_opt.zero_grad()
            MSE.backward(retain_graph=True)
            total_norm = 0
            for p in gen.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
            MSE_norm[epoch * nbatches + i] = total_norm

            gen_opt.zero_grad()
            STD.backward(retain_graph=True)
            total_norm = 0
            for p in gen.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
            STD_norm[epoch * nbatches + i] = total_norm

            gen_opt.zero_grad()
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            total_norm = 0
            for p in gen.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
            BCE_norm[epoch * nbatches + i] = total_norm
            gen_opt.step()

    alpha = torch.mean(torch.where(PnL_norm != 0, BCE_norm / PnL_norm, torch.tensor(0.0)))
    beta = torch.mean(torch.where(MSE_norm != 0, BCE_norm / MSE_norm, torch.tensor(0.0)))
    gamma = torch.mean(torch.where(SR_norm != 0, BCE_norm / SR_norm, torch.tensor(0.0)))
    delta = torch.mean(torch.where(STD_norm != 0, BCE_norm / STD_norm, torch.tensor(0.0)))
    print("Completed. ")
    print(r"$\alpha$:", alpha)
    print(r"$\beta$:", beta)
    print(r"$\gamma$:", gamma)
    print(r"$\delta$:", delta)

    gradients = {"PnL_norm": PnL_norm.cpu(),
                 "MSE_norm": MSE_norm.cpu(),
                 "SR_norm": SR_norm.cpu(),
                 "STD_norm": STD_norm.cpu(),
                 "BCE_norm": BCE_norm.cpu()}



    return gen, disc, gen_opt, disc_opt, alpha, beta, gamma, delta, gradients