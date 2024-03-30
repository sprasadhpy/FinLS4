import torch
from tqdm import tqdm
from Utils.combine_vectors import combine_vectors


def train_loop(gen, disc, gen_opt, disc_opt, criterion, train_loader, alpha, beta, gamma, delta, cfg=None):
    """
    Training loop for the BCE GAN (ForGAN)
    """

    batch_size = cfg.batch_size
    n_epochs = cfg.n_epochs
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
    discloss = [False] * (nbatches * n_epochs)
    genloss = [False] * (nbatches * n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False


    # currstep = 0

    # train the discriminator more

    dscpred_real = [False] * (nbatches * n_epochs)
    dscpred_fake = [False] * (nbatches * n_epochs)
    PnL_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        for i, train_data in enumerate(train_loader):
            train_data = train_data.to(device)
            curr_batch_size = train_data.size(0)

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

                if cfg.model == "ForGAN-LSTM":
                    fake = gen(noise, condition, h_0g, c_0g)
                    fake_and_condition = combine_vectors(condition, fake, dim=-1)
                    fake_and_condition.to(torch.float)
                    real_and_condition = combine_vectors(condition, real, dim=-1)

                elif cfg.model == "ForGAN-SegRNN":
                    fake = gen(noise, condition, h_0g, c_0g)
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

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch * nbatches + i] = dscr
            dscpred_fake[epoch * nbatches + i] = dscfk

            # fksmpl.append(fake.detach())
            # rlsmpl.append(real.detach())

            # Get the predictions from the discriminator

            dloss = disc_loss.detach().item()
            discloss[epoch * nbatches + i] = dloss

            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1, curr_batch_size, z_dim, device=device, dtype=torch.float)

            fake = gen(noise, condition, h_0g, c_0g)

            # fake1 = fake1.unsqueeze(0).unsqueeze(2)

            if cfg.model == "ForGAN-LSTM":
                fake_and_condition = combine_vectors(condition, fake, dim=-1)
            elif cfg.model == "ForGAN-SegRNN":
                fake_and_condition = combine_vectors(condition[:, :, 1:], fake, dim=-1)

            disc_fake_pred = disc(fake_and_condition, h_0d, c_0d)

            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch * nbatches + i] = gloss

    return gen, disc, gen_opt, disc_opt