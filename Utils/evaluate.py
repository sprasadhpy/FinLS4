import torch
import numpy as np
import pandas as pd


def evaluate(gen, test_loader, val_loader, cfg=None):
    """
    Evaluation of a GAN model on a single stock
    """
    ticker = cfg.current_ticker
    h=cfg.h
    l=cfg.l
    pred=cfg.pred
    hid_d=cfg.hid_d
    hid_g=cfg.hid_g
    z_dim=cfg.z_dim
    lrg=cfg.lrg_s
    lrd=cfg.lrd_s
    n_epochs=cfg.n_epochs
    losstype=cfg.model
    device=cfg.device

    test_data = next(iter(test_loader))
    val_data = next(iter(val_loader))

    df_temp = False
    dt = {'lrd': lrd, 'lrg': lrg, 'type': losstype, 'epochs': n_epochs, 'ticker': ticker, 'hid_g': hid_g,
          'hid_d': hid_d}
    # print("Validation set best PnL (in bp): ",PnL_best)
    # print("Checkpoint epoch: ",checkpoint_last_epoch+1)
    ntest = len(test_data)
    gen.eval()
    with torch.no_grad():
        condition1 = test_data[:, 0:l]
        condition1 = condition1.unsqueeze(0)
        condition1 = condition1.to(device)
        condition1 = condition1.to(torch.float)

        ntest = test_data.shape[0]
        h0 = torch.zeros((1, ntest, hid_g), device=device, dtype=torch.float)
        c0 = torch.zeros((1, ntest, hid_g), device=device, dtype=torch.float)
        fake_noise = torch.randn(1, ntest, z_dim, device=device, dtype=torch.float)
        fake1 = gen(fake_noise, condition1, h0, c0)
        fake1 = fake1.unsqueeze(0).unsqueeze(2)
        generated1 = torch.empty([1, 1, 1, ntest, 1000])
        generated1[0, 0, 0, :, 0] = fake1[0, 0, 0, :, 0].detach()

        for i in range(999):
            fake_noise = torch.randn(1, ntest, z_dim, device=device, dtype=torch.float)
            fake1 = gen(fake_noise, condition1, h0, c0)
            fake1 = fake1.unsqueeze(0).unsqueeze(2)
            # print(fake.shape)
            generated1[0, 0, 0, :, i + 1] = fake1[0, 0, 0, :, 0].detach()

            del fake1
            del fake_noise
        # rmse = torch.sqrt(torch.mean((fake-real)**2))
        # mae = torch.mean(torch.abs(fake-real))
    # print("RMSE: ", rmse)
    # print("MAE: ",mae)
    b1 = generated1.squeeze()
    mn1 = torch.mean(b1, dim=1).to(device)
    real1 = test_data[:, -1]
    rl1 = real1.squeeze().to(device)
    rmse1 = torch.sqrt(torch.mean((mn1 - rl1) ** 2))
    mae1 = torch.mean(torch.abs(mn1 - rl1))
    # print("RMSE: ",rmse,"MAE: ",mae)
    dt['RMSE'] = rmse1.item()
    dt['MAE'] = mae1.item()
    ft1 = mn1.clone().detach().to(device)


    # look at the Sharpe Ratio
    n_b1 = b1.shape[1]
    PnL_ws1 = torch.empty(ntest)
    for i1 in range(ntest):
        fk1 = b1[i1, :]
        pu1 = (fk1 >= 0).sum()
        pu1 = pu1 / n_b1
        pd1 = 1 - pu1
        PnL_temp1 = 10000 * (pu1 * rl1[i1].item() - pd1 * rl1[i1].item())
        PnL_ws1[i1] = PnL_temp1.item()
    PnL_ws1 = np.array(PnL_ws1)
    PnL_wd1 = np.zeros(int(0.5 * len(PnL_ws1)))
    PnL_even = np.zeros(int(0.5 * len(PnL_ws1)))
    PnL_odd = np.zeros(int(0.5 * len(PnL_ws1)))
    for i1 in range(len(PnL_wd1)):
        PnL_wd1[i1] = PnL_ws1[2 * i1] + PnL_ws1[2 * i1 + 1]
        PnL_even[i1] = PnL_ws1[2 * i1]
        PnL_odd[i1] = PnL_ws1[2 * i1 + 1]
    PnL_test = PnL_wd1
    PnL_w_m1 = np.mean(PnL_wd1)
    PnL_w_std1 = np.std(PnL_wd1)
    SR1 = PnL_w_m1 / PnL_w_std1
    # print("Sharpe Ratio: ",SR)
    dt['SR_w scaled'] = SR1 * np.sqrt(252)
    dt['PnL_w'] = PnL_w_m1

    if (ntest % 2) == 0:
        dt['Close-to-Open SR_w'] = np.sqrt(252) * np.mean(PnL_even) / np.std(PnL_even)
        dt['Open-to-Close SR_w'] = np.sqrt(252) * np.mean(PnL_odd) / np.std(PnL_odd)
    else:
        dt['Open-to-Close SR_w'] = np.sqrt(252) * np.mean(PnL_even) / np.std(PnL_even)
        dt['Close-to-Open SR_w'] = np.sqrt(252) * np.mean(PnL_odd) / np.std(PnL_odd)
    print("Annualised (test) SR_w: ", SR1 * np.sqrt(252))

    distcheck = np.array(b1[1, :].cpu())
    means = np.array(mn1.detach().cpu())
    reals = np.array(rl1.detach().cpu())
    dt['Corr'] = np.corrcoef([means, reals])[0, 1]
    dt['Pos mn'] = np.sum(means > 0) / len(means)
    dt['Neg mn'] = np.sum(means < 0) / len(means)
    print('Correlation ', np.corrcoef([means, reals])[0, 1])

    dt['narrow dist'] = (np.std(distcheck) < 0.0002)

    means_gen = means
    reals_test = reals
    distcheck_test = distcheck
    rl_test = reals[1]

    mn = torch.mean(b1, dim=1)
    mn = np.array(mn.cpu())
    dt['narrow means dist'] = (np.std(mn) < 0.0002)

    ntest = val_data.shape[0]
    gen.eval()
    with torch.no_grad():
        condition1 = val_data[:, 0:l]
        condition1 = condition1.unsqueeze(0)
        condition1 = condition1.to(device)
        condition1 = condition1.to(torch.float)
        ntest = val_data.shape[0]
        h0 = torch.zeros((1, ntest, hid_g), device=device, dtype=torch.float)
        c0 = torch.zeros((1, ntest, hid_g), device=device, dtype=torch.float)
        fake_noise = torch.randn(1, ntest, z_dim, device=device, dtype=torch.float)
        fake1 = gen(fake_noise, condition1, h0, c0)
        fake1 = fake1.unsqueeze(0).unsqueeze(2)
        generated1 = torch.empty([1, 1, 1, ntest, 1000])
        generated1[0, 0, 0, :, 0] = fake1[0, 0, 0, :, 0].detach()
        # generated1 = fake1.detach()
        for i in range(999):
            fake_noise = torch.randn(1, ntest, z_dim, device=device, dtype=torch.float)
            fake1 = gen(fake_noise, condition1, h0, c0)
            fake1 = fake1.unsqueeze(0).unsqueeze(2)
            # print(fake.shape)
            generated1[0, 0, 0, :, i + 1] = fake1[0, 0, 0, :, 0].detach()
            # generated1 = combine_vectors(generated1, fake1.detach(), dim=-1)
            #             print(generated1.shape)
            del fake1
            del fake_noise

    b1 = generated1.squeeze()
    mn1 = torch.mean(b1, dim=1).to(device)
    real1 = val_data[:, -1]
    rl1 = real1.squeeze().to(device)
    rmse1 = torch.sqrt(torch.mean((mn1 - rl1) ** 2))
    mae1 = torch.mean(torch.abs(mn1 - rl1))
    # print("RMSE: ",rmse,"MAE: ",mae)
    dt['RMSE val'] = rmse1.item()
    dt['MAE val'] = mae1.item()
    ft1 = mn1.clone().detach().to(device)
    # print("PnL in bp", PnL)

    # look at the Sharpe Ratio
    n_b1 = b1.shape[1]
    PnL_ws1 = torch.empty(ntest)
    for i1 in range(ntest):
        fk1 = b1[i1, :]
        pu1 = (fk1 >= 0).sum()
        pu1 = pu1 / n_b1
        pd1 = 1 - pu1
        PnL_temp1 = 10000 * (pu1 * rl1[i1].item() - pd1 * rl1[i1].item())
        PnL_ws1[i1] = PnL_temp1.item()
    PnL_ws1 = np.array(PnL_ws1)
    PnL_wd1 = np.zeros(int(0.5 * len(PnL_ws1)))
    for i1 in range(len(PnL_wd1)):
        PnL_wd1[i1] = PnL_ws1[2 * i1] + PnL_ws1[2 * i1 + 1]
    PnL_w_m1 = np.mean(PnL_wd1)
    PnL_w_std1 = np.std(PnL_wd1)
    SR1 = PnL_w_m1 / PnL_w_std1
    # print("Sharpe Ratio: ",SR)
    dt['PnL_w val'] = PnL_w_m1
    dt['SR_w scaled val'] = SR1 * np.sqrt(252)

    print("Annualised (val) SR_w : ", SR1 * np.sqrt(252))

    means = np.array(mn1.detach().cpu())
    reals = np.array(rl1.detach().cpu())
    dt['Corr val'] = np.corrcoef([means, reals])[0, 1]
    dt['Pos mn val'] = np.sum(means > 0) / len(means)
    dt['Neg mn val'] = np.sum(means < 0) / len(means)

    df_temp = pd.DataFrame(data=dt, index=[0])

    cumPnL = np.cumsum(PnL_test)

    if (test_data.shape[0] % 2 == 0):
        intradayCumPnL = np.cumsum(PnL_odd)
        overnightCumPnL = np.cumsum(PnL_even)
    else:
        overnightCumPnL = np.cumsum(PnL_odd)
        intradayCumPnL = np.cumsum(PnL_even)

    sample_dist = distcheck_test
    means = means_gen

    PnLs = {
        "intradayCumPnL": [float(i) for i in intradayCumPnL],
        "overnightCumPnL": [float(i) for i in overnightCumPnL],
        "cumPnL": [float(i) for i in cumPnL],
        "sample_dist": [float(i) for i in sample_dist],
        "means": [float(i) for i in means]
    }


    return df_temp,  PnLs