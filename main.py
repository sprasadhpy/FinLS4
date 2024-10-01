import torch
import pandas as pd
from Configuration import Config
from Utils.data_preprocessing import data_preprocessing, findETF
from ExecuteModels.execute_for_gan_lstm import execute_for_gan_lstm
from ExecuteModels.execute_for_gan_segrnn import execute_for_gan_segrnn
from ExecuteModels.execute_for_gan_tln import execute_for_gan_tln
from ExecuteModels.execute_for_gan_ls4 import execute_for_gan_ls4

# from Utils.regime_analysis import analyse_regime

import os
os.environ['CXX'] = 'g++-8'


def main():

    models= ['ForGAN-LSTM', 'ForGAN-SegRNN', 'ForGAN-FT-Matrix', 'ForGAN-F-SVD', 'ForGAN-FFT-Conv', 'ForGAN-Conv-SVD', 'ForGAN-LS4']
    model = 'ForGAN-LS4'
    # model = 'ForGAN-SegRNN'
    # model = models[-1]
    regime_analysis = False

    tickers = ['AMZN','AZO','GS','EL']
    tickers = ['AMZN']
    cfg = Config(model=model, tickers=tickers)

    # Set the device and print the details of the s
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    cfg.device = device

    print("Device: ", device)


    # Run data preprocessing
    data_preprocessing(cfg)

    for ticker in tickers:
        print('\n' + '-'*50 + ticker + '-'*50 + '\n')

        print("Model: ", cfg.model)
        print("Processing data for ", ticker)
        # Read the data

        if ticker[0] != 'X':
            cfg.current_ticker = ticker
            cfg.current_etf = findETF(pd.read_csv(cfg.etflistloc), cfg.current_ticker)
        else:
            cfg.current_ticker = ticker
            cfg.current_etf = ticker


        if regime_analysis:
            analyse_regime(cfg)



        if cfg.model == 'ForGAN-LSTM':
            execute_for_gan_lstm(cfg)

        elif cfg.model == 'ForGAN-SegRNN':
            execute_for_gan_segrnn(cfg)

        elif cfg.model in ['ForGAN-FT-Matrix', 'ForGAN-F-SVD', 'ForGAN-FFT-Conv', 'ForGAN-Conv-SVD']:
            execute_for_gan_tln(cfg)

        elif cfg.model == 'ForGAN-LS4':
            cfg.hid_g = 16
            cfg.hid_d = 16
            execute_for_gan_ls4(cfg)


if __name__ == "__main__":
    main()