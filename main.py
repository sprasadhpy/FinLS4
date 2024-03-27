import torch
import pandas as pd
from Configuration import Config
from Utils.data_preprocessing import data_preprocessing, findETF
from ExecuteModels.execute_for_gan_lstm import execute_for_gan_lstm

def main():

    model = 'ForGAN LSTM'
    tickers = ['AMZN','AZO','GS','EL']
    # tickers = ['AMZN']
    cfg = Config(model=model, tickers=tickers)


    # Run data preprocessing
    data_preprocessing(cfg)

    for ticker in tickers:
        print("Processing data for ", ticker)
        # Read the data
        cfg.current_ticker = ticker
        cfg.current_etf = findETF(pd.read_csv(cfg.etflistloc), cfg.current_ticker)

        if cfg.model == 'ForGAN LSTM':

            execute_for_gan_lstm(cfg)


if __name__ == "__main__":
    main()