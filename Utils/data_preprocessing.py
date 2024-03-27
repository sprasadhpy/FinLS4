import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm

def findETF(data,stock):
    """
    reading a file containing information on stock memberships
    input: stock ticker
    output: corresponding ETF ticker
    """
    try:
        out = np.array(data['ticker_y'][data['ticker_x'] == stock])[0]
    except:
        out = None
    return out

def data_preprocessing(cfg):

    print("\nInitializing data preprocessing...\n")
    etfinfo = pd.read_csv(cfg.etflistloc)

    tickers = cfg.tickers
    stocks_list = [ticker for ticker in tickers if ticker[0] != 'X']
    etfs_list = [ticker for ticker in tickers if ticker[0] == 'X']

    print("Stocks list: ", stocks_list)
    print("ETFs list: ", etfs_list)

    files = os.listdir(cfg.dataloc)
    csv_files = [os.path.splitext(file)[0] for file in files if file.endswith('.csv')]

    # Convert the lists to sets
    stocks_required_set = set(stocks_list)
    csv_files_set = set(csv_files)

    # Find the stocks that are in stocks_required but not in csv_files
    missing_stocks = stocks_required_set - csv_files_set

    # Convert the set back to a list
    missing_stocks_list = list(missing_stocks)

    print("Missing stocks: ", missing_stocks_list)

    # Similarly find the missing ETFs

    for stock in stocks_list:
        etf= findETF(etfinfo,stock)
        etfs_list.append(etf)

    etf_set = set(etfs_list)
    missing_etfs = etf_set - csv_files_set

    missing_etfs_list = list(missing_etfs)
    print("Missing ETFs: ", missing_etfs_list)

    if len(missing_stocks_list) > 0:
        print("\nThe generating data for the missing stocks...\n")

        stocks_data = pd.read_csv(cfg.rawstocksloc)
        stocks_data['date_dt'] = pd.to_datetime(stocks_data['date'])
        stocks_data['AdjClose'] = stocks_data['PRC'] / stocks_data['CFACPR']
        stocks_data['AdjOpen'] = stocks_data['OPENPRC'] / stocks_data['CFACPR']

        # filter date after 1999
        stocks_data = stocks_data[stocks_data['date_dt'] > cfg.start_date]

        print("\nData Loading Completed...\n")

        for stock in missing_stocks_list:
            stock_df = stocks_data[stocks_data.TICKER == stock].copy().reset_index(drop=True)
            # format the date column to datetime
            stock_df['date'] = pd.to_datetime(stock_df['date'])
            # Select only the required columns
            stock_df = stock_df[['date', 'AdjClose', 'AdjOpen']]

            stock_df.to_csv(os.path.join(cfg.dataloc, stock + ".csv"))
            print("Data for ", stock, " saved successfully")

    else:
        print("Data for all stocks already exists")

    if len(missing_etfs_list) > 0:
        print("\nThe generating data for the missing ETFs...\n")

        etf_data = pd.read_csv(cfg.rawetfsloc)
        etf_data['date_dt'] = pd.to_datetime(etf_data['date'])
        etf_data['AdjClose'] = etf_data['PRC'] / etf_data['CFACPR']
        etf_data['AdjOpen'] = etf_data['OPENPRC'] / etf_data['CFACPR']

        # filter date after 1999
        etf_data = etf_data[etf_data['date_dt'] > cfg.start_date]

        print("\nData Loading Completed...\n")

        for etf in missing_etfs_list:
            etf_df = etf_data[etf_data.TICKER == etf].copy().reset_index(drop=True)
            # format the date column to datetime
            etf_data['date_dt'] = pd.to_datetime(etf_data['date'], dayfirst=True)
            # Select only the required columns
            etf_df = etf_df[['date', 'AdjClose', 'AdjOpen']]

            etf_df.to_csv(os.path.join(cfg.dataloc, etf + ".csv"))
            print("Data for ", etf, " saved successfully")
    else:
        print("Data for all ETFs already exists")

    print("Data for all tickers saved successfully")

    print("\nData preprocessing completed successfully.....\n")


