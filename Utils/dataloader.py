import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from data_provider.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', stock_path='', etf_path='',
                 cfg=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 20
            self.label_len = 1
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag


        self.root_path = root_path
        self.stock_path = stock_path
        self.etf_path = etf_path
        self.cfg = cfg
        self.__read_data__()
        # self.len = self.__len__()



    def __read_data__(self):

        stock = self.stock_path
        etf = self.etf_path

        stock_df = pd.read_csv(stock)
        etf_df = pd.read_csv(etf)

        excess_returns, dates_dt = self.excessreturns(stock_df, etf_df)

        N = len(excess_returns)
        N_tr = int(self.cfg.tr * N)
        N_vl = int(self.cfg.vl * N)
        N_tst = N - N_tr - N_vl
        train_sr = excess_returns[0:N_tr]
        val_sr = excess_returns[N_tr:N_tr + N_vl]
        train_sr = excess_returns[0:N_tr]
        val_sr = excess_returns[N_tr:N_tr + N_vl]
        test_sr = excess_returns[N_tr + N_vl:]
        n = int((N_tr - self.cfg.l - self.cfg.pred) / self.cfg.h) + 1
        train_data = np.zeros(shape=(n, self.cfg.l + self.cfg.pred))
        l_tot = 0
        for i in range(n):
            train_data[i, :] = train_sr[l_tot:l_tot + self.cfg.l + self.cfg.pred]
            l_tot = l_tot + self.cfg.h
        n = int((N_vl - self.cfg.l - self.cfg.pred) / self.cfg.h) + 1
        val_data = np.zeros(shape=(n, self.cfg.l + self.cfg.pred))
        l_tot = 0
        for i in range(n):
            val_data[i, :] = val_sr[l_tot:l_tot + self.cfg.l + self.cfg.pred]
            l_tot = l_tot + self.cfg.h
        n = int((N_tst - self.cfg.l - self.cfg.pred) / self.cfg.h) + 1
        test_data = np.zeros(shape=(n, self.cfg.l + self.cfg.pred))
        l_tot = 0
        for i in range(n):
            test_data[i, :] = test_sr[l_tot:l_tot + self.cfg.l + self.cfg.pred]
            l_tot = l_tot + self.cfg.h

        if self.flag == 'train':
            self.data_x = torch.tensor(train_data, dtype=torch.float32)
            self.dates = torch.tensor(dates_dt[0:N_tr].astype(int) // 10 ** 9, dtype=torch.float32)
            print(f"train data shape: {self.data_x.shape}")
        elif self.flag == 'val':
            self.data_x = torch.tensor(val_data, dtype=torch.float32)
            self.dates = torch.tensor(dates_dt[N_tr:N_tr + N_vl].astype(int) // 10 ** 9, dtype=torch.float32)
        else:
            self.data_x = torch.tensor(test_data, dtype=torch.float32)
            self.dates = torch.tensor(dates_dt[N_tr + N_vl:].astype(int) // 10 ** 9, dtype=torch.float32)


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        seq_x = self.data_x[s_begin,:]
        # seq_dates = self.dates[s_begin:s_end,:]

        # print(f"seq_x shape: {seq_x.shape}, seq_dates shape: {seq_dates.shape}")

        return seq_x#, seq_dates

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def excessreturns(self,s_df, e_df):
        """
        function to get a time series of alternating close and open
        etf-excess log returns for a given stock
        all prices are adjusted for stock events
        input: location of datasets, stock ticker, etf ticker
        output: time series of etf excess log returns
        optional: plot sanity check
        """
        # s_df = pd.read_csv(dataloc + stock + ".csv")
        # e_df = pd.read_csv(dataloc + etf + ".csv")
        dates_dt = pd.to_datetime(s_df['date'])
        d1 = pd.to_datetime(self.cfg.test_start_date)
        smp = (dates_dt < d1)
        s_df = s_df[smp]
        e_df = e_df[smp]
        s_logclose = np.log(s_df['AdjClose'])
        e_logclose = np.log(e_df['AdjClose'])
        s_logopen = np.log(s_df['AdjOpen'])
        e_logopen = np.log(e_df['AdjOpen'])
        s_log = np.zeros(2 * len(s_logclose))
        e_log = np.zeros(2 * len(s_logclose))
        for i in range(len(s_logclose)):
            s_log[2 * i] = s_logopen[i]
            s_log[2 * i + 1] = s_logclose[i]
            e_log[2 * i] = e_logopen[i]
            e_log[2 * i + 1] = e_logclose[i]
        s_ret = np.diff(s_log)
        e_ret = np.diff(e_log)
        s_ret[s_ret > 0.15] = 0.15
        s_ret[s_ret < -0.15] = -0.15
        e_ret[e_ret > 0.15] = 0.15
        e_ret[e_ret < -0.15] = -0.15
        excessret = s_ret - e_ret
        dates_dt = pd.to_datetime(s_df['date'])
        return excessret, dates_dt


