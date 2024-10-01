import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from data_provider.timefeatures import time_features
import warnings
from Utils.excessreturns import excessreturns
from Utils.rawreturns import rawreturns

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

        if self.cfg.current_ticker[0] != 'X':
            excess_returns, dates_dt = excessreturns(self.cfg,stock_df, etf_df)
        else:
            excess_returns, dates_dt = rawreturns(self.cfg,stock_df, etf_df)

        N = len(excess_returns)
        N_tr = int(self.cfg.tr * N)
        N_vl = int(self.cfg.vl * N)
        N_tst = N - N_tr - N_vl

        train_sr = excess_returns[0:N_tr]
        train_dt = dates_dt[0:N_tr]

        val_sr = excess_returns[N_tr:N_tr + N_vl]
        val_dt = dates_dt[N_tr:N_tr + N_vl]

        test_sr = excess_returns[N_tr + N_vl:]
        test_dt = dates_dt[N_tr + N_vl:]

        results_dir = os.path.join(self.cfg.results_loc, self.cfg.model, self.cfg.current_ticker)
        excess_returns_train_df = pd.DataFrame(train_sr, columns=['excess_returns'])
        excess_returns_train_df.to_csv(os.path.join(results_dir, 'excess_returns_train.csv'), index=False)

        # print(f"Excess returns for {self.cfg.current_ticker} saved to {results_dir}")


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
            # print(f"train data shape: {self.data_x.shape}")
        elif self.flag == 'val':
            self.data_x = torch.tensor(val_data, dtype=torch.float32)
        else:
            self.data_x = torch.tensor(test_data, dtype=torch.float32)



    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        seq_x = self.data_x[s_begin,:]
        # seq_dates = self.dates[s_begin:s_end,:]

        # print(f"seq_x shape: {seq_x.shape}, seq_dates shape: {seq_dates.shape}")

        return seq_x#, seq_dates

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Basis(Dataset):
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
        self.len = self.__len__()



    def __read_data__(self):

        stock = self.stock_path
        etf = self.etf_path

        stock_df = pd.read_csv(stock)
        etf_df = pd.read_csv(etf)

        if self.cfg.current_ticker[0] != 'X':
            excess_returns, dates_dt = excessreturns(self.cfg,stock_df, etf_df)
        else:
            excess_returns, dates_dt = rawreturns(self.cfg,stock_df, etf_df)

        N = len(excess_returns)
        N_tr = int(self.cfg.tr * N)
        N_vl = int(self.cfg.vl * N)
        N_tst = N - N_tr - N_vl
        train_sr = excess_returns[0:N_tr]
        val_sr = excess_returns[N_tr:N_tr + N_vl]
        train_sr = excess_returns[0:N_tr]
        val_sr = excess_returns[N_tr:N_tr + N_vl]
        test_sr = excess_returns[N_tr + N_vl:]

        results_dir = os.path.join(self.cfg.results_loc, self.cfg.model, self.cfg.current_ticker)
        excess_returns_train_df = pd.DataFrame(train_sr, columns=['excess_returns'])
        excess_returns_train_df.to_csv(os.path.join(results_dir, 'excess_returns_train.csv'), index=False)

        print(f"Excess returns for {self.cfg.current_ticker} saved to {results_dir}")



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

        # convert dates_dt from series to dataframe with column name 'date'
        df_stamp = pd.DataFrame(dates_dt, columns=['date'])
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = df_stamp.drop(['date'], axis=1).values

        train_dt = data_stamp[0:N_tr]
        val_dt = data_stamp[N_tr:N_tr + N_vl]
        test_dt = data_stamp[N_tr + N_vl:]

        n = int((N_tr - self.cfg.l - self.cfg.pred) / self.cfg.h) + 1
        train_dates = np.zeros(shape=(n, self.cfg.l + self.cfg.pred, data_stamp.shape[1]))
        l_tot = 0
        for i in range(n):
            train_dates[i, :] = train_dt[l_tot:l_tot + self.cfg.l + self.cfg.pred]
            l_tot = l_tot + self.cfg.h
        n = int((N_vl - self.cfg.l - self.cfg.pred) / self.cfg.h) + 1
        val_dates = np.zeros(shape=(n, self.cfg.l + self.cfg.pred, data_stamp.shape[1]))
        l_tot = 0
        for i in range(n):
            val_dates[i, :] = data_stamp[l_tot:l_tot + self.cfg.l + self.cfg.pred]
            l_tot = l_tot + self.cfg.h
        n = int((N_tst - self.cfg.l - self.cfg.pred) / self.cfg.h) + 1
        test_dates = np.zeros(shape=(n, self.cfg.l + self.cfg.pred, data_stamp.shape[1]))
        l_tot = 0
        for i in range(n):
            test_dates[i, :] = data_stamp[l_tot:l_tot + self.cfg.l + self.cfg.pred]
            l_tot = l_tot + self.cfg.h

        if self.flag == 'train':
            self.data_x = torch.tensor(train_data, dtype=torch.float32)
            self.dates = torch.tensor(train_dates, dtype=torch.float32)
            print(f"train data shape: {self.data_x.shape}")
            print(f"train dates shape: {self.dates.shape}")
        elif self.flag == 'val':
            self.data_x = torch.tensor(val_data, dtype=torch.float32)
            self.dates = torch.tensor(val_dates, dtype=torch.float32)
        else:
            self.data_x = torch.tensor(test_data, dtype=torch.float32)
            self.dates = torch.tensor(test_dates, dtype=torch.float32)


    def __getitem__(self, index):
        s_begin = index
        r_begin = s_begin + self.seq_len

        seq_x = self.data_x[s_begin,:]
        seq_y = self.data_x[r_begin,:]

        seq_x_mark = self.dates[s_begin,:]
        seq_y_mark = self.dates[r_begin,:]

        index_list = np.arrange(index, index + self.seq_len + self.pred_len,1)

        norm_index = index_list / self.len

        # print(f"seq_x shape: {seq_x.shape}, seq_dates shape: {seq_dates.shape}")

        return seq_x, seq_y, seq_x_mark, seq_y_mark, norm_index

    def __len__(self):
        return len(self.data_x)




