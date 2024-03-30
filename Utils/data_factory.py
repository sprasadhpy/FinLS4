from Utils.dataloader import Dataset_Custom, Dataset_Basis
from torch.utils.data import DataLoader
import os

data_dict = {
    'custom': Dataset_Custom,
    'basis': Dataset_Basis,
}



def data_provider(cfg, flag):
    Data = data_dict[cfg.data]



    data_set = Data(
        root_path=cfg.dataloc,
        stock_path=cfg.dataloc + r'/'+cfg.current_ticker + '.csv',
        etf_path=cfg.dataloc + r'/'+cfg.current_etf + '.csv',
        flag=flag,
        size=[cfg.l, cfg.h, cfg.pred],
        cfg=cfg
    )

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = len(data_set)  # Set batch size to the entire data
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = cfg.batch_size


    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=cfg.num_workers,
        drop_last=drop_last)
    return data_set, data_loader