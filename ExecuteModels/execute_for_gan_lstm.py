import torch
import pandas as pd

from Utils.data_factory import data_provider

def execute_for_gan_lstm(cfg):



    cfg.data = "custom"

    print("\nCalculating excess returns and splitting data into train, validation and test sets...\n")

    train_data, train_loader = data_provider(cfg, 'train')
    val_data, val_loader = data_provider(cfg, 'val')
    test_data, test_loader = data_provider(cfg, 'test')

    print("\nData Splitting Completed...\n")

