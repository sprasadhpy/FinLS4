import numpy as np
import pandas as pd

def rawreturns(cfg,s_df, e_df):
    """
    function to get a time series of raw log returns for a given stock/etf
    all prices are adjusted for stock events
    input: location of datasets, stock ticker, etf ticker
    output: time series of etf excess log returns
    optional: plot sanity check
    """
    dates_dt = pd.to_datetime(s_df['date'], format='%d/%m/%Y', errors='coerce')
    d1 = pd.to_datetime(cfg.test_start_date)
    smp = (dates_dt < d1)
    s_df = s_df[smp]
    dates_dt = pd.to_datetime(s_df['date'], format='%d/%m/%Y', errors='coerce')
    s_logclose = np.log(s_df['AdjClose'])
    s_logopen = np.log(s_df['AdjOpen'])
    s_log = np.zeros(2 * len(s_logclose))
    for i in range(len(s_logclose)):
        s_log[2 * i] = s_logopen[i]
        s_log[2 * i + 1] = s_logclose[i]
    s_ret = np.diff(s_log)
    s_ret[s_ret > 0.15] = 0.15
    s_ret[s_ret < -0.15] = -0.15
    dates_dt = pd.to_datetime(s_df['date'], format='%d/%m/%Y', errors='coerce')

    return s_ret, dates_dt