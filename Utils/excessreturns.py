import numpy as np
import pandas as pd

def excessreturns(cfg,s_df, e_df):
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
    d1 = pd.to_datetime(cfg.test_start_date)
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