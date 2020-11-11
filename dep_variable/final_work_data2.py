import pandas as pd
import numpy as np
import datetime

def refurbish_dataframe(dataframe_, nan_option='Discrete', lag_option=0) -> pd.DataFrame: # from now on use this
    dataframe_ = dataframe_.set_index(dataframe_.columns[0])
    dataframe_.index = pd.to_datetime(dataframe_.index)

    if nan_option == 'Discrete':
        dataframe_ = dataframe_.replace(0, np.nan)
        dataframe_ = dataframe_.fillna(-1)
    elif nan_option == 'Continuous_no_future':
        dataframe_ = dataframe_.fillna(method='ffill')
    elif nan_option == 'Continuous_yes_future':
        dataframe_ = dataframe_.interpolate(method='time')
    else: # don't touch np.nan
        pass

    dataframe_ = dataframe_.shift(lag_option)
    return dataframe_


def renamecolumns(original: pd.DataFrame, rename_with: str) -> pd.DataFrame:
    """
    Renames the DataFrame with particular str(rename_with)
    """
    o = list(original.columns)
    res = list()
    for names in o:
        ren = names + ' ' + rename_with
        res.append(ren)
    # Renaming purposed Dictionary
    rendic = dict()
    for i in range(len(res)):
        rendic[o[i]] = res[i]

    renamed = original.rename(columns=rendic)

    return renamed

additional = pd.read_csv(r"D:\Data\Grad\add_data.csv") # Addition data(i.e. gold, oil)
fx = pd.read_csv(r"D:\Data\Grad\fx_data.csv") # Exchange_rate
bond = pd.read_csv(r"D:\Data\Grad\bonds_dataset.csv") # Raw bond data
stock = pd.read_csv(r"D:\Data\Grad\total_stock_index_dataset.csv") # Raw stock data (not return)
data_kor = pd.read_csv(r"D:\Data\Grad\Kor_log_ret_class_lag1.csv") # Korea classified Data
data_volume = pd.read_csv(r'D:\Data\Grad\Kospi_volume.csv')

