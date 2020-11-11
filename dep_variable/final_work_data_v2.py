import pandas as pd
import numpy as np

import datetime
import pickle


def set_date_index(dataframe_:pd.DataFrame, date_col_loc=0) -> pd.DataFrame:
    day = dataframe_.columns[date_col_loc]
    dataframe_ = dataframe_.set_index(dataframe_[day])
    dataframe_ = dataframe_[dataframe_.columns[1:]]

    dataframe_.index = pd.to_datetime(dataframe_.index)

    return dataframe_


data1 = pd.read_csv(r"D:\Data\Grad\add_data.csv") # Addition data(i.e. gold, oil)
data2 = pd.read_csv(r"D:\Data\Grad\fx_data.csv") # Exchange_rate
data3 = pd.read_csv(r"D:\Data\Grad\bonds_dataset.csv") # Raw bond data
data4 = pd.read_csv(r"D:\Data\Grad\total_stock_index_dataset.csv") # Raw stock data (not return)
data5 = pd.read_csv(r"D:\Data\Grad\Kor_log_ret_class_lag1.csv") # Korea classified Data
data6 = pd.read_csv(r'D:\Data\Grad\Kospi_volume.csv')
data7 = pd.read_csv(r'D:\Data\Grad\individual_stock_dataset.csv')

total_dat = [data1, data2, data3, data4, data5, data6, data7]
total_dat = list(map(set_date_index, total_dat))

total_df = pd.concat([*total_dat], axis=1)

for col in total_df.columns:
    if col != total_dat[4].columns[0]:  # This is for Classification
        total_df[[col]] = np.log(total_df[[col]])

test_df = total_df[total_df.index >= datetime.datetime(2014, 1, 3)]
test_df = test_df[test_df.index < datetime.datetime(2020, 5, 1)]
test_df = test_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.fillna(method='ffill')
test_df = test_df.dropna(axis='columns')  # Original Dataset without the lagged variable has 192 columns

# Starting Date from (2014, 1, 3) to ending date of (2020, 5, 1)

def create_diff(dataframe_:pd.DataFrame, diff_:int) -> pd.DataFrame:
    cols = dataframe_.columns
    y_name = 'korclf1'
    res = pd.DataFrame(None)
    for col in cols:
        if col != y_name:
            addit = dataframe_[[col]] - dataframe_[[col]].shift(diff_)
            addit = addit.rename(columns={col : f'{col}_{diff_}diff'})
            res = pd.concat([res, addit], axis=1)

    return res

dff1 = create_diff(test_df, 1)
dff2 = create_diff(test_df, 7)
dff3 = create_diff(test_df, 30)

fwd = pd.concat([test_df, dff1, dff2, dff3], axis=1)
fwd = fwd.dropna(axis='rows')
print(fwd)
fwd.to_pickle(r'.\fwd.pkl')