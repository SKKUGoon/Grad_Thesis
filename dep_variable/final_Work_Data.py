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

data1 = pd.read_csv(r"D:\Data\Grad\add_data.csv") # Addition data(i.e. gold, oil)
data2 = pd.read_csv(r"D:\Data\Grad\fx_data.csv") # Exchange_rate
data3 = pd.read_csv(r"D:\Data\Grad\bonds_dataset.csv") # Raw bond data
data4 = pd.read_csv(r"D:\Data\Grad\total_stock_index_dataset.csv") # Raw stock data (not return)
data_kor = pd.read_csv(r"D:\Data\Grad\Kor_log_ret_class_lag1.csv") # Korea classified Data
data_volume = pd.read_csv(r'D:\Data\Grad\Kospi_volume.csv')

use = ['data1', 'data2', 'data3', 'data4',
       'data_kor', 'data_volume']
using = [data1, data2, data3, data4,
         data_kor, data_volume]
type_ = ['Continuous_no_future', 'Continuous_no_future', 'Continuous_no_future', 'Continuous_no_future',
         'Discrete', 'Continuous_no_future']

for i in range(len(using)):
    using[i] = refurbish_dataframe(using[i], nan_option=type_[i], lag_option=0)


# Create Lag
lag_value = [1, 2, 7] # week, and month lag.
lagged_data = {}
for lag in lag_value:
    for d in range(len(using)):
        lagged = using[d].shift(lag)
        col = {}
        for k in lagged.columns:
            col[k] = k + ' ' + str(lag) + 'lag'
        lagged = lagged.rename(columns=col)
        lagged_data[use[d] + ' with lag' + str(lag)] = lagged

keyvalue = list(lagged_data.keys())
lag_temp = lagged_data[keyvalue[0]]
for i in range(len(lagged_data.keys())):
    lag_temp = pd.concat([lag_temp, lagged_data[keyvalue[i]]], axis=1)

# Create log returns for stock indexes(over a day)
ln_d4 = np.log(using[3]) - np.log(using[3].shift(7))
ln_d4 = renamecolumns(ln_d4, '7dayrt')
lnsq_d4 = ln_d4 ** 2 # sqrd to take in the effect of return-volatility
lnsq_d4 = renamecolumns(lnsq_d4, '7dayvol')
# Cut data
s = '2001-12-06'
sd = datetime.date(2001, 12, 6)
e = '2020-05-14'
ed = datetime.date(2020, 5, 15)

fwd = pd.concat([using[4].shift(-1), # using[4] and contains dependent data.
                 using[0], using[1], using[2], using[3],
                 using[5], lag_temp, lnsq_d4.shift(1), ln_d4.shift(1)], axis=1)

fwd = fwd[s : e]
# U.S 1Y 10Y Japan 1Y 10Y has nans.
# nz has inf when taken log
# Delete them for now
fwd = fwd.replace([np.inf, -np.inf, 0], np.nan)
fwd = fwd.dropna(axis='columns')
fwd.to_csv(r'D:\Data\Grad\test_Work_Data_w_lag.csv', index=True)
print(fwd)