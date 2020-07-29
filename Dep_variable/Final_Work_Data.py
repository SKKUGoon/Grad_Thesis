import pandas as pd
import numpy as np
import datetime
from Dep_variable.Methods import preprocessing
# Cleaning Data
Data1 = pd.read_csv(r"D:\Data\Grad\20global_regional10p.csv") # Transformed global crisis Data lag 20
Data1 = Data1.set_index(Data1.columns[0])
Data1.index = pd.to_datetime(Data1.index)

Data2 = pd.read_csv(r"D:\Data\Grad\add_data.csv") # Addition data(i.e. gold, oil)
Data2 = Data2.set_index(Data2.columns[0])
Data2.index = pd.to_datetime(Data2.index)

Data3 = pd.read_csv(r"D:\Data\Grad\fx_data.csv") # Exchange_rate
Data3 = Data3.set_index(Data3.columns[0])
Data3.index = pd.to_datetime(Data3.index)

Data4 = pd.read_csv(r"D:\Data\Grad\1lag_crisis.csv") # Stock Crisis 1lag
Data4 = Data4.set_index(Data4.columns[0])
Data4.index = pd.to_datetime(Data4.index)

Data6 = pd.read_csv(r"D:\Data\Grad\bonds_dataset.csv") # Raw bond data
Data6 = Data6.set_index(Data6.columns[0])
Data6.index = pd.to_datetime(Data6.index)
Data6 = Data6.interpolate(method = 'time')
Data6 = Data6.fillna(0)

Data5 = pd.read_csv(r"D:\Data\Grad\total_stock_index_dataset.csv") # Raw stock data (not return)
Data5 = Data5.set_index(Data5.columns[0])
Data5.index = pd.to_datetime(Data5.index)
Data5 = Data5.replace(0, np.nan)
Data5 = Data5.interpolate(method = 'time')
need_log_t = Data5.columns
lnData5 = np.log(Data5[[i for i in need_log_t]])
col_lnData5 = {}
for i in range(len(lnData5.columns)):
    col_lnData5[lnData5.columns[i]] = 'log' + lnData5.columns[i]
lnData5 = lnData5.rename(columns = col_lnData5)
d5 = preprocessing(lnData5)

# Korea Classification Datas
Data_kor = pd.read_csv(r"D:\Data\Grad\Kor_log_ret_class_lag1.csv") # Korea classified Data
Data_kor = Data_kor.set_index(Data_kor.columns[0])
Data_kor.index = pd.to_datetime(Data_kor.index)
Data_kor = Data_kor.replace(0, np.nan)
Data_kor = Data_kor.fillna(-1)

def refurbish_dataframe(DataFrame_, nan_option='Discrete', lag_option=0) -> pd.DataFrame: # from now on use this
    DataFrame_ = DataFrame_.set_index(DataFrame_.columns[0])
    DataFrame_.index = pd.to_datetime(DataFrame_.index)

    if nan_option == 'Discrete':
        DataFrame_ = DataFrame_.replace(0, np.nan)
        DataFrame_ = DataFrame_.fillna(-1)
    else: # Continuous Data
        DataFrame_ = DataFrame_.interpolate(method='time')

    DataFrame_ = DataFrame_.shift(lag_option)
    return DataFrame_

# 3 days return result "UNTIL YESTERDAY"
Data_kor_3days_return = pd.read_csv(r"D:\Data\Grad\Kor_log_ret_class_lag3.csv") # Korea 3 days Data
Data_kor_3days_return = refurbish_dataframe(Data_kor_3days_return, lag_option=1)

# KOSPI Trade Volume 'UNTIL YESTERDAY'
Data_Volume = pd.read_csv(r'D:\Data\Grad\Kospi_volume.csv')
Data_volume = refurbish_dataframe(Data_Volume, nan_option='Continuous', lag_option=1)

# Take Log
log_return_stock = {} # Null Dict to contain log returns
log_return_sqr_stock = {}
for i in range(len(lnData5.columns)):
    lnData5_log_return, lnData5_log_return_2 = d5.empirical_dist(1, i), d5.empirical_dist(1, i)**2 # log return, volatility
    log_return_stock[lnData5.columns[i]] = lnData5_log_return
    log_return_sqr_stock[lnData5.columns[i]] = lnData5_log_return_2
ln_return_Data5 = pd.DataFrame.from_dict(log_return_stock) # Dataframe from dict
volatility_Data5 = pd.DataFrame.from_dict(log_return_stock)

col_ln_return = {} # change the column name of the logged variables.
col_ln_volatility = {} # This one is for the sqrd ones.
for k in range(len(ln_return_Data5.columns)): # Change the columns' name including the lag value
    col_ln_return[ln_return_Data5.columns[k]] = ln_return_Data5.columns[k] + ' ' + 'ln_ret'
    col_ln_volatility[volatility_Data5.columns[k]] = volatility_Data5.columns[k] + ' ' + 'ln_vol'

ln_return_Data5 = ln_return_Data5.rename(columns = col_ln_return) # column name change
volatility_Data5 = volatility_Data5.rename(columns = col_ln_volatility)

# Create Lag
lag_value = [1, 2, 3, 4, 5, 20, 30, 60, 90] # 1day, 2day, 3day, ... , 3 months
lagging_data = [Data2, Data3, Data5, lnData5, Data6] # lagging only exchange_rate, raw stock data, and additional data
lagged_data_key = ['Data2', 'Data3', 'Data5', 'lnData5', 'Data6']
lagged_data = {}
for lag in lag_value:
    for d in range(len(lagging_data)):
        la = lagging_data[d].shift(lag) # shift by lag value list
        col = {}
        for k in range(len(lagging_data[d].columns)): # Change the columns' name including the lag value
            col[la.columns[k]] = la.columns[k] + ' ' + str(lag) + 'lag'
        la = la.rename(columns = col) # rename the columns with the col dictionary
        lagged_data[lagged_data_key[d] +' with lag'+ str(lag)] = la

keyvalue = list(lagged_data.keys()) # turning the dictionary into dataframe.
lag_temp = lagged_data[keyvalue[0]]
for i in range(len(lagged_data.keys())-1):
    lag_temp = pd.concat([lag_temp, lagged_data[keyvalue[i+1]]], axis = 1)

# Congregate all of the defined data.
starting_date = '2001-08-13'
start_date = datetime.date(2001, 8, 13)
ending_date = '2020-05-15'
end_date = datetime.date(2020, 5, 15)

Work_Data = pd.concat([Data1,
                       Data2[starting_date : ending_date],
                       Data3[starting_date : ending_date],
                       Data5[starting_date : ending_date],
                       lnData5[starting_date : ending_date],
                       Data6[starting_date : ending_date],
                       ln_return_Data5[starting_date : ending_date],
                       volatility_Data5[starting_date : ending_date],
                       lag_temp[starting_date : ending_date]], axis = 1)

Work_Data_without_lag = pd.concat([Data2[starting_date : ending_date],
                                   Data3[starting_date : ending_date],
                                   Data5[starting_date : ending_date],
                                   Data6[starting_date : ending_date],
                                   Data_volume[starting_date : ending_date]], axis = 1)

Work_Data_with_lag = pd.concat([Data2[starting_date : ending_date],
                                Data3[starting_date : ending_date],
                                Data5[starting_date : ending_date],
                                Data6[starting_date : ending_date],
                                ln_return_Data5[starting_date: ending_date],
                                volatility_Data5[starting_date: ending_date],
                                Data_volume[starting_date : ending_date],
                                lag_temp[starting_date : ending_date]], axis = 1)


# Testing. Use Full Data Next Time
t_starting_date = '2001-12-06'
t_start_date = datetime.date(2001, 12, 6)
t_ending_date = '2020-05-15'
t_end_date = datetime.date(2020, 5, 15)

# This is the data without lag - From now use this. 2020-07-17 Original Data
t_Work_Data_without_lag = Work_Data_without_lag[t_starting_date : t_ending_date]
t_Work_Data_without_lag.index = pd.to_datetime(t_Work_Data_without_lag.index)
t_Work_Data_without_lag = t_Work_Data_without_lag.interpolate(method = 'time')

# Data_kor and Data_kor_3days is separated because they are discrete values
Data_kor = Data_kor[t_starting_date : t_ending_date]
Data_kor_3days_return = Data_kor_3days_return[starting_date : ending_date]

t_Work_Data_without_lag = pd.concat([Data_kor, Data_kor_3days_return, t_Work_Data_without_lag], axis = 1)
t_Work_Data_without_lag = t_Work_Data_without_lag.fillna(method = 'ffill') # Use the value before the NaN to fill it.
t_Work_Data_without_lag.to_csv(r'D:\Data\Grad\test_Work_Data_wo_lag.csv', index = True) # save as csv

# Data with lag. Original Data add ln_return and volatility
t_Work_Data_with_lag = Work_Data_with_lag[t_starting_date : t_ending_date]
t_Work_Data_with_lag.index = pd.to_datetime(t_Work_Data_with_lag.index)
t_Work_Data_with_lag = t_Work_Data_with_lag.interpolate(method = 'time')

# Data_kor and Data_kor_3days is separated because they are discrete values
Data_kor = Data_kor[t_starting_date : t_ending_date]
Data_kor_3days_return = Data_kor_3days_return[t_starting_date : t_ending_date]

t_Work_Data_with_lag = pd.concat([Data_kor, Data_kor_3days_return, t_Work_Data_with_lag], axis = 1)
t_Work_Data_with_lag = t_Work_Data_with_lag.fillna(method = 'ffill') # Use the value before the NaN to fill it.
t_Work_Data_with_lag.to_csv(r'D:\Data\Grad\test_Work_Data_w_lag.csv', index = True) # save as csv

# without lag 6736 * 117  size
# with lag 6736 * 1550  size