import pandas as pd
import numpy as np
import datetime
from dep_variable.methods import Preprocessing

def refurbish_dataframe(dataframe_, nan_option='Discrete', lag_option=0) -> pd.DataFrame: # from now on use this
    dataframe_ = dataframe_.set_index(dataframe_.columns[0])
    dataframe_.index = pd.to_datetime(dataframe_.index)

    if nan_option == 'Discrete':
        dataframe_ = dataframe_.replace(0, np.nan)
        dataframe_ = dataframe_.fillna(-1)
    elif nan_option == 'Continuous':
        dataframe_ = dataframe_.interpolate(method='time')
    else: # don't touch np.nan
        pass

    dataframe_ = dataframe_.shift(lag_option)
    return dataframe_

data1 = pd.read_csv(r"D:\Data\Grad\add_data.csv") # Addition data(i.e. gold, oil)
data2 = pd.read_csv(r"D:\Data\Grad\fx_data.csv") # Exchange_rate
data3 = pd.read_csv(r"D:\Data\Grad\bonds_dataset.csv") # Raw bond data
data4 = pd.read_csv(r"D:\Data\Grad\total_stock_index_dataset.csv") # Raw stock data (not return)
data_kor = pd.read_csv(r"D:\Data\Grad\Kor_log_ret_class_lag1.csv") # Korea classified Data
data_kor_3days_return = pd.read_csv(r"D:\Data\Grad\Kor_log_ret_class_lag3.csv") # Korea 3 days Data
data_volume = pd.read_csv(r'D:\Data\Grad\Kospi_volume.csv')

using = [data1, data2, data3, data4, data_kor, data_kor_3days_return, data_volume]
type_ = ['Continuous', 'Continuous', 'Continuous', 'Continuous', 'Discrete', 'Discrete', 'Continuous']

for i in range(len(using)):
    using[i] = refurbish_dataframe(using[i], nan_option=type_[i], lag_option=0)




# Create Lag
lag_value = [1, 2, 3, 4, 5, 20, 30, 60, 90] # 1day, 2day, 3day, ... , 3 months
lagging_data = [data2, data3, Data5, lnData5, Data6] # lagging only exchange_rate, raw stock data, and additional data
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





Work_Data = pd.concat([data1,
                       data2[starting_date: ending_date],
                       data3[starting_date: ending_date],
                       Data5[starting_date : ending_date],
                       lnData5[starting_date : ending_date],
                       Data6[starting_date : ending_date],
                       ln_return_Data5[starting_date : ending_date],
                       volatility_Data5[starting_date : ending_date],
                       lag_temp[starting_date : ending_date]], axis = 1)

Work_Data_without_lag = pd.concat([data2[starting_date: ending_date],
                                   data3[starting_date: ending_date],
                                   Data5[starting_date : ending_date],
                                   Data6[starting_date : ending_date],
                                   Data_volume[starting_date : ending_date]], axis = 1)

Work_Data_with_lag = pd.concat([data2[starting_date: ending_date],
                                data3[starting_date: ending_date],
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
data_kor = data_kor[t_starting_date: t_ending_date]
data_kor_3days_return = data_kor_3days_return[starting_date: ending_date]

t_Work_Data_without_lag = pd.concat([data_kor, data_kor_3days_return, t_Work_Data_without_lag], axis = 1)
t_Work_Data_without_lag = t_Work_Data_without_lag.fillna(method = 'ffill') # Use the value before the NaN to fill it.
t_Work_Data_without_lag.to_csv(r'D:\Data\Grad\test_Work_Data_wo_lag.csv', index = True) # save as csv

# Data with lag. Original Data add ln_return and volatility
t_Work_Data_with_lag = Work_Data_with_lag[t_starting_date : t_ending_date]
t_Work_Data_with_lag.index = pd.to_datetime(t_Work_Data_with_lag.index)
t_Work_Data_with_lag = t_Work_Data_with_lag.interpolate(method = 'time')

# Data_kor and Data_kor_3days is separated because they are discrete values
data_kor = data_kor.shift(-1) #
data_kor = data_kor[t_starting_date: t_ending_date]
data_kor_3days_return = data_kor_3days_return.shift(-1)
data_kor_3days_return = data_kor_3days_return[t_starting_date: t_ending_date]

t_Work_Data_with_lag = pd.concat([data_kor, data_kor_3days_return, t_Work_Data_with_lag], axis = 1)
t_Work_Data_with_lag = t_Work_Data_with_lag.fillna(method = 'ffill') # Use the value before the NaN to fill it.
t_Work_Data_with_lag.to_csv(r'D:\Data\Grad\test_Work_Data_w_lag.csv', index = True) # save as csv

# without lag 6736 * 117  size
# with lag 6736 * 1550  size