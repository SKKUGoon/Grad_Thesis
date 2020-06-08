import pandas as pd
from Methods import preprocessing
# Creating Total Work Space
Data1 = pd.read_csv(r"D:\Data\Grad\total_stock_index_dataset.csv") # Stock data
Data2 = pd.read_csv(r"D:\Data\Grad\add_data.csv") # Addition data(i.e. gold, oil)
Data3 = pd.read_csv(r"D:\Data\Grad\fx_data.csv") # Exchange_rate
Data_list = [Data1, Data2, Data3] # If you add something, add to this Data_list
Indexed_list = []
for i in Data_list:
    i = i.set_index(i.columns[0])
    Indexed_list.append(i)
Work_Data = pd.concat([Indexed_list[i] for i in range(len(Indexed_list))], axis = 1)
Work_Data.index = pd.to_datetime(Work_Data.index)
Work_Data = Work_Data.interpolate(method = 'time') # Interpolate := fill in the missing numbers according to time.
                                                   # Which means it fills in the Saturday and
Work_Data = Work_Data[1:] # Erase January the First, Since it's all NaN, and interpolate doesn't work.

# Empirical Distribution of Returns for Stocks
WD = preprocessing(Work_Data)

yield_dist_temp_1 = {} # To make it into DataFrame, we use Dictionary first.
for i in range((len(Data1.columns)-1)): # Data1 = stock. Stock column number - 1(-1 because index was included in Data 1)
    yield_dist_temp_temp_temp = [] # Empty list to contain empirical dist from the class_preprocessing
    yield_dist_temp_temp = WD.empirical_dist(1, i) # lag1, ith column
    yield_dist_temp_temp_temp.append(yield_dist_temp_temp) # empty list containing dataframe
    yield_dist_temp_1[Data1.columns[i+1]] = list(yield_dist_temp_temp_temp[0]) # temp_temp_temp no need to exist. Fix it later
yield_dist_1 = pd.DataFrame.from_dict(yield_dist_temp_1) # Dictionary to DataFrame.

yield_dist_temp_20 = {}
for i in range((len(Data1.columns)-1)):
    yield_dist_temp_temp_temp = []
    yield_dist_temp_temp = WD.empirical_dist(1, i)
    yield_dist_temp_temp_temp.append(yield_dist_temp_temp)
    yield_dist_temp_20[Data1.columns[i+1]] = list(yield_dist_temp_temp_temp[0])
yield_dist_20 = pd.DataFrame.from_dict(yield_dist_temp_20)

# Create Crisis ; Dependent Variable. For lag (1)
YD = preprocessing(yield_dist_1)
crisis_code_temp_01_lag1 = {}
crisis_code_temp_05_lag1 = {} # again make dictionary

for i in range((len(yield_dist_1.columns))):
    crisis_code_temp_temp1 = list(YD.crisis_coding(0.01, i)[0]) # 1% percentile
    crisis_code_temp_01_lag1[Data1.columns[i+1]+'1p'] = crisis_code_temp_temp1
    crisis_code_temp_temp5 = list(YD.crisis_coding(0.05, i)[0]) # 5% percentile
    crisis_code_temp_05_lag1[Data1.columns[i+1]+'5p'] = crisis_code_temp_temp5

crisis_var_1_1lag = pd.DataFrame.from_dict(crisis_code_temp_01_lag1) # 1 lag, 1% percentile
crisis_var_5_1lag = pd.DataFrame.from_dict(crisis_code_temp_05_lag1) # 1 lag, 5% percentile

crisis_var_1lag = pd.concat([crisis_var_1_1lag, crisis_var_5_1lag], axis = 1) # add them together
data_indexing = pd.date_range(start='1/2/1997', periods=8535)
date = pd.DataFrame(data_indexing, columns = ['date'])
crisis_var_1lag = pd.concat([date, crisis_var_1lag], axis = 1)
crisis_var_1lag = crisis_var_1lag.set_index('date')
crisis_var_1lag.to_csv(r"D:\Data\Grad\1lag_crisis.csv")


# Create Crisis ; Dependent Variable. For lag (20)
YD = preprocessing(yield_dist_20)
crisis_code_temp_01 = {}
crisis_code_temp_05 = {} # again make dictionary

for i in range((len(yield_dist_20.columns))):
    crisis_code_temp_temp1 = list(YD.crisis_coding(0.01, i)[0])
    crisis_code_temp_01[Data1.columns[i+1]+'1p'] = crisis_code_temp_temp1
    crisis_code_temp_temp5 = list(YD.crisis_coding(0.05, i)[0])
    crisis_code_temp_05[Data1.columns[i+1]+'5p'] = crisis_code_temp_temp5

crisis_var_1 = pd.DataFrame.from_dict(crisis_code_temp_01)
crisis_var_5 = pd.DataFrame.from_dict(crisis_code_temp_05)

crisis_var = pd.concat([crisis_var_1, crisis_var_5], axis = 1)
crisis_var = pd.concat([date, crisis_var], axis = 1) # already defined date from above
crisis_var = crisis_var.set_index('date')
crisis_var.to_csv(r"D:\Data\Grad\20lag_crisis.csv")
