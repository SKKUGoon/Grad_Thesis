import pandas as pd
from Dep_variable.Methods import preprocessing
# Creating Total Work Space
# Deprecated

Data1 = pd.read_csv(r"D:\Data\Grad\total_stock_index_dataset.csv") # Stock data
Data2 = pd.read_csv(r"D:\Data\Grad\add_data.csv") # Addition data(i.e. gold, oil)
Data3 = pd.read_csv(r"D:\Data\Grad\fx_data.csv") # Exchange_rate
Data_list = [Data1] # If you add something, add to this Data_list
Indexed_list = []
for i in Data_list:
    i = i.set_index(i.columns[0])
    Indexed_list.append(i)
Work_Data = pd.concat([Indexed_list[i] for i in range(len(Indexed_list))], axis = 1)
Work_Data.index = pd.to_datetime(Work_Data.index)
Work_Data = Work_Data.interpolate(method = 'time') # Interpolate := fill in the missing numbers according to time.
                                                   # Which means it fills in the Saturday and
Work_Data = Work_Data[1:] # Erase January the First, Since it's all NaN, and interpolate doesn't work.

# 1 lag crisis coding 1%
WD = preprocessing(Work_Data)
crisis_var_1lag_temp = {}
for i in range(len(Work_Data.columns)):
    crisis_var_1lag_temp[Work_Data.columns[i]] = WD.crisis_code_200(i, 1, 0.01)
    print('processing column', i)
crisis_var_1lag = pd.DataFrame(None)
for key in crisis_var_1lag_temp.keys():
    crisis_var_1lag = pd.concat([crisis_var_1lag, crisis_var_1lag_temp[key]], axis = 1)
crisis_var_1lag.to_csv(r"D:\Data\Grad\1lag_crisis.csv")

# 30 lag crisis coding 1% named(20lag_crisis) 20 working days
crisis_var_20lag_temp = {}
for i in range(len(Work_Data.columns)):
    crisis_var_20lag_temp[Work_Data.columns[i]] = WD.crisis_code_200(i, 20, 0.01)
    print('processing column', i)
crisis_var_20lag = pd.DataFrame(None)
for key in crisis_var_20lag_temp.keys():
    crisis_var_20lag = pd.concat([crisis_var_20lag, crisis_var_20lag_temp[key]], axis = 1)
crisis_var_20lag.to_csv(r"D:\Data\Grad\20lag_crisis.csv")

# 1 lag crisis coding 5%
WD = preprocessing(Work_Data)
crisis_var_1lag_temp = {}
for i in range(len(Work_Data.columns)):
    crisis_var_1lag_temp[Work_Data.columns[i]] = WD.crisis_code_200(i, 1, 0.05)
    print('processing column', i)
crisis_var_1lag = pd.DataFrame(None)
for key in crisis_var_1lag_temp.keys():
    crisis_var_1lag = pd.concat([crisis_var_1lag, crisis_var_1lag_temp[key]], axis = 1)
crisis_var_1lag.to_csv(r"D:\Data\Grad\1lag_crisis_5p.csv")

# 30 lag crisis coding 5% named(20lag_crisis) 20 working days
crisis_var_20lag_temp = {}
for i in range(len(Work_Data.columns)):
    crisis_var_20lag_temp[Work_Data.columns[i]] = WD.crisis_code_200(i, 20, 0.05)
    print('processing column', i)
crisis_var_20lag = pd.DataFrame(None)
for key in crisis_var_20lag_temp.keys():
    crisis_var_20lag = pd.concat([crisis_var_20lag, crisis_var_20lag_temp[key]], axis = 1)
crisis_var_20lag.to_csv(r"D:\Data\Grad\20lag_crisis_5p.csv")

# 30 lag crisis coding 10% named(20lag_crisis) 20 working days
crisis_var_20lag_temp = {}
for i in range(len(Work_Data.columns)):
    crisis_var_20lag_temp[Work_Data.columns[i]] = WD.crisis_code_200(i, 20, 0.10)
    print('processing column', i)
crisis_var_20lag = pd.DataFrame(None)
for key in crisis_var_20lag_temp.keys():
    crisis_var_20lag = pd.concat([crisis_var_20lag, crisis_var_20lag_temp[key]], axis = 1)
crisis_var_20lag.to_csv(r"D:\Data\Grad\20lag_crisis_10p.csv")