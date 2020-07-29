import pandas as pd
import numpy as np
import datetime
from Dep_variable.Methods import preprocessing

t_starting_date = '2001-12-06'
t_starting_date = datetime.date(2001, 12, 6)
t_ending_date = '2020-05-15'
t_end_date = datetime.date(2020, 5, 15)

# Get Asia Data
Data = pd.read_csv(r"D:\Data\Grad\total_stock_index_dataset.csv")
Data = Data.set_index(Data.columns[0])
Data_kor = pd.DataFrame(Data['kor']) # Working Data

# Lag : 1
WorkData = preprocessing(Data_kor)
WorkData = WorkData.log_return(1, 0)
WD = preprocessing(WorkData)
WD = WD.log_return_class(0)
WD.to_csv(r"D:\Data\Grad\Kor_log_ret_class_lag1.csv")

# Lag : 3
WorkData = preprocessing(Data_kor)
WorkData = WorkData.log_return(3, 0) # Even though the front most value is np.nan / Data will be cutted from 2001. So no worries
WD = preprocessing(WorkData)
WD = WD.log_return_class(0)
WD = WD.rename(columns={'kor class': 'kor class_lag3'})
WD.to_csv(r"D:\Data\Grad\Kor_log_ret_class_lag3.csv")
print(WD)