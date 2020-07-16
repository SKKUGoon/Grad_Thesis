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

# Lag : 20
WorkData = preprocessing(Data_kor)
WorkData = WorkData.log_return(20, 0)
WD = preprocessing(WorkData)
WD = WD.log_return_class(0)
