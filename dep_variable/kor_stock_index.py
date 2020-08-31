import pandas as pd
import datetime
from dep_variable.methods import Preprocessing

t_starting_date = '2001-12-06'
t_starting_date = datetime.date(2001, 12, 6)
t_ending_date = '2020-05-15'
t_end_date = datetime.date(2020, 5, 15)

# Get Asia Data
Data = pd.read_csv(r"D:\Data\Grad\total_stock_index_dataset.csv")
Data = Data.set_index(Data.columns[0])
Data.index = pd.to_datetime(Data.index)
Data_kor = pd.DataFrame(Data['kor'])
Data_kor = Data_kor.fillna(method='ffill') # Working Data.

# Lag : 1
WD1 = Preprocessing(Data_kor)
WD1 = WD1.log_return_class(1, 0)
WD1.to_csv(r"D:\Data\Grad\Kor_log_ret_class_lag1.csv")
