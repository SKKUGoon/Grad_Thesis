import pandas as pd
import matplotlib.pyplot as plt
# Graph crisis events by date
data_indexing = pd.date_range(start='1/2/1997', periods=8535)
# Lag1
CrisisData_1lag = pd.read_csv(r"D:\Data\Grad\1lag_crisis.csv", index_col = 0) # lag 1
CrisisData1p1_lag = CrisisData_1lag[[CrisisData_1lag.columns[i] for i in range(30)]] # 1%
CrisisData5p1_lag = CrisisData_1lag[[CrisisData_1lag.columns[i] for i in range(30, 60)]]
CrisisData1_1lag = pd.DataFrame(CrisisData1p1_lag.sum(axis = 1))
CrisisData5_1lag = pd.DataFrame(CrisisData5p1_lag.sum(axis = 1))
# 1%, lag1
plt.plot(data_indexing, CrisisData1_1lag, 'r', label = '1% - lag1')
plt.legend()
plt.show()
# 5%, Lag1
plt.plot(data_indexing, CrisisData5_1lag, 'b', label = '5% - lag1')
plt.legend()
plt.show()

# Lag20
CrisisData_20lag = pd.read_csv(r"D:\Data\Grad\20lag_crisis.csv", index_col = 0) # lag 1
CrisisData1p_20lag = CrisisData_20lag[[CrisisData_20lag.columns[i] for i in range(30)]] # 1%
CrisisData5p_20lag = CrisisData_20lag[[CrisisData_20lag.columns[i] for i in range(30, 60)]]
CrisisData1_20lag = pd.DataFrame(CrisisData1p_20lag.sum(axis = 1))
CrisisData5_20lag = pd.DataFrame(CrisisData5p_20lag.sum(axis = 1))
# 1%, lag20
plt.plot(data_indexing, CrisisData1_20lag, 'r', label = '1% - lag20')
plt.legend()
plt.show()
# 5%, Lag20
plt.plot(data_indexing, CrisisData5_20lag, 'b', label = '5% - lag20')
plt.legend()
plt.show()