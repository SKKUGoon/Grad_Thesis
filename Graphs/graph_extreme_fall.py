import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

# Graph crisis events by date
t_starting_date = '2001-12-06'
t_start_date = datetime.date(2001, 12, 6)
t_ending_date = '2020-05-15'
t_end_date = datetime.date(2020, 5, 15)

countries_ame_dat = pd.read_csv(r"D:\Data\Grad\ame_total_dataset.csv")
countries_ame_dat = countries_ame_dat.set_index(countries_ame_dat.columns[0])
countries_eu_dat = pd.read_csv(r"D:\Data\Grad\eu_total_dataset.csv")
countries_eu_dat = countries_eu_dat.set_index(countries_eu_dat.columns[0])
countries_asia_dat = pd.read_csv(r"D:\Data\Grad\asia_total_dataset.csv")
countries_asia_dat = countries_asia_dat.set_index(countries_asia_dat.columns[0])

# Lag20
CrisisData_20lag = pd.read_csv(r"D:\Data\Grad\20lag_crisis_10p.csv", index_col = 0) # lag 1
print(CrisisData_20lag)

for_graph_temp = CrisisData_20lag[1502:6809]
for_graph = pd.DataFrame(for_graph_temp.sum(), columns = ['incidents'])
# Country-wise Crisis Incidents
plt.bar(for_graph.index, for_graph['incidents'])
plt.show()

# country example
Data5 = pd.read_csv(r"D:\Data\Grad\total_stock_index_dataset.csv") # Raw stock data (not return)
Data5 = Data5.set_index(Data5.columns[0])
Data5.index = pd.to_datetime(Data5.index)
Data5 = Data5.replace(0, np.nan)
Data5 = Data5.interpolate(method = 'time')

CrisisData_20lag2 = pd.read_csv(r"D:\Data\Grad\20global_regional10p.csv")
us_stock = Data5['US'][t_starting_date : t_ending_date]
us = pd.DataFrame(for_graph_temp['US'])
plt.plot(us_stock, 'r')
plt.bar(us.index, us['US']*10000)
plt.show()


# How many countries were in crisis(5% percentile)
Cris = CrisisData_20lag.sum(axis = 1)
for_graph2 = pd.DataFrame(Cris[1502:6809])
Cris = pd.DataFrame(Cris)
Cris.to_csv(r'D:\Data\Grad\Crisis_for_countries.csv')