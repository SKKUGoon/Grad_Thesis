import pandas as pd
import matplotlib.pyplot as plt
# Graph crisis events by date
data_indexing = pd.date_range(start='1/2/1997', periods=8535)

countries_ame_dat = pd.read_csv(r"D:\Data\Grad\ame_total_dataset.csv")
countries_ame_dat = countries_ame_dat.set_index(countries_ame_dat.columns[0])
countries_eu_dat = pd.read_csv(r"D:\Data\Grad\eu_total_dataset.csv")
countries_eu_dat = countries_eu_dat.set_index(countries_eu_dat.columns[0])
countries_asia_dat = pd.read_csv(r"D:\Data\Grad\asia_total_dataset.csv")
countries_asia_dat = countries_asia_dat.set_index(countries_asia_dat.columns[0])

# Lag20
CrisisData_20lag = pd.read_csv(r"D:\Data\Grad\20lag_crisis_5p.csv", index_col = 0) # lag 1
print(CrisisData_20lag)

for_graph_temp = CrisisData_20lag[1502:6809]
for_graph = pd.DataFrame(for_graph_temp.sum(), columns = ['incidents'])
# Country-wise Crisis Incidents
plt.bar(for_graph.index, for_graph['incidents'])
plt.show()

# country example
CrisisData_20lag2 = pd.read_csv(r"D:\Data\Grad\20global_regional5p.csv")
us = for_graph_temp['US']

# How many countries were in crisis(5% percentile)
Cris = CrisisData_20lag.sum(axis = 1)
for_graph2 = pd.DataFrame(Cris[1502:6809])
for_graph2.to_csv(r"D:\Data\Grad\for_graph2.csv")