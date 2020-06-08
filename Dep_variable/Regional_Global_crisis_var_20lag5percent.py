import pandas as pd
import datetime
# Dataset in Use # 20 lags
Data1 = pd.read_csv(r"D:\Data\Grad\20lag_crisis_5p.csv") # Crisis Variable.

countries_ame_dat = pd.read_csv(r"D:\Data\Grad\ame_total_dataset.csv")
countries_ame_dat = countries_ame_dat.set_index(countries_ame_dat.columns[0])
countries_eu_dat = pd.read_csv(r"D:\Data\Grad\eu_total_dataset.csv")
countries_eu_dat = countries_eu_dat.set_index(countries_eu_dat.columns[0])
countries_asia_dat = pd.read_csv(r"D:\Data\Grad\asia_total_dataset.csv")
countries_asia_dat = countries_asia_dat.set_index(countries_asia_dat.columns[0])

countries_ame = list(countries_ame_dat.columns)
countries_eu = list(countries_eu_dat.columns)
countries_asia = list(countries_asia_dat.columns)

countries_ame_1p = [countries_ame[i] + '1p' for i in range(len(countries_ame))]
countries_eu_1p = [countries_eu[i] + '1p' for i in range(len(countries_eu))]
countries_asia_1p = [countries_asia[i] + '1p' for i in range(len(countries_asia))]

countries_ame_5p = [countries_ame[i] + '5p' for i in range(len(countries_ame))]
countries_eu_5p = [countries_eu[i] + '5p' for i in range(len(countries_eu))]
countries_asia_5p = [countries_asia[i] + '5p' for i in range(len(countries_asia))]

# Regional Crisis 5% 20lag
ame_crisis = [] # regional crisis variable 1
for j in range(len(Data1)):
    if sum(list(Data1.loc[j, [i for i in countries_ame]])) >= int(len(countries_ame)/2): # if more than int(len(countries_ame_1p)/2) number of countries experience 1% crisis
        ame_crisis.append(1) # it is regional crisis
    else:
        ame_crisis.append(0)
eu_crisis = [] # regional crisis variable 2
for j in range(len(Data1)):
    if sum(list(Data1.loc[j, [i for i in countries_eu]])) >= (int(len(countries_eu)/2) - 2):
        eu_crisis.append(1)
    else:
        eu_crisis.append(0)
asia_crisis = [] # regional crisis variable 3
for j in range(len(Data1)):
    if sum(list(Data1.loc[j, [i for i in countries_asia]])) >= int(len(countries_asia)/2):
        asia_crisis.append(1)
    else:
        asia_crisis.append(0)

ame_crisis_df = pd.DataFrame(ame_crisis, columns = ['ame_crisis'])
eu_crisis_df = pd.DataFrame(eu_crisis, columns = ['eu_crisis'])
asia_crisis_df = pd.DataFrame(asia_crisis, columns = ['asia_crisis'])
region_crisis = pd.concat([ame_crisis_df, eu_crisis_df, asia_crisis_df], axis = 1)

# Global Crisis 5% 20lag
glb_crisis = []
region_list = ['ame_crisis', 'eu_crisis', 'asia_crisis']
for j in range(len(region_crisis)):
    if sum(list(region_crisis.loc[j, [i for i in region_list]])) > 2:
        glb_crisis.append(1)
    else:
        glb_crisis.append(0)
glb_crisis_df = pd.DataFrame(glb_crisis, columns = ['global crisis'])
global_regional_df = pd.concat([region_crisis, glb_crisis_df], axis = 1)

# Index Date
data_indexing = pd.date_range(start='1/2/1997', periods=8535)
date = pd.DataFrame(data_indexing, columns = ['date'])
global_regional_df = pd.concat([date, global_regional_df], axis = 1)
global_regional_df = global_regional_df.set_index('date')

starting_date = '2001-08-13'
start_date = datetime.date(2001, 8, 13)
ending_date = '2020-05-15'
end_date = datetime.date(2020, 5, 15)

global_regional_df = global_regional_df[starting_date : ending_date]
global_regional_df.to_csv(r"D:\Data\Grad\20global_regional5p.csv")