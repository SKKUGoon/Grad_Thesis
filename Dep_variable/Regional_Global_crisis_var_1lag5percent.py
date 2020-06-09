import pandas as pd
import datetime

t_starting_date = '2001-12-06'
t_start_date = datetime.date(2001, 12, 6)
t_ending_date = '2020-05-15'
t_end_date = datetime.date(2020, 5, 15)

# Dataset in Use # 20 lags
Data1 = pd.read_csv(r"D:\Data\Grad\1lag_crisis_5p.csv") # Crisis Variable.
Data1 = Data1.set_index(Data1.columns[0])
Data1 = Data1[t_starting_date : t_ending_date]

countries_ame_dat = pd.read_csv(r"D:\Data\Grad\ame_total_dataset.csv")
countries_ame_dat = countries_ame_dat.set_index(countries_ame_dat.columns[0])
countries_ame_dat = countries_ame_dat[t_starting_date : t_ending_date]

countries_eu_dat = pd.read_csv(r"D:\Data\Grad\eu_total_dataset.csv")
countries_eu_dat = countries_eu_dat.set_index(countries_eu_dat.columns[0])
countries_eu_dat = countries_eu_dat[t_starting_date : t_ending_date]

countries_asia_dat = pd.read_csv(r"D:\Data\Grad\asia_total_dataset.csv")
countries_asia_dat = countries_asia_dat.set_index(countries_asia_dat.columns[0])
countries_asia_dat = countries_asia_dat[t_starting_date : t_ending_date]

countries_ame = list(countries_ame_dat.columns)
countries_eu = list(countries_eu_dat.columns)
countries_asia = list(countries_asia_dat.columns)

# Regional Crisis 1%
ame_crisis = [] # regional crisis variable 1
for j in range(len(countries_ame_dat)):
    if Data1[countries_ame].sum(axis = 1)[j] >= int(len(countries_ame)/2): # if more than int(len(countries_ame_1p)/2) number of countries experience 1% crisis
        ame_crisis.append(1) # it is regional crisis
    else:
        ame_crisis.append(0)
eu_crisis = [] # regional crisis variable 2
for j in range(len(countries_eu_dat)):
    if Data1[countries_eu].sum(axis = 1)[j] >= (int(len(countries_eu)/2) - 2):
        eu_crisis.append(1)
    else:
        eu_crisis.append(0)
asia_crisis = [] # regional crisis variable 3
for j in range(len(countries_asia_dat)):
    if Data1[countries_asia].sum(axis = 1)[j] >= 4:
        asia_crisis.append(1)
    else:
        asia_crisis.append(0)

ame_crisis_df = pd.DataFrame(ame_crisis, columns = ['ame_crisis'])
ame_crisis_df.index = countries_ame_dat.index

eu_crisis_df = pd.DataFrame(eu_crisis, columns = ['eu_crisis'])
eu_crisis_df.index = countries_eu_dat.index

asia_crisis_df = pd.DataFrame(asia_crisis, columns = ['asia_crisis'])
asia_crisis_df.index = countries_asia_dat.index
region_crisis = pd.concat([ame_crisis_df, eu_crisis_df, asia_crisis_df], axis = 1)
region_crisis = region_crisis.fillna(method = 'ffill')

# Global Crisis 1% 20lag
glb_crisis = []
for j in range(len(region_crisis)):
    if region_crisis.sum(axis = 1)[j] >= 2:
        glb_crisis.append(1)
    else:
        glb_crisis.append(0)
glb_crisis_df = pd.DataFrame(glb_crisis, columns = ['global crisis'])
glb_crisis_df.index = region_crisis.index
global_regional_df = pd.concat([region_crisis, glb_crisis_df], axis = 1)

global_regional_df.to_csv(r"D:\Data\Grad\1global_regional5p.csv")
print(global_regional_df)