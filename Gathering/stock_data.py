from fredapi import Fred
from pandas_datareader import data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import datetime
import investpy
# Importing Data From Federal Reserve Economic Data(FRED)
fred = Fred(api_key='4df5dd2d4ee169c0ce8ebf3b37a0ca67')

starting_date = '1997-01-02'
start_date = datetime.date(1997, 1, 2)
ending_date = '2020-05-15'
end_date = datetime.date(2020, 5, 15)

# NORTH AND SOUTH AMERICAN REGION STOCK INDEX
# United States : NASDAQ_Composite - FRED
# The observations for the NASDAQ Composite Index represent the daily index value at "market close".
United_States = fred.get_series('NASDAQCOM')
United_States = pd.DataFrame(United_States, columns = ['US'])
stock_start_index = list(United_States.index).index(start_date)
stock_end_index = list(United_States.index).index(end_date)
United_States = United_States[(stock_start_index) : (stock_end_index)]

# From Yahoo
tickers_ame = ['GSPTSE', 'MERV']
countries_ame = ['Canada', 'Argentina']
# Canada : S&P/TSX Composite Index - PD.Datareader
Canada = data.DataReader('^GSPTSE', 'yahoo', starting_date, ending_date)
Canada = pd.DataFrame(Canada['Close'])
Canada = Canada.rename(columns = {"Close" : "CA"})
# Argentina : MERVAL (^MERV)
Argentina = data.DataReader('^MERV', 'yahoo', starting_date, ending_date)
Argentina = pd.DataFrame(Argentina['Close'])
Argentina = Argentina.rename(columns = {"Close" : "ARG"})
# Brazil : IBOVESPA
Brazil = data.DataReader('^BVSP', 'yahoo', starting_date, ending_date)
Brazil = pd.DataFrame(Brazil['Close'])
Brazil = Brazil.rename(columns = {"Close" : "BRA"})
# Mexico : IPC MEXICO
Mexico = data.DataReader('^MXX', 'yahoo', starting_date, ending_date)
Mexico = pd.DataFrame(Mexico['Close'])
Mexico = Mexico.rename(columns = {"Close" : "MEX"})

NS_America_Region = pd.concat([United_States, Canada, Argentina, Brazil, Mexico], axis = 1) # Chile was deleted due to Data Shortage. NaN Exists
NS_Ame_region = NS_America_Region

# EUROPEAN STOCK INDEX
European_dataset = {}
countries_eu = ['GER', 'FR',
                'SPA', 'NET', 'SWZ',
                'BEL', 'AUS', 'DEN']
tickers_eu = ['^GDAXI', '^FCHI',
              '^IBEX', '^AEX', '^SSMI',
              '^BFX', '^ATX', '^OMXC20',]
for country in range(len(countries_eu)):
    temp = data.DataReader(tickers_eu[country], 'yahoo', starting_date, ending_date)
    temp = temp['Close'].rename(columns = {"Close" : countries_eu[country]})
    European_dataset[countries_eu[country]] = temp
European_Region = pd.DataFrame.from_dict(European_dataset)
EU_region = European_Region

# ASIAN STOCK INDEX
Asian_dataset = {}
countries_asia = ['CHI', 'JAP', 'INDA',
                  'TAI', 'HK', 'PHIL', 'AUSTL']
tickers_asia = ['000001.SS', '^N225', '^BSESN',
                '^TWII', '^HSI', 'PSEI.PS', '^AXJO']
for country in range(len(countries_asia)):
    temp = data.DataReader(tickers_asia[country], 'yahoo', starting_date, ending_date)
    temp = temp['Close'].rename(columns = {"Close" : countries_asia[country]})
    Asian_dataset[countries_asia[country]] = temp
Asian_Region = pd.DataFrame.from_dict(Asian_dataset)
Asia_region = Asian_Region

########################### AME ###############################
# peru 2000/07/03 ~
peru = investpy.get_index_historical_data(index = 'FTSE Peru',
                                       country = 'peru',
                                       from_date = '02/01/2000',
                                       to_date = '06/06/2020')
# chile
chile = investpy.get_index_historical_data(index = 'S&P CLX IPSA',
                                       country = 'chile',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
ame_add = [peru, chile]
ame_add_str = ['peru', 'chile']
ame_add_temp = {}
for i in range(len(ame_add)):
    index = list(ame_add[i].index)
    ame_add[i] = list(ame_add[i]['Close'])
    temp_dict = {'date' : index, ame_add_str[i] : ame_add[i]}
    temp_df = pd.DataFrame.from_dict(temp_dict)
    temp_df = temp_df.set_index('date')
    ame_add_temp[ame_add_str[i]] = temp_df
ame_additional = ame_add_temp[list(ame_add_temp.keys())[0]]
for i in range(len(ame_add_temp.keys()) - 1):
    ame_additional = pd.concat([ame_additional,
                                ame_add_temp[list(ame_add_temp.keys())[i + 1]]], axis = 1)

########################### ASIA/PACIFIC ###############################
# indonesia
inda = investpy.get_index_historical_data(index = 'IDX Composite',
                                       country = 'indonesia',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# korea
kor = investpy.get_index_historical_data(index = 'KOSPI',
                                       country = 'south korea',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# thailand
tha = investpy.get_index_historical_data(index = 'SET',
                                       country = 'thailand',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# pakistan
pak = investpy.get_index_historical_data(index = 'Karachi 100',
                                       country = 'pakistan',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# New Zealand
nz = investpy.get_index_historical_data(index = 'NZX All',
                                       country = 'new zealand',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# Malaysia 2001/08/13 ~
mal = investpy.get_index_historical_data(index = '	FTSE Malaysia',
                                       country = 'malaysia',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')

asia_add = [inda, kor, tha, pak, nz, mal]
asia_add_str = ['inda', 'kor', 'tha', 'pak', 'nz', 'mal']
asia_add_temp = {}
for i in range(len(asia_add)):
    index = list(asia_add[i].index)
    asia_add[i] = list(asia_add[i]['Close'])
    temp_dict = {'date' : index, asia_add_str[i] : asia_add[i]}
    temp_df = pd.DataFrame.from_dict(temp_dict)
    temp_df = temp_df.set_index('date')
    asia_add_temp[asia_add_str[i]] = temp_df
asia_additional = asia_add_temp[list(asia_add_temp.keys())[0]]
for i in range(len(asia_add_temp.keys()) - 1):
    asia_additional = pd.concat([asia_additional,
                                asia_add_temp[list(asia_add_temp.keys())[i + 1]]], axis = 1)

########################### EUROPE ###############################
# England 2001/01/03 ~
uk = investpy.get_index_historical_data(index = 'FTSE 100',
                                       country = 'united kingdom',
                                       from_date = '02/01/1999',
                                       to_date = '06/06/2020')
# russia 1997/01/05 ~
russia = investpy.get_index_historical_data(index = 'RTSI',
                                       country = 'russia',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# sweden 1997/01/02 ~
sweden = investpy.get_index_historical_data(index = 'OMXS30',
                                       country = 'sweden',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# norway
norway = investpy.get_index_historical_data(index = 'Oslo OBX',
                                       country = 'norway',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# ireland 2001/08/13 ~
ireland = investpy.get_index_historical_data(index = 'FTSE Ireland',
                                       country = 'ireland',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# poland
pol = investpy.get_index_historical_data(index = 'WIG20',
                                       country = 'poland',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# greece 1997/11/18 ~
grc = investpy.get_index_historical_data(index = 'FTSE/Athex 20',
                                       country = 'greece',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# czech
czh = investpy.get_index_historical_data(index = 'FTSE czech republic',
                                       country = 'czech republic',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# hungary 2001/08/13 ~
hun = investpy.get_index_historical_data(index = 'FTSE Hungary',
                                       country = 'hungary',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
eu_add = [uk, russia, sweden, norway, ireland, pol, grc, czh, hun]
eu_add_str = ['uk', 'russia', 'sweden', 'norway', 'ireland', 'pol', 'grc', 'czh', 'hun']
eu_add_temp = {}
for i in range(len(eu_add)):
    index = list(eu_add[i].index)
    eu_add[i] = list(eu_add[i]['Close'])
    temp_dict = {'date' : index, eu_add_str[i] : eu_add[i]}
    temp_df = pd.DataFrame.from_dict(temp_dict)
    temp_df = temp_df.set_index('date')
    eu_add_temp[eu_add_str[i]] = temp_df
eu_additional = eu_add_temp[list(eu_add_temp.keys())[0]]
for i in range(len(eu_add_temp.keys()) - 1):
    eu_additional = pd.concat([eu_additional,
                                eu_add_temp[list(eu_add_temp.keys())[i + 1]]], axis = 1)

ame_total = pd.concat([NS_Ame_region, ame_additional], axis = 1)
ame_total.to_csv(r"D:\Data\Grad\ame_total_dataset.csv")
eu_total = pd.concat([EU_region, eu_additional], axis = 1)
eu_total.to_csv(r"D:\Data\Grad\eu_total_dataset.csv")
asia_total = pd.concat([Asia_region, asia_additional], axis = 1)
asia_total.to_csv(r"D:\Data\Grad\asia_total_dataset.csv")

total_stock_index_dataset = pd.concat([NS_Ame_region, ame_additional,
                                       EU_region, eu_additional,
                                       Asia_region, asia_additional], axis = 1)
total_stock_index_dataset = total_stock_index_dataset.interpolate(method = 'time')
total_stock_index_dataset.to_csv(r"D:\Data\Grad\total_stock_index_dataset.csv")
print(total_stock_index_dataset)