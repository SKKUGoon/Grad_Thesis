from fredapi import Fred
from pandas_datareader import data
from typing import List, Dict

import pandas as pd

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
# The observations for the NASDAQ Composite Index represent the daily index value at 'market close'.
United_States = fred.get_series('NASDAQCOM')
United_States = pd.DataFrame(United_States, columns=['US'])
stock_start_index = list(United_States.index).index(start_date)
stock_end_index = list(United_States.index).index(end_date)
United_States = United_States[(stock_start_index) : (stock_end_index)]

# From Yahoo
America_dataset = {}
tickers_ame = ['^GSPTSE', '^MERV', '^BVSP', '^MXX']
countries_ame = ['CAN', 'ARG', 'BRA', 'MEX']

for country in range(len(countries_ame)):
    temp = data.DataReader(tickers_ame[country], 'yahoo', starting_date, ending_date)
    temp = temp['Close'].rename(columns={'Close' : countries_ame[country]})
    America_dataset[countries_ame[country]] = temp
Ame_region = pd.DataFrame.from_dict(America_dataset)

NS_Ame_region = pd.concat([United_States, Ame_region], axis=1) # Chile was deleted due to Data Shortage. NaN Exists


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
    temp = temp['Close'].rename(columns={'Close' : countries_eu[country]})
    European_dataset[countries_eu[country]] = temp
EU_region = pd.DataFrame.from_dict(European_dataset)


# ASIAN STOCK INDEX
Asian_dataset = {}
countries_asia = ['CHI', 'JAP', 'INDA',
                  'TAI', 'HK', 'PHIL', 'AUSTL']
tickers_asia = ['000001.SS', '^N225', '^BSESN',
                '^TWII', '^HSI', 'PSEI.PS', '^AXJO']
for country in range(len(countries_asia)):
    temp = data.DataReader(tickers_asia[country], 'yahoo', starting_date, ending_date)
    temp = temp['Close'].rename(columns={'Close' : countries_asia[country]})
    Asian_dataset[countries_asia[country]] = temp
Asian_Region = pd.DataFrame.from_dict(Asian_dataset)
Asia_region = Asian_Region

"""peru 2000/07/03 ~, chile, indonesia, korea, thailand, pakistan, New Zealand,
Malaysia 2001/08/13 ~, England 2001/01/03 ~, russia 1997/01/05 ~, sweden 1997/01/02 ~,
norway, ireland 2001/08/13 ~, poland, greece 1997/11/18 ~, czech, hungary 2001/08/13 ~"""

start_date = '02/01/2000'
end_date = '06/06/2020'

# Missing files append
# count_list_ count_tickers country: order of these list should match for api request.

def MVGenerater(total_file: Dict, region: List, column_name='Close') -> pd.DataFrame:
    """
    :param total_file: Total missing file collected from investpy
    :param region: ame asia or eu
    :param column_name: mostly closing price
    :return: base is pandas DataFrame of missing files. from 2000-01-02 to 2020-06-05
    """
    saving_spot = dict()
    # Save specific column into dictionary
    for c in region:
        saving_spot[c] = pd.DataFrame(total_file[c][column_name]).rename(columns={column_name : c})

    # Create empty Dataframe with date index
    start = datetime.datetime.strptime("02-01-2000", "%d-%m-%Y")
    end = datetime.datetime.strptime("06-06-2020", "%d-%m-%Y")
    date_gen = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]
    date_dat = list()
    for date in date_gen:
        date_dat.append(date.strftime("%Y-%m-%d"))
    base = pd.DataFrame(date_dat, columns=['Date'])
    base = base.set_index('Date')
    base.index = pd.to_datetime(base.index)

    # Dictionary to pd.Dataframe
    for i in range(len(saving_spot.keys())):
        base = pd.concat([base, saving_spot[list(saving_spot.keys())[i]]], axis=1)

    # Index to datetime
    base.index = pd.to_datetime(base.index)

    return base

ame_add_str = ['peru', 'chile']
asia_add_str = ['indone', 'kor', 'tha', 'pak', 'nz', 'mal']
eu_add_str = ['uk', 'russia', 'sweden', 'norway', 'ireland', 'pol', 'grc', 'czh', 'hun']
count_list_ = ame_add_str + asia_add_str + eu_add_str

count_tickers = ['FTSE Peru', 'S&P CLX IPSA', 'IDX Composite', 'KOSPI',
                 'SET', 'Karachi 100', 'NZX All', 'FTSE Malaysia',
                 'FTSE 100', 'RTSI', 'OMXS30', 'Oslo OBX',
                 'FTSE Ireland', 'WIG20', 'FTSE/Athex 20', 'FTSE czech republic','FTSE Hungary']

country = ['peru', 'chile', 'indonesia', 'south korea',
           'thailand', 'pakistan', 'new zealand', 'malaysia',
           'united kingdom', 'russia', 'sweden', 'norway',
           'ireland', 'poland', 'greece', 'czech republic', 'hungary']

missing = {}
for i in range(len(count_list_)):
    missing[count_list_[i]] = investpy.get_index_historical_data(index=count_tickers[i],
                                                                 country=country[i],
                                                                 from_date=start_date,
                                                                 to_date=end_date)


ame_additional = MVGenerater(missing, ame_add_str)
eu_additional = MVGenerater(missing, eu_add_str)
asia_additional = MVGenerater(missing, asia_add_str)

ame_total = pd.concat([NS_Ame_region, ame_additional], axis = 1)
ame_total.to_csv(r'D:\Data\Grad\ame_total_dataset.csv')
eu_total = pd.concat([EU_region, eu_additional], axis = 1)
eu_total.to_csv(r'D:\Data\Grad\eu_total_dataset.csv')
asia_total = pd.concat([Asia_region, asia_additional], axis = 1)
asia_total.to_csv(r'D:\Data\Grad\asia_total_dataset.csv')

total_stock_index_dataset = pd.concat([NS_Ame_region, ame_additional,
                                       EU_region, eu_additional,
                                       Asia_region, asia_additional], axis = 1)
total_stock_index_dataset = total_stock_index_dataset.interpolate(method = 'time')
total_stock_index_dataset.to_csv(r'D:\Data\Grad\total_stock_index_dataset.csv')
print(total_stock_index_dataset)