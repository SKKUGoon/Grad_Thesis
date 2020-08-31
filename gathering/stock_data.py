from fredapi import Fred
from pandas_datareader import data
import pandas as pd

from typing import List, Dict

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

start_date = '02/01/2000'
end_date = '06/06/2020'
United_States = United_States[start_date : end_date]
# Get rid of yahoo. Integrate into investing.com api sources.

def v_generator(total_file: Dict, region: List, column_name='Close') -> pd.DataFrame:
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
    for val in saving_spot.values():
        base = pd.concat([base, val], axis=1)

    # Index to datetime
    base.index = pd.to_datetime(base.index)

    return base

# count_list_ count_tickers country: order of these list should match for api request.
ame_add_str = ['canada', 'argentina', 'brazil', 'mexico', 'peru', 'chile']
asia_add_str = ['china', 'japan', 'india', 'taiwan', 'hong kong',
                'phil', 'austl', 'indone', 'kor', 'tha',
                'pak', 'nz', 'mal']
eu_add_str = ['ger', 'fra', 'spa', 'net', 'swz',
              'bel', 'ausr', 'den', 'uk', 'russia',
              'sweden', 'norway', 'ireland', 'pol', 'grc',
              'czh', 'hun']
count_list_ = ame_add_str + asia_add_str + eu_add_str

count_tickers_ame = ['S&P/TSX', 'S&P Merval', 'Bovespa', 'S&P/BMV IPC', 'FTSE Peru', 'S&P CLX IPSA']
count_tickers_asia = ['Shanghai', 'Nikkei 225', 'BSE Sensex', 'Taiwan Weighted', 'Hang Seng',
                      'PSEi Composite', 'S&P/ASX 200', 'IDX Composite', 'KOSPI', 'SET',
                      'Karachi 100', 'NZX All', 'FTSE Malaysia']
count_tickers_eu = ['DAX', 'CAC 40', 'IBEX 35', 'AEX', 'SMI',
                    'BEL 20', 'ATX', 'OMXC20', 'FTSE 100', 'RTSI',
                    'OMXS30', 'Oslo OBX', 'FTSE Ireland', 'WIG20', 'FTSE/Athex 20',
                    'FTSE czech republic','FTSE Hungary']
count_tickers = count_tickers_ame + count_tickers_asia + count_tickers_eu

country_ame = ['canada', 'argentina', 'brazil', 'mexico', 'peru', 'chile']
country_asia = ['china', 'japan', 'india', 'taiwan', 'hong kong',
                'philippines', 'australia', 'indonesia', 'south korea','thailand',
                'pakistan', 'new zealand', 'malaysia']
country_eu = ['germany', 'france', 'spain', 'netherlands', 'switzerland',
              'belgium', 'austria', 'denmark', 'united kingdom', 'russia',
              'sweden', 'norway', 'ireland', 'poland', 'greece',
              'czech republic', 'hungary']
country = country_ame + country_asia + country_eu
missing = {}
for i in range(len(count_list_)):
    missing[count_list_[i]] = investpy.get_index_historical_data(index=count_tickers[i],
                                                                 country=country[i],
                                                                 from_date=start_date,
                                                                 to_date=end_date)

ame_additional = v_generator(missing, ame_add_str)
eu_additional = v_generator(missing, eu_add_str)
asia_additional = v_generator(missing, asia_add_str)

# Delete columns that has too little data: austria, india, taiwan, philippinese
col_ = list(eu_additional.columns)
col_.remove('ausr')
eu_additional = eu_additional[col_]

col_2 = list(asia_additional.columns)
rv = ['india', 'taiwan', 'phil']
for i in rv:
    col_2.remove(i)
asia_additional = asia_additional[col_2]

# India, taiwan, philippines and Austria needed to be scrapped from yahoo.
countries_asia = ['india', 'taiwan', 'phil']
tickers_asia = ['^BSESN', '^TWII', 'PSEI.PS']

countries_eu = ['ausr']
tickers_eu = ['^ATX']

countries = countries_asia + countries_eu
tickers = tickers_asia + tickers_eu

dataset = {}
for cntry in range(len(countries)):
    temp = data.DataReader(tickers[cntry], 'yahoo', starting_date, ending_date)
    temp = temp['Close'].rename(columns={'Close' : countries[cntry]})
    dataset[countries[cntry]] = temp

more = pd.DataFrame.from_dict(dataset)[start_date : end_date]
asia_more = more[countries_asia]
eu_more = more[countries_eu]

# starting_date = '1997-01-02'
# ending_date = '2020-05-15'
# start_date = '02/01/2000'
# end_date = '06/06/2020'

alt_start_date = '31/01/2000'
ame_total = pd.concat([United_States, ame_additional], axis = 1)[alt_start_date:]
ame_total = ame_total.shift(1) # Since at the time of asia market closing time,
                               # we don't have the same date's closure information of America
ame_total.to_csv(r'D:\Data\Grad\ame_total_dataset.csv')

eu_total = pd.concat([eu_more, eu_additional], axis = 1)[start_date:]
eu_total = eu_total.shift(1) # Since at the time of asia market closing time
                             # EU market haven't closed yet
eu_total.to_csv(r'D:\Data\Grad\eu_total_dataset.csv')

asia_total = pd.concat([asia_more, asia_additional], axis = 1)[start_date:]

# Asia lag - except (austl nz kor japan)
as_total = pd.DataFrame(None)

behind_kor = ['india', 'taiwan', 'phil', 'china', 'hong kong', 'indone', 'tha', 'pak', 'mal'] # UTC + 9< countries
for cntry in behind_kor:
    asia_ = pd.DataFrame(asia_total[cntry]).shift(1)
    as_total = pd.concat([as_total, asia_], axis=1)

front_kor = ['japan', 'kor', 'nz', 'austl'] # UTC + 9> countries
for cntry in front_kor:
    asia_ = pd.DataFrame(asia_total[cntry])
    as_total = pd.concat([as_total, asia_], axis=1)

print(as_total)
as_total.to_csv(r'D:\Data\Grad\asia_total_dataset.csv')


total_stock_index_dataset = pd.concat([ame_total, eu_total, asia_total], axis = 1)
total_stock_index_dataset.to_csv(r'D:\Data\Grad\total_stock_index_dataset.csv')
