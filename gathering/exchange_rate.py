from fredapi import Fred
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import datetime
# Importing Data From Federal Reserve Economic Data(FRED)
fred = Fred(api_key='4df5dd2d4ee169c0ce8ebf3b37a0ca67')

starting_date = '1997-01-02'
ending_date = '2020-05-15'
start_date = '02/01/2000'
end_date = '06/06/2020'

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

fred_tickers = ['DEXCHUS', 'DEXDNUS', 'DEXHKUS', 'DEXINUS', 'DEXJPUS', 'DEXKOUS', 'DEXMAUS', 'DEXMXUS',
                'DEXNOUS', 'DEXSIUS', 'DEXSDUS', 'DEXSZUS', 'DEXTAUS', 'DEXUSEU', 'DEXUSUK', 'DEXCAUS', 'DEXUSAL', 'DEXUSNZ']
fred_names = ['CNYUSD', 'DKKUSD', 'HKDUSD', 'INRUSD', 'JPYUSD', 'KRWUSD', 'MYRUSD', 'MXNUSD',
              'NOKUSD', 'SINUSD', 'SEKUSD', 'CHFUSD', 'TWDUSD', 'EURUSD', 'GBPUSD', 'CADUSD', 'AUDUSD', 'NZDUSD']
# n to 1 USD
# except n USD to 1GBP, EUR, AUD, NZD

exchange_rate = {}
for i in range(len(fred_tickers)):
    temp = pd.DataFrame(fred.get_series(fred_tickers[i]), columns =['exc'])
    exchange_rate[fred_names[i]] = temp

fxrate = v_generator(exchange_rate, fred_names, column_name='exc')
fx_data = fxrate[start_date : ending_date]
print(fx_data)

fx_data.to_csv(r"D:\Data\Grad\fx_data.csv")