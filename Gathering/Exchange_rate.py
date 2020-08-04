from fredapi import Fred
from pandas_datareader import data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import datetime
# Importing Data From Federal Reserve Economic Data(FRED)
fred = Fred(api_key='4df5dd2d4ee169c0ce8ebf3b37a0ca67')

starting_date = '1997-01-02'
start_date = datetime.date(1997, 1, 2)
ending_date = '2020-05-15'
end_date = datetime.date(2020, 5, 15)


Fred_tickers = ['DEXCHUS', 'DEXDNUS', 'DEXHKUS', 'DEXINUS', 'DEXJPUS','DEXKOUS', 'DEXMAUS', 'DEXMXUS',
                'DEXNOUS', 'DEXSIUS', 'DEXSDUS', 'DEXSZUS', 'DEXTAUS', 'DEXUSEU', 'DEXUSUK', 'DEXCAUS', 'DEXUSAL', 'DEXUSNZ']
Fred_names = ['CNYUSD', 'DKKUSD', 'HKDUSD', 'INRUSD', 'JPYUSD', 'KRWUSD', 'MYRUSD', 'MXNUSD',
              'NOKUSD', 'SINUSD', 'SEKUSD', 'CHFUSD', 'TWDUSD', 'EURUSD', 'GBPUSD', 'CADUSD', 'AUDUSD', 'NZDUSD']
# n to 1 USD
# except n USD to 1GBP, EUR, AUD, NZD

exchange_rate = {}
for i in range(len(Fred_tickers)):
    temp = fred.get_series(Fred_tickers[i])
    exchange_rate[Fred_names[i]] = temp

fxrate = pd.DataFrame.from_dict(exchange_rate)
add_start_index = list(fxrate.index).index(start_date)
add_end_index = list(fxrate.index).index(end_date)
# Cut out the data we need
fx_data = fxrate[(add_start_index) : (add_end_index)]
print(fx_data)

fx_data.to_csv(r"D:\Data\Grad\fx_data.csv")
# Graph
for i in range(len(list(exchange_rate.keys()))):
    exchange_rate[list(exchange_rate.keys())[i]].plot(label = list(exchange_rate.keys())[i])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()