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
ending_date = '2020-05-20'
end_date = datetime.date(2020, 5, 20)

# NO BOND DATA!
bond_name = ['USA_B', 'JAPAN_B']
tickers_bond = ['^TNX', ]