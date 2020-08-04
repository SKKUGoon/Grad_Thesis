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

Fred_tickers = ['USD3MTD156N', 'VIXCLS', 'GOLDAMGBD228NLBM', 'TEDRATE', 'DCOILWTICO', 'DCOILBRENTEU', 'DFF', 'BAMLHYH0A3CMTRIV']
Fred_names = ['LIBOR', 'VIX', 'Gold', 'TED spread', 'Oil_WTI', 'Oil_Brent', 'Effective Fed', 'High_return_bond']

additional_data = {}
for i in range(len(Fred_tickers)):
    temp = fred.get_series(Fred_tickers[i])
    additional_data[Fred_names[i]] = temp

additional = pd.DataFrame.from_dict(additional_data)
add_start_index = list(additional.index).index(start_date)
add_end_index = list(additional.index).index(end_date)
# Cut out the data we need
add_data = additional[(add_start_index) : (add_end_index)]
print(add_data)

add_data.to_csv(r"D:\Data\Grad\add_data.csv")
# Graph
for i in range(len(list(additional_data.keys()))):
    additional_data[list(additional_data.keys())[i]].plot(label=list(additional_data.keys())[i])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()