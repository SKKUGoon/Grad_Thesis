import investpy
import pandas as pd

import datetime
# Importing Data From Federal Reserve Economic Data(FRED)

# Country Name
country = ['U.S.', 'Japan', 'Germany', 'U.K.', 'France', 'India', 'South Korea', 'Russia', 'Spain', 'Mexico',
           'Indonesia', 'Netherlands', 'Switzerland', 'Taiwan', 'Poland', 'Belgium', 'Thailand', 'Austria',
           'Norway', 'Hong Kong', 'Israel', 'Philippines', 'Malaysia', 'Ireland', 'Greece',
           'Czech Republic', 'Hungary']
years = ['1Y', '10Y']
investpy_tickers = list()
for count in country:
    for yrs in years:
        full_name = count + ' ' + yrs
        investpy_tickers.append(full_name) # Create tickers. Save in investpy_tickers list

# Remove what I don't have
investpy_tickers.remove('Netherlands 1Y')
investpy_tickers.remove('Taiwan 1Y')
investpy_tickers.remove('Greece 1Y')


bonds_all = []
for i in range(len(investpy_tickers)):
    bonds = investpy.bonds.get_bond_historical_data(bond = investpy_tickers[0],
                                                    from_date = '02/01/1997',
                                                    to_date = '06/06/2020')
    bonds_all.append(bonds)

bonds_close = []

# Gather Closing price(Daily)
for i in range(len(bonds_all)):
    close = pd.DataFrame(bonds_all[i]['Close'])
    close = close.rename(columns = {'Close' : investpy_tickers[i]})
    bonds_close.append(close)

bonds_dataset = bonds_close[0]
for i in range(len(investpy_tickers) - 1):
    bonds_dataset = pd.concat([bonds_dataset, bonds_close[i+1]], axis = 1)

bonds_dataset.to_csv(r"D:\Data\Grad\bonds_dataset.csv")

