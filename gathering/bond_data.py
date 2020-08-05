import investpy
import pandas as pd
import copy
from typing import List, Dict

import datetime
# Importing Data From Federal Reserve Economic Data(FRED)
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

investpy_tics = copy.copy(investpy_tickers)

ame_tickers = ['U.S.', 'Mexico']
years = ['1Y', '10Y']
ame_l = list()
for ame in ame_tickers:
    for yrs in years:
        ame_full = ame + ' ' + yrs
        ame_l.append(ame_full)

for element in ame_l:
    if element in investpy_tickers:
        investpy_tics.remove(element)

# Get bond data
bonds_all = {}
for tic in investpy_tickers:
    bonds = investpy.bonds.get_bond_historical_data(bond=tic,
                                                    from_date='02/01/1997',
                                                    to_date='06/06/2020')
    bonds_all[tic] = bonds

bonds = v_generator(bonds_all, investpy_tickers)

ame_part = bonds[ame_l]
rest_part = bonds[investpy_tics]
ame_part = ame_part.shift(1) # at the date of Korean market closure, ame_ bond market haven't closed yet
bonds_final = pd.concat([ame_part, rest_part], axis = 1)

bonds_final.to_csv(r"D:\Data\Grad\bonds_dataset.csv")

