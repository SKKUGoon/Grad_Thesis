import pandas as pd

from typing import List, Dict

import datetime
import investpy

start_date = '02/01/2000'
end_date = '06/06/2020'

tickers = [
    '105560', '030200', '033780', '003550', '066570', '034220',
    '051900', '032640', '051910', '010950', '017670', '326030',
    '034730', '096770', '000660', '035250', '010130', '000270',
    '024110', '035420', '251270', '023530', '011170', '006400',
    '028260', '207940', '032830', '018260', '009150', '005930',
    '000810', '068270', '055550', '002790', '090430', '036570',
    '316140', '139480', '035720', '021240', '005490', '086790',
    '015760', '009540', '161390', '000720', '086280', '012330',
    '004020', '005380'
]  # KOSPI 50 items

individual_stock = pd.DataFrame(None)
for stock_items in tickers:
    try:
        ind = investpy.get_stock_historical_data(stock=stock_items, country='south korea', from_date=start_date, to_date=end_date)
        ind = ind[['Close', 'Volume']]
        ind = ind.rename(columns={'Close': f'{stock_items}_close',
                                  'Volume' : f'{stock_items}_volume'})
        print(ind)
        individual_stock = pd.concat([individual_stock, ind], axis=1)
    except RuntimeError:
        print(f"{stock_items} does not exist")
        pass

individual_stock.to_csv(r'D:\Data\Grad\individual_stock_dataset.csv')