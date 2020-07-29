import investpy
import pandas as pd
# Insufficient Data
KOSPIVIX = investpy.get_index_historical_data(index='KOSPI Volatility',
                                              country='south korea',
                                              from_date='02/01/1997',
                                              to_date='06/06/2020')
kvix = list(KOSPIVIX['Close'])
KOSPIVIX_ = pd.DataFrame(kvix, columns=['KOSPI_VIX'])
KOSPIVIX_.index = KOSPIVIX.index
KOSPIVIX_.index = pd.to_datetime(KOSPIVIX_.index)
KOSPIVIX_.to_csv(r'D:\Data\Grad\Kospivix.csv')


# Get Trade Volume
Kor = investpy.get_index_historical_data(index='KOSPI',
                                         country='south korea',
                                         from_date='02/01/1997',
                                         to_date='06/06/2020')
Kospi_volume = list(Kor['Volume'])
K_volume = pd.DataFrame(list(Kor['Volume']), columns=['K_trade_volume'])
K_volume.index = Kor.index
K_volume.index = pd.to_datetime(K_volume.index)
K_volume.to_csv(r'D:\Data\Grad\Kospi_volume.csv')
