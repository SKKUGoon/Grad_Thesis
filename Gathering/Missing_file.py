"""peru 2000/07/03 ~, chile, indonesia, korea, thailand, pakistan, New Zealand,
Malaysia 2001/08/13 ~, England 2001/01/03 ~, russia 1997/01/05 ~, sweden 1997/01/02 ~,
norway, ireland 2001/08/13 ~, poland, greece 1997/11/18 ~, czech, hungary 2001/08/13 ~"""

import investpy

start_date = '02/01/2000'
end_date = '06/06/2020'

# Missing files append
count_list_ = ['peru', 'chile', 'indone', 'kor',
               'tha', 'pak', 'nz', 'mal',
               'uk', 'russia', 'sweden', 'norway',
               'ireland', 'pol', 'grc', 'czh', 'hun']

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
