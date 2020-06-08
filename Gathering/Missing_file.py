import investpy
########################### AME ###############################
# peru 2000/07/03 ~
peru = investpy.get_index_historical_data(index = 'FTSE Peru',
                                       country = 'peru',
                                       from_date = '02/01/2000',
                                       to_date = '06/06/2020')
# chile
chile = investpy.get_index_historical_data(index = 'S&P CLX IPSA',
                                       country = 'chile',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
########################### ASIA/PACIFIC ###############################
# indonesia
inda = investpy.get_index_historical_data(index = 'IDX Composite',
                                       country = 'indonesia',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# korea
kor = investpy.get_index_historical_data(index = 'KOSPI',
                                       country = 'south korea',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# thailand
tha = investpy.get_index_historical_data(index = 'SET',
                                       country = 'thailand',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# pakistan
pak = investpy.get_index_historical_data(index = 'Karachi 100',
                                       country = 'pakistan',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# New Zealand
nz = investpy.get_index_historical_data(index = 'NZX All',
                                       country = 'new zealand',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# Malaysia 2001/08/13 ~
mal = investpy.get_index_historical_data(index = '	FTSE Malaysia',
                                       country = 'malaysia',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
########################### EUROPE ###############################
# England 2001/01/03 ~
UK = investpy.get_index_historical_data(index = 'FTSE 100',
                                       country = 'united kingdom',
                                       from_date = '02/01/1999',
                                       to_date = '06/06/2020')
# russia 1997/01/05 ~
russia = investpy.get_index_historical_data(index = 'RTSI',
                                       country = 'russia',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# sweden 1997/01/02 ~
sweden = investpy.get_index_historical_data(index = 'OMXS30',
                                       country = 'sweden',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# norway
norway = investpy.get_index_historical_data(index = 'Oslo OBX',
                                       country = 'norway',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# ireland 2001/08/13 ~
ireland = investpy.get_index_historical_data(index = 'FTSE Ireland',
                                       country = 'ireland',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# poland
pol = investpy.get_index_historical_data(index = 'WIG20',
                                       country = 'poland',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# greece 1997/11/18 ~
grc = investpy.get_index_historical_data(index = 'FTSE/Athex 20',
                                       country = 'greece',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# czech
czh = investpy.get_index_historical_data(index = 'FTSE czech republic',
                                       country = 'czech republic',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')
# hungary 2001/08/13 ~
hun = investpy.get_index_historical_data(index = 'FTSE Hungary',
                                       country = 'hungary',
                                       from_date = '02/01/1997',
                                       to_date = '06/06/2020')