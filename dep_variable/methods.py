import pandas as pd
import numpy as np

class Preprocessing:
    def __init__(self, data):
        """data should be python pandas dataframe"""
        self.data = data
        self.df_columns = data.columns # List of columns name
        self.index = data.index

    def log_return(self, lag_distance: int, column_number: int) -> pd.DataFrame:
        """
        :param lag_distance: how much time I should go back.
        :param column_number: start from 0(just like python list indexing scheme)
        :return: DataFrame
        log return(t) = log(Price_t) - log(Price_(t-1)). Loose first data in the process
        """
        lag = lag_distance
        col_num = column_number
        target = self.data[list(self.df_columns)[col_num]] # select particular column

        logreturn = np.log(target) - np.log(target.shift(lag))
        res = pd.DataFrame(logreturn, columns=[self.df_columns[col_num]])
        return res


    def log_return_class(self, lag_distance: int, column_number: int, simplicity='-1/1') -> pd.DataFrame:
        """
        :param lag_distance: how much time I should go back.
        :param column_number: start from 0
        :param simplicity: '-1/1' '0/1' 'n class'
        :return: DataFrame
        log return class(t) = sign(log(Price_t) - log(Price_(t-1)). Lose first data in the process)
        """
        sim = simplicity
        col_num = column_number
        lag = lag_distance
        play = self.log_return(lag, col_num)[self.df_columns[0]]

        if sim == '-1/1':
            resultlist = list()
            for element in play:
                if element < 0:
                    resultlist.append(-1) # return -1 if negative
                elif element >= 0:
                    resultlist.append(1)
                else:
                    resultlist.append(np.nan)

        elif sim == '0/1':
            resultlist = list()
            for element in play:
                if element < 0:
                    resultlist.append(0) # return 0 if negative
                elif element >= 0:
                    resultlist.append(1)
                else:
                    resultlist.append(np.nan)
        elif sim == 'n class': # Make more differentiated class
            ...
        else: # Make another classification standards
            ...
        # As DataFrame
        lrc = pd.DataFrame(data=resultlist,
                           index=list(self.index),
                           columns=[self.df_columns[col_num]+'clf'+str(lag)])
        return lrc