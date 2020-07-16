import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
class preprocessing:
    def __init__(self, data):
        """data should be python pandas dataframe"""
        self.data = data
        self.df_columns = data.columns # List of columns name


    def empirical_dist(self, lag_distance, column_number):
        self.lag = lag_distance # exogenous
        self.col_num = column_number # exogenous
        self.play = self.data[list(self.df_columns)[self.col_num]] # defines the dataset's column you want to process
        self.yield_dist = self.play - self.play.shift(self.lag) # Lagging process
        print('Deprecated. When you calculate yield_dist, Use crisis_code_200 instead')
        return self.yield_dist


    def log_return(self, lag_distance, column_number):
        self.lag = lag_distance
        self.col_num = column_number
        self.play = self.data[list(self.df_columns)[self.col_num]]

        self.logreturn = np.log(self.play) - np.log(self.play.shift(self.lag))
        self.result = pd.DataFrame(self.logreturn, columns = [self.df_columns[self.col_num]])
        return self.result


    def log_return_class(self, column_number, simplicity = '2 class'):
        self.sim = simplicity
        self.col_num = column_number
        self.index = self.data.index
        self.play = self.data[list(self.df_columns)[self.col_num]]
        if self.sim == '2 class':
            self.playlist = list(self.play)
            self.resultlist = list()
            for i in range(len(self.playlist)):
                if self.playlist[i] < 0:
                    self.resultlist.append(-1) # return -1 if negative
                elif self.playlist[i] >= 0:
                    self.resultlist.append(1)
                else:
                    self.resultlist.append(np.nan)
        elif self.sim == 'n class': # Make more differentiated class
            ...
        else: # Make another classification standards
            ...
        # As DataFrame
        self.lrc = pd.DataFrame(data = self.resultlist, index = list(self.index), columns = [self.df_columns[self.col_num] + ' class'])
        return self.lrc

    def crisis_code_200(self, column_number, lag_distance, quantile_exo):
        self.col_num = column_number # Data's column
        self.lag = lag_distance
        self.q = quantile_exo
        self.using = self.data[list(self.df_columns)[self.col_num]]
        self.crisis_code = [np.nan]
        for length in range(len(self.using)):
            self.emp_dist = self.using[0:length] - self.using[0:length].shift(self.lag)
            self.qntl = self.emp_dist.quantile(self.q) # emp_dist and qntl are decided at each iteration
            if length < 200: # pass if the length of using data is under 200
                pass
            elif length == 200: # if the using data's length is 200. create crisis_code for the first 200 observation
                for i in range(len(self.using[0:length])):
                    if self.emp_dist[0:length][i] < self.qntl:
                        self.crisis_code.append(1) # append 1 if it's less than designated quantile.
                    else:
                        self.crisis_code.append(0)
            else:
                if self.emp_dist[0:length][-1] < self.qntl:
                    self.crisis_code.append(1)
                else:
                    self.crisis_code.append(0)

        # Return as DataFrame
        self.crisis_code = pd.DataFrame(self.crisis_code, columns = [self.df_columns[self.col_num]])
        self.crisis_code.index = self.data.index
        return self.crisis_code


    def empirical_dist_graph(self, data):
        return None


    def crisis_coding(self, quantile, column_number):
        self.col_num = column_number # exogenous
        self.workspace = self.data[list(self.data.columns)[self.col_num]] # define workspace as particular column of the data
        self.q = quantile # 0.01 ~ 1.00, exogenous
        self.quant = self.workspace.quantile(self.q) # Calculate the qth percentile of data
        self.crisis_coded = [] # null list where you append 1 if <percentile , 0 if > percentile
        for i in range(len(self.workspace)):
            if self.workspace[i] < self.quant:
                self.crisis_coded.append(1)
            else:
                self.crisis_coded.append(0)
        print('Deprecated. Use crisis_code_200 instead') # No longer useful
        return pd.DataFrame(self.crisis_coded) # Report as DataFrame

