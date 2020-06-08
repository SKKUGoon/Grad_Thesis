import pandas as pd
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
        return self.yield_dist

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
        return pd.DataFrame(self.crisis_coded) # Report as DataFrame
