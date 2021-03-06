from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np
import timeit
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

start = timeit.default_timer()
WD = pd.read_csv(r'D:\Data\Grad\test_Work_Data_w_lag.csv')
WD = WD.set_index(WD.columns[0])
WD.index = pd.to_datetime(WD.index)

WD1 = WD[WD.columns[0:4]].fillna(method = 'ffill')
WD2 = WD[WD.columns[4:1078]].interpolate(method = 'time')
WD = pd.concat([WD1, WD2], axis = 1)

X_col = list(WD.columns)
X_col.remove('kor class') # X use all of the columns from the WD minus 'global crisis'column which is y

y = WD['kor class']
X = WD[[i for i in X_col]]

# Create trainig set, testing set.
X_train, X_test,y_train, y_test = train_test_split(X, y,
                                                   test_size = 0.3,
                                                   shuffle = False) # Split the data. For now no validation set
# Scale the Data
scale = StandardScaler() # Scale the data.
scale.fit(X_train)

# Variable Selection using LASSO
selection = SelectFromModel(LogisticRegression(C=0.1, penalty='l1')) #penalty is L1, which is LASSO
selection.fit(scale.transform(X_train), y_train)
selected_var = list(selection.get_support())
selected_var = X_train.columns[selected_var]
print(selected_var)
# Should include Borute algorithm - More on that later
# https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a
X_selected = list(selected_var)
X_filtered_lasso = WD[[i for i in X_selected]]
X_filtered_lasso.to_csv(r"D:\Data\Grad\X_filitered_lasso.csv")
stop = timeit.default_timer()
print('Time: ', stop - start) # time my script