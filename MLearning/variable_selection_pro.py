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

WD = pd.read_csv(r'D:\Data\Grad\test_Work_Data.csv')
WD = WD.set_index(WD.columns[0])
WD.index = pd.to_datetime(WD.index)

X_col = list(WD.columns)
X_col.remove('global crisis') # X use all of the columns from the WD minus 'global crisis'column which is y

y = WD['global crisis']
X = WD[[i for i in X_col]]
X = X.drop(['ame_crisis', 'eu_crisis', 'asia_crisis'], axis = 1)

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

stop = timeit.default_timer()
print('Time: ', stop - start) # time my script