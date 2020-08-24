import numpy as np
import pandas as pd
import timeit

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

import matplotlib.pyplot as plt

from mlearning.scoringMethods import scoring_model

import warnings
warnings.filterwarnings(action = "ignore", category = FutureWarning)

# Data_using
start = timeit.default_timer()

wd = pd.read_csv(r'D:\Data\Grad\test_Work_Data_w_lag.csv')
wd = wd.set_index(wd.columns[0])
wd.index = pd.to_datetime(wd.index)

x = wd[wd.columns[1:]]
y = wd['korclf1']


# Create trainig set, testing set.
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.3,
                                                    shuffle=False) # Split the data. For now no validation set
                                                                    # no random split. so shuffle = false
# Original Data
scale = MinMaxScaler() # Scale the data.
scale.fit(x_train)
x_train_s = pd.DataFrame(scale.transform(x_train), columns = list(x_train.columns))
x_test_s = pd.DataFrame(scale.transform(x_test), columns = list(x_test.columns))
date = pd.DataFrame(x_train.index)
date_t = pd.DataFrame(x_test.index)
x_train_s = pd.concat([date, x_train_s], axis = 1)
x_test_s = pd.concat([date_t, x_test_s], axis = 1) # X_train scaled and ready to go
x_train_s = x_train_s.set_index('Unnamed: 0')
x_test_s = x_test_s.set_index('Unnamed: 0') # X_test scaled and ready to go


# Support Vector Machine
# sklearn specifies their kernel as np.exp(-gamma || x-x' || **2)
# gamma = 1/(2sigma**2)

def mysvc(c, sigma, degree):
    g = 1 / (2 * sigma**2)
    svm_clf = svm.SVC(C = c, gamma=g, kernel='rbf')
    return svm_clf

for i in [1, 50, 100]:
    for j in range(1,20):
        m = mysvc(i, j)
        m.fit(x_train_s, y_train)
        pred = m.predict(x_test_s)
        t = scoring_model(y_test, pred)
        print(f'C = {i}, sigma = {j}', t.accuracy())
