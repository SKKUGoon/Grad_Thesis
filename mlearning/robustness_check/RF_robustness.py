import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from mlearning.scoringMethods import scoring_model

from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
import tensorflow as tf

import random
import warnings
warnings.filterwarnings(action = "ignore", category = FutureWarning)

# Set Seed

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
scale = StandardScaler() # Scale the data.
scale.fit(x_train)
x_train_s = pd.DataFrame(scale.transform(x_train), columns=list(x_train.columns))
x_test_s = pd.DataFrame(scale.transform(x_test), columns=list(x_test.columns))
date = pd.DataFrame(x_train.index)
date_t = pd.DataFrame(x_test.index)
x_train_s = pd.concat([date, x_train_s], axis=1)
x_test_s = pd.concat([date_t, x_test_s], axis=1) # X_train scaled and ready to go
x_train_s = x_train_s.set_index('Unnamed: 0')
x_test_s = x_test_s.set_index('Unnamed: 0') # X_test scaled and ready to go

# Model _ Random Forest
robust_sens = {}
robust_spec = {}
gm = {}
acc = {}
for j in range(1,30):
    robust_check1 = []
    robust_check2 = []
    robust_gmean = []
    robust_acc = []
    for i in range(20):
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=j)
        rf_clf.fit(x_train_s, y_train)
        y_pred_rf = rf_clf.predict((x_test_s[x_train_s.columns]))
        acc_rf = scoring_model(y_test, y_pred_rf)

        robust_check1.append(acc_rf.sensitivity())
        robust_check2.append(acc_rf.specificity())
        robust_gmean.append(acc_rf.gmean())
        robust_acc.append(acc_rf.accuracy())
    robust_sens[f'max_depth = {j}'] = np.mean(robust_check1)
    robust_spec[f'max_depth = {j}'] = np.mean(robust_check2)
    gm[f'max_depth = {j}'] = np.mean(robust_gmean)
    acc[f'max_depth = {j}'] = np.mean(robust_acc)
    print(f'max_depth {j} checked')



print(robust_sens, '\n', robust_spec, '\n', gm)
