import numpy as np
import pandas as pd
import timeit

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlearning.scoringMethods import scoring_model

import matplotlib.pyplot as plt

import random
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
                                                    test_size = 0.3,
                                                    shuffle = False) # Split the data. For now no validation set
                                                                    # no random split. so shuffle = false
# Original Data
scale = StandardScaler() # Scale the data.
scale.fit(x_train)
x_train_s = pd.DataFrame(scale.transform(x_train), columns = list(x_train.columns))
x_test_s = pd.DataFrame(scale.transform(x_test), columns = list(x_test.columns))
date = pd.DataFrame(x_train.index)
date_t = pd.DataFrame(x_test.index)
x_train_s = pd.concat([date, x_train_s], axis = 1)
x_test_s = pd.concat([date_t, x_test_s], axis = 1) # X_train scaled and ready to go
x_train_s = x_train_s.set_index('Unnamed: 0')
x_test_s = x_test_s.set_index('Unnamed: 0') # X_test scaled and ready to go

# Robustness Check
CART_sens = {}
CART_spec = {}
CART_gm = {}
for depth in range(1, 30):
    robust_check1 = []
    robust_check2 = []
    robust_gmean = []
    for i in range(20): # average over 20
        CART_clf = DecisionTreeClassifier(max_depth = depth) # deal with max_depth later
        CART_clf.fit(x_train_s, y_train)
        y_pred_CART = CART_clf.predict(x_test_s)
        acc_CART = scoring_model(y_test, y_pred_CART)

        robust_check1.append(acc_CART.sensitivity())
        robust_check2.append(acc_CART.specificity())
        robust_gmean.append(acc_CART.gmean())
    CART_sens[f'max_depth = {depth}'] = np.mean(robust_check1)
    CART_spec[f'max_depth = {depth}'] = np.mean(robust_check2)
    CART_gm[f'max_depth = {depth}'] = np.mean(robust_gmean)
    print(f'max_depth {depth} checked')

sens = list()
spec = list()
gm = list()
for i in range(len(list(CART_sens.keys()))):
    sens.append(CART_sens[list(CART_sens.keys())[i]])
    spec.append(CART_spec[list(CART_spec.keys())[i]])
    gm.append(CART_gm[list(CART_gm.keys())[i]])
x_axis = list(range(1, 30))

plt.plot(x_axis, sens, color = 'r', label = 'sensitivity')
plt.plot(x_axis, spec, color = 'b', label = 'specificity')
plt.plot(x_axis, gm, color = 'g', label = 'gmean')
plt.xlabel('maximum depth')
plt.title('CART')
plt.legend()
plt.show()