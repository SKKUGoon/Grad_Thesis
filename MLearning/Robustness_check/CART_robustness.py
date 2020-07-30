import numpy as np
import pandas as pd
import timeit

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from MLearning.ScoringMethods import scoring_model

import random
import warnings
warnings.filterwarnings(action = "ignore", category = FutureWarning)


# Data_using
start = timeit.default_timer()

WD = pd.read_csv(r'D:\Data\Grad\test_Work_Data_w_lag.csv')
WD = WD.set_index(WD.columns[0])
WD.index = pd.to_datetime(WD.index)

WD1 = WD[WD.columns[0:4]].fillna(method = 'ffill')
WD2 = WD[WD.columns[4:1078]].interpolate(method = 'time')
WD = pd.concat([WD1, WD2], axis = 1)

Indep_var = pd.read_csv(r"D:\Data\Grad\X_filitered_lasso.csv")
Indep_var = Indep_var.set_index(Indep_var.columns[0])
Indep_var.index = pd.to_datetime(Indep_var.index)

y = WD['kor class']
X = WD[Indep_var.columns]

# Create trainig set, testing set.
X_train, X_test,y_train, y_test = train_test_split(X, y,
                                                   test_size = 0.3,
                                                   shuffle = False) # Split the data. For now no validation set
                                                                    # no random split. so shuffle = false
# Original Data
scale = StandardScaler() # Scale the data.
scale.fit(X_train)
X_train_s = pd.DataFrame(scale.transform(X_train), columns = list(X_train.columns))
X_test_s = pd.DataFrame(scale.transform(X_test), columns = list(X_test.columns))
date = pd.DataFrame(X_train.index)
date_t = pd.DataFrame(X_test.index)
X_train_s = pd.concat([date, X_train_s], axis = 1)
X_test_s = pd.concat([date_t, X_test_s], axis = 1) # X_train scaled and ready to go
X_train_s = X_train_s.set_index('Unnamed: 0')
X_test_s = X_test_s.set_index('Unnamed: 0') # X_test scaled and ready to go

# Robustness Check
robust_sens = {}
robust_spec = {}
gm = {}
for depth in range(1, 30):
    robust_check1 = []
    robust_check2 = []
    robust_gmean = []
    for i in range(20): # average over 20
        CART_clf = DecisionTreeClassifier(max_depth = depth) # deal with max_depth later
        CART_clf.fit(X_train_s, y_train)
        y_pred_CART = CART_clf.predict(X_test_s)
        acc_CART = scoring_model(y_test, y_pred_CART)

        robust_check1.append(acc_CART.sensitivity())
        robust_check2.append(acc_CART.specificity())
        robust_gmean.append(acc_CART.gmean())
    robust_sens[f'max_depth = {depth}'] = np.mean(robust_check1)
    robust_spec[f'max_depth = {depth}'] = np.mean(robust_check2)
    gm[f'max_depth = {depth}'] = np.mean(robust_gmean)
    print(f'max_depth {depth} checked')

print(robust_sens, '\n', robust_spec, '\n', gm)