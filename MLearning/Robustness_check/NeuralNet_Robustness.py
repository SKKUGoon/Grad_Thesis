import numpy as np
import pandas as pd
import timeit


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from MLearning.ScoringMethods import scoring_model


from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout


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
                                                   test_size=0.3,
                                                   shuffle=False) # Split the data. For now no validation set
                                                                  # no random split. so shuffle = false
# Original Data
scale = StandardScaler() # Scale the data.
scale.fit(X_train)
X_train_s = pd.DataFrame(scale.transform(X_train), columns = list(X_train.columns))
X_test_s = pd.DataFrame(scale.transform(X_test), columns = list(X_test.columns))
date = pd.DataFrame(X_train.index)
date_t = pd.DataFrame(X_test.index)
X_train_s = pd.concat([date, X_train_s], axis=1)
X_test_s = pd.concat([date_t, X_test_s], axis=1) # X_train scaled and ready to go
X_train_s = X_train_s.set_index('Unnamed: 0')
X_test_s = X_test_s.set_index('Unnamed: 0') # X_test scaled and ready to go


# Neural Networks
sens = list()
spec = list()
gm = list() # store 20 iterations result
for j in range(20):
    model = Sequential()
    model.add(Dense(units=45, input_dim=len(X_train_s.columns), activation='elu'))
    model.add(Dense(units=20, activation='relu'))
    model.add(Dense(units=10, activation='linear'))
    model.add(Dense(units=1, activation='tanh'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    result1 = model.fit(X_train_s, y_train, verbose=0)
    y_pred_NN_temp = model.predict(X_test_s)
    y_pred_NN = []
    for i in range(len(y_pred_NN_temp)):
        if y_pred_NN_temp[i] >= 0:
            y_pred_NN.append(1)
        else:
            y_pred_NN.append(-1)
    acc_NN = scoring_model(y_test, y_pred_NN)

    # add metrics
    sens.append(acc_NN.sensitivity())
    spec.append(acc_NN.specificity())
    gm.append(acc_NN.gmean())

    # print out phase number 1 ~ 20
    print('NN iteration phase', (j + 0))

print('sensitivity is ', sens, '\nspecificity is ', spec, '\ngmean is ', gm)
