import numpy as np
import pandas as pd
import timeit


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from mlearning.scoringMethods import scoring_model


from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout


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
