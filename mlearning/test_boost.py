import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm

import numpy as np

import matplotlib.pyplot as plt

from mlearning.scoringMethods import scoring_model
from mlearning.AdaboostM import AdaboostClassifierES

import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import time
import copy

import warnings
warnings.filterwarnings('ignore')


def nn1():
    inp = x_train_s.shape[1]
    # create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(inp//2, input_dim=inp, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='tanh'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def nn2():
    inp = x_train_s.shape[1]
    # create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(inp//2, input_dim=inp, activation='relu'))
    model.add(tf.keras.layers.Dense(inp//4, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='tanh'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def nn3():
    inp = x_train_s.shape[1]
    # create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(inp//2, input_dim=inp, activation='relu'))
    model.add(tf.keras.layers.Dense(inp//4, activation='relu'))
    model.add(tf.keras.layers.Dense(inp//8, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='tanh'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


# Data_using
wd = pd.read_csv(r'D:\Data\Grad\test_Work_Data_w_lag.csv')
wd = wd.set_index(wd.columns[0])
wd.index = pd.to_datetime(wd.index)

x = wd[wd.columns[1:]]
y = wd['korclf1']

not_rt, is_rt = [[] for _ in range(2)]

for i in x.columns:
    if 'rt' in i or 'clf' in i or 'lag' in i:
        is_rt.append(i)
    elif 'rt' not in i:
        not_rt.append(i)
    else:
        raise ValueError

x_org_not_rt = copy.deepcopy(x[not_rt])
col = dict()
for i in x_org_not_rt.columns:
    col[i] = i + '_original'
x_org_not_rt = x_org_not_rt.rename(columns=col)
x[not_rt] = (x[not_rt] - x[not_rt].shift(1)) / x[not_rt].shift(1)
x = pd.concat([x_org_not_rt, x], axis=1)
x = x[1:]  # Since we imposed 1 day log return
y = y[1:]

# Create trainig set, testing set.
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.3,
                                                    shuffle=False)  # Split the data. For now no validation set

# Original Data
scale = StandardScaler() # Scale the data.
scale.fit(x_train)
x_train_s = pd.DataFrame(scale.transform(x_train), columns = list(x_train.columns))
scale.fit(x_test)
x_test_s = pd.DataFrame(scale.transform(x_test), columns = list(x_test.columns))
date = pd.DataFrame(x_train.index)
date_t = pd.DataFrame(x_test.index)
x_train_s = pd.concat([date, x_train_s], axis = 1)
x_test_s = pd.concat([date_t, x_test_s], axis = 1) # X_train scaled and ready to go
x_train_s = x_train_s.set_index('Unnamed: 0')
x_test_s = x_test_s.set_index('Unnamed: 0') # X_test scaled and ready to go


# Voting Classifier
dt1 = DecisionTreeClassifier(max_depth=2) # Not Random
dt2 = DecisionTreeClassifier(max_depth=3) # Not Random
dt3 = DecisionTreeClassifier(max_depth=4) # Not Random
dt4 = DecisionTreeClassifier(max_depth=5) # Not Random
dt5 = DecisionTreeClassifier(max_depth=6) # Not Random
dt6 = DecisionTreeClassifier(max_depth=7) # Not Random
dt7 = DecisionTreeClassifier(max_depth=8) # Not Random
dt8 = DecisionTreeClassifier(max_depth=9) # Not Random
dt9 = DecisionTreeClassifier(max_depth=10) # Not Random
dt10 = DecisionTreeClassifier(max_depth=11) # Not Random
dt11 = DecisionTreeClassifier(max_depth=12) # Not Random

rf1 = RandomForestClassifier(n_estimators=1000, max_depth=2, n_jobs=-1)
rf2 = RandomForestClassifier(n_estimators=1000, max_depth=3, n_jobs=-1)
rf3 = RandomForestClassifier(n_estimators=1000, max_depth=4, n_jobs=-1)
rf4 = RandomForestClassifier(n_estimators=1000, max_depth=5, n_jobs=-1)
rf5 = RandomForestClassifier(n_estimators=1000, max_depth=6, n_jobs=-1)
rf6 = RandomForestClassifier(n_estimators=1000, max_depth=7, n_jobs=-1)
rf7 = RandomForestClassifier(n_estimators=1000, max_depth=8, n_jobs=-1)
rf8 = RandomForestClassifier(n_estimators=1000, max_depth=9, n_jobs=-1)
rf9 = RandomForestClassifier(n_estimators=1000, max_depth=10, n_jobs=-1)
rf10 = RandomForestClassifier(n_estimators=500, max_depth=2, n_jobs=-1)
rf11 = RandomForestClassifier(n_estimators=500, max_depth=3, n_jobs=-1)
rf12 = RandomForestClassifier(n_estimators=500, max_depth=4, n_jobs=-1)
rf13 = RandomForestClassifier(n_estimators=500, max_depth=5, n_jobs=-1)
rf14 = RandomForestClassifier(n_estimators=500, max_depth=6, n_jobs=-1)
rf15 = RandomForestClassifier(n_estimators=500, max_depth=7, n_jobs=-1)
rf16 = RandomForestClassifier(n_estimators=500, max_depth=8, n_jobs=-1)
rf17 = RandomForestClassifier(n_estimators=500, max_depth=9, n_jobs=-1)
rf18 = RandomForestClassifier(n_estimators=500, max_depth=10, n_jobs=-1)

n1 = KerasClassifier(build_fn=nn1, epochs=50, verbose=0)
n2 = KerasClassifier(build_fn=nn1, epochs=100, verbose=0)
n3 = KerasClassifier(build_fn=nn1, epochs=150, verbose=0)
n4 = KerasClassifier(build_fn=nn1, epochs=200, verbose=0)
n5 = KerasClassifier(build_fn=nn2, epochs=50, verbose=0)
n6 = KerasClassifier(build_fn=nn2, epochs=100, verbose=0)
n7 = KerasClassifier(build_fn=nn2, epochs=150, verbose=0)
n8 = KerasClassifier(build_fn=nn2, epochs=200, verbose=0)
n9 = KerasClassifier(build_fn=nn3, epochs=50, verbose=0)
n10 = KerasClassifier(build_fn=nn3, epochs=100, verbose=0)
n11 = KerasClassifier(build_fn=nn3, epochs=150, verbose=0)
n12 = KerasClassifier(build_fn=nn3, epochs=200, verbose=0)

svc = SVC(gamma='auto')

clf_ls =[dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9, dt10, dt11,
         rf1, rf2, rf3, rf4, rf5, rf6, rf7, rf8, rf9, rf10, rf11, rf12, rf13, rf14, rf15, rf16, rf17, rf18,
         n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, svc]

y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
y_ls = y_test[y_test.columns[0]]

#a = AdaboostClassifierES(clf_ls, max_iter=10, early_stopping=False)
#start_time = time.time()
#a.fit(x_train_s, y_train)
#end_time = time.time()
#p2 = a.predict(x_test_s)
#b2 = scoring_model(y_ls, list(p2))
#print(b2.accuracy(weight='weighted'))
#print(end_time - start_time)

b = AdaboostClassifierES(clf_ls, max_iter=150, early_stopping=False)
start_time = time.time()
b.stochastic_fit(x_train_s, y_train)
end_time = time.time()
p3 = b.predict(x_test_s)
b3 = scoring_model(y_ls, list(p3))
b3acc = b3.accuracy(weight='weighted')
print(b3acc)
print(end_time - start_time)

# Rolling Forecast
c = AdaboostClassifierES(clf_ls, max_iter=100, early_stopping=False)
c.stochastic_fit(x_train_s, y_train)

refit = 30  # Refit after a month
it = len(x_test_s)//refit + (len(x_test_s) % refit != 0)

pr = list()
iter_ = 1
for i in range(it):
    # Prediction
    p4 = c.predict(x_test_s[refit*(iter_-1) : refit*iter_])
    pr.append(p4)

    # Remake training set for x and y
    x_tr = pd.concat([x_train_s[refit*iter_:], x_test_s[refit*(iter_-1) : refit*iter_]])
    y_tr = pd.concat([y_train[refit*iter_:], y_test[refit*(iter_-1) : refit*iter_]])

    iter_ += 1

    # Reset the Model
    c = AdaboostClassifierES(clf_ls, max_iter=10, early_stopping=False)
    c.stochastic_fit(x_tr, y_tr)
    print(f'{i+1}/{it}')