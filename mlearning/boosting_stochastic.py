import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score

import numpy as np

import matplotlib.pyplot as plt

from mlearning.AdaboostM import AdaboostClassifierES

import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import copy
import datetime

import warnings


def nn1():
    inp = 66
    # create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(inp//2, input_dim=inp, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='tanh'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def nn2():
    inp = 66
    # create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(inp//2, input_dim=inp, activation='relu'))
    model.add(tf.keras.layers.Dense(inp//4, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='tanh'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def nn3():
    inp = 66
    # create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(inp//2, input_dim=inp, activation='relu'))
    model.add(tf.keras.layers.Dense(inp//2, activation='relu'))
    model.add(tf.keras.layers.Dense(inp//8, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='tanh'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


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

#clf_ls =[dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9, dt10, dt11,
#         rf1, rf2, rf3, rf4, rf5, rf6, rf7, rf8, rf9, rf10, rf11, rf12, rf13, rf14, rf15, rf16, rf17, rf18,
#         n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, svc]

#clf_ls =[dt1, dt3, dt5, dt7, dt9, dt11,
#         rf1, rf3, rf5, rf7, rf9, rf11, rf13, rf15, rf17,
#         n1, n3, n5, n7, n9, n11, svc]

clf_ls =[dt1, dt5, dt9,
         rf1, rf5, rf9, rf13, rf17,
         n1, n5, n9, svc]


# Data
X = pd.read_pickle(r'D:\Data\Grad\X_selected.pkl')
y = pd.read_pickle(r'D:\Data\Grad\y_selected.pkl')

# Training, Test Size
train_size = 365 * 3  # 3 years
test_size = 30  # 1 month

actual = list()
pr = list()
for i in range(6):  # FIXME: for 48 month
    # Split Data
    X_train, y_train = (X[(0 + test_size * i) : (train_size + test_size * i)],
                        y[(0 + test_size * i) : (train_size + test_size * i)])

    X_test, y_test = (X[(train_size + test_size * i):(train_size + test_size * (i + 1))],
                      y[(train_size + test_size * i):(train_size + test_size * (i + 1))])

    # Scale
    scale = StandardScaler()
    scale.fit(X_train)
    X_train_np = scale.transform(X_train)  # Scaled. np.array
    X_train = pd.DataFrame(X_train_np, index=X_train.index, columns=X_train.columns)

    # Fit
    c = AdaboostClassifierES(clf_ls, max_iter=50, early_stopping=False)
    c.stochastic_fit(X_train, y_train)
    print(f'fit: {i + 1}/{12}')

    # Predict
    X_test_np = scale.transform(X_test)
    X_test = pd.DataFrame(X_test_np, index=X_test.index, columns=X_test.columns)
    print(f'predict: {i + 1}/{12}')

    pred = c.predict(X_test)
    pr.append(pred)
    actual.append(y_test.to_numpy().tolist())