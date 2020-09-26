import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm

import numpy as np

import matplotlib.pyplot as plt

from mlearning.scoringMethods import scoring_model
from mlearning.AdaboostM import AdaboostClassifierES

import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

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

# Data_using
wd = pd.read_csv(r'D:\Data\Grad\test_Work_Data_w_lag.csv')
wd = wd.set_index(wd.columns[0])
wd.index = pd.to_datetime(wd.index)

x = wd[wd.columns[1:]]
y = wd['korclf1']


# Create trainig set, testing set.
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.3,
                                                    shuffle=True) # Split the data. For now no validation set
                                                                    # no random split. so shuffle = false
# Original Data
scale = MinMaxScaler() # Scale the data.
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
dt1 = DecisionTreeClassifier(max_depth=2, random_state=42) # Not Random
dt2 = DecisionTreeClassifier(max_depth=5, random_state=42) # Not Random
dt3 = DecisionTreeClassifier(max_depth=10, random_state=42) # Not Random
dt4 = DecisionTreeClassifier(max_depth=15, random_state=42) # Not Random
dt5 = DecisionTreeClassifier(max_depth=20, random_state=42) # Not Random
dt6 = DecisionTreeClassifier(max_depth=25, random_state=42) # Not Random

rf1 = RandomForestClassifier(n_estimators=1000, max_depth=2, n_jobs=-1, random_state=42)
rf2 = RandomForestClassifier(n_estimators=1000, max_depth=5, n_jobs=-1, random_state=42)
rf3 = RandomForestClassifier(n_estimators=1000, max_depth=10, n_jobs=-1, random_state=42)
rf4 = RandomForestClassifier(n_estimators=500, max_depth=2, n_jobs=-1, random_state=42)
rf5 = RandomForestClassifier(n_estimators=500, max_depth=5, n_jobs=-1, random_state=42)
rf6 = RandomForestClassifier(n_estimators=500, max_depth=10, n_jobs=-1, random_state=42)

n1 = KerasClassifier(build_fn=nn1, epochs=50, verbose=0)
n2 = KerasClassifier(build_fn=nn1, epochs=100, verbose=0)
n3 = KerasClassifier(build_fn=nn1, epochs=150, verbose=0)
n4 = KerasClassifier(build_fn=nn2, epochs=50, verbose=0)
n5 = KerasClassifier(build_fn=nn2, epochs=100, verbose=0)
n6 = KerasClassifier(build_fn=nn2, epochs=150, verbose=0)

clf_ls =[dt1, dt2, dt3, dt4, dt5, dt6]

y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
y_ls = y_test[y_test.columns[0]]

a = AdaboostClassifierES(clf_ls, max_iter=100, early_stopping=False)

a.fit(x_train_s, y_train)
p2 = a.predict(x_test_s)
b2 = scoring_model(y_ls, list(p2))
print(b2.accuracy(weight='weighted'))