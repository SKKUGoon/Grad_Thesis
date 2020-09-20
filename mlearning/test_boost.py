from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

import random
import timeit
import pandas as pd
import numpy as np

from mlearning.scoringMethods import scoring_model

import warnings

warnings.filterwarnings(action="ignore", category=FutureWarning)
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
                                                    test_size = 0.3,
                                                    shuffle = True) # Split the data. For now no validation set
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


# Overall

t = list()
def mysvc(c, sigma, k='rbf'):
    g = 1 / (2 * sigma**2)
    svm_clf = SVC(C = c, gamma=g, kernel=k, probability=True)
    return svm_clf

clfnum = list(range(11, 40))
ind = 0
for c_val in [1, 5, 10]:
    for sigma_val in [10]:
        clf = (f'clf{clfnum[ind]}', mysvc(c_val, sigma_val))
        t.append(clf)
        ind += 1

votingClf = VotingClassifier(t)
print(votingClf)

# Adaboost itself

formean = list()

for i in range(5):
    # 10 iteration and take mean / variance of the result
    adaboost = AdaBoostClassifier(base_estimator=votingClf, n_estimators=len(t), algorithm='SAMME')
    adaboost.fit(x_train_s, y_train)
    y_pred_adaboost = adaboost.predict(x_test_s)
    acc_adaboost = scoring_model(y_test, y_pred_adaboost)
    print(acc_adaboost.sensitivity(), acc_adaboost.specificity(), acc_adaboost.accuracy())

    formean.append(acc_adaboost.accuracy())
m = np.mean(formean)
v = np.var(formean)
print(f'mean accuracy = {m}, variance = {v}')