import pandas as pd
import numpy as np
import timeit

from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm

import matplotlib.pyplot as plt

from mlearning.scoringMethods import scoring_model
from mlearning.AdaboostM import AdaboostClassifierES

import warnings

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
# Logistic Regression
dt1 = DecisionTreeClassifier(max_depth=2, random_state=42) # Not Random
dt2 = DecisionTreeClassifier(max_depth=10, random_state=42) # Not Random
dt3 = DecisionTreeClassifier(max_depth=17, random_state=42) # Not Random
dt4 = DecisionTreeClassifier(max_depth=25, random_state=42) # Not Random

def mysvc(c, sigma, k='rbf'):
    g = 1 / (2 * sigma**2)
    svm_clf = svm.SVC(C = c, gamma=g, kernel=k, probability=True)
    return svm_clf
b = mysvc(1,10)
c = mysvc(1,5)
d = mysvc(1,1)

clf_ls =[dt1, dt2, dt3]
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
y_ls = y_test[y_test.columns[0]]


a = AdaboostClassifierES(clf_ls, max_iter=50, early_stopping=False)

a.fit(x_train_s,y_train)
p = a.predict(x_test_s)
b = scoring_model(y_ls, list(p))
print(b.accuracy(weight='weighted'))

ss = a.get_iter_prediction()
for i in range(len(ss)-1):
    print(np.cov(ss[i], ss[i+1]))
