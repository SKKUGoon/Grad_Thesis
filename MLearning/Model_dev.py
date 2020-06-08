import pandas as pd
import timeit

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
warnings.filterwarnings(action = "ignore", category = FutureWarning)


# Data_using
start = timeit.default_timer()

WD = pd.read_csv(r'D:\Data\Grad\test_Work_Data.csv')
WD = WD.set_index(WD.columns[0])
WD.index = pd.to_datetime(WD.index)

X_col = list(WD.columns)
X_col.remove('global crisis') # X use all of the columns from the WD minus 'global crisis'column which is y

y = WD['global crisis']
X = WD[[i for i in X_col]]

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

# Filtered Data with Boruta
X_fil_train = pd.read_csv(r"D:\Data\Grad\X_filtered.csv")
X_fil_train = X_fil_train.set_index(X_fil_train.columns[0])

# logistic regression
from sklearn.linear_model import LogisticRegression
logR_clf = LogisticRegression().fit(X_fil_train, y_train)
y_pred_logR = logR_clf.predict(X_test_s[X_fil_train.columns])
acc = metrics.accuracy_score(y_test, y_pred_logR)
print('logistic_regression', acc)
# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators = 100)
rf_clf.fit(X_fil_train, y_train)
y_pred_rf = rf_clf.predict((X_test_s[X_fil_train.columns]))
acc2 = metrics.accuracy_score(y_test, y_pred_rf)
print('Random forest', acc2)
# support vector machine
from sklearn import svm
svm_clf = svm.SVC()
svm_clf.fit(X_fil_train, y_train)
y_pred_svm = svm_clf.predict((X_test_s[X_fil_train.columns]))
sv = svm_clf.support_vectors_
acc3 = metrics.accuracy_score(y_test, y_pred_svm)
print('Support Vector machine', acc3)
# Neural Networks

# CART
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#from sklearn import tree
#cv = KFold(n = X_fil_train.shape[0], n_folds = 5, shuffle = True, random_state = 1)
#for i in range(1,10):
#    tree_clf = tree.DecisionTreeClassifier(max_depth = i, random_state = 0)
#    if tree_clf.fit(X_fil_train, y_train).tree_
from sklearn import tree


# XGBoost - Later

# Deep Feed Forward Network(MXNET)
import tensorflow as tf