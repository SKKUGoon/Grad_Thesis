import pandas as pd
import timeit

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from MLearning.ScoringMethods import scoring_model

import random
import warnings
warnings.filterwarnings(action = "ignore", category = FutureWarning)

# Set Seed
random.seed(41)

# Data_using
start = timeit.default_timer()

WD = pd.read_csv(r'D:\Data\Grad\test_Work_Data.csv')
WD = WD.set_index(WD.columns[0])
WD.index = pd.to_datetime(WD.index)

WD1 = WD[WD.columns[0:4]].fillna(method = 'ffill')
WD2 = WD[WD.columns[4:1078]].interpolate(method = 'time')
WD = pd.concat([WD1, WD2], axis = 1)

Indep_var = pd.read_csv(r'D:\Data\Grad\X_filtered.csv')
Indep_var = Indep_var.set_index(Indep_var.columns[0])
Indep_var.index = pd.to_datetime(Indep_var.index)

y = WD['global crisis']
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

# logistic regression
from sklearn.linear_model import LogisticRegression
logR_clf = LogisticRegression().fit(X_train_s, y_train)
y_pred_logR = logR_clf.predict(X_test_s[X_train_s.columns])
acc_logR = scoring_model(y_test, y_pred_logR)
print(acc_logR.sensitivity(), acc_logR.specificity())
acc1 = {'gmean' : acc_logR.gmean(),
        'LP' : acc_logR.LP(),
        'LR' : acc_logR.LR(),
        'DP' : acc_logR.DP(),
        'Youden' : acc_logR.Youden(),
        'BA' : acc_logR.BA()}
print('logistic_regression', acc1)
# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators = 100)
rf_clf.fit(X_train_s, y_train)
y_pred_rf = rf_clf.predict((X_test_s[X_train_s.columns]))
acc_rf = scoring_model(y_test, y_pred_rf)

print(acc_rf.sensitivity(), acc_rf.specificity())
acc2 = {'gmean' : acc_rf.gmean(),
        'LP' : acc_rf.LP(),
        'LR' : acc_rf.LR(),
        'DP' : acc_rf.DP(),
        'Youden' : acc_rf.Youden(),
        'BA' : acc_rf.BA()}
print('Random forest', acc2)
# support vector machine
from sklearn import svm
svm_clf = svm.SVC()
svm_clf.fit(X_train_s, y_train)
y_pred_svm = svm_clf.predict((X_test_s[X_train_s.columns]))
sv = svm_clf.support_vectors_
acc_svm = scoring_model(y_test, y_pred_svm)
print(acc_svm.sensitivity(), acc_svm.specificity())
acc3 = {'gmean' : acc_svm.gmean(),
        'LP' : acc_svm.LP(),
        'LR' : acc_svm.LR(),
        'DP' : acc_svm.DP(),
        'Youden' : acc_svm.Youden(),
        'BA' : acc_svm.BA()}
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