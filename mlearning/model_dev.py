import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from mlearning.scoringMethods import scoring_model

from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
import tensorflow as tf

import random
import warnings
warnings.filterwarnings(action = "ignore", category = FutureWarning)

# Set Seed
random.seed(41)

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

# logistic regression
from sklearn.linear_model import LogisticRegression
logR_clf = LogisticRegression().fit(x_train_s, y_train)
y_pred_logR = logR_clf.predict(x_test_s[x_train_s.columns])
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
rf_clf = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 42)
rf_clf.fit(x_train_s, y_train)
y_pred_rf = rf_clf.predict((x_test_s[x_train_s.columns]))
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
svm_clf.fit(x_train_s, y_train)
y_pred_svm = svm_clf.predict((x_test_s[x_train_s.columns]))
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

# CART
CART_clf = DecisionTreeClassifier(max_depth = 1, random_state = 42) # deal with max_depth later
CART_clf.fit(x_train_s, y_train)
y_pred_CART = CART_clf.predict(x_test_s)
acc_CART = scoring_model(y_test, y_pred_CART)
acc4 = {'gmean' : acc_CART.gmean(),
        'LP' : acc_CART.LP(),
        'LR' : acc_CART.LR(),
        'DP' : acc_CART.DP(),
        'Youden' : acc_CART.Youden(),
        'BA' : acc_CART.BA()}
print(acc_CART.sensitivity(), acc_CART.specificity())
print('Classification and Regression Tree', acc4)

# XGBoost - Later

# Deep Feed Forward Network(MXNET)
# import tensorflow as tf

# Neural Networks
acc_NN_dict = {'gmean' : [],
               'LP' : [],
               'LR' : [],
               'DP' : [],
               'Youden' : [],
               'BA' : []} # store 20 iterations result
for iter_ in range(10):
    model = Sequential()
    model.add(Dense(units=45, input_dim=len(x_train_s.columns), activation='elu'))
    model.add(Dense(units=20, activation='relu'))
    model.add(Dense(units=10, activation='linear'))
    model.add(Dense(units=1, activation='tanh'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    result1 = model.fit(x_train_s, y_train, verbose=0)
    y_pred_NN_temp = model.predict(x_test_s)
    y_pred_NN = []
    for i in range(len(y_pred_NN_temp)):
        if y_pred_NN_temp[i] >= 0:
            y_pred_NN.append(1)
        else:
            y_pred_NN.append(-1)
    acc_NN = scoring_model(y_test, y_pred_NN)

    # add metrics
    acc_NN_dict['gmean'].append(acc_NN.gmean())
    acc_NN_dict['LP'].append(acc_NN.LP())
    acc_NN_dict['LR'].append(acc_NN.LR())
    acc_NN_dict['DP'].append(acc_NN.DP())
    acc_NN_dict['Youden'].append(acc_NN.Youden())
    acc_NN_dict['BA'].append(acc_NN.BA())
    # print out phase number 1 ~ 20
    print('NN iteration phase', (iter_ + 0))

dictkey = list(acc_NN_dict.keys())
acc5 = {'gmean' : np.mean(acc_NN_dict['gmean']),
        'LP' : np.mean(acc_NN_dict['LP']),
        'LR' : np.mean(acc_NN_dict['LR']),
        'DP' : np.mean(acc_NN_dict['DP']),
        'Youden' : np.mean(acc_NN_dict['Youden']),
        'BA' : np.mean(acc_NN_dict['BA'])}
print('Neural Net', acc5)

# Adaboost itself
from sklearn.ensemble import AdaBoostClassifier,VotingClassifier
votingClf = VotingClassifier([('clf1', logR_clf), ('clf2', rf_clf),
                              ('clf3', CART_clf)], voting='soft')

adaboost = AdaBoostClassifier(base_estimator=votingClf)
adaboost.fit(x_train_s, y_train)
y_pred_adaboost = adaboost.predict(x_test_s)
acc_adaboost = scoring_model(y_test, y_pred_adaboost)
acc4 = {'gmean' : acc_adaboost.gmean(),
        'LP' : acc_adaboost.LP(),
        'LR' : acc_adaboost.LR(),
        'DP' : acc_adaboost.DP(),
        'Youden' : acc_adaboost.Youden(),
        'BA' : acc_adaboost.BA()}
print(acc_adaboost.sensitivity(), acc_adaboost.specificity())
print('acc_adaboost', acc4)


# ROC Curve

# Logistic Regression
logR_proba = []
pr = logR_clf.predict_proba(x_test_s)
for i in range(len(x_test_s)):
    logR_proba.append(pr[i][1])
fpr_NN, tpr_NN, thresholds_NN = roc_curve(y_test, logR_proba)
AUC_logR = auc(fpr_NN, tpr_NN)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_NN, tpr_NN, label='LogR (area = {:.3f})'.format(AUC_logR))

# Random Forest
rf_proba = []
pr = rf_clf.predict_proba(x_test_s)
for i in range(len(x_test_s)):
    rf_proba.append(pr[i][1])
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf_proba)
AUC_RF = auc(fpr_rf, tpr_rf)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(AUC_RF))

# Support Vector Machine
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, y_pred_svm)
auc_svm = auc(fpr_svm, tpr_svm)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_svm, tpr_svm, label='SVM (area = {:.3f})'.format(auc_svm))

# Neural Net
fpr_NN, tpr_NN, thresholds_NN = roc_curve(y_test, y_pred_NN_temp.ravel())
AUC_NN = auc(fpr_NN, tpr_NN)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_NN, tpr_NN, label='Neural Net (area = {:.3f})'.format(AUC_NN))

# Adaboost
ada_proba = []
a = adaboost.predict_proba(x_test_s)
for i in range(len(y_test)):
    ada_proba.append(a[i][1])

fpr_ADA, tpr_ADA, thresholds_ADA = roc_curve(y_test, ada_proba)
AUC_ADA = auc(fpr_ADA, tpr_ADA)
plt.plot([0, 1], [0,1 ], 'k--')
plt.plot(fpr_ADA, tpr_ADA, label='ADABoost (area = {:.3f}'.format(AUC_ADA))


plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')

plt.legend(loc='best')

plt.show()

# Save Prediction Data
logR_df = pd.DataFrame(y_pred_logR, columns = ['logR'])
RF_df = pd.DataFrame(y_pred_rf, columns = ['RF'])
SVM_df = pd.DataFrame(y_pred_svm, columns = ['SVM'])
CART_df = pd.DataFrame(y_pred_CART, columns = ['CART'])
NN_df = pd.DataFrame(y_pred_NN, columns = ['NN'])
real = pd.DataFrame(list(y_test), columns = ['global crisis'])
Ttl = pd.concat([logR_df, RF_df, SVM_df, CART_df, NN_df, real], axis = 1)
Ttl.index = y_test.index
Ttl.to_csv(r'D:\Data\Grad\y_pred_all.csv')