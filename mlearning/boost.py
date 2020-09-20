from sklearn.ensemble import AdaBoostClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm

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


# Voting Classifier
# Logistic Regression
logR_clf = LogisticRegression() # Not Random

boostadd = list()

# Random Forest
for j in range(1,30):
    cv_ls = list()
    for i in range(100):
        rf_clf = RandomForestClassifier(n_estimators=1000, max_depth=j, n_jobs=-1)
        rf_clf.fit(x_train_s, y_train)
        y_pred_rf = rf_clf.predict((x_test_s[x_train_s.columns]))
        acc_rf = scoring_model(y_test, y_pred_rf)

    if min(cv_ls) > 0.5:
        boost_ele = RandomForestClassifier(n_estimators=100, max_depth=j, n_jobs=-1)
        boostadd.append(boost_ele)
    print('RandomForest iteration completed')

print('RandomForestClassifier added to boostadd')

# Names of classifierl
num_cls = len(boostadd)
names = list()
for i in range(num_cls):
    names.append('clf'+str(i+1))


# Overall
votingClf = VotingClassifier(list(zip(names, boostadd)), voting='soft')
print(list(zip(names, boostadd)))

# Adaboost itself
adaboost = AdaBoostClassifier(base_estimator=votingClf, n_estimators=num_cls, algorithm='SAMME')
adaboost.fit(x_train_s, y_train)
y_pred_adaboost = adaboost.predict(x_test_s)
acc_adaboost = scoring_model(y_test, y_pred_adaboost)
acc4 = {'gmean' : acc_adaboost.gmean(),
        'LP' : acc_adaboost.LP(),
        'LR' : acc_adaboost.LR(),
        'DP' : acc_adaboost.DP(),
        'Youden' : acc_adaboost.Youden(),
        'BA' : acc_adaboost.BA(),
        'accuracy' : acc_adaboost.accuracy(weight='weighted')}
print(acc_adaboost.sensitivity(), acc_adaboost.specificity(), acc_adaboost.accuracy())