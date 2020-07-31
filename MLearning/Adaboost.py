from sklearn.ensemble import AdaBoostClassifier,VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasRegressor

import random
import timeit
import pandas as pd
import numpy as np

from MLearning.ScoringMethods import scoring_model

import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)
# Set Seed
random.seed(41)

# Data_using
start = timeit.default_timer()

WD = pd.read_csv(r'D:\Data\Grad\test_Work_Data_w_lag.csv')
WD = WD.set_index(WD.columns[0])
WD.index = pd.to_datetime(WD.index)

WD1 = WD[WD.columns[0:4]].fillna(method = 'ffill')
WD2 = WD[WD.columns[4:1078]].interpolate(method = 'time')
WD = pd.concat([WD1, WD2], axis = 1)

Indep_var = pd.read_csv(r"D:\Data\Grad\X_filitered_lasso.csv")
Indep_var = Indep_var.set_index(Indep_var.columns[0])
Indep_var.index = pd.to_datetime(Indep_var.index)

y = WD['kor class']
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


# Voting Classifier
logR_clf = LogisticRegression() # Not Random
rf_clf = RandomForestClassifier(n_estimators = 100, max_depth = 5)
CART_clf = DecisionTreeClassifier(max_depth = 1)
svm_clf = svm.SVC()

def simple_model():
    # create model
    model = Sequential()
    model.add(Dense(units=25, input_dim=len(X_train_s.columns), activation='relu'))
    model.add(Dense(units=1, activation='tanh'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    return model

ann_estimator = KerasRegressor(build_fn=simple_model, epochs=100, batch_size=10, verbose=0)

votingClf = VotingClassifier([('clf1', logR_clf), ('clf2', rf_clf),
                              ('clf3', CART_clf)], voting='soft')

# Adaboost itself
adaboost = AdaBoostClassifier(base_estimator=votingClf)
adaboost.fit(X_train_s, y_train)
y_pred_adaboost = adaboost.predict(X_test_s)
acc_adaboost = scoring_model(y_test, y_pred_adaboost)
acc4 = {'gmean' : acc_adaboost.gmean(),
        'LP' : acc_adaboost.LP(),
        'LR' : acc_adaboost.LR(),
        'DP' : acc_adaboost.DP(),
        'Youden' : acc_adaboost.Youden(),
        'BA' : acc_adaboost.BA()}
print(acc_adaboost.sensitivity(), acc_adaboost.specificity())
print('acc_adaboost', acc4)