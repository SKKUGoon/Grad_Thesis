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

import random
import warnings
warnings.filterwarnings(action = "ignore", category = FutureWarning)

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

# Neural Networks
model = Sequential()
model.add(Dense(units = 50, input_dim = len(X_train_s.columns), activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'sgd')

acc_NN_dict = {'gmean' : [],
               'LP' : [],
               'LR' : [],
               'DP' : [],
               'Youden' : [],
               'BA' : []} # store 20 iterations result
for iter_ in range(0,2):
        np.random.seed(iter_)
        result1 = model.fit(X_train_s, y_train, verbose = 0)
        y_pred_NN_temp = model.predict(X_test_s)
        y_pred_NN = []
        for i in range(len(y_pred_NN_temp)):
            if y_pred_NN_temp[i] >= 0.5:
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

fpr_NN, tpr_NN, thresholds_NN = roc_curve(y_test, y_pred_NN_temp.ravel())
auc_keras = auc(fpr_NN, tpr_NN)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_NN, tpr_NN, label='NN (area = {:.3f})'.format(auc_keras))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')

plt.legend(loc='best')

plt.show()