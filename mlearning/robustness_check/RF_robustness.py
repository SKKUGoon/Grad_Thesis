import numpy as np
import pandas as pd
import timeit

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

from mlearning.scoringMethods import scoring_model

import warnings
warnings.filterwarnings(action = "ignore", category = FutureWarning)

# Data_using
start = timeit.default_timer()

wd = pd.read_csv(r'D:\Data\Grad\test_Work_Data_w_lag.csv')
wd = wd.set_index(wd.columns[0])
wd.index = pd.to_datetime(wd.index)

x = wd[wd.columns[1:]]
y = wd['korclf1']


# Create trainig set, testing set.
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.3,
                                                    shuffle=False) # Split the data. For now no validation set
                                                                    # no random split. so shuffle = false
# Original Data
scale = MinMaxScaler() # Scale the data.
scale.fit(x_train)
x_train_s = pd.DataFrame(scale.transform(x_train), columns = list(x_train.columns))
x_test_s = pd.DataFrame(scale.transform(x_test), columns = list(x_test.columns))
date = pd.DataFrame(x_train.index)
date_t = pd.DataFrame(x_test.index)
x_train_s = pd.concat([date, x_train_s], axis = 1)
x_test_s = pd.concat([date_t, x_test_s], axis = 1) # X_train scaled and ready to go
x_train_s = x_train_s.set_index('Unnamed: 0')
x_test_s = x_test_s.set_index('Unnamed: 0') # X_test scaled and ready to go

# Model _ Random Forest
robust_sens = {}
robust_spec = {}
gm = {}
acc = {}
boostadd = {}

for j in range(1,30):
    robust_check1 = []
    robust_check2 = []
    robust_gmean = []
    robust_acc = []
    for i in range(100):
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=j, n_jobs=-1)
        rf_clf.fit(x_train_s, y_train)
        y_pred_rf = rf_clf.predict((x_test_s[x_train_s.columns]))
        acc_rf = scoring_model(y_test, y_pred_rf)

        robust_check1.append(acc_rf.sensitivity())
        robust_check2.append(acc_rf.specificity())
        robust_gmean.append(acc_rf.gmean())
        robust_acc.append(acc_rf.accuracy())
    robust_sens[f'max_depth = {j}'] = robust_check1
    robust_spec[f'max_depth = {j}'] = robust_check2
    gm[f'max_depth = {j}'] = robust_gmean
    acc[f'max_depth = {j}'] = robust_acc
    if min(robust_acc) > 0.5:
        boostadd[f'max_depth = {j}'] = robust_acc
    print(f'max_depth {j} checked')



print('Final results are:\n', robust_sens, '\n', robust_spec, '\n', gm, '\n', acc)
print(boostadd.keys())

# Graph drawing
x_ = list(range(1,30))
sens100 = list()
spec100 = list()
gm100 = list()
acc100 = list()
acc_sd100 = list()
acc_minmax100 = list()

for i in range(29):
    sens100.append(np.mean(list(robust_sens.values())[i]))
    spec100.append(np.mean(list(robust_spec.values())[i]))
    gm100.append(np.mean(list(gm.values())[i]))
    acc100.append(np.mean(list(acc.values())[i]))
    acc_sd100.append(np.var(list(acc.values())[i]))
    acc_minmax100.append(max(list(acc.values())[i]) - min(list(acc.values())[i]))


plt.plot(x_, sens100, color='r', label='sensitivity')
plt.plot(x_, spec100, color='b', label='specificity')
plt.plot(x_, gm100, color='g', label='gmean')
plt.xlabel('maximum depth')
plt.title('Number of Trees : 100')
plt.legend()
plt.show()

plt.plot(x_, acc100, color='c', label='accuracy')
plt.bar(x_, acc_sd100, color='r', label='standard deviation')
plt.plot(x_, acc_minmax100, color = 'g', label='min max difference')
plt.plot(x_, [0.5]*29, color='b', linestyle='dashed', label='50% line')
plt.xlabel('maximum depth')
plt.title('Number of Trees : 100')
plt.legend()
plt.show()
