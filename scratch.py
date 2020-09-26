import pandas as pd
import numpy as np
import timeit

from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
dt1 = DecisionTreeClassifier(max_depth=2, random_state=42) # Not Random
dt2 = DecisionTreeClassifier(max_depth=10, random_state=42) # Not Random
dt3 = DecisionTreeClassifier(max_depth=17, random_state=42) # Not Random
dt4 = DecisionTreeClassifier(max_depth=25, random_state=42) # Not Random

rf1 = RandomForestClassifier(n_estimators=1000, max_depth=2, n_jobs=-1, random_state=42)
rf2 = RandomForestClassifier(n_estimators=1000, max_depth=6, n_jobs=-1, random_state=42)
rf3 = RandomForestClassifier(n_estimators=1000, max_depth=10, n_jobs=-1, random_state=42)

clf_ls =[dt1, dt2, dt3, dt4, rf1, rf2, rf3]
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
y_ls = y_test[y_test.columns[0]]

big_sss = list()
big_pppp = list()
for i in range(1):
    a = AdaboostClassifierES(clf_ls, max_iter=20, early_stopping=False)

    a.fit(x_train_s,y_train)
    p = a.predict(x_test_s)
    b = scoring_model(y_ls, list(p))
    print(b.accuracy(weight='weighted'))

    ss = a.get_iter_prediction()
    for i in range(len(ss)-1):
        print(np.cov(ss[i], ss[i+1]))


    #
    sss = list()
    for i in range(len(ss)-1):
        if len(sss) == 0:
            sss.append(np.cov(ss[i], ss[i+1])[0][1])
        else:
            if max(sss) > np.cov(ss[i], ss[i+1])[0][1]:
                sss.append(max(sss))
            else:
                sss.append(np.cov(ss[i], ss[i+1])[0][1])
    sss = np.array(sss)
    big_sss.append(sss)

    pp = b.accuracy(weight='weighted')

    cp = a.get_iter_cumul_prediction()
    pppp = list()
    for i in cp:
        bb = scoring_model(y_ls, i)
        pppp.append(bb.accuracy(weight='weighted'))
    pppp = np.array(pppp)
    big_pppp.append(pppp)

big_sss = np.array(big_sss)
big_pppp = np.array(big_pppp)

big_sss_ = big_sss/20
big_pppp_ = big_pppp/20


# Draw a graph
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_ylabel('absolute cov', color=color)
ax1.plot(list(map(abs, sss)), color=color, label='cov')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('prediction acc', color=color)  # we already handled the x-label with ax1
ax2.plot(pppp, color=color, label='prediction_acc')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


plt.plot(pppp, color='b', label='prediction_acc')