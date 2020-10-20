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
scale.fit(x_test)
x_test_s = pd.DataFrame(scale.transform(x_test), columns = list(x_test.columns))
date = pd.DataFrame(x_train.index)
date_t = pd.DataFrame(x_test.index)
x_train_s = pd.concat([date, x_train_s], axis = 1)
x_test_s = pd.concat([date_t, x_test_s], axis = 1) # X_train scaled and ready to go
x_train_s = x_train_s.set_index('Unnamed: 0')
x_test_s = x_test_s.set_index('Unnamed: 0') # X_test scaled and ready to go

# Model _ Random Forest
acc = {}

num_trees = 1000

for j in range(1,50,2):
    robust_acc = []
    for i in range(1):
        rf_clf = RandomForestClassifier(n_estimators=num_trees, max_depth=j, n_jobs=-1)
        rf_clf.fit(x_train_s, y_train)
        y_pred_rf = rf_clf.predict((x_test_s[x_train_s.columns]))
        acc_rf = scoring_model(y_test, y_pred_rf)

        robust_acc.append(acc_rf.accuracy(weight='weighted'))
    acc[f'max_depth = {j}'] = robust_acc
    print(f'max_depth {j} checked')



print(f'Final results are:{acc}')

# Graph drawing
x_ = list(range(1,50,2))

plt.plot(x_, list(acc.values()), color='b', label='accuracy')
plt.plot(x_, [0.5]*len(x_), color='b', linestyle='dashed', label='50% line')
plt.xlabel('maximum depth')
plt.title(f'Number of Trees : {num_trees}')
plt.legend()
plt.show()
