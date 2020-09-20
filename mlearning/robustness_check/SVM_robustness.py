import pandas as pd
import timeit

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm

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


# Support Vector Machine
# sklearn specifies their kernel as np.exp(-gamma || x-x' || **2)
# gamma = 1/(2sigma**2)

def mysvc(c, sigma, k='rbf'):
    g = 1 / (2 * sigma**2)
    svm_clf = svm.SVC(C = c, gamma=g, kernel=k, probability=True)
    return svm_clf

boostadd = list()
sens, spec, gm, acc = dict(), dict(), dict(), dict()

for c_val in [1, 50, 100]:
    sens_sigma, spec_sigma, gm_sigma, acc_sigma = list(), list(), list(), list()
    for sigma_val in range(1, 20):
        m = mysvc(c_val, sigma_val)
        m.fit(x_train_s, y_train)
        pred = m.predict(x_test_s)

        # Metrics
        t = scoring_model(y_test, pred)
        sens_sigma.append(t.sensitivity())
        spec_sigma.append(t.specificity())
        gm_sigma.append(t.gmean())
        acc_sigma.append(t.accuracy(weight='weighted'))

        # Boosting element
        if t.accuracy() >= 0.5:
            cdt = [c_val, sigma_val]
            boostadd.append(cdt)
        print(f'sigma value {sigma_val} checked')
        print(cross_val_score(m, x_train_s, y_train, cv=5, n_jobs=-1))

    sens[f'C = {c_val}'] = sens_sigma
    spec[f'C = {c_val}'] = spec_sigma
    gm[f'C = {c_val}'] = gm_sigma
    acc[f'C = {c_val}'] = acc_sigma
    print(f'C value {c_val} checked')

print(sens, '\n', spec, '\n', gm, '\n', acc)
print(boostadd)

# Graph
x_ = list(range(1,20))
tlt = [1, 50, 100]
for i in range(3):
    plt.plot(x_, gm[list(gm.keys())[i]], color='b', linestyle='dashed', label='gmean')
    plt.plot(x_, acc[list(acc.keys())[i]], color='b', label='accuracy')
    plt.xlabel('sigma')
    plt.title(f'C={tlt[i]}')
    plt.legend()
    plt.show()

