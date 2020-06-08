from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import pandas as pd

import timeit
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
# Scale the Data
scale = StandardScaler() # Scale the data.
scale.fit(X_train)
X_train_s = pd.DataFrame(scale.transform(X_train), columns = list(X_train.columns))

# define Random Forest
for i in range(1,100):
    rf = RandomForestClassifier(n_jobs = -1, class_weight = 'balanced', max_depth = i)
    rf.fit(X_train_s, y_train)
    y_pred = rf.predict((X_test))
    print('max_depth : ' + str(i) + ' = ', metrics.accuracy_score(y_test, y_pred))

stop = timeit.default_timer()
print('Time: ', stop - start) # time my script