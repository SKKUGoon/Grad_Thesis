from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np
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

# Variable Selection using LASSO
selection = SelectFromModel(LogisticRegression(C=0.1, penalty='l1')) #penalty is L1, which is LASSO
selection.fit(scale.transform(X_train), y_train)
selected_var = list(selection.get_support())
selected_var = X_train.columns[selected_var]

# Should include Borute algorithm - More on that later
# https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a
# Create Boruta. Find Different more sophisticated Dataset from google finance api.
X_train_s = pd.DataFrame(scale.transform(X_train), columns = list(X_train.columns))
X_train_Shadow = X_train_s.apply(np.random.permutation)
X_train_Shadow.columns = ['shadow_' + i for i in X.columns]
X_train_boruta = pd.concat([X_train_s, X_train_Shadow], axis = 1)

forest = RandomForestRegressor(max_depth = 5, random_state = 42)
forest.fit(X_train_boruta, y_train)
feat_imp_X = forest.feature_importances_[:len(X_train_s.columns)]
feat_imp_shadow = forest.feature_importances_[len(X_train_s.columns):]
hits_single = feat_imp_X > feat_imp_shadow.max()

hits = np.zeros((len(X.columns))) # initialize hits counter
# repeat 20 times
for iter_ in range(100):
   np.random.seed(iter_)
   X_train_Shadow = X_train_s.apply(np.random.permutation) # make X_shadow by randomly permuting each column of X
   X_train_boruta = pd.concat([X_train_s, X_train_Shadow], axis = 1)
   forest = RandomForestRegressor(max_depth = 5, random_state = 42) # fit a random forest (suggested max_depth between 3 and 7)
   forest.fit(X_train_boruta, y_train)

   feat_imp_X = forest.feature_importances_[:len(X_train_s.columns)] # store feature importance
   feat_imp_shadow = forest.feature_importances_[len(X_train_s.columns):]
   # compute hits for this trial and add to counter
   hits += (feat_imp_X > feat_imp_shadow.max())
   print('phase' + str(iter_)) # to show where you at

selected_col = []
for i in range(len(hits)):
    if hits[i] > 0:
        selected_col.append(i)
    else:
        pass
for i in selected_col:
    print(X_train_s.columns[i])

stop = timeit.default_timer()
print('Time: ', stop - start) # time my script