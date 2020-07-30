from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import pandas as pd
import numpy as np
import timeit
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

start = timeit.default_timer()
WD = pd.read_csv(r'D:\Data\Grad\test_Work_Data_w_lag.csv')
WD = WD.set_index(WD.columns[0])
WD.index = pd.to_datetime(WD.index)

WD1 = WD[WD.columns[0:4]].fillna(method = 'ffill')
WD2 = WD[WD.columns[4:1078]].interpolate(method = 'time')
WD = pd.concat([WD1, WD2], axis = 1)

X_col = list(WD.columns)
X_col.remove('kor class') # X use all of the columns from the WD minus 'global crisis'column which is y

y = WD['kor class']
X = WD[[i for i in X_col]]

# Create trainig set, testing set.
X_train, X_test,y_train, y_test = train_test_split(X, y,
                                                   test_size = 0.3,
                                                   shuffle = False) # Split the data. For now no validation set
scale = StandardScaler() # Scale the data.
scale.fit(X_train)
X_train_s = pd.DataFrame(scale.transform(X_train), columns = list(X_train.columns))
date = pd.DataFrame(X_train.index)
X_train_s = pd.concat([date, X_train_s], axis = 1)
X_train_s = X_train_s.set_index('Unnamed: 0')

# To Boruta
X_train_b = X_train_s.to_numpy() # change dataframe to array / Boruta to acknowledge
y_train_b = y_train.to_numpy()
X_test_b = X_test.to_numpy()
y_test_b = y_test.to_numpy()

# define Random Forest
rf = RandomForestClassifier(n_jobs = -1, class_weight = 'balanced', max_depth = int(np.sqrt(len(X_train_s.columns))))
feature_select = BorutaPy(rf, n_estimators= 'auto', verbose = 2, random_state= 1)
feature_select.fit(X_train_b, y_train_b)
selected_var = feature_select.support_
selected_var_rank = feature_select.ranking_
X_filtered = feature_select.transform(X_train_b)
print(selected_var)
print(selected_var_rank)
print(X_filtered)
X_fil = pd.DataFrame(X_filtered, columns = X_train_s.columns[selected_var]) # pandas dataframe only with selected variables
date = pd.DataFrame(X_train.index)
X_fil = pd.concat([date, X_fil], axis = 1)
X_fil = X_fil.set_index('Unnamed: 0')
X_fil.to_csv(r"D:\Data\Grad\X_filtered.csv")