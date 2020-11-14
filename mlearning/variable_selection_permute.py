from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

import pandas as pd


dat = pd.read_pickle(r'D:\Data\Grad\fwd.pkl')

# Separate y and X data
y = dat[['korclf1']].shift(-1)[:2279]

x_col = list(dat.columns)
x_col.remove('korclf1')
X = dat[x_col][:2279]

# 3 years of training, predict 30 days.
train_size = 365 * 3
trs = 0.002

# Get a sense of the threshold
clf = RandomForestClassifier(n_estimators=1000, n_jobs=1)
clf.fit(X[:train_size], y[:train_size])
f_imp = list()
for feats in zip(X.columns, clf.feature_importances_):
    f_imp.append(feats)

f_imp = sorted(f_imp, key=lambda x: x[1])

sel = SelectFromModel(RandomForestClassifier(n_estimators=1000), threshold=trs)
sel.fit(X[:train_size], y[:train_size])
sel_var = X.columns[(sel.get_support())]  # selected features
X_sel = X[sel_var]

X_sel.to_pickle(r'D:\Data\Grad\X_selected.pkl')
y.to_pickle(r'D:\Data\Grad\y_selected.pkl')