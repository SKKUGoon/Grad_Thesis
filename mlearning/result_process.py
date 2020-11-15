import pickle
import pandas as pd

# Boost with 3 decision tree, 5 random forest, 3 neural net, support vector machine
# 50 iteration
with open(r'D:\Data\Grad\res_sto_boost.pkl', 'rb') as file:
    res_boost = pickle.load(file)

with open(r'D:\Data\Grad\res_raw_sto_boost.pkl', 'rb') as file:
    raw_res_boost = pickle.load(file)

# Boost with 6 neural net
# 50 iteration
with open(r'D:\Data\Grad\res_sto_boost_nn.pkl', 'rb') as file:
    res_nn_boost = pickle.load(file)
with open(r'D:\Data\Grad\res_raw_sto_boost_nn.pkl', 'rb') as file:
    raw_res_nn_boost = pickle.load(file)

# Individual result
with open(r'D:\Data\Grad\res_individual.pkl', 'rb') as file:
    res_individual = pickle.load(file)



individual_acc = list()
individual_stats = list()
for key in res_individual:
    if 'acc' in key:
        individual_acc.append(res_individual[key])
    elif 'stats' in key:
        individual_stats.append(res_individual[key])

all = individual_acc + [res_boost]  # add boosting result

pd.DataFrame(all).to_csv(r'D:\Data\result.csv')
