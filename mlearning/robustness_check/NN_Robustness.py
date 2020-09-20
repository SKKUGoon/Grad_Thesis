import numpy as np
import pandas as pd
import timeit
from typing import List

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout

from mlearning.scoringMethods import scoring_model

import warnings
import matplotlib.pyplot as plt


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
                                                    test_size = 0.3,
                                                    shuffle = True) # Split the data. For now no validation set
                                                                    # no random split. so shuffle = false
# Original Data
scale = StandardScaler() # Scale the data.
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


# Deep NN
class DNN(Sequential):
    def __init__(self, input: int, drop: float, h_output: List, num_layers: int, output=1, d_actf='relu'):
        """
        :param input: input dimension
        :param drop: dropout rate, less than zero
        :param h_output: list of hidden layer
        :param output: output layer. Default set to 1
        :param d_actf: activation function. Default set to ReLU
        """
        super().__init__()

        self.add(Dense(units=h_output[0], activation=d_actf, input_dim=input, name='Input_Layer'))  # first layer
        self.add(Dropout(drop))

        for i in range(num_layers):
            self.add(Dense(units=h_output[i], activation=d_actf, name=f'Hidden_{i+1}'))
            self.add(Dropout(drop))

        self.add(Dense(units=output, activation='tanh'))  # output layer
        self.compile(loss='binary_crossentropy', optimizer='adam')

def classid(prob: List, classtype='-1,1') -> List:
    cls = list()

    if classtype == '-1,1':
        for i in prob:
            if i > 0:
                cls.append(1)
            else:
                cls.append(-1)

    else:
        ...
    return cls


def dnn_(outp: List, layers, dor, acf: str, x_, y_):
    """
    :param outp: hidden layer input
    :param layers: number of layers
    :param acf: activation function, write in str
    :param x: training independent data
    :param y: training dependent data(label)
    """
    model = DNN(len(x_.columns), drop=dor, h_output=outp, num_layers=layers, d_actf=acf)
    dnn = model.fit(x_, y_, epochs=1, verbose=0)

    performance_test = model.predict(x_test_s)

    return model

def res_class(model: DNN, x):
    prd = model.predict(x)
    prd_class = classid(prd)
    return prd_class

def res_graph(model: DNN):
    m = model
    # Graph for overfitting
    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(m.history['loss'], 'y', label='train loss')

    acc_ax.plot(m.history['accuracy'], 'b', label='train acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    return plt.show()


def floatspace(start: float, finish: float):
    """
    generate listed float space.
    """

    und1 = len(str(start).split('.')[1])
    und2 = len(str(finish).split('.')[1])

    s = int(start * (10 ** und1))
    f = int(finish * (10 ** und2))
    fs = list(map(lambda num: num / 10 ** (und2), list(range(s, f))))

    return fs


op = [300, 300, 100, 100, 50, 50, 25, 10]
nnacc = dict()
for i in floatspace(0.31, 0.49):
    print(i)
    acclist = list()
    for j in range(10):
        t = dnn_(op, len(op), i, 'relu', x_train, y_train)
        pred = res_class(t, x_test)

        m = scoring_model(y_test, pred)
        acclist.append(m.accuracy())
    nnacc[f'dropout: {i}'] = acclist
    print(np.mean(acclist))