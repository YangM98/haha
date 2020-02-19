#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller as ADF, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import AR,ARMA, ARIMA
#
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

data = pd.read_table(r"C:\Users\Administrator\Desktop\姑咱水化.txt",engine='python',names = ['time','value'],sep=' ',parse_dates=True)
plt.rcParams['font.sans-serif'] = ['SimHei']
#data['time'] = pd.to_datetime(data['time'])
data = data.set_index('time')
print(data.index)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)
reframed = series_to_supervised(scaled_data, 1, 1)
print(reframed)
value = reframed.values
rate = 0.8
lenth = len(data)
da_size = int(np.floor(rate*lenth))
train = value[:da_size,:]
test = value[da_size:,:]
print(train.shape,test.shape)

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
#将数据重构为符合lstm的数据格式
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape,  test_X.shape, test_y.shape)
#建立lstm模型
model = Sequential()
model.add(LSTM(50, activation='relu',input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# fit network
LSTM = model.fit(train_X, train_y,epochs=100, batch_size=32, verbose=2, shuffle=False)
# plot history
#plt.plot(LSTM['loss'], label='train')
#plt.plot(LSTM['val_loss'], label='valid')
test_predict = model.predict(test_X)
train_predict = model.predict(train_X)

# 反归一化
train_predict = scaler.inverse_transform(train_predict)
trainY = scaler.inverse_transform([train_y])
test_predict = scaler.inverse_transform(test_predict)
testY = scaler.inverse_transform([test_y])

plt.plot(data.values, c='b',label='true')
plt.plot([i for i in train_predict] + [x for x in test_predict], c='r',label='predict')

plt.legend()
plt.xlabel(u'时间')
plt.ylabel(u'氡的浓度')
plt.show()