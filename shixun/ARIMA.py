#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller as ADF, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import AR,ARMA, ARIMA
data = pd.read_table(r"C:\Users\Administrator\Desktop\姑咱水化.txt",engine='python',sep=' ',names=['日期', '浓度'],index_col=u'日期')
plt.rcParams['font.sans-serif'] = ['SimHei']
print(data.index)
#data.plot()
#plt.show()
print(u'原始序列的ADF检验结果为：', ADF(data[u'浓度']))
D_data = data.diff().dropna()
D_data.columns=[u'浓度差分']
D_data.plot()
plt.show()
plot_acf(D_data).show()
plot_pacf(D_data).show()
print(u'差分序列的ADF检验结果为:',ADF(D_data[u'浓度差分']))
print(D_data)
lenth = len(D_data)
print(lenth)
rata = 0.8
train_size = int(np.floor(lenth*rata))
#print(train_size)

train = D_data[:train_size]
test = D_data[train_size:]
print(train[u'浓度差分'],test)
#adftest = adfuller(train, autolag='AIC')
print(train)
model = ARIMA(train,order=(1,1,0))
arima = model.fit()
pred = arima.predict()#预测
#mes =  mean_squared_error(train,pred)
plt.clf()
plt.plot(pred, label="pre" )
plt.plot(test, label = "true" )
plt.xlabel(u'日期')
plt.ylabel(u'差分浓度')
plt.show()
#print(mes)
