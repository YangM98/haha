#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import numpy as np
from keras import backend as K, regularizers
from keras import Input, Model, Sequential
from keras.layers import Dense, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.decomposition import PCA
from sklearn import tree, preprocessing, metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

print("读取数据中.........")
train_X=np.load('C:\\Users\\Administrator\\Desktop\\No4\\一维npy\\train.npy')
test_X=np.load('C:\\Users\\Administrator\\Desktop\\No4\\一维npy\\test.npy')
train_y = np.loadtxt(r"C:\Users\Administrator\Desktop\No4\一维npy\train_y.txt",dtype=str)
test_y = np.loadtxt(r"C:\Users\Administrator\Desktop\No4\一维npy\test_y.txt",dtype=str)
print("读取成功........")
train_y = np.array(train_y)
print(train_y.shape,test_y.shape)
total_X = np.vstack((train_X,test_X))
print(total_X.shape)
min_max_scaler = preprocessing.MinMaxScaler()
X_total_minmax = min_max_scaler.fit_transform(total_X)

train_size = train_X.shape[0]
X_train = X_total_minmax[0:train_size,:]
X_test = X_total_minmax[train_size:,:]
#print(X_total_minmax)

# y值one-hot
encoder = LabelEncoder()
encoder.fit(train_y)
y_train = encoder.transform(train_y)
y_train = np_utils.to_categorical(y_train,num_classes=4)
encoder.fit(test_y)
y_test = encoder.transform(test_y)
y_test = np_utils.to_categorical(y_test,num_classes=4)

encoding_dam = 32
input_img=Input(shape=(16384,))

#建立编码层
encoded = Dense(1000,activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_img)
encoded = Dense(256 ,activation='relu')(encoded)
encoded = Dense(64,activation='relu')(encoded)
encoder_output = Dense(encoding_dam,name='mid')(encoded)

#建立解码层
decoded = Dense(64,activation='relu')(encoder_output)
decoded = Dense(256,activation='relu')(decoded)
decoded = Dense(1000,activation='relu')(decoded)
decoded = Dense(16384,activation='tanh')(decoded)

# 构建自编码模型
autoencoder = Model(inputs=input_img, outputs=decoded)

# 构建编码模型
encoder = Model(inputs=input_img, outputs=encoder_output)

#编译模型
autoencoder.compile(optimizer='adam',loss='mse')
autoencoder.summary()
# training
autoencoder.fit(X_train, X_train, epochs=20, batch_size=256, shuffle=True)

auto_trainX=encoder.predict(X_train)
auto_testX = encoder.predict(X_test)
print(auto_trainX.shape)
print(auto_testX.shape)

#使用决策树
print("Training---------- 决策树")
clf = tree.DecisionTreeClassifier()
clf.fit(auto_trainX,train_y)
clf_pre = clf.predict(auto_testX)
score = metrics.accuracy_score(test_y,clf_pre)
rec = classification_report(test_y,clf_pre)
print(score)
print(rec)

#使用BP网络
print("Training---------- BP")
bp_c = MLPClassifier(solver='lbfgs', alpha=0.00001, hidden_layer_sizes=(5, 5, 4), activation='relu')
bp_c.fit(auto_trainX,train_y)
score_bp  =bp_c.score(auto_testX,test_y)
bp_pre = bp_c.predict(auto_testX)
#bp_score = bp_c.score(Pca_test,y_test,sample_weight=None)
bp_f=bp_pre==test_y
bp_t=test_y.shape[0]-0.5*np.count_nonzero(bp_f == False)
bp_score=bp_t/test_y.shape[0]
print(score_bp)
print( bp_score)
#逻辑回归
print("Training---------- 逻辑回归")
logist = LogisticRegression()
logist.fit(auto_trainX,train_y)
#logist_score_benshen = logist.score(auto_testX,y_test)
logist_pre = logist.predict(auto_testX)
logist_f=logist_pre==test_y
logist_t=test_y.shape[0]-0.5*np.count_nonzero(logist_f == False)
logist_score=logist_t/test_y.shape[0]

print(logist_score)

#支持向量机
print("Training---------- SVM")
svc = SVC()
svc.fit(auto_trainX,train_y)
predict = svc.predict(auto_testX)
svc_score = metrics.accuracy_score(test_y, predict)
print(svc_score)


plt.figure()
namelist=['Tree','BP','logi','SVM']
numlist = [score,bp_score,logist_score,svc_score]
plt.bar(range(len(namelist)), numlist, color='rgby')
index=[0,1,2,3,]
plt.xticks(index, namelist)
plt.show()