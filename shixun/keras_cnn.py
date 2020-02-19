 #!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import numpy as np
import pandas as pd
from keras import Input, Model, Sequential
from keras.layers import Dense, BatchNormalization, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.decomposition import PCA
from sklearn import tree, preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

print("读取数据中.........")
train_X=np.load('C:\\Users\\Administrator\\Desktop\\No4\\train.npy')
test_X=np.load('C:\\Users\\Administrator\\Desktop\\No4\\test.npy')
train_y = np.loadtxt(r"C:\Users\Administrator\Desktop\No4\训练样本\train_y.txt",dtype=str)
test_y = np.loadtxt(r"C:\Users\Administrator\Desktop\No4\测试样本\test_y.txt",dtype=str)
print("读取成功........")
print(train_X.shape,test_X.shape)
train_y = np.array(train_y)
print(train_y.shape,test_y.shape)
x_train = train_X.reshape(train_X.shape[0],128,128,1)
x_test = test_X.reshape(test_X.shape[0],128,128,1)

#数字图像特征标准化
x_train = x_train/255
x_test = x_test/255
#total_X = np.vstack((train_X,test_X))
print(x_train.shape,x_test.shape)
#min_max_scaler = preprocessing.MinMaxScaler()
#X_total_minmax = min_max_scaler.fit_transform(total_X)
#print(X_total_minmax)

# y值one-hot
encoder = LabelEncoder()
encoder.fit(train_y)
y_train = encoder.transform(train_y)
y_train = np_utils.to_categorical(y_train,num_classes=4)
encoder.fit(test_y)
y_test = encoder.transform(test_y)
y_test = np_utils.to_categorical(y_test,num_classes=4)
print(y_train,y_test.shape)

#构建CNN模型
model = Sequential()
#第一层卷积
model.add(Conv2D(filters=16,
                 kernel_size=(3,3),
                 padding='same',
                 input_shape=(128,128,1),

                 activation='relu'))
#最大池化层将（128,128）变成（64,64）
model.add(MaxPooling2D(pool_size=(2,2)))



#第二层卷积
model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                 padding='same',

                 activation='relu'))
#第二层池化(64,64)(32,32)
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.3))
#第三层卷积
model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                 padding='same',
                 activation='relu'))
#第三层池化(32,32)（16,16）

model.add(MaxPooling2D(pool_size=(2,2)))

#第四层卷积
model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 padding='same',

                 activation='relu'))
#第四层池化（16,16）(8,8)
model.add(MaxPooling2D(pool_size=(2,2)))

#Dropout
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(2048,activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(4,activation='softmax'))
#输出模型
print(model.summary())

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print("\n Testing -------- ")
model.fit(x_train,y_train,epochs=40,batch_size=32,verbose=1)
print("\n Testing -------- ")
loss, accuracy = model.evaluate(x_test, y_test)
print("test loss:", loss)
print("test accuracy:", accuracy)
#predict = model.predict(x_test)
#conf = confusion_matrix(test_y,predict)
#print(conf)