# !usr/bin/python
# -*- coding: utf-8 -*-
# author:ly

import numpy as np
import pandas as pd
from keras import Input, Model, Sequential
from keras.layers import Dense, BatchNormalization, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Convolution2D, \
    UpSampling2D
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
x_train = train_X.astype('float32') / 255.
x_test = test_X.astype('float32') / 255.
x_train = x_train.reshape(train_X.shape[0],1,128,128)
x_test = test_X.reshape(test_X.shape[0],1,128,128)
#加入高斯噪声 像素值降到（0,1）
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

input_img = Input(shape=(1, 128, 128))

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (32, 7, 7)

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
encoder = Model(input_img,)








