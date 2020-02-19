#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn import preprocessing, metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

print("读取数据中.........")
train_X=np.load('C:\\Users\\Administrator\\Desktop\\No4\\一维npy\\train.npy')
test_X=np.load('C:\\Users\\Administrator\\Desktop\\No4\\一维npy\\test.npy')
train_y = np.loadtxt(r"C:\Users\Administrator\Desktop\No4\一维npy\train_y.txt",dtype=str)
test_y = np.loadtxt(r"C:\Users\Administrator\Desktop\No4\一维npy\test_y.txt",dtype=str)
print("读取成功........")

total_X = np.vstack((train_X,test_X))
print(total_X.shape)
min_max_scaler = preprocessing.MinMaxScaler()
X_total = min_max_scaler.fit_transform(total_X)

encoder = LabelEncoder()
encoder.fit(train_y)
y_train = encoder.transform(train_y)
y_train = np_utils.to_categorical(y_train,num_classes=4)
encoder.fit(test_y)
y_test = encoder.transform(test_y)
y_test = np_utils.to_categorical(y_test,num_classes=4)

a,score=[],[]
for n in range(10,2010,50):
    a.append(n)
    print(n)
    print("----------------------")
    pca = PCA(n_components=n)
    main_v = pca.fit_transform(X_total)
    #print(main_v)
    train_size = train_X.shape[0]
    Pca_train = main_v[0:train_size, :]
    Pca_test = main_v[train_size:, :]

    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                             algorithm='SAMME',n_estimators=400,learning_rate=0.8)
    bdt.fit(Pca_train,train_y)
    pre=bdt.predict(Pca_test)
    acc=metrics.accuracy_score(test_y, pre)
    print(acc)
    score.append(acc)
plt.figure()
plt.plot(a,score)
plt.show()





