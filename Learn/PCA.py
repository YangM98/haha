#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import tree, preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC

train_X = np.array(np.loadtxt(r"C:\Users\Administrator\Desktop\No4\训练样本\train.txt",delimiter=','))
test_X = np.array(np.loadtxt(r"C:\Users\Administrator\Desktop\No4\测试样本\test.txt",delimiter=','))
train_y = np.loadtxt(r"C:\Users\Administrator\Desktop\No4\训练样本\train_y.txt",dtype=str)
test_y = np.loadtxt(r"C:\Users\Administrator\Desktop\No4\测试样本\test_y.txt",dtype=str)
train_y = np.array(train_y)
print(train_y.shape,test_y.shape)
total_X = np.vstack((train_X,test_X))
min_max_scaler = preprocessing.MinMaxScaler()
X_total_minmax = min_max_scaler.fit_transform(total_X)
print(X_total_minmax.shape)
a=[]
b=[]

'''
pca = PCA(n_components=2000)
main_v = pca.fit_transform(total_X)
print(main_v.shape)

#得到pca降维处理后的训练测试数据
train_size = train_X.shape[0]
Pca_train = main_v[0:train_size,:]
Pca_test = main_v[train_size:,:]

print(Pca_train.shape,Pca_test.shape)

#使用决策树
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Pca_train,train_y)
pre_y = clf.predict(Pca_test)
print(pre_y.shape)
t= [ i for i in range(1,1343,1)]
plt.figure()
plt.scatter(t,test_y,color='b',  label=u'真实数据')
plt.scatter(t, pre_y,color='r',  label=u'预测数据')
plt.show()
'''


for n in range(10,2100,40):
    pca = PCA(n_components=n)
    main_v = pca.fit_transform(X_total_minmax)
    print(main_v.shape)

    #得到pca降维处理后的训练测试数据
    train_size = train_X.shape[0]
    Pca_train = main_v[0:train_size,:]
    Pca_test = main_v[train_size:,:]

    #使用决策树
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(Pca_train,train_y)
    score = clf.score(Pca_test,test_y)
    pre = clf.predict(Pca_test)
    #混淆矩阵
    conf = confusion_matrix(test_y,pre)
    print(conf)
    a.append(n)
    b.append(score)
    #print(n+"维状态下准确率为:")
    print(score)


    #使用svm
    svc = SVC(n_neihhbors=4)
    svc.fit(Pca_train,train_y)
    predict = svc.predict(Pca_test)
    print("accoury_score:",accuracy_score(predict,test_y))



plt.figure()
plt.plot(a,b,'r')
plt.xlabel("维度")
plt.ylabel("准确率")
plt.show()








