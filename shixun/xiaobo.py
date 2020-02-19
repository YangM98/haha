
#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly

import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn import preprocessing, tree, metrics
import matplotlib.pyplot as  plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

def test():
    img = cv2.imread(r"C:\Users\Administrator\Desktop\0.JPG")
    # 将多通道图像变为单通道图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    plt.figure('二维小波一级变换')
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    print(np.mean(cA),np.std(cA),np.linalg.norm(cA,ord=1),np.linalg.norm(cA,ord=2))
    plt.subplot(221), plt.imshow(cA, 'gray'), plt.title("A")
    plt.subplot(222), plt.imshow(cH, 'gray'), plt.title("H")
    plt.subplot(223), plt.imshow(cV, 'gray'), plt.title("V")
    plt.subplot(224), plt.imshow(cD, 'gray'), plt.title("D")
    plt.show()
def get_fea(x):
    mean = np.mean(x)
    std = np.std(x)
    l1 = np.linalg.norm(x, ord=1)
    l2 = np.linalg.norm(x, ord=2)
    fea = [mean,std,l1,l2]
    return fea
def train_xiaobo():
    print("读取数据中.........")
    train_X = np.load('C:\\Users\\Administrator\\Desktop\\No4\\train.npy')
    test_X = np.load('C:\\Users\\Administrator\\Desktop\\No4\\test.npy')
    train_y = np.loadtxt(r"C:\\Users\Administrator\\Desktop\\No4\训练样本\train_y.txt", dtype=str)
    test_y = np.loadtxt(r"C:\\Users\\Administrator\Desktop\\No4\\测试样本\test_y.txt", dtype=str)
    print("读取成功........")
    print(train_X.shape, test_X.shape)
    train_y = train_y.reshape(train_y.shape[0], -1)
    test_y = test_y.reshape(test_y.shape[0], -1)
    # 得到整个数据集
    data_X = np.vstack((train_X, test_X))
    data_y = np.vstack((train_y, test_y))
    # 数据规范化
    data_X = data_X.astype('float32') / 255.
    print(data_X.shape, data_y.shape)
    bo_data= []
    # 提取小波特征保存数组
    for im in data_X:
        coeffs = pywt.dwt2(im, 'haar')
        cA, (cH, cV, cD) = coeffs
        a = get_fea(cA)
        h = get_fea(cH)
        v = get_fea(cV)
        d = get_fea(cD)
        da = [a,h,v,d]
        da = np.array(da).flatten()
        bo_data.append(da)
    bo_data = np.array(bo_data)
    print(bo_data.shape)
    X_train, X_test, y_train, y_test = train_test_split(bo_data, data_y, test_size=0.4, random_state=42)
    # print(X_train.shape,X_test.shape)
    # print(y_train,y_test)
    # y值one-hot
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_Train = encoder.transform(y_train)
    y_train_hot = np_utils.to_categorical(y_Train, num_classes=4)
    encoder.fit(y_test)
    y_Test = encoder.transform(y_test)
    y_test_hot = np_utils.to_categorical(y_Test, num_classes=4)

    # 使用决策树
    print("Training---------- 决策树")
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train_hot)
    clf_pre = clf.predict(X_test)
    score = metrics.accuracy_score(y_test_hot, clf_pre)
    rec = classification_report(y_test_hot, clf_pre)
    print(score)
    print(rec)
    # 逻辑回归
    print("Training---------- 逻辑回归")
    logist = LogisticRegression()
    logist.fit(X_train, y_train)  # 不支持one-hot
    logist_pre = logist.predict(X_test)
    logist_score = metrics.accuracy_score(y_test, logist_pre)
    logist_rec = classification_report(y_test, logist_pre)
    conf = confusion_matrix(y_test, logist_pre)
    # logist_s = logist.score(X_test,y_test_hot)
    print(logist_score)
    print(logist_rec)
    print(conf)

    # 支持向量机
    print("Training---------- SVM")
    svc = SVC()
    svc.fit(X_train, y_train)
    predict = svc.predict(X_test)
    svc_score = metrics.accuracy_score(y_test, predict)
    svc_rec = classification_report(y_test, predict)
    onf = confusion_matrix(y_test, predict)
    print(svc_score)
    print(svc_rec)
    print(onf)
    # 使用BP网络
    print("Training---------- BP")
    bp_c = MLPClassifier(solver='lbfgs', alpha=0.00001, hidden_layer_sizes=(5, 5, 4), activation='relu')
    bp_c.fit(X_train, y_train)
    bp_pre = bp_c.predict(X_test)
    bp_score = metrics.accuracy_score(y_test, bp_pre)
    bp_rec = classification_report(y_test, bp_pre)
    nf = confusion_matrix(y_test, bp_pre)
    print(bp_score)
    print(bp_rec)
    print(nf)
    plt.figure()
    namelist = ['tree', 'BP', 'logi', 'SVM']
    numlist = [score, bp_score, logist_score, svc_score]
    plt.bar(range(len(namelist)), numlist, color='rgby')
    index = [0, 1, 2, 3, ]
    plt.xticks(index, namelist)
    plt.show()

if __name__=='__main__':
    test()
    train_xiaobo()