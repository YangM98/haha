#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import column_or_1d
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

print("读取数据中.........")
data_X=np.load('C:\\Users\\Administrator\\Desktop\\128No4\\data_X.npy')
data_Y = np.loadtxt(r"C:\Users\Administrator\Desktop\128No4\data_y.txt",dtype=str)
print("读取成功........")
print(data_X.shape,data_Y.shape)
data_y = column_or_1d(data_Y, warn=True)
#data_y = data_Y.reshape(data_Y.shape[0], -1)

# 数据规范化
min_max_scaler = preprocessing.MinMaxScaler()
X_std = min_max_scaler.fit_transform(data_X)

# 得到pca降维处理后的训练测试数据
for n in range(10,2100,40):
    pca = PCA(n_components=n)
    main_v = pca.fit_transform(X_std)
    print(main_v.shape)
    X_train, X_test, y_train, y_test = train_test_split(main_v, data_y, test_size=0.4, random_state=42)

    # 支持向量机
    print("Training---------- SVM")
    svc = SVC()
    svc.fit(X_train, y_train)
    predict = svc.predict(X_test)
    svc_score = metrics.accuracy_score(y_test, predict)
    svc_rec = classification_report(y_test, predict)
    print(svc_score)
    print(svc_rec)
    # AdaBoost
    print("Training--------AdaBoost")
    clf = AdaBoostClassifier(n_estimators=100)
    scores = cross_val_score(clf, main_v, data_Y)
    print('AdaBoost准确率：', scores.mean())
    # Bagging
    print("Training--------Bagging")
    bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
    scores = cross_val_score(bagging, main_v, data_Y)
    print('Bagging准确率：',scores.mean())
