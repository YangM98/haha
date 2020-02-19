#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn import preprocessing, metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import tree

print("读取数据中.........")
train_X=np.load('C:\\Users\\Administrator\\Desktop\\No4\\一维npy\\train.npy')
test_X=np.load('C:\\Users\\Administrator\\Desktop\\No4\\一维npy\\test.npy')
train_y = np.loadtxt(r"C:\Users\Administrator\Desktop\No4\一维npy\train_y.txt",dtype=str)
test_y = np.loadtxt(r"C:\Users\Administrator\Desktop\No4\一维npy\test_y.txt",dtype=str)
print("读取成功........")
#
total_X = np.vstack((train_X,test_X))
print(total_X.shape)
min_max_scaler = preprocessing.MinMaxScaler()
X_total = min_max_scaler.fit_transform(total_X)
'''
# y值one-hot
encoder = LabelEncoder()
encoder.fit(train_y)
y_train = encoder.transform(train_y)
y_train = np_utils.to_categorical(y_train,num_classes=4)
encoder.fit(test_y)
y_test = encoder.transform(test_y)
y_test = np_utils.to_categorical(y_test,num_classes=4)
'''
w,cll,bp,svm,logi=[],[],[],[],[]

for n in range(50,510,100):
    w.append(n)
    pca = PCA(n_components=n)
    main_v = pca.fit_transform(X_total)
    print(main_v)
    #np.savetxt("C:\\Users\\Administrator\\Desktop\\main.txt", main_v, fmt='%0.4f', delimiter=',', newline='\n')

    #得到pca降维处理后的训练测试数据
    train_size = train_X.shape[0]
    Pca_train = main_v[0:train_size,:]
    Pca_test = main_v[train_size:,:]
    '''
    #使用决策树
    print("Training---------- 决策树")
    clf = tree.DecisionTreeClassifier()
    clf.fit(Pca_train,train_y)
    score = clf.score(Pca_test,test_y)
    cll.append(score)

    print(score)

    #逻辑回归
    print("Training---------- 逻辑回归")
    logist = LogisticRegression()
    logist.fit(Pca_train,train_y)
    #logist_score = logist.score(Pca_test,y_test)
    logist_pre = logist.predict(Pca_test)
    logist_f=logist_pre==test_y
    logist_t=test_y.shape[0]-0.5*np.count_nonzero(logist_f == False)
    logist_score=logist_t/test_y.shape[0]
    logi.append(logist_score)
    print(logist_score)
    '''
    #使用BP网络
    print("Training---------- BP")
    bp_c = MLPClassifier(solver='lbfgs', alpha=0.00001, hidden_layer_sizes=(5, 5, 4), activation='relu',max_iter=1000)
    bp_c.fit(Pca_train,train_y)
    bp_pre = bp_c.predict(Pca_test)
    #bp_score = bp_c.score(Pca_test,y_test,sample_weight=None)
    bp_f=bp_pre==test_y
    bp_t=test_y.shape[0]-0.5*np.count_nonzero(bp_f == False)
    bp_score=bp_t/test_y.shape[0]
    bp.append(bp_score)
    print( bp_score)
    #支持向量机
    print("Training---------- SVM")
    svc = SVC()
    svc.fit(Pca_train,train_y)
    predict = svc.predict(Pca_test)
    svc_score = metrics.accuracy_score(test_y, predict)
    svm.append(svc_score)
    print(svc_score)

plt.figure()
#plt.plot(w,cll,'p-',label=u"tree")
plt.plot(w,bp,'b-',label=u"BP")
#plt.plot(w,logi,'y-',label=u"Logist")
plt.plot(w,svm,'k-',label=u"SVM")
plt.legend()
plt.rcParams['font.family'] = 'SimHei'  #黑体
plt.xlabel(u"维度")
plt.ylabel(u"准确率")
plt.show()