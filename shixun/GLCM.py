#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import numpy as np
from skimage.feature import hog, greycomatrix, greycoprops
from skimage import io
from PIL import Image
import numpy as np
from keras.utils import np_utils
from sklearn import preprocessing, tree, metrics
import matplotlib.pyplot as  plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import cv2

def train_glcm():
    print("读取数据中.........")
    train_X=np.load('C:\\Users\\Administrator\\Desktop\\No4\\train.npy')
    test_X=np.load('C:\\Users\\Administrator\\Desktop\\No4\\test.npy')
    train_y = np.loadtxt(r"C:\\Users\Administrator\\Desktop\\No4\训练样本\train_y.txt",dtype=str)
    test_y = np.loadtxt(r"C:\\Users\\Administrator\Desktop\\No4\\测试样本\test_y.txt",dtype=str)
    print("读取成功........")
    print(train_X.shape,test_X.shape)
    train_y = train_y.reshape(train_y.shape[0],-1)
    test_y = test_y.reshape(test_y.shape[0],-1)
    #得到整个数据集
    data_X = np.vstack((train_X,test_X))
    data_y = np.vstack((train_y,test_y))
    #数据规范化
    #data_X = int(data_X.astype('float32') / 255.)
    #print(data_X.shape,data_y.shape)
    glcm_data=[]
    for img in data_X:
        aa = []
        glcm = greycomatrix(img, [1,5], [0,np.pi/2,np.pi * 3 / 4], levels=256,symmetric=True, normed=True)
        for prop in {'contrast', 'dissimilarity',
                         'homogeneity', 'energy', 'correlation', 'ASM'}:
            temp = greycoprops(glcm, prop)
            # temp=np.array(temp).reshape(-1)
            #print(prop,temp)
            aa.append(temp)
        #print('------')
        aa = np.array(aa).flatten()
        #aa = aa.reshape(1,-1)
        #print(aa.shape)
        glcm_data.append(aa)

    glcm_data = np.array(glcm_data)
    print(glcm_data.shape)
    X_train, X_test, y_train, y_test = train_test_split(glcm_data,data_y, test_size=0.4, random_state=42)
    #print(X_train.shape,X_test.shape)
    #print(y_train,y_test)
    # y值one-hot
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_Train = encoder.transform(y_train)
    y_train_hot = np_utils.to_categorical(y_Train,num_classes=4)
    encoder.fit(y_test)
    y_Test = encoder.transform(y_test)
    y_test_hot = np_utils.to_categorical(y_Test,num_classes=4)

    #使用决策树
    print("Training---------- 决策树")
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train,y_train_hot)
    clf_pre = clf.predict(X_test)
    score = metrics.accuracy_score(y_test_hot,clf_pre)
    rec = classification_report(y_test_hot,clf_pre)
    #nf = confusion_matrix(y_test, clf_pre)
    print(score)
    print(rec)
    #print(nf)
    #逻辑回归
    print("Training---------- 逻辑回归")
    logist = LogisticRegression()
    logist.fit(X_train,y_train)#不支持one-hot
    logist_pre = logist.predict(X_test)
    logist_score=metrics.accuracy_score(y_test,logist_pre)
    logist_rec = classification_report(y_test,logist_pre)
    conf = confusion_matrix(y_test, logist_pre)
    #logist_s = logist.score(X_test,y_test_hot)
    print(logist_score)
    print(logist_rec)
    print(conf)

    #支持向量机
    print("Training---------- SVM")
    svc = SVC()
    svc.fit(X_train,y_train)
    predict = svc.predict(X_test)
    svc_score = metrics.accuracy_score(y_test, predict)
    svc_rec = classification_report(y_test,predict)
    onf = confusion_matrix(y_test, predict)
    print(svc_score)
    print(svc_rec)
    print(onf)
    #使用BP网络
    print("Training---------- BP")
    bp_c = MLPClassifier(solver='lbfgs', alpha=0.00001, hidden_layer_sizes=(5, 5, 4), activation='relu')
    bp_c.fit(X_train,y_train)
    bp_pre = bp_c.predict(X_test)
    bp_score=metrics.accuracy_score(y_test,bp_pre)
    bp_rec = classification_report(y_test,bp_pre)
    print(bp_score)
    print(bp_rec)

    plt.figure()
    namelist=['tree','BP','logi','SVM']
    numlist = [score,bp_score,logist_score,svc_score]
    plt.bar(range(len(namelist)), numlist, color='rgby')
    index=[0,1,2,3,]
    plt.xticks(index, namelist)
    plt.show()

def glcm():
    img = cv2.cvtColor(cv2.imread(r"C:\Users\Administrator\Desktop\0.JPG"), cv2.COLOR_BGR2GRAY)
    print(img.shape)
    glcm = greycomatrix(img, [1,5,8], [0, np.pi * 1/4,np.pi/2,np.pi * 3 / 4], levels=256,symmetric=True, normed=True)
    for prop in {'contrast', 'dissimilarity',
                     'homogeneity', 'energy', 'correlation', 'ASM'}:
            temp = greycoprops(glcm, prop)
            # temp=np.array(temp).reshape(-1)
            print(prop, temp)

if __name__=='__main__':
    #glcm()
    train_glcm()