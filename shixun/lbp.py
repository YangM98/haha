#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import numpy as np
from keras.datasets import mnist
from sklearn import preprocessing,metrics
#读取mnist数据
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# settings for LBP
radius = 3
n_points = 8 * radius
def mnist_lbp():
    (X_train_image,y_train_label),\
    (X_test_image,y_test_label) = mnist.load_data()
    print(X_train_image.shape)
    train_X,test_X=[],[]
    for image in X_train_image:
        print('===========')
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(image, n_points, radius)
        im = lbp.flatten()
        train_X.append(im)
    train_X = np.array(train_X)
    print(train_X.shape)

    for img in X_test_image:
        lbp = local_binary_pattern(img, n_points, radius)
        # 统计图像的直方图
        #max_bins = int(lbp.max() + 1);
        # hist size:256
        #test_hist, bin_edge= np.histogram(lbp, bins=max_bins, range=(0, max_bins));
        img = lbp.flatten()
        test_X.append(img)
    test_X = np.array(test_X)
    #数据规范化
    total = np.vstack((train_X,test_X))
    Stand_scaler = preprocessing.StandardScaler()
    X_total = Stand_scaler.fit_transform(total)
    train_X = X_total[0:train_X.shape[0],:]
    test_X = X_total[train_X.shape[0]:,:]

    print(test_X)
    print(test_X.shape,train_X.shape)
    #支持向量机

    print("Training---------- SVM")
    svc = SVC()
    svc.fit(train_X,y_train_label)
    predict = svc.predict(test_X)
    svc_score = metrics.accuracy_score(y_test_label, predict)
    print(svc_score)

    print('Training-------Bp')
    bp_c = MLPClassifier(solver='lbfgs', alpha=0.00001, hidden_layer_sizes=(5,10,10), activation='relu',max_iter=1000)
    bp_c.fit(train_X,y_train_label)
    bp_pre = bp_c.predict(test_X)
    score = metrics.accuracy_score(y_test_label, bp_pre)
    print(score)

def lbp():
    # 读取图像
    image = cv2.imread('image/0.jpg')

    #显示到plt中，需要从BGR转化到RGB，若是cv2.imshow(win_name, image)，则不需要转化
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(131)
    plt.imshow(image1),plt.title("image")

    # 转换为灰度图显示
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.subplot(132)
    plt.imshow(image, cmap='gray'),plt.title("gray_img")

    # 处理
    lbp = local_binary_pattern(image, n_points, radius)

    plt.subplot(133)
    plt.imshow(lbp, cmap='gray'),plt.title("lbp_img")
    plt.show()

if __name__ =='__main__':
    lbp()
    mnist_lbp()