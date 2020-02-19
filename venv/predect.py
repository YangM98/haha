#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt

def computeCost(X,y,theta):
    inner = np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))
    #result = np.sum((np.dot(X,theta)-y)**2)/(2*len(X))
    #return result


def gradientDescent(X, y, theta, alpha, epoch):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.flatten().shape[1])  # 参数theta的数量
    cost = np.zeros(epoch)  # 初始一个ndarray，包含每次epoch的cost
    m = X.shape[0]  # 样本数量m

    for i in range(epoch):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - (alpha / len(X) * np.sum(term))

        # temp = theta-(alpha/m)*(X*theta.T-y).T*X
        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost


def plotData(epoch, cost):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(epoch), cost2, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Epoch')
    plt.show()


def plotData(epoch, cost):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(epoch), cost2, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Epoch  loss')
    plt.show()

alpha = 0.01
epoch = 1000
path="C:\\Users\\Administrator\\Desktop\\ex1data2.txt"
data = pd.read_csv(path,names =['size','Bedrooms','price'])
data2=(data-data.mean())/data.std()
data2.insert(0,'Ones',1)

cols = data2.shape[1] #列数
X2 = data2.iloc[:,0:cols-1]# 取前cols-1列，即输出向量
y2 = data2.iloc[:,cols-1:cols]#取最后一列，即目标向量



X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)

theta2 = np.matrix(np.array([0,0,0]))


g2,cost2 = gradientDescent(X2,y2,theta2,alpha,epoch)
computeCost(X2,y2,g2),g2
plotData(epoch,cost2)
