#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly

import  numpy as np
from sklearn import datasets
from sklearn.neural_network import MLPClassifier

iris = datasets.load_iris()
y = iris.target
x = iris.data

clf = MLPClassifier(solver='lbfgs',alpha=0.00001,hidden_layer_sizes=(5,2))
clf.fit(x,y)
res = clf.predict([[6.3,2.5,5.,1.9]])
print(res)

