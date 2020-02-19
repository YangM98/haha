#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import numpy as np
import pandas as pd
from keras.utils import np_utils
np.random.seed(10)
from keras.datasets import mnist
import matplotlib.pyplot as plt
(X_train_image,y_train_label),(X_test_image,y_test_label) = mnist.load_data()
print("train_data:",len(X_train_image))
print("test_data",len(X_test_image))

def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image,cmap='binary')
    plt.show()

plot_image(X_train_image[0])
