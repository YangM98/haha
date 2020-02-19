#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image/c1.png')  # 直接读为灰度图像
ret, th1 = cv2.threshold(img, 38, 255, cv2.THRESH_BINARY_INV)
#th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# 换行符号
#th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# 换行符号
plt.imshow(th1)
plt.show()

