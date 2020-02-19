#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import cv2
import matplotlib.pyplot as plt
im = cv2.imread('image/AI.jpg')
#print(im)

#均值滤波
img_mean = cv2.blur(im, (5,5))
# 高斯滤波
img_Guassian = cv2.GaussianBlur(im,(5,5),2)
# 中值滤波
img_median = cv2.medianBlur(im, 5)
# 双边滤波
img_bilater = cv2.bilateralFilter(im,9,75,75)

# 展示不同的图片
titles = ['img','mean', 'Gaussian', 'median', 'bilateral']
imgs = [im, img_mean, img_Guassian, img_median, img_bilater]
for i in range(5):
    plt.subplot(2,3,i+1)#注意，这和matlab中类似，没有0，数组下标从1开始
    plt.imshow(imgs[i])
    plt.title(titles[i])
plt.show()

