#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import numpy as np
import cv2
import matplotlib.pyplot as plt
image = cv2.imread("image/0.jpg")
print(image)
#image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#将图像转化为灰度图像
#拉普拉斯算法
lap = cv2.Laplacian(image,cv2.CV_64F)#拉普拉斯边缘检测
lap = np.uint8(np.absolute(lap))##对lap去绝对值
plt.subplot(1,2,1)
plt.imshow(image),plt.title('Image')
plt.subplot(1,2,2)
plt.imshow(lap),plt.title('Lap')
plt.show()
'''
#Sobel边缘检测
sobelX = cv2.Sobel(image,cv2.CV_64F,1,0)#x方向的梯度
sobelY = cv2.Sobel(image,cv2.CV_64F,0,1)#y方向的梯度

sobelX = np.uint8(np.absolute(sobelX))#x方向梯度的绝对值
sobelY = np.uint8(np.absolute(sobelY))#y方向梯度的绝对值
sobelCombined = cv2.bitwise_or(sobelX,sobelY)#

plt.subplot(2,2,1),plt.imshow(image),plt.title('Image')
plt.subplot(2,2,2),plt.imshow(sobelX),plt.title('Sobel_X')
plt.subplot(2,2,3),plt.imshow(sobelY),plt.title('Sobel_Y')
plt.subplot(2,2,4),plt.imshow(sobelCombined),plt.title('Sobel')
plt.show()
'''