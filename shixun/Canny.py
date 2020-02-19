#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('image/0.jpg')
edges = cv2.Canny(img,100,200)

plt.subplot(1,2,1)
plt.imshow(img,cmap = 'gray')
plt.title('Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2)
plt.imshow(edges,cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
plt.show()
