#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import cv2
import matplotlib.pyplot as plt
im = cv2.imread('image/f3.tif')
print(im.shape)
'''
gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
im_ZF = cv2.equalizeHist(gray)
plt.imshow(im_ZF)
plt.show()
'''
g = im[:,:,0]
b = im[:,:,1]
r = im[:,:,2]
g2 = cv2.equalizeHist(g)
b2 = cv2.equalizeHist(b)
r2 = cv2.equalizeHist(r)
im2 = im.copy()
im2[:,:,0] = g2
im2[:,:,1] = b2
im2[:,:,2] = r2
plt.subplot(1,2,1)
plt.imshow(im),plt.title('img')
plt.subplot(1,2,2)
plt.imshow(im2),plt.title('After')
plt.show()