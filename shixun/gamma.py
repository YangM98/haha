#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly

from skimage import exposure
import cv2
import numpy as np
import matplotlib.pyplot as plt
#img = cv2.imread(r"C:\Users\Administrator\Desktop\0.JPG")
img = cv2.imread('image/f1.tif')
gamma_img = exposure.adjust_gamma(img, 0.8)
img1 = np.power(img/float(np.max(img)), 1/1.5)
img2 = np.power(img/float(np.max(img)), 1.5)

plt.subplot(221), plt.imshow(img), plt.title("src")
plt.subplot(222), plt.imshow(img1), plt.title("1/1.5")
plt.subplot(223), plt.imshow(img2), plt.title("1.5")
plt.subplot(224), plt.imshow(gamma_img), plt.title("gamma=0.8")
plt.show()
