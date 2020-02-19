#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import cv2
import numpy as np
import matplotlib.pyplot as plt
im = cv2.imread('image/0.jpg')
print(im)
#图像平移

imgShape = im.shape
height =imgShape[0]
width = imgShape[1]
deep = imgShape[2]
img_move = np.zeros(imgShape,np.uint8)
print(img_move.shape)
n = 200#移动像素值大小
for i in range( height ):
    for j in range( width - n ):
        img_move[i, j + n] = im[i, j]
img_move[0:height,0:n] = im[0:height,-n:]

#图片镜像
img_mirr = np.zeros([height*2,width,deep],np.uint8)
for i in range( height ):
    for j in range( width ):
        img_mirr[i,j] = im[i,j]
        img_mirr[height*2-i-1,j] = im[i,j]
for i in range(width):
    img_mirr[height, i] = (0, 0, 255)

#图片缩放
reheight =int(height*1.2)
rewidth  =int(width*1.1)
img_resize = cv2.resize(im,(reheight,rewidth))

#图片旋转

matRotate = cv2.getRotationMatrix2D((height*0.5, width*0.6), 45, 1) # mat rotate 1 center 2 angle 3 缩放系数
img_cir = cv2.warpAffine(im, matRotate, (height, width))

# 展示不同的图片
titles = ['img','move', 'mirror', 'resize', 'circle']
imgs = [im, img_move, img_mirr, img_resize, img_cir]
for i in range(5):
    plt.subplot(2,3,i+1)#注意，这和matlab中类似，没有0，数组下标从1开始
    plt.imshow(imgs[i])
    plt.title(titles[i])
plt.show()


