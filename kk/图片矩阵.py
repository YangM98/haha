#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import glob
import numpy as np
import cv2

img=[]
filepath = "C:\\Users\\Administrator\\Desktop\\No4\\训练样本\\经向疵点"
list = [filenames for filenames in glob.glob(filepath+'/*.bmp')]
#print (list)
for i in list:
    #print(i)
    im = cv2.imdecode(np.fromfile(i, dtype=np.uint8), -1)#路径有中文
    #im = cv2.imread(i)
    aa = im.flatten()
    np.savetxt("C:\\Users\\Administrator\\Desktop\\No4\\训练样本\\径向疵点.txt",aa)
    #with open("C:\\Users\\Administrator\\Desktop\\No4\\训练样本\\径向疵点.txt","a+") as f:
        #f.write(aa+'\n')
    #img.append(im)

