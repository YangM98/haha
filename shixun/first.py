#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import cv2
import os
import numpy as np
n,m =2,2
filename = r"C:\Users\Administrator\Desktop\aa"
filepaths = os.listdir(filename)
for file in filepaths:
    name,jpg = os.path.splitext(file)
    print(name)
    filepath = os.path.join(filename,file)
    im = cv2.imread(filepath)
    im =cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lengh = im.shape[0]
    width = im.shape[1]
    len1 = np.floor(lengh/m).astype(int)
    wid1 = np.floor(width/n).astype(int)
    print(len1)
    sum=0
    for i in range(1,m+1):
        for j in range(1,n+1):
            img = im[len1*(j-1)+1:len1*j,wid1*(i-1)+1:wid1*i]
            sum = sum+1
            cv2.imwrite("C:\\Users\\Administrator\\Desktop\\aa\\"+name+"___"+str(sum)+".jpg",img)
    print(file +'-------剪切成功！！')