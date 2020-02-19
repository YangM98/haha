#!usr/bin/python
# -*- coding: utf-8 -*-
'''
批量视频截取成图片
'''
import cv2
import os

filepath = r"G:\trainData\3th\video"
pathlist = os.listdir(filepath)
a = 1
for alldir in pathlist:
    videopath = r'G:\\trainData\\3th\\video\\'+alldir
    print(videopath)
    vc = cv2.VideoCapture(videopath)
    c = 1
    # 判断是否正常打开
    if vc.isOpened():
        rval,frame = vc.read()
    else:
        rval = False
    # 视频帧数间隔
    timeOut = 800

    while rval:
        rval,frame = vc.read()
        b = a
        if(c%timeOut==0):
            str1 = str(b)
            cv2.imwrite("G:\\trainData\\3th\\picture\\"+str1.zfill(6)+'.jpg',frame)
            print("已截取第%d张图片" % a)
            a= a+1
        c=c+1
       # cv2.waitKey(0)
    vc.release()