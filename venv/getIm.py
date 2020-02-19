#!usr/bin/python
# -*- coding: utf-8 -*-
'''
单个视频截取成图片
'''
import cv2
videopath = "G:\\test.mp4"
vc = cv2.VideoCapture(videopath)
a = 1 #图片计数
c = 1 #帧数计数
# 判断是否正常打开
if vc.isOpened():
    rval, frame = vc.read() #返回一个布尔类型的 frame为图片
    print("正在截取文件，路径为：" + videopath)
else:
    rval = False
    print("加载文件失败，请检查文件路径！")
# 视频帧数间隔
timeOut = 1

while rval:
    rval, frame = vc.read()

    if (c%timeOut == 0):
        str1 = str(a)
        cv2.imwrite('G:\\aa\\' + str1.zfill(0) + '.jpg', frame)
        print("已截取第%d张图片"%a)
        a = a + 1
    c = c+1
    #cv2.waitKey(1)#表示等待键盘输入 1 表示1ms切换到下一帧图像
vc.release()#释放