#!usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import importlib
import time

importlib.reload(sys)
#sys.setdefaultencoding('utf8')

path = u"E:/data/"
#dirPath = "E:/data/"
for root, dir, files in os.walk(path):
    for file in files:
        full_path = os.path.join(root, file)
        mtime = os.stat(full_path).st_mtime
        #file_modify_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        #print(time.localtime(mtime))
        #print(time.localtime(mtime).tm_hour)
        #获取小时数匹配
        #if(time.localtime(mtime).tm_hour < 7 or time.localtime(mtime).tm_hour > 19 ):
        #测试：获取分钟
        if(time.localtime(mtime).tm_min < 7 or time.localtime(mtime).tm_min > 19 ):
            os.remove(path + file)