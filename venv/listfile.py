#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import os
filepath=r"G:\pic"
txtpath=r"C:\Users\Administrator\Desktop\aa.txt"
for file in os.listdir(filepath):
    with open (txtpath,"a") as f:
        list =os.path.join(filepath,file)
        print(list)
        f.write(list+"\n")
    f.close()
