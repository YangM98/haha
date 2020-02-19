#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import os
import shutil
filepath=r"F:\data2\jpg"
picpath=r"F:\data2\xml"
outpicpath=r"F:\data2/newxml"
for file in os.listdir(filepath):
    filename=os.path.splitext(file)
    filename = filename[0]+".xml"
    name = os.path.join(picpath,filename)
    newname =os.path.join(outpicpath,filename)
    shutil.copyfile(name,newname)





