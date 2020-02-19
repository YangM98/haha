#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import glob
file = r'C:\Users\Administrator\Pictures\Saved Pictures'
for filenames in glob.glob( file+'\*.jpg'):
    print(filenames)