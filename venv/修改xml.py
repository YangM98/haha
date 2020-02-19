#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import xml.etree.ElementTree as ET
doc = ET.parse('C:/Users/Administrator/Desktop/hand1/新建文件夹/000001.xml')
root = doc.getroot()
print(root.tag)
