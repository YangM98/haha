#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import os
import  re
import cv2
import numpy as np
import xml.etree.ElementTree as ET

filepath = 'C:/Users/Administrator/Pictures/Saved Pictures/timg.jpg'  #此处有更改
label = 'normal'
xmlpath = "C:/Users/Administrator/Desktop"

def WriteXml( filepath, label, xmlpath):
    labelname = label  # 标签
    (file1,filename) = os.path.split(str(filepath))  #此处有更改
    (file,ext) = os.path.splitext(filename)
    print(filename)    #此处有更改，写xml中的filename做了切片处理，单独打印filename，结果为后面这一串0.jpg' mode='r' encoding='cp936'><_io.TextIOWrapper name='E:/abc/wang 0 .jpg' mode='r' encoding='cp936'>
    print(file1,file,ext)
    img = cv2.imread(filepath)

    #img1 = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    shape = img.shape
    height = shape[0]
    width = shape[1]
    print(shape)

    xmlsave = xmlpath+"/"+file+".xml"  # 保存XML文件路径
    print (xmlsave)

    # 创建根节点
    a = ET.Element("annotation")
    # folder节点
    b = ET.SubElement(a, "folder")
    b.text = "JPEGImages"
    # b.attrib = {}
    # filename节点
    c = ET.SubElement(a, "filename")
    c.text = file+ext   #此处有更改
    # source节点
    d = ET.SubElement(a, "source")
    e = ET.SubElement(d, "database")
    e.text = "Unknown"
    # size节点
    f = ET.SubElement(a, "size")
    # width,height,depth 节点
    g = ET.SubElement(f, "width")
    g.text = str(width)   #此处有更改
    h = ET.SubElement(f, "height")
    h.text = str(height)  #此处有更改
    ii = ET.SubElement(f, "depth")
    ii.text = "3"
    # segmented节点
    j = ET.SubElement(a, "segmented")
    j.text = "0"
    # object节点
    k = ET.SubElement(a, "object")
    # name,pose,truncated,difficult节点
    l = ET.SubElement(k, "name")
    l.text = labelname
    m = ET.SubElement(k, "pose")
    m.text = "Unspecifiied"
    n = ET.SubElement(k, "truncated")
    n.text = "0"
    o = ET.SubElement(k, "difficult")
    o.text = "0"
    # bndbox节点
    p = ET.SubElement(k, "bndbox")
    # xmin,ymin,xmax,ymax节点
    q = ET.SubElement(p, "xmin")
    q.text = "2"
    r = ET.SubElement(p, "ymin")
    r.text = "2"
    s = ET.SubElement(p, "xmax")
    s.text = str(width)   #此处有更改
    t = ET.SubElement(p, "ymax")
    t.text = str(height)  #此处有更改

    # 创建elementtree对象，写文件
    tree = ET.ElementTree(a)
    tree.write(xmlsave)
if __name__ == '__main__':
    WriteXml(filepath,label,xmlpath)