#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from QT.SImage import *


class MyWindow(QMainWindow, Ui_SImage):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)





import cv2
import numpy as np
import os
import re
import xml.etree.ElementTree as ET


def WriteXml():
    labelname = ""  # 标签
    filename = ""  # 图片，xml，txt文件名
    width = " "
    height = " "

    xmax = " "
    ymax = " "
    xmlsave = "E:/asdf/xmls/1.xml"  # 保存XML文件路径

    # 创建根节点
    a = ET.Element("annotation")
    # folder节点
    b = ET.SubElement(a, "folder")
    b.text = "JPEGImages"
    # b.attrib = {}
    # filename节点
    c = ET.SubElement(a, "filename")
    c.text = filename[:-4] + '.jpg'
    # source节点
    d = ET.SubElement(a, "source")
    e = ET.SubElement(d, "database")
    e.text = "Unknown"
    # size节点
    f = ET.SubElement(a, "size")
    # width,height,depth 节点
    g = ET.SubElement(f, "width")
    g.text = width
    h = ET.SubElement(f, "height")
    h.text = height
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
    s.text = xmax
    t = ET.SubElement(p, "ymax")
    t.text = ymax

    # 创建elementtree对象，写文件
    tree = ET.ElementTree(a)
    xmlpath = xmlsave
    tree.write(xmlpath)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
