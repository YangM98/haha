# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SeIMG.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!
import glob

import cv2
import os
import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QStringListModel, QVariant
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QListView, QMessageBox
import xml.etree.ElementTree as ET


global filepath, xmlpath,list
filepath=""
xmlpath=""

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(877, 606)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 621, 461))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setText("")
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(660, 250, 211, 311))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.listView = QtWidgets.QListView(self.horizontalLayoutWidget)
        self.listView.setObjectName("listView")
        self.horizontalLayout.addWidget(self.listView)

        #self.listView.setViewMode(QListView.ListMode)





        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(620, -1, 41, 561))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_2.addItem(spacerItem)
        self.pre = QtWidgets.QPushButton(self.centralwidget)
        self.pre.setGeometry(QtCore.QRect(50, 470, 91, 81))
        self.pre.setObjectName("pre")
        self.next = QtWidgets.QPushButton(self.centralwidget)
        self.next.setGeometry(QtCore.QRect(200, 470, 91, 81))
        self.next.setObjectName("next")
        self.delete_2 = QtWidgets.QPushButton(self.centralwidget)
        self.delete_2.setGeometry(QtCore.QRect(450, 470, 91, 81))
        self.delete_2.setObjectName("delete_2")
        self.pushButton_1 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_1.setGeometry(QtCore.QRect(660, 0, 101, 121))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.pushButton_1.setFont(font)
        self.pushButton_1.setObjectName("pushButton_1")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(770, 0, 101, 121))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(770, 130, 101, 111))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(660, 130, 101, 111))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 877, 23))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionOpen_Dir = QtWidgets.QAction(MainWindow)
        self.actionOpen_Dir.setObjectName("actionOpen_Dir")
        self.actionChange_xml = QtWidgets.QAction(MainWindow)
        self.actionChange_xml.setObjectName("actionChange_xml")
        self.actionexit = QtWidgets.QAction(MainWindow)
        self.actionexit.setObjectName("actionexit")
        self.actionTip = QtWidgets.QAction(MainWindow)
        self.actionTip.setObjectName("actionTip")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionOpen_Dir)
        self.menuFile.addAction(self.actionChange_xml)
        self.menuFile.addAction(self.actionexit)
        self.menuHelp.addAction(self.actionTip)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        #self.listView.setEditTriggers(self,QAbstractItemView_EditTriggers=)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        #self.pre.setText(_translate("MainWindow", "PushButton"))
        #self.next.setText(_translate("MainWindow", "PushButton"))
        #self.delete_2.setText(_translate("MainWindow", "PushButton"))
        self.pushButton_1.setText(_translate("MainWindow", "Normal"))
        self.pushButton_2.setText(_translate("MainWindow", "Phone"))
        self.pushButton_3.setText(_translate("MainWindow", "Homework"))
        self.pushButton_4.setText(_translate("MainWindow", "Sleep"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionOpen.setText(_translate("MainWindow", "Open file"))
        self.actionOpen_Dir.setText(_translate("MainWindow", "Open Dir"))
        self.actionChange_xml.setText(_translate("MainWindow", "Change xml "))
        self.actionexit.setText(_translate("MainWindow", "Exit"))
        self.actionTip.setText(_translate("MainWindow", "Tip"))



class MyWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)


        # 图标
        self.actionOpen.setIcon(QIcon('ic/open.png'))
        self.actionOpen_Dir.setIcon(QIcon('ic/open.png'))
        self.actionChange_xml.setIcon(QIcon('ic/open.png'))
        self.actionexit.setIcon(QIcon('ic/quit.png'))
        self.pre.setIcon(QIcon('ic/prev.png'))
        self.next.setIcon(QIcon('ic/next.png'))
        self.delete_2.setIcon(QIcon('ic/cancel.png'))
        # self.pre.setToolButtonStyle(Qt.ToolButtonIconOnly)
        # self.next.setToolButtonStyle(Qt.ToolButtonIconOnly)
        # self.delete_2.setToolButtonStyle(Qt.ToolButtonIconOnly)

        # 快捷键
        self.pre.setShortcut('A')
        self.next.setShortcut('D')
        self.delete_2.setShortcut('E')

        # 删除
        #self.delete_2.setChecked(False)
        self.delete_2.clicked.connect(lambda:self.btn_delete(filepath))

        # 退出
        self.actionexit.setShortcut('Ctrl+P')
        self.actionexit.triggered.connect(self.close)
        self.actionexit.setStatusTip('退出')
        # 打开文件
        self.actionOpen.triggered.connect(self.openFile)
        self.actionOpen.setShortcut('Ctrl+Q')


        # 打开文件夹
        self.actionOpen_Dir.triggered.connect(self.openFiles)


        #打开要保存的xml文件夹
        self.actionChange_xml.triggered.connect(self.open_xml)

        #双击item
        self.listView.doubleClicked.connect(self.itemshow)


        #前一张Img
        #self.pre.clicked.connect(self.preImg1)
        self.pre.clicked.connect(lambda :self.preImg(filepath))

        #后一张Img
        #self.next.clicked.connect(self.nextImg1)
        self.next.clicked.connect(lambda :self.nextImg(filepath))


        #生成xml
        self.pushButton_1.clicked.connect(lambda: self.WriteXml(filepath,'normal',xmlpath))
        self.pushButton_2.clicked.connect(lambda: self.WriteXml(filepath,'phone',xmlpath))
        self.pushButton_3.clicked.connect(lambda: self.WriteXml(filepath,'homework',xmlpath))
        self.pushButton_4.clicked.connect(lambda: self.WriteXml(filepath,'sleep',xmlpath))

    def itemshow(self,index):
        global list
        aa = index.row()
        filepath =list[aa]
        jpg = QtGui.QPixmap(filepath).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
        self.statusBar().showMessage(filepath)
        #print (filepath)

    def openFile(self):  #打开图片，选择保存xml文件夹
        fileName1, filetype = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg")
        jpg = QtGui.QPixmap(fileName1).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)

        global filepath, xmlpath,list

        filepath = fileName1
        xmlpath = self.open_xml()
        list = [filepath]
        if xmlpath == '':
            xmlpath = fileName1

        model = QStandardItemModel()
        self.item = QStandardItem(filepath)
        model.appendRow(self.item)
        self.listView.setModel(model)

        self.statusBar().showMessage(filepath)
        #print (filepath,xmlpath)

    def openFiles(self):

        try:
            directory = QFileDialog.getExistingDirectory(self,
                                                          "选取图片文件夹",
                                                          "./")
            global filepath, xmlpath,list

            xmlpath = self.open_xml()
            if xmlpath=='':
                xmlpath = directory

            model = QStandardItemModel()
            #self.model  = QStringListModel(self)

            list = [filenames for filenames in glob.glob(directory +'/*.jpg')]
            #print(list)
            for item  in list:
                aa = QStandardItem(item)
                model.appendRow(aa)
            self.listView.setModel(model)
            filepath = list[0]
            jpg = QtGui.QPixmap(filepath).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(jpg)
            self.statusBar().showMessage(filepath)

            #print(filepath,xmlpath)
            return filepath,xmlpath
        except :
            return

    def open_xml(self):
        global xmlpath
        xmlpath = QFileDialog.getExistingDirectory(self,
                                                   "选择XML文件夹",
                                                   "./")
        return xmlpath


    def preImg (self,filepaths):
        global filepath
        if filepaths== "":
            return
        else :
            index = list.index(filepaths)
            if index == 0:
                QMessageBox.warning(self,
                                    "提示",
                                    "这是第一张，请往后走！！！",
                                    QMessageBox.Yes)
                #print("当前为第一张！！！")
                filepath = filepaths
            else:
                index1 = index - 1
                filepath = list[index1]
                jpg = QtGui.QPixmap(filepath).scaled(self.label.width(), self.label.height())
                self.label.setPixmap(jpg)
                self.statusBar().showMessage(filepath)

        return filepath


    def nextImg (self,filepaths):
        global filepath
        if filepaths =="":
            return
        else:
            index = list.index(filepaths)
            if index ==len(list)-1:
                QMessageBox.warning(self,
                                    "提示",
                                    "后边没有了！！！",
                                    QMessageBox.Yes)

                filepath = filepaths
            else:
                index1 =index+1
                filepath = list[index1]
                jpg = QtGui.QPixmap(filepath).scaled(self.label.width(), self.label.height())
                self.label.setPixmap(jpg)
                self.statusBar().showMessage(filepath)

        return filepath





    def btn_delete(self,filepaths):

        global filepath, model
        model = QStandardItemModel()

        if filepaths=='':
            QMessageBox.warning(self,
                                "提示！",
                                "当前没有图片，请导入！",
                                QMessageBox.Yes)
            return
        else:
            index = list.index(filepaths)#获取当前图片的index
            reply =QMessageBox.warning(self,
                                "警告！",
                                "确认删除！",
                                QMessageBox.Yes|QMessageBox.No)
            if reply ==QMessageBox.No:
                return
            elif reply ==QMessageBox.Yes:
                list.remove(filepaths)
                for item in list:
                    aa = QStandardItem(item)
                    model.appendRow(aa)
                self.listView.setModel(model)

                # os.remove(filepath)
                if index == len(list):  #获取下一张图片的路径
                    QMessageBox.warning(self,
                                        "提示",
                                        "后边没有了,又重头再来了！",
                                        QMessageBox.Yes)

                    if len(list)==0:
                        filepath = ""
                        return  filepath
                    else:
                        filepath = list[0]
                        jpg = QtGui.QPixmap(filepath).scaled(self.label.width(), self.label.height())
                        self.label.setPixmap(jpg)
                        self.statusBar().showMessage(filepath)
                        print("liu")
                else:
                    filepath = list[index]
                    jpg = QtGui.QPixmap(filepath).scaled(self.label.width(), self.label.height())
                    self.label.setPixmap(jpg)
                    self.statusBar().showMessage(filepath)
                   # print("删除........")
                    return filepath


    def WriteXml(self,filepath, label, xmlpath):
        if filepath=="" or xmlpath =="":
            return
        else:
            labelname = label  # 标签
            (file1, filename) = os.path.split(str(filepath))  # 此处有更改
            (file, ext) = os.path.splitext(filename)
            #print(filename)  # 此处有更改，写xml中的filename做了切片处理，单独打印filename，结果为后面这一串0.jpg' mode='r' encoding='cp936'><_io.TextIOWrapper name='E:/abc/wang 0 .jpg' mode='r' encoding='cp936'>
            #print(file1, file, ext)
            img = cv2.imread(filepath)

            # img1 = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            shape = img.shape
            height = shape[0]
            width = shape[1]
            #print(shape)

            xmlsave = xmlpath + "/" + file + ".xml"  # 保存XML文件路径
            #print(xmlsave)

            # 创建根节点
            a = ET.Element("annotation")
            # folder节点
            b = ET.SubElement(a, "folder")
            b.text = "JPEGImages"
            # b.attrib = {}
            # filename节点
            c = ET.SubElement(a, "filename")
            c.text = file + ext  # 此处有更改
            # source节点
            d = ET.SubElement(a, "source")
            e = ET.SubElement(d, "database")
            e.text = "Unknown"
            # size节点
            f = ET.SubElement(a, "size")
            # width,height,depth 节点
            g = ET.SubElement(f, "width")
            g.text = str(width)  # 此处有更改
            h = ET.SubElement(f, "height")
            h.text = str(height)  # 此处有更改
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
            s.text = str(width)  # 此处有更改
            t = ET.SubElement(p, "ymax")
            t.text = str(height)  # 此处有更改

            # 创建elementtree对象，写文件
            tree = ET.ElementTree(a)
            tree.write(xmlsave)
            self.statusbar.showMessage("生成xml成功!!!     "+xmlsave)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())


