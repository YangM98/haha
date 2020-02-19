# -*- coding: utf-8 -*-

import  sys
import  os
import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog

filepath = ""

class Ui_SImage(object):
    def setupUi(self, SImage):
        SImage.setObjectName("SImage")
        SImage.resize(855, 637)

        self.centralwidget = QtWidgets.QWidget(SImage)
        self.centralwidget.setObjectName("centralwidget")
        self.phone = QtWidgets.QToolButton(self.centralwidget)
        self.phone.setGeometry(QtCore.QRect(710, 0, 121, 111))
        self.phone.setObjectName("phone")
        self.sleep = QtWidgets.QToolButton(self.centralwidget)
        self.sleep.setGeometry(QtCore.QRect(710, 110, 121, 111))
        self.sleep.setObjectName("sleep")
        self.normal = QtWidgets.QToolButton(self.centralwidget)
        self.normal.setGeometry(QtCore.QRect(590, 0, 121, 111))
        self.normal.setObjectName("normal")
        self.homework = QtWidgets.QToolButton(self.centralwidget)
        self.homework.setGeometry(QtCore.QRect(590, 110, 121, 111))
        self.homework.setObjectName("homework")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(590, 240, 241, 361))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 239, 359))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.toolButton_5 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_5.setGeometry(QtCore.QRect(60, 520, 81, 71))
        self.toolButton_5.setObjectName("toolButton_5")
        self.toolButton_6 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_6.setGeometry(QtCore.QRect(190, 520, 81, 71))
        self.toolButton_6.setObjectName("toolButton_6")
        self.toolButton_7 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_7.setGeometry(QtCore.QRect(420, 520, 81, 71))
        self.toolButton_7.setObjectName("toolButton_7")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 551, 501))
        self.label.setText("")
        self.label.setObjectName("label")
        SImage.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(SImage)
        self.statusbar.setObjectName("statusbar")
        SImage.setStatusBar(self.statusbar)
        self.menuBar = QtWidgets.QMenuBar(SImage)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 855, 23))
        self.menuBar.setObjectName("menuBar")
        self.menuFile = QtWidgets.QMenu(self.menuBar)
        self.menuFile.setObjectName("menuFile")
        SImage.setMenuBar(self.menuBar)
        self.actionOpen_file = QtWidgets.QAction(SImage)
        self.actionOpen_file.setObjectName("actionOpen_file")
        self.actionOpen_files = QtWidgets.QAction(SImage)
        self.actionOpen_files.setObjectName("actionOpen_files")
        self.actionChange_xml_file = QtWidgets.QAction(SImage)
        self.actionChange_xml_file.setObjectName("actionChange_xml_file")
        self.actionexit = QtWidgets.QAction(SImage)
        self.actionexit.setObjectName("actionexit")
        self.menuFile.addAction(self.actionOpen_file)
        self.menuFile.addAction(self.actionOpen_files)
        self.menuFile.addAction(self.actionChange_xml_file)
        self.menuFile.addAction(self.actionexit)
        self.menuBar.addAction(self.menuFile.menuAction())

        #图标
        self.actionOpen_file.setIcon(QIcon('ic/open.png'))
        self.actionOpen_files.setIcon(QIcon('ic/open.png'))
        self.actionChange_xml_file.setIcon(QIcon('ic/open.png'))
        self.actionexit.setIcon(QIcon('ic/quit.png'))
        self.toolButton_5.setIcon(QIcon('ic/prev.png'))
        self.toolButton_6.setIcon(QIcon('ic/next.png'))
        self.toolButton_7.setIcon(QIcon('ic/cancel.png'))
        #self.toolButton_5.setToolButtonStyle(Qt.ToolButtonIconOnly)
        #self.toolButton_6.setToolButtonStyle(Qt.ToolButtonIconOnly)
        #self.toolButton_7.setToolButtonStyle(Qt.ToolButtonIconOnly)
        #快捷键
        self.toolButton_5.setShortcut('A')
        self.toolButton_6.setShortcut('D')
        self.toolButton_7.setShortcut('DELETE')

        #删除
        self.toolButton_7.setChecked(False)
        #self.toolButton_7.clicked.connect(self.btn_delete())


        #退出
        self.actionexit.setShortcut('Ctrl+P')
        self.actionexit.triggered.connect(self.close)
        self.actionexit.setStatusTip('退出')
        #打开文件
        self.actionOpen_file.triggered.connect(self.openFile)
        self.actionOpen_file.setShortcut('Ctrl+Q')

        #打开文件夹
        self.actionOpen_files.triggered.connect(self.openFiles)








        self.retranslateUi(SImage)
        QtCore.QMetaObject.connectSlotsByName(SImage)


    def retranslateUi(self, SImage):
        _translate = QtCore.QCoreApplication.translate
        SImage.setWindowTitle(_translate("SImage", "MainWindow"))
        self.phone.setText(_translate("SImage", "phone"))
        self.sleep.setText(_translate("SImage", "sleep"))
        self.normal.setText(_translate("SImage", "normal"))
        self.homework.setText(_translate("SImage", "homework"))
        self.toolButton_5.setText(_translate("SImage", "上一个"))
        self.toolButton_6.setText(_translate("SImage", "下一个"))
        self.toolButton_7.setText(_translate("SImage", "删除"))
        self.menuFile.setTitle(_translate("SImage", "File"))
        self.actionOpen_file.setText(_translate("SImage", "Open"))
        self.actionOpen_files.setText(_translate("SImage", "Open file"))
        self.actionChange_xml_file.setText(_translate("SImage", "Change xml file "))
        self.actionexit.setText(_translate("SImage", "exit"))

    def openFile(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg")
        jpg = QtGui.QPixmap(fileName1).scaled(self.label.width(), self.label.height())

        self.label.setPixmap(jpg)
        filepath = fileName1
        print (filepath)
        return filepath
    def openFiles(self):

        directory = QFileDialog.getExistingDirectory(self,
                                                      "选取文件夹",
                                                      "./",'*.jpg')
        totallist=""
        for im in os.listdir(directory):
            list = (directory+"/"+im)
            print(list)
            #totallist.append(list)

        #jpg = QtGui.QPixmap(totallist[0]).scaled(self.label.width(), self.label.height())
        #self.label.setPixmap(jpg)
    def btn_delete(self):
        if filepath=="":
            return
        else:
            os.remove(filepath)








