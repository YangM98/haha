#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import sys
import gc
import numpy  as np
from PyQt5 import QtGui
from PyQt5.QtCore import QStringListModel
from PyQt5.QtGui import QStandardItem, QStandardItemModel

from Animal.Dbcon import Dbcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QAbstractItemView, QMessageBox, QFileDialog
from Animal.ani import Ui_MainWindow
from Animal.animals import Ui_MainWindowTwo

index,fileName,mes,mess='','','',''

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        self.db = Dbcon()
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.secondShow)
        self.BtAllshow.clicked.connect(self.showrule)
        #双击不可修改item
        self.listView.setEditTriggers(QAbstractItemView.NoEditTriggers)
        #点击item获得行号
        self.listView.clicked.connect(self.SingleClick)
        self.btdelete.clicked.connect(lambda: self.deleterule(index))
        self.combox()
        self.pushButton_3.clicked.connect(self.selectpic)
        self.BtAdd.clicked.connect(lambda: self.addrule(fileName))


    #显示combox中的信息
    def combox(self):
        sql = "select property from property"
        pro =[]
        result = self.db.show(sql)
        for i in result:
            for a in i:
               pro.append(a)
        pro.insert(0,'无')
        #print(pro)
        self.comboBox.addItems(pro)
        self.comboBox_2.addItems(pro)
        self.comboBox_3.addItems(pro)
        self.comboBox_3.addItems(pro)
        self.comboBox_4.addItems(pro)
        self.comboBox_5.addItems(pro)




    #显示推理界面
    def secondShow(self):
        self.hide()
        self.s = SecondM()
        self.s.show()
    #显示全部规则
    def showrule(self):
        rlue=[]
        sql = "select 规则  from rule "
        db = Dbcon()
        result = db.show(sql)
        for aa in result:
            for a in aa:
                rlue.append(a)
        slm = QStringListModel()
        self.qList = rlue
        slm.setStringList(self.qList)
        self.listView.setModel(slm)

    def SingleClick(self,qModelIndex):
        global index
        index = qModelIndex.row()
        #print(index)
        return index




    def deleterule(self,indexs):
        global index
        index = indexs
        if index=='':
            return
        else:
            mess = self.qList[indexs]
            reply = QMessageBox.warning(self,
                                        "警告！",
                                        "确认删除！",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
            elif reply == QMessageBox.Yes:

                sql = "delete from rule where 规则='{}'".format(mess)
                self.db.delete(sql)
                self.showrule()
                index = ''
                return index
    def selectpic(self):
        global  fileName
        fileName, filetype = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg")
        jpg = QtGui.QPixmap(fileName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
        return fileName

    def addrule(self,filepath):
        global file
        dbcon = Dbcon()

        file = filepath
        comb1=self.comboBox.currentText()
        comb2=self.comboBox_2.currentText()
        comb3=self.comboBox_3.currentText()
        comb4=self.comboBox_4.currentText()
        comb5=self.comboBox_5.currentText()
        anim = self.lineEdit.text()
        #print (file,comb1,comb2,comb3,comb4,comb5,anim)
        comclass,com =[], [comb1,comb2,comb3,comb4,comb5]
        for i in com:
            if i!='无':
                comclass.append(i)
        temp = set(comclass)
        if anim=='':
            QMessageBox.warning(self,
                                "提示！",
                                "请输入该动物名称！！！",
                                QMessageBox.Yes)
            return
        else:
            if comb1 == comb2 == comb3 == comb4 == comb5:
                QMessageBox.warning(self,
                                    "提示！",
                                    "请选择该动物的属性！！！",
                                    QMessageBox.Yes)
                return
            else:
                if len(temp) == len(comclass):
                    if file == '':
                        reply = QMessageBox.warning(self,
                                                    "提示！",
                                                    "还未添加图片，是否添加？",
                                                    QMessageBox.Yes | QMessageBox.No)
                        if reply == QMessageBox.Yes:
                            file = self.selectpic()
                        if reply == QMessageBox.No:
                            file = 'C:\\Users\\Administrator\\PycharmProjects\\haha\\Animal\\AnimalPic\\timg 1.jpg'
                    print(anim,comclass[0],file )
                    Re=''
                    if len(comclass)==1:
                        Re = comclass[0] + "、则该动物是" + anim
                    elif len(comclass)==2:
                        Re = comclass[0]+"、"+comclass[1]+"、则该动物是"+anim
                    elif len(comclass) == 3:
                        Re = comclass[0] + "、" + comclass[1] + "、" + comclass[2] + "、则该动物是" + anim
                    elif len(comclass) == 4:
                        Re = comclass[0] + "、" + comclass[1] + "、" + comclass[2] + "、" + comclass[3] +"、则该动物是" + anim
                    elif len(comclass) == 5:
                        Re = comclass[0] + "、" + comclass[1] + "、" + comclass[2] + "、" + comclass[3] + "、" + comclass[4] + "、则该动物是" + anim


                    sql = "insert into rule(animal,规则,url)values ('%s','%s','%s')"%(anim,Re,file)
                    dbcon.insert(sql)
                    num = len(comclass)
                    for i in range(0,num):
                        sqlup = "update rule set %s = 1 where animal='%s'" %(comclass[i],anim)
                        dbcon.update(sqlup)
                    #dbcon.close()
                    QMessageBox.warning(self,
                                        "提示！",
                                        "添加成功！！！",
                                        QMessageBox.Yes)
                    self.showrule()

                else:
                    QMessageBox.warning(self,
                                        "提示！",
                                        "请选择不一样的属性！！！",
                                        QMessageBox.Yes)
                    return





class SecondM(QMainWindow, Ui_MainWindowTwo):
    def __init__(self, parent=None):
        super(SecondM, self).__init__(parent)
        self.setupUi(self)
        self.db = Dbcon()
        self.listView.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.listView_2.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.pushButton_4.clicked.connect(self.FirstShow)
        self.protrey()
        self.pushButton_3.clicked.connect(self.re)
        self.listView.clicked.connect(self.SingleClick)
        self.pushButton.clicked.connect(lambda:self.ledright(mes))
        self.qList2 = []
        self.listView_2.clicked.connect(self.SingleClickleft)
        self.pushButton_2.clicked.connect(lambda:self.letleft(mess))
        self.pushButton_5.clicked.connect(self.detector)
        self.lineEdit.setReadOnly(False)
        self.lineEdit_2.setReadOnly(False)
        file='C:\\Users\\Administrator\\PycharmProjects\\haha\\Animal\\AnimalPic\\timg 1.jpg'
        jpg = QtGui.QPixmap(file).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)

    #显示推理界面
    def FirstShow(self):
        self.hide()
        self.f = MyWindow()
        self.f.show()
    #重置选择
    def re(self):
        self.protrey()
        self.qList2=[]
        UserSelect = QStringListModel()
        UserSelect.setStringList(self.qList2)
        self.listView_2.setModel(UserSelect)
    #将动物属性显示界面
    def protrey(self):
        pro = []
        sql = 'select property from property '
        result = self.db.show(sql)
        for i in result:
            for j in i:
                pro.append(j)
        pr = QStringListModel()
        self.qList = pro
        pr.setStringList(self.qList)
        self.listView.setModel(pr)
        #加入复选框
        '''
        self.model = QStandardItemModel()
        for li in pro:
            item = QStandardItem(li)
        # Add a checkbox to it
            item.setCheckable(True)
        # Add the item to the model
            self.model.appendRow(item)
        '''
    def SingleClick(self,qModelIndex):
        global mes
        index = qModelIndex.row()
        mes = self.qList[index]
        return mes


    def ledright(self,mes1):
        global mes
        if mes1=='':
            return
        self.qList2.append(mes1)
        UserSelect = QStringListModel()
        #print(self.qList2)
        UserSelect.setStringList(self.qList2)
        self.listView_2.setModel(UserSelect)

        self.qList.remove(mes1)
        pr = QStringListModel()
        pr.setStringList(self.qList)
        self.listView.setModel(pr)
        mes=''
        return mes


    def SingleClickleft(self,qModelIndex):
        global mess
        index = qModelIndex.row()
        mess = self.qList2[index]
        #print(mess)
        return mess

    def letleft(self,mess1):
        global mess
        if mess1=='':
            return
        self.qList.append(mess1)
        UserSelect = QStringListModel()
        #print(self.qList)
        UserSelect.setStringList(self.qList)
        self.listView.setModel(UserSelect)

        self.qList2.remove(mess1)
        pr = QStringListModel()
        pr.setStringList(self.qList2)
        self.listView_2.setModel(pr)
        mess=''
        return mess
    #推理模块的编写
    def detector(self):
        if self.qList2 ==[]:
            return
        aa=[]
        pro = self.qList2
        for i in pro :
            aa.append(i)
        sql = 'insert into rule (id,animal,规则,url)values (1,"预测","None","None")'
        self.db.insert(sql)
        for i in range(0,len(aa)):
            sqlup = 'update rule set %s=1  where animal="预测" '%(aa[i])
            self.db.update(sqlup)
        seclectsql = 'select * from rule  '
        result = self.db.show(seclectsql)
        bb = []
        aa = list(result)
        for i in aa:
            aa[aa.index(i)] = list(i)
        aa = np.array(aa)
        data = aa[:, 2:-2]
        data = np.array(data,dtype=int)
        #print(data)
        pre = data[0, :]
        #print(pre)
        print(data.shape[0])
        for i in range(1,data.shape[0]):
            ma = np.sum(np.abs(pre - data[i,:]))
            bb.append(ma)
            #print(ma)
        if min(bb)<=5:
            index = bb.index(min(bb))
            dect = aa[index+1,:]
            #print(dect)
            pic = dect[-1]
            #print(pic)
            name = dect[1]
            re = dect[-2]
            if pic!='':
                jpg = QtGui.QPixmap(pic).scaled(self.label.width(), self.label.height())
                self.label.setPixmap(jpg)
            self.lineEdit.setText(name)
            self.lineEdit_2.setText(re)
            sqldel = 'delete from rule where id=1'
            self.db.delete(sqldel)
            self.re()
        else:
            QMessageBox.warning(self,
                                "提示！",
                                "推理失败，请选择其他属性！",
                                QMessageBox.Yes)
            sqldel = 'delete from rule where id=1'
            self.db.delete(sqldel)
            self.re()
            self.lineEdit.setText('')




if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())