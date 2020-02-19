# coding=utf-8
import wx  # 导入必须的Python包
from tkinter import *
class MenuForm(wx.Frame):
    def OnQuit(self,event):
        self.Close()

    def __init__(self,parent,ID,title):
        wx.Frame.__init__(self,parent,ID,title)
        #mnuFile
        mnuFile=wx.Menu()
        mnuFile.Append(100,'&Open\tCtrl+O','Open File')
        mnuFile.AppendSeparator()
        mnuFile.Append(105,'&Quit\tCtrl+Q','Quit Application')
        #EVT_MENU
        wx.EVT_MENU(self,105,self.OnQuit)
        #menuBar
        menuBar = wx.MenuBar()
        menuBar.Append(mnuFile,"&File")
        self.SetMenuBar(menuBar)
        self.Centre()

class App(wx.App):  # 子类化wxPython应用程序类
    def OnInit(self):  # 定义一个应用程序的初始化方法
        frame = MenuForm(parent=None,ID=-1,title="GUI with Menu")
        frame.Show(True)
        return True

app = App()  # 创建一个应用程序类的实例
app.MainLoop()  # 进入这个应用程序的主事件循环