#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import pymysql
import numpy as np

class Dbcon() :
    def __init__(self) -> object:
        self.db=pymysql.connect("localhost","root","123456","animal")

    def show (self,sql):
        cursor = self.db.cursor()
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
            self.db.commit()
            return  result
        except:
            self.db.rollback()
        self.db.close()
    def delete(self,sql):
        cursor = self.db.cursor()
        try:
            cursor.execute(sql)
            self.db.commit()
        except:
            self.db.rollback()

    def insert(self,sql):
        cursor = self.db.cursor()
        try:
            cursor.execute(sql)
            self.db.commit()
        except:
            self.db.rollback()





    def update(self,sql):
        cursor = self.db.cursor()
        try:
            cursor.execute(sql)
            self.db.commit()
        except:
            self.db.rollback()



if __name__ == '__main__':
    db = Dbcon()
    sql="select * from rule "
    result = db.show(sql)

    bb=[]
    b=list(result)
    b[:] = [list(c) for c in b]
    print(type(b))
   #for i in b:
       # b[b.index(i)]=list(i)
        #aa[aa.index(i)]=list(i)
    #aa = np.array(b,dtype=)
    aa = np.array(b)
    #print(b)
    print(aa)
    data =aa[:,2:-2]
    data = np.array(data,dtype=int)
    print(data)

    #data =list(map(int,data))
    pre = data[0,:]
    print(pre)
    for i in range(1,data.shape[0]):
        ma = np.sum(np.abs(pre - data[i,:]))
        bb.append(ma)
        print(ma)
    index = bb.index(min(bb))
    print(index)
    print(aa[index+1,:])
    x = aa[14,:]
    x=list(x)
    print(x)
