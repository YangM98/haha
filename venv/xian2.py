#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel("C:\\Users\\Administrator\\Desktop\\w.xls" )
#data2 = data.sort_values(by=['FIPS_Combined','YYYY'])
#data2.to_excel("C:\\Users\\Administrator\\Desktop\\2018_MCMProblemC_DATA\\2.xlsx")
#t=[10,18,0.2]
#y=[339]
data2=data.T
#print(data.mean)
data2.plot()
#plt.plot(t,y,'b^')
plt.grid(True)
plt.legend(bbox_to_anchor=(0,10,10,0),ncol=2,loc=0,mode='expand',borderaxespad=0)

plt.annotate('drug identification threshold levels',xy=(15,365),xytext=(13,700,),arrowprops=dict(arrowstyle='->'))
plt.title('WV -Trends in the change of opioids in counties ')
plt.xlabel("Year(year)")
plt.ylabel("Total number of opioid and heroin incidents")
plt.ylim(0, 800)
plt.show()