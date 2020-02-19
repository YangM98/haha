#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel("C:\\Users\\Administrator\\Desktop\\v.xls" )
#data2 = data.sort_values(by=['FIPS_Combined','YYYY'])
#data2.to_excel("C:\\Users\\Administrator\\Desktop\\2018_MCMProblemC_DATA\\2.xlsx")
data2=data.T
#print(data.mean)
data.plot()
plt.title('VA')
plt.xlabel("FIPS_Combined")
plt.ylabel("County total count of Drugs")
plt.ylim(0, 600)
plt.show()