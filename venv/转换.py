#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.DataFrame(pd.read_csv(r"C:\Users\Administrator\Desktop\2018_MCMProblemC_DATA\16\ACS_16_5YR_DP02_with_ann.csv",encoding='UTF-8' ))
#print (data)

data2=data.iloc[:,3:-1:4]
data2.to_excel(r"C:\Users\Administrator\Desktop\16_.xls")
#data3 = data.iloc[:,200:399]
#data3.to_excel(r"C:\Users\Administrator\Desktop\16_2.xls")
#data4 = data.iloc[:,400:-1]
#data4.to_excel(r"C:\Users\Administrator\Desktop\16_3.xls")