#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel("C:\\Users\\Administrator\\Desktop\\1.xlsx" )
#data2 = data.sort_values(by=['FIPS_Combined','YYYY'])
#data2.to_excel("C:\\Users\\Administrator\\Desktop\\2018_MCMProblemC_DATA\\2.xlsx")
data.plot()
plt.xlabel("Year")
plt.ylabel("TotalDrugReportsState")
plt.show()


