# -*- coding:utf-8 -*-

import pandas as pd

data = pd.read_csv("student.csv")
print(data)

data.to_pickle("student.pickle")


data2 = pd.read_pickle("student.pickle")
print(data2)

