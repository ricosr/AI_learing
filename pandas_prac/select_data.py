# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np


def show(mark, para=''):
    print("\n********{0}:********".format(mark))
    if not isinstance(para, str):
        print(para)


dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
#              A   B   C   D
# 2013-01-01   0   1   2   3
# 2013-01-02   4   5   6   7
# 2013-01-03   8   9  10  11
# 2013-01-04  12  13  14  15
# 2013-01-05  16  17  18  19
# 2013-01-06  20  21  22  23

print("1.按行或列简单获取：")
show("按列名", df['A'])
show("按列名2", df.A)
show("跨多行", df[0:3])
show("按行标跨多行", df["2013-01-02":"2013-01-04"])

print("2.按label获取数据：")
show("按行标", df.loc["2013-01-02"])
show("按列标", df.loc[:, ['A', 'B']])
show("按行标&&列标", df.loc["2013-01-02", ['A', 'B']])

print("3.按坐标位置获取时候数据:")
show("第4行", df.iloc[3])
show("4行2列", df.iloc[3, 1])
show("4到5行，2到3列", df.iloc[3:5, 1:3])
show("246行的2到3列", df.iloc[[1, 3, 5], 1:3])

print("4.按坐标和label:")
show("前3行的AC两列", df.ix[:3, ['A', 'C']])

print("5.按条件获取数据：")
show("", df[df.A < 8])
