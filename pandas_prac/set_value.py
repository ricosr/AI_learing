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

df.iloc[2, 3] = 1111
show("只改3行4列的值", df)

df.loc['20130101', 'B'] = 2222
show("按label改值", df)

df[df.A > 4] = 0
show("将符合A列大于4的行全部改为0", df)

df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
df.A[df.A > 4] = 0
show("只改A列大于4的改为0", df)

df['F'] = np.nan
show("加空列", df)

df['E'] = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130101', periods=6))
show("加一列非空的", df)
