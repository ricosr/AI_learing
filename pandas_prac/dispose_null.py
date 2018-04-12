# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np


def show(mark, para=''):
    print("\n********{0}:********".format(mark))
    if not isinstance(para, str):
        print(para)


dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
df.iloc[0, 1] = np.nan
df.iloc[1, 2] = np.nan
#              A     B     C   D
# 2013-01-01   0   NaN   2.0   3
# 2013-01-02   4   5.0   NaN   7
# 2013-01-03   8   9.0  10.0  11
# 2013-01-04  12  13.0  14.0  15
# 2013-01-05  16  17.0  18.0  19
# 2013-01-06  20  21.0  22.0  23

show("去掉每列横向有NaN的行", df.dropna(axis=0, how='any'))

# show("去掉每列横全为NaN的行", df.dropna(axis=0, how='all'))

show("把为NaN的项改为0", df.fillna(value=0))

show("显示有数据缺失的项", df.isnull())

show("判断是否有数据缺失", np.any(df.isnull() == True))
