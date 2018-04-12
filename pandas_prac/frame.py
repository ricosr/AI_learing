# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

df = pd.Series([1, 3, 5, np.nan, 44, 1])
# 0     1.0
# 1     3.0
# 2     5.0
# 3     NaN
# 4    44.0
# 5     1.0

dates = pd.date_range('20160301', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
#                    a         b         c         d
# 2016-03-01 -0.971804  1.175639  0.094365  0.247782
# 2016-03-02 -0.093352  0.049045 -0.479096 -1.111607
# 2016-03-03 -0.084437 -0.329073 -1.411246  0.810504
# 2016-03-04  1.759887 -0.996337  0.331909  0.284511
# 2016-03-05 -0.698617  0.263004  0.021155  0.791914
# 2016-03-06  0.573120  2.409110 -0.215235  1.069406

df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})
#      A          B    C  D      E    F
# 0  1.0 2013-01-02  1.0  3   test  foo
# 1  1.0 2013-01-02  1.0  3  train  foo
# 2  1.0 2013-01-02  1.0  3   test  foo
# 3  1.0 2013-01-02  1.0  3  train  foo


def show(mark, para=''):
    print("\n********{0}:********".format(mark))
    if not isinstance(para, str):
        print(para)


show("数据类型", df2.dtypes)
show("看下标", df2.index)
show("列名", df2.columns)
show("所有值", df2.values)
show("数据总结", df2.describe())
show("转秩", df2.T)
show("按列名(横轴)降序排列", df2.sort_index(axis=1, ascending=False))
show("按E列排序", df2.sort_values(by="E"))
