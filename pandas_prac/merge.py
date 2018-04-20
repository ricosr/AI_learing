# -*- coding:utf-8 -*-

import pandas as pd


def show(mark, para=''):
    print("\n********{0}:********".format(mark))
    if not isinstance(para, str):
        print(para)


left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

#     A   B key
# 0  A0  B0  K0
# 1  A1  B1  K1
# 2  A2  B2  K2
# 3  A3  B3  K3
#     C   D key
# 0  C0  D0  K0
# 1  C1  D1  K1
# 2  C2  D2  K2
# 3  C3  D3  K3

res = pd.merge(left, right, on="key")
show("按key列合并", res)


left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

#     A   B key1 key2
# 0  A0  B0   K0   K0
# 1  A1  B1   K0   K1
# 2  A2  B2   K1   K0
# 3  A3  B3   K2   K1
#     C   D key1 key2
# 0  C0  D0   K0   K0
# 1  C1  D1   K1   K0
# 2  C2  D2   K1   K0
# 3  C3  D3   K2   K0


res = pd.merge(left, right, on=["key1", "key2"], how="inner")
show("按key1, key2列inner合并", res)

res = pd.merge(left, right, on=["key1", "key2"], how="outer")
show("按key1, key2列outer合并", res)

res = pd.merge(left, right, on=["key1", "key2"], how="left")
show("按key1, key2列left合并", res)

res = pd.merge(left, right, on=["key1", "key2"], how="right")
show("按key1, key2列right合并", res)


df1 = pd.DataFrame({'col1': [0, 1], 'col_left': ['a', 'b']})
df2 = pd.DataFrame({'col1': [1, 2, 2], 'col_right': [2, 2, 2]})
#    col1 col_left
# 0     0        a
# 1     1        b
#    col1  col_right
# 0     1          2
# 1     2          2
# 2     2          2

res = pd.merge(df1, df2, on='col1', how="outer", indicator=True)
show("显示merge方式", res)

res = pd.merge(df1, df2, on='col1', how="outer", indicator="indicator_column")
show("定义indicator列名称", res)


left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                    index=['K0', 'K1', 'K2'])

right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']},
                     index=['K0', 'K2', 'K3'])

#      A   B
# K0  A0  B0
# K1  A1  B1
# K2  A2  B2
#      C   D
# K0  C0  D0
# K2  C2  D2
# K3  C3  D3

res = pd.merge(left, right, left_index=True, right_index=True, how="outer")
show("按index outer合并", res)


res = pd.merge(left, right, left_index=True, right_index=True, how="inner")
show("按index inner合并", res)


boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})

res = pd.merge(boys, girls, on='k', suffixes=["_boy", "_girl"], how="inner")
show("解决列名相同问题", res)
