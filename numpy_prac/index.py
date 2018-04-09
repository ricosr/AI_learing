# -*- coding:utf-8 -*-
import numpy as np


def show(mark, para=''):
    print("\n********{0}:********".format(mark))
    if not isinstance(para, str):
        print(para)


A = np.arange(0, 16, 1).reshape(4, 4)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]]

show("A[3]", A[3])
show("A[1][1]", A[1][1])
show("A[2, 1]", A[2, 1])
show("第三行所有元素", A[2, :])
show("第二列所有元素", A[:, 1])

show("行迭代")
for row in A:
    print(row)

show("列迭代")
for row in A.T:
    print(row)

show("逐项迭代")
for item in A.flat:
    print(item)
