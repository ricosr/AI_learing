# -*- coding:utf-8 -*-
import numpy as np


def show(mark, para=''):
    print("\n********{0}:********".format(mark))
    if not isinstance(para, str):
        print(para)


A = np.arange(16).reshape((4, 4))
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]]

show("等分1维度", np.split(A, 2, axis=1))
show("等分0维度", np.split(A, 2, axis=0))
show("不等分1维度", np.array_split(A, 3, axis=1))
show("不等分0维度", np.array_split(A, 3, axis=0))
show("横向等分", np.vsplit(A, 2))
show("竖向等分", np.hsplit(A, 2))


