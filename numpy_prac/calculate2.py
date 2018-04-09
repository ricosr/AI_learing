# -*- coding:utf-8 -*-
import numpy as np


def show(mark, para):
    print("\n********{0}:********".format(mark))
    print(para)


A = np.arange(0, 9, 1).reshape(3, 3)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]

show("最小索引", np.argmin(A))
show("最大索引", np.argmax(A))
show("平均值", np.mean(A))
show("平均值", A.mean())
show("行平均值", np.mean(A, axis=1))
show("平均值2", np.average(A))
show("中位数", np.median(A))
show("逐项累加和", np.cumsum(A))
show("逐项累差", np.diff(A))
show("非0元素", np.nonzero(A))

B = np.array([
    [3, 5, 2],
    [7, 4, 1],
    [8, 8, 0]
])
show("行排序", np.sort(B))
show("列排序", np.sort(B, axis=0))

show("转秩", np.transpose(A))
show("转秩2", A.T)

show("统一范围外的元素", np.clip(A, 3, 7))
