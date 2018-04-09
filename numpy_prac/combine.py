# -*- coding:utf-8 -*-
import numpy as np


def show(mark, para=''):
    print("\n********{0}:********".format(mark))
    if not isinstance(para, str):
        print(para)


a1 = np.array([1, 2, 3])
a2 = np.array([4, 5, 6])

show("行合并", np.vstack((a1, a2, a2)))
show("列合并", np.hstack((a1, a2, a2)))


show("增加一个维度变成矩阵合并", np.hstack((a1[:, np.newaxis], a2[:, np.newaxis])))

a1 = np.array([1, 2, 3])[:, np.newaxis]
a2 = np.array([4, 5, 6])[:, np.newaxis]
show("按1维度合并多个", np.concatenate((a1, a2, a2), axis=1))
show("按0维度合并多个", np.concatenate((a1, a2, a2), axis=0))
