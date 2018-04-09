# -*- coding:utf-8 -*-
import numpy as np

# create a array

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(a.shape)    # 几行几列
print(a.size)     # 多少个元素
print(a.ndim)     # 矩阵的秩
print(a.dtype)    # 元素的类型

print("\n**************zeros:***************")
az = np.zeros((2, 2))    # 生成2×2元素全为0
print(az)

print("\n**************ones:***************")
oa = np.ones((2, 2), dtype=float)    # 生成2×2元素全为1
print(oa)

print("\n**************empty:***************")
ea = np.empty((3, 3))        # 生成3×3元素全接近0
print(ea)

print("\n**************arange:***************")
aa = np.arange(1, 20, 2).reshape(2, 5)        # 步长2, 1~20生成2×5矩阵
print(aa)

print("\n**************linespace:***************")
al = np.linspace(1, 20, 4, dtype=float).reshape(2, 2)    # 1~20生成4个元素生成2×2矩阵
print(al)

print("\n**************random:***************")
ar = np.random.random((2, 5))    # 0~1随机生成2×5矩阵
print(ar)


