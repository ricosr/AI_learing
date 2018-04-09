# -*- coding:utf-8 -*-
import numpy as np
# **********array1:**********
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]
# **********array2:**********
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]


a1 = np.arange(0, 9, 1).reshape(3, 3)
a2 = np.arange(1, 10, 1).reshape(3, 3)
print("**********array1:**********")
print(a1)
print("**********array2:**********")
print(a2)

print("**********sum = array1 + array2:**********")
sum_a = a1 + a2
print(sum_a)

print("**********sum = array2 - array1:**********")
sum_a = a2 - a1
print(sum_a)

print("**********multiplication one by one**********")
mul_result1 = a2 * a1
print(mul_result1)

print("**********matrix multiplication**********")
mul_result2 = np.dot(a1, a2)
print(mul_result2)

print("**********suqare one by one**********")
square = a1 ** 2
print(square)

print("**********三角函数：**********")
print(np.sin(a1))
print(np.cos(a1))
print(np.tan(a1))

print("**********找出特定大小元素**********")
print(a1 > 0)
print(a1 == 5)
print(a1 <= 7)

print("**********按条件求和：**********")
print(np.sum(a1))
print(np.sum(a1, axis=0))   # 列
print(np.sum(a1, axis=1))   # 行
