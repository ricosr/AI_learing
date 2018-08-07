# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

# 原理：组成n*n的网格, 每个点对应一个高度值

def f(x, y):
    return (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)

n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-4, 4, n)
print(np.meshgrid(x, y))
X, Y = np.meshgrid(x, y)    # 在二维平面中将每一个x和每一个y分别对应起来
print(f(X, Y))


plt.figure()
plt.scatter(X, Y, c=X)

plt.show()

# 添加颜色：
plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)    # 透明度0.75, f(X,Y) 的值对应到color map的暖色组中寻找对应颜色

# 等高线：
C = plt.contour(X, Y, f(X, Y), 8, colors="black", linewidths=.5)    # 颜色选黑色，线条宽度选0.5
# contour up to 8+1 automatically chosen contour levels

# 添加高度数字：
plt.clabel(C, inline=True, fontsize=10)    # inline控制是否将Label画在线里面，字体大小为10

# plt.xticks(())
# plt.yticks(())

plt.show()
