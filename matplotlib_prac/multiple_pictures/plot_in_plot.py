# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt

fig = plt.figure()

x = [1, 2, 3, 4, 5, 6, 7]    # 准备数据
y = [1, 3, 4, 2, 5, 8, 6]

# 大图：
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
# ax1 的位置信息：距离零点位置分别占整图的 1%, 1%, 8%, 8%
ax1 = fig.add_axes([left, bottom, width, height])
ax1.plot(x, y, 'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title("title")

# 小图：
left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(y, x, 'b')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title("title inside 1")

# 小图2：
plt.axes([0.6, 0.2, 0.25, 0.25])    # 直接加坐标
plt.plot(y[::-1], x, 'g')
plt.xlabel('x')
plt.ylabel('y')
plt.title("title inside 2")

plt.show()
