# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2

plt.figure()
plt.plot(x, y2)
plt.plot(x, y1, color="red", linewidth=1.0, linestyle="--")
plt.xlim((-1, 2))
plt.ylim((-2, 3))

new_ticks = np.linspace(-1, 2, 5)    # 设置刻度范围，-1到2设置5个刻度
# print(new_ticks)
plt.xticks(new_ticks)    # 设置x轴刻度
plt.yticks([-2, -1.8, -1, 1.22, 3], [r"$really\ bad$", r"$bad$", r"$normal$", r"$good$", r"$really\ good$"])    # 设置y轴刻度和刻度名称

ax = plt.gca()    # 获取当前坐标轴信息
ax.spines["right"].set_color("red")    # 设置边框颜色
ax.spines["top"].set_color("blue")
# plt.show()

ax.xaxis.set_ticks_position("bottom")    # 设置x的刻度位置    paras: top, bottom, both, default, none
ax.spines["bottom"].set_position(("data", 0))    # 设置x=0的位置,    paras: outward, axes, data ????

ax.yaxis.set_ticks_position("left")    # 设置y的刻度位置
ax.spines["left"].set_position(("data", 0))    # 设置y=0的位置

plt.show()

