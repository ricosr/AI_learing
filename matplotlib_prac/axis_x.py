# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2

plt.figure(figsize=(10, 8))
plt.plot(x, y2)
plt.plot(x, y1, color="red", linewidth=1.0, linestyle="--")

plt.xlim((-1, 2))    # 设置x轴范围
plt.ylim((-2, 3))    # 设置y轴范围
plt.xlabel("I am x axis")    # 设置x轴名称
plt.ylabel("I am y axis")    # 设置y轴名称
# plt.show()

new_ticks = np.linspace(-1, 2, 5)    # 设置刻度范围，-1到2设置5个刻度
# print(new_ticks)
plt.xticks(new_ticks)    # 设置x轴刻度
plt.yticks([-2, -1.8, -1, 1.22, 3], [r"$really\ bad$", r"$bad$", r"$normal$", r"$good$", r"$really\ good$"])    # 设置y轴刻度和刻度名称

plt.show()
