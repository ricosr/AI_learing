# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y = 2*x + 1
# y2 = x**2

plt.figure(num=1, figsize=(8, 5),)
plt.plot(x, y,)

ax = plt.gca()
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")
# ax.spines["top"].set_color("none")

ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_position(("data", 0))
ax.yaxis.set_ticks_position("left")
ax.spines["left"].set_position(("data", 0))

x0 = 1
y0 = 2*x0 + 1
plt.plot([x0, x0], [0, y0], "k--", linewidth=2.5)    # 两个点连成垂直x轴的线
plt.scatter([x0, ], [y0, ], s=50, color='b')    # 画点
# plt.scatter(x0, y0, s=50, color='b')    # 没区别

plt.annotate(r"$2x+1=%s$" % y0, xy=(x0, y0), xycoords="data", xytext=(+30, -30),
             textcoords="offset points", fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.2"))

# plt.annotate(注释内容, xy=(要注释的点), 基于数据来选位置, 注释内容距离注释点xy的偏差,
#              设置为距离注释点xy的偏差, 字体大小,
#              对图中箭头类型的一些设置)


# text注释:
plt.text(-3.7, 3, "$This\ is\ the\ some\ text.\ \mu\ \sigma_i\ \alpha_t$",
         fontdict={"size": 16, "color": 'r'})    # paras: 注释位置, 注释内容, 注释字体

plt.show()

