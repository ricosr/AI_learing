# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y = 0.001*x

plt.figure()
plt.plot(x, y, linewidth=10, zorder=1)    # set zorder for ordering the plot in plt 2.0.2 or higher
plt.ylim(-2, 2)
ax=plt.gca()
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_position(("data", 0))
ax.yaxis.set_ticks_position("left")
ax.spines["left"].set_position(("data", 0))

# 调节透明度：
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_bbox(dict(facecolor="white", edgecolor="red", alpha=0.7, zorder=2))
# set zorder for ordering the plot in plt 2.0.2 or higher
# 在 plt 2.0.2 或更高的版本中, 设置 zorder 给 plot 在 z 轴方向排序
# label.set_fontsize(12)重新调节字体大小, bbox设置目的内容的透明度相关参, facecolor调节box前景色, edgecolor设置边框, alpha设置透明度

plt.show()
