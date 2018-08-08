# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# subplot2grid：
plt.figure(num=1)
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
# (3,3)表示将整个图像窗口分成3行3列, (0,0)表示从第0行第0列开始作图,
# colspan=3表示列的跨度为3(占多少列), rowspan=1表示行的跨度为1. colspan和rowspan缺省, 默认跨度为1
ax1.plot([1, 2], [1, 2])
ax1.set_title("ax1_title")

ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0))
ax5 = plt.subplot2grid((3, 3), (2, 1))

ax4.scatter([1, 2], [2, 2])
ax4.set_xlabel("ax4_x")
ax4.set_ylabel("ax4_y")


# GridSpec：
plt.figure(num=2)
gs = gridspec.GridSpec(3, 3)    # 将整个图像窗口分成3行3列
ax6 = plt.subplot(gs[0, :])    # 占第0行和所有列
ax7 = plt.subplot(gs[1, :2])    # 占第1行和第2列前的所有列
ax8 = plt.subplot(gs[1:, 2])    # 占第1行后的所有行和第2列
ax9 = plt.subplot(gs[-1, 0])
ax9 = plt.subplot(gs[-1, 1])


# subplots:
f, ((ax11, ax12), (ax13, ax14)) = plt.subplots(2, 2, sharex=True, sharey=True)
# 使用plt.subplots建立一个2行2列的图像窗口,sharex=True表示共享x轴坐标, sharey=True表示共享y轴坐标.
# ((ax11, ax12), (ax13, ax14))表示第1行从左至右依次放ax11和ax12, 第2行从左至右依次放ax13和ax14.
# f是picture的object, 可操作各种属性
ax11.scatter([1, 2], [1, 2])
ax11.set_xlabel("ax11x")
ax11.set_ylabel("ax11y")
ax12.set_xlabel("ax12x")
ax12.set_ylabel("ax12y")
ax13.set_xlabel("ax13x")
ax13.set_ylabel("ax13y")
ax14.set_xlabel("ax14x")
ax14.set_ylabel("ax14y")
plt.tight_layout()    # 表示紧凑显示图像

plt.show()
