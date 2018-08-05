# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 2)    # 定义区间
print(x)
y = 2*x + 1

plt.figure()    # 定义一个图像窗口
plt.plot(x, y)    # 画曲线
plt.show()    # 显示图像
