# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

# 大小相同的图：
plt.figure(num=1)
plt.subplot(2, 2, 1)    # 将图分成2x2, 当前操作第一个小块
plt.plot([0, 1], [0, 1])
plt.subplot(2, 2, 2)
plt.plot([0, 1], [0, 2])
plt.subplot(223)
plt.plot([0, 1], [0, 3])
plt.subplot(224)
plt.plot([0, 1], [0, 4])


# 大小不同的图：
plt.figure(num=2)
plt.subplot(2, 1, 1)
plt.plot([0, 1], [0, 1])

plt.subplot(2, 3, 4)    # 因为第一个subplot分了两行, 按照现在的分发, 第一次的第一行已经占了3个位置, 所以现在要选第4个
plt.plot([0, 1], [0, 2])

plt.subplot(235)
plt.plot([0, 1], [0, 3])

plt.subplot(236)
plt.plot([0, 1], [0, 4])

plt.show()
