# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


# 用这样 3x3 的 2D-array 来表示点的颜色, 每一个点就是一个pixel

a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
              0.365348418405, 0.439599930621, 0.525083754405,
              0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)

plt.imshow(a, interpolation="nearest", cmap="bone", origin="lower")    # 选择的原点的位置
# interpolation:(出图方式)
# methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
#            'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
#            'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

plt.colorbar(shrink=.92)    # 添加一个colorbar, 其中我们添加一个shrink参数, 使colorbar的长度变短为原来的92%
plt.xticks(())
plt.yticks(())
plt.show()
