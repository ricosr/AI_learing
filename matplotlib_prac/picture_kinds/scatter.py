# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

n = 5
X = np.random.normal(0, 1, n)
print(X)
Y = np.random.normal(0, 1, n)
print(Y)
T = np.arctan2(Y, X)     # for color value
print(T)

plt.scatter(X, Y, s=75, c=T, alpha=.5)    # size=75, 颜色为T, 透明度alpha 为 50%

plt.xlim(-1.5, 1.5)
plt.xticks(())     # ignore xticks
plt.ylim(-1.5, 1.5)
plt.yticks(())

plt.show()
