# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

n = 12
X = np.arange(n)
# Y1 = (1 - X/float(n))*np.random.uniform(0.5, 1.0, n)
Y1 = np.random.uniform(0.5, 1.0, n)

Y2 = np.random.uniform(0.5, 1.0, n)
# print(Y1)
# print(Y2)

plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

plt.xlim(-.5, n)
plt.xticks(())
plt.ylim(-1.25, 1.25)
plt.yticks(())

for x, y in zip(X, Y1):
    plt.text(x, y, "%.2f" % y, ha="center", va="bottom")
    # ha: horizontal alignment
    # va: vertical alignment
for x, y in zip(X, Y2):
    plt.text(x, -y, "%.2f" % y, ha="center", va="top")

plt.show()

