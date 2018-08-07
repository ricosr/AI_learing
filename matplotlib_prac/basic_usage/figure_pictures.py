# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

x1 = np.linspace(-3, 3, 50)
y1 = 2*x1 + 1
y2 = x1**2

plt.figure()
plt.plot(x1, y1)
plt.show()

plt.figure(num=3, figsize=(8, 5),)    # 编号为3；大小为(8, 5)
plt.plot(x1, y2)
plt.plot(x1, y1, color="red", linewidth=2.0, linestyle="-.")
# [‘solid’ | ‘dashed’, ‘dashdot’, ‘dotted’ | (offset, on-off-dash-seq) | '-' | '--' | '-.' | ':' | 'None' | ' ' | '']
plt.show()
