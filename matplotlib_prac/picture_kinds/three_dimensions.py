# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D    # 用于3D坐标轴显示


fig = plt.figure()
ax = Axes3D(fig)    # 在窗口上添加3D坐标轴

X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)    # 形成x-y网格
R = np.sqrt(X**2 + Y**2)    # height
Z = np.sin(R)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap("rainbow"))    # rstride和cstride分别代表row和column的跨度
"""
============= ================================================
        Argument      Description
        ============= ================================================
        *X*, *Y*, *Z* Data values as 2D arrays
        *rstride*     Array row stride (step size), defaults to 10
        *cstride*     Array column stride (step size), defaults to 10
        *color*       Color of the surface patches
        *cmap*        A colormap for the surface patches.
        *facecolors*  Face colors for the individual patches
        *norm*        An instance of Normalize to map values to colors
        *vmin*        Minimum value to map
        *vmax*        Maximum value to map
        *shade*       Whether to shade the facecolors
        ============= ================================================
"""
ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.get_cmap("rainbow"))    # zdir选择了z, 那么效果将会是对于XY平面的投影
"""
==========  ================================================
        Argument    Description
        ==========  ================================================
        *X*, *Y*,   Data values as numpy.arrays
        *Z*
        *zdir*      The direction to use: x, y or z (default)
        *offset*    If specified plot a projection of the filled contour
                    on this position in plane normal to zdir
        ==========  ================================================
"""
ax.contour(X, Y, Z, zdir='z', offset=-2, colors="black", linewidths=2)    # 颜色选黑色，线条宽度选0.5
ax.set_zlim(-2, 2)

plt.show()
