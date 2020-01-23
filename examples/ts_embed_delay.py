# -*- coding: utf-8 -*-
"""

"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
np.set_printoptions(precision=3, linewidth=256)

from dyconnmap.ts import embed_delay


if __name__ == "__main__":
    ts = np.load('data/ts_lorenz.npy')
    x, y, z = embed_delay(ts, 3, 10).T

    plt.figure()
    plt.plot(ts)
    plt.show()

    figure = plt.figure()
    axes = Axes3D(figure)
    axes.plot3D(x, y, z)
    figure.add_axes(axes)
    plt.show()
