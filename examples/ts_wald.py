# -*- coding: utf-8 -*-
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=3, linewidth=256)

from dyconnmap.ts import wald


import matplotlib.pyplot as plt


if __name__ == "__main__":
    data = np.load("../examples/data/wald_test_ts.npy").item()
    x = data['x']
    y = data['y']

    print(np.shape(x))

    # Test
    w, r, e, we = wald(x, y)

    print(w)

    # Plot
    e = e - 1

    data = np.vstack((x, y))

    n_range = np.max(we) - np.min(we)
    we = (we - np.min(we) / n_range)

    plt.figure()
    for i in range(len(e)):
        x1, y1 = data[e[i, 0], 0], data[e[i, 1], 0]
        x2, y2 = data[e[i, 0], 1], data[e[i, 1], 1]
        plt.plot([x1, y1], [x2, y2], c='k', linewidth=1.5 * (0.25 + we[i]), zorder=1)

    plt.scatter(data[e, 0], data[e, 1],
                    s=50, edgecolor='w', zorder=1000)
    plt.axis('square')
    plt.show()
