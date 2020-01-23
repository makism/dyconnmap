# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(precision=3, linewidth=256)

from dyconnmap.ts import teager_kaiser_energy


if __name__ == "__main__":
    ts = np.load('/home/makism/Github/dyconnmap/examples/data/10secs.npy')
    ts = ts[0:1, 0:128].ravel()

    teo = teager_kaiser_energy(ts)

    print(teo)
