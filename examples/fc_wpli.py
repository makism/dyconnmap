# -*- coding: utf-8 -*-
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
np.set_printoptions(precision=3, linewidth=256)

from dyconnmap.fc import wpli, dwpli


if __name__ == "__main__":
    data = np.load(
        "/home/makism/Github/dyconnmap/examples/data/eeg_32chans_10secs.npy")
    data = data[0:2, :]

    csdparams = {'NFFT': 128, 'noverlap': 128 / 2.0}

    wpliv = wpli(data, [1.0, 4.0], 128.0, **csdparams)
    dwpliv = dwpli(data, [1.0, 4.0], 128.0, **csdparams)

    print(wpliv)

    print(dwpliv)
