# -*- coding: utf-8 -*-
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
np.set_printoptions(precision=2, linewidth=256)

from dyfunconn import analytic_signal
from dyfunconn.fc import corr, crosscorr, partcorr


if __name__ == "__main__":
    data = np.load("data/eeg_32chans_10secs.npy")
    n_channels, n_samples = np.shape(data)

    fb = [1.0, 4.0]
    fs = 128.0

    filtered, _, _ = analytic_signal(data, fb, fs)

    # Correlation
    r = corr(data, fb, fs)
    print(r)

    # Cross correlation
    # xr = np.correlate(filtered[0, ], filtered[1, ])#, mode='valid')
    # xr1 = crosscorr(data, fb, fs)

    # Partial correlation
    pr = partcorr(data, fb, fs)

    # print(xr, xr1)

    print(pr)
