# -*- coding: utf-8 -*-
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
np.set_printoptions(precision=2, linewidth=256)

from dyconnmap import analytic_signal
from dyconnmap.fc import corr, crosscorr, partcorr
from dyconnmap.fc import Corr


if __name__ == "__main__":
    data = np.load("data/eeg_32chans_10secs.npy")
    n_channels, n_samples = np.shape(data)

    fb = [1.0, 4.0]
    fs = 128.0

    # Correlation
    r = corr(data, fb, fs)
    print(r)

    # Partial correlation
    # pr = partcorr(data, fb, fs)
    # print(pr)

    ro = Corr(fb, fs)
    pp_data = ro.preprocess(data)
    r = ro.estimate(pp_data)

    print(r)
