# -*- coding: utf-8 -*-
import time

import numpy as np
np.set_printoptions(precision=3, linewidth=160, suppress=True)

from dyconnmap.fc import Corr
from dyconnmap import tvfcg


if __name__ == '__main__':
    data = np.load(
        "/home/makism/Github/dyconnmap/examples/data/eeg_32chans_10secs.npy")

    n_channels, n_samples = np.shape(data)

    fb = [1.0, 4.0]
    cc = 3.0
    fs = 128
    step = 5
    estimator = Corr(fb, fs)

    fcgs = tvfcg(data, estimator, fb, fs)

    print(fcgs)
