# -*- coding: utf-8 -*-
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
np.set_printoptions(precision=2, linewidth=256)

import sys
sys.path.append('/opt/src/wmtsa-python')

import wmtsa.modwt

from dyconnmap import analytic_signal



if __name__ == '__main__':
    data = np.load("data/eeg_32chans_10secs.npy")
    n_channels, n_samples = np.shape(data)

    fb = [1.0, 4.0]
    fs = 128.0

    filtered, _, _ = analytic_signal(data, fb, fs)


    wcoef = wmtsa.modwt.modwt(filtered, wtf='la8', nlevels='conservative', boundary='reflection', RetainVJ=False)
