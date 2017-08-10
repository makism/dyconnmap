# -*- coding: utf-8 -*-
""" Partial Correlation


"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from ..analytic_signal import analytic_signal

import numpy as np


def partcorr(data, fb, fs, pairs=None):
    """

    """
    n_channels, n_samples = np.shape(data)
    filtered, _, _ = analytic_signal(data, fb, fs)

    P_corr = np.zeros((n_channels, n_channels), dtype=np.float32)

    r = np.corrcoef(filtered)
    rinv = np.linalg.inv(r)

    for i in range(n_channels):
        for ii in range(n_channels):
            P_corr[i, ii] = -rinv[i, ii] / \
                (np.sqrt(rinv[i, i] * rinv[ii, ii]))

    return P_corr
