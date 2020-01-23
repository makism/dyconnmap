# -*- coding: utf-8 -*-
""" Partial Correlation


"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from ..analytic_signal import analytic_signal

import numpy as np


def partcorr(data, fb, fs, pairs=None):
    """ Partial correlation

    Parameters
    ----------


    Returns
    -------

    """
    n_channels, _ = np.shape(data)
    filtered, _, _ = analytic_signal(data, fb, fs)

    P_corr = np.zeros((n_channels, n_channels), dtype=np.float32)

    r = np.cov(filtered, rowvar=False)
    r = np.float32(r)

    rinv = np.linalg.inv(r)
    rinv = np.float32(rinv)

    for i in range(n_channels):
        for ii in range(n_channels):
            P_corr[i, ii] = -rinv[i, ii] / \
                np.float32(np.sqrt(rinv[i, i] * rinv[ii, ii]))

    P_corr = np.float32(P_corr)

    return P_corr
