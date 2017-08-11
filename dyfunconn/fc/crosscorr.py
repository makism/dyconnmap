# -*- coding: utf-8 -*-
""" Cross Correlation


see @https://docs.scipy.org/doc/numpy/reference/generated/numpy.correlate.html

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from .estimator import Estimator
from ..analytic_signal import analytic_signal

import numpy as np


def crosscorr(data, fb, fs, pairs=None):
    """

    """
    n_channels, n_samples = np.shape(data)
    filtered, _, _ = analytic_signal(data, fb, fs)

    r = np.zeros([n_channels, n_channels], dtype=np.float32)

    for i in range(n_channels):
        for ii in range(n_channels):
            r[i, ii] = np.correlate(filtered[i, ], filtered[ii, ], mode='valid')

    return r


class CrossCorr(Estimator):
    """ Cross correlation

    """
    pass
