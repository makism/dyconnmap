# -*- coding: utf-8 -*-
""" Dwell Time

Dwell time measures the time (when used in the context of functional connectivity
microstates) a which a state is active consecutive temporal segments (Dimitriadis2019_).


|

.. [Dimitriadis2019] Dimitriadis, S. I., López, M. E., Maestu, F., & Pereda, E. (2019). Modeling the Switching behavior of Functional Connectivity Microstates (FCμstates) as a Novel Biomarker for Mild Cognitive Impairment. Frontiers in Neuroscience, 13.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np


def dwell_time(x):
    """ Dwell Time

    Compute the dwell time for the given symbolic, 1d time series.


    Parameters
    ----------
    x : array-like, shape(N)
        Input symbolic time series.


    Returns
    -------
    dwell : dictionary
        KVP, where K=symbol id and V=array of dwell time.

    mean : dictionary
        KVP, where K=symbol id and V=mean dwell time.

    std : dictionary
        KVP, where K=symbol id and V=std dwell time.

    """
    data = x
    symbols = np.unique(data)

    dwell = {}
    dwell_mean = {}
    dwell_std = {}
    for symbol in symbols:
        r = np.where(data == symbol)[0]

        r_diff = np.diff(r)
        r_diff_without_one = np.where(r_diff != 1)

        x = r[r_diff_without_one]
        segments = len(x)

        dur = np.zeros((segments, 1))

        len_r = len(r)
        tmp1 = np.squeeze(x)
        tmp2 = r[len_r - 1]

        xx = np.hstack([tmp1, tmp2])
        for l in range(segments - 1):
            r1 = np.where(r == xx[l + 1])[0]
            r2 = np.where(r == xx[l])[0]
            dur[l] = r1 - r2

        r1 = np.where(r == xx[segments])[0]
        r2 = np.where(r == xx[segments - 1])[0]
        dur[segments - 1] = r1 - r2 + 1

        dwell[symbol] = dur / len(data)
        dwell_mean[symbol] = np.mean(dur) / len(data)
        dwell_std[symbol] = np.std(dur) / len(data)

    return (dwell, dwell_mean, dwell_std)
