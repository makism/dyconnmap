# -*- coding: utf-8 -*-
""" Correlation

@see https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from .estimator import Estimator
from ..analytic_signal import analytic_signal

import numpy as np


def corr(data, fb, fs, pairs=None):
    """

    """
    n_channels, n_samples = np.shape(data)
    filtered, _, _ = analytic_signal(data, fb, fs)

    r = np.corrcoef(filtered[:])
    r = np.float32(r)

    return r


class Corr(Estimator):
    """ Phase Locking Value (PLV)


    See also
    --------
    dyfunconn.fc.corr: Correlation
    dyfunconn.tvfcg: Time-Varying Functional Connectivity Graphs
    """

    def __init__(self, fb, fs, pairs=None):
        Estimator.__init__(self, fs, pairs)

        self.fb = fb
        self.fs = fs

    def preprocess(self, data):
        filtered, _, _ = analytic_signal(data, self.fb, self.fs)

        return filtered

    def estimate_pair(self, ts1, ts2):
        """

        Returns
        -------
        ts : array-like, shape(1, n_samples)
            Estimated Corr time series (real valued).

        avg : float
            Average.


        Notes
        -----
        Called from :mod:`dyfunconn.tvfcgs.tvfcg`.
        """
        n_samples = len(ts1)

        ts = None
        avg = None

        return ts, avg

    def estimate(self, data):
        """


        Returns
        -------
        ts : complex array-like, shape(n_channels, n_channels, n_samples)
            Estimated Corr time series.

        avg : array-like, shape(n_channels, n_channels)
            Average corr


        Notes
        -----
        Called from :mod:`dyfunconn.tvfcgs.tvfcg`.
        """
        n_channels, n_samples = np.shape(data)

        ts = np.zeros((n_channels, n_channels, n_samples), dtype=np.complex)
        avg = np.zeros((n_channels, n_channels))

        if self.pairs is None:
            self.pairs = [(r1, r2) for r1 in range(n_channels)
                          for r2 in range(r1, n_channels)
                          if r1 != r2]

        for pair in self.pairs:
            u_phases1, u_phases2 = data[pair, ]
            ts = None
            avg = None

            ts[pair] = ts
            avg[pair] = avg

        return ts, avg
