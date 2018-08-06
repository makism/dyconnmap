# -*- coding: utf-8 -*-
""" Correlation

@see https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from .estimator import Estimator
from ..analytic_signal import analytic_signal

import numpy as np


def corr(data, fb, fs, pairs=None):
    """ Correlation

    Compute the correlation for the given :attr:`data`, between the :attr:`pairs` (if given)
    of channels.


    Parameters
    ----------
    data : array-like, shape(n_channels, n_samples)
        Multichannel recording data.

    fb : list of length 2
        The lower and upper frequency.

    fs : float
        Sampling frequency.

    pairs : array-like or `None`
        - If an `array-like` is given, notice that each element is a tuple of length two.
        - If `None` is passed, complete connectivity will be assumed.


    Returns
    -------
    r : array-like, shape(n_channels, n_channels)
        Estimated correlation values.

    See also
    --------
    dyfunconn.fc.Corr: Correlation (Class Estimator)
    """
    filtered, _, _ = analytic_signal(data, fb, fs)

    r = np.corrcoef(filtered[:])
    r = np.float32(r)

    return r


class Corr(Estimator):
    """ Correlation


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
        r : array-like, shape(1, n_samples)
            Estimated correlation values (real valued).

        _ : None
            None.


        Notes
        -----
        Called from :mod:`dyfunconn.tvfcgs.tvfcg`.
        """
        # n_samples = len(ts1)

        r = np.corrcoef(ts1, ts2)[0, 1]

        return r, None

    def mean(self, value):
        return value

    def estimate(self, data):
        """


        Returns
        -------
        r : array-like, shape(n_channels, n_channels, n_samples)
            Estimated correlation values.


        Notes
        -----
        Called from :mod:`dyfunconn.tvfcgs.tvfcg`.
        """
        n_channels, n_samples = np.shape(data)

        r = np.zeros((n_channels, n_channels), dtype=self.data_type)

        if self.pairs is None:
            self.pairs = [(r1, r2) for r1 in range(n_channels)
                          for r2 in range(n_channels)]

        for pair in self.pairs:
            f_data1, f_data2 = data[pair, ]
            r[pair] = np.corrcoef(f_data1, f_data2)[0, 1]

        return r
