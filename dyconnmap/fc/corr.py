# -*- coding: utf-8 -*-
""" Correlation

@see https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
from .estimator import Estimator
from ..analytic_signal import analytic_signal


def corr(data, fb=None, fs=None, pairs=None):
    """ Correlation

    Compute the correlation for the given :attr:`data`, between the :attr:`pairs` (if given)
    of channels.


    Parameters
    ----------
    data : array-like, shape(n_rois, n_samples)
        Multichannel recording data.

    fb : list of length 2, optional
        The low and high frequencies.

    fs : float, optional
        Sampling frequency.

    pairs : array-like or `None`
        - If an `array-like` is given, notice that each element is a tuple of length two.
        - If `None` is passed, complete connectivity will be assumed.


    Returns
    -------
    r : array-like, shape(n_rois, n_rois)
        Estimated correlation values.

    See also
    --------
    dyconnmap.fc.Corr: Correlation
    """
    X = None
    if fb is not None and fs is not None:
        _, _, filtered = analytic_signal(data, fb, fs)
        X = filtered
    else:
        X = data

    r = np.corrcoef(X[:])
    r = np.float32(r)

    return r


class Corr(Estimator):
    """ Correlation


    See also
    --------
    dyconnmap.fc.corr: Correlation
    dyconnmap.tvfcg: Time-Varying Functional Connectivity Graphs
    """

    def __init__(self, fb=None, fs=None, pairs=None):
        Estimator.__init__(self, fb, fs, pairs)

    def preprocess(self, data):
        if self._skip_filter:
            return data
        else:
            _, _, filtered = analytic_signal(data, self.fb, self.fs)
            return filtered

    def estimate_pair(self, signal1, signal2):
        """

        Returns
        -------
        r : array-like, shape(1, n_samples)
            Estimated correlation values (real valued).

        _ : None
            None.


        Notes
        -----
        Called from :mod:`dyconnmap.tvfcgs.tvfcg`.
        """
        # n_samples = len(ts1)

        r = np.corrcoef(signal1, signal2)[0, 1]

        return r, None

    def mean(self, value):
        return value

    def estimate(self, data, data_against=None):
        """


        Returns
        -------
        r : array-like, shape(n_rois, n_rois, n_samples)
            Estimated correlation values.


        Notes
        -----
        Called from :mod:`dyconnmap.tvfcgs.tvfcg`.
        """
        n_rois, _ = np.shape(data)

        r = np.zeros((n_rois, n_rois), dtype=self.data_type)

        super().prepare_pairs(n_rois, True)

        # if self.pairs is None:
        # self.pairs = [(r1, r2) for r1 in range(n_rois) for r2 in range(n_rois)]

        for pair in self.pairs:
            f_data1, f_data2 = data[pair,]
            r[pair] = np.corrcoef(f_data1, f_data2)[0, 1]

        return r
