# -*- coding: utf-8 -*-
""" Imaginary part of Phase Locking Value


Imaginary Phase Locking Value (*IPLV*) was proposed to resolve PLV's sensitivity to
volume conduction and common reference effects.

IPLV is computed similarly as PLV, but taking the imaginary part of the summation:

.. math::
    ImPLV = \\frac{1}{N} \\left | Im \\left ( \\sum_{t=1}^{N} e^{i (\phi_{j1}(t)  - \phi_{j2}(t))} \\right ) \\right |


|

-----

.. [Sadaghiani2012] Sadaghiani, S., Scheeringa, R., Lehongre, K., Morillon, B., Giraud, A. L., D'Esposito, M., & Kleinschmidt, A. (2012). Alpha-band phase synchrony is related to activity in the fronto-parietal adaptive control network. The Journal of Neuroscience, 32(41), 14305-14310.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from .estimator import Estimator
from ..analytic_signal import analytic_signal

import numpy as np


def iplv_fast(data, pairs=None):
    """ Imaginary part of Phase Locking Value

    """
    _, n_samples = np.shape(data)

    _, u_phases = analytic_signal(data)
    Q = np.exp(1j * u_phases)

    Q = np.matrix(Q)
    W = np.abs(np.imag(Q @ Q.conj().transpose())) / np.float32(n_samples)

    return W


def iplv(data, fb=None, fs=None, pairs=None):
    """ Imaginary part of Phase Locking Value

    Compute the Imaginary part of Phase Locking Value for the given *data*,
    between the *pairs* (if given) of rois.


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
    ts : array-like, shape(n_rois, n_rois, n_samples)
        Estimated IPLV time series.

    avg : array-like, shape(n_rois, n_rois)
        Average IPLV.
    """
    iplv = IPLV(fb, fs, pairs)
    pp_data = iplv.preprocess(data)

    return iplv.estimate(pp_data)


class IPLV(Estimator):
    """ Imaginary part of PLV (iPLV)


    See also
    --------
    dyconnmap.fc.iplv: Imaginary part of PLV
    dyconnmap.fc.plv: Phase Locking Value
    dyconnmap.tvfcg: Time-Varying Functional Connectivity Graphs
    """

    def __init__(self, fb=None, fs=None, pairs=None):
        Estimator.__init__(self, fb, fs, pairs)
        self.data_type = np.complex

    def preprocess(self, data):
        if self._skip_filter:
            _, u_phases = analytic_signal(data)
        else:
            _, u_phases, _ = analytic_signal(data, self.fb, self.fs)

        return u_phases

    def estimate_pair(self, ts1, ts2):
        """

        Returns
        -------
        ts : array-like, shape(1, n_samples)
            Estimated iPLV time series.

        avg : float
            Average iPLV.


        Notes
        -----
        Called from :mod:`dyconnmap.tvfcgs.tvfcg`.
        """
        n_samples = len(ts1)

        ts_plv = np.exp(1j * (ts1 - ts2))
        avg_plv = np.abs(np.imag(np.sum((ts_plv)))) / float(n_samples)

        return np.imag(ts_plv), avg_plv

    def mean(self, ts):
        l = float(np.shape(ts)[0])
        return np.abs(np.sum(ts)) / l

    def estimate(self, data):
        """

        Returns
        -------
        ts :

        avg:


        Notes
        -----
        Called from :mod:`dyconnmap.tvfcgs.tvfcg`.
        """
        n_rois, n_samples = np.shape(data)

        ts = np.zeros((n_rois, n_rois, n_samples), dtype=np.complex)
        avg = np.zeros((n_rois, n_rois))

        if self.pairs is None:
            self.pairs = [
                (r1, r2) for r1 in range(n_rois) for r2 in range(r1, n_rois) if r1 != r2
            ]

        for pair in self.pairs:
            u_phases1, u_phases2 = data[pair,]
            ts_plv = np.exp(1j * (u_phases1 - u_phases2))
            avg_plv = np.abs(np.imag(np.sum((ts_plv)))) / float(n_samples)

            ts[pair] = ts_plv
            avg[pair] = avg_plv

        return ts, avg
