# -*- coding: utf-8 -*-
""" Phase Locking Value

One of the pioneer methods called Phase Locking Value (*PLV*) is discussed in
[Lachaux1998]_; it utilizes the Hilbert representation (consult
:py:mod:`dyconnmap.analytic_signal` for more details) an EEG time
series (of :math:`N_{sensors}`) and quantifies their interaction based on their
instantaneous phase in a specific band frequency.

So, for a pair of Instantaneous Phases of two time series of equal length,
:math:`\phi_{j1}(t)` and :math:`\phi_{j2}(t)`, the Phase Locking Value for each
sample in time (:math:`t`) is computed as:

.. math::
    e^{i (\phi_{j1}(t)  - \phi_{j2}(t))}

A value of zero means that no coupling (or negligible) observed between two
phases, while a value of one denotes a perfect synchronization.

|

-----

.. [Lachaux1998] Lachaux, J., Rodriguez, E., Martinerie, J., Varela, F., & others,. (1999). Measuring phase synchrony in brain signals. Human Brain Mapping, 8(4), 194-208.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from .estimator import Estimator
from ..analytic_signal import analytic_signal

import numpy as np


def plv_fast(data, pairs=None):
    """ Phase Locking Value

    """
    _, n_samples = np.shape(data)

    _, u_phases = analytic_signal(data)
    Q = np.exp(1j * u_phases)

    Q = np.matrix(Q)
    W = np.abs(Q @ Q.conj().transpose()) / np.float32(n_samples)

    return W


def plv(data, fb=None, fs=None, pairs=None):
    """ Phase Locking Value

    Compute the PLV for the given :attr:`data`, between the :attr:`pairs` (if given)
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
    ts : array-like, shape(n_rois, n_rois, n_samples)
        Estimated PLV time series.

    avg : array-like, shape(n_rois, n_rois)
        Average PLV.


    See also
    --------
    dyconnmap.fc.PLV: Phase Locking Value (Class Estimator)
    dyconnmap.fc.iplv: Imaginary part of PLV
    dyconnmap.fc.pli: Phase Lag Index
    """
    estimator = PLV(fb, fs, pairs)
    pp_data = estimator.preprocess(data)

    return estimator.estimate(pp_data)


class PLV(Estimator):
    """ Phase Locking Value (PLV)


    See also
    --------
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

    def estimate_pair(self, signal1, signal2):
        """

        Returns
        -------
        ts : array-like, shape(1, n_samples)
            Estimated PLV time series (real valued).

        avg : float
            Average PLV.


        Notes
        -----
        Called from :mod:`dyconnmap.tvfcgs.tvfcg`.
        """
        n_samples = len(signal1)

        ts_plv = np.exp(1j * (signal1 - signal2))
        avg_plv = np.abs(np.sum((ts_plv))) / float(n_samples)

        return ts_plv, avg_plv

    def mean(self, ts):
        l = float(np.shape(ts)[0])
        return np.abs(np.sum(ts)) / l

    def estimate(self, data, data_against=None):
        """


        Returns
        -------
        ts : complex array-like, shape(n_channels, n_channels, n_samples)
            Estimated PLV time series (complex valued).

        avg : array-like, shape(n_channels, n_channels)
            Average PLV.


        Notes
        -----
        Called from :mod:`dyconnmap.tvfcgs.tvfcg`.
        """
        n_rois, n_samples = np.shape(data)

        ts = np.zeros((n_rois, n_rois, n_samples), dtype=np.complex)
        avg = np.zeros((n_rois, n_rois))

        super().prepare_pairs(n_rois)

        for pair in self.pairs:
            u_phases1, u_phases2 = data[pair,]
            ts_plv = np.exp(1j * (u_phases1 - u_phases2))
            avg_plv = np.abs(np.sum((ts_plv))) / float(n_samples)

            ts[pair] = ts_plv
            avg[pair] = avg_plv

        return ts, avg
