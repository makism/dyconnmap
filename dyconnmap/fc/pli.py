# -*- coding: utf-8 -*-
""" Phase Lag Index

Phase Lag Index (*PLI*) [Stam2007_], proposed  as an alternative (to PLV) phase
synchronization estimator that is less prone to the effects of common sources
(namely, volume conduction and active reference electrodes). These effects can
artificially generate functional connectivity as the same signal signal is
measured at different electrodes [Hardmeier2014_].

PLI estimates the asymmetry in the distribution of two time series' instantaneous phase differences.

Given two time series of equal length :math:`x(t)` and :math:`y(t)`, we extract
their respective instantaneous phases :math:`\phi_x(t)` and :math:`\phi_y(t)`
using the Hilbert transform (consult :py:mod:`dyconnmap.analytic_signal` for
more details).
Then, for such a pair of phases, PLI is computed as follows:

.. math::
    PLI = | \\left \\langle sign [ sin ( \\phi_x(t) - \\phi_y(t) ) ] \\right \\rangle |

Where, :math:`sign` refers to the signum function, \\left \\langle \\right \\rangle
denotes the mean value and || the absolute value.

|

-----


.. [Stam2007] Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). Phase lag index: assessment of functional connectivity from multi channel EEG and MEG with diminished bias from common sources. Human brain mapping, 28(11), 1178-1193.
.. [Hardmeier2014] Hardmeier, M., Hatz, F., Bousleiman, H., Schindler, C., Stam, C. J., & Fuhr, P. (2014). Reproducibility of functional connectivity and graph measures based on the phase lag index (PLI) and weighted phase lag index (wPLI) derived from high resolution EEG. PloS one, 9(10), e108648.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
from .estimator import Estimator
from ..analytic_signal import analytic_signal


def pli(data, fb=None, fs=None, pairs=None):
    """ Phase Lag Index

    Compute the PLI for the given :attr:`data`, between the :attr:`pairs` (if given)
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
        Estimated PLI time series.

    avg : array-like, shape(n_rois, n_rois)
        Average PLI.


    See also
    --------
    dyconnmap.fc.PLI: Phase Lag Index
    """
    estimator = PLI(fb, fs, pairs)
    pp_data = estimator.preprocess(data)

    return estimator.estimate(pp_data)


class PLI(Estimator):
    """ Phase Lag Index (PLI)


    """

    def __init__(self, fb=None, fs=None, pairs=None):
        Estimator.__init__(self, fb, fs, pairs)

    def preprocess(self, data):
        if self._skip_filter:
            _, u_phases = analytic_signal(data)
        else:
            _, u_phases, _ = analytic_signal(data, self.fb, self.fs)

        return u_phases

    def mean(self, ts):
        return np.abs(np.mean(ts))

    def estimate_pair(self, signal1, signal2):
        ts_pli = np.sin(signal1 - signal2)
        avg_pli = np.abs(np.mean(np.sign(ts_pli)))

        return ts_pli, avg_pli

    def estimate(self, data, data_against=None):
        n_rois, n_samples = np.shape(data)

        # if self.pairs is None:
        # self.pairs = [
        # (r1, r2) for r1 in range(n_rois) for r2 in range(r1, n_rois) if r1 != r2
        # ]

        super().prepare_pairs(n_rois)

        ts = np.zeros((n_rois, n_rois, n_samples))
        avg = np.zeros((n_rois, n_rois))

        for pair in self.pairs:
            u_phases1, u_phases2 = data[pair,]
            ts_pli = np.sin(u_phases1 - u_phases2)
            avg_pli = np.abs(np.sum(np.sign(ts_pli)) / float(n_samples))

            ts[pair] = ts_pli
            avg[pair] = np.squeeze(avg_pli)

        return ts, avg
