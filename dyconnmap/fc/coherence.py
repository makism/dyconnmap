# -*- coding: utf-8 -*-
""" Coherence

Coherence (*Coh*) is one of the most commonly utilized connectivity estimators; it is a
measurement of the linear relationship of two signals at a specific frequency [Nolte2004_].

Given two time series :math:`x` and :math:`y`, coherece is given by:

.. math::
    coh^2_{xy}(f) = \\frac{ |G_{xy}(f)^2| }{ G_{xx}(f) G_{yy}(f) }

Where :math:`G_{xy}(f)` is the estimated cross-spectral density between :math:`x` and
:math:`y`, while :math:`G_{xx}(f)` and :math:`G_{yy}(f)` are the autospectrum of
:math:`x` and :math:`y` respectively.

The result is a symmetric matrix of size :math:`[n\_channels \\times n\_channels]`
bearing no information about the directionality of the interaction, with values
within the range :math:`[0,1]`.


|

-----

.. [Nolte2004] Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., & Hallett, M. (2004). Identifying true brain interaction from EEG data using the imaginary part of coherency. Clinical neurophysiology, 115(10), 2292-2307.
.. [Thatcher2005] Thatcher, R. W., North, D., & Biver, C. (2005). EEG and intelligence: relations between EEG coherence, EEG phase delay and power. Clinical neurophysiology, 116(9), 2129-2141.
.. [Vinck2011] Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F., & Pennartz, C. M. (2011). An improved index of phase-synchronization for electrophysiological data in the presence of volume-conduction, noise and sample-size bias. Neuroimage, 55(4), 1548-1565.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from .estimator import Estimator
from ..analytic_signal import analytic_signal

import numpy as np
import matplotlib.mlab as mlab


def coherence(data, fb, fs, pairs=None, **kwargs):
    """ Coherence

    Estimate the Coherence for the given :attr:`data`,
    between the :attr:`pairs (if given) of channels.


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

    **kwargs :
        Keyword arguments to be passed to :meth:`matplotlib.mlab.csd`.


    Returns
    -------
    coh : array-like, shape(n_channels, n_channels)
        Estimated Coherence.


    See also
    --------
    dyconnmap.fc.Coherece: Coherece (Class Estimator)
    dyconnmap.fc.icoherence: Imaginary Coherence
    """
    n_channels, _ = np.shape(data)
    _, _, filtered = analytic_signal(data, fb, fs)

    if pairs is None:
        pairs = [(r1, r2) for r1 in range(n_channels) for r2 in range(n_channels)]

    coh = np.zeros((n_channels, n_channels))

    for pair in pairs:
        filt1, filt2 = filtered[pair,]

        csdxx, _ = mlab.csd(
            x=filt1, y=filt1, Fs=fs, scale_by_freq=True, sides="onesided", **kwargs
        )
        csdyy, _ = mlab.csd(
            x=filt2, y=filt2, Fs=fs, scale_by_freq=True, sides="onesided", **kwargs
        )
        csdxy, _ = mlab.csd(
            x=filt1, y=filt2, Fs=fs, scale_by_freq=True, sides="onesided", **kwargs
        )

        cohv = np.abs(csdxy * np.conj(csdxy)) / (csdxx * csdyy)

        coh[pair] = np.sum(cohv) / len(cohv)

    return coh


class Coherence(Estimator):
    """ Coherence

    An :mod:`dyconnmap.fc.Estimator` class that implements :mod:`dyconnmap.fc.coherence`.


    See also
    --------
    dyconnmap.fc.coherence: Coherence
    dyconnmap.tvfcg: Time-Varying Functional Connectivity Graphs
    """

    def __init__(self, fb, fs, pairs=None, **kwargs):
        Estimator.__init__(self, fs, pairs)

        self.fb = fb
        self.pairs = pairs
        self.csdargs = kwargs

    def preprocess(self, data):
        n_channels, _ = np.shape(data)

        _, _, filtered = analytic_signal(data, self.fb, self.fs)

        super().prepare_pairs(n_channels)

        return filtered

    def estimate_pair(self, ts1, ts2):
        csdxx, _ = mlab.csd(
            x=ts1,
            y=ts1,
            Fs=self.fs,
            scale_by_freq=True,
            sides="onesided",
            **self.csdargs
        )
        csdyy, _ = mlab.csd(
            x=ts2,
            y=ts2,
            Fs=self.fs,
            scale_by_freq=True,
            sides="onesided",
            **self.csdargs
        )
        csdxy, _ = mlab.csd(
            x=ts1,
            y=ts2,
            Fs=self.fs,
            scale_by_freq=True,
            sides="onesided",
            **self.csdargs
        )

        cohv = np.abs(csdxy * np.conj(csdxy)) / (csdxx * csdyy)

        return np.sum(cohv) / len(cohv)

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
        n_rois, _ = np.shape(data)

        super().prepare_pairs(n_rois)

        avg = np.zeros((n_rois, n_rois))

        for pair in self.pairs:
            ts1, ts2 = data[pair,]
            avg[pair] = self.estimate_pair(ts1, ts2)

        return (None, avg)
