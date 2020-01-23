# -*- coding: utf-8 -*-
""" Imaginary Coherence


Imaginary Coherence (*ICoh*)


Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., & Hallett, M. (2004). Identifying true brain interaction from EEG data using the imaginary part of coherency. Clinical neurophysiology, 115(10), 2292-2307.
Chicago


"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from .estimator import Estimator
from ..analytic_signal import analytic_signal

from matplotlib import mlab
import numpy as np


def icoherence(data, fb, fs, pairs=None, **kwargs):
    """ Imaginary Coherence

    Compute the Imaginary part of Coherence for the given :attr:`data`,
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
    icoh : array-like, shape(n_channels, n_channels)
        Estimated Imaginary part of Coherence.

    See also
    --------
    dyconnmap.fc.coherence: Coherence
    """
    n_channels, _ = np.shape(data)
    _, _, filtered = analytic_signal(data, fb, fs)

    if pairs is None:
        pairs = [(r1, r2) for r1 in range(n_channels) for r2 in range(n_channels)]

    icoh = np.zeros((n_channels, n_channels))

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

        num = np.sum(np.abs(np.imag(csdxy)))
        denom = np.sqrt(np.sum(np.abs(csdxx)) * np.sum(np.abs(csdyy)))

        icoh[pair] = num / denom

    return icoh


class ICoherence(Estimator):
    """ Imaginary Coherence


    """

    def __init__(self, fb, fs, pairs=None):
        Estimator.__init__(self, fs, pairs)

        self.csd_nfft = 256
        self.csd_noverlap = self.csd_nfft / 2.0
        self.fb = fb
        self.fs = fs

    def preprocess(self, data):
        n_channels, _ = np.shape(data)

        _, _, filtered = analytic_signal(data, self.fb, self.fs)

        super().prepare_pairs(n_channels)

        # Store all the pair-wise auto/cross spectra.
        #  2nd axis, 1st dimension is the autospectra of the 1st channel (within a pair)
        #  2nd axis, 2nd dimension is the autospectra of the 2nd channel (within a pair)
        #  2nd axis, 3rd dimension is the crosspectra between the two channels (within a pair)
        samples = (self.csd_nfft / 2.0) + 1
        csds = np.zeros((n_channels, n_channels, 3, samples))

        for pair in self.pairs:
            filt1, filt2 = filtered[pair,]

            csdxx, _ = mlab.csd(
                filt1,
                filt1,
                NFFT=self.csd_nfft,
                Fs=self.fs,
                noverlap=self.csd_noverlap,
                scale_by_freq=True,
                sides="onesided",
            )
            csdyy, _ = mlab.csd(
                filt2,
                filt2,
                NFFT=self.csd_nfft,
                Fs=self.fs,
                noverlap=self.csd_noverlap,
                scale_by_freq=True,
                sides="onesided",
            )
            csdxy, _ = mlab.csd(
                filt1,
                filt2,
                NFFT=self.csd_nfft,
                Fs=self.fs,
                noverlap=self.csd_noverlap,
                scale_by_freq=True,
                sides="onesided",
            )

            r1, r2 = pair
            csds[r1, r2, 0] = csdxx
            csds[r1, r2, 1] = csdyy
            csds[r1, r2, 2] = csdxy

        return csds

    def estimate_pair(self, signal1, signal2):
        csdxx = None
        csdyy = None
        csdxy = None

        # num = np.sum(np.abs(np.imag(csdxy)))
        # denom = np.sqrt(np.sum(np.abs(csdxx)) * np.sum(np.abs(csdyy)))
        # icoh = num / denom

        icoh = 0.0

        return icoh, 0.0

    def estimate(self, data, data_against=None):
        return None
