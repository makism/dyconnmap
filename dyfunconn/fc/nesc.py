# -*- coding: utf-8 -*-
""" Amplitude-Normalized Envelope-to-Signal Correlation


-----

.. [Penny2008] Penny, W. D., Duzel, E., Miller, K. J., & Ojemann, J. G. (2008). Testing for nested oscillation. Journal of neuroscience methods, 174(1), 50-61. Chicago
.. [Bruns2004] Bruns, A., & Eckhorn, R. (2004). Task-related coupling from high-to low-frequency signals among visual cortical areas in human subdural recordings. International Journal of Psychophysiology, 51(2), 97-116.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from ..analytic_signal import analytic_signal

import numpy as np


def nesc(data, f_lo, f_hi, fs, pairs=None):
    """ Amplitude-Normalized Envelope-to-Signal-Correlation


    Parameters
    ----------
    data : array-like, shape(n_channels, n_samples)
        Multichannel recording data.

    fb_lo : list of length 2
        The low and high frequencies of the lower band.

    fb_hi : list of length 2
        The low and high frequencies of the upper band.

    fs : float
        Sampling frequency.

    pairs : array-like or `None`
        - If an `array-like` is given, notice that each element is a tuple of length two.
        - If `None` is passed, complete connectivity will be assumed.


    Returns
    -------
    r : array-like, shape(n_channels, n_channels)
        Estimated Pearson correlation coefficient.
    """
    hilbert_lo, _, _ = analytic_signal(data, f_lo, fs)
    hilbert_hi, _, _ = analytic_signal(data, f_hi, fs)

    phi = np.angle(hilbert_lo)

    nesc = np.corrcoef(np.cos(phi), np.abs(hilbert_hi))
    nesc = np.float32(nesc)

    return nesc
