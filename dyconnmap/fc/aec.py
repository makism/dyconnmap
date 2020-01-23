# -*- coding: utf-8 -*-
""" Amplitude Envelope Correlation

Amplitude Envelope Correlation (*AEC*), estimates the coupling
(without phase coherence and even among different frequencies [Bruns2000_]) by
computing the correlation coefficient of a signal's amplitude envelope.

.. math::
   r_{AEC} = \\text{corr}(\\alpha_{lo}, \\alpha_{hi})

Where :math:`\\alpha` denotes the Instantaneous Amplitude of a given signal,
filtered in a specfic frequency band (:math:`lo` or :math:`hi`).


|

-----

.. [Bruns2000] Bruns, A., Eckhorn, R., Jokeit, H., & Ebner, A. (2000). Amplitude envelope correlation detects coupling among incoherent brain signals. Neuroreport, 11(7), 1509-1514.
.. [Penny2008] Penny, W. D., Duzel, E., Miller, K. J., & Ojemann, J. G. (2008). Testing for nested oscillation. Journal of neuroscience methods, 174(1), 50-61.
.. [Friston1996] Friston, K. J. (1997). Another neural code?. Neuroimage, 5(3), 213-220.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from ..analytic_signal import analytic_signal

import numpy as np


def aec(data, fb_lo, fb_hi, fs):
    """ Amplitude Envelope Correlation

    Estimate the Amplitude-Envelope Correlation for the given :attr:`data`.


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


    Returns
    -------
    r : array-like, shape(n_channels, n_channels)
        Estimated Pearson correlation coefficient.
    """
    h_lo, _, _ = analytic_signal(data, fb_lo, fs)
    h_hi, _, _ = analytic_signal(data, fb_hi, fs)

    r = np.corrcoef(np.abs(h_lo), np.abs(h_hi))
    r = np.float32(r)

    return r
