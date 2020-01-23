# -*- coding: utf-8 -*-
""" Power-Envelope Correlation

Similarly to :mod:`dyconnmap.fc.aec`, we can use the followig formula to estimate the correlations in power
between the different frequency bands [Friston1996_].

.. math::
   r_{PAEC} = \\text{corr}(\\alpha_{lo}^2, \\alpha_{hi}^2)


|

-----

.. [Hipp2012] Hipp, J. F., Hawellek, D. J., Corbetta, M., Siegel, M., & Engel, A. K. (2012). Large-scale cortical correlation structure of spontaneous oscillatory activity. Nature neuroscience, 15(6), 884-890.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from ..analytic_signal import analytic_signal

import numpy as np


def pec(data, fb_lo, fb_hi, fs):
    """ Power Envelope Correlation


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

    r = np.corrcoef(np.power(np.abs(h_lo), 2.0), np.power(np.abs(h_hi), 2.0))
    r = np.float32(r)

    return r
