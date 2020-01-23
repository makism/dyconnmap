# -*- coding: utf-8 -*-
""" Envelope-to-Signal Correlation


Proposed by Bruns and Eckhorn [Bruns2004_], Envelope-to-Signal Correlation (*ESC*)
is similar to Amplitude Envelope Correlation (:mod:`dyconnmap.fc.aec`), but the
the amplitude of the lower frequency oscillation is signed; and thus the phase
information is preserved.

.. math::
    r_{ESC} = \\text{corr}(\\chi_{lo}, \\alpha_{hi})

Where :math:`\\chi` is the input signal filtered to the frequency band :math:`lo`
and :math:`\\alpha` denotes the Instantaneous Amplitude of the same input signal
at the frequency band :math:`hi`.

|

-----

.. [Bruns2004] Bruns, A., & Eckhorn, R. (2004). Task-related coupling from high-to low-frequency signals among visual cortical areas in human subdural recordings. International Journal of Psychophysiology, 51(2), 97-116.
.. [Penny2008] Penny, W. D., Duzel, E., Miller, K. J., & Ojemann, J. G. (2008). Testing for nested oscillation. Journal of neuroscience methods, 174(1), 50-61. Chicago
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from ..analytic_signal import analytic_signal

import numpy as np


def esc(data, fb_lo, fb_hi, fs):
    """ Envelope-Signal-Correlation

    Estimate the Envelope-Signal-Correlation the given :attr:`data`.


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
    _, _, f = analytic_signal(data, fb_lo, fs)
    h, _, _ = analytic_signal(data, fb_hi, fs)

    escv = np.corrcoef(f, np.abs(h))
    escv = np.float32(escv)

    return escv
