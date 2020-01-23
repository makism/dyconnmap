# -*- coding: utf-8 -*-
""" General Linear Model


General linear modeling (*GLM*) is used widely in neuroimaging [Penny2006_] to detect
coupling between a low and higher frequency.

.. math::
    \\chi_{hf} = X_{\\beta} + e

Where :math:`\\beta` are the corresponding regression coefficients and
:math:`e` is the additive Gaussian noise. Finally, :math:`X` is the design
matrix of size :math:`n \\times 3` (:math:`n` the number of samples). Columns
1 and 2, contain the cosines and sines counterparts of the instantaneous
phases (of the low frequency) of the predictors, while the third row only 1s.

|

-----

.. [Penny2006] Penny, W. D., Friston, K. J., Ashburner, J. T., Kiebel, S. J., & Nichols, T. E. (Eds.). (2011). Statistical parametric mapping: the analysis of functional brain images. Academic press.
.. [Penny2008] Penny, W. D., Duzel, E., Miller, K. J., & Ojemann, J. G. (2008). Testing for nested oscillation. Journal of neuroscience methods, 174(1), 50-61.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from ..analytic_signal import analytic_signal

import numpy as np
import statsmodels.api as sm


def glm(data, fb_lo, fb_hi, fs, pairs=None, window_size=-1):
    """ General Linear Model

    Estimate the :math:`r^2` for the given :attr:`data`,
    between the :attr:`pairs (if given) of channels.

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

    window_size : int
        The number of samples that will be used in each window.


    Returns
    -------
    ts : complex array-like, shape(n_windows, n_channels, n_channels)
        Estimated :math:`r^2` time series (in each window).

    ts_avg: complex array-like, shape(n_channels, n_channels)
        Average :math:`r^2` (across all windows).
    """
    window_size = int(window_size)

    n_channels, n_samples = np.shape(data)

    windows = n_samples / window_size
    windows = np.int32(windows)
    if windows <= 0:
        windows = 1
        window_size = -1

    if pairs is None:
        pairs = [(r1, r2) for r1 in range(n_channels) for r2 in range(n_channels)]

    l_hilb, _, _ = analytic_signal(data, fb_lo, fs)
    h_hilb, _, _ = analytic_signal(data, fb_hi, fs)

    lf = np.angle(l_hilb) % np.pi
    hfa = np.abs(h_hilb)

    ts = np.zeros((windows, n_channels, n_channels))
    ts_avg = np.zeros((n_channels, n_channels))

    start_window = 0
    end_window = start_window + window_size
    for win in range(windows):
        for pair in pairs:
            source_channel, target_channel = pair

            slide_lf = lf[source_channel, start_window:end_window]
            slide_hfa = hfa[target_channel, start_window:end_window]

            s = np.size(slide_lf)
            y = np.reshape(slide_hfa, (s, 1))
            ax = np.ones((s))
            X = np.vstack((np.cos(slide_lf), np.sin(slide_lf), ax)).T

            result = sm.OLS(y, X).fit()
            r2 = result.rsquared
            r2 = np.float32(r2)

            ts[win, source_channel, target_channel] = r2

        start_window = end_window
        end_window = start_window + window_size

    ts_avg = np.average(ts, 0)

    return ts, ts_avg
