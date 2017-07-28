# -*- coding: utf-8 -*-
""" Phase Coherence

Phase Coherence (*PCOH*)


.. math::
    PCV_{p, q}(t) = \\frac{H_{max} - H}{H_{max}}

Where :math:`H` is the Shannon entropy estimated within :math:`M` number of
phase bins, and :math:`H_{max} = ln(M)` is the maximal entropy and
and :math:`p_k` is the relative frequency of finding frequency difference
in the :math:`k` th phase bin.

.. math::
    H = - \\sum_{k=1}^M p_k ln(p_k)

The computed value varies within the range :math:`[0, 1]`.

|

-----

.. [Ziqiang2007] Ziqiang, Z., & Puthusserypady, S. (2007, July). Analysis of schizophrenic EEG synchrony using empirical mode decomposition. In Digital Signal Processing, 2007 15th International Conference on (pp. 131-134). IEEE.
.. [Tass1998] Tass, P., Rosenblum, M. G., Weule, J., Kurths, J., Pikovsky, A., Volkmann, J., ... & Freund, H. J. (1998). Detection of n: m phase locking from noisy data: application to magnetoencephalography. Physical review letters, 81(15), 3291.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from ..analytic_signal import analytic_signal

import numpy as np


def pcoh(data, fb, fs, n_bins=10, unwrap=False, pairs=None):
    """ Phase Coherence

    Compute the Phase Coherence for the given :attr:`data`,
    between the :attr:`pairs (if given) of channels.


    Parameters
    ----------
    data : array-like, shape(n_channels, n_samples)
        Multichannel recording data.

    fb : list of length 2
        The low and high frequencies.

    fs : float
        Sampling frequency.

    n_bins : int
        Number of bins. Default `10`.

    unrwap : boolean
        Whether or not to unwap the extracted phases. Default `False`.

    pairs : array-like or `None`
        - If an `array-like` is given, notice that each element is a tuple of length two.
        - If `None` is passed, complete connectivity will be assumed.


    Returns
    -------
    pcoh : array-like, shape(n_channels, n_channels)
        Estimated PCOH.
    """
    n_channels, n_samples = np.shape(data)

    _, hilberted, u_phases = analytic_signal(data, fb, fs)

    if unwrap:
        data = u_phases
    else:
        data = np.angle(hilberted)

    if pairs is None:
        pairs = [(r1, r2) for r1 in range(n_channels)
                 for r2 in range(r1, n_channels)
                 if r1 != r2]

    pcv = np.zeros((n_channels, n_channels))

    for pair in pairs:
        phase1, phase2 = data[pair, ]

        dtheta = phase1 - phase2
        binwidth = (np.max(dtheta) - np.min(dtheta)) / float(n_bins)

        bins = np.arange(np.min(dtheta) - binwidth /
                         2.0, np.max(dtheta) + binwidth, binwidth)

        hist, edges = np.histogram(dtheta, bins=bins)
        hist = hist / float(np.sum(hist))

        Hmax = np.log(n_bins)
        H = -1.0 * np.sum(hist * np.log(hist + np.finfo(float).eps))
        pcv[pair] = (Hmax - H) / Hmax

    return pcv
