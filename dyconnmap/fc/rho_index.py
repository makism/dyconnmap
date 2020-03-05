# -*- coding: utf-8 -*-
""" ρ index

.. math::
    \\rho_{p, q}(t) = \\frac{H_{max} - H}{H_{max}}

Where :math:`H` is the Shannon entropy estimated within :math:`M` number of
phase bins, and :math:`H_{max} = ln(M)` is the maximal entropy and
and :math:`p_k` is the relative frequency of finding frequency difference
in the :math:`k` th phase bin.

.. math::
    H = - \\sum_{k=1}^M p_k ln(p_k)

The computed value varies within the range :math:`[0, 1]`

-----

.. [Tass1998] Tass, P., Rosenblum, M. G., Weule, J., Kurths, J., Pikovsky, A., Volkmann, J., ... & Freund, H. J. (1998). Detection of n: m phase locking from noisy data: application to magnetoencephalography. Physical review letters, 81(15), 3291.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from ..analytic_signal import analytic_signal

import numpy as np


def rho_index(data, n_bins, fb, fs, pairs=None):
    """ Synchronization Index

    Compute the synchronization index for the given :attr:`data`, between the :attr:`pairs (if given)
    of channels.


    Parameters
    ----------
    data : array-like, shape(n_channels, n_samples)
        Multichannel recording data.

    n_bins : int
        Number of bins.

    fb : list of length 2
        The low and high frequencies.

    fs : float
        Sampling frequency.

    pairs : array-like or `None`
        - If an `array-like` is given, notice that each element is a tuple of length two.
        - If `None` is passed, complete connectivity will be assumed.


    Returns
    -------
    rho : array-likem, shape(n_channels, n_channels)
        Estimated rho index.
    """
    n_channels, _ = np.shape(data)

    _, u_phases, _ = analytic_signal(data, fb, fs=128, order=3)

    if pairs is None:
        pairs = [
            (r1, r2)
            for r1 in range(n_channels)
            for r2 in range(r1, n_channels)
            if r1 != r2
        ]

    rho_mtx = np.zeros((n_channels, n_channels))
    for pair in pairs:
        u_phase1, u_phase2 = u_phases[pair,]

        du = (u_phase1 - u_phase2) % (2.0 * np.pi)

        hist, _ = np.histogram(du, n_bins)
        n_hist = hist / float(np.sum(hist))

        Smax = np.log(n_bins)
        S = -np.sum(n_hist * np.log(n_hist))
        H = (Smax - S) / Smax

        rho_mtx[pair] = H

    return rho_mtx
