# -*- coding: utf-8 -*-
""" Directed Phase Lag Index

Directed Phase Lag Index (*dPLI*) was introduced in [Stam2012_] to capture
the phase and lag relationship as a measure of directed functional connectivity.

* if :math:`0.5 \le dPLI_{xy} \leq 1.0`, :math:`x` is leading :math:`y`
* if :math:`0.0 \leq dPLI_{xy} = 0.5`, :math:`y` is leading :math:`x`
* if :math:`dPLI_{xy} = 0.5`, neither :math:`x` nor :math:`y` is leading or lagging


|

-----

.. [Stam2012] Stam, C. J., & van Straaten, E. C. (2012). Go with the flow: use of a directed phase lag index (dPLI) to characterize patterns of phase relations in a large-scale model of brain dynamics. Neuroimage, 62(3), 1415-1428.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from ..analytic_signal import analytic_signal

import numpy as np


def dpli(data, fb, fs, pairs=None):
    """ Directed Phase Lag Index

    Estimate the Directed Phase Lag Index for the given :attr:`data`,
    between the :attr:`pairs (if given) of channels.


    Parameters
    ----------
    data : array-like, shape(n_channels, n_samples)
        Multichannel recording data

    fb : list of length 2
        The lower and upper frequency.

    fs : float
        Sampling frequency

    pairs : array-like or `None`
        - If an `array-like` is given, notice that each element is a tuple of length two.
        - If `None` is passed, complete connectivity will be assumed.


    Returns
    -------
    dpliv : array-like, shape(n_channels, n_channels)
        Estimated Directed PLI.


    See also
    --------
    dyconnmap.fc.PLI: Phase Lag Index
    """
    n_channels, n_samples = np.shape(data)

    dpliv = np.zeros((n_channels, n_channels))

    h_signal, _, _ = analytic_signal(data, fb, fs)
    phases = np.angle(h_signal)

    if pairs is None:
        pairs = [(r1, r2) for r1 in range(n_channels) for r2 in range(n_channels)]

    for pair in pairs:
        phase1, phase2 = phases[pair,]

        diff_phases = phase1 - phase2
        cyclic_rel_phase = np.mod(diff_phases, 2.0 * np.pi)

        r1 = (np.where(0 <= cyclic_rel_phase)) and (np.where(cyclic_rel_phase < np.pi))[
            0
        ]
        r2 = np.intersect1d(
            np.where(-np.pi <= cyclic_rel_phase), np.where(cyclic_rel_phase < 0)
        )
        r3 = np.where(cyclic_rel_phase == 0)[0]

        dpliv[pair] = float(len(r1) + len(r2) - 0.5 * len(r3)) / n_samples

    return dpliv
