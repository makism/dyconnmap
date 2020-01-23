# -*- coding: utf-8 -*-
""" Weighted Phase Lag Index and Debiased Weighted Phase Lag Index

PLI is prone to noise and volume conduction effects; thus, Weighted Lag Index (*wPLI*) [Vinck2011]_
was proposed in [Vinck, 2011] alongside with an alternative, debiased design (*dwPLI*).
Similar to PLI, wPLI operates on the cross-spectrum of two real-valued signals; but,
it furthermore weights the cross-spectrum with the magnitude of the imaginary component.

.. math::
    wPLI = \\frac{|E\{ \\Im(Z) \} |}{ E\{ \\Im(Z) \} } = \\frac{ | E\{ |\\Im(Z)| sign(\\Im(Z)) \} |  }{ E\{ |\\Im(Z)| \} }


Furthermore, to overcome the possible sample-bias, the authors defined a debiased variant of wPLI:

.. math::
    dwPLI = \\frac{\\sum_{j=1}^N \\sum_{k \\neq j} \\Im\{X_j\} \\Im\{X_k\}}{\\sum_{j=1}^N \\sum_{k \\neq j} \\left| \\Im\{X_j\} \\Im\{X_k\} \\right| }


|

-----

.. [Vinck2011] Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F., & Pennartz, C. M. (2011). An improved index of phase-synchronization for electrophysiological data in the presence of volume-conduction, noise and sample-size bias. Neuroimage, 55(4), 1548-1565.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from ..analytic_signal import analytic_signal

import numpy as np
import matplotlib.mlab as mlab


def wpli(data, fb, fs, pairs=None, **kwargs):
    """ Weighted Phase Lag Index

    Compute the Weight Phase Lad Index for the given *data*, between the specified *pairs* of
    channels.


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
    wpli : array-like, shape(n_channels, n_channels)
        Estimated Weighted PLI.


    Notes
    -----
    1. The resulting wpli value has a phase shift.
    2. The results do not match those from MATLAB because of the `mlab.cpsd`.


    Seer also
    ---------
    dyconnmap.wpli.dwpli: Debiased Weighted Phase Lag Index
    """
    n_channels, _ = np.shape(data)

    if pairs is None:
        pairs = __prepare_pairs(n_channels)

    _, _, filtered = analytic_signal(data, fb, fs)
    filtered = data

    wpliv = np.zeros((n_channels, n_channels))

    for pair in pairs:
        filt1, filt2 = filtered[pair,]
        csdxy, _ = mlab.csd(
            filt1, filt2, Fs=fs, scale_by_freq=True, sides="onesided", **kwargs
        )

        Ixy = np.imag(csdxy)

        num = np.nansum(np.abs(Ixy) * np.sign(Ixy))
        denom = np.nansum(np.abs(Ixy))

        wpliv[pair] = num / denom

    return wpliv


def dwpli(data, fb, fs, pairs=None, **kwargs):
    """ Debiased Weighted Phase Lag Index

    Compute the Debiased Weight Phase Lad Index for the given *data*, between the specified *pairs* of
    channels.


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
    dwpli : array-like, shape(n_channels, n_channels)
        Estimated Debiased Weighted PLI.


    Seer also
    ---------
    dyconnmap.wpli.wpli: Weighted Phase Lag Index
    """
    n_channels, _ = np.shape(data)

    if pairs is None:
        pairs = __prepare_pairs(n_channels)

    _, _, filtered = analytic_signal(data, fb, fs)
    filtered = data

    dwpliv = np.zeros((n_channels, n_channels))

    for pair in pairs:
        filt1, filt2 = filtered[pair,]
        csdxy, _ = mlab.csd(
            filt1, filt2, Fs=fs, scale_by_freq=True, sides="onesided", **kwargs
        )

        Ixy = np.imag(csdxy)

        num = np.nansum(np.abs(Ixy) * np.sign(Ixy))
        denom = np.nansum(np.abs(Ixy))

        sumsquare = np.nansum(np.power(Ixy, 2.0))
        dwpliv[pair] = (np.power(num, 2.0) - sumsquare) / (
            np.power(denom, 2.0) - sumsquare
        )

    return dwpliv


def __prepare_pairs(rois):
    return [(r1, r2) for r1 in range(rois) for r2 in range(rois)]
