# -*- coding: utf-8 -*-
""" Imaginary part of Phase Locking Value


Imaginary Phase Locking Value (*IPLV*) was proposed to resolve PLV's sensitivity to
volume conduction and common reference effects.

IPLV is computed similarly as PLV, but taking the imaginary part of the summation:

.. math::
    ImPLV = \\frac{1}{N} \\left | Im \\left ( \\sum_{t=1}^{N} e^{i (\phi_{j1}(t)  - \phi_{j2}(t))} \\right ) \\right |


|

-----

.. [Sadaghiani2012] Sadaghiani, S., Scheeringa, R., Lehongre, K., Morillon, B., Giraud, A. L., D'Esposito, M., & Kleinschmidt, A. (2012). Alpha-band phase synchrony is related to activity in the fronto-parietal adaptive control network. The Journal of Neuroscience, 32(41), 14305-14310.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from .estimator import Estimator
from ..analytic_signal import analytic_signal

import numpy as np


def iplv(data, fb, fs, pairs=None):
    """ Imaginary part of Phase Locking Value

    Compute the Imaginary part of Phase Locking Value for the given *data*,
    between the *pairs* (if given) of channels.


    Parameters
    ----------
    data : array-like, shape(n_channels, n_samples)
        Multichannel recording data.

    fb : list of length 2
        The low and high frequencies.

    fs : float
        Sampling frequency.

    pairs : array-like or `None`
        - If an `array-like` is given, notice that each element is a tuple of length two.
        - If `None` is passed, complete connectivity will be assumed.


    Returns
    -------
    ts : array-like, shape(n_channels, n_channels, n_samples)
        Estimated IPLV time series.

    avg : array-like, shape = [n_electrodes, n_electrodes]
        Average IPLV.
    """
    n_channels, n_samples = np.shape(data)
    _, _, u_phases = analytic_signal(data, fb, fs)

    if pairs is None:
        pairs = [(r1, r2) for r1 in range(n_channels)
                 for r2 in range(r1, n_channels)
                 if r1 != r2]

    ts = np.zeros((n_channels, n_channels, n_samples))
    avg = np.zeros((n_channels, n_channels))

    for pair in pairs:
        u_phases1, u_phases2 = u_phases[pair, ]

        ts_iplv = np.exp(1j * (u_phases1 - u_phases2))
        average_iplv = np.abs(np.sum((ts_iplv)) / float(n_samples))

        ts[pair] = ts_iplv
        avg[pair] = average_iplv

    return ts, avg


class IPLV(Estimator):

    def __init__(self, fb, fs, pairs=None):
        pass

    def preprocess(self, data):
        pass

    def estimate(self, data):
        pass
