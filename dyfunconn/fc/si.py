# -*- coding: utf-8 -*-
""" Synchronization Index



-----


https://vis.caltech.edu/~rodri/papers/performance.pdf

http://researchcommons.waikato.ac.nz/bitstream/handle/10289/770/EEG_entropies.pdf;jsessionid=837973D1FD53B66585F7CBFB6FEBADFB?sequence=1

file:///home/makism/Downloads/pereda2005.pdf

https://en.wikipedia.org/wiki/Brain_connectivity_estimators#Bivariate_versus_multivariate_estimators

http://journal.frontiersin.org/article/10.3389/fneur.2013.00057/full

http://jn.physiology.org/content/87/2/937

http://www.stat.physik.uni-potsdam.de/~mros/Moss_book.pdf

http://www.fulviofrisone.com/attachments/article/412/synchronization%20an%20universal%20concept%20in%20nonlinear%20sciences.pdf

.. [Cohen2008] Cohen, M. X. (2008). Assessing transient cross-frequency coupling in EEG data. Journal of neuroscience methods, 168(2), 494-499."""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from ..analytic_signal import analytic_signal

import numpy as np


def si(data, n_bins, fb, fs, pairs=None):
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
    si : array-likem, shape(n_channels, n_channels)
        Estimated Synchronization Index.
    """
    n_channels, n_samples = np.shape(data)

    _, _, u_phases = analytic_signal(data, fb, fs=128, order=3)

    if pairs is None:
        pairs = [(r1, r2) for r1 in range(n_channels)
                 for r2 in range(r1, n_channels)
                 if r1 != r2]

    si_mtx = np.zeros((n_channels, n_channels))
    for pair in pairs:
        u_phase1, u_phase2 = u_phases[pair, ]

        du = (u_phase1 - u_phase2) % (2.0 * np.pi)

        hist, bins = np.histogram(du, n_bins)
        n_hist = hist / float(np.sum(hist))

        Smax = np.log(n_bins)
        S = -np.sum(n_hist * np.log(n_hist))
        H = (Smax - S) / Smax

        si_mtx[pair] = H

    return si_mtx
