# -*- coding: utf-8 -*-
""" Ordinal Pattern Similarity




|

-----


.. [Stam2007] Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). Phase lag index: assessment of functional connectivity from multi channel EEG and MEG with diminished bias from common sources. Human brain mapping, 28(11), 1178-1193.
.. [Hardmeier2014] Hardmeier, M., Hatz, F., Bousleiman, H., Schindler, C., Stam, C. J., & Fuhr, P. (2014). Reproducibility of functional connectivity and graph measures based on the phase lag index (PLI) and weighted phase lag index (wPLI) derived from high resolution EEG. PloS one, 9(10), e108648.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
import scipy
import itertools
from sklearn import preprocessing

from .embed_delay import embed_delay


def ordinal_pattern_similarity(signal1, signal2, m, tau):
    """ Ordinal Pattern Similarity


    Parameters
    ----------
    signal1 :

    signal2 :

    m : int
        Embedding dimension.

    tau : int
        Time delay parameter.


    Notes
    -----
    * The results may vary from the original MATLAB script because of
      the permutations' order.

    * The permutations are generated from :math:`[1, dim+1]` so there are
      no occurances of :math:`0.`

    * The extra :math:`+1` in the lines
      .. python: I = sklearn.preprocessing.normalize(I + 1)
      is in order to avoid :math:`0`s.


    Returns
    -------
    dissimilarity : float
        The dissimilarity index as computed from the ordinal patterns.

    ordinal_patterns : array
        The time series of ordinal patterns for input signals.

    patterns_distribution : array
        Distribution of the patterns.
    """
    if len(signal1) != len(signal2):
        raise Exception("")

    x = embed_delay(signal1, m, tau)
    y = embed_delay(signal2, m, tau)

    len1 = len(x)
    len2 = len(y)

    factorial_dim = scipy.special.factorial(m, exact=True)

    ipermlist = itertools.permutations(list(range(1, m + 1)))
    npermlist = np.zeros((np.int32(factorial_dim), m))
    for index, perm in enumerate(ipermlist):
        perm = np.reshape(perm, (1, -1)).astype(np.float32)
        npermlist[index, :] = preprocessing.normalize(perm)

    # Signal 1
    I = np.argsort(x).astype(np.float32)
    I = preprocessing.normalize(I + 1)
    X = np.dot(I, npermlist.T).T
    X = np.round(X, decimals=4)

    ct1, _ = np.where(X == 1.0)

    # Signal 2
    I = np.argsort(y).astype(np.float32)
    I = preprocessing.normalize(I + 1)
    Y = np.dot(I, npermlist.T).T

    Y = np.round(Y, decimals=4)

    ct2, _ = np.where(Y == 1.0)

    c1 = np.zeros((len(npermlist)))
    c2 = np.zeros((len(npermlist)))

    for i in range(factorial_dim):
        r = np.where(ct1 == i)
        c1[i] = len(r[0].squeeze())
        r = np.where(ct2 == i)
        c2[i] = len(r[0].squeeze())

    p1 = np.zeros((len(npermlist)))
    p2 = np.zeros((len(npermlist)))

    for i in range(factorial_dim):
        p1[i] = c1[i] / (len1 - (m - 1) * tau)
        p2[i] = c2[i] / (len2 - (m - 1) * tau)

    stable = np.sqrt(factorial_dim / (factorial_dim - 1))

    d = np.sum(np.power(p1 - p2, 2.0))

    dissimilarity = stable * np.sqrt(d)
    ordinal_patterns = np.vstack([ct1, ct2])
    patterns_distribution = np.vstack([c1, c2])

    return dissimilarity, ordinal_patterns, patterns_distribution
