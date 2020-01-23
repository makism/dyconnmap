# -*- coding: utf-8 -*-
""" Permutation Entropy


|

-----


.. [Stam2007] Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). Phase lag index: assessment of functional connectivity from multi channel EEG and MEG with diminished bias from common sources. Human brain mapping, 28(11), 1178-1193.
.. [Hardmeier2014] Hardmeier, M., Hatz, F., Bousleiman, H., Schindler, C., Stam, C. J., & Fuhr, P. (2014). Reproducibility of functional connectivity and graph measures based on the phase lag index (PLI) and weighted phase lag index (wPLI) derived from high resolution EEG. PloS one, 9(10), e108648.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
import scipy
import itertools

from .embed_delay import embed_delay


def permutation_entropy(signal, m, tau):
    """ Permutation Entropy


    Parameters
    ----------
    signal : array-like, shape(N)
        Symblic time series (1D).

    m : int
        Embedding dimension.

    tau : int
        Time delay parameter.


    Returns
    -------
    pe : float
        Permutation entropy.

    npe : float
        Normalized permutation entropy.
    """
    x = embed_delay(signal, m, tau)

    lx = len(x)

    factorial_dim = np.int(scipy.special.factorial(m))

    ipermlist = itertools.permutations(list(range(1, m + 1)))
    npermlist = np.zeros((factorial_dim, m))
    for index, perm in enumerate(ipermlist):
        perm = np.reshape(perm, (1, -1)).astype(np.float32)
        npermlist[index, :] = perm

    c = np.zeros((len(npermlist)))

    for j in range(lx):
        I = np.argsort(x[j, :]).astype(np.float32)

        for jj in range(factorial_dim):
            if np.all([v == 1.0 for v in np.abs(npermlist[jj, :] - I)]):
                c[jj] += 1

    pe = 0.0
    p = 0.0

    for i in range(factorial_dim):
        if c[i] != 0.0:
            p = c[i] / ((lx - (m - 1)) * tau)
            pe = pe + p * np.log2(p)

    pe = -pe
    npe = pe / np.log2(factorial_dim)

    return pe, npe
