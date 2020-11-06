# -*- coding: utf-8 -*-
""" Order Reccurence Plot




|

-----


.. [Stam2007] Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). Phase lag index: assessment of functional connectivity from multi channel EEG and MEG with diminished bias from common sources. Human brain mapping, 28(11), 1178-1193.
.. [Hardmeier2014] Hardmeier, M., Hatz, F., Bousleiman, H., Schindler, C., Stam, C. J., & Fuhr, P. (2014). Reproducibility of functional connectivity and graph measures based on the phase lag index (PLI) and weighted phase lag index (wPLI) derived from high resolution EEG. PloS one, 9(10), e108648.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
import scipy
import itertools
import sklearn

from .embed_delay import embed_delay


def rr_order_patterns(
    signal1: "np.ndarray[np.int32]", signal2: "np.ndarray[np.int32]", m: int, tau: int
) -> float:
    """



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

    * The results may vary from the original MATLAB script because of
      the order of the indices in the :python:`numpy.where`.

    * The permutations are generated from :math:`[1, dim+1]` so there are
      no occurances of :math:`0.`

    * The extra :math:`+1` in the lines
      .. python: I = sklearn.preprocessing.normalize(I + 1)
      is in order to avoid :math:`0`s.


    Returns
    -------
    cstr : float
        Coupling strength.
    """
    if len(signal1) != len(signal2):
        raise Exception("")

    x = embed_delay(signal1, m, tau)
    y = embed_delay(signal2, m, tau)

    lenx = len(x)

    factorial_dim = scipy.special.factorial(m)

    ipermlist = itertools.permutations(list(range(1, m + 1)))
    npermlist = np.zeros((np.int32(factorial_dim), m))
    for index, perm in enumerate(ipermlist):
        perm = np.reshape(perm, (1, -1)).astype(np.float32)
        npermlist[index, :] = sklearn.preprocessing.normalize(perm)

    # Signal 1
    I = np.argsort(x).astype(np.float32)
    I = sklearn.preprocessing.normalize(I + 1)
    X = np.dot(I, npermlist.T).T
    X = np.round(X, decimals=4)

    ct1, _ = np.where(X == 1.0)

    # Signal 2
    I = np.argsort(y).astype(np.float32)
    I = sklearn.preprocessing.normalize(I + 1)
    Y = np.dot(I, npermlist.T).T

    Y = np.round(Y, decimals=4)

    ct2, _ = np.where(Y == 1.0)

    total_sum = 0.0
    for k in range(lenx):
        if ct1[k] == ct2[k]:
            total_sum += 1

    cstr = total_sum / lenx

    return cstr
