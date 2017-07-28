# -*- coding: utf-8 -*-
"""




|

-----


.. [Stam2007] Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). Phase lag index: assessment of functional connectivity from multi channel EEG and MEG with diminished bias from common sources. Human brain mapping, 28(11), 1178-1193.
.. [Hardmeier2014] Hardmeier, M., Hatz, F., Bousleiman, H., Schindler, C., Stam, C. J., & Fuhr, P. (2014). Reproducibility of functional connectivity and graph measures based on the phase lag index (PLI) and weighted phase lag index (wPLI) derived from high resolution EEG. PloS one, 9(10), e108648.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np


def embed_delay(ts, dim, tau):
    """ Embed delay

    Build a set of embedding sequences from given time series X with lag Tau
    and embedding dimension DE. Let X = [x(1), x(2), ... , x(N)], then for each
    i such that 1 < i <  N - (D - 1) * Tau, we build an embedding sequence,
    Y(i) = [x(i), x(i + Tau), ... , x(i + (D - 1) * Tau)]. All embedding
    sequence are placed in a matrix Y.


    Parameters
    ----------
    ts : 1d array

    dim : int
        The embedding dimension.

    tau : int
        Time delay factor.


    Returns
    -------
    new_ts : array
        The embedding sequeces.

    """
    ts = ts.flatten()
    new_ts = np.zeros([len(ts), dim])

    l = np.int32(np.floor(dim / 2.0))
    for i, o in zip(range(0, dim), range(-l, l + 1)):
        data = np.roll(ts, -tau + o)
        new_ts[:, i] = data
    new_ts = new_ts[:-(dim - 1) * tau, ]

    return new_ts
