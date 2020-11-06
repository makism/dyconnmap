# -*- coding: utf-8 -*-
""" Entropy

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np


def entropy(x: "np.ndarray[np.float32]") -> float:
    """ Entropy

    Parameters
    ----------
    x : array-like, shape(N)
        Input symbolic time series.


    Returns
    -------
    entropy : float
        The computed entropy.
    """
    l = len(x)

    # unique, counts = np.unique(dts, return_counts=True)
    _, counts = np.unique(x, return_counts=True)
    len_counts = len(counts)

    v = 0.0
    for i in range(len_counts):
        v += float(counts[i] / l) * np.log10(counts[i] / l)

    return v
