# -*- coding: utf-8 -*-
""" Teager–Kaiser Operator

ref: https://www.c-motion.com/v3dwiki/index.php/Teager_Kaiser_Energy

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np


def teager_kaiser_energy(ts: "np.ndarray[np.float32]") -> "np.ndarray[np.float32]":
    """ Teager Kaiser Energy


    Parameters
    ----------
    ts : array of size [1 x samples]


    Returns
    -------

    """
    ts = ts.ravel()
    l = len(ts)

    ts = np.hstack([0.0, ts, 0.0])
    new_ts = np.zeros(len(ts))

    for i in range(1, l):
        new_ts[i] = np.power(ts[i + 1], 2.0) - ts[i + 1] * ts[i - 1]

    return ts
