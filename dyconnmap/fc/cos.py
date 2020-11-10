# -*- coding: utf-8 -*-
""" Cosine


"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from typing import Optional, List

from ..analytic_signal import analytic_signal

import numpy as np


def cos(
    data: np.ndarray,
    fb: Optional[float] = None,
    fs: Optional[float] = None,
    pairs: Optional[List[List[int]]] = None,
):
    """ Cosine

    Compute the correlation for the given :attr:`data`, between the :attr:`pairs` (if given)
    of channels.


    Parameters
    ----------
    data : array-like, shape(n_rois, n_samples)
        Multichannel recording data.

    fb : list of length 2, optional
        The low and high frequencies.

    fs : float, optional
        Sampling frequency.

    pairs : array-like or `None`
        - If an `array-like` is given, notice that each element is a tuple of length two.
        - If `None` is passed, complete connectivity will be assumed.


    Returns
    -------
    c : array-like, shape(n_rois, n_rois)
        Estimated connectivity matrix.
    """
    n_rois, n_samples = np.shape(data)

    X = None
    if fb is not None and fs is not None:
        _, uphases, _ = analytic_signal(data, fb, fs)
        X = uphases
    else:
        X = data

    conn_mtx = np.zeros((n_rois, n_rois), dtype=np.float32)
    for k in range(n_rois):
        for l in range(k + 1, n_rois):
            val = np.sum(np.cos(X[k, :] - X[l, :])) / np.float32(n_samples)
            val = np.abs(val)

            conn_mtx[k, l] = val

    return conn_mtx
