#
"""


"""

from ..analytic_signal import analytic_signal

import numpy as np


def cos(data, fb=None, fs=None, pairs=None):
    """

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
