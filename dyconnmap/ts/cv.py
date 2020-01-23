""" Coefficient of Variation


"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np


def cv(x):
    """ Coefficient of Variation


    Parameters
    ----------
    x : array-like, shape(n_samples)
        Input time series


    Returns
    -------
    cv : float
        The computed coefficient of variation.
    """
    return np.std(x) / np.mean(x)
