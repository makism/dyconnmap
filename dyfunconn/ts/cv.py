""" Coefficient of Variation


"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np


def cv(X):
    """


    Parameters
    ----------
    X : 


    Returns
    -------
    cv : float
        The computed coefficient of variation.
    """
    return np.std(X) / np.mean(X)
