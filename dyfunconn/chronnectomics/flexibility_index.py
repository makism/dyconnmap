# -*- coding: utf-8 -*-
""" Flexibility Index

In the context of graph clustering it was defined in (Basset2011_), flexbility is the frequency
of a nodea change module allegiance; the transition of brain states between consecutive
temporal segments. The higher the number of changes, the larger the FI will be.

.. math::
   FI = \\frac{\\text{number of transitions}} {\\text{total symbols - 1}}

|

.. [Basset2011] Bassett, D. S., Wymbs, N. F., Porter, M. A., Mucha, P. J., Carlson, J. M., & Grafton, S. T. (2011). Dynamic reconfiguration of human brain networks during learning. Proceedings of the National Academy of Sciences, 108(18), 7641-7646.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np


def flexibility_index(x):
    """ Flexibility Index

    Compute the flexibility index for the given symbolic, 1d time series.


    Parameters
    ----------
    x : array-like, shape(N)
        Input symbolic time series.


    Returns
    -------
    fi : float
        The flexibility index.
    """
    l = len(x)

    counter = 0
    for k in range(l - 1):
        if x[k] != x[k + 1]:
            counter += 1

    fi = counter / np.float32(l - 1)

    return fi
