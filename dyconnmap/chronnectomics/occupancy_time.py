# -*- coding: utf-8 -*-
""" Occupancy Time

The fraction of number of distinct symbols occuring in the symbolic time series (Dimitriadis2019_).


|

.. [Dimitriadis2019] Dimitriadis, S. I., López, M. E., Maestu, F., & Pereda, E. (2019). Modeling the Switching behavior of Functional Connectivity Microstates (FCμstates) as a Novel Biomarker for Mild Cognitive Impairment. Frontiers in Neuroscience, 13.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np


def occupancy_time(x):
    """ Occupancy Time

    Compute the occupancy time for the given symbolic, 1d time series.


    Parameters
    ----------
    x : array-like, shape(N)
        Input symbolic time series.


    Returns
    -------
    ot : dictionary
        KVP, where K=symbol id and V=occupancy time.
    """
    l = len(x)
    u = np.unique(x)

    ot = {}
    for symbol in u:
        r = np.where(x == symbol)[0]
        ot[symbol] = len(r)

    ot_res = {k: v / np.float32(l) for k, v in ot.items()}

    return ot_res
