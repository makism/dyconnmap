# -*- coding: utf-8 -*-
""" Mutual Information

Comparing two partitions using the mutual information theoretical metric.


Notes
-----
Based on the code available at http://users.auth.gr/~stdimitr/files/software/vi.rar

|

-----
.. [Fred2005] Fred, A. L., & Jain, A. K. (2005). Combining multiple clusterings using evidence accumulation. IEEE transactions on pattern analysis and machine intelligence, 27(6), 835-850.
.. [Strehl2002] Strehl, A., & Ghosh, J. (2002). Cluster ensembles---a knowledge reuse framework for combining multiple partitions. Journal of machine learning research, 3(Dec), 583-617.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>"

import numpy as np

from dyfunconn.ts.entropy import entropy


def mutual_information(indices_a, indices_b):
    """ Mutual Information



    Parameters
    ----------


    Returns
    -------
    MI : float
        Mutual information.

    NMI : float
        Normalized mutual information.
    """
    indices_a = indices_a.flatten()
    indices_b = indices_b.flatten()

    entropy_a = -entropy(indices_a)
    entropy_b = -entropy(indices_b)

    N = len(indices_a)
    unique, counts = np.unique(indices_a, return_counts=True)
    len_counts = len(counts)
    Ua = np.zeros((len_counts, N))
    for i in range(len_counts):
        tmp = np.where(indices_a == unique[i])
        Ua[i, tmp[0]] = 1

    unique, counts = np.unique(indices_b, return_counts=True)
    len_counts = len(counts)
    Ub = np.zeros((len_counts, N))
    for i in range(len_counts):
        tmp = np.where(indices_b == unique[i])
        Ub[i, tmp[0]] = 1

    Sab = Ua.dot(Ub.T) / np.float32(N)
    Sa = np.diag(Ua.dot(Ua.T) / np.float32(N))
    Sb = np.diag(Ub.dot(Ub.T) / np.float32(N))

    # Add dummy dimension (needed for following computations).
    Sa = np.expand_dims(Sa, axis=1)
    Sb = np.expand_dims(Sb, axis=1)

    SS = Sab * np.log10(Sab / (Sa * Sb.T))

    MI = np.nansum(np.nansum(SS))
    NMI = 2 * MI / (entropy_a + entropy_b)

    return MI, NMI
