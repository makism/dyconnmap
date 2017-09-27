# -*- coding: utf-8 -*-
""" Variation of Information



Notes
-----
Based on the code available at http://users.auth.gr/~stdimitr/files/software/vi.rar

|

-----
.. [Meilla2007] Meilă, M. (2007). Comparing clusterings—an information based distance. Journal of multivariate analysis, 98(5), 873-895.
.. [Dimitriadis2009] Dimitriadis, S. I., Laskaris, N. A., Del Rio-Portilla, Y., & Koudounis, G. C. (2009). Characterizing dynamic functional connectivity across sleep stages from EEG. Brain topography, 22(2), 119-133.
.. [Dimitriadis2012] Dimitriadis, S. I., Laskaris, N. A., Michael Vourkas, V. T., & Micheloyannis, S. (2012). An EEG study of brain connectivity dynamics at the resting state. Nonlinear Dynamics-Psychology and Life Sciences, 16(1), 5.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com"

import numpy as np

from dyfunconn.ts.entropy import entropy


def variation_information(indices_a, indices_b):
    """

    indices_a:

    indices_b:

    """
    n1 = len(indices_a)
    n2 = len(indices_b)

    if n1 != n2:
        pass

    entropy1 = entropy(indices_a)
    entropy2 = entropy(indices_b)

    MI, NMI = __mi(indices_a, -entropy1, indices_b, -entropy2)

    entropy1 = -entropy1
    entropy2 = -entropy2
    VI_value = entropy1 + entropy2 - 2 * MI

    NVI = VI_value / np.log(n1)

    return VI_value, NVI


def __mi(indices_a, entropy_a, indices_b, entropy_b):
    N = len(indices_a)

    unique, counts = np.unique(indices_a, return_counts=True)
    len_counts = len(counts)
    Ua = np.zeros((len_counts, N))
    indices_a = indices_a.flatten()
    for i in range(len_counts):
        tmp = np.where(indices_a == unique[i])
        Ua[i, tmp[0]] = 1

    unique, counts = np.unique(indices_b, return_counts=True)
    len_counts = len(counts)
    Ub = np.zeros((len_counts, N))
    indices_b = indices_b.flatten()
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
