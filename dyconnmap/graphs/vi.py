# -*- coding: utf-8 -*-
""" Variation of Information

Variation of Information (*VI*) [Meilla2007]_ is an information theoretic criterion
for comparing two partitions. It is based on the classic notions of entropy and mutual information.
In a nutshell, VI measures the amount of information that is lost or gained in changing from
clustering :math:`A` to clustering :math:`B`. VI is a true metric, is always non-negative and symmetric.
The following formula is used to compute the VI between two groups:

.. math::
    VI(A, B) = [H(A) - I(A, B)] + [H(B) - I(A, B)]

Where :math:`H` denotes the entropy computed for each partition separately,
and :math:`I` the mutual information between clusterings :math:`A` and :math:`B`.

The resulting distance score can be adjusted to bound it between :math:`[0, 1]` as follows:

.. math::
    VI^{*}(A,B) = \\frac{1}{\\log{n}}VI(A, B)


|

-----

.. [Meilla2007] Meilă, M. (2007). Comparing clusterings—an information based distance. Journal of multivariate analysis, 98(5), 873-895.
.. [Dimitriadis2009] Dimitriadis, S. I., Laskaris, N. A., Del Rio-Portilla, Y., & Koudounis, G. C. (2009). Characterizing dynamic functional connectivity across sleep stages from EEG. Brain topography, 22(2), 119-133.
.. [Dimitriadis2012] Dimitriadis, S. I., Laskaris, N. A., Michael Vourkas, V. T., & Micheloyannis, S. (2012). An EEG study of brain connectivity dynamics at the resting state. Nonlinear Dynamics-Psychology and Life Sciences, 16(1), 5.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>
# Author: Stavros Dimitriadis <stdimitr@gmail.com>

import numpy as np

from dyconnmap.ts.entropy import entropy


def variation_information(indices_a, indices_b):
    """ Variation of Information


    Parameters
    ----------
    indices_a : array-like, shape(n_samples)
        Symbolic time series.

    indices_b : array-like, shape(n_samples)
        Symbolic time series.


    Returns
    -------
    vi : float
        Variation of information.
    """
    n1 = len(indices_a)
    n2 = len(indices_b)

    if n1 != n2:
        pass

    entropy1 = entropy(indices_a)
    entropy2 = entropy(indices_b)

    MI, _ = __mi(indices_a, -entropy1, indices_b, -entropy2)

    entropy1 = -entropy1
    entropy2 = -entropy2
    VI_value = entropy1 + entropy2 - 2 * MI

    NVI = VI_value / np.log(n1)

    return VI_value, NVI


def __unique_symbols(indices):
    """

    """
    N = len(indices)
    unique, counts = np.unique(indices, return_counts=True)
    len_counts = len(counts)
    U = np.zeros((len_counts, N))
    indices = indices.flatten()
    for i in range(len_counts):
        tmp = np.where(indices == unique[i])
        U[i, tmp[0]] = 1

    return U


def __mi(indices_a, entropy_a, indices_b, entropy_b):
    """

    """
    N = len(indices_a)

    Ua = __unique_symbols(indices_a)
    Ub = __unique_symbols(indices_b)

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
