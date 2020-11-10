""" Intra-class Correlation (3, 1)


Notes
-----
* Based on the code available at <https://github.com/ekmolloy/fmri_test-retest>

|

.. [McGraw1996] McGraw, K. O., & Wong, S. P. (1996). Forming inferences about some intraclass correlation coefficients. Psychological methods, 1(1), 30.
.. [Birn2013] Birn, R. M., Molloy, E. K., Patriat, R., Parker, T., Meier, T. B., Kirk, G. R., ... & Prabhakaran, V. (2013). The effect of scan length on the reliability of resting-state fMRI connectivity estimates. Neuroimage, 83, 550-558.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
import scipy


def icc_31(X: "np.ndarray[np.float32]") -> float:
    """ ICC (3,1)


    Parameters
    ----------
    X :
        Input data


    Returns
    -------
    icc : float
        Intra-class correlation.

    """
    _, k = np.shape(X)  # type: ignore
    _, ms, _, _ = _anova(X)

    BMS = ms[2]
    EMS = ms[4]
    icc = (BMS - EMS) / (BMS + (k - 1) * EMS)

    return icc


def _anova(X):
    """

    """
    m, n = np.shape(X)
    total = m * n

    A = np.sum(np.sum(np.power(X, 2.0)))
    Bc = np.sum(np.power(np.sum(X, 0), 2.0)) / np.float32(m)
    Br = np.sum(np.power(np.sum(X, 1), 2.0)) / np.float32(n)
    D = np.power(np.sum(np.sum(X)), 2.0) / np.float32(total)

    ss_bc = Bc - D  # Columns - between
    ss_wc = A - Bc  # Columns - within

    ss_br = Br - D  # Rows - between
    ss_wr = A - Br  # Rows - within

    ss_e = A - Br - Bc + D  # Residual

    # degrees of freedom for columns/rows/residual and between/within
    df_bc = n - 1
    df_wc = n * (m - 1)
    df_br = m - 1
    df_wr = m * (n - 1)
    df_e = df_bc * df_br
    df = np.array([df_bc, df_wc, df_br, df_wr, df_e])

    #
    ms_bc = ss_bc / df_bc
    ms_wc = ss_wc / df_wc
    ms_br = ss_br / df_br
    ms_wr = ss_wr / df_wr
    ms_e = ss_e / df_e
    ms = np.array([ms_bc, ms_wc, ms_br, ms_wr, ms_e])

    #
    F_bc = ms_bc / ms_e
    F_br = ms_br / ms_e
    F = np.array([F_bc, F_br])

    #
    p_bc = 1.0 - scipy.stats.f.cdf(F_bc, df_bc, df_e)
    p_br = 1.0 - scipy.stats.f.cdf(F_br, df_br, df_e)
    p = np.array([p_bc, p_br])

    return df, ms, F, p
