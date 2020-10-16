# -*- coding: utf-8 -*-
""" Fisher's Z Transformation


|

-----


.. [Stam2007] Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). Phase lag index: assessment of functional connectivity from multi channel EEG and MEG with diminished bias from common sources. Human brain mapping, 28(11), 1178-1193.
.. [Hardmeier2014] Hardmeier, M., Hatz, F., Bousleiman, H., Schindler, C., Stam, C. J., & Fuhr, P. (2014). Reproducibility of functional connectivity and graph measures based on the phase lag index (PLI) and weighted phase lag index (wPLI) derived from high resolution EEG. PloS one, 9(10), e108648.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np


def fisher_z(data):
    """ Fisher's z-transformation

    For a given dataset :math:`p` bound to :math:`[0.0, 1.0]`, we can use Fisher's z-transformation to normalize it
    in an approximately Gaussian distribution.

    This transformation is computed as follows:

    .. math::
        z_p := \\frac{1}{2} \\text{ln} \\left ( \\frac{1+p}{1-p} \\right ) = \\text{arctanh}(p)


    Parameters
    ----------
    data :

    Returns
    -------

    """
    return np.arctanh(data)


def fisher_z_plv(data):
    """

    .. math::
        z^p_j = sin^{-1}(2 * PLV_j - 1)


    Parameters
    ----------

    Returns
    -------


    |

    -----
    .. [Mormann2005] Mormann, F., Fell, J., Axmacher, N., Weber, B., Lehnertz, K., Elger, C. E., & Fernández, G. (2005). Phase/amplitude reset and theta–gamma interaction in the human medial temporal lobe during a continuous word recognition memory task. Hippocampus, 15(7), 890-900.

    """
    tmp = 2 * data - 1
    return np.apply_along_axis(np.arcsin, 1, tmp)
