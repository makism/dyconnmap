# -*- coding: utf-8 -*-
"""Fisher Scoring Algorithm


|

-----

.. [Stam2007] Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). Phase lag index: assessment of functional connectivity from multi channel EEG and MEG with diminished bias from common sources. Human brain mapping, 28(11), 1178-1193.
.. [Hardmeier2014] Hardmeier, M., Hatz, F., Bousleiman, H., Schindler, C., Stam, C. J., & Fuhr, P. (2014). Reproducibility of functional connectivity and graph measures based on the phase lag index (PLI) and weighted phase lag index (wPLI) derived from high resolution EEG. PloS one, 9(10), e108648.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np


def fisher_score(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """

    Parameters
    ---------
    x :

    y :

    Returns
    -------

    """
    lx = len(x)
    ly = len(y)

    if lx != ly:
        raise Exception("")

    fsc = np.abs(np.mean(x) - np.mean(y))  # type: ignore
    fsc /= np.sqrt(np.var(x) + np.var(y))  # type: ignore

    return fsc
