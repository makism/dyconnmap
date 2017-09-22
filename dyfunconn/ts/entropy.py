# -*- coding: utf-8 -*-
""" Entropy

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com"

import numpy as np

def __entropy(dts):
    """ Entropy

    """
    l = len(dts)

    unique, counts = np.unique(dts, return_counts=True)
    len_counts = len(counts)
    counts = np.float32(counts)

    v = 0.0
    for i in range(len_counts):
        v += (counts[i] / l) * np.log10(counts[i] / l)

    return v
