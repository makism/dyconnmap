# -*- coding: utf-8 -*-
"""

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from .estimator import Estimator
from .plv import PLV, plv, plv_fast
from .pli import PLI, pli
from .iplv import IPLV, iplv, iplv_fast
from .aec import aec
from .esc import esc
from .nesc import nesc
from .cos import cos
from .pec import pec
from .glm import glm
from .pac import PAC, pac
from .mui import mutual_information
from .dpli import dpli
from .wpli import wpli, dwpli
from .coherence import coherence, Coherence
from .icoherence import icoherence
from .corr import corr, Corr
from .crosscorr import crosscorr
from .partcorr import partcorr
from .rho_index import rho_index


__all__ = [
    "Estimator",
    "PLV",
    "plv",
    "plv_fast",
    "PLI",
    "pli",
    "IPLV",
    "iplv",
    "iplv_fast",
    "aec",
    "esc",
    "nesc",
    "pec",
    "glm",
    "rho_index",
    "PAC",
    "pac",
    "mutual_information",
    "dpli",
    "wpli",
    "dwpli",
    "coherence",
    "Coherence",
    "icoherence",
    "corr",
    "Corr",
    "crosscorr",
    "partcorr",
    "cos",
]
