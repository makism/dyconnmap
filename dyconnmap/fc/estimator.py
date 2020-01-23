# -*- coding: utf-8 -*-
""" Base class for estimators

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from abc import ABCMeta
import numpy as np


class Estimator(object, metaclass=ABCMeta):
    """ Base class for estimators.

    Through this abstract class, an estimator can provide the necessary methods
    to be used for a time-varying functional connectivity analysis.


    See also
    --------
    dynfunconn.tvfcgs.tvfcgs: Time-Varying Functional Connectivity Graphs
    dynfunconn.tvfcgs.tvfcgs_cfc: Time-Varying Functional Connectivity Graphs (for Cross frequency Coupling)
    dynfunconn.tvfcgs.tvfcgs_ts: Time-Varying Functional Connectivity Graphs (from time series)
    """

    def __init__(self, fb=None, fs=None, pairs=None):
        self.fs = fs
        self.fb = fb
        self.pairs = pairs
        self.data_type = np.float32
        self._skip_filter = fb is None and fs is None

    def preprocess(self, data):
        """ Preprocess the data.

        """
        pass

    def estimate(self, data, data_against=None):
        """ Estimate the connectivity within the given dataset.

        """
        pass

    def estimate_pair(self, signal1, signal2):
        """ Estimate the connectivity between two signals (time series).

        Notes
        -----
        This is invoked from cross-frequency coupling methods.
        """
        pass

    def mean(self, ts):
        """ The function used to compute the mean synchronization in a timeseries.

        This is needed because some estimators produce complex (imaginary), and
        special treatment is needed (i.e. taking only the real part).


        Returns
        -------
        mtx : array-like
            The average synchronization.

        """
        return np.mean(ts)

    def prepare_pairs(self, rois, symmetric=False):
        """ Prepares a list of indices of ROIs sourced in an estimator.


        Parameters
        ==========
        rois : int
            Number of rois

        """
        if self.pairs is None:
            if symmetric:
                self.pairs = [(r1, r2) for r1 in range(rois) for r2 in range(rois)]
            else:
                self.pairs = [
                    (r1, r2) for r1 in range(rois) for r2 in range(r1, rois) if r1 != r2
                ]

    def typeCast(self, data, cast_type=np.float32):
        return data.astype(cast_type)
