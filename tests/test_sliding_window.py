# -*- coding: utf-8 -*-
"""


"""
import nose
from nose import tools
import numpy as np
from numpy import testing

# dynfunconn
from dyfunconn import sliding_window_indx


def test_sliding_window_indx():
    result_fcgs = np.load("data/test_tvfcgs_plv.npy")
    np.testing.assert_array_equal(tvfcg_plv_fcgs, result_fcgs)

    # win_id, start_offset, end_offset, source, target = sliding_window_indx(data, )
