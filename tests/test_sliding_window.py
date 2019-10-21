# -*- coding: utf-8 -*-
"""


"""
import nose
from nose import tools
import numpy as np
from numpy import testing

# dynfunconn
from dyfunconn import sliding_window_indx, sliding_window


def test_sliding_window():
    rng = np.random.RandomState(1)

    data = rng.rand(64, 100)

    dfcg = sliding_window(data, window_length=25, step=1, pairs=None)

    expected = np.load("data/test_sliding_window.npy")

    np.testing.assert_array_equal(dfcg, expected)


def test_sliding_window_indx():
    result_indices1 = np.load("data/test_sliding_indices1.npy")
    result_indices3 = np.load("data/test_sliding_indices3.npy")
    result_indices6 = np.load("data/test_sliding_indices6.npy")

    ts = np.zeros((4, 100))
    wlen = 10

    indices1 = sliding_window_indx(ts, window_length=wlen, overlap=0.5)
    indices3 = sliding_window_indx(ts, window_length=wlen, overlap=0.75)
    indices6 = sliding_window_indx(ts, window_length=wlen, overlap=0.90)

    np.testing.assert_array_equal(result_indices1, indices1)
    np.testing.assert_array_equal(result_indices3, indices3)
    np.testing.assert_array_equal(result_indices6, indices6)
