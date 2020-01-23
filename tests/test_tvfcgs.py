# -*- coding: utf-8 -*-
"""


"""
import nose
from nose import tools
import scipy as sp
from scipy import io
import numpy as np
import os
from numpy import testing

# dynfunconn
from dyconnmap import tvfcg, tvfcg_ts, tvfcg_cfc, tvfcg_compute_windows
from dyconnmap.fc import PAC, PLV, plv


tvfcg_plv_ts = None
tvfcg_plv_fcgs = None
tvfcg_pac_plv_fcgs = None


def sample_ufunc(data):
    return np.abs(np.real(data))


def setup_module(module):
    global tvfcg_plv_ts
    global tvfcg_plv_fcgs
    global tvfcg_pac_plv_fcgs

    original_data = np.load("../examples/data/eeg_32chans_10secs.npy")

    # TVFCGS with PLV
    data = original_data[0:2, 0:1024]
    fb = [1.0, 4.0]
    fs = 128
    estimator = PLV(fb, fs)
    tvfcg_plv_fcgs = tvfcg(data, estimator, fb, fs)

    # TVFCGS with PAC and PLV
    data = original_data[..., 0:1024]
    fb = [1.0, 4.0]
    fs = 128
    f_lo = fb
    f_hi = [20.0, 30.0]
    estimator = PLV(fb, fs)
    pac = PAC(f_lo, f_hi, fs, estimator)
    tvfcg_pac_plv_fcgs = tvfcg_cfc(data, pac, f_lo, f_hi, fs)

    # TVFCGS with PLV (ts)
    fb = [1.0, 4.0]
    fs = 128.0
    estimator = PLV(fb, fs)
    u_phases = estimator.preprocess(data)
    ts, avg = estimator.estimate(u_phases)
    tvfcg_plv_ts = tvfcg_ts(ts, [1.0, 4.0], 128, avg_func=estimator.mean)


def test_tvfcgs_compute_windows():
    data = np.load("data/test_iplv_ts.npy")

    fb = [1.0, 4.0]
    fs = 128.0
    cc = 2.0
    step = 5

    windows, window_length = tvfcg_compute_windows(data, fb, fs, cc, step)

    result_windows = np.load("data/test_tvfcgs_compute_windows_windows.npy")
    np.testing.assert_array_equal(windows, result_windows)

    result_window_length = np.load("data/test_tvfcgs_compute_windows_window_length.npy")
    np.testing.assert_array_equal(window_length, result_window_length)


def test_tvfcgs_plv():
    result_fcgs = np.load("data/test_tvfcgs_plv.npy")
    np.testing.assert_array_equal(tvfcg_plv_fcgs, result_fcgs)


def test_tvfcgs_pac_plv():
    result_ts = np.load("data/test_tvfcgs_pac_plv.npy")
    np.testing.assert_array_equal(tvfcg_pac_plv_fcgs, result_ts)


def test_tvfcgs_from_plv_ts():
    result_fcgs = np.load("data/test_tvfcgs_from_plv_ts.npy")

    if "TRAVIS" in os.environ:
        # We have to use the following to make the test work on Travis
        np.testing.assert_allclose(tvfcg_plv_ts, result_fcgs, rtol=1e-10, atol=0.0)
    else:
        # The following tests pass locally; but they fail on Travis o_O
        np.testing.assert_array_equal(tvfcg_plv_ts, result_fcgs)
