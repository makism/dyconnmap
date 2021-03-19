# -*- coding: utf-8 -*-
"""Tests for time-varying functional connectivity graphs module."""

import pytest
import scipy as sp
from scipy import io
import numpy as np
import os
from numpy import testing

from dyconnmap import tvfcg, tvfcg_ts, tvfcg_cfc, tvfcg_compute_windows
from dyconnmap.fc import PAC, PLV, plv


def sample_ufunc(data):
    return np.abs(np.real(data))


@pytest.fixture()
def initialize():
    tvfcg_plv_ts = None
    tvfcg_plv_fcgs = None
    tvfcg_pac_plv_fcgs = None

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

    return tvfcg_plv_ts, tvfcg_plv_fcgs, tvfcg_pac_plv_fcgs


def test_tvfcgs_compute_windows():
    """Test the generated windows from the TVFCGs method."""

    # Groundtruth
    result_windows = np.load("groundtruth/tvfcgs/compute_windows_windows.npy")
    result_window_length = np.load("groundtruth/tvfcgs/compute_windows_window_length.npy")

    # Data
    data = np.load("groundtruth/fc/iplv_ts.npy")

    # Run
    fb = [1.0, 4.0]
    fs = 128.0
    cc = 2.0
    step = 5

    windows, window_length = tvfcg_compute_windows(data, fb, fs, cc, step)

    # Test
    np.testing.assert_array_equal(windows, result_windows)
    np.testing.assert_array_equal(window_length, result_window_length)


def test_tvfcgs_plv(initialize):
    """Test the TVFCGs with the PLV estimator."""

    # Groundtruth
    result_fcgs = np.load("groundtruth/tvfcgs/plv.npy")

    # Data (and Run)
    tvfcg_plv_ts, tvfcg_plv_fcgs, tvfcg_pac_plv_fcgs = initialize

    f32_1 = np.float32(result_fcgs)
    f32_2 = np.float32(tvfcg_plv_fcgs)

    # Test
    np.testing.assert_array_almost_equal(f32_1, f32_2)


def test_tvfcgs_pac_plv(initialize):
    """Test the TVFCGs with the PAC/PLV estimator."""

    # Groundtruth
    result_ts = np.load("groundtruth/tvfcgs/pac_plv.npy")

    # Data (and Run)
    tvfcg_plv_ts, tvfcg_plv_fcgs, tvfcg_pac_plv_fcgs = initialize

    f32_1 = np.float32(result_ts)
    f32_2 = np.float32(tvfcg_pac_plv_fcgs)

    # Test
    np.testing.assert_array_almost_equal(f32_1, f32_2)


def test_tvfcgs_from_plv_ts(initialize):
    """Test the TVFCGs as extracted from the PLV timeseries."""

    # Groundtruth
    result_fcgs = np.load("groundtruth/tvfcgs/from_plv_ts.npy")

    # Data
    tvfcg_plv_ts, tvfcg_plv_fcgs, tvfcg_pac_plv_fcgs = initialize

    # Test
    np.testing.assert_array_almost_equal(tvfcg_plv_ts, result_fcgs)
