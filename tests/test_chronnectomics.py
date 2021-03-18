# -*- coding: utf-8 -*-
"""Tests for the chronnectomics module."""
import numpy as np
from numpy import testing

from dyconnmap.chronnectomics import dwell_time, flexibility_index, occupancy_time


def test_dwell_time():
    """Test compute the Dwell Time for a symbolic time series."""

    # Groundtruth
    dwell_expected = np.load(
        "groundtruth/chronnectomics/dwell_time.npy", allow_pickle=True
    ).item()
    mean_expected = np.load(
        "groundtruth/chronnectomics/dwell_mean.npy", allow_pickle=True
    ).item()
    std_expected = np.load(
        "groundtruth/chronnectomics/dwell_std.npy", allow_pickle=True
    ).item()

    # Compute DT
    sts = np.load("sample_data/chronnectomics/symbolic_timeseries.npy")
    dwell, mean, std = dwell_time(sts)

    for symbol_id in dwell.keys():
        np.testing.assert_equal(dwell[symbol_id], dwell_expected[symbol_id])

    for symbol_id in mean.keys():
        np.testing.assert_equal(mean[symbol_id], mean_expected[symbol_id])

    for symbol_id in std.keys():
        np.testing.assert_equal(std[symbol_id], std_expected[symbol_id])


def test_flexibility_index():
    """Test compute the Flexibility Index for a symbolic time series."""

    # Groundtruth
    fi_expected = np.load("groundtruth/chronnectomics/flexibility_index.npy")

    # Compute FI
    sts = np.load("sample_data/chronnectomics/symbolic_timeseries.npy")
    fi = flexibility_index(sts)

    np.testing.assert_equal(fi, fi_expected)


def test_occupancy_time():
    """Test compute the Occupancy Time for a symbolic time series."""

    # Groundtruth
    ot_expected = np.load(
        "groundtruth/chronnectomics/occupancy_time.npy", allow_pickle=True
    ).item()

    # Compute OT
    sts = np.load("sample_data/chronnectomics/symbolic_timeseries.npy")
    ot = occupancy_time(sts)

    np.testing.assert_equal(ot, ot_expected)
