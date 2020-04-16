# -*- coding: utf-8 -*-

import numpy as np
from numpy import testing

# dyconnmap
from dyconnmap.chronnectomics import dwell_time, flexibility_index, occupancy_time


sts = None


def setup_module(module):
    global sts
    sts = np.load("../examples/data/chronnectomics_sts.npy")


def test_dwell_time():
    global sts

    dwell, mean, std = dwell_time(sts)

    dwell_expected = np.load(
        "data/chronnectomics_dwell_time.npy", allow_pickle=True
    ).item()
    mean_expected = np.load(
        "data/chronnectomics_dwell_mean.npy", allow_pickle=True
    ).item()
    std_expected = np.load(
        "data/chronnectomics_dwell_std.npy", allow_pickle=True
    ).item()

    for symbol_id in dwell.keys():
        np.testing.assert_equal(dwell[symbol_id], dwell_expected[symbol_id])

    for symbol_id in mean.keys():
        np.testing.assert_equal(mean[symbol_id], mean_expected[symbol_id])

    for symbol_id in std.keys():
        np.testing.assert_equal(std[symbol_id], std_expected[symbol_id])


def test_flexibility_index():
    global sts

    fi = flexibility_index(sts)

    fi_expected = np.load("data/chronnectomics_flexibility_index.npy")

    np.testing.assert_equal(fi, fi_expected)


def test_occupancy_time():
    global sts

    ot = occupancy_time(sts)

    ot_expected = np.load(
        "data/chronnectomics_occupancy_time.npy", allow_pickle=True
    ).item()

    np.testing.assert_equal(ot, ot_expected)
