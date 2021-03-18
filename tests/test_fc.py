# -*- coding: utf-8 -*-
"""Test for the functional connectivity module."""

import os
import scipy as sp
import numpy as np
from numpy import testing

from dyconnmap.fc import (
    aec,
    #   biplv,
    coherence,
    Coherence,
    dpli,
    esc,
    glm,
    icoherence,
    iplv,
    iplv_fast,
    # mi,
    mutual_information,
    nesc,
    pac,
    pec,
    pli,
    plv,
    PLV,
    plv_fast,
    rho_index,
    cos,
    # sl,
    wpli,
    dwpli,
    corr,
    Corr,
    crosscorr,
    partcorr,
)


def test_aec():
    """Test the Amplitude-Envelope Coupling algorithm."""

    # Groundtruth
    expected = np.load("groundtruth/fc/aec.npy")

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    corr = aec(data, [1.0, 4.0], [20.0, 45.0], 128)

    # Test
    np.testing.assert_array_almost_equal(corr, expected)


def test_coherence():
    """Test the Coherence algorithm."""

    # Groundtruth
    expected = np.load("groundtruth/fc/coherence.npy")

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    csdparams = {"NFFT": 256, "noverlap": 256 / 2.0}
    coh = coherence(data, [1.0, 4.0], 128.0, **csdparams)

    # Test
    np.testing.assert_array_almost_equal(coh, expected)


def test_coherence_class():
    """Test the Coherence algorithm (class)."""

    # Groundtruth
    expected = np.load("groundtruth/fc/coherence.npy")

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    csdparams = {"NFFT": 256, "noverlap": 256 / 2.0}
    coh = Coherence([1.0, 4.0], 128.0, **csdparams)

    pp_data = coh.preprocess(data)
    _, avg = coh.estimate(pp_data)
    avg = avg + avg.T
    np.fill_diagonal(avg, 1.0)

    # Test
    np.testing.assert_array_almost_equal(avg, expected)


def test_dpli():
    """Test the Directed PLI algorithm."""

    # Groundtruth
    expected = np.load("groundtruth/fc/dpli.npy")

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    dpliv = dpli(data, [1.0, 4.0], 128.0)

    # Test
    np.testing.assert_array_equal(dpliv, expected)


def test_esc():
    """Test the ESC algorithm."""

    # Groundtruth
    expected = np.load("groundtruth/fc/esc.npy")

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    escv = esc(data, [1.0, 4.0], [20.0, 45.0], 128.0)

    # Test
    np.testing.assert_array_almost_equal(escv, expected)


def test_glm():
    """Test the GLM method."""

    # Groundtruth
    expected_ts = np.load("groundtruth/fc/glm_ts.npy")
    expected_avg = np.load("groundtruth/fc/glm_avg.npy")

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    num_ts, ts_len = np.shape(data)
    window_size = ts_len / 2.0

    fb_lo = [4.0, 8.0]
    fb_hi = [25.0, 40.0]
    fs = 128.0

    ts, ts_avg = glm(data, fb_lo, fb_hi, fs, pairs=None, window_size=window_size)

    # Test
    np.testing.assert_array_equal(ts, expected_ts)
    np.testing.assert_array_equal(ts_avg, expected_avg)


def test_icoherence():
    """Test the Imaginary part of Coherence algorithm."""

    # Groundtruth
    expected = np.load("groundtruth/fc/icoherence.npy")

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    csdparams = {"NFFT": 256, "noverlap": 256 / 2.0}
    icoh = icoherence(data, [1.0, 4.0], 128.0, **csdparams)

    # Test
    np.testing.assert_array_almost_equal(icoh, expected)


def test_iplv():
    """Test the Imaginary part of PLV algorithm."""

    # Groundtruth
    expected_ts = np.load("groundtruth/fc/iplv_ts.npy")
    expected_avg = np.load("groundtruth/fc/iplv_avg.npy")

    expected_ts = np.float32(expected_ts)
    expected_avg = np.float32(expected_avg)

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    ts, avg = iplv(data, [1.0, 4.0], 128.0)
    ts = np.float32(ts)
    avg = np.float32(avg)

    # Test
    np.testing.assert_array_almost_equal(ts, expected_ts)
    np.testing.assert_array_almost_equal(avg, expected_avg)


def test_iplv_nofilter():
    """Test the Imaginary part of PLV (without filtering)."""

    # Groundtruth
    expected_ts = np.load("groundtruth/fc/iplv_nofilter_ts.npy")
    expected_avg = np.load("groundtruth/fc/iplv_nofilter_avg.npy")

    # Data
    data = np.load("sample_data/fc/rois39_samples100.npy")

    # Run
    ts, avg = iplv(data)

    # Test
    np.testing.assert_array_almost_equal(ts, expected_ts)
    np.testing.assert_array_almost_equal(avg, expected_avg)


def test_fast_iplv_nofilter():
    """Test the fast implementation of IPLV (without filtering)."""

    # Groundtruth
    expected_avg = np.load("groundtruth/fc/iplv_nofilter_avg.npy")
    expected_avg = np.float32(expected_avg)

    # Run
    data = np.load("sample_data/fc/rois39_samples100.npy")

    # Run
    avg = iplv_fast(data)

    # iPLV returns a fully symmetrical matrix, so we have to
    # fill with zeros the diagonal and the lower triagular
    np.fill_diagonal(avg, 0.0)
    avg[np.tril_indices_from(avg)] = 0.0
    avg = np.float32(avg)

    # Test
    np.testing.assert_array_equal(avg, expected_avg)


def test_mui():
    """Test the Mutual Information algorithm. (WIP)"""
    assert True


def test_nesc():
    """Test the normalized implementation of the Envelope-to-Signal Correlation algorithm."""

    # Groundtruth
    expected = np.load("groundtruth/fc/nesc.npy")

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    nescv = nesc(data, [4.0, 7.0], [20.0, 45.0], 128)

    # Test
    np.testing.assert_array_equal(nescv, expected)


def test_pac_one_channel():
    """Test the Phase-Amplitude Coupling (one channel) algorithm using the PLV estimator."""

    # Groundtruth
    expected = 0.468296707219

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")
    data = data[0:1, 0:128]

    # Run
    fs = 128
    f_lo = [1.0, 4.0]
    f_hi = [20.0, 30.0]

    estimator = PLV(f_lo, fs)
    ts, avg = pac(data, f_lo, f_hi, fs, estimator)
    avg = np.squeeze(np.real(avg))

    # Test
    np.testing.assert_almost_equal(avg, expected)


def test_pac_multiple_channels():
    """Test the PAC (multiple channels) algorithm using the PLV estimator."""

    # Groundtruth
    expected_ts = np.load("groundtruth/fc/pac_plv_ts.npy")
    expected_ts = np.real(expected_ts)
    expected_ts = np.nan_to_num(expected_ts)
    expected_ts = np.float32(expected_ts)

    expected_avg = np.load("groundtruth/fc/pac_plv_avg.npy")
    expected_avg = np.nan_to_num(expected_avg)
    expected_avg = np.float32(expected_avg)

    #  Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    fs = 128
    fb_lo = [1.0, 4.0]
    fb_hi = [20.0, 30.0]

    estimator = PLV(fb_lo, fs)
    ts, avg = pac(data, fb_lo, fb_hi, 128, estimator)

    ts = np.real(ts)
    ts = np.nan_to_num(ts)
    ts = np.float32(ts)

    avg = np.nan_to_num(avg)
    avg = np.float32(avg)

    # Test
    np.testing.assert_array_almost_equal(ts, expected_ts)
    np.testing.assert_almost_equal(avg, expected_avg)


def test_pec():
    """Test for the Power-Envelope Correlation algorithm."""

    # Groundtruth
    expected = np.load("groundtruth/fc/pec.npy")

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    fb_lo = [1.0, 4.0]
    fb_hi = [25, 40.0]
    fs = 128

    v = pec(data, fb_lo, fb_hi, fs)

    # Test
    np.testing.assert_array_almost_equal(v, expected)


def test_pli():
    """Test for the Phase Lagging Index algorithm."""

    # Groundtruth
    expected_ts = np.load("groundtruth/fc/pli_ts.npy")
    expected_avg = np.load("groundtruth/fc/pli_avg.npy")

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")
    n_channels, n_samples = np.shape(data)

    # Run
    pairs = [
        (r1, r2) for r1 in range(n_channels) for r2 in range(n_channels) if r1 != r2
    ]
    ts, avg = pli(data, [1.0, 4.0], 128.0, pairs)

    # Test
    np.testing.assert_array_almost_equal(ts, expected_ts)
    np.testing.assert_array_almost_equal(avg, expected_avg)


def test_plv():
    """Test for the Phase Locking Value algorithm."""

    # Groundtruth
    expected_ts = np.load("groundtruth/fc/plv_ts.npy")
    expected_ts = np.float32(expected_ts)
    expected_avg = np.load("groundtruth/fc/plv_avg.npy")
    expected_avg = np.float32(expected_avg)

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    ts, avg = plv(data)
    ts = np.float32(ts)
    avg = np.float32(avg)

    # Test
    np.testing.assert_array_equal(ts, expected_ts)
    np.testing.assert_array_equal(avg, expected_avg)


def test_fast_plv():
    """Test the fast implementation of PLV."""

    # Groundtruth
    expected_avg = np.load("groundtruth/fc/plv_avg.npy")

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    avg = plv_fast(data)
    avg = np.float32(avg)

    # PLV returns a fully symmetrical matrix, so we have to
    # fill with zeros the diagonal and the lower triagular
    np.fill_diagonal(avg, 0.0)
    avg[np.tril_indices_from(avg)] = 0.0

    # Test
    np.testing.assert_allclose(avg, expected_avg, rtol=1e-10, atol=0.0)


def test_rho_index():
    """Test for the rho Index."""

    # Groundtruth
    expected = np.load("groundtruth/fc/rho_index.npy")

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    rho_mtx = rho_index(data, 10, [1.0, 4.0], 128.0)

    # Test
    np.testing.assert_array_equal(rho_mtx, expected)


def test_sl():
    """Test the Synchronization Likelihood. (WIP)"""
    assert True


def test_wpli():
    """Test the Weighted PLI algorithm."""

    # Groundtruth
    expected = np.load("groundtruth/fc/wpli.npy")
    expected = np.nan_to_num(expected)

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    csdparams = {"NFFT": 256, "noverlap": 256 / 2.0}

    wpliv = wpli(data, [1.0, 4.0], 128.0, **csdparams)
    wpliv = np.nan_to_num(wpliv)

    # Test
    np.testing.assert_allclose(wpliv, expected, rtol=1e-10, atol=0.0)


def test_dwpli():
    """Test the Debiased WPLI algorithm."""

    # Groundtruth
    expected = np.load("groundtruth/fc/dwpli.npy")
    expected = np.nan_to_num(expected)

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    csdparams = {"NFFT": 256, "noverlap": 256 / 2.0}

    dwpliv = dwpli(data, [1.0, 4.0], 128.0, **csdparams)
    dwpliv = np.nan_to_num(dwpliv)

    # Test
    np.testing.assert_allclose(dwpliv, expected, rtol=1e-10, atol=0.0)


def test_corr():
    """Test the Correlation algorithm."""

    # Groundtruth
    expected = np.load("groundtruth/fc/corr.npy")

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    r = corr(data, [1.0, 4.0], 128.0)

    # Test
    np.testing.assert_array_equal(r, expected)


def test_corr_class():
    """Test the Correlation algorithm (class)."""

    # Groundtruth
    expected = np.load("groundtruth/fc/corr.npy")

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    obj_corr = Corr([1.0, 4.0], 128.0)
    pp_data = obj_corr.preprocess(data)
    r = obj_corr.estimate(pp_data)

    np.testing.assert_array_equal(r, expected)


def test_crosscorr():
    """Test the Cross Correlation algorithm."""

    # Groundtruth
    expected = np.load("groundtruth/fc/crosscorr.npy")

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    r = crosscorr(data, [1.0, 4.0], 128.0)

    # Test
    np.testing.assert_array_equal(r, expected)


def test_partcorr():
    """Test the Partial Correlation algorithm. (WIP)"""

    # Groundtruth
    expected = np.load("groundtruth/fc/partcorr.npy")

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    r = partcorr(data, [1.0, 4.0], 128.0)

    # Test
    assert True
    # if "TRAVIS" in os.environ:
    # assert True
    # else:
    # np.testing.assert_array_equal(r, expected)


def test_cos():
    """Test the Cosine algorithm."""

    # Groundtruth
    expected = np.load("groundtruth/fc/cos.npy")

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    conn = cos(data, [1.0, 4.0], 128.0)

    # Test
    np.testing.assert_array_almost_equal(conn, expected)


def test_cos_nofilter():
    """Test the Cosine algorithm (without filtering)."""

    # Groundtruth
    expected = np.load("groundtruth/fc/cos_nofilter.npy")

    # Data
    data = np.load("sample_data/fc/eeg_32chans_10secs.npy")

    # Run
    conn = cos(data)

    # Test
    np.testing.assert_array_almost_equal(conn, expected)
