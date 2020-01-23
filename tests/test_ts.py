# -*- coding: utf-8 -*-

import numpy as np
from numpy import testing

# dynfunconn
from dyconnmap.ts import (
    aaft,
    fdr,
    phase_rand,
    surrogate_analysis,
    entropy_reduction_rate,
    symoblic_transfer_entropy,
    embed_delay,
    sample_entropy,
    ordinal_pattern_similarity,
    permutation_entropy,
    entropy,
    rr_order_patterns,
    wald,
    markov_matrix,
    transition_rate,
    occupancy_time,
    teager_kaiser_energy,
    dcorr,
    fisher_score,
    fisher_z_plv,
    fisher_z,
    complexity_index,
    fnn,
    cv,
    icc_31,
)


ts = None


def setup_module(module):
    global ts

    ts = np.load("../examples/data/eeg_32chans_10secs.npy")
    ts = ts[0:1, 0:128].ravel()


def test_surrogate_analysis():
    rng = np.random.RandomState(0)

    data = np.load("../examples/data/eeg_32chans_10secs.npy")
    ts1 = data[0, 0:512].ravel()
    ts2 = data[1, 0:512].ravel()

    p_vals, surr_vals, surrogates, r_value = surrogate_analysis(
        ts1, ts2, num_surr=1000, estimator_func=None, ts1_no_surr=False, rng=rng
    )

    expected_result = np.load("data/test_ts_surrogates.npy").squeeze()
    np.testing.assert_array_equal(surrogates, expected_result)

    expected_result = np.load("data/test_ts_surrogates_p_val.npy").squeeze()
    assert p_vals == expected_result

    np.save("/tmp/out.npy", surr_vals)

    expected_result = np.load("data/test_ts_surrogates_corr.npy").squeeze()
    np.testing.assert_array_almost_equal(surr_vals, expected_result)


def test_surrogate_analysis_fdr():
    rng = np.random.RandomState(0)

    data = np.load("../examples/data/eeg_32chans_10secs.npy")
    ts1 = data[0, 0:512].ravel()
    ts2 = data[1, 0:512].ravel()

    p_val, surr_vals, surrogates, r_value = surrogate_analysis(
        ts1, ts2, num_surr=1000, estimator_func=None, ts1_no_surr=False, rng=rng
    )

    num_ts = 2
    num_pvals = num_ts * (num_ts - 1) / 2.0
    num_pvals = np.int32(num_pvals)
    p_vals = np.ones([num_pvals, 1]) * p_val
    h, crit_p = fdr(p_vals, 0.01, "pdep")

    expected_result = np.load("data/test_ts_surrogates_fdr_h.npy")
    assert h == expected_result

    expected_result = np.load("data/test_ts_surrogates_fdr_crit_p.npy")
    assert crit_p == expected_result


def test_surrogate_analysis2():
    rng = np.random.RandomState(0)

    data = np.load("../examples/data/eeg_32chans_10secs.npy")
    ts1 = data[0, 0:512].ravel()
    ts2 = data[1, 0:512].ravel()

    p_val, surr_vals, surrogates, r_value = surrogate_analysis(
        ts1, ts2, num_surr=1000, ts1_no_surr=True, rng=rng
    )

    expected_result = np.load("data/test_ts_surrogates2.npy")
    np.testing.assert_array_equal(surrogates, expected_result)

    expected_result = np.load("data/test_ts_surrogates2_p_val.npy")
    assert p_val == expected_result


def test_surrogate_analysis2_fdr():
    rng = np.random.RandomState(0)

    data = np.load("../examples/data/eeg_32chans_10secs.npy")
    ts1 = data[0, 0:512].ravel()
    ts2 = data[1, 0:512].ravel()

    p_val, surr_vals, surrogates, r_value = surrogate_analysis(
        ts1, ts2, num_surr=1000, ts1_no_surr=True, rng=rng
    )

    num_ts = 2
    num_pvals = num_ts * (num_ts - 1) / 2.0
    num_pvals = np.int32(num_pvals)
    p_vals = np.ones([num_pvals, 1]) * p_val
    h, crit_p = fdr(p_vals, 0.01, "pdep")

    expected_result = np.load("data/test_ts_surrogates2_fdr_h.npy")
    assert h == expected_result


def test_rr_order_similarity():
    data = np.load("../examples/data/eeg_32chans_10secs.npy")
    ts1 = data[0, 0:128].ravel()
    ts2 = data[1, 0:128].ravel()

    m = 3
    tau = 1
    cstr = rr_order_patterns(ts1, ts2, m, tau)

    expected_result = np.load("data/test_ts_cstr.npy")
    assert cstr == expected_result


def test_pemutation_entropy():
    dim = 3
    tau = 1

    pe, npe = permutation_entropy(ts, dim, tau)

    expected_result = np.load("data/test_ts_permutation_entropy.npy")
    assert pe == expected_result

    expected_result = np.load("data/test_ts_n_permutation_entropy.npy")
    assert npe == expected_result


def test_ordinal_pattern_similariy():
    data = np.load("../examples/data/eeg_32chans_10secs.npy")
    ts1 = data[0, 0:128].ravel()
    ts2 = data[1, 0:128].ravel()

    dim = 3
    tau = 1
    diss, ts, distr = ordinal_pattern_similarity(ts1, ts2, dim, tau)

    expected_result = np.load("data/test_ts_dissimilarity.npy")
    assert diss == expected_result

    expected_result = np.load("data/test_ts_ordinal_patterns.npy")
    np.testing.assert_array_equal(ts, expected_result)

    expected_result = np.load("data/test_ts_patterns_distribution.npy")
    np.testing.assert_array_equal(distr, expected_result)


def test_ts_embedded_ts():
    dim = 3
    tau = 1
    embedded_ts = embed_delay(ts, dim, tau)

    expected_result = np.load("data/test_ts_embedded_ts.npy")
    np.testing.assert_array_equal(embedded_ts, expected_result)


def test_phase_rand():
    global ts
    num_surr = 1
    rng = np.random.RandomState(0)

    rp = phase_rand(ts, num_surr, rng)
    rp = np.float32(rp)

    expected_result = np.load("data/test_ts_phase_rand.npy")
    expected_result = np.float32(expected_result)
    np.testing.assert_array_equal(rp, expected_result)


def test_aaft():
    global ts
    num_surr = 1
    rng = np.random.RandomState(0)

    s = aaft(ts, num_surr, rng)

    expected_result = np.load("data/test_ts_afft.npy")
    np.testing.assert_array_equal(s, expected_result)


def test_fdr():
    num_pvals = 2 * (2 - 1) / 2.0
    num_pvals = np.int32(num_pvals)

    pvals = np.ones((1, num_pvals)) * 0.0025
    q = 0.01
    method = "pdep"

    h, crit_p = fdr(pvals, q, method)

    h = h.ravel()[0]
    crit_p = crit_p.ravel()[0]

    assert h == True
    assert crit_p == 0.0025


def test_entropy_reduction_rate():
    mean1 = np.mean(ts)
    sx = ts
    sx[ts > mean1] = 1
    sx[ts < mean1] = 2

    val = entropy_reduction_rate(sx)

    np.testing.assert_almost_equal(val, 0.622877964074)


def test_symbolic_transfer_entropy():
    data = np.load("../examples/data/eeg_32chans_10secs.npy")
    ts1 = data[0, 0:128].ravel()
    ts2 = data[1, 0:128].ravel()

    # Symobolization scheme based on the mean value
    mean1 = np.mean(ts1)
    sx = ts1
    sx[ts1 > mean1] = 1
    sx[ts1 < mean1] = 0

    mean2 = np.mean(ts2)
    sy = ts2
    sy[ts2 > mean2] = 1
    sy[ts2 < mean2] = 0

    diff, xy, yx = symoblic_transfer_entropy(sx, sy)

    np.testing.assert_almost_equal(diff, 0.0458895478268)
    np.testing.assert_almost_equal(xy, 0.124565647163)
    np.testing.assert_almost_equal(yx, 0.0786760993358)


def test_wald():
    # data = np.load("../examples/data/wald_test_ts.npy").item()
    data = np.load("../examples/data/wald_test_ts.npy", encoding="latin1").item()

    x = data["x"]
    y = data["y"]

    w, r, e, we = wald(x, y)
    np.testing.assert_almost_equal(w, -6.07394435273)


def test_markov_chain():
    rng = np.random.RandomState(0)

    symts = rng.randint(0, 4, 100)
    mtx = markov_matrix(symts)

    expected_result = np.load("data/ts_markov_chain.npy")
    np.testing.assert_array_almost_equal(mtx, expected_result)


def test_markov_tr():
    rng = np.random.RandomState(0)

    symts = rng.randint(0, 4, 100)
    tr = transition_rate(symts)

    expected_result = np.load("data/test_ts_markov_tr.npy")
    assert expected_result == tr


def test_markov_oc():
    rng = np.random.RandomState(0)

    symts = rng.randint(0, 4, 100)
    oc, _ = occupancy_time(symts)

    expected_result = np.load("data/test_ts_markov_oc.npy")
    np.testing.assert_array_equal(oc, expected_result)


def test_markov_oc_2():
    rng = np.random.RandomState(0)

    symts = rng.randint(0, 4, 100)
    oc, _ = occupancy_time(symts, symbol_states=4)

    expected_result = np.load("data/test_ts_markov_oc.npy")
    np.testing.assert_array_equal(oc, expected_result)


def test_sample_entropy():
    rng = np.random.RandomState(0)

    symts = rng.randint(10, size=100)

    sampen = sample_entropy(symts)
    np.testing.assert_almost_equal(sampen, 2.37954613413)


def test_entropy():
    rng = np.random.RandomState(0)

    symts = rng.randint(10, size=100)

    result = -entropy(symts)
    np.testing.assert_almost_equal(result, 0.981787607585)


def test_teager_kaiser_energy():
    rng = np.random.RandomState(0)

    ts = rng.rand(1, 100)

    tko = teager_kaiser_energy(ts)

    expected_result = np.load("data/ts_tko.npy")
    np.testing.assert_array_almost_equal(tko, expected_result)


def test_dcorr():
    rng = np.random.RandomState(0)

    x = rng.rand(100)
    y = rng.rand(100)

    result = dcorr(x, y)
    np.testing.assert_array_almost_equal(result, 0.144839163386)


def test_fisher_score():
    rng = np.random.RandomState(0)

    x = rng.rand(100)
    y = rng.rand(100)

    result = fisher_score(x, y)
    np.testing.assert_array_almost_equal(result, 0.138355761177)


def test_fisher_z_plv():
    """ WIP """
    from dyconnmap.fc import plv

    np.set_printoptions(precision=2, linewidth=256)

    # print ""
    data = np.load("../examples/data/eeg_32chans_10secs.npy")
    ts, avg = plv(data, [1.0, 4.0], 128.0)

    symm_avg = avg + avg.T
    np.fill_diagonal(symm_avg, 1.0)

    # print symm_avg
    ts = fisher_z_plv(avg)
    # print ts


def test_complexity_index():
    expected_ci = np.float32(
        np.load("data/test_ts_ci_complexity_index.npy").flatten()[0]
    )
    expected_spectrum = np.load("data/test_ts_ci_spectrum.npy")

    ts = np.load("data/test_ts_ci_ts.npy")
    ci, spectrum = complexity_index(ts, sub_len=50)

    # np.testing.assert_array_equal(expected_ci, ci)
    assert expected_ci == ci
    np.testing.assert_array_equal(expected_spectrum, spectrum)


def test_fnn():
    ts = np.load("data/test_ts_fnn.npy")

    min_dimension = fnn(ts, tau=5)
    assert min_dimension == 19

    min_dimension = fnn(ts, tau=5)
    assert min_dimension == 19

    min_dimension = fnn(ts, tau=15)
    assert min_dimension == 7


def test_icc_31():
    X = [
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ]

    icc_val = icc_31(X)
    np.testing.assert_almost_equal(icc_val, 0.7148407148407155)


def test_cv():
    rng = np.random.RandomState(0)
    X = rng.randint(0, 10, 100)

    cv_val = cv(X)
    np.testing.assert_almost_equal(cv_val, 0.64834403973994137)
