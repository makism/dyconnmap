# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(precision=3, linewidth=256)

from dyconnmap.ts import fdr, surrogate_analysis


if __name__ == "__main__":
    rng = np.random.RandomState(0)

    data = np.load(
        "/home/makism/Github/dyconnmap/examples/data/eeg_32chans_10secs.npy")
    ts1 = data[0, :].ravel()
    ts2 = data[1, :].ravel()

    p_val, corr_surr, surrogates, r_value = surrogate_analysis(
        ts1, ts2, num_surr=1000, ts1_no_surr=True, rng=rng)

    num_ts = 2
    p_vals = np.ones([num_ts * (num_ts - 1) / 2, 1]) * p_val
    q = 0.01
    method = 'pdep'
    h, crit_p = fdr(p_vals, q, method)

    print("p-value: {0}, h: {1} (critical p-value: {2})".format(p_val, h, crit_p))
