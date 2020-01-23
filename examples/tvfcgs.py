# -*- coding: utf-8 -*-
import time

import numpy as np
np.set_printoptions(precision=3, linewidth=160, suppress=True)

from dyconnmap.fc import plv, PLV, pac, PAC
from dyconnmap import tvfcg, tvfcg_ts, tvfcg_cfc


if __name__ == '__main__':

    data = np.load("../examples/data/eeg_32chans_10secs.npy")

    # PLV
    fb = [1.0, 4.0]
    fs = 128.0

    now = time.time()
    ts, avg = plv(data, fb, fs)
    print("Finished in", time.time() - now, "sec")

    # TVFCGs from time seriess
    now = time.time()
    fcgs = tvfcg_ts(ts, [1.0, 4.0], 128)
    print("Finished in", time.time() - now, "sec")

    # TVFCGs
    fb = [1.0, 4.0]
    fs = 128.0

    estimator = PLV(fb, fs)
    fcgs = tvfcg(data, estimator, fb, fs)

    # PAC
    lo = [1.0, 4.0]
    hi = [8.0, 13.0]
    fs = 128.0
    estimator = PLV(fb, fs)

    now = time.time()
    cfc_ts, cfc_avg = pac(data, lo, hi, fs, estimator)
    print("Finished in", time.time() - now, "sec")

    # TVFCGs + PAC
    pac_estimator = PAC(lo, hi, fs, estimator)

    now = time.time()
    fcgs = tvfcg_cfc(data, pac_estimator, lo, hi, fs)
    print("Finished in", time.time() - now, "sec")
