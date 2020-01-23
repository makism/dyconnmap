# -*- coding: utf-8 -*-

import time
import numpy as np
np.set_printoptions(precision=3, linewidth=80)

from dyconnmap import tvfcg, tvfcg_cfc
from dyconnmap.fc import MI, PLV, PLI
from dyconnmap.fc import PAC
from dyconnmap.fc import plv_within


def try_PLV(data):
    fb = [1.0, 4.0]
    fs = 128
    pairs = None

    ts, avg = plv_within(data, fb, fs)

    print(avg)


def try_MI(data):
    bins = 10
    pairs= None
    fb_lo = [1.0, 4.0]
    fb_hi = [8.0, 13.0]
    fs = 128
    n_jobs = 1
    estimator_instance = MI(bins, fb_lo, fb_hi, fs, pairs, n_jobs)

    preprocess = getattr(estimator_instance, "preprocess")
    result = preprocess(data)

    estimator = getattr(estimator_instance, "estimate")
    result, result_norm = estimator(result)

    print("MI mtx shape:", np.shape(result))


def try_TVFCG(data):
    fb = [1.0, 4.0]
    fs = 128
    pairs = None
    n_jobs = 1
    cc = 2.0
    step = 5

    pli = PLI(fb, fs, pairs)
    fcgs = tvfcg(data, pli, fb, fs, cc, step)

    return fcgs


def try_PAC(data):
    fb = [1.0, 4.0]
    fs = 128
    pairs = None
    n_jobs = 1

    plv = PLV(fb, fs, pairs)

    f_lo = [1.0, 4.0]
    f_hi = [20.0, 30.0]
    pac = PAC(f_lo, f_hi, fs, plv)
    phases, phases_lohi = pac.preprocess(data)
    cfc, avg = pac.estimate(phases, phases_lohi)

    return cfc, avg

def try_TVFCG_PAC(data):

    fb = [1.0, 4.0]
    fs = 128
    f_lo = fb
    f_hi = [20.0, 30.0]
    estimator = PLV(fb, fs)
    pac = PAC(f_lo, f_hi, fs, estimator)
    fcgs = tvfcg_cfc(data, pac, f_lo, f_hi, fs)

    return fcgs


if __name__ == '__main__':
    data = np.load("/home/makism/Github/dyconnmap/examples/data/10secs.npy")

    # try_MI(data)
    try_PLV(data)
    #
    # ts, avg = plv_within(data, [1.0, 4.0], 128)
    # print avg
    #
    # now = time.time()
    # fcgs = try_TVFCG(data)
    # print "Finished in", time.time() - now, "sec"
    #
    # now = time.time()
    # cfc, avg = try_PAC(data)
    # print "Finished in", time.time() - now, "sec"

    # now = time.time()
    # fcgs = try_TVFCG_PAC(data)
    # print "Finished in", time.time() - now, "sec"
    #
    # fcg0 = fcgs[0, ]
    # print fcg0
