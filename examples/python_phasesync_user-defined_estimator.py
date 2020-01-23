# -*- coding: utf-8 -*-
import numpy as np
from numpy import random
import scipy as sp
from scipy import io

np.set_printoptions(precision=3, linewidth=160, suppress=True)

from dyconnmap import PhaseSync


def myestimator(data, synchpairs, ts, avg, fb, fs):
    print(ts)
    print(avg)
    print(fb)
    print(fs)

    ts = None
    avg = -1.0

    return (ts, avg)

if __name__ == '__main__':
    data = sp.io.loadmat("/home/makism/Github/dyconnmap/examples/data/10secs.mat")['X1']
    data = data[0:2, 0:1024]
    psync = PhaseSync([1.0, 4.0], 128, estimator=myestimator)
    psync.timeseries(data)

    print(psync.ts)
    print(psync.ts_avg)
