# -*- coding: utf-8 -*-
""" Phase-Amplitude Coupling

Phase-Amplitude Coupling (*PAC*) is the most famous and prominent approach for
studuying the Cross Frequency Coupling between slow and faster oscillations. The
phase of a low frequency drive the power of a higher frequency.


"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from .estimator import Estimator
from ..analytic_signal import analytic_signal

import numpy as np


def pac(data, f_lo, f_hi, fs, estimator, pairs=None):
    """ Phase-Amplitude Coupling

    Compute the Phase Amplitude Couplgin using the given estimator for the given *data*,
    between the specified *pairs* of channels.


    Parameters
    ----------
    data : array-like, shape = [n_electrodes, n_samples]
        Multichannel recording data.

    pairs : array-like
        Each element is a tuple of length two.

    f_lo : list of length 2
        The low and high frequencies.

    f_hi : list of length 2
        The low and high frequencies.

    fs : float
        Sampling frequency.

    estimator: iplv | plv | pli | corr
        Estimator used
        Valid options:
            'iplv' : Imaginary Phase Locking Value
            'plv'  : Phase Locking Value
            'pli'  : Phase Lag Index

    Returns
    -------
    ts : complex array-like, shape = [n_electrodes, n_electrodes, n_samples]
        The PAC computed each time series.

    avg : complex array-like, shape = [n_electrodes, n_electrodes]
        The average PAC across all samples.
    """
    pac = PAC(f_lo, f_hi, fs, estimator, pairs)
    phases, phases_lohi = pac.preprocess(data)

    return pac.estimate(phases, phases_lohi)


class PAC(Estimator):
    """ Phase Amplitude Coupling (PAC)


    """

    def __init__(self, f_lo, f_hi, fs, estimator, pairs=None):
        self.f_lo = f_lo
        self.f_hi = f_hi
        self.fs = fs
        self.estimator = estimator
        self.pairs = pairs

    def preprocess(self, data):
        hilberted_lo, _, _ = analytic_signal(data, self.f_lo, self.fs)
        phase = np.angle(hilberted_lo)

        hilberted_hi, _, _ = analytic_signal(data, self.f_hi, self.fs)
        amp = np.abs(hilberted_hi)

        hilberted_lohi, _, _ = analytic_signal(amp, self.f_lo, self.fs)
        phase_lohi = np.angle(hilberted_lohi)

        return phase, phase_lohi

    def mean(self, ts):
        return self.estimator.mean(ts)

    def estimate(self, phases, phases_lohi):
        num_ts, ts_len = np.shape(phases)

        self.pairs = [(r1, r2) for r1 in range(0, num_ts) for r2 in range(r1, num_ts)]

        pacs_ts = np.zeros((num_ts, num_ts, ts_len), dtype=np.complex)
        pacs_avg = np.zeros((num_ts, num_ts))

        for pair in self.pairs:
            p1, p2 = pair
            phase1 = phases[p1,]
            phase1_lohi = phases_lohi[p2,]

            ts, avg = self.estimator.estimate_pair(phase1, phase1_lohi)

            pacs_ts[pair] = ts
            pacs_avg[pair] = avg

        return pacs_ts, pacs_avg

    def estimate_pair(self, phase1, phase1_lohi):
        ts, avg = self.estimator.estimate_pair(phase1, phase1_lohi)

        return ts, avg
