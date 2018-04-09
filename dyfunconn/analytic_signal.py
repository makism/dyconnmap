# -*- coding: utf-8 -*-
""" Analytic Signal

For a time series :math:`x(t)`, filtered with a passband :math:`N`th order Butterworth filter (Butter1930_)
in the frequency range :math:`F_{lo} - F_{hi}`; first we compute its analytic representation (Cohen1995_ , Freeman2007_):

.. math::
    u_j(t) = \\frac{1}{\pi} \\textrm{PV} \\int_{+\\infty}^{-\\infty} { \\frac{V_j(t')}{t-t'} dt'}

where :math:`PV` signifies the Cauchy Principal Value. The above equation results
into a complex time-series :math:`V_j(t)`, with a real part :math:`v_j(t)` (the original neuroelectric
time-series) and an imaginary part :math:`u_j(t)`, both as functions of time. Where :math:`j` the
EEG recording electrode id.

From these parts, we are capable to determine the Instantaneous Amplitude:

.. math::
    A_j(t) = \sqrt{ v_j^2 (t) + u_j^2 (t) }

and its Instantaneous Phase counterpart, from:

.. math::
    \phi_j (t) = atan \\frac{u_j(t)} {v_j(t)}

The values in :math:`\phi_j(t)` are originally bound to :math:`[-\pi, \pi]`,
however we employed an *unwrap* transformation (a phase correction
algorithm) in order to eliminate the discontinuities (Dimitriadis2010_, Freeman2002_).

|

-----


.. [Cohen1995] Cohen, L. (1995). Time-frequency analysis (Vol. 1, No. 995,299). Prentice hall.
.. [Freeman2007] Walter J. Freeman (2007) Hilbert transform for brain waves. Scholarpedia, 2(1):1338.
.. [Dimitriadis2010] Dimitriadis, S. I., Laskaris, N. A., Tsirka, V., Vourkas, M., Micheloyannis, S., & Fotopoulos, S. (2010). Tracking brain dynamics via time-dependent network analysis. Journal of neuroscience methods, 193(1), 145-155.
.. [Freeman2002] Freeman, W. J., & Rogers, L. J. (2002). Fine temporal resolution of analytic phase reveals episodic synchronization by state transitions in gamma EEGs. Journal of neurophysiology, 87(2), 937-945.
.. [Butter1930] Butterworth, S. (1930). On the theory of filter amplifiers. Wireless Engineer, 7(6), 536-541.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
import scipy


def analytic_signal(signal, fb, fs=128, order=3):
    """ Passband filtering and Hilbert transformation


    Parameters
    ----------
    signal: real array-like, shape(n_channels, n_samples)
        Input signal

    fb: list of length 2
        The low and high frequencies

    fs: int
        Sampling frequency

    order : int
        Filter order

    Returns
    -------
    filtered_signal: real array-like, shape(n_channels, n_samples)
        The input signal, filtered within the given frequencies

    hilberted_signal: complex array-like, shape(n_channels, n_samples)
        The Hilbert representation of the input signal

    unwrapped_phase: real array-like, shape(n_channels, n_samples)
        The unwrapped phase of the Hilbert representation


    Notes
    -----
    Internally, we use SciPy's Butterworth implementation (`scipy.signal.butter`)
    and the two-pass filter `scipy.signal.filtfilt` to achieve results identical
    to MATLAB.
    """
    fs = float(fs)

    passband = [fb[0] / (fs / 2.0), fb[1] / (fs / 2.0)]
    passband = np.ravel(passband)
    b, a = scipy.signal.butter(
        order, passband, 'bandpass', analog=False, output='ba')

    filtered_signal = scipy.signal.filtfilt(b, a, signal)
    hilberted_signal = scipy.signal.hilbert(filtered_signal)
    unwrapped_phase = np.unwrap(np.angle(hilberted_signal))

    return (filtered_signal, hilberted_signal, unwrapped_phase)
