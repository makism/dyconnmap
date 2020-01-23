# -*- coding: utf-8 -*-
""" Time-Varying Functional Connectivity Graphs

Time-varying functional connectivity graphs (TVFCGs) (Dimitriadis2010_, Falani2008_) introduce the
idea of processing overlapping segments of neuroelectric signals by defining a
frequency-dependent time window in which the synchronization is estimated;
and then tabulating the results as adjacency matrices. These matrices have a
natural graph-based representation called “functional connectivity graphs”
(FCGs).

An important aspect of the TVFCGs is the “cycle-criterion” (CC) (Cohen2008_).
It regulates the amount of the oscillation cycles that will be considered in
measuring the phase synchrony. In the original proposal :math:`CC = 2.0` was
introduced, resulting into a time-window with width twice the lower period.
TVFCGs on the other, consider the given lower frequency that correspond to the
possibly synchronized oscillations of each brain rhythm and the sampling
frequency. This newly defined frequency-depedent time-window is sliding over
the time series and the network connectivity is estimated. The overlapping
is determined by an arbitrary step parameter.

Given a multi-channel recording data matrix :math:`X^{m \\times n}` of
size :math:`m \\times n` (with :math:`m` channels, and :math:`n` samples), a
frequency range with :math:`F_{up}` and :math:`F_{lo}` the upper and lower
limits, :math:`fs` the sampling frequency, :math:`step` and :math:`CC`, the
computation of these graphs proceeds as follows:

Firstly, based on the :math:`CC` and the specified frequency range
(:math:`F_{lo}` and :math:`fs` ) the window size calculated:

.. math::
    w_{len} = \\frac{ CC }{ F_{lo} } fs

Then, this window is moving per :math:`step` samples and the average
synchronization is computed (between the channels, in a pairwise manner)
resulting into :math:`\\frac{n}{step}` adjacency matrices of size
:math:`n \\times n`.


|

-----

.. [Cohen2008] Cohen, M. X. (2008). Assessing transient cross-frequency coupling in EEG data. Journal of neuroscience methods, 168(2), 494-499.
.. [Dimitriadis2010] Dimitriadis, S. I., Laskaris, N. A., Tsirka, V., Vourkas, M., Micheloyannis, S., & Fotopoulos, S. (2010). Tracking brain dynamics via time-dependent network analysis. Journal of neuroscience methods, 193(1), 145-155.
.. [Falani2008] Fallani, F. D. V., Latora, V., Astolfi, L., Cincotti, F., Mattia, D., Marciani, M. G., ... & Babiloni, F. (2008). Persistent patterns of interconnection in time-varying cortical networks estimated from high-resolution EEG recordings in humans during a simple motor act. Journal of Physics A: Mathematical and Theoretical, 41(22), 224014.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np

from .fc.estimator import Estimator


def tvfcg(data, estimator_instance, fb, fs, cc=2.0, step=5.0, pairs=None):
    """ Time-Varying Functional Connectivity Graphs

    The TVFCGs are computed from the input ``data`` by employing the given
    synchronization estimator (``estimator_instance``).


    Parameters
    ----------
    data : array-like, shape(n_channels, n_samples)
        Multichannel recording data.

    estimator_instance : object
        An object of type :mod:`dyconnmap.fc.Estimator`.

    fb : list of length 2
        The lower and upper frequency.

    fs : float
        Sampling frequency.

    cc : float
        Cycle criterion.

    step : int
        The amount of samples the window will move/slide over the time series.

    pairs : array-like or `None`
        - If an `array-like` is given, notice that each element is a tuple of length two.
        - If `None` is passed, complete connectivity will be assumed.


    Returns
    -------
    fcgs : array-like, shape(n_windows, n_channels, n_channels)
        The computed FCGs.
    """
    preprocess, estimator, avg_func = _validate_estimator(estimator_instance)

    # Preprocess the data (estimator function)
    pp_data = preprocess(data)

    #
    n_channels, n_samples = np.shape(data)
    # window_length = np.int32(np.round((cc / fb[0]) * fs))
    # windows = np.int32(np.round((n_samples - window_length) / step))
    windows, window_length = tvfcg_compute_windows(data, fb, fs, cc, step)

    if window_length >= n_samples:
        raise Exception(
            "The size of window cannot be greater than the number of samples"
        )

    fcgs = np.zeros(
        (windows, n_channels, n_channels), dtype=estimator_instance.data_type
    )

    if pairs is None:
        pairs = [
            (win_id, int(win_id * step), int(window_length + (win_id * step)), c1, c2)
            for win_id in range(windows)
            for c1 in range(0, n_channels)
            for c2 in range(c1, n_channels)
            if c1 != c2
        ]

    for pair in pairs:
        win_id, start, end, c1, c2 = pair

        slice1 = pp_data[c1, ..., start:end]
        slice2 = pp_data[c2, ..., start:end]

        # slice = None
        # try:
        slice_ts, _ = estimator(slice1, slice2)
        # except:
        #   slice = estimator(slice1, slice2)

        fcgs[win_id, c1, c2] = avg_func(slice_ts)

    return fcgs


def tvfcg_cfc(
    data, estimator_instance, fb_lo, fb_hi, fs=128, cc=2.0, step=5, pairs=None
):
    """ Time-Varying Functional Connectivity Graphs (for Cross frequency Coupling)

    The TVFCGs are computed from the input ``data`` by employing the given
    cross frequency coupling synchronization estimator (``estimator_instance``).


    Parameters
    ----------
    data : array-like, shape(n_channels, n_samples)
        Multichannel recording data.

    estimator_instance : object
        An object of type :mod:`dyconnmap.fc.Estimator`.

    fb_lo : list of length 2
        The low and high frequencies.

    fb_hi : list of length 2
        The low and high frequencies.

    fs : float
        Sampling frequency.

    cc : float
        Cycle criterion.

    step : int
        The amount of samples the window will move/slide over the time series.

    pairs : array-like or `None`
        - If an `array-like` is given, notice that each element is a tuple of length two.
        - If `None` is passed, complete connectivity will be assumed.


    Returns
    -------
    fcgs : array-like, shape(n_windows, n_channels, n_channels)
        The computed Cross-Frequency FCGs.


    Notes
    -----
    Not all available estimators in the :mod:`dyconnmap.fc` are valid for estimating
    cross frequency coupling.
    """
    preprocess, estimator, avg_func = _validate_estimator(estimator_instance)

    # Preprocess the data (estimator function)
    pp_data1, pp_data2 = preprocess(data)

    #
    n_channels, n_samples = np.shape(data)
    # window_length = np.int32(np.round((cc / fb_lo[0]) * fs))
    # windows = np.int32(np.round((n_samples - window_length) / step))
    windows, window_length = tvfcg_compute_windows(data, fb_lo, fs, cc, step)

    if window_length >= n_samples:
        raise Exception(
            "The size of window cannot be greater than the number of samples"
        )

    fcgs = np.zeros((windows, n_channels, n_channels))

    if pairs is None:
        pairs = [
            (win_id, (win_id * step), window_length + (win_id * step), c1, c2)
            for win_id in range(windows)
            for c1 in range(0, n_channels)
            for c2 in range(c1, n_channels)
            if c1 != c2
        ]

    for pair in pairs:
        win_id, start, end, c1, c2 = pair
        slice1 = pp_data1[c1, ..., start:end]
        slice2 = pp_data2[c2, ..., start:end]
        slice_ts, _ = estimator(slice1, slice2)
        aslice = avg_func(slice_ts)

        fcgs[win_id, c1, c2] = aslice

    return fcgs


def tvfcg_ts(ts, fb, fs=128, cc=2.0, step=5, pairs=None, avg_func=np.mean):
    """ Time-Varying Function Connectivity Graphs (from time series)

    This implementation operates directly on the given estimated synchronization
    time series (``ts``) and the mean value inside the window is computed.


    Parameters
    ----------
    ts : array-like, shape(n_channels, n_samples)
        Multichannel synchronization time series.

    fb : list of length 2
        The lower and upper frequency.

    fs : float
        Sampling frequency.

    cc : float
        Cycle criterion.

    step : int
        The amount of samples the window will move/slide over the time series.

    pairs : array-like or `None`
        - If an `array-like` is given, notice that each element is a tuple of length two.
        - If `None` is passed, complete connectivity will be assumed.


    Returns
    -------
    fcgs : array-like
        The computed FCGs.
    """
    n_channels, n_channels, n_samples = np.shape(ts)

    window_length = np.int32(np.round((cc / fb[0]) * fs))
    windows = np.int32(np.round((n_samples - window_length) / step))

    if window_length >= n_samples:
        raise Exception(
            "The size of window cannot be greater than the number of samples"
        )

    fcgs = np.zeros((windows, n_channels, n_channels))

    if pairs is None:
        pairs = [
            (win_id, (win_id * step), window_length + (win_id * step), c1, c2)
            for win_id in range(windows)
            for c1 in range(n_channels)
            for c2 in range(c1, n_channels)
        ]

    for pair in pairs:
        win_id, start, end, c1, c2 = pair
        slice_ts = ts[c1, c2, start:end]

        fcgs[win_id, c1, c2] = avg_func(slice_ts)

    return fcgs


def tvfcg_compute_windows(data, fb_lo, fs, cc, step):
    """ Compute TVFCGs Sliding Windows

    A helper function that computes the size and number of sliding windows
    given the parameters.


    Parameters
    ----------
    data : array-like, shape(n_channels, n_samples)
        Multichannel recording data.

    fb_lo :

    fb : list of length 2
        The lower and upper frequency.

    fs : float
        Sampling frequency.

    cc : float
        Cycle criterion.

    step : int
        Stepping.


    Returns
    -------
    windows : int
        The total number of sliding windows.

    window_length : int
        The length of a sliding window; number of samples used to estimated the connectivity.
    """
    *_, n_samples = np.shape(data)
    window_length = np.int32(np.round((cc / fb_lo[0]) * fs))
    windows = np.int32(np.round((n_samples - window_length) / step))

    # print("window_length = {0}".format(window_length))

    if window_length >= n_samples:
        raise Exception(
            "The size of window cannot be greater than the number of samples"
        )

    return windows, window_length


def _validate_estimator(estimator_instance):
    """

    Perform common validity checks for a given estimator.


    Parameters
    ----------
    estimator_instance : object
        An instance of `dyconnmap.fc.Estimator`


    Returns
    -------
    preprocess : function
        A callable function for preprocessing the data.

    estimator : function
        A callable function for estimating the synchronization.

    avg : function
        A callable function for computing the average on each slice.


    Notes
    -----
    This function is used mainly internally.
    """
    if not isinstance(estimator_instance, Estimator):
        raise Exception("Given object is not an Estimator.")

    preprocess = getattr(estimator_instance, "preprocess")
    estimator = getattr(estimator_instance, "estimate_pair")
    avg = getattr(estimator_instance, "mean")

    if not callable(preprocess):
        raise Exception("Preprocess method is not callable.")

    if not callable(estimator):
        raise Exception("Estimator method is not callabled.")

    if not callable(avg):
        raise Exception("Mean method is not callable.")

    return preprocess, estimator, avg
