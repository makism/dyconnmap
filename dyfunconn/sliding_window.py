# -*- coding: utf-8 -*-
""" Sliding Window




"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from .tvfcgs import _validate_estimator

import numpy as np


def sliding_window(data, estimator_instance, window_length=25, step=1, pairs=None):
    """

    """
    preprocess, estimator, avg_func = _validate_estimator(estimator_instance)

    # Preprocess the data (estimator function)
    pp_data = preprocess(data)

    n_rois, n_samples = np.shape(data)

    if window_length >= n_samples:
        raise Exception(
            "The size of window cannot be greater than the number of samples"
        )

    n_slides = np.int32(np.ceil((n_samples - window_length) / step + 1.0))

    dfcg = np.zeros((n_slides, n_rois, n_rois), dtype=estimator_instance.data_type)

    if pairs is None:
        pairs = [
            (
                win_id,
                np.int32(win_id * step),
                np.int32(win_id * step + window_length),
                c1,
                c2,
            )
            for win_id in range(n_slides)
            for c1 in range(0, n_rois)
            for c2 in range(c1, n_rois)
            if c1 != c2
        ]

    for pair in pairs:
        slide_id, start, end, roi1, roi2 = pair

        slice1 = pp_data[roi1, ..., start:end]
        slice2 = pp_data[roi2, ..., start:end]

        slice_ts, _ = estimator(slice1, slice2)

        dfcg[slide_id, roi1, roi2] = avg_func(slice_ts)

    return dfcg


def sliding_window_indx(data, window_length, overlap=0.75, pairs=None):
    """ Compute the indices and pairs using a sliding window.

    Slide a window over ``data``, and return the indices and offsets.


    Parameters
    ----------
    data : array-like, shape(n_channels, n_samples)
        Multichannel recording data.

    window_length : int
        Number of samples to be used in the computation of the connectivity.

    overlap : float
        Percentage of the ``window_length`` by which the window will overlap when
        sliding forward.

    pairs : array-like or `None`
        - If an `array-like` is given, notice that each element is a tuple of length two.
        - If `None` is passed, complete connectivity will be assumed.


    Returns
    -------
    indices: array - like, shape(n_windows, start_offset, end_offset, n_channels, n_channels)
        Indices of pairs.
    """
    n_channels, n_samples = np.shape(data)

    if window_length >= n_samples:
        raise Exception(
            "The size of window cannot be greater than the number of samples"
        )

    if overlap >= 0.99 or overlap <= 0.05:
        raise Exception("Illegal value for overlap parameter.")

    step = window_length - np.int32(window_length * overlap)
    windows = np.int32(np.round((n_samples - window_length) / step))

    indices = [
        (win_id, int(win_id * step), int(window_length + (win_id * step)), c1, c2)
        for win_id in range(windows)
        for c1 in range(0, n_channels)
        for c2 in range(c1, n_channels)
        if c1 != c2
    ]

    return indices
