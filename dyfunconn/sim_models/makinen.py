# -*- coding: utf-8 -*-
"""


Notes
-----
Based on the MATLAB code from http://www.cs.bris.ac.uk/home/rafal/phasereset/phase.zip

|

-----

.. [Yeung] Yeung, N., Bogacz, R., Holroyd, C. B., Nieuwenhuis, S., & Cohen, J. D. (2007). Theta phase resetting and the error‐related negativity. Psychophysiology, 44(1), 39-49.
.. [Makinen] Mäkinen, V., Tiitinen, H., & May, P. (2005). Auditory event-related responses are generated independently of ongoing brain activity. Neuroimage, 24(4), 961-968.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np


def makinen(frames, epochs, fs, min_fr, max_fr, rng=None):
    """ Makinen

    Parameters
    ----------
    frames : int
        Number of signal frames per each trial.

    epochs : int
        Number of simulated trials.

    fs : float
        Sampling frequency.

    min_fr :
        Minimum frequency of the sinusoid which is being reset.

    max_fr :
        Maximum frequency of the sinusoid which is being reset.

    rng : object or None
        An object of type numpy.random.RandomState


    Returns
    -------

    """
    signal = np.zeros(epochs * frames)

    fs = np.float32(fs)

    for i in range(0, 4):
        pass1 = phasereset(frames, epochs, fs, min_fr, max_fr, rng)

        for frame in range(frames):
            signal[frame] += pass1[frame]

    return signal


def phasereset(frames, epochs, fs, min_fr, max_fr, rng=None):
    """ Phasereset


    Parameters
    ----------
    frames :

    epochs :

    fs :

    min_fr :

    max_fr :

    rng : object or None
        An object of type numpy.random.RandomState


    Returns
    -------

    """
    if rng is None:
        rng = np.random.RandomState()

    position = frames / 2.0
    tjitter = 0.0

    signal = np.zeros(epochs * frames)

    for trial in range(epochs):
        wavefr = np.random.random_sample() * (max_fr - min_fr) + min_fr
        initphase = np.random.random_sample() * 2.0 * np.pi
        phase = 0.0
        pos = position + np.round(np.random.random_sample() * tjitter)

        for i in range(0, frames):
            if i < pos:
                phase = i / fs * 2.0 * np.pi * wavefr + initphase
            else:
                phase = (i - pos) / fs * 2.0 * np.pi * wavefr

            offset = (trial - 1) * frames + i
            sample = np.sin(phase)
            signal[offset] = sample

    return signal
