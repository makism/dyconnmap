"""Basic filtering methods."""
# author Avraam Marimpis <avraam.marimpis@gmail.com>

from typing import Tuple

import numpy as np

import scipy
import scipy.signal


def passband_filter(
    data: np.ndarray, fs: float, fb: Tuple[float, float], **kwargs
) -> np.ndarray:
    """Passband filter."""
    if fb[1] <= fb[0]:
        raise Exception(
            f"Upper frequency ({fb[1]} cannot be less or equal than {fb[0]}."
        )

    order = kwargs.get("order", 3)

    passband = [fb[0] / (fs / 2.0), fb[1] / (fs / 2.0)]
    passband = np.ravel(passband)
    b, a = scipy.signal.butter(
        order, passband, "bandpass", analog=False, output="ba"
    )

    filtered = scipy.signal.filtfilt(b, a, data)

    return filtered


def analytic_signal(data: np.ndarray, unwrap: bool = True) -> np.ndarray:
    """Extract the analytic signal."""
    h = scipy.signal.hilbert(data)
    phase = np.angle(h)

    if unwrap:
        u_phase = np.unwrap(phase)
        return u_phase

    return phase
