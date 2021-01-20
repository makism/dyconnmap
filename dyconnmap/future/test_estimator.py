import numpy as np

import sys

sys.path.append("/home/makism/Github/dyconnmap-feature_dataset/")
sys.path.append("/home/makism/Github/dyconnmap-feature_dataset/dyconnmap/future/")
from dyconnmap import tvfcg, sliding_window
from dyconnmap.fc import PLV, Corr, plv_fast

from .dataset import Dataset, Modality
from .basicfilter import passband_filter
from .phaselockingvalue import PhaseLockingValue
from .correlation import Correlation, correlation
from .slidingwindow import SlidingWindow
from .timevar import TimeVarying


def test_phaselockingvalue():
    """Test Phase Locking Value estimator."""
    rng = np.random.RandomState(0)

    n_subjects = 10
    n_rois = 4
    n_samples = 128

    data = rng.rand(n_rois, n_samples)

    ds = Dataset(data, modality=Modality.Raw, fs=128.0)
    conn = PhaseLockingValue(
        filter=passband_filter, filter_opts={"fs": 128.0, "fb": [1.0, 4.0]}
    )

    result = conn(ds)

    f_data = passband_filter(data, fs=128.0, fb=[1.0, 4.0])
    legacy_plv = np.asarray(plv_fast(f_data))

    result = np.float32(result)
    legacy_plv = np.float32(legacy_plv)

    np.testing.assert_array_equal(legacy_plv, result)


def test_phaselockingvalue_time_varying():
    """Test Phase Locking Value estimator (Time-Varying)."""
    rng = np.random.RandomState(0)

    n_subjects = 1
    n_rois = 4
    n_samples = 128 * 3
    fs = 128.0
    cc = 2.0
    step = 10
    fb = [1.0, 4.0]

    cc = 2.0
    step = 5

    data = rng.rand(n_subjects, n_rois, n_samples)
    ds = Dataset(data, modality=Modality.Raw, fs=fs)

    win = TimeVarying(step=step, cc=cc)
    conn = PhaseLockingValue(
        rois=None, filter=passband_filter, filter_opts={"fs": fs, "fb": fb}
    )
    result = conn(ds, win)

    estimator = PLV(fb, fs)
    fcgs = tvfcg(data[0, :, :], estimator, fb, fs, cc, step)

    np.testing.assert_array_equal(result, fcgs)


def test_correlation():
    """Test Correlation estimator."""
    rng = np.random.RandomState(0)

    n_subjects = 1
    n_rois = 32
    n_samples = 128

    data = rng.rand(n_subjects, n_rois, n_samples)

    ds = Dataset(data, modality=Modality.Raw, fs=128.0)

    conn = Correlation(rois=[0, 3])
    result1 = conn(ds)
    result1 = np.array(result1)

    result2 = correlation(data, rois=[0, 3])
    result2 = result2[0]

    np.testing.assert_array_equal(result1, result2)


def test_correlation_sliding_window():
    """Test Correlation estimator (Sliding Window)."""
    rng = np.random.RandomState(0)

    n_subjects = 1
    n_rois = 32
    n_samples = 128 * 3
    fs = 128.0

    window_length = 128
    step = 10

    data = rng.rand(n_subjects, n_rois, n_samples)
    ds = Dataset(data, modality=Modality.Raw, fs=fs)

    conn = Correlation(rois=None)
    win = SlidingWindow(step=step, window_length=window_length)
    result = conn(ds, win)
    result = np.array(result)

    estimator = Corr()
    fcgs = sliding_window(
        data[0, :, :], estimator, window_length=window_length, step=step
    )

    np.testing.assert_array_equal(result, fcgs)
