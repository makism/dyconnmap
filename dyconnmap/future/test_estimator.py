import numpy as np

import sys

sys.path.append("/home/makism/Github/dyconnmap-feature_dataset/")
sys.path.append("/home/makism/Github/dyconnmap-feature_dataset/dyconnmap/future/")
from dyconnmap.fc import plv_fast

from .dataset import Dataset, Modality
from .basicfilter import passband_filter
from .phaselockingvalue import PhaseLockingValue
from .correlation import Correlation, correlation


def test_phaselockingvalue():

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


def test_correlation():
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
