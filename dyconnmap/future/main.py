import numpy as np

import sys

sys.path.append("/home/makism/Github/dyconnmap-feature_dataset/")
from dyconnmap.fc import plv_fast


from dataset import Dataset, Modality
from correlation import Correlation
from slidingwindow import SlidingWindow
from phaselockingvalue import PhaseLockingValue

if __name__ == "__main__":
    rng = np.random.RandomState(0)

    n_subjects = 10
    n_rois = 4
    n_samples = 128

    data = rng.rand(n_subjects, n_rois, n_samples)
    data = rng.rand(n_rois, n_samples)
    data2 = rng.rand(n_rois, n_samples)

    ds = Dataset(data, modality=Modality.FMRI, tr=1.5)
    ds.labels = ["a", "b"]
    ds += data2
    print(ds)

    sw = SlidingWindow(step=10, samples=128, rois=32, window_length=10)
    print(sw)

    conn = Correlation()
    conn(ds)
    print(conn)

    conn = PhaseLockingValue()
    print(conn)

    conn(ds)

    # Check if our new class yields the same results as the previous `plv_fast`.
    result = conn(ds)[0]
    legacy_plv = np.asarray(plv_fast(data))

    result = np.float32(result)
    legacy_plv = np.float32(legacy_plv)

    np.testing.assert_array_equal(legacy_plv, result)
