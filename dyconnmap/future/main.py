import numpy as np

import sys

sys.path.append("/home/makism/Github/dyconnmap-feature_dataset/")
from dyconnmap.fc import plv_fast


from dataset import Dataset, Modality

from correlation import Correlation
from timevar import TimeVarying
from slidingwindow import SlidingWindow
from phaselockingvalue import PhaseLockingValue
from pipeline import Pipeline
from basicfilter import passband_filter

if __name__ == "__main__":
    # import pdb
    # pdb.set_trace()

    rng = np.random.RandomState(0)

    n_subjects = 10
    n_rois = 4
    n_samples = 128

    data = rng.rand(n_subjects, n_rois, n_samples)
    data = rng.rand(n_rois, n_samples)
    data2 = rng.rand(n_rois, n_samples)

    # ds = Dataset(data, modality=Modality.FMRI, tr=1.5)
    ds = Dataset(data, modality=Modality.Raw, fs=128.0)
    ds.labels = ["a", "b"]
    ds += data2
    print(ds)

    # sw = SlidingWindow(step=10, samples=128, rois=32, window_length=10)
    # print(sw)

    # fb does where???

    # timevar = TimeVarying(step=10, samples=128, rois=32, window_length=10)
    # print(timevar)

    conn = Correlation()
    # conn(ds, timevar)
    print(conn)

    conn = PhaseLockingValue(filter=passband_filter, filter_opts={"fb": [1.0, 4.0]})
    print(conn)

    # pipeline = Pipeline(
    # stages=[
    # {"filter": passband_filter, "args": {}},
    # {"estimator": PhaseLockingValue()},
    # ]
    # )
    # print(pipeline)

    # # Check if our new class yields the same results as the previous `plv_fast`.
    # result = conn(ds)[0]
    # legacy_plv = np.asarray(plv_fast(data))
    #
    # result = np.float32(result)
    # legacy_plv = np.float32(legacy_plv)
    #
    # np.testing.assert_array_equal(legacy_plv, result)
