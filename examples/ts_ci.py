# -*- coding: utf-8 -*-
"""

"""
import numpy as np
np.set_printoptions(precision=3, linewidth=256)

from dyconnmap.ts import complexity_index


if __name__ == "__main__":
    rng = np.random.RandomState(0)

    ts = rng.randint(1, 10, [100])
    norm_ci, ci, spectrum = complexity_index(ts, sub_len=25, normalize=True)

    print(("Normalized complexity index: %f" % (norm_ci)))
    print(("Complexity index: %f" % (ci)))
