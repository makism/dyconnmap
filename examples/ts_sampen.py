# -*- coding: utf-8 -*-
"""

"""

import numpy as np
# np.set_printoptions(precision=3, linewidth=256)
import scipy
from scipy import io

from dyconnmap.ts import sample_entropy


if __name__ == "__main__":
    rng = np.random.RandomState(0)

    x = rng.randint(10, size=1000)
    sampen = sample_entropy(x)

    print(sampen)
