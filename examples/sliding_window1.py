"""


"""

import numpy as np

from dyconnmap import sliding_window_indx


if __name__ == '__main__':
    ts = np.zeros((4, 100))
    wlen = 10

    indices1 = sliding_window_indx(ts, window_length=wlen, overlap=0.5)
    indices3 =  sliding_window_indx(ts, window_length=wlen, overlap=0.75)
    indices6 = sliding_window_indx(ts, window_length=wlen, overlap=0.90)

    print(indices1)
    print(indices3)
    print(indices6)
