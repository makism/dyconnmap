import matplotlib.pyplot as plt
import numpy as np

from dyconnmap.plot import EEGLabChanLocs
from dyconnmap.plot import plot_head_outline, plot_montage


#
locs = EEGLabChanLocs().fromFile("/home/makism/Github/dyconnmap/snippets/emotiv.ced")
electrodes = locs.project2d(0.9)
labels = locs.chans_labels

fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (6, 5), frameon = False)

#
plot_head_outline(scale = 1.5, shift = (0, 0), color = 'k', linewidth = '3', ax=ax[0])
plot_montage(electrodes, ax=ax[0])

#
plot_head_outline(scale = 1.5, shift = (0, 0), color = 'k', linewidth = '3', ax=ax[1])
plot_montage(electrodes, labels, ax=ax[1])

#
ax[1].axis('equal')
ax[0].axis('equal')
plt.show()
