import matplotlib.pyplot as plt
import numpy as np

from dyfunconn.plot import EEGChanLocs, EEGLabChanLocs
from dyfunconn.plot import topoplot, conn_topoplot, plot_head_outline, plot_montage


locs = EEGLabChanLocs().fromFile("/home/makism/Github/dyfunconn/snippets/emotiv.ced")


# Create a random connectivity matrix
np.random.RandomState(0)
r = np.random.rand(14, 14)

conn_topoplot(r, locs)

#
# def filter_out(val):
#     return val > 0.75
#
# fig, axes = conn_topoplot(r, locs, 'PLV', filter=filter_out, line_width="prop")
#
#
#
# # Render only the 25% of the strongest links
# fig, axes = conn_topoplot(r, locs, 'PLV', view=0.25, line_width=3.0)
#
# from dyfunconn.plot import volume_conduction
# volume_conduction()
#
# from dyfunconn.plot import states_transition, ciruclar_states_transition
# states_transition(r)
#
# ciruclar_states_transition(r, threshold=0.70, filter_out=[1, 2, 3])
