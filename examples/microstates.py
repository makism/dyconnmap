# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 20:59:54 2014

@author: makism
"""

#%%
import os

#%%
import numpy as np
import scipy as sp
from scipy import io
from numpy import random

np.set_printoptions(precision=3, linewidth=256)


#%%
import dyconnmap
from dyconnmap.microstates import microstates


#%%
x = sp.io.loadmat("data/10secs.mat")['X1']
# x = x[0:2, :]

#%%
mstates = microstates(x, 4)
