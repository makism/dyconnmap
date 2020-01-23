
# coding: utf-8

# In[1]:

#get_ipython().magic(u'matplotlib inline')


# In[2]:

import os

import numpy as np
np.set_printoptions(precision=3, linewidth=250)

import scipy as sp
from scipy import signal, io

import pandas as pd
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.manifold import MDS
from sklearn import svm, cross_validation, metrics
from sklearn import discriminant_analysis

import sys
sys.path.append('/mnt/Other/DEAP/tmp/')

from dyconnmap.fc import PLI, PAC
from dyconnmap import bands
from dyconnmap import tvfcg_cfc
from dyconnmap import analytic_signal


# ### Process subject

# In[3]:

subject_id = 1
basedir = "/mnt/Other/DEAP/data/data_preprocessed_python/"
data = np.load(basedir + "/s01.dat")
eeg = data['data']

n_trials, n_channels, n_samples = eeg.shape


# In[4]:

fname = ("/mnt/Other/DEAP/data/data_preprocessed_python/s%02d.dat" % (subject_id))

data = np.load(fname)
eeg = data['data'][:, 0:32, :]


# In[5]:

n_trials, n_channels, n_samples = np.shape(eeg)

print(n_trials, n_channels, n_samples)


# In[7]:

trial1 = eeg[0, :]


# In[91]:

p_range = (1, 20)
a_range = (1, 60)
dp = 0.5
da = 0.5

f_phases = np.arange(p_range[0], p_range[1], dp)
f_amps = np.arange(a_range[0], a_range[1], da)
P = len(f_phases)
A = len(f_amps)


# In[92]:

print(f_phases)
print(f_amps)


# In[ ]:

comodu = np.zeros((P, A))

estimator = PLI(0, 0)

for p in range(P):
    f_lo = (f_phases[p], f_phases[p] + dp)
    _, hilberted_lo, _ = analytic_signal(trial1, f_lo)
    phase = np.angle(hilberted_lo)

    for a in range(A):
        f_hi = (f_amps[a], f_amps[a] + da)

        _, hilberted_hi, _ = analytic_signal(trial1, f_hi)
        amp = np.abs(hilberted_hi)

        _, hilberted_lohi, _ = analytic_signal(amp, f_lo)
        phase_lohi = np.angle(hilberted_lohi)

        ts, avg = estimator.estimate_pair(phase, phase_lohi)

        comodu[p, a] = avg


# In[ ]:

np.shape(comodu)


# In[ ]:

np.shape(f_phases)


# In[ ]:

np.shape(f_amps)


# In[ ]:

plt.figure(figsize=(10,5))
plt.pcolor(f_phases, f_amps + da, comodu.T, cmap=cm.jet)
plt.axis([f_phases[0], f_phases[-1], f_amps[0] + da, f_amps[-1]])
plt.colorbar()
plt.title('PAC PLI', size=20)
plt.yticks(np.arange(6, 30, 10),size=20)
plt.xticks(np.arange(1, 20, 5),size=20)
plt.xlabel('Phase frequency (Hz)', size=20)
plt.ylabel('Amplitude frequency (Hz)', size=20)


# In[ ]:

f_phases


# In[ ]:

f_amps
