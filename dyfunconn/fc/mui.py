# -*- coding: utf-8 -*-
""" Mutual Information

Mutual Information (*MI*),



|

-----

.. [Vinck2011] Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F., & Pennartz, C. M. (2011). An improved index of phase-synchronization for electrophysiological data in the presence of volume-conduction, noise and sample-size bias. Neuroimage, 55(4), 1548-1565.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np

EPS = np.finfo(float).eps

#function [Iab,Pab,Pa,Pb] = my_mutualInformation(a,b,normalize,nbins)


def mutual_information(x, y, n_bins = 10):
    """

    Parameters
    ----------


    Returns
    -------

    """
    if n_bins is None:
        n_bins = np.round(np.sqrt(len(a) / 10.0))

# % Joint histogram
# abHist = hist2(a,b,nbins);
#
# % Marginal histograms
# aHist = sum(abHist,1);
# bHist = sum(abHist,2);
#
# % Probabilities
# N = sum(aHist);
# Pa = aHist/N;
# Pb = bHist/N;
# Pab = abHist/N;
#
# % Disable divide by 0 and log of 0 warnings
# warning('off');
# Ha = (Pa .* log(Pa));
# id = isfinite(Ha);
# Ha = - sum(Ha(id));
#
# Hb = (Pb .* log(Pb));
# id = isfinite(Hb);
# Hb = - sum(Hb(id));
#
# Hab = (Pab .* log(Pab));
# id = isfinite(Hab);
# Hab = - sum(Hab(id));
# warning('on');
#
#
#     %normalized
#     if normalize
#         Iab=[Ha + Hb] / (2*Hab);
#     else
#         Iab = Ha + Hb - Hab;
#     end
#
# return

    # bins = (5, 5)
    # sigma = 1.0
    # normalized = True
    #
    # jh = np.histogram2d(x, y, bins=bins)[0]
    #
    # # smooth the jh with a gaussian filter of given sigma
    # ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
    #                         output=jh)
    #
    # # compute marginal histograms
    # jh = jh + EPS
    # sh = np.sum(jh)
    # jh = jh / sh
    # s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    # s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))
    #
    # # Normalised Mutual Information of:
    # # Studholme,  jhill & jhawkes (1998).
    # # "A normalized entropy measure of 3-D medical image alignment".
    # # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    # if normalized:
    #     mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
    #           / np.sum(jh * np.log(jh))) - 1
    # else:
    #     mi = (np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
    #           - np.sum(s2 * np.log(s2)))
    #
    # return mi
