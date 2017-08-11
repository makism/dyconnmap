# -*- coding: utf-8 -*-
""" Bi-Phase Locking Value




|

.. [Darvas2009] Darvas, F., Ojemann, J. G., & Sorensen, L. B. (2009). Bi-phase locking—a tool for probing non-linear interaction in the human brain. NeuroImage, 46(1), 123-132.


"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from ..analytic_signal import analytic_signal

"""
function bplv_value=b_plv_hilbert(filtered1_a,filtered1_b,filtered2_a,filtered2_b)

%Darvas-"Bi-phase locking - a tool probing non-linear interaction in the
%human brain"-Neuroimage - 2009
%Input:
%filtered1_a,filtered1_b =  filtered  EEG/MEG signals in the two studying
%frequency bands

%B-PLV measure search for cross frequency
%interplays between two frequency bands in a time-window (e.g. [4 8; 30 45])
%Output:
%        bplv_value varies from 0 to 1

%We extract the instantaneous phase of the signals via Hilbert transform


tic


%STAVROS DIMITRIADIS 3/2010
%http://users.auth.gr/stdimitr/
"""


def biplv(data, fb_lo, fb_hi, fs, pairs=None):
    """ Bi-Phase Locking Value

    Estimate the Bi-Phase Locking Value for the given :attr:`data`,
    between the :attr:`pairs (if given) of channels


    Parameters
    ----------
    data : array-like, shape(n_channels, n_samples)
        Multichannel recording data.

    fb_lo : list of length 2
        The low and high frequencies of the lower band.

    fb_hi : list of length 2
        The low and high frequencies of the upper band.

    fs : float
        Sampling frequency.


    Returns
    -------

    """

    """
    %for the first signal
    %for the first frequency band
    hilbert1_a=hilbert(filtered1_a);
    phase1_a=angle(hilbert1_a);
    unwrap1_a=unwrap(phase1_a);

    %for the second frequency band
    hilbert1_b=hilbert(filtered1_b);
    phase1_b=angle(hilbert1_b);
    unwrap1_b=unwrap(phase1_b);
    """
    _, _, u_phases1 = analytic_signal(data, fb1, fs)

    """
    %for the second signal
    %for the first frequency band
    hilbert2_a=hilbert(filtered2_a);
    phase2_a=angle(hilbert2_a);
    unwrap2_a=unwrap(phase2_a);

    %for the second frequency band
    hilbert2_b=hilbert(filtered2_b);
    phase2_b=angle(hilbert2_b);
    unwrap2_b=unwrap(phase2_b);
    """
    _, _, u_phases2 = analytic_signal(data, fb2, fs)

    """
    bplv_value=0;

    [dim_x dim_y]=size(filtered1_a);

    for k=1:dim_x
        for l=1:dim_y
            bplv_value = bplv_value + (
                exp(
                    i * (
                        unwrap1_a(k, l) + unwrap1_b(k, l) -
                        (unwrap2_a(k, l) * unwrap2_b(k, l))
                    )
                )
            );
        end
    end

    bplv_value=abs(bplv_value/(dim_x*dim_y));
    """
    pass
