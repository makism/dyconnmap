# -*- coding: utf-8 -*-
""" Commonly used band frequencies

For your convenience we have predefined some widely adopted brain rhythms.
You can access them with

.. code-block:: python
   :linenos:

    from dyconnmap.bands import *
    print(bands['alpha'])


=============  ==================  =================
  brainwave     frequency (Hz)      variable/index
=============  ==================  =================
     δ           [1.0, 4.0]          bands['delta']
     θ           [4.0, 8.0]          bands['theta']
     α1          [7.0, 10.0]         bands['alpha1']
     α2          [10.0, 13.0]        bands['alpha2']
     α           [7.0, 13.0]         bands['alpha']
     μ           [8.0, 13.0]         band['mu']
     β           [13.0, 25.0]        bands['beta']
     γ           [25.0, 40.0]        bands['gamma']
=============  ==================  =================

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>


bands = {
    'delta': [1.0, 4.0],
    'theta': [4.0, 8.0],
    'mu': [8.0, 13.0],
    'alpha': [7.0, 13.0], 'alpha1': [7.0, 10.0], 'alpha2': [10.0, 13.0],
    'beta': [13.0, 25.0],
    'gamma': [25.0, 40.0]
}
