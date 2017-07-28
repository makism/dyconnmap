__version__ = '0.99-git'

from .tvfcgs import tvfcg, tvfcg_ts, tvfcg_cfc, tvfcg_compute_windows
from .analytic_signal import analytic_signal
from .cluster import NeuralGas, MergeNeuralGas

__all__ = ['analytic_signal',
           'fc_estimators',
           'bands',
           'tvfcg', 'tvfcg_cfc', 'tvfcg_ts',
           'sim_models',
           'surr_analysis']
