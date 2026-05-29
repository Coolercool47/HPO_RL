"""Классические алгоритмы для оптимизации гиперпараметров.

- :class:`BOHB` — Bayesian Optimization and Hyperband
- :class:`hyperband` — алгоритм Hyperband
- :class:`TPE` — Tree-structured Parzen Estimator
- :class:`HMM_MCMC` — HMM + MCMC для HPO
- :class:`CMA_ES` — Covariance Matrix Adaptation Evolution Strategy
- :class:`SimpleGA` — простой генетический алгоритм
"""

from hpo_rl.baselines.BOHB import BOHB
from hpo_rl.baselines.hyperband import hyperband
from hpo_rl.baselines.TPE import TPE
from hpo_rl.baselines.HMM_MCMC import HMM_MCMC
from hpo_rl.baselines.CMA_ES import CMA_ES
from hpo_rl.baselines.SimpleGA import SimpleGA

__all__ = [
    "BOHB",
    "hyperband",
    "TPE",
    "HMM_MCMC",
    "CMA_ES",
    "SimpleGA",
]
