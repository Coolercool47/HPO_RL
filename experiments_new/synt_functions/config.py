"""E1: synthetic benchmark suite (19 configurations, 10-D, B=500, 20 seeds)."""

from experiments_new.common.spaces import synthetic_specs

DIMS = 10
BUDGET = 500
SEEDS = list(range(20))
TASKS = synthetic_specs(DIMS)
METHODS = ["RS", "TPE", "GP", "CMAES", "FMP", "FMP_DREAM"]
# SMAC is registered but not run on Windows: `python run.py --methods SMAC` adds it later.
