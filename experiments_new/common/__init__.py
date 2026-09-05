"""Shared infrastructure for the rebuttal experiments (experiments_new/*).

Modules
-------
seeding   seed_everything
spaces    synthetic benchmark specs (correct optima), search spaces, task factory
yahpo     LCBench (YAHPO Gym surrogate) task factory
methods   method registry: RS / TPE / GP / CMAES / TPE_HB / SMAC / FMP variants
runner    job grid execution with resume, dry-run and smoke modes
io        loading per-run JSON results into pandas
stats     significance tests, ranks, critical-difference diagrams
plots     convergence / regret / state-trajectory / sensitivity figures
"""
