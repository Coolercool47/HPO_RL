# experiments_new — rebuttal experiments for H-MCMC-FMP

All experiments share one job runner, one method registry and one result format,
so any table/figure can be rebuilt from the stored per-run JSON files and methods
(e.g. SMAC later) or seeds can be added without re-running anything.

```
experiments_new/
  common/            seeding, task factories (synthetic + LCBench/YAHPO), method registry,
                     runner (dry-run / smoke / resume / parallel), io, stats, plots, analysis
  configs/           lcbench_instances.json (seeded hardness survey), fmp_tuned.json (E0),
                     baselines_default.json / baselines_tuned.json
  tuning_heldout/    E0  held-out meta-tuning of one shared config per method (LCBench 'tune' instances)
  synt_functions/    E1  19 synthetic configurations, 10-D, B=500, 20 seeds
  lcbench/           E2  10 LCBench test instances, B=200 full-fidelity evals, 20 seeds (+TPE_HB on epochs)
  ablation_ladder/   E3  single-factor ladder FMP -> +chains -> -subsample -> soft -> +DREAM
  ablation_controller/ E4 HMM vs fixed / random / rule-based / learned-emission controllers
  diagnostics/       E5  state trajectories, O_t histograms vs emissions, Baum-Welch A evolution
  sensitivity/       E6  one-at-a-time sweeps of 12 knobs + random configs for fANOVA
```

## Method names

| name | what |
|---|---|
| `RS`, `TPE`, `GP`, `CMAES` | Optuna `RandomSampler`, `TPESampler`, `GPSampler`, `CmaEsSampler` (n_startup = FMP n_init) |
| `TPE_HB` | TPE + Hyperband pruner (BOHB-style), LCBench only, cost counted in epochs |
| `SMAC` | SMAC3 `HyperparameterOptimizationFacade` runner (needs `pip install smac`; not run on Windows) |
| `FMP` | ladder (i): 1 chain, Viterbi, EXPLORE coordinate subsampling, no orchestrator (paper's "FMP-only" arm) |
| `FMP_MC`, `FMP_MC_NOSUB`, `FMP_SOFT` | ladder (ii)–(iv) |
| `FMP_DREAM` | ladder (v): + symmetric DREAM(ZS) kernel, p_dream = 0.5 (paper's full method) |
| `FMP_CTRL_*` | controller ablation on top of `FMP_SOFT` |

All FMP variants are configurations of `hpo_rl/baselines/HMM_MCMC_FMP.py`.

## Workflow

```
python experiments_new/lcbench/select_instances.py          # once: test / tuning instances
python experiments_new/<exp>/run.py --dry-run               # grid + determinism checks, no workload
python experiments_new/<exp>/run.py --smoke                 # n_init+5 evals, 1 seed, 2 tasks -> results_smoke/
python experiments_new/<exp>/run.py --workers 10            # full run (resumable; re-run to fill gaps)
python experiments_new/<exp>/analyze.py                     # tables, tests, CD diagram, curves -> figures/
```

Order for the rebuttal: E6 (sensitivity at Table-5 defaults) -> E0 (held-out tuning) ->
E1/E2 with `--fmp-config tuned --baseline-config tuned` -> E3/E4 -> E5.

Results: `<exp>/results/<task>/<method>/seed_XXX.json`. Each file holds the complete
evaluation trace (configs, values, noise-free values, costs, fidelities, timing) plus,
for FMP, the per-evaluation controller diagnostics (state, posterior, T, scale,
acceptance) and Baum-Welch snapshots.

## Reproducibility

`HMM_MCMC_FMP(seed=...)` routes every draw (Sobol initial design included) through
`numpy.random.default_rng(seed)`; Optuna samplers get the same seed. The dry-run
verifies that identical seeds reproduce identical traces and different seeds differ.
