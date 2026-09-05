# Runbook: rebuttal experiments in the right order

All commands run from the code repo root (`HPO_RL/HPO_RL`) with the project venv
(`.venv\Scripts\python.exe` on Windows, `.venv/bin/python` elsewhere). Every runner is
resumable: re-running the same command skips finished runs and fills gaps. Add
`--workers N` to use N processes (default: CPU count minus 2).

## 0. Environment (new machine)

```
powershell -ExecutionPolicy Bypass -File experiments_new\setup_env.ps1     # Windows
bash experiments_new/setup_env.sh                                           # Linux / WSL / macOS
```

The script creates `.venv`, installs `experiments_new/requirements.txt`, installs
`yahpo-gym` without its outdated dependency pins, clones the LCBench surrogate data
(sparse checkout, ~150 MB) into `yahpo_data/`, and runs a dry-run. To move results
between machines copy the `experiments_new/*/results*/` directories and
`experiments_new/configs/`; nothing else is stateful.

## 1. Instance selection (once, ~1 min)

```
python experiments_new/lcbench/select_instances.py --n-test 10 --n-tune 8
```

Writes `configs/lcbench_instances.json`: 10 test instances (paper's 5 + 5 hardest by a
seeded random survey) and 8 disjoint tuning instances. Already done on this machine;
re-running reproduces the same file (seed 12345).

## 2. Sanity checks (5 min)

```
python experiments_new/synt_functions/run.py --dry-run
python experiments_new/lcbench/run.py --dry-run
python experiments_new/sensitivity/run.py --dry-run
python experiments_new/synt_functions/run.py --smoke
python experiments_new/lcbench/run.py --smoke
```

Dry-run validates every method (determinism per seed, seed sensitivity, kwargs) with
no workload. Smoke runs n_init + 5 evaluations on two tasks into `results_smoke/`.

## 3. E6 sensitivity at the Table-5 defaults (~25 min on 10 workers)

```
python experiments_new/sensitivity/run.py --workers 10
python experiments_new/sensitivity/analyze.py
```

Figures in `sensitivity/figures/`: `oat_synthetic.png`, `oat_lcbench.png` (one panel
per knob, dashed line = base configuration), `oat_influence.csv` (knob ranking),
`fanova_importance.csv`. Read this before tuning: knobs with flat curves can be
frozen at their defaults; the paper's sensitivity paragraph comes from here.

## 4. E0 held-out tuning (~1.5 h total)

```
python experiments_new/tuning_heldout/run.py --method FMP   --n-trials 100 --workers 10
python experiments_new/tuning_heldout/run.py --method TPE   --n-trials 100 --workers 10
python experiments_new/tuning_heldout/run.py --method CMAES --n-trials 100 --workers 10
```

Each meta-trial runs the candidate on the 8 tuning instances x 3 seeds (200 evals).
Outputs `configs/fmp_tuned.json` and `configs/baselines_tuned.json`. The study is
stored in `tuning_heldout/results/<method>/meta_study.db`, so an interrupted run
continues from where it stopped. The script refuses to run if tuning and test
instances overlap.

## 5. E1 synthetic and E2 LCBench, main comparison (~3 h with GP-BO)

Run the untuned reference first (cheap, separate output directory), then the tuned
configuration used in the paper tables:

```
python experiments_new/synt_functions/run.py --workers 10 --out experiments_new/synt_functions/results_table5
python experiments_new/lcbench/run.py        --workers 10 --out experiments_new/lcbench/results_table5

python experiments_new/synt_functions/run.py --workers 10 --fmp-config tuned --baseline-config tuned
python experiments_new/lcbench/run.py        --workers 10 --fmp-config tuned --baseline-config tuned

python experiments_new/synt_functions/analyze.py            # -> synt_functions/figures/
python experiments_new/lcbench/analyze.py                   # -> lcbench/figures/
python experiments_new/synt_functions/analyze.py --results experiments_new/synt_functions/results_table5
python experiments_new/lcbench/analyze.py        --results experiments_new/lcbench/results_table5
```

GP-BO dominates the run time (about 150 s per 500-evaluation run). To get everything
else first: `--methods RS TPE CMAES FMP FMP_DREAM` now and `--methods GP` later; the
analysis merges whatever is present. Outputs per suite: `table_<suite>.tex` (mean ±
SEM, bold best, * = Holm-corrected Mann-Whitney p < 0.05 vs TPE), `tests_vs_TPE.csv`,
`wins_vs_TPE.csv`, `cross_task.json` (Wilcoxon over tasks, Friedman, average ranks,
Nemenyi CD), `cd_<suite>.png`, `convergence_<suite>.png`, `overhead.csv`.

## 6. E3 ablation ladder and E4 controller ablation (~30 min)

```
python experiments_new/ablation_ladder/run.py     --workers 10 --fmp-config tuned
python experiments_new/ablation_controller/run.py --workers 10 --fmp-config tuned
python experiments_new/ablation_ladder/analyze.py
python experiments_new/ablation_controller/analyze.py
```

Ladder reference = `FMP` (step i); controller reference = `FMP_CTRL_HMM`. Decision
gates from the plan: if `FMP_DREAM` is not significantly better than the best earlier
step, the paper is reframed around H-MCMC-FMP; if `FMP_CTRL_RULE` or `FMP_CTRL_RANDOM`
match `FMP_CTRL_HMM`, the HMM contribution is not supported.

## 7. E5 diagnostics (1 min, no new runs)

```
python experiments_new/diagnostics/run.py --results experiments_new/synt_functions/results --method FMP_DREAM --seeds 0
python experiments_new/diagnostics/run.py --results experiments_new/lcbench/results        --method FMP_DREAM --seeds 0
```

Produces state-trajectory plots, O_t histograms with the hand-set emission densities
and an offline 3-component GMM, Baum-Welch transition-matrix evolution, and
`occupancy_<method>.csv` (state occupancy, acceptance per state, rescues).

## 8. Adding SMAC later (Linux / WSL)

```
pip install "smac>=2.1"
python experiments_new/synt_functions/run.py --methods SMAC --workers 10 --fmp-config tuned
python experiments_new/lcbench/run.py        --methods SMAC --workers 10 --fmp-config tuned
```

Only the SMAC runs are executed; re-running the analysis scripts merges them into the
existing tables and figures.

## Reading the critical-difference diagram

The axis is the average rank over tasks (1 = best). Each method hangs from its rank.
A thick bar joins methods whose ranks differ by less than the critical difference
(Nemenyi test at alpha = 0.05), i.e. methods that are **not** significantly different.
With only 2 to 5 tasks the CD is larger than the whole axis and every method is joined;
the diagram is informative from about 10 tasks upward (19 synthetic configurations,
10 LCBench instances).
