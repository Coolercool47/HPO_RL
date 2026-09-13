# Next runs (after the first full run of 2026-09-13)

State of the results after the first run and the fixes that followed it:

| experiment | status | what to do |
|---|---|---|
| E6 sensitivity | complete (4050 runs), analysed | nothing |
| E0 held-out tuning | complete (FMP, TPE, CMA-ES; 100 trials each) | nothing; `configs/fmp_tuned.json` regenerated with `n_chains` and `p_dream` |
| E2 LCBench, Table-5 config (`results_table5/`) | complete (7 methods x 10 tasks x 20 seeds) | nothing; the old `FMP` arm was renamed `L1_K1_VITERBI_SUB` in place |
| E2 LCBench, tuned config (`results/`) | baselines complete, **FMP / FMP_DREAM missing** (crashed on `_meta`) | step 2 |
| E1 synthetic, both configs | barely started (1 to 2 tasks) | steps 5 and 6 |
| E3 ladder, E4 controller | not started | steps 3 and 4 |
| E5 diagnostics | not started (needs E2 / E4 results) | step 7 |

All commands run from the code repo root with the project venv. Every runner is
resumable: a re-run skips finished runs and only fills the gaps, so an interrupted
command is simply repeated. Logs go to `experiments_new/<exp>/logs/`, and every
invocation appends its outcome to `<results>/_run_summary.json`.

## 1. Pull and check (2 min)

```
git pull
python experiments_new/lcbench/run.py --dry-run --fmp-config tuned --baseline-config tuned
python experiments_new/ablation_controller/run.py --dry-run --fmp-config tuned
```

Both must end with `[dry-run] all checks passed`. The first line of the dry-run output
prints the tuned FMP configuration; it must contain `n_chains` and `p_dream` and no
`_meta`.

Memory: budget 0.5 GB per worker. Without `--workers` the runner uses CPU count minus
2 capped by available RAM (needs `psutil`, in `requirements.txt`). On the 16 GB machine
use `--workers 8` for anything that includes GP-BO and up to `--workers 12` otherwise.

## 2. Fill the tuned LCBench FMP runs (~5 min)

```
python experiments_new/lcbench/run.py --workers 12 --fmp-config tuned --baseline-config tuned
python experiments_new/lcbench/analyze.py
```

Expected: `[runner] 1400 jobs, 1000 done, 400 to run`. The analysis then covers all
seven methods with the tuned configuration; compare against
`lcbench/figures_table5/` (untuned).

## 3. E4 controller ablation (~15 min) — run this before anything else

```
python experiments_new/ablation_controller/run.py --workers 12 --fmp-config tuned
python experiments_new/ablation_controller/analyze.py
```

2030 runs (7 controllers x 29 tasks x 10 seeds). Reference method `CTRL_HMM`. This
decides how the paper is framed: if `CTRL_RULE` or `CTRL_RANDOM` are not significantly
worse than `CTRL_HMM` (see `wins_vs_CTRL_HMM.csv` and `cross_task.json`), the HMM
controller is not supported by the data and the method is presented as an annealed
adaptive Metropolis search with an interchangeable regime controller.

## 4. E3 ablation ladder (~15 min)

```
python experiments_new/ablation_ladder/run.py --workers 12 --fmp-config tuned
python experiments_new/ablation_ladder/analyze.py
```

1740 runs, reference `L1_K1_VITERBI_SUB` (the submitted paper's FMP-only arm). Steps
L2 to L5 fix `n_chains = 4` by design even though the tuned config says 1; every other
knob comes from the tuned config. Decision gate: if `L5_DREAM` is not significantly
better than the best of L1 to L4 on at least one suite, DREAM becomes an appendix
extension.

## 5. E1 synthetic, tuned configuration (~2 h, GP-BO dominates)

```
python experiments_new/synt_functions/run.py --workers 8 --fmp-config tuned --baseline-config tuned --methods RS TPE CMAES FMP FMP_DREAM
python experiments_new/synt_functions/run.py --workers 8 --fmp-config tuned --baseline-config tuned --methods GP
python experiments_new/synt_functions/analyze.py
```

Splitting GP off gives usable tables after the first command (~10 min); the second
adds GP (about 150 s per run, 380 runs). The analysis merges whatever is present.

## 6. E1 synthetic, Table-5 reference (~2 h, can run on a second machine)

```
python experiments_new/synt_functions/run.py --workers 8 --out experiments_new/synt_functions/results_table5 --methods RS TPE CMAES FMP FMP_DREAM
python experiments_new/synt_functions/run.py --workers 8 --out experiments_new/synt_functions/results_table5 --methods GP
python experiments_new/synt_functions/analyze.py --results experiments_new/synt_functions/results_table5
```

The existing `results_table5/` already holds the first task; the runner continues
from there. Note that its `FMP` directory was renamed to `L1_K1_VITERBI_SUB`, so the
new `FMP` (soft filter, orchestrator, no DREAM) will be run fresh.

## 7. E5 diagnostics (1 min, no new runs)

```
python experiments_new/diagnostics/run.py --results experiments_new/lcbench/results --method FMP_DREAM --seeds 0
python experiments_new/diagnostics/run.py --results experiments_new/synt_functions/results --method FMP_DREAM --seeds 0
python experiments_new/diagnostics/run.py --results experiments_new/ablation_controller/results --method CTRL_HMM --seeds 0
```

Figures in `diagnostics/figures/<results-name>/`: state trajectories, observation
histograms against the hand-set emissions and an offline GMM, Baum-Welch transition
evolution, occupancy tables.

## 8. Optional: SMAC (Linux only, after everything else)

```
python experiments_new/synt_functions/run.py --workers 8 --fmp-config tuned --baseline-config tuned --methods SMAC
python experiments_new/lcbench/run.py        --workers 8 --fmp-config tuned --baseline-config tuned --methods SMAC
```

Re-run the two `analyze.py` scripts afterwards; SMAC is merged into the existing tables.

## Running on two machines

The runner only touches `<results>/<task>/<method>/seed_XXX.json`, so the grid can be
split by method or by task between machines and merged by copying the result
directories together:

```
# machine A
python experiments_new/synt_functions/run.py --workers 8 --fmp-config tuned --baseline-config tuned --tasks cont__
# machine B
python experiments_new/synt_functions/run.py --workers 8 --fmp-config tuned --baseline-config tuned --tasks noisy__ cat__
```

`--tasks` takes substrings of task keys (`cont__`, `noisy__`, `cat__`, `lcbench_7593`, ...).
Copy `experiments_new/*/results*/` back and run the analysis once.

## What to send back

`experiments_new/*/results*/` (per-run JSON), `experiments_new/*/figures*/`,
`experiments_new/*/logs/` and every `_run_summary.json`. The figures are regenerated
from the JSON files, so the results directories are the only thing that cannot be
recreated.
