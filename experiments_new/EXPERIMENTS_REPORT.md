# Experiment inventory (state of 2026-09-19)

Counts are taken from the result files on disk (`*/results*/**/seed_*.json`), not from the
plan. Every set has 0 error files. Numbers and conclusions are in `RESULTS.md`; this file
only says what was run.

## Totals

| | runs | objective evaluations |
|---|---|---|
| all experiments | 22 760 | 8 561 820 |
| of which LCBench (YAHPO surrogate) | 12 520 (55 %) | 3 441 789 (40 %) |
| of which synthetic functions | 10 240 (45 %) | 5 120 031 (60 %) |

LCBench instances used: 10 test instances (7593, 189873, 168329, 167185, 167152, 168910,
167168, 189906, 168331, 168330) and 8 disjoint tuning instances (189905, 167201, 189354,
189866, 167200, 167184, 189909, 167161). An LCBench evaluation is one surrogate query at
52 epochs, except for TPE_HB, whose evaluations are partial-fidelity (budget counted in
epochs, 200 x 52).

## Per experiment

| ID | folder | design | runs | LCBench runs | budget |
|---|---|---|---|---|---|
| E0 held-out tuning | `tuning_heldout/` | 3 methods (FMP, TPE, CMA-ES) x 100 meta-trials x 8 tuning instances x 3 seeds | 7 200 | 7 200 | 200 |
| E2 LCBench, tuned | `lcbench/results` | 7 methods (RS, TPE, GP, CMAES, TPE_HB, FMP, FMP_DREAM) x 10 instances x 20 seeds | 1 400 | 1 400 | 200 |
| E2 LCBench, Table-5 | `lcbench/results_table5` | 7 methods (as above, with L1 in place of FMP) x 10 x 20 | 1 400 | 1 400 | 200 |
| E1 synthetic, tuned | `synt_functions/results` | 6 methods x 19 tasks (9 cont, 5 noisy, 5 cat; 10-D) x 20 seeds | 2 280 | 0 | 500 |
| E1 synthetic, Table-5 | `synt_functions/results_table5` | 7 methods (+ L1) x 19 x 20 | 2 660 | 0 | 500 |
| E3 ablation ladder | `ablation_ladder/` | 6 variants x (19 synthetic + 10 LCBench) x 10 seeds | 1 740 | 600 | 500 / 200 |
| E4 controller ablation | `ablation_controller/` | 7 controllers x 29 tasks x 10 seeds | 2 030 | 700 | 500 / 200 |
| E6 sensitivity | `sensitivity/` | one-at-a-time: 12 knobs x 5 levels + base = 61 configs x 5 tasks (3 synthetic, 2 LCBench) x 10 seeds = 3 050; random: 200 configs x 5 seeds on 10-D Rastrigin = 1 000 | 4 050 | 1 220 | 500 |
| E5 diagnostics | `diagnostics/` | no new runs; figures from E1/E2/E4 histories | – | – | – |
| E7 overhead | `*/figures*/overhead.csv` | no new runs; timing recorded in every run | – | – | – |

## Deviations from the plan

* Main results use a held-out-tuned configuration for FMP, TPE and CMA-ES (E0, user
  decision), with the Table-5 defaults kept as a second full run. The plan had fixed
  defaults only.
* Sensitivity covers 12 knobs, not 10.
* SMAC is not run. The runner works in a Linux container (`docker/Dockerfile.smac`,
  smoke-tested 2026-09-19: budget respected, seeds differ, 0.5 GB peak); see RUNBOOK section 8.
* Ablations and sensitivity use 10 seeds (as planned); main comparisons use 20.

## Known caveats

* About 1 % of multi-chain FMP runs finished 1-2 evaluations past the budget; the loader
  now scores every run on its first `budget` evaluations.
* Synthetic Table-5 FMP_DREAM and L1 arms were rerun on 2026-09-19 (see the correction
  note in `RESULTS.md` section 7).
* GP-BO results may differ in the last digits between machines (torch); all GP runs in
  the repository come from one machine per result set.
