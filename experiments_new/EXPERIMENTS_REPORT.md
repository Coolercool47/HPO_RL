# Experiment inventory (final state of 2026-09-21)

Counts are taken from the result files on disk, not from the plan. Every finished set has
0 error files. Numbers and conclusions are in `RESULTS.md`; this file says what was run.
LCBench (YAHPO surrogate of real neural-network training runs on OpenML datasets) is the
primary benchmark; synthetic functions are the secondary robustness check.

## Totals (finished runs)

| | runs | share |
|---|---|---|
| all experiments | 60 370 | |
| LCBench | 37 700 | 62 % |
| rbv2 SVM / XGBoost (real data, categorical + conditional) | 5 600 | 9 % |
| synthetic functions | 17 070 | 28 % |

All planned experiments are complete, including SMAC on LCBench and rbv2 (SMAC on the synthetic
functions was not run: lowest priority under plan revision 2, about 7.5 h).

LCBench instances: 34 exist. 8 are reserved for tuning (189905, 167201, 189354, 189866,
167200, 167184, 189909, 167161). All other 26 form the test set: the 10 originally selected
(7593, 189873, 168329, 167185, 167152, 168910, 167168, 189906, 168331, 168330) plus the 16
added on 2026-09-19 (3945, 34539, 126025, 126026, 126029, 146212, 167104, 167149, 167181,
167190, 168335, 168868, 168908, 189862, 189865, 189908). No test instance is selected any more.
An LCBench evaluation is one surrogate query at 52 epochs, except for TPE_HB (partial
fidelity, budget counted in epochs, 200 x 52).

## Per experiment

| ID | folder | design | runs | LCBench runs | LCBench instances | budget |
|---|---|---|---|---|---|---|
| E0 held-out tuning | `tuning_heldout/` | 3 methods (FMP, TPE, CMA-ES) x 100 meta-trials x 8 tuning instances x 3 seeds | 7 200 | 7 200 | 8 (tuning) | 200 |
| E2 LCBench, tuned | `lcbench/results` | 8 methods (RS, TPE, GP, SMAC, CMAES, TPE_HB, FMP, FMP_DREAM) x 26 instances x 20 seeds | 4 160 | 4 160 | 26 | 200 |
| E2 LCBench, Table-5 | `lcbench/results_table5` | 7 methods (L1 in place of FMP) x 26 instances x 20 seeds | 3 640 | 3 640 | 26 | 200 |
| E8 rbv2 (SVM, XGBoost) | `rbv2/results` | 7 methods (RS, TPE, GP, SMAC, CMAES, FMP, FMP_DREAM) x 40 instances (20 per scenario) x 20 seeds; nothing tuned on it | 5 600 | – | 40 rbv2 | 200 |
| E1 synthetic, tuned | `synt_functions/results` | 6 methods x 19 tasks (9 cont, 5 noisy, 5 cat; 10-D) x 20 seeds | 2 280 | 0 | – | 500 |
| E1 synthetic, Table-5 | `synt_functions/results_table5` | 7 methods (+ L1) x 19 x 20 | 2 660 | 0 | – | 500 |
| E3 ablation ladder | `ablation_ladder/` | 6 variants x (26 LCBench + 19 synthetic) x 10 seeds | 2 700 | 1 560 | 26 | 200 / 500 |
| E4 controller ablation | `ablation_controller/` | 7 controllers x 45 tasks x 10 seeds | 3 150 | 1 820 | 26 | 200 / 500 |
| E6 sensitivity, Table-5 centred | `sensitivity/results` | one-at-a-time: 61 configs x 9 tasks (6 LCBench + 3 synthetic) x 10 seeds = 5 490; random: 200 configs x 9 tasks x 5 seeds = 9 000 | 14 490 | 9 660 | 6 | 200 / 500 |
| E6 sensitivity, tuned centred | `sensitivity/results_tuned` | same design around the tuned configuration (levels x0.25 ... x4) | 14 490 | 9 660 | 6 | 200 / 500 |
| E5 diagnostics | `diagnostics/` | no new runs; figures from E1/E2/E4 histories | – | – | – | – |
| E7 overhead | `*/figures*/overhead.csv` | no new runs; timing recorded in every run | – | – | – | – |
| SMAC timing | `smac_timing/` | 4 LCBench + 4 synthetic full-budget runs in Docker | 8 | 4 | 1 | 200 / 500 |

Sensitivity LCBench instances: 7593, 168329, 167185 (original set), 126026, 167181, 189908 (added).

## What is stored where

* Raw per-run JSON for every experiment is in git, except the two sensitivity studies
  (2.8 GB): their raw files stay on the machine that ran them; git holds
  `sensitivity/figures*/sensitivity_runs.csv` (every per-run scalar the analysis uses),
  the summaries and the figures.
* `smac_timing/` is kept outside the result sets so that 8 runs do not enter the tables.

## Deviations from the plan

* Main results use a held-out-tuned configuration for FMP, TPE and CMA-ES (E0, user
  decision), with the Table-5 defaults kept as a second full run. The plan had fixed defaults only.
* LCBench test set is 26 instances (plan: 10).
* Sensitivity: 12 knobs (plan: 10), two centres, LCBench-first; fANOVA on all 9 tasks
  (first version: Rastrigin only).
* SMAC runs in a Linux container (`docker/Dockerfile.smac`): done on LCBench (520 runs, 2.7 h) and
  rbv2 (800 runs, 4.5 h) with 6 concurrent jobs; not run on the synthetic functions.
* E8 (rbv2) was added by plan revision 2 because LCBench has no categorical hyperparameters.
* Ablations and sensitivity use 10 seeds (as planned); main comparisons use 20.

## Corrections made after the first digest

* 2026-09-19: synthetic Table-5 FMP_DREAM arm had run with DREAM off on 18/19 tasks
  (p_dream default), L1 arm existed for one task; both rerun.
* 2026-09-19: scoring capped at the budget (about 1 % of multi-chain FMP runs used 1-2
  evaluations more).
* 2026-09-20: sensitivity redone. The first version used 2 LCBench instances at 500
  instead of 200 evaluations, ran fANOVA on one synthetic function, averaged two different
  units into one influence number, and the digest misread the sign of the LCBench score.

## Known caveats

* GP-BO results may differ in the last digits between machines (torch); all GP runs in a
  result set come from one machine.
* HMM diagnostics figures (E5) cover all 26 LCBench instances (regenerated 2026-09-20).
* 2026-09-20: the first E8 run was discarded (instance id not applied by the surrogate wrapper); fixed and rerun.
