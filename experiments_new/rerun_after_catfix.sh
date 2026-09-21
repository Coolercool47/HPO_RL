#!/usr/bin/env bash
# Re-tune H-MCMC-FMP and rerun every experiment that contains FMP-type runs, after the fix of the
# categorical-only step (it re-proposed the current configuration on spaces without categorical
# coordinates and wasted up to 30 % of the budget). Baseline runs (RS, TPE, TPE_HB, CMAES, GP, SMAC)
# are untouched. Run from the repository root:  bash experiments_new/rerun_after_catfix.sh
set -u
PY=.venv/Scripts/python.exe
E=experiments_new
W=8
step() { echo; echo "=== $(date '+%F %T') $*"; }

step "1. re-tune FMP on the 8 held-out LCBench tasks"
if [ -d $E/tuning_heldout/results/FMP ] && [ ! -d $E/tuning_heldout/results_before_catfix/FMP ]; then
  mkdir -p $E/tuning_heldout/results_before_catfix && mv $E/tuning_heldout/results/FMP $E/tuning_heldout/results_before_catfix/FMP
  cp $E/configs/fmp_tuned.json $E/configs/fmp_tuned_before_catfix.json
fi
$PY $E/tuning_heldout/run.py --method FMP --n-trials 100 --workers $W || exit 1

step "2. remove FMP-type results"
for d in $E/lcbench/results $E/lcbench/results_table5 $E/rbv2/results $E/synt_functions/results $E/synt_functions/results_table5; do
  for m in FMP FMP_DREAM L1_K1_VITERBI_SUB; do rm -rf $d/*/$m; done
done
rm -rf $E/ablation_ladder/results $E/ablation_controller/results $E/sensitivity/results $E/sensitivity/results_tuned

step "3. main comparisons"
$PY $E/lcbench/run.py        --workers $W --fmp-config tuned --baseline-config tuned --methods FMP FMP_DREAM
$PY $E/lcbench/run.py        --workers $W --out $E/lcbench/results_table5 --methods FMP_DREAM L1_K1_VITERBI_SUB
$PY $E/rbv2/run.py           --workers $W --fmp-config tuned --baseline-config tuned --methods FMP FMP_DREAM
$PY $E/synt_functions/run.py --workers $W --fmp-config tuned --baseline-config tuned --methods FMP FMP_DREAM
$PY $E/synt_functions/run.py --workers $W --out $E/synt_functions/results_table5 --methods FMP FMP_DREAM L1_K1_VITERBI_SUB

step "4. ablations"
$PY $E/ablation_ladder/run.py     --workers $W --fmp-config tuned
$PY $E/ablation_controller/run.py --workers $W --fmp-config tuned

step "5. analyses of everything above"
$PY $E/lcbench/analyze.py; $PY $E/lcbench/analyze.py --results $E/lcbench/results_table5
$PY $E/rbv2/analyze.py
$PY $E/synt_functions/analyze.py; $PY $E/synt_functions/analyze.py --results $E/synt_functions/results_table5
$PY $E/ablation_ladder/analyze.py; $PY $E/ablation_controller/analyze.py
$PY $E/common/anytime.py $E/lcbench/results; $PY $E/common/anytime.py $E/rbv2/results
$PY $E/diagnostics/run.py --results $E/lcbench/results --method FMP_DREAM --seeds 0
$PY $E/diagnostics/run.py --results $E/ablation_controller/results --method CTRL_HMM --seeds 0

step "6. sensitivity (longest)"
$PY $E/sensitivity/run.py --workers $W
$PY $E/sensitivity/run.py --workers $W --fmp-config tuned --out $E/sensitivity/results_tuned
$PY $E/sensitivity/analyze.py
$PY $E/sensitivity/analyze.py --results $E/sensitivity/results_tuned --figs $E/sensitivity/figures_tuned

step "done"
