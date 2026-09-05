"""E4: does the HMM controller infer anything useful? All variants share the
multi-chain FMP configuration (ladder step iv); only the regime controller differs.

    FMP_CTRL_HMM            3-state HMM, soft filter, Baum-Welch on A (hand-set emissions)
    FMP_CTRL_HMM_VITERBI    same with hard Viterbi decoding
    FMP_CTRL_HMM_NOBW       hand-set A, no online learning
    FMP_CTRL_HMM_LEARNEMIS  Baum-Welch also updates emission means / variances
    FMP_CTRL_FIXED          always EXPLOIT weights (TRAPPED only via the rejection counter)
    FMP_CTRL_RANDOM         uniformly random EXPLOIT / EXPLORE each step
    FMP_CTRL_RULE           threshold rule: EXPLOIT if mean(O_window) <= 0.05 else EXPLORE

    python experiments_new/ablation_controller/run.py --dry-run | --smoke | --workers 10
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments_new.common.runner import cli  # noqa: E402
from experiments_new.common.spaces import synthetic_specs  # noqa: E402
from experiments_new.lcbench import config as L  # noqa: E402

METHODS = ["FMP_CTRL_HMM", "FMP_CTRL_HMM_VITERBI", "FMP_CTRL_HMM_NOBW", "FMP_CTRL_HMM_LEARNEMIS",
           "FMP_CTRL_FIXED", "FMP_CTRL_RANDOM", "FMP_CTRL_RULE"]
TASKS = [synthetic_specs(10)[0], L.TASKS[0]] + synthetic_specs(10)[1:] + L.TASKS[1:]   # smoke uses the first two
SEEDS = list(range(10))

if __name__ == "__main__":
    cli("ablation_controller", TASKS, METHODS, SEEDS, 500, description=__doc__)
