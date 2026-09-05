"""E3: single-factor ablation ladder (paper's FMP-only arm -> full DREAM method).

    FMP            (i)   one chain, Viterbi, EXPLORE coordinate subsampling, no orchestrator
    FMP_MC         (ii)  + 4 chains and orchestrator
    FMP_MC_NOSUB   (iii) - coordinate subsampling
    FMP_SOFT       (iv)  soft forward-filter posterior mixing
    FMP_DREAM      (v)   + symmetric DREAM(ZS) kernel (p_dream = 0.5)
    FMP_DREAM_LEGACYKERNEL   (v) with the legacy asymmetric kernel and likelihood-only acceptance

Tasks: the 19 synthetic configurations + the LCBench test instances. 10 seeds.
    python experiments_new/ablation_ladder/run.py --dry-run | --smoke | --workers 10
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments_new.common.runner import cli  # noqa: E402
from experiments_new.common.spaces import synthetic_specs  # noqa: E402
from experiments_new.lcbench import config as L  # noqa: E402

METHODS = ["FMP", "FMP_MC", "FMP_MC_NOSUB", "FMP_SOFT", "FMP_DREAM", "FMP_DREAM_LEGACYKERNEL"]
TASKS = [synthetic_specs(10)[0], L.TASKS[0]] + synthetic_specs(10)[1:] + L.TASKS[1:]   # smoke uses the first two
SEEDS = list(range(10))

if __name__ == "__main__":
    cli("ablation_ladder", TASKS, METHODS, SEEDS, 500, description=__doc__,
        overrides=None)
