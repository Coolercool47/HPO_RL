"""E3: single-factor ablation ladder (paper's FMP-only arm -> full DREAM method).

    L1_K1_VITERBI_SUB      (i)   one chain, Viterbi, EXPLORE coordinate subsampling, no orchestrator
    L2_K4_ORCH             (ii)  + 4 chains and orchestrator
    L3_NOSUB               (iii) - coordinate subsampling
    L4_SOFT                (iv)  soft forward-filter posterior mixing
    L5_DREAM               (v)   + symmetric DREAM(ZS) kernel (p_dream = 0.5)
    L5_DREAM_LEGACYKERNEL  (v) with the legacy asymmetric kernel and likelihood-only acceptance

n_chains is fixed by the ladder (1 / 4) even when a tuned shared config is used; all other
knobs come from --fmp-config. Tasks: 19 synthetic configurations + LCBench test instances,
10 seeds.
    python experiments_new/ablation_ladder/run.py --dry-run | --smoke | --workers 10
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments_new.common.runner import cli  # noqa: E402
from experiments_new.common.spaces import synthetic_specs  # noqa: E402
from experiments_new.lcbench import config as L  # noqa: E402

METHODS = ["L1_K1_VITERBI_SUB", "L2_K4_ORCH", "L3_NOSUB", "L4_SOFT", "L5_DREAM", "L5_DREAM_LEGACYKERNEL"]
TASKS = [synthetic_specs(10)[0], L.TASKS[0]] + synthetic_specs(10)[1:] + L.TASKS[1:]   # smoke uses the first two
SEEDS = list(range(10))

if __name__ == "__main__":
    cli("ablation_ladder", TASKS, METHODS, SEEDS, 500, description=__doc__,
        overrides=None)
