"""E1 runner. Examples:
    python experiments_new/synt_functions/run.py --dry-run
    python experiments_new/synt_functions/run.py --smoke
    python experiments_new/synt_functions/run.py --workers 10
    python experiments_new/synt_functions/run.py --methods GP --n-seeds 10      # add / extend later
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments_new.common.runner import cli  # noqa: E402
from experiments_new.synt_functions import config as C  # noqa: E402

if __name__ == "__main__":
    cli("synt_functions", C.TASKS, C.METHODS, C.SEEDS, C.BUDGET,
        description="E1: synthetic benchmarks, all methods, fixed configuration")
