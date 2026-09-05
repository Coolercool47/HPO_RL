"""E2 runner (LCBench). Examples:
    python experiments_new/lcbench/select_instances.py       # once
    python experiments_new/lcbench/run.py --dry-run
    python experiments_new/lcbench/run.py --smoke
    python experiments_new/lcbench/run.py --workers 10
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments_new.common.runner import cli  # noqa: E402
from experiments_new.lcbench import config as C  # noqa: E402

if __name__ == "__main__":
    cli("lcbench", C.TASKS, C.METHODS, C.SEEDS, C.BUDGET,
        description="E2: LCBench test instances, all methods, single fidelity (TPE_HB multi-fidelity reference)")
