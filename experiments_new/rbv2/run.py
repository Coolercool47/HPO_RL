"""E8 runner (rbv2_svm, rbv2_xgboost). Examples:
    python experiments_new/rbv2/run.py --dry-run
    python experiments_new/rbv2/run.py --workers 8 --heavy-gb 1.0 --fmp-config tuned --baseline-config tuned
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments_new.common.runner import cli  # noqa: E402
from experiments_new.rbv2 import config as C  # noqa: E402

if __name__ == "__main__":
    cli("rbv2", C.TASKS, C.METHODS, C.SEEDS, C.BUDGET,
        description="E8: real-data HPO with categorical/conditional hyperparameters (YAHPO rbv2_svm, rbv2_xgboost)")
