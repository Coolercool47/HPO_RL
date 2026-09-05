"""E2: LCBench (YAHPO surrogate), single fidelity (52 epochs), B=200 evaluations, 20 seeds.
Instances come from configs/lcbench_instances.json (run select_instances.py once);
until then the paper's 5 instances are used."""

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "lcbench_instances.json"
PAPER_INSTANCES = ["7593", "189873", "168329", "167185", "167152"]


def load_instances() -> tuple[list[str], list[str]]:
    if CONFIG.is_file():
        with open(CONFIG, encoding="utf-8") as fh:
            d = json.load(fh)
        return list(d["test"]), list(d["tune"])
    return list(PAPER_INSTANCES), []


TEST_INSTANCES, TUNE_INSTANCES = load_instances()
BUDGET = 200
SEEDS = list(range(20))
TASKS = [dict(kind="lcbench", instance=i, budget=BUDGET) for i in TEST_INSTANCES]
TUNE_TASKS = [dict(kind="lcbench", instance=i) for i in TUNE_INSTANCES]
METHODS = ["RS", "TPE", "GP", "CMAES", "TPE_HB", "FMP", "FMP_DREAM"]
