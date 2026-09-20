"""E8: second real-data benchmark with consequential categorical / conditional hyperparameters.

YAHPO Gym scenarios rbv2_svm (kernel in {linear, polynomial, radial}; degree and gamma
conditional on it) and rbv2_xgboost (booster in {gblinear, gbtree, dart}; 8 conditional
hyperparameters): surrogates of real SVM / XGBoost training on OpenML tasks.
Protocol as for LCBench: B = 200 full-fidelity evaluations (trainsize = 1, repl = 10),
20 seeds, objective = validation accuracy. 20 instances per scenario, drawn once with a
fixed seed from the scenario's instance list (configs/rbv2_instances.json). No method was
tuned on these scenarios: FMP / TPE / CMA-ES use the configuration tuned on the 8 held-out
LCBench instances, so this is an out-of-benchmark generalisation test for all of them.
"""

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "rbv2_instances.json"
SCENARIOS = ["rbv2_svm", "rbv2_xgboost"]
N_INSTANCES = 20
BUDGET = 200
SEEDS = list(range(20))
METHODS = ["RS", "TPE", "GP", "CMAES", "FMP", "FMP_DREAM"]


def load_instances() -> dict:
    if CONFIG.is_file():
        return json.loads(CONFIG.read_text(encoding="utf-8"))["instances"]
    import numpy as np

    from experiments_new.common.yahpo import rbv2_instances

    rng = np.random.default_rng(2026)
    inst = {sc: sorted(rng.choice(sorted(rbv2_instances(sc), key=int), N_INSTANCES, replace=False).tolist(), key=int)
            for sc in SCENARIOS}
    CONFIG.write_text(json.dumps({"rule": "numpy default_rng(2026).choice over the sorted instance list, 20 per scenario, no selection",
                                  "instances": inst}, indent=2), encoding="utf-8")
    return inst


INSTANCES = load_instances()
TASKS = [dict(kind="rbv2", scenario=sc, instance=i, budget=BUDGET) for sc in SCENARIOS for i in INSTANCES[sc]]
