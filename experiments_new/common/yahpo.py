"""LCBench tasks through the YAHPO Gym surrogate.

Single-fidelity protocol (paper): every evaluation trains for the full 52 epochs
and costs 52 fidelity units; `Task.eval_cost = 52`. The multi-fidelity helper
`evaluate_at_epoch` is used only by the BOHB-style TPE+Hyperband baseline.
"""

from __future__ import annotations

import functools
import os
import warnings
from pathlib import Path

from experiments_new.common.spaces import Task, spec_key

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.environ.get("YAHPO_DATA_PATH", REPO_ROOT / "yahpo_data"))

LCBENCH_TARGET = "val_accuracy"
LCBENCH_FIDELITY = "epoch"

ALL_LCBENCH_INSTANCES = [
    "3945", "7593", "34539", "126025", "126026", "126029", "146212", "167104", "167149", "167152",
    "167161", "167168", "167181", "167184", "167185", "167190", "167200", "167201", "168329", "168330",
    "168331", "168335", "168868", "168908", "168910", "189354", "189862", "189865", "189866", "189873",
    "189905", "189906", "189908", "189909",
]


def _init_yahpo():
    warnings.filterwarnings("ignore", message=".*default.*should be.*default_value.*")
    import ConfigSpace as CS

    # yahpo-gym 1.0.x calls ConfigurationSpace._sort_hyperparameters (removed in ConfigSpace >= 1.2)
    if not hasattr(CS.ConfigurationSpace, "_sort_hyperparameters"):
        CS.ConfigurationSpace._sort_hyperparameters = lambda self: None
    from yahpo_gym import local_config

    # Set the data path in-process only. `set_data_path` rewrites the user's settings
    # file, which races when many worker processes start at once (a partially written
    # YAML loads as None -> AttributeError on .update).
    local_config._config = {"data_path": str(DATA_PATH)}
    if not (DATA_PATH / "lcbench" / "encoding.json").is_file():
        raise FileNotFoundError(
            f"YAHPO lcbench data not found under {DATA_PATH}. Clone https://github.com/slds-lmu/yahpo_data "
            "(sparse checkout of 'lcbench' is enough) or set YAHPO_DATA_PATH."
        )


@functools.lru_cache(maxsize=None)
def _benchmark(instance: str):
    _init_yahpo()
    from yahpo_gym import benchmark_set

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bench = benchmark_set.BenchmarkSet("lcbench")
        bench.set_instance(str(instance))
    bench.check = False
    return bench


def lcbench_space(instance: str) -> tuple[dict, float, float]:
    """Return (hpo_rl space, min_epoch, max_epoch) for an instance."""
    import ConfigSpace.hyperparameters as CSH

    bench = _benchmark(instance)
    cs = bench.get_opt_space(drop_fidelity_params=True)
    space: dict = {}
    for hp in list(cs.values()):
        if isinstance(hp, CSH.Constant):
            continue
        if isinstance(hp, CSH.UniformFloatHyperparameter):
            space[hp.name] = {"type": "float", "values": [float(hp.lower), float(hp.upper)], "log": bool(hp.log)}
        elif isinstance(hp, CSH.UniformIntegerHyperparameter):
            space[hp.name] = {"type": "int", "values": [int(hp.lower), int(hp.upper)], "log": bool(hp.log)}
        elif isinstance(hp, CSH.CategoricalHyperparameter):
            space[hp.name] = {"type": "categorical", "values": list(hp.choices)}
        elif isinstance(hp, CSH.OrdinalHyperparameter):
            space[hp.name] = {"type": "categorical", "values": list(hp.sequence)}
        else:
            raise TypeError(f"unsupported hyperparameter {hp}")
    f_hp = bench.get_fidelity_space()[LCBENCH_FIDELITY]
    return space, float(f_hp.lower), float(f_hp.upper)


def evaluate_at_epoch(instance: str, cfg: dict, epoch: int, target: str = LCBENCH_TARGET) -> float:
    bench = _benchmark(instance)
    q = dict(cfg)
    q["OpenML_task_id"] = str(instance)
    q[LCBENCH_FIDELITY] = int(round(epoch))
    return float(bench.objective_function(q)[0][target])


def build_lcbench_task(spec: dict) -> Task:
    instance = str(spec["instance"])
    space, min_epoch, max_epoch = lcbench_space(instance)
    max_epoch_i = int(round(max_epoch))

    def objective(cfg: dict) -> float:  # minimize negative accuracy
        return -evaluate_at_epoch(instance, cfg, max_epoch_i)

    return Task(
        key=spec_key(spec), spec=spec, space=space, objective=objective, f_star=None,
        true_objective=None, maximize_raw=True, raw_of_loss=lambda v: -v, eval_cost=float(max_epoch_i),
        extra={"instance": instance, "min_epoch": int(round(min_epoch)), "max_epoch": max_epoch_i},
    )
