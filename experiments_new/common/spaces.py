"""Synthetic benchmark tasks with correct global optima and the hpo_rl space format.

A *task spec* is a plain, picklable dict so it can be shipped to worker processes:
    {"kind": "synthetic", "function": "rastrigin", "dims": 10, "noise_std": 0.0, "categorical": False}
    {"kind": "lcbench", "instance": "7593"}
`build_task(spec)` turns it into a `Task` with callables.
"""

from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Known optima (minimization). Michalewicz depends on d (m = 10):
# d=2: -1.8013, d=5: -4.687658, d=10: -9.66015 (Molga & Smutnicki 2005).
# ---------------------------------------------------------------------------

_MICHALEWICZ_OPT = {1: -0.8013, 2: -1.8013, 5: -4.687658, 10: -9.66015}


def global_optimum(function: str, dims: int) -> float:
    if function == "styblinski_tang":
        return -39.16616570377142 * dims
    if function == "michalewicz":
        if dims not in _MICHALEWICZ_OPT:
            raise ValueError(f"Michalewicz optimum unknown for d={dims}; add it to _MICHALEWICZ_OPT")
        return _MICHALEWICZ_OPT[dims]
    return 0.0


SYNTH_BOUNDS = {
    "sphere": (-5.0, 5.0),
    "rosenbrock": (-1.0, 1.0),
    "rastrigin": (-5.12, 5.12),
    "ackley": (-32.768, 32.768),
    "griewank": (-600.0, 600.0),
    "schwefel": (-500.0, 500.0),
    "levy": (-10.0, 10.0),
    "michalewicz": (0.0, math.pi),
    "styblinski_tang": (-5.0, 5.0),
}

CONTINUOUS_FUNCTIONS = tuple(SYNTH_BOUNDS.keys())
NOISY_FUNCTIONS = {"sphere": 0.5, "rastrigin": 8.0, "ackley": 2.0, "schwefel": 300.0, "levy": 1.5}
CATEGORICAL_FUNCTIONS = ("sphere", "rastrigin", "ackley", "schwefel", "levy")

DUMMY_CATEGORIES = {
    "optimizer": {"values": ["adam", "sgd", "rmsprop", "adamw"], "type": "categorical"},
    "activation": {"values": ["relu", "tanh", "gelu", "silu"], "type": "categorical"},
    "scheduler": {"values": ["cosine", "step", "plateau"], "type": "categorical"},
}


def synthetic_specs(dims: int = 10) -> list[dict]:
    """The 19 configurations of the paper: 9 continuous, 5 noisy, 5 categorical."""
    specs = [dict(kind="synthetic", function=f, dims=dims, noise_std=0.0, categorical=False) for f in CONTINUOUS_FUNCTIONS]
    specs += [dict(kind="synthetic", function=f, dims=dims, noise_std=s, categorical=False) for f, s in NOISY_FUNCTIONS.items()]
    specs += [dict(kind="synthetic", function=f, dims=dims, noise_std=0.0, categorical=True) for f in CATEGORICAL_FUNCTIONS]
    return specs


def spec_key(spec: dict) -> str:
    if spec["kind"] == "synthetic":
        suite = "cat" if spec["categorical"] else ("noisy" if spec["noise_std"] > 0 else "cont")
        return f"{suite}__{spec['function']}_{spec['dims']}d"
    if spec["kind"] == "lcbench":
        return f"lcbench_{spec['instance']}"
    raise ValueError(spec)


def spec_suite(spec: dict) -> str:
    if spec["kind"] == "synthetic":
        return "cat" if spec["categorical"] else ("noisy" if spec["noise_std"] > 0 else "cont")
    return spec["kind"]


@dataclass
class Task:
    key: str
    spec: dict
    space: dict                                  # hpo_rl format {name: {type, values, log}}
    objective: Callable[[dict], float]           # minimize
    f_star: float | None = None                  # known optimum of the *noise-free* objective
    true_objective: Callable[[dict], float] | None = None  # noise-free twin (synthetic noisy)
    maximize_raw: bool = False                   # reporting: raw metric is maximized (accuracy)
    raw_of_loss: Callable[[float], float] = field(default=lambda v: v)
    eval_cost: float = 1.0                       # cost units per full evaluation (LCBench: epochs)
    extra: dict = field(default_factory=dict)


def make_space(function: str, dims: int, categorical: bool) -> dict:
    lo, hi = SYNTH_BOUNDS[function]
    space = {f"x{i}": {"values": [float(lo), float(hi)], "type": "float", "log": False} for i in range(dims)}
    if categorical:
        space.update({k: dict(v) for k, v in DUMMY_CATEGORIES.items()})
    return space


def _make_backend(function: str, dims: int, noise_std: float, seed: int):
    from hpo_rl.backends.function import OptimizationBenchmarkBackend

    # the backend draws its noise instance seed from the legacy global RNG -> pin it
    state = np.random.get_state()
    np.random.seed(seed % (2**32 - 1))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            be = OptimizationBenchmarkBackend(function_name=function, dimensions=dims, noise_std=noise_std, use_cache=False)
    finally:
        np.random.set_state(state)
    return be


def build_synthetic_task(spec: dict, seed: int = 0) -> Task:
    function, dims = spec["function"], int(spec["dims"])
    noise_std = float(spec.get("noise_std", 0.0))
    categorical = bool(spec.get("categorical", False))
    be = _make_backend(function, dims, noise_std, seed)
    true_be = _make_backend(function, dims, 0.0, seed) if noise_std > 0 else None

    def objective(cfg: dict) -> float:
        return float(be.evaluate(cfg))

    true_objective = (lambda cfg: float(true_be.evaluate(cfg))) if true_be is not None else None
    return Task(
        key=spec_key(spec), spec=spec, space=make_space(function, dims, categorical),
        objective=objective, f_star=global_optimum(function, dims), true_objective=true_objective,
        maximize_raw=False, eval_cost=1.0,
    )


def build_task(spec: dict, seed: int = 0) -> Task:
    if spec["kind"] == "synthetic":
        return build_synthetic_task(spec, seed)
    if spec["kind"] == "lcbench":
        from experiments_new.common.yahpo import build_lcbench_task

        return build_lcbench_task(spec)
    raise ValueError(f"unknown task kind {spec['kind']!r}")
