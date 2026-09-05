"""Method registry and single-run executors.

All methods minimize `task.objective`. A run returns a plain dict (JSON-serialisable)
with the full evaluation trace so that any table/figure can be rebuilt without
re-running:

    method, task, suite, seed, budget, params
    values        loss per evaluation (minimize orientation), evaluation order
    raw           raw metric (accuracy for LCBench, f for synthetic)
    true_values   noise-free f per evaluation (synthetic noisy tasks) or None
    costs         cumulative cost after each evaluation (1 per eval; epochs for LCBench)
    fidelity      fidelity of each evaluation (epochs; only differs for TPE_HB)
    configs       evaluated configurations (user space)
    best_value, best_config, best_true_value, best_raw
    time_total, time_objective, n_evals
    fmp           diagnostics for FMP variants (summary, history, A_history, emission_history)

Method specs
------------
    {"kind": "optuna", "sampler": "random"|"tpe"|"gp"|"cmaes", "pruner": None|"hyperband"}
    {"kind": "fmp", "params": {...HMM_MCMC_FMP kwargs...}}
    {"kind": "smac"}
"""

from __future__ import annotations

import copy
import math
import time
import warnings
from typing import Any

import numpy as np

from experiments_new.common.spaces import Task, spec_suite

# ---------------------------------------------------------------------------
# FMP variant presets (single-factor ablation ladder + controller ablation)
# ---------------------------------------------------------------------------

# (i)  FMP-only as in the submitted paper's "FMP-only" arm: one chain, Viterbi,
#      EXPLORE coordinate subsampling, no orchestration.
FMP_STEP1 = dict(p_dream=0.0, decoder="viterbi", explore_subsample=True, n_chains=1, orchestrate_every=0)
# (ii) + parallel chains and orchestrator
FMP_STEP2 = dict(p_dream=0.0, decoder="viterbi", explore_subsample=True, n_chains=4, orchestrate_every=5)
# (iii) - EXPLORE coordinate subsampling (DREAM class never had it)
FMP_STEP3 = dict(p_dream=0.0, decoder="viterbi", explore_subsample=False, n_chains=4, orchestrate_every=5)
# (iv) soft forward-filter posterior mixing instead of hard Viterbi
FMP_STEP4 = dict(p_dream=0.0, decoder="soft", explore_subsample=False, n_chains=4, orchestrate_every=5)
# (v)  + DREAM(ZS) crossover kernel (symmetric version)
FMP_STEP5 = dict(p_dream=0.5, decoder="soft", explore_subsample=False, n_chains=4, orchestrate_every=5)

FMP_VARIANTS: dict[str, dict] = {
    "FMP": FMP_STEP1,
    "FMP_MC": FMP_STEP2,
    "FMP_MC_NOSUB": FMP_STEP3,
    "FMP_SOFT": FMP_STEP4,
    "FMP_DREAM": FMP_STEP5,
    "FMP_DREAM_LEGACYKERNEL": {**FMP_STEP5, "dream_symmetric": False},
    # controller ablation (on top of the multi-chain FMP configuration, step iv)
    "FMP_CTRL_HMM": FMP_STEP4,
    "FMP_CTRL_HMM_VITERBI": FMP_STEP3,
    "FMP_CTRL_HMM_NOBW": {**FMP_STEP4, "learn_transitions": False},
    "FMP_CTRL_HMM_LEARNEMIS": {**FMP_STEP4, "learn_emissions": True},
    "FMP_CTRL_FIXED": {**FMP_STEP4, "controller": "fixed"},
    "FMP_CTRL_RANDOM": {**FMP_STEP4, "controller": "random"},
    "FMP_CTRL_RULE": {**FMP_STEP4, "controller": "rule"},
}

BASELINES: dict[str, dict] = {
    "RS": {"kind": "optuna", "sampler": "random", "pruner": None},
    "TPE": {"kind": "optuna", "sampler": "tpe", "pruner": None},
    "GP": {"kind": "optuna", "sampler": "gp", "pruner": None},
    "CMAES": {"kind": "optuna", "sampler": "cmaes", "pruner": None},
    "TPE_HB": {"kind": "optuna", "sampler": "tpe", "pruner": "hyperband"},   # LCBench only
    "SMAC": {"kind": "smac"},                                                # not run on Windows
}


def method_spec(name: str, fmp_base: dict | None = None, baseline_params: dict | None = None) -> dict:
    """Resolve a method name to a spec. `fmp_base` = shared FMP hyperparameters
    (Table 5 defaults when empty); variant flags override it."""
    if name in FMP_VARIANTS:
        params = {**(fmp_base or {}), **FMP_VARIANTS[name]}
        return {"kind": "fmp", "params": params}
    if name in BASELINES:
        spec = dict(BASELINES[name])
        spec["params"] = dict((baseline_params or {}).get(name, {}))
        return spec
    raise KeyError(f"unknown method {name!r}; known: {sorted(FMP_VARIANTS) + sorted(BASELINES)}")


# ---------------------------------------------------------------------------
# Trace recorder
# ---------------------------------------------------------------------------

class Trace:
    def __init__(self, task: Task, budget_evals: int):
        self.task = task
        self.budget = budget_evals
        self.values: list[float] = []
        self.raw: list[float] = []
        self.true_values: list[float] | None = [] if task.true_objective else None
        self.costs: list[float] = []
        self.fidelity: list[float] = []
        self.configs: list[dict] = []
        self.cum_cost = 0.0
        self.t_obj = 0.0

    def evaluate(self, cfg: dict, fidelity: float | None = None, objective=None) -> float:
        """Evaluate and record; returns the loss (minimize orientation)."""
        t = time.perf_counter()
        v = float(self.task.objective(cfg)) if objective is None else float(objective(cfg))
        self.t_obj += time.perf_counter() - t
        self.record(cfg, v, fidelity)
        return v

    def record(self, cfg: dict, v: float, fidelity: float | None = None, cost: float | None = None):
        self.values.append(v)
        self.raw.append(float(self.task.raw_of_loss(v)))
        self.configs.append(_jsonable(cfg))
        if self.true_values is not None:
            self.true_values.append(float(self.task.true_objective(cfg)))
        c = self.task.eval_cost if cost is None else cost
        self.cum_cost += c
        self.costs.append(self.cum_cost)
        self.fidelity.append(float(self.task.eval_cost if fidelity is None else fidelity))

    @property
    def n_full(self) -> int:
        return sum(1 for f in self.fidelity if f >= self.task.eval_cost)

    def to_record(self, method: str, seed: int, params: dict, t_total: float) -> dict:
        full = [i for i, f in enumerate(self.fidelity) if f >= self.task.eval_cost]
        if full:
            i_best = min(full, key=lambda i: self.values[i])
        else:
            i_best = int(np.argmin(self.values))
        rec = {
            "method": method, "task": self.task.key, "suite": spec_suite(self.task.spec), "task_spec": self.task.spec,
            "seed": seed, "budget": self.budget, "params": _jsonable(params),
            "values": self.values, "raw": self.raw, "true_values": self.true_values,
            "costs": self.costs, "fidelity": self.fidelity, "configs": self.configs,
            "best_value": self.values[i_best], "best_raw": self.raw[i_best], "best_config": self.configs[i_best],
            "best_true_value": (self.true_values[i_best] if self.true_values is not None else self.values[i_best]),
            "f_star": self.task.f_star, "maximize_raw": self.task.maximize_raw, "eval_cost": self.task.eval_cost,
            "time_total": t_total, "time_objective": self.t_obj, "n_evals": len(self.values),
        }
        return rec


def _jsonable(obj: Any):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Optuna baselines
# ---------------------------------------------------------------------------

def _suggest(trial, space: dict) -> dict:
    cfg = {}
    for name, info in space.items():
        t = info["type"]
        if t == "float":
            lo, hi = info["values"]
            cfg[name] = trial.suggest_float(name, float(lo), float(hi), log=bool(info.get("log", False)))
        elif t == "int":
            lo, hi = info["values"][0], info["values"][-1]
            cfg[name] = trial.suggest_int(name, int(lo), int(hi), log=bool(info.get("log", False)))
        elif t == "categorical":
            cfg[name] = trial.suggest_categorical(name, list(info["values"]))
        else:
            raise ValueError(t)
    return cfg


def _make_sampler(name: str, seed: int, n_startup: int, params: dict, noisy: bool):
    import optuna

    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "tpe":
        kw = dict(seed=seed, n_startup_trials=n_startup)
        kw.update(params)
        return optuna.samplers.TPESampler(**kw)
    if name == "gp":
        kw = dict(seed=seed, n_startup_trials=n_startup, deterministic_objective=not noisy)
        kw.update(params)
        return optuna.samplers.GPSampler(**kw)
    if name == "cmaes":
        kw = dict(seed=seed, n_startup_trials=n_startup, warn_independent_sampling=False)
        kw.update(params)
        return optuna.samplers.CmaEsSampler(**kw)
    raise ValueError(name)


def run_optuna(task: Task, seed: int, budget: int, spec: dict, n_startup: int) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    noisy = task.spec.get("noise_std", 0.0) > 0
    params = dict(spec.get("params", {}))
    sampler_params = {k: v for k, v in params.items() if k != "n_startup_trials"}
    n_startup = int(params.get("n_startup_trials", n_startup))
    sampler = _make_sampler(spec["sampler"], seed, n_startup, sampler_params, noisy)
    t0 = time.perf_counter()
    trace = Trace(task, budget)
    if spec.get("pruner") == "hyperband":
        rec = _run_optuna_hyperband(task, seed, budget, sampler, trace)
    else:
        study = optuna.create_study(direction="minimize", sampler=sampler)
        for _ in range(budget):
            trial = study.ask()
            cfg = _suggest(trial, task.space)
            v = trace.evaluate(cfg)
            study.tell(trial, v)
        rec = {}
    used = {"sampler": spec["sampler"], "pruner": spec.get("pruner"), "n_startup_trials": n_startup, **sampler_params}
    out = trace.to_record(spec["name"], seed, used, time.perf_counter() - t0)
    out.update(rec)
    return out


def _run_optuna_hyperband(task: Task, seed: int, budget: int, sampler, trace: Trace) -> dict:
    """BOHB-style TPE + Hyperband on LCBench: multi-fidelity with epoch cost accounting.
    Stops when the cumulative epoch cost reaches budget * max_epoch."""
    import optuna

    if task.spec["kind"] != "lcbench":
        raise ValueError("hyperband pruner is only wired for LCBench tasks")
    from experiments_new.common.yahpo import evaluate_at_epoch

    inst = task.extra["instance"]
    min_e, max_e = task.extra["min_epoch"], task.extra["max_epoch"]
    rungs = []
    r = max(1, min_e)
    while r < max_e:
        rungs.append(int(r))
        r *= 3
    rungs.append(int(max_e))
    total_budget = budget * task.eval_cost
    pruner = optuna.pruners.HyperbandPruner(min_resource=rungs[0], max_resource=int(max_e), reduction_factor=3)
    # HyperbandPruner assigns brackets by crc32(study_name + trial number): a fixed
    # study name makes the bracket schedule reproducible for a given seed.
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner,
                                study_name=f"tpe_hb_seed{seed}")
    n_pruned = 0
    while trace.cum_cost < total_budget:
        trial = study.ask()
        cfg = _suggest(trial, task.space)
        prev = 0
        last = None
        pruned = False
        for e in rungs:
            t = time.perf_counter()
            v = -evaluate_at_epoch(inst, cfg, e)
            trace.t_obj += time.perf_counter() - t
            trace.record(cfg, v, fidelity=float(e), cost=float(e - prev))
            prev = e
            last = v
            trial.report(v, e)
            if e != rungs[-1] and trial.should_prune():
                pruned = True
                break
            if trace.cum_cost >= total_budget:
                break
        if pruned:
            n_pruned += 1
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        else:
            study.tell(trial, last)
    return {"n_pruned": n_pruned, "rungs": rungs}


# ---------------------------------------------------------------------------
# FMP
# ---------------------------------------------------------------------------

def run_fmp(task: Task, seed: int, budget: int, spec: dict, keep_history: bool = True) -> dict:
    from hpo_rl.baselines.HMM_MCMC_FMP import HMM_MCMC_FMP

    params = copy.deepcopy(spec["params"])
    trace = Trace(task, budget)
    t0 = time.perf_counter()
    alg = HMM_MCMC_FMP(lambda cfg: trace.evaluate(cfg), budget, task.space, seed=seed, **params)
    alg.main_loop()
    out = trace.to_record(spec["name"], seed, {**params, "seed": seed}, time.perf_counter() - t0)
    fmp = {"summary": alg.summary(), "A_history": alg.A_history, "emission_history": alg.emission_history}
    if keep_history:
        fmp["history"] = [{k: v for k, v in r.items() if k != "config"} for r in alg.history]
    out["fmp"] = _jsonable(fmp)
    return out


# ---------------------------------------------------------------------------
# SMAC (written, not run on Windows; requires `pip install smac`)
# ---------------------------------------------------------------------------

def _configspace_from_space(space: dict, seed: int):
    import ConfigSpace as CS

    cs = CS.ConfigurationSpace(seed=seed)
    for name, info in space.items():
        t = info["type"]
        if t == "float":
            lo, hi = info["values"]
            cs.add(CS.UniformFloatHyperparameter(name, float(lo), float(hi), log=bool(info.get("log", False))))
        elif t == "int":
            lo, hi = info["values"][0], info["values"][-1]
            cs.add(CS.UniformIntegerHyperparameter(name, int(lo), int(hi), log=bool(info.get("log", False))))
        elif t == "categorical":
            cs.add(CS.CategoricalHyperparameter(name, list(info["values"])))
    return cs


def run_smac(task: Task, seed: int, budget: int, spec: dict, n_startup: int) -> dict:
    try:
        from smac import HyperparameterOptimizationFacade, Scenario
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("SMAC3 is not installed (no official Windows support). Install `smac` on Linux/WSL "
                           "and re-run only the SMAC jobs; all other results are kept.") from e
    import tempfile

    trace = Trace(task, budget)
    t0 = time.perf_counter()
    cs = _configspace_from_space(task.space, seed)
    with tempfile.TemporaryDirectory() as tmp:
        scenario = Scenario(cs, n_trials=budget, seed=seed, deterministic=True, output_directory=tmp)
        init = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=n_startup)

        def target(config, seed: int = 0) -> float:
            return trace.evaluate(dict(config))

        smac = HyperparameterOptimizationFacade(scenario, target, initial_design=init, overwrite=True, logging_level=40)
        smac.optimize()
    return trace.to_record(spec["name"], seed, {"n_startup_trials": n_startup}, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_method(name: str, spec: dict, task: Task, seed: int, budget: int, n_startup: int = 16,
               keep_history: bool = True) -> dict:
    spec = dict(spec)
    spec["name"] = name
    kind = spec["kind"]
    if kind == "fmp":
        return run_fmp(task, seed, budget, spec, keep_history=keep_history)
    if kind == "optuna":
        return run_optuna(task, seed, budget, spec, n_startup)
    if kind == "smac":
        return run_smac(task, seed, budget, spec, n_startup)
    raise ValueError(kind)
