"""HMM-HMC hybrid ablation: BO vs NUTS burst hybrid vs burst without HMM on lcbench."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from yahpo_gym import benchmark_set, local_config

from hpo_rl.baselines.HMM_MCMC_HMC import HMM_MCMC_HMC

REPO_ROOT = ROOT
DATA_PATH = Path(__import__("os").environ.get("YAHPO_DATA_PATH", REPO_ROOT / "yahpo_data"))
TUNING_RESULTS_DIR = REPO_ROOT / "experiments" / "yahpo_tuning_results"
BEST_PARAMS_PATH = TUNING_RESULTS_DIR / "best_params_hmm_mcmc_hmc.json"
PLOTS_DIR = REPO_ROOT / "experiments" / "yahpo_ablation_plots"
RESULTS_DIR = REPO_ROOT / "experiments" / "yahpo_ablation_results"

LCBENCH_CFG = {"target": "val_accuracy", "fidelity": "epoch"}

ARM_COLORS = {
    "bo": "#1f77b4",
    "hybrid": "#2ca02c",
    "mcmc_no_hmm": "#ff7f0e",
}
ARM_LABELS = {
    "bo": "Pure BO (burst disabled)",
    "hybrid": "Hybrid NUTS burst + HMM",
    "mcmc_no_hmm": "Burst on stuck only (no HMM)",
}

HMM_FIXED = {
    "n_chains": 1,
    "orchestrate_every": 1000,
    "T_min": None,
    "show_progress": False,
    "verbose_history": False,
}


@dataclass
class ScenarioContext:
    instance: str
    bench: Any
    target: str
    fidelity: str
    max_fidelity: float
    fidelity_is_int: bool
    hpo_rl_space: dict


@dataclass
class RunResult:
    curve: np.ndarray
    best_acc: float
    frac_argmax: float
    frac_burst: float
    frac_explore: float


def _hp_bounds(hp: CSH.Hyperparameter) -> tuple[float, float]:
    return float(hp.lower), float(hp.upper)


def _hp_max_constant(hp: CSH.Hyperparameter):
    if isinstance(hp, CSH.UniformIntegerHyperparameter):
        return int(hp.upper)
    return float(hp.upper)


def configspace_to_hpo_rl_space(cs: CS.ConfigurationSpace, exclude: set[str] | None = None) -> dict:
    exclude = exclude or set()
    space: dict = {}
    for hp in cs.get_hyperparameters():
        name = hp.name
        if name in exclude or isinstance(hp, CSH.Constant):
            continue
        if isinstance(hp, CSH.UniformFloatHyperparameter):
            space[name] = {
                "type": "float",
                "values": [float(hp.lower), float(hp.upper)],
                "log": bool(hp.log),
            }
        elif isinstance(hp, CSH.UniformIntegerHyperparameter):
            space[name] = {
                "type": "int",
                "values": [int(hp.lower), int(hp.upper)],
                "log": bool(hp.log),
            }
        elif isinstance(hp, CSH.CategoricalHyperparameter):
            space[name] = {"type": "categorical", "values": list(hp.choices)}
        elif isinstance(hp, CSH.OrdinalHyperparameter):
            space[name] = {"type": "categorical", "values": list(hp.sequence)}
        else:
            raise TypeError(f"Unsupported hyperparameter: {name} ({type(hp)})")
    return space


def build_lcbench_context(instance: str) -> ScenarioContext:
    bench = benchmark_set.BenchmarkSet("lcbench")
    bench.set_instance(instance)
    bench.check = False

    target = LCBENCH_CFG["target"]
    if target not in bench.targets:
        raise ValueError(f"Target {target} not in {bench.targets}")

    fidelity = LCBENCH_CFG["fidelity"]
    fspace = bench.get_fidelity_space()
    f_hp = fspace[fidelity]
    _, max_f = _hp_bounds(f_hp)
    fidelity_is_int = isinstance(f_hp, CSH.UniformIntegerHyperparameter)

    for fp in bench.config.fidelity_params:
        if fp == fidelity:
            continue
        bench.set_constant(fp, _hp_max_constant(fspace[fp]))

    opt_space = bench.get_opt_space(drop_fidelity_params=True)
    hpo_rl_space = configspace_to_hpo_rl_space(opt_space, exclude=set(bench.config.fidelity_params))

    return ScenarioContext(
        instance=instance,
        bench=bench,
        target=target,
        fidelity=fidelity,
        max_fidelity=max_f,
        fidelity_is_int=fidelity_is_int,
        hpo_rl_space=hpo_rl_space,
    )


def evaluate_at_max_fidelity(ctx: ScenarioContext, config: dict) -> float:
    cfg = dict(config)
    val = int(round(ctx.max_fidelity)) if ctx.fidelity_is_int else float(ctx.max_fidelity)
    cfg[ctx.fidelity] = val
    return float(ctx.bench.objective_function(cfg)[0][ctx.target])


def load_base_params() -> dict:
    if not BEST_PARAMS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing tuned params at {BEST_PARAMS_PATH}. "
            "Run experiments/yahpo_tune_hmm_lcbench.ipynb first."
        )
    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        tuned_params = json.load(f)
    return {
        **HMM_FIXED,
        **{
            k: v
            for k, v in tuned_params.items()
            if k
            not in (
                "use_hmm",
                "inner_burst_steps",
                "explore_every",
                "warmup_fraction",
                "show_progress",
                "verbose_history",
            )
        },
    }


def make_arms(base_params: dict) -> dict[str, dict]:
    return {
        "bo": {
            **base_params,
            "use_hmm": False,
            "burst_steps_escape": 0,
            "burst_steps_stuck": 0,
        },
        "hybrid": {
            **base_params,
            "use_hmm": True,
            "burst_steps_escape": 64,
            "burst_steps_stuck": 32,
        },
        "mcmc_no_hmm": {
            **base_params,
            "use_hmm": False,
            "burst_steps_escape": 0,
            "burst_steps_stuck": 32,
        },
    }


def kernel_fractions(history_table: list[dict], n_init: int) -> tuple[float, float, float]:
    post = [row for row in history_table if int(row["Eval"]) > n_init]
    if not post:
        return 0.0, 0.0, 0.0
    n = len(post)
    frac_argmax = sum(1 for row in post if row["Kernel"] == "argmax") / n
    frac_burst = sum(1 for row in post if row["Kernel"] == "burst") / n
    frac_explore = sum(1 for row in post if row["Kernel"] == "explore") / n
    return float(frac_argmax), float(frac_burst), float(frac_explore)


def run_hmc_curve(
    ctx: ScenarioContext,
    hmm_params: dict,
    n_evals: int,
    seed: int,
) -> RunResult:
    curve: list[float] = []
    best_acc = -np.inf

    def objective(config_dict: dict) -> float:
        nonlocal best_acc
        acc = evaluate_at_max_fidelity(ctx, config_dict)
        best_acc = max(best_acc, acc)
        curve.append(best_acc)
        return -acc

    np.random.seed(seed)
    alg = HMM_MCMC_HMC(
        objective_func=objective,
        budget=n_evals,
        dict_to_optimize=ctx.hpo_rl_space,
        seed=seed,
        **hmm_params,
    )
    alg.main_loop()
    frac_argmax, frac_burst, frac_explore = kernel_fractions(alg.history_table, alg.n_init)
    return RunResult(
        curve=np.asarray(curve, dtype=float),
        best_acc=float(best_acc),
        frac_argmax=frac_argmax,
        frac_burst=frac_burst,
        frac_explore=frac_explore,
    )


def _mean_sem(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = arr.mean(axis=0)
    if arr.shape[0] > 1:
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    else:
        sem = np.zeros_like(mean)
    return mean, sem


def plot_ablation(
    raw_curves: dict[tuple[str, str, int], np.ndarray],
    arms: dict[str, dict],
    instances: list[str],
    n_evals: int,
    budget: int,
    out_all: Path,
    out_per_inst: Path,
) -> None:
    x = np.arange(1, n_evals + 1)

    def stack(arm: str, instance: str | None = None) -> np.ndarray:
        curves = []
        for (a, inst, _seed), curve in raw_curves.items():
            if a != arm:
                continue
            if instance is not None and inst != instance:
                continue
            curves.append(curve)
        return np.vstack(curves)

    fig, ax = plt.subplots(figsize=(9, 5))
    for arm in arms:
        arr = stack(arm)
        mean, sem = _mean_sem(arr)
        color = ARM_COLORS[arm]
        ax.plot(x, mean, color=color, label=ARM_LABELS[arm], lw=2)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2)
    ax.set_xlabel("Evaluation index")
    ax.set_ylabel("Best-so-far val_accuracy (mean ± SEM)")
    ax.set_title(f"HMM-HMC hybrid ablation (budget={budget}, all instances)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_all, bbox_inches="tight")
    plt.close(fig)

    n_inst = len(instances)
    fig, axes = plt.subplots(1, n_inst, figsize=(5 * n_inst, 4), sharey=True)
    if n_inst == 1:
        axes = [axes]
    for ax, inst in zip(axes, instances):
        for arm in arms:
            arr = stack(arm, instance=inst)
            mean, sem = _mean_sem(arr)
            color = ARM_COLORS[arm]
            ax.plot(x, mean, color=color, label=ARM_LABELS[arm], lw=2)
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2)
        ax.set_title(f"instance {inst}")
        ax.set_xlabel("Evaluation index")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Best-so-far val_accuracy (mean ± SEM)")
    fig.suptitle(f"HMM-HMC hybrid ablation (budget={budget}, per instance)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_per_inst, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="HMM_MCMC_HMC hybrid ablation on lcbench")
    parser.add_argument("--smoke", action="store_true", help="Quick sanity run")
    parser.add_argument("--budget", type=int, default=200, help="Evaluation budget")
    args = parser.parse_args()

    if not hasattr(CS.ConfigurationSpace, "_sort_hyperparameters"):
        CS.ConfigurationSpace._sort_hyperparameters = lambda self: None

    local_config.init_config() if not local_config.settings_path.exists() else None
    local_config.set_data_path(str(DATA_PATH))
    encoding_probe = DATA_PATH / "lcbench" / "encoding.json"
    if not encoding_probe.is_file():
        raise FileNotFoundError(f"yahpo data not found at {DATA_PATH}")

    if args.smoke:
        instances = ["3945"]
        seeds = [42]
        n_evals = 40
    else:
        instances = ["3945", "7593", "189873"]
        seeds = [42, 43, 44, 45, 46]
        n_evals = int(args.budget)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_params = load_base_params()
    arms = make_arms(base_params)

    print("HMM-HMC hybrid ablation arms:")
    for arm_name, arm_params in arms.items():
        print(
            f"  {arm_name:12s}  use_hmm={arm_params['use_hmm']}  "
            f"burst_escape={arm_params['burst_steps_escape']}  "
            f"burst_stuck={arm_params['burst_steps_stuck']}"
        )
    print(f"instances={instances}, seeds={seeds}, n_evals={n_evals}")

    contexts = {inst: build_lcbench_context(inst) for inst in instances}
    raw_curves: dict[tuple[str, str, int], np.ndarray] = {}
    final_rows: list[dict] = []

    runs = [(arm, inst, seed) for arm in arms for inst in instances for seed in seeds]
    for i, (arm, inst, seed) in enumerate(runs, start=1):
        print(f"[{i}/{len(runs)}] arm={arm} inst={inst} seed={seed}", flush=True)
        result = run_hmc_curve(contexts[inst], arms[arm], n_evals, seed)
        raw_curves[(arm, inst, seed)] = result.curve
        arm_params = arms[arm]
        final_rows.append(
            {
                "arm": arm,
                "instance": inst,
                "seed": seed,
                "budget": n_evals,
                "use_hmm": arm_params["use_hmm"],
                "burst_steps_escape": arm_params["burst_steps_escape"],
                "burst_steps_stuck": arm_params["burst_steps_stuck"],
                "best_val_accuracy": result.best_acc,
                "final_curve": result.curve[-1] if len(result.curve) else np.nan,
                "frac_argmax": result.frac_argmax,
                "frac_burst": result.frac_burst,
                "frac_explore": result.frac_explore,
            }
        )

    df_runs = pd.DataFrame(final_rows)
    summary_rows = []
    for arm in arms:
        sub = df_runs[df_runs["arm"] == arm]
        summary_rows.append(
            {
                "arm": arm,
                "instance": "all",
                "mean_best": sub["best_val_accuracy"].mean(),
                "std_best": sub["best_val_accuracy"].std(ddof=1) if len(sub) > 1 else 0.0,
                "mean_frac_burst": sub["frac_burst"].mean(),
                "mean_frac_explore": sub["frac_explore"].mean(),
                "n_runs": len(sub),
            }
        )
        for inst in instances:
            sub_i = sub[sub["instance"] == inst]
            summary_rows.append(
                {
                    "arm": arm,
                    "instance": inst,
                    "mean_best": sub_i["best_val_accuracy"].mean(),
                    "std_best": sub_i["best_val_accuracy"].std(ddof=1) if len(sub_i) > 1 else 0.0,
                    "mean_frac_burst": sub_i["frac_burst"].mean(),
                    "mean_frac_explore": sub_i["frac_explore"].mean(),
                    "n_runs": len(sub_i),
                }
            )
    df_summary = pd.DataFrame(summary_rows)

    suffix = f"b{n_evals}"
    summary_path = RESULTS_DIR / f"hmc_hmm_ablation_summary_{suffix}.csv"
    runs_path = RESULTS_DIR / f"hmc_hmm_ablation_runs_{suffix}.csv"
    df_summary.to_csv(summary_path, index=False)
    df_runs.to_csv(runs_path, index=False)

    np.savez(
        RESULTS_DIR / f"hmc_hmm_ablation_raw_curves_{suffix}.npz",
        arms=np.array(list(arms.keys())),
        instances=np.array(instances),
        seeds=np.array(seeds),
        budget=np.array(n_evals),
        **{
            f"curve_{arm}_{inst}_{seed}": raw_curves[(arm, inst, seed)]
            for arm, inst, seed in raw_curves
        },
    )

    plot_ablation(
        raw_curves,
        arms,
        instances,
        n_evals,
        n_evals,
        PLOTS_DIR / f"hmc_hmm_ablation_convergence_all_{suffix}.png",
        PLOTS_DIR / f"hmc_hmm_ablation_convergence_per_instance_{suffix}.png",
    )

    print("\n=== Summary (mean best val_accuracy) ===")
    for arm in arms:
        sub = df_runs[df_runs["arm"] == arm]
        print(
            f"  {ARM_LABELS[arm]:36s}  "
            f"{sub['best_val_accuracy'].mean():.4f} ± "
            f"{sub['best_val_accuracy'].std(ddof=1) if len(sub) > 1 else 0.0:.4f}  "
            f"burst={sub['frac_burst'].mean():.2%}  "
            f"explore={sub['frac_explore'].mean():.2%}"
        )

    hybrid_mean = df_runs[df_runs["arm"] == "hybrid"]["best_val_accuracy"].mean()
    bo_mean = df_runs[df_runs["arm"] == "bo"]["best_val_accuracy"].mean()
    no_hmm_mean = df_runs[df_runs["arm"] == "mcmc_no_hmm"]["best_val_accuracy"].mean()
    print(f"\nLift hybrid - bo: {hybrid_mean - bo_mean:+.4f}")
    print(f"Lift hybrid - mcmc_no_hmm: {hybrid_mean - no_hmm_mean:+.4f}")
    print(f"Saved {summary_path}")
    print(f"Saved {runs_path}")
    print(f"Saved plots under {PLOTS_DIR}")


if __name__ == "__main__":
    main()
