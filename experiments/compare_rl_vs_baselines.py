"""
Compare DQN and Recurrent DQN inference vs grid search and random search on the same
benchmarks as in run_experiment.ipynb (2D classics; extend via ``FUNCTION_ORDER``).

Budget: 200 function evaluations for RL rollouts and random search; grid search uses
14x14 = 196 points (padded to 200 on the plot for alignment).

Usage:
    python compare_rl_vs_baselines.py

Checkpoints (`--ckpt-recurrent`, `--ckpt-dqn`; use forward slashes on Windows).
Training saves `algo.state_dict()` — `config_*` hyperparameters must match the run
that produced the checkpoint (especially net hidden sizes / optimizer for recurrent DQN).

Set WANDB_MODE=disabled to avoid W&B uploads (default for this script).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

# Before imports that may init W&B
os.environ.setdefault("WANDB_MODE", "disabled")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam, AdamW

from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy

from hpo_rl.backends.function import OptimizationBenchmarkBackend
from hpo_rl.controller.check import check
from hpo_rl.controller.controller import controller
from hpo_rl.nets.gradient_monitor import GradientMonitoredNet
from hpo_rl.nets.masked_recurrent_net import MaskedRecurrentNet

# ── Experiment constants (plan) ─────────────────────────────────────────
BUDGET = 200
N_SEEDS = 3
GRID_PER_AXIS = 14  # 14*14 = 196 evaluations

# Order matters for ``SequentialBackend`` indices: keep the first blocks aligned with how
# ``final_policy.pth`` was trained whenever you reuse old checkpoints (append new benchmarks).
FUNCTION_ORDER = [
    # N-D functions (work in 2D)
    "rastrigin",
    "rosenbrock",
    "schwefel",
    "ackley",
    "sphere",
    "griewank",
    "levy",
    "michalewicz",
    "styblinski_tang",
    # 2D-only functions
    "booth",
    "beale",
    "goldstein_price",
    "bukin_n6",
    "cross_in_tray",
    "drop_wave",
    "eggholder",
    "holder_table",
    "schaffer_n2",
    "schaffer_n4",
    "shubert",
    "dejong_n5",
    "easom",
    "levy_n13",
    "langermann",
]

# noise_std ≈ 10 % of each function's typical value range.
# Large enough to be significant, small enough not to bury the landscape signal.
NOISE_STD_MAP: dict[str, float] = {
    # N-D functions
    "sphere":           5.0,    # range [0, ~50]
    "rosenbrock":       20.0,   # range [0, ~400]  (2D, bounds [-1,1])
    "rastrigin":        8.0,    # range [0, ~80]
    "ackley":           2.0,    # range [0, ~22]
    "griewank":         10.0,   # range [0, ~100]
    "schwefel":         150.0,  # range [0, ~1677]
    "levy":             8.0,    # range [0, ~100]
    "michalewicz":      0.1,    # range [-2, 0]
    "styblinski_tang":  10.0,   # range [-78, ~250]
    # 2D-only functions
    "booth":            30.0,   # range [0, ~1200]
    "beale":            5.0,    # typical near-optimum values
    "goldstein_price":  50.0,   # range [3, ~1e4]
    "bukin_n6":         20.0,   # range [0, ~500]
    "cross_in_tray":    0.05,   # range [-2.06, 0]
    "drop_wave":        0.05,   # range [-1, 0.5]
    "eggholder":        50.0,   # range [-960, ~1000]
    "holder_table":     1.0,    # range [-19, 0]
    "schaffer_n2":      0.05,   # range [0, 1]
    "schaffer_n4":      0.05,   # range [0, 1]
    "shubert":          15.0,   # range [-186, ~200]
    "dejong_n5":        10.0,   # range [~1, ~500]
    "easom":            0.05,   # range [-1, 0]
    "levy_n13":         15.0,   # range [0, ~300]
    "langermann":       0.1,    # range [-1.5, ~1]
}

BACKENDS_LIST = [
    {"name": "function", "function": fn, "dimensions": 2,
     "noise_std": NOISE_STD_MAP[fn]}
    for fn in FUNCTION_ORDER
]

BACKENDS_LIST_CLEAN = [
    {"name": "function", "function": fn, "dimensions": 2, "noise_std": 0.0}
    for fn in FUNCTION_ORDER
]

DEFAULT_CKPT_RECURRENT = "log/recurrent_dqn/20260509-235607/final_policy.pth"
DEFAULT_CKPT_DQN = "log/dqn/20260509-233450/final_policy.pth"

OUT_DIR = Path(__file__).parent.parent / "logs" / "compare_rl_vs_baselines"
RESULTS_TXT = "compare_rl_vs_baselines_results.txt"
PLOT_FILE = "compare_rl_vs_baselines_convergence.png"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_recurrent_dqn(load_checkpoint: str | None, backends_list: list | None = None) -> dict:
    """Match recurrent DQN block in run_experiment.ipynb."""
    cfg = {
        "full_args": {
            "algorithm": {
                "name": "recurrent_dqn",
                "gamma": 0.99,
                "seq_len": 10,
                "target_update_freq": 500,
            },
            "buffer": {
                "total_size": 100000,
                "buffer_num": 20,
                "stack_num": 1,
            },
            "optim": {
                "name": "TorchOptimizerFactory",
                "optim_class": Adam,
                "lr": 1e-3,
            },
            "net": {
                "hidden_sizes": [256, 256, 256],
                "net": MaskedRecurrentNet,
                "rnn_layers": 1,
            },
            "trainer": {
                "max_epochs": 60,
                "epoch_num_steps": 6000,
                "batch_size": 64,
                "collection_step_num_env_steps": 2000,
                "update_step_num_gradient_steps_per_sample": 1.0,
            },
            "policy": {
                "class": DiscreteQLearningPolicy,
                "eps_training": 0.25,
                "eps_inference": 0.0,
            },
            "inference": {
                "n_episode": 1,
                "reset_before_collect": True,
            },
            "num_training_envs": 20,
            "num_test_envs": 20,
        },
        "env": {
            "name": "new_cycle_move_pipeline",
            "num_bins": 500,
            "max_steps": 200,
            "step_sizes": [1, 2, 5, 10, 25, 50],
            "history_window": 0,
            "reward_mode": "absolute",
        },
        "backend": {
            "name": "sequential",
            "mode": "shuffle",
            "backends": deepcopy(backends_list if backends_list is not None else BACKENDS_LIST),
        },
    }
    if load_checkpoint:
        cfg["full_args"]["load_checkpoint"] = str(Path(load_checkpoint).as_posix())
    return cfg


def config_dqn(load_checkpoint: str | None, backends_list: list | None = None) -> dict:
    """Match DQN block in run_experiment.ipynb (sequential backends from ``FUNCTION_ORDER``)."""
    cfg = {
        "full_args": {
            "algorithm": {
                "name": "dqn",
                "gamma": 0.99,
                "target_update_freq": 200,
                "huber_loss_delta": 0.5,
            },
            "buffer": {
                "total_size": 20000,
                "buffer_num": 20,
                "stack_num": 1,
            },
            "optim": {
                "name": "TorchOptimizerFactory",
                "optim_class": AdamW,
                "lr": 3e-4,
                "weight_decay": 0.1,
            },
            "net": {
                "net": GradientMonitoredNet,
                "hidden_sizes": [256, 256, 256],
                "grad_log_interval": 200_000,  # quiet during comparison
                "grad_verbose": False,
            },
            "trainer": {
                "max_epochs": 20,
                "epoch_num_steps": 4000,
                "batch_size": 20,
                "collection_step_num_env_steps": 200,
            },
            "policy": {
                "class": DiscreteQLearningPolicy,
                "eps_training": 0.1,
                "eps_inference": 0.0,
            },
            "inference": {
                "n_episode": 1,
                "reset_before_collect": True,
            },
            "num_training_envs": 20,
            "num_test_envs": 20,
        },
        "env": {
            "name": "new_cycle_move_pipeline",
            "num_bins": 500,
            "max_steps": 200,
            "step_sizes": [1, 2, 5, 10, 25, 50],
            "history_window": 3,
            "reward_mode": "absolute",
        },
        "backend": {
            "name": "sequential",
            "mode": "shuffle",
            "backends": deepcopy(backends_list if backends_list is not None else BACKENDS_LIST),
        },
    }
    if load_checkpoint:
        cfg["full_args"]["load_checkpoint"] = str(Path(load_checkpoint).as_posix())
    return cfg


def _scores_to_best_curve(scores: list[float], maximize: bool) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if maximize:
        return np.maximum.accumulate(arr)
    return np.minimum.accumulate(arr)


def _pad_curve(y: np.ndarray, budget: int) -> np.ndarray:
    if len(y) >= budget:
        return y[:budget].astype(float)
    if len(y) == 0:
        return np.full(budget, np.nan)
    pad = np.full(budget - len(y), y[-1], dtype=float)
    return np.concatenate([y.astype(float), pad])


def run_rl_episode(
    raw_config: dict,
    function_idx: int,
    seed: int,
    budget: int,
) -> np.ndarray:
    """One inference rollout on a locked Sequential child; returns best-so-far curve, len=budget."""
    _seed_all(seed)
    cfg = deepcopy(raw_config)
    parsed = check(cfg)
    ctrl = controller(**parsed)
    if not getattr(ctrl, "load_loc", None):
        raise ValueError("RL config must set full_args.load_checkpoint for this script.")
    if not getattr(ctrl, "_checkpoint_loaded", False):
        raise RuntimeError(
            f"Checkpoint was NOT loaded from {ctrl.load_loc!r}. "
            "Use forward slashes in the path (e.g. log/recurrent_dqn/.../final_policy.pth)."
        )
    if not hasattr(ctrl.backend, "set_active_backend"):
        raise RuntimeError("Expected SequentialBackend.")

    ctrl.backend.set_active_backend(int(function_idx), lock=True)
    ctrl.inference()
    history = ctrl.return_history()
    if ctrl.backend.maximize:
        scores = [float(t[1]) for t in history]
        # Stored metrics follow backend; for benchmark we minimize — convert if needed
        best_curve = _scores_to_best_curve(scores, maximize=True)
    else:
        scores = [float(t[1]) for t in history]
        best_curve = _scores_to_best_curve(scores, maximize=False)

    return _pad_curve(best_curve, budget)


def run_grid_search(function_name: str, budget: int = BUDGET, noise_std: float = 0.0) -> np.ndarray:
    """Full 2D grid on native bounds; best-so-far curve, padded to `budget`."""
    backend = OptimizationBenchmarkBackend(
        function_name=function_name, dimensions=2, noise_std=noise_std,
    )
    lo0, hi0 = backend.bounds[0]
    lo1, hi1 = backend.bounds[1]
    n = GRID_PER_AXIS
    xs = np.linspace(lo0, hi0, n, dtype=np.float64)
    ys = np.linspace(lo1, hi1, n, dtype=np.float64)
    scores: list[float] = []
    for x0 in xs:
        for x1 in ys:
            cfg = {"x0": float(x0), "x1": float(x1)}
            scores.append(float(backend.evaluate(cfg)))
    best = _scores_to_best_curve(scores, maximize=backend.maximize)
    return _pad_curve(best, budget)


def run_random_search(function_name: str, seed: int, budget: int = BUDGET, noise_std: float = 0.0) -> np.ndarray:
    backend = OptimizationBenchmarkBackend(
        function_name=function_name, dimensions=2, noise_std=noise_std,
    )
    rng = np.random.default_rng(seed)
    lo0, hi0 = backend.bounds[0]
    lo1, hi1 = backend.bounds[1]
    scores = []
    for _ in range(budget):
        cfg = {
            "x0": float(rng.uniform(lo0, hi0)),
            "x1": float(rng.uniform(lo1, hi1)),
        }
        scores.append(float(backend.evaluate(cfg)))
    best = _scores_to_best_curve(scores, maximize=backend.maximize)
    return _pad_curve(best, budget)


def _plot_results(
    curves: dict[str, dict[str, list[np.ndarray]]],
    outfile: str,
    *,
    budget: int,
    n_seeds: int,
) -> None:
    """
    curves[method][function_name] = list of seed curves (or single for grid), each length ``budget``.
    """
    evals = np.arange(1, budget + 1)
    n_fn = len(FUNCTION_ORDER)
    cols = min(4, max(1, n_fn))
    rows = max(1, (n_fn + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.8 * rows), squeeze=False)

    methods_styles = [
        ("DQN", "C0", "-"),
        ("Recurrent_DQN", "C1", "-"),
        ("Grid", "C2", "--"),
        ("Random", "C3", "-"),
    ]

    for idx, fname in enumerate(FUNCTION_ORDER):
        ax = axes[idx // cols][idx % cols]
        for label, color, linestyle in methods_styles:
            key = label.replace("_", " ") if label == "Recurrent_DQN" else label
            if key == "Recurrent DQN":
                stack_key = "Recurrent_DQN"
            else:
                stack_key = label
            if stack_key not in curves or fname not in curves[stack_key]:
                continue
            runs = curves[stack_key][fname]
            if not runs:
                continue
            stack = np.vstack(runs)
            if not np.isfinite(stack).any():
                continue
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
            disp = "Recurrent DQN" if stack_key == "Recurrent_DQN" else stack_key
            ax.plot(evals, mean, label=disp, color=color, linestyle=linestyle, linewidth=2)
            if len(runs) > 1:
                ax.fill_between(evals, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_title(fname)
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("Best objective so far")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    for j in range(n_fn, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(
        f"Best-so-far vs evaluations (budget={budget}, RL/Random: mean ± std over {n_seeds} seeds)"
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {outfile}")


def _plot_per_function(
    curves: dict[str, dict[str, list[np.ndarray]]],
    out_dir: str,
    *,
    budget: int,
    n_seeds: int,
) -> None:
    """Saves one PNG per benchmark function into *out_dir*."""
    os.makedirs(out_dir, exist_ok=True)
    evals = np.arange(1, budget + 1)
    methods_styles = [
        ("DQN",          "C0", "-"),
        ("Recurrent_DQN","C1", "-"),
        ("Grid",         "C2", "--"),
        ("Random",       "C3", "-"),
    ]

    for fname in FUNCTION_ORDER:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label, color, linestyle in methods_styles:
            stack_key = label
            runs = curves.get(stack_key, {}).get(fname, [])
            if not runs:
                continue
            stack = np.vstack(runs)
            if not np.isfinite(stack).any():
                continue
            mean = np.nanmean(stack, axis=0)
            std  = np.nanstd(stack, axis=0)
            disp = "Recurrent DQN" if stack_key == "Recurrent_DQN" else stack_key
            ax.plot(evals, mean, label=disp, color=color, linestyle=linestyle, linewidth=2)
            if len(runs) > 1:
                ax.fill_between(evals, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_title(f"{fname}  (budget={budget})")
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("Best objective so far")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{fname}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")


def _print_table(
    final_vals: dict[str, dict[str, list[float]]],
    out_stream=sys.stdout,
) -> None:
    """final_vals[method][func] -> list of final best values per seed (single elem for grid)."""
    hdr = (
        f"{'Function':<16} | {'DQN':>22} | {'Recurrent DQN':>22} | "
        f"{'Grid':>12} | {'Random':>22}"
    )
    sep = "-" * len(hdr)
    print(sep, file=out_stream)
    print(hdr, file=out_stream)
    print(sep, file=out_stream)
    for fn in FUNCTION_ORDER:
        row_dqn = final_vals.get("DQN", {}).get(fn, [float("nan")])
        row_rdqn = final_vals.get("Recurrent_DQN", {}).get(fn, [float("nan")])
        glist = final_vals.get("Grid", {}).get(fn, [])
        row_g = float(glist[0]) if glist else float("nan")
        r_runs = final_vals.get("Random", {}).get(fn, [float("nan")])

        def fmt_runs(xs):
            if not xs or (len(xs) == 1 and np.isnan(xs[0])):
                return "n/a"
            m, s = float(np.mean(xs)), float(np.std(xs))
            return f"{m:.4f} ± {s:.4f}"

        g_str = f"{row_g:12.4f}" if np.isfinite(row_g) else "n/a".rjust(12)

        print(
            f"{fn:<16} | {fmt_runs(row_dqn):>22} | {fmt_runs(row_rdqn):>22} | "
            f"{g_str} | {fmt_runs(r_runs):>22}",
            file=out_stream,
        )
    print(sep, file=out_stream)


def main():
    # Avoid UnicodeEncodeError on Windows when dependencies print non-ASCII hints.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-dqn",
        type=str,
        default=DEFAULT_CKPT_DQN,
        help="Path to DQN checkpoint (final_policy.pth)",
    )
    parser.add_argument(
        "--ckpt-recurrent",
        type=str,
        default=DEFAULT_CKPT_RECURRENT,
        help="Path to recurrent DQN checkpoint (final_policy.pth)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=N_SEEDS,
        help="Number of random seeds for RL and random search",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=BUDGET,
        help="Evaluation budget for RL, random search, and padded plot length",
    )
    args = parser.parse_args()
    budget = args.budget
    n_seeds = args.seeds

    dqn_ckpt  = Path(args.ckpt_dqn).as_posix()
    rdqn_ckpt = Path(args.ckpt_recurrent).as_posix()

    base_seeds = [42 + k for k in range(n_seeds)]

    variants = [
        ("clean", BACKENDS_LIST_CLEAN, {}),
        ("noisy", BACKENDS_LIST,       NOISE_STD_MAP),
    ]

    for variant_name, bl, noise_map in variants:
        print(f"\n{'='*60}")
        print(f"  Variant: {variant_name.upper()}")
        print(f"{'='*60}\n")

        dqn_cfg  = config_dqn(dqn_ckpt, bl)
        rdqn_cfg = config_recurrent_dqn(rdqn_ckpt, bl)

        curves: dict[str, dict[str, list[np.ndarray]]] = {
            "DQN": {fn: [] for fn in FUNCTION_ORDER},
            "Recurrent_DQN": {fn: [] for fn in FUNCTION_ORDER},
            "Grid": {fn: [] for fn in FUNCTION_ORDER},
            "Random": {fn: [] for fn in FUNCTION_ORDER},
        }
        finals: dict[str, dict[str, list[float]]] = {
            "DQN": {fn: [] for fn in FUNCTION_ORDER},
            "Recurrent_DQN": {fn: [] for fn in FUNCTION_ORDER},
            "Grid": {fn: [] for fn in FUNCTION_ORDER},
            "Random": {fn: [] for fn in FUNCTION_ORDER},
        }

        # ── RL ────────────────────────────────────────────────────────────
        for seed in base_seeds:
            for fi, fname in enumerate(FUNCTION_ORDER):
                print(f"[DQN] seed={seed} function={fname} ...")
                try:
                    y = run_rl_episode(dqn_cfg, fi, seed, budget)
                except Exception as e:
                    print(f"  ERROR DQN {fname} seed {seed}: {e}", file=sys.stderr)
                    y = np.full(budget, np.nan)
                curves["DQN"][fname].append(y)
                finals["DQN"][fname].append(float(y[-1]))

                print(f"[Recurrent DQN] seed={seed} function={fname} ...")
                try:
                    y_r = run_rl_episode(rdqn_cfg, fi, seed, budget)
                except Exception as e:
                    print(f"  ERROR Recurrent DQN {fname} seed {seed}: {e}", file=sys.stderr)
                    y_r = np.full(budget, np.nan)
                curves["Recurrent_DQN"][fname].append(y_r)
                finals["Recurrent_DQN"][fname].append(float(y_r[-1]))

        # ── Grid (deterministic) ──────────────────────────────────────────
        for fname in FUNCTION_ORDER:
            fn_noise = noise_map.get(fname, 0.0)
            g = run_grid_search(fname, budget, noise_std=fn_noise)
            curves["Grid"][fname].append(g)
            finals["Grid"][fname].append(float(g[-1]))

        # ── Random search ─────────────────────────────────────────────────
        for seed in base_seeds:
            for fname in FUNCTION_ORDER:
                fn_noise = noise_map.get(fname, 0.0)
                r = run_random_search(fname, seed, budget, noise_std=fn_noise)
                curves["Random"][fname].append(r)
                finals["Random"][fname].append(float(r[-1]))

        plot_file   = str(OUT_DIR / PLOT_FILE.replace(".png", f"_{variant_name}.png"))
        plot_dir    = str(OUT_DIR / f"plots_per_function/{variant_name}")
        results_txt = str(OUT_DIR / RESULTS_TXT.replace(".txt", f"_{variant_name}.txt"))

        _plot_results(curves, plot_file, budget=budget, n_seeds=n_seeds)
        _plot_per_function(curves, plot_dir, budget=budget, n_seeds=n_seeds)

        print()
        _print_table(finals)
        summary = {
            "variant": variant_name,
            "budget": budget,
            "n_seeds": n_seeds,
            "checkpoints": {"dqn": args.ckpt_dqn, "recurrent_dqn": args.ckpt_recurrent},
            "final_best_per_method": {
                m: {fn: finals[m][fn] for fn in FUNCTION_ORDER} for m in finals
            },
        }
        with open(results_txt, "w", encoding="utf-8") as f:
            f.write(json.dumps(summary, indent=2))
            f.write("\n\n")
            _print_table(finals, out_stream=f)
        print(f"\nWrote table + JSON summary to {results_txt}")


if __name__ == "__main__":
    main()
