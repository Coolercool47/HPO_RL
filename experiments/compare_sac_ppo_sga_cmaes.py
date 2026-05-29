"""
Compare SAC and PPO (trained checkpoints, inference only) vs SimpleGA and CMA-ES on the
same 2D benchmark suite as ``compare_rl_vs_baselines.py``.

Budget: 200 objective evaluations per method per seed (RL episode length matches ``max_steps``
in ``instant_continuous_pipeline``).

SAC / PPO use your saved hyperparameter layouts (see ``config_sac``, ``config_ppo``) with the
full ``FUNCTION_ORDER`` sequential backend — checkpoints must match that ordering and net specs.

Usage:
    python compare_sac_ppo_sga_cmaes.py --ckpt-sac log/sac/<run>/final_policy.pth \\
        --ckpt-ppo log/ppo/<run>/final_policy.pth

Use forward slashes on Windows for checkpoint paths. Training saves ``algo.state_dict()`` —
``config_sac`` / ``config_ppo`` hyperparameters (net sizes, env deltas, AutoAlpha, etc.) must
match the run that produced each checkpoint.

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

os.environ.setdefault("WANDB_MODE", "disabled")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import tianshou.algorithm.optim as opt
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.modelfree.sac import SACPolicy, AutoAlpha
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic

from hpo_rl.controller.check import check
from hpo_rl.controller.controller import controller
from hpo_rl.nets.base_net import BaseNet

BUDGET = 200
N_SEEDS = 3

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

DEFAULT_CKPT_SAC = "log/sac/20260509-144135/20260509-144135/best_policy.pth"
DEFAULT_CKPT_PPO = "log/ppo/20260509-200654/final_policy.pth"

OUT_DIR = Path(__file__).parent.parent / "logs" / "compare_sac_ppo_sga_cmaes"
RESULTS_TXT = "compare_sac_ppo_sga_cmaes_results.txt"
PLOT_FILE = "compare_sac_ppo_sga_cmaes_convergence.png"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_sac(load_checkpoint: str | None, backends_list: list | None = None) -> dict:
    """SAC hyperparameters from your saved config; sequential backend = ``FUNCTION_ORDER``."""
    cfg = {
        "full_args": {
            "algorithm": {
                "name": "sac",
                "gamma": 0.99,
                "tau": 0.005,
                # JSON dumps show ``AutoAlpha()``; this tianshou build requires explicit args.
                "alpha": AutoAlpha(
                    target_entropy=-2.0,
                    log_alpha=0.0,
                    optim=opt.AdamOptimizerFactory(lr=1e-4),
                ),
                "n_step_return_horizon": 1,
            },
            "optim": {
                "name": "TorchOptimizerFactory",
                "optim_class": optim.Adam,
                "lr": 3e-4,
            },
            "net": {
                "actor": ContinuousActorProbabilistic,
                "critic": ContinuousCritic,
                "net": Net,
                "hidden_sizes": [256, 256, 256],
            },
            "buffer": {
                "total_size": 100000,
                "buffer_num": 20,
                "stack_num": 1,
            },
            "trainer": {
                "max_epochs": 20,
                "epoch_num_steps": 4000,
                "batch_size": 256,
                "collection_step_num_env_steps": 2000,
                "update_step_num_gradient_steps_per_sample": 1.0,
                "test_step_num_episodes": 20,
            },
            "policy": {
                "class": SACPolicy,
                "action_scaling": True,
                "actor_kwargs": {"unbounded": True, "conditioned_sigma": False},
            },
            "inference": {
                "n_episode": 1,
                "reset_before_collect": True,
            },
            "num_training_envs": 20,
            "num_test_envs": 20,
        },
        "env": {
            "name": "instant_continuous_pipeline",
            "max_delta_frac": 0.05,
            "max_steps": 200,
            "history_window": 1,
            "reward_mode": "absolute",
            "terminate_on_oob": False,
            "oob_penalty": 0.0,
            "oob_tolerance": 3,
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


def config_ppo(load_checkpoint: str | None, backends_list: list | None = None) -> dict:
    """PPO hyperparameters from your saved config; sequential backend = ``FUNCTION_ORDER``."""
    cfg = {
        "full_args": {
            "algorithm": {
                "name": "ppo",
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "vf_coef": 0.5,
                "ent_coef": 0.0,
                "max_grad_norm": 0.5,
                "value_clip": True,
                "return_scaling": True,
                "recompute_advantage": True,
            },
            "optim": {
                "name": "TorchOptimizerFactory",
                "optim_class": optim.Adam,
                "lr": 3e-4,
            },
            "net": {
                "actor": ContinuousActorProbabilistic,
                "critic": ContinuousCritic,
                "net": BaseNet,
                "hidden_sizes": [256, 256],
                "norm_layer": torch.nn.LayerNorm,
            },
            "trainer": {
                "max_epochs": 100,
                "epoch_num_steps": 4000,
                "batch_size": 256,
                "collection_step_num_env_steps": 2000,
                "update_step_num_repetitions": 10,
                "test_step_num_episodes": 20,
            },
            "policy": {
                "class": ProbabilisticActorPolicy,
                "dist_fn": lambda mu_sigma: torch.distributions.Independent(
                    torch.distributions.Normal(*mu_sigma), 1
                ),
                "action_scaling": True,
                "action_bound_method": "clip",
                "actor_kwargs": {"unbounded": True, "conditioned_sigma": True},
            },
            "inference": {
                "n_episode": 1,
                "reset_before_collect": True,
            },
            "num_training_envs": 20,
            "num_test_envs": 20,
        },
        "env": {
            "name": "instant_continuous_pipeline",
            "max_delta_frac": 0.1,
            "max_steps": 200,
            "history_window": 1,
            "reward_mode": "absolute",
            "terminate_on_oob": False,
            "oob_penalty": -1.0,
            "oob_tolerance": 3,
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
    """One inference rollout on a locked Sequential child; returns best-so-far curve."""
    _seed_all(seed)
    cfg = deepcopy(raw_config)
    parsed = check(cfg)
    ctrl = controller(**parsed)
    if not getattr(ctrl, "load_loc", None):
        raise ValueError("RL config must set full_args.load_checkpoint for this script.")
    if not getattr(ctrl, "_checkpoint_loaded", False):
        raise RuntimeError(
            f"Checkpoint was NOT loaded from {ctrl.load_loc!r}. "
            "Use forward slashes in the path (e.g. log/sac/.../final_policy.pth)."
        )
    if not hasattr(ctrl.backend, "set_active_backend"):
        raise RuntimeError("Expected SequentialBackend.")

    ctrl.backend.set_active_backend(int(function_idx), lock=True)
    ctrl.inference()
    history = ctrl.return_history()
    maximize = ctrl.backend.maximize
    scores = [float(t[1]) for t in history]
    best_curve = _scores_to_best_curve(scores, maximize=maximize)
    return _pad_curve(best_curve, budget)


def run_baseline(
    algorithm_name: str,
    function_name: str,
    seed: int,
    budget: int,
    noise_std: float = 0.0,
) -> np.ndarray:
    """SimpleGA or CMA_ES on a single function backend; best-so-far curve length ``budget``."""
    _seed_all(seed)
    backend_cfg = {"name": "function", "function": function_name, "dimensions": 2,
                   "noise_std": noise_std}

    if algorithm_name == "SimpleGA":
        full_args_alg = {
            "name": "SimpleGA",
            "N_pop": 20,
            "budget": budget,
            "mutation_prob": 0.1,
            "crossover_prob": 0.8,
            "tournament_size": 5,
            "elitism": True,
        }
    elif algorithm_name == "CMA_ES":
        full_args_alg = {
            "name": "CMA_ES",
            "N_pop": None,
            "budget": budget,
            "initial_step_size": 0.5,
        }
    else:
        raise ValueError(f"Unknown baseline algorithm: {algorithm_name}")

    raw = {"backend": backend_cfg, "full_args": {"algorithm": full_args_alg}}
    parsed = check(raw)
    ctrl = controller(**parsed)
    ctrl.inference()
    history = ctrl.return_history()
    maximize = ctrl.backend.maximize
    scores = [float(t[1]) for t in history]
    best_curve = _scores_to_best_curve(scores, maximize=maximize)
    return _pad_curve(best_curve, budget)


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
        ("SAC",      "C0", "-"),
        ("PPO",      "C1", "-"),
        ("SimpleGA", "C2", "--"),
        ("CMA_ES",   "C3", "-."),
    ]

    for fname in FUNCTION_ORDER:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label, color, linestyle in methods_styles:
            runs = curves.get(label, {}).get(fname, [])
            if not runs:
                continue
            stack = np.vstack(runs)
            if not np.isfinite(stack).any():
                continue
            mean = np.nanmean(stack, axis=0)
            std  = np.nanstd(stack, axis=0)
            ax.plot(evals, mean, label=label, color=color, linestyle=linestyle, linewidth=2)
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


def _plot_results(
    curves: dict[str, dict[str, list[np.ndarray]]],
    outfile: str,
    *,
    budget: int,
    n_seeds: int,
) -> None:
    evals = np.arange(1, budget + 1)
    n_fn = len(FUNCTION_ORDER)
    cols = min(4, max(1, n_fn))
    rows = max(1, (n_fn + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.8 * rows), squeeze=False)

    methods_styles = [
        ("SAC", "C0", "-"),
        ("PPO", "C1", "-"),
        ("SimpleGA", "C2", "--"),
        ("CMA_ES", "C3", "-."),
    ]

    for idx, fname in enumerate(FUNCTION_ORDER):
        ax = axes[idx // cols][idx % cols]
        for label, color, linestyle in methods_styles:
            if label not in curves or fname not in curves[label]:
                continue
            runs = curves[label][fname]
            if not runs:
                continue
            stack = np.vstack(runs)
            if not np.isfinite(stack).any():
                continue
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
            ax.plot(evals, mean, label=label, color=color, linestyle=linestyle, linewidth=2)
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
        f"Best-so-far vs evaluations (budget={budget}, RL/SGA/CMA: mean ± std over {n_seeds} seeds)"
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {outfile}")


def _print_table(
    final_vals: dict[str, dict[str, list[float]]],
    out_stream=sys.stdout,
) -> None:
    hdr = (
        f"{'Function':<16} | {'SAC':>22} | {'PPO':>22} | "
        f"{'SimpleGA':>22} | {'CMA_ES':>22}"
    )
    sep = "-" * len(hdr)
    print(sep, file=out_stream)
    print(hdr, file=out_stream)
    print(sep, file=out_stream)

    def fmt_runs(xs: list[float]) -> str:
        if not xs or (len(xs) == 1 and np.isnan(xs[0])):
            return "n/a"
        m, s = float(np.mean(xs)), float(np.std(xs))
        return f"{m:.4f} ± {s:.4f}"

    for fn in FUNCTION_ORDER:
        row_sac = final_vals.get("SAC", {}).get(fn, [float("nan")])
        row_ppo = final_vals.get("PPO", {}).get(fn, [float("nan")])
        row_sga = final_vals.get("SimpleGA", {}).get(fn, [float("nan")])
        row_cma = final_vals.get("CMA_ES", {}).get(fn, [float("nan")])
        print(
            f"{fn:<16} | {fmt_runs(row_sac):>22} | {fmt_runs(row_ppo):>22} | "
            f"{fmt_runs(row_sga):>22} | {fmt_runs(row_cma):>22}",
            file=out_stream,
        )
    print(sep, file=out_stream)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-sac",
        type=str,
        default=DEFAULT_CKPT_SAC,
        help="Path to SAC checkpoint (final_policy.pth or algo state dict)",
    )
    parser.add_argument(
        "--ckpt-ppo",
        type=str,
        default=DEFAULT_CKPT_PPO,
        help="Path to PPO checkpoint (final_policy.pth or algo state dict)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=N_SEEDS,
        help="Number of random seeds for RL and stochastic baselines",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=BUDGET,
        help="Evaluation budget per method / padded curve length",
    )
    args = parser.parse_args()
    budget = args.budget
    n_seeds = args.seeds

    sac_ckpt = Path(args.ckpt_sac).as_posix()
    ppo_ckpt = Path(args.ckpt_ppo).as_posix()

    base_seeds = [42 + k for k in range(n_seeds)]

    variants = [
        ("clean", BACKENDS_LIST_CLEAN, {}),
        ("noisy", BACKENDS_LIST,       NOISE_STD_MAP),
    ]

    for variant_name, bl, noise_map in variants:
        print(f"\n{'='*60}")
        print(f"  Variant: {variant_name.upper()}")
        print(f"{'='*60}\n")

        sac_cfg = config_sac(sac_ckpt, bl)
        ppo_cfg = config_ppo(ppo_ckpt, bl)

        curves: dict[str, dict[str, list[np.ndarray]]] = {
            "SAC": {fn: [] for fn in FUNCTION_ORDER},
            "PPO": {fn: [] for fn in FUNCTION_ORDER},
            "SimpleGA": {fn: [] for fn in FUNCTION_ORDER},
            "CMA_ES": {fn: [] for fn in FUNCTION_ORDER},
        }
        finals: dict[str, dict[str, list[float]]] = {
            "SAC": {fn: [] for fn in FUNCTION_ORDER},
            "PPO": {fn: [] for fn in FUNCTION_ORDER},
            "SimpleGA": {fn: [] for fn in FUNCTION_ORDER},
            "CMA_ES": {fn: [] for fn in FUNCTION_ORDER},
        }

        for seed in base_seeds:
            for fi, fname in enumerate(FUNCTION_ORDER):
                fn_noise = noise_map.get(fname, 0.0)

                print(f"[SAC] seed={seed} function={fname} ...")
                try:
                    y = run_rl_episode(sac_cfg, fi, seed, budget)
                except Exception as e:
                    print(f"  ERROR SAC {fname} seed {seed}: {e}", file=sys.stderr)
                    y = np.full(budget, np.nan)
                curves["SAC"][fname].append(y)
                finals["SAC"][fname].append(float(y[-1]))

                print(f"[PPO] seed={seed} function={fname} ...")
                try:
                    y_p = run_rl_episode(ppo_cfg, fi, seed, budget)
                except Exception as e:
                    print(f"  ERROR PPO {fname} seed {seed}: {e}", file=sys.stderr)
                    y_p = np.full(budget, np.nan)
                curves["PPO"][fname].append(y_p)
                finals["PPO"][fname].append(float(y_p[-1]))

                print(f"[SimpleGA] seed={seed} function={fname} ...")
                try:
                    y_g = run_baseline("SimpleGA", fname, seed, budget, noise_std=fn_noise)
                except Exception as e:
                    print(f"  ERROR SimpleGA {fname} seed {seed}: {e}", file=sys.stderr)
                    y_g = np.full(budget, np.nan)
                curves["SimpleGA"][fname].append(y_g)
                finals["SimpleGA"][fname].append(float(y_g[-1]))

                print(f"[CMA_ES] seed={seed} function={fname} ...")
                try:
                    y_c = run_baseline("CMA_ES", fname, seed, budget, noise_std=fn_noise)
                except Exception as e:
                    print(f"  ERROR CMA_ES {fname} seed {seed}: {e}", file=sys.stderr)
                    y_c = np.full(budget, np.nan)
                curves["CMA_ES"][fname].append(y_c)
                finals["CMA_ES"][fname].append(float(y_c[-1]))

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
            "checkpoints": {"sac": args.ckpt_sac, "ppo": args.ckpt_ppo},
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
