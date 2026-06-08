from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import get_args

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
for p in (ROOT, EXPERIMENTS):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch

from hpo_rl.backends.function import OptimizationBenchmarkBackend
from hpo_rl.controller import plot_and_save
from hpo_rl.controller.check import check
from hpo_rl.controller.controller import controller

from compare_sac_ppo_sga_cmaes import FUNCTION_ORDER, config_ppo, config_sac

warnings.filterwarnings("ignore")

BUDGET = 200
OUT_DIR = ROOT / "logs" / "plot_2d_sac_ppo_cmaes_sga"

ALL_2D_FUNCTIONS: list[str] = list(get_args(OptimizationBenchmarkBackend.FUNCTIONS))

# RL checkpoints were trained on FUNCTION_ORDER; index must match that list.
BACKENDS_LIST_RL = [
    {"name": "function", "function": fn, "dimensions": 2, "noise_std": 0.0}
    for fn in FUNCTION_ORDER
]

DEFAULT_CKPT_SAC = ROOT / "log/sac/20260509-144135/final_policy.pth"
DEFAULT_CKPT_PPO = ROOT / "log/ppo/20260509-200654/final_policy.pth"

BASELINE_METHODS = frozenset({"cma_es", "sga"})


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_rl_history(
    raw_config: dict,
    function_idx: int,
    seed: int,
) -> list:
    """Run RL inference on a single benchmark function and return history."""
    _seed_all(seed)
    cfg = deepcopy(raw_config)
    parsed = check(cfg)
    ctrl = controller(**parsed)
    if not getattr(ctrl, "load_loc", None):
        raise ValueError("RL config must set full_args.load_checkpoint.")
    if not getattr(ctrl, "_checkpoint_loaded", False):
        raise RuntimeError(
            f"Checkpoint was NOT loaded from {ctrl.load_loc!r}. "
            "Use forward slashes in the path (e.g. log/sac/.../final_policy.pth)."
        )
    if not hasattr(ctrl.backend, "set_active_backend"):
        raise RuntimeError("Expected SequentialBackend.")

    ctrl.backend.set_active_backend(int(function_idx), lock=True)
    ctrl.inference()
    return list(ctrl.return_history())


def run_baseline_history(
    algorithm_name: str,
    function_name: str,
    seed: int,
    budget: int,
) -> list:
    """Run CMA_ES or SimpleGA on a single 2D benchmark and return history."""
    _seed_all(seed)
    backend_cfg = {
        "name": "function",
        "function": function_name,
        "dimensions": 2,
        "noise_std": 0.0,
    }

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
    # check() uses symmetric bounds from its registry; use per-dimension backend bounds.
    benchmark = OptimizationBenchmarkBackend(
        function_name=function_name, dimensions=2, noise_std=0.0,
    )
    parsed["algorithm"]["params"]["dict_to_optimize"] = benchmark.hp_space

    ctrl = controller(**parsed)
    ctrl.inference()
    return list(ctrl.return_history())


def _validate_baseline_history(
    history: list,
    func_name: str,
    budget: int,
) -> None:
    """Ensure baseline runs produce exactly ``budget`` in-bounds evaluations."""
    backend = OptimizationBenchmarkBackend(
        function_name=func_name, dimensions=2, noise_std=0.0,
    )
    if len(history) != budget:
        raise ValueError(
            f"Expected {budget} evaluations, got {len(history)} for {func_name}"
        )
    for i, (cfg, _) in enumerate(history):
        for dim, (lo, hi) in enumerate(backend.bounds):
            key = f"x{dim}"
            val = float(cfg[key])
            if not lo <= val <= hi:
                raise ValueError(
                    f"Evaluation {i + 1}: {key}={val} outside [{lo}, {hi}]"
                )


def _plot_history(
    func_name: str,
    method_name: str,
    history: list,
    out_dir: str,
) -> None:
    backend = OptimizationBenchmarkBackend(
        function_name=func_name, dimensions=2, noise_std=0.0,
    )
    if backend.maximize:
        best_result = max(history, key=lambda x: x[1])
    else:
        best_result = min(history, key=lambda x: x[1])

    save_path = Path(out_dir) / func_name / method_name
    save_path.mkdir(parents=True, exist_ok=True)

    plotter = plot_and_save(
        history=history,
        best_result=best_result,
        save_path=save_path,
        backend=backend,
        experiment_number=func_name,
    )
    plotter.plot_3d(suffix=f"_{method_name}")
    plotter.plot_trajectory(suffix=f"_{method_name}")
    print(
        f"  {func_name}/{method_name}: OK  → {save_path} "
        f"({len(history)} points)"
    )


def process_function(
    func_name: str,
    function_idx: int,
    budget: int,
    seed: int,
    out_dir: str,
    sac_cfg: dict,
    ppo_cfg: dict,
) -> None:
    methods: list[tuple[str, list]] = []

    for method_name, runner in [
        ("sac", lambda: run_rl_history(sac_cfg, function_idx, seed)),
        ("ppo", lambda: run_rl_history(ppo_cfg, function_idx, seed)),
        ("cma_es", lambda: run_baseline_history("CMA_ES", func_name, seed, budget)),
        ("sga", lambda: run_baseline_history("SimpleGA", func_name, seed, budget)),
    ]:
        try:
            history = runner()
        except Exception as exc:
            print(f"  {func_name}/{method_name}: ОШИБКА — {exc}")
            continue

        if not history:
            print(f"  {func_name}/{method_name}: пустая история, пропуск")
            continue

        if method_name in BASELINE_METHODS:
            try:
                _validate_baseline_history(history, func_name, budget)
            except ValueError as exc:
                print(f"  {func_name}/{method_name}: ОШИБКА валидации — {exc}")
                continue

        methods.append((method_name, history))

    for method_name, history in methods:
        try:
            _plot_history(func_name, method_name, history, out_dir)
        except Exception as exc:
            print(f"  {func_name}/{method_name}: ОШИБКА при построении — {exc}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = argparse.ArgumentParser(
        description="Plot 2D benchmark landscapes: SAC, PPO, CMA-ES, SGA",
    )
    parser.add_argument("--budget", type=int, default=BUDGET,
                        help="Число пробных точек для baseline-алгоритмов")
    parser.add_argument("--seed", type=int, default=42, help="Семя ГСЧ")
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR),
                        help="Корневая папка для PNG/PGF (по умолчанию logs/plot_2d_sac_ppo_cmaes_sga)")
    parser.add_argument("--funcs", type=str, default="",
                        help="Через запятую: subset функций (по умолчанию — все из function.py)")
    parser.add_argument("--ckpt-sac", type=str, default=str(DEFAULT_CKPT_SAC),
                        help="Путь к чекпоинту SAC (final_policy.pth)")
    parser.add_argument("--ckpt-ppo", type=str, default=str(DEFAULT_CKPT_PPO),
                        help="Путь к чекпоинту PPO (final_policy.pth)")
    args = parser.parse_args()

    funcs = (
        [f.strip() for f in args.funcs.split(",") if f.strip()]
        if args.funcs
        else list(ALL_2D_FUNCTIONS)
    )
    unknown = [f for f in funcs if f not in ALL_2D_FUNCTIONS]
    if unknown:
        print(f"Неизвестные функции: {unknown}")
        print(f"Доступные: {ALL_2D_FUNCTIONS}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sac_ckpt = Path(args.ckpt_sac)
    if not sac_ckpt.is_absolute():
        sac_ckpt = ROOT / sac_ckpt
    ppo_ckpt = Path(args.ckpt_ppo)
    if not ppo_ckpt.is_absolute():
        ppo_ckpt = ROOT / ppo_ckpt
    sac_cfg = config_sac(sac_ckpt.as_posix(), BACKENDS_LIST_RL)
    ppo_cfg = config_ppo(ppo_ckpt.as_posix(), BACKENDS_LIST_RL)

    print(
        f"Функций: {len(funcs)}, budget={args.budget}, seed={args.seed}, "
        f"out={out_dir}\n"
        f"SAC checkpoint: {sac_ckpt}\n"
        f"PPO checkpoint: {ppo_ckpt}\n"
    )

    for func_name in funcs:
        function_idx = FUNCTION_ORDER.index(func_name)
        print(f"── {func_name} ──")
        try:
            process_function(
                func_name, function_idx, args.budget, args.seed, str(out_dir),
                sac_cfg, ppo_cfg,
            )
        except Exception as exc:
            print(f"  ОШИБКА: {exc}")

    print(f"\nГотово. Графики сохранены в {out_dir}/")


if __name__ == "__main__":
    main()
