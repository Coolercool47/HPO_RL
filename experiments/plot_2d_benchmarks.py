"""
Визуализация ландшафта всех 2D функций из OptimizationBenchmarkBackend:
сравнение TPE (Optuna) и HMM_MCMC (FMP-MCMC).

Использует plot_and_save из hpo_rl.controller:
  - plot_3d()        — контурная карта + 3D поверхность + траектория точек
  - plot_trajectory() — кривая сходимости best-so-far

Для каждой функции и каждого метода создаётся подпапка:
  out_dir/<func_name>/tpe/   — графики TPE
  out_dir/<func_name>/hmm/   — графики HMM-MCMC

Запуск:
    python plot_2d_benchmarks.py
    python plot_2d_benchmarks.py --budget 150 --seeds 1 --out_dir plots_2d
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna

from hpo_rl.backends.function import OptimizationBenchmarkBackend
from hpo_rl.baselines.HMM_MCMC import HMM_MCMC
from hpo_rl.controller import plot_and_save

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── Параметры по умолчанию ─────────────────────────────────────────────────
BUDGET = 400
N_SEEDS = 1
OUT_DIR = "plots_2d"

HMM_PARAMS = dict(
    n_init=20,
    n_chains=1,
    orchestrate_every=1000,
    T_mcmc=0.01,
    sigma_fraction=0.0055,
    wide_sigma_fraction=0.5,
    temperature=0.3,
    hmm_window=4,
    hmm_obs_epsilon=1e-8,
    hmm_lambda_noise=0.01,
    clone_noise=0.05,
    burnin_fraction=0.0,
    p_cat_step=0.0,
    kde_tau=0.05,
    anneal_T=True,
)

# ── Все 2D функции из function.py ──────────────────────────────────────────
ALL_2D_FUNCTIONS: list[str] = [
    # 2D-only
    "booth", "beale", "goldstein_price", "bukin_n6",
    "cross_in_tray", "drop_wave", "eggholder", "holder_table",
    "schaffer_n2", "schaffer_n4", "shubert", "dejong_n5",
    "easom", "levy_n13", "langermann",
    # N-D в 2D-режиме
    "sphere", "rosenbrock", "rastrigin", "ackley",
    "griewank", "schwefel", "levy", "michalewicz", "styblinski_tang",
]


# ── Запуск методов ─────────────────────────────────────────────────────────
def _make_space(backend: OptimizationBenchmarkBackend) -> dict:
    return {
        f"x{i}": {"values": [float(backend.bounds[i][0]), float(backend.bounds[i][1])], "type": "float"}
        for i in range(2)
    }


def run_tpe(backend: OptimizationBenchmarkBackend, seed: int, budget: int) -> list:
    """Запускает Optuna TPE; возвращает history в формате [(config, score), ...]."""
    space = _make_space(backend)

    def objective(trial: optuna.Trial) -> float:
        cfg = {name: trial.suggest_float(name, *info["values"]) for name, info in space.items()}
        return backend.evaluate(cfg)

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=max(10, budget // 10))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)

    history = []
    for t in sorted(study.trials, key=lambda tr: tr.number):
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
            history.append((t.params, float(t.value)))
    return history


def run_hmm(backend: OptimizationBenchmarkBackend, seed: int, budget: int) -> list:
    """Запускает HMM_MCMC; возвращает history в формате [(config, score), ...]."""
    np.random.seed(seed)
    space = _make_space(backend)
    alg = HMM_MCMC(
        objective_func=backend.evaluate,
        budget=budget,
        dict_to_optimize=space,
        **HMM_PARAMS,
    )
    with open(os.devnull, "w") as devnull:
        saved_out, saved_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            alg.main_loop()
        finally:
            sys.stdout, sys.stderr = saved_out, saved_err
    return [(cfg, float(val)) for cfg, val in alg.data]


# ── Основная логика ────────────────────────────────────────────────────────
def process_function(func_name: str, budget: int, seed: int, out_dir: str) -> None:
    """Запускает оба метода для одной функции и сохраняет графики через plot_and_save."""
    backend = OptimizationBenchmarkBackend(function_name=func_name, dimensions=2, noise_std=0.0)

    for method_name, history in [
        ("tpe", run_tpe(backend, seed, budget)),
        ("hmm", run_hmm(backend, seed, budget)),
    ]:
        if not history:
            print(f"  {func_name}/{method_name}: пустая история, пропуск")
            continue

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
        print(f"  {func_name}/{method_name}: OK  → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot 2D benchmark landscapes: TPE vs HMM-MCMC")
    parser.add_argument("--budget", type=int, default=BUDGET, help="Число пробных точек")
    parser.add_argument("--seed",   type=int, default=42,     help="Семя ГСЧ")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR, help="Корневая папка для PNG/PGF")
    parser.add_argument("--funcs",  type=str, default="",
                        help="Через запятую: subset функций (по умолчанию — все 24)")
    args = parser.parse_args()

    funcs = [f.strip() for f in args.funcs.split(",") if f.strip()] if args.funcs else ALL_2D_FUNCTIONS
    print(f"Функций: {len(funcs)}, budget={args.budget}, seed={args.seed}, out={args.out_dir}\n")

    for func_name in funcs:
        print(f"── {func_name} ──")
        try:
            process_function(func_name, args.budget, args.seed, args.out_dir)
        except Exception as exc:
            print(f"  ОШИБКА: {exc}")

    print(f"\nГотово. Графики сохранены в ./{args.out_dir}/")


if __name__ == "__main__":
    main()
