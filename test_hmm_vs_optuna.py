"""
Сравнение HMM_MCMC vs Optuna TPE на 10-мерных бенчмарках.

Тесты:
  1. Непрерывные 10D функции (schwefel, rastrigin, ackley, levy, styblinski_tang)
  2. Функции с фиктивными категориальными параметрами (schwefel, rastrigin)

Запуск:
    python test_hmm_vs_optuna.py
"""

import numpy as np
import sys
import os
import optuna

from hpo_rl.baselines.HMM_MCMC import HMM_MCMC
from hpo_rl.backends.function import OptimizationBenchmarkBackend

# ── Общие настройки ──────────────────────────────────────────────
N_SEEDS = 5
BUDGET = 100
DIMENSIONS = 10

# Гиперпараметры HMM_MCMC (из run_exp_HMM_MCMC.py)
HMM_PARAMS = dict(
    n_init=32,
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

# ── Функции-бенчмарки для 10D ───────────────────────────────────
# noise_std ≈ 20% от типичных значений функции при оптимизации
CONTINUOUS_BENCHMARKS = {
    "schwefel":        {"bounds": (-500.0, 500.0),   "noise_std": 300},
    "rastrigin":       {"bounds": (-5.12, 5.12),     "noise_std": 8},
    "ackley":          {"bounds": (-32.768, 32.768),  "noise_std": 2},
    "levy":            {"bounds": (-10.0, 10.0),      "noise_std": 1.5},
    "styblinski_tang": {"bounds": (-5.0, 5.0),        "noise_std": 65},
}

CATEGORICAL_BENCHMARKS = {
    "schwefel":  {"bounds": (-500.0, 500.0),  "noise_std": 300},
    "rastrigin": {"bounds": (-5.12, 5.12),    "noise_std": 8},
}

# Фиктивные категориальные параметры (не влияют на loss)
DUMMY_CATEGORIES = {
    "optimizer":   {"values": ["adam", "sgd", "rmsprop", "adamw"], "type": "categorical"},
    "activation":  {"values": ["relu", "tanh", "gelu", "silu"],   "type": "categorical"},
    "scheduler":   {"values": ["cosine", "step", "plateau"],      "type": "categorical"},
}


# ── Утилиты ──────────────────────────────────────────────────────
def make_continuous_space(func_name: str) -> dict:
    lo, hi = CONTINUOUS_BENCHMARKS[func_name]["bounds"]
    return {
        f"x{i}": {"values": [lo, hi], "type": "float", "log": False}
        for i in range(DIMENSIONS)
    }


def make_categorical_space(func_name: str) -> dict:
    lo, hi = CATEGORICAL_BENCHMARKS[func_name]["bounds"]
    space = {
        f"x{i}": {"values": [lo, hi], "type": "float", "log": False}
        for i in range(DIMENSIONS)
    }
    space.update(DUMMY_CATEGORIES)
    return space


def run_hmm_mcmc(backend, space: dict, seed: int) -> float:
    np.random.seed(seed)
    alg = HMM_MCMC(
        objective_func=backend.evaluate,
        budget=BUDGET,
        dict_to_optimize=space,
        **HMM_PARAMS,
    )
    # Подавляем verbose-вывод HMM-таблицы и tqdm
    with open(os.devnull, "w") as devnull:
        _saved_out, _saved_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            _best_config, best_loss = alg.main_loop()
        finally:
            sys.stdout, sys.stderr = _saved_out, _saved_err
    return best_loss


def run_optuna_tpe(backend, space: dict, seed: int) -> float:
    """Запускает Optuna TPE с тем же бюджетом."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        config = {}
        for name, info in space.items():
            if info["type"] == "float":
                lo, hi = info["values"]
                config[name] = trial.suggest_float(name, lo, hi)
            elif info["type"] == "categorical":
                config[name] = trial.suggest_categorical(name, info["values"])
            elif info["type"] == "int":
                lo, hi = info["values"][0], info["values"][-1]
                config[name] = trial.suggest_int(name, lo, hi)
        return backend.evaluate(config)

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=32)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=BUDGET, show_progress_bar=False)
    return study.best_value


# ── Основной цикл ───────────────────────────────────────────────
def run_benchmark(func_name: str, space_builder, label: str, noise_std: float = 0.0):
    backend = OptimizationBenchmarkBackend(
        function_name=func_name, dimensions=DIMENSIONS, noise_std=noise_std
    )
    space = space_builder(func_name)

    hmm_losses = []
    optuna_losses = []

    for s in range(N_SEEDS):
        seed = 42 + s
        # HMM_MCMC
        hl = run_hmm_mcmc(backend, space, seed)
        hmm_losses.append(hl)
        # Optuna TPE
        ol = run_optuna_tpe(backend, space, seed)
        optuna_losses.append(ol)

    return hmm_losses, optuna_losses


def print_table(results: list[tuple[str, str, list[float], list[float]]], out=None):
    p = lambda s: print(s, file=out, flush=True) if out else print(s)
    header = f"{'Benchmark':<30} | {'HMM_MCMC mean+/-std':>22} | {'HMM min':>8} | {'Optuna TPE mean+/-std':>22} | {'Opt min':>8} | {'delta%':>8}"
    sep = "-" * len(header)
    p(sep)
    p(header)
    p(sep)

    for name, label, hmm, opt in results:
        hm, hs = np.mean(hmm), np.std(hmm)
        h_min = np.min(hmm)
        om, os_ = np.mean(opt), np.std(opt)
        o_min = np.min(opt)
        delta = (hm - om) / (om + 1e-30) * 100
        tag = f"{label} {name}"
        p(f"{tag:<30} | {hm:10.2f} +/- {hs:8.2f} | {h_min:8.2f} | {om:10.2f} +/- {os_:8.2f} | {o_min:8.2f} | {delta:+7.1f}%")
    p(sep)


RESULTS_FILE = "hmm_vs_optuna_results.txt"


if __name__ == "__main__":
    f = open(RESULTS_FILE, "w", encoding="utf-8")
    def log(s=""):
        print(s, file=f, flush=True)
        print(s)

    all_results = []

    # 1) Continuous 10D
    log(f"\n{'='*70}")
    log(f"  CONTINUOUS 10D  (budget={BUDGET}, seeds={N_SEEDS})")
    log(f"{'='*70}")
    for func_name, binfo in CONTINUOUS_BENCHMARKS.items():
        ns = binfo["noise_std"]
        log(f"\n>>> {func_name} (noise_std={ns}) ...")
        hmm, opt = run_benchmark(func_name, make_continuous_space, "cont", noise_std=ns)
        all_results.append((func_name, "cont", hmm, opt))
        log(f"    HMM_MCMC: {np.mean(hmm):.2f} +/- {np.std(hmm):.2f}  (min={np.min(hmm):.2f})")
        log(f"    Optuna:   {np.mean(opt):.2f} +/- {np.std(opt):.2f}  (min={np.min(opt):.2f})")

    # 2) Categorical
    log(f"\n{'='*70}")
    log(f"  CATEGORICAL 10D  (budget={BUDGET}, seeds={N_SEEDS})")
    log(f"{'='*70}")
    for func_name, binfo in CATEGORICAL_BENCHMARKS.items():
        ns = binfo["noise_std"]
        log(f"\n>>> {func_name} + categorical (noise_std={ns}) ...")
        hmm, opt = run_benchmark(func_name, make_categorical_space, "cat", noise_std=ns)
        all_results.append((func_name, "cat", hmm, opt))
        log(f"    HMM_MCMC: {np.mean(hmm):.2f} +/- {np.std(hmm):.2f}  (min={np.min(hmm):.2f})")
        log(f"    Optuna:   {np.mean(opt):.2f} +/- {np.std(opt):.2f}  (min={np.min(opt):.2f})")

    # 3) Summary table
    log(f"\n{'='*70}")
    log(f"  SUMMARY  (budget={BUDGET}, dims={DIMENSIONS}, seeds={N_SEEDS})")
    log(f"{'='*70}")
    print_table(all_results, out=f)
    print_table(all_results)

    f.close()
    log = print
    log(f"\nResults saved to {RESULTS_FILE}")
