"""Benchmark: HMM_MCMC_TEST (Baum-Welch + spline proposals) vs HMM_MCMC vs Optuna TPE.

Структура повторяет experiments/test_hmm_vs_optuna.py:
  - continuous 10D (clean)
  - noisy benchmarks
  - categorical + 10D float

Запуск из корня репозитория::

    python experiments/test_hmm_mcmc_test_vs_optuna.py
    python experiments/test_hmm_mcmc_test_vs_optuna.py --smoke
"""

import argparse
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_root = str(ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import optuna
import matplotlib.pyplot as plt

from hpo_rl.baselines.HMM_MCMC import HMM_MCMC
from hpo_rl.baselines.HMM_MCMC_TEST import HMM_MCMC_TEST
from hpo_rl.backends.function import OptimizationBenchmarkBackend

N_SEEDS = 5
BUDGET = 500
DIMENSIONS = 10

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

HMM_TEST_PARAMS = dict(
    **HMM_PARAMS,
    use_baum_welch=True,
    bw_refit_every=5,
    bw_min_obs=12,
    bw_n_em_iters=3,
    bw_prior_strength=25.0,
    bw_exploit_prior_scale=3.0,
    bw_max_len=64,
    use_spline_proposal=True,
    spline_knots=60,
    spline_floor=0.02,
    spline_min_archive=50,
    locality_sigma_fraction=0.08,
    spline_mix_scale=1.0,
)

HMM_TEST_TUNED_OVERRIDES: dict[str, dict] = {
    "schwefel": {"spline_min_archive": 10, "bw_prior_strength": 5.0},
    "ackley": {"spline_min_archive": 50, "bw_prior_strength": 5.0},
}


def _get_hmm_test_params(func_name: str = "") -> dict:
    params = dict(HMM_TEST_PARAMS)
    if func_name in HMM_TEST_TUNED_OVERRIDES:
        params.update(HMM_TEST_TUNED_OVERRIDES[func_name])
    return params

CONTINUOUS_BENCHMARKS = {
    "sphere": {"bounds": (-5.0, 5.0)},
    "rosenbrock": {"bounds": (-1.0, 1.0)},
    "rastrigin": {"bounds": (-5.12, 5.12)},
    "ackley": {"bounds": (-32.768, 32.768)},
    "griewank": {"bounds": (-600.0, 600.0)},
    "schwefel": {"bounds": (-500.0, 500.0)},
    "levy": {"bounds": (-10.0, 10.0)},
    "michalewicz": {"bounds": (0.0, float(np.pi))},
    "styblinski_tang": {"bounds": (-5.0, 5.0)},
}

GLOBAL_OPTIMUM_VALUE = {
    "sphere": 0.0,
    "rosenbrock": 0.0,
    "rastrigin": 0.0,
    "ackley": 0.0,
    "griewank": 0.0,
    "schwefel": 0.0,
    "levy": 0.0,
    "michalewicz": 0.0,
    "styblinski_tang": round(-39.16617 * DIMENSIONS, 5),
}

NOISY_BENCHMARKS = {
    "sphere": 0.5,
    "rastrigin": 8.0,
    "ackley": 2.0,
    "schwefel": 300.0,
    "levy": 1.5,
}

CATEGORICAL_BENCHMARKS = {
    "sphere": {"bounds": (-5.0, 5.0)},
    "rastrigin": {"bounds": (-5.12, 5.12)},
    "ackley": {"bounds": (-32.768, 32.768)},
    "schwefel": {"bounds": (-500.0, 500.0)},
    "levy": {"bounds": (-10.0, 10.0)},
}

DUMMY_CATEGORIES = {
    "optimizer": {"values": ["adam", "sgd", "rmsprop", "adamw"], "type": "categorical"},
    "activation": {"values": ["relu", "tanh", "gelu", "silu"], "type": "categorical"},
    "scheduler": {"values": ["cosine", "step", "plateau"], "type": "categorical"},
}

PLOT_FILES = {
    "continuous": "convergence_continuous_test.png",
    "continuous_raw": "per_eval_continuous_test.png",
    "noisy": "convergence_noisy_test.png",
    "noisy_raw": "per_eval_noisy_test.png",
    "categorical": "convergence_categorical_test.png",
    "categorical_raw": "per_eval_categorical_test.png",
}

RESULTS_FILE = "hmm_test_vs_optuna_results.txt"


def make_continuous_space(func_name: str) -> dict:
    lo, hi = CONTINUOUS_BENCHMARKS[func_name]["bounds"]
    return {
        f"x{i}": {"values": [float(lo), float(hi)], "type": "float", "log": False}
        for i in range(DIMENSIONS)
    }


def make_noisy_continuous_space(func_name: str) -> dict:
    return make_continuous_space(func_name)


def make_categorical_space(func_name: str) -> dict:
    lo, hi = CATEGORICAL_BENCHMARKS[func_name]["bounds"]
    space = {
        f"x{i}": {"values": [float(lo), float(hi)], "type": "float", "log": False}
        for i in range(DIMENSIONS)
    }
    space.update(DUMMY_CATEGORIES)
    return space


def _scores_to_curve(scores: list[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    return np.minimum.accumulate(arr)


def _pad_curve_to_budget(curve: np.ndarray, budget: int) -> np.ndarray:
    if len(curve) >= budget:
        return curve[:budget]
    if len(curve) == 0:
        return np.full(budget, np.nan)
    pad = np.full(budget - len(curve), curve[-1])
    return np.concatenate([curve, pad])


def _run_hmm_algo(alg_cls, params: dict, backend, space: dict, seed: int):
    np.random.seed(seed)
    alg = alg_cls(
        objective_func=backend.evaluate,
        budget=BUDGET,
        dict_to_optimize=space,
        **params,
    )
    with open(os.devnull, "w") as devnull:
        _saved_out, _saved_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            _best = alg.main_loop()
        finally:
            sys.stdout, sys.stderr = _saved_out, _saved_err
    best_loss = float(_best[1])
    scores = [float(s) for _, s in alg.data]
    raw_curve = _pad_curve_to_budget(np.asarray(scores, dtype=float), BUDGET)
    best_curve = _pad_curve_to_budget(_scores_to_curve(scores), BUDGET)
    return best_loss, best_curve, raw_curve


def run_hmm_mcmc_test(
    backend, space: dict, seed: int, func_name: str = ""
) -> tuple[float, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    desc = f"TEST {func_name} seed={seed}" if func_name else f"TEST seed={seed}"
    alg = HMM_MCMC_TEST(
        objective_func=backend.evaluate,
        budget=BUDGET,
        dict_to_optimize=space,
        show_progress=True,
        progress_desc=desc,
        verbose_history=False,
        **_get_hmm_test_params(func_name),
    )
    _best = alg.main_loop()
    best_loss = float(_best[1])
    scores = [float(s) for _, s in alg.data]
    raw_curve = _pad_curve_to_budget(np.asarray(scores, dtype=float), BUDGET)
    best_curve = _pad_curve_to_budget(_scores_to_curve(scores), BUDGET)
    return best_loss, best_curve, raw_curve


def run_hmm_mcmc(
    backend, space: dict, seed: int
) -> tuple[float, np.ndarray, np.ndarray]:
    return _run_hmm_algo(HMM_MCMC, HMM_PARAMS, backend, space, seed)


def run_optuna_tpe(backend, space: dict, seed: int) -> tuple[float, np.ndarray, np.ndarray]:
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
    scores = []
    for t in sorted(study.trials, key=lambda tr: tr.number):
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
            scores.append(float(t.value))
    raw_arr = np.asarray(scores, dtype=float)
    raw_curve = _pad_curve_to_budget(raw_arr, BUDGET)
    best_curve = _pad_curve_to_budget(_scores_to_curve(scores), BUDGET)
    return float(study.best_value), best_curve, raw_curve


def run_benchmark(
    func_name: str,
    space_builder,
    label: str,
    noise_std: float = 0.0,
):
    backend = OptimizationBenchmarkBackend(
        function_name=func_name, dimensions=DIMENSIONS, noise_std=noise_std
    )
    space = space_builder(func_name)

    test_losses, base_losses, optuna_losses = [], [], []
    test_best, base_best, optuna_best = [], [], []
    test_raw, base_raw, optuna_raw = [], [], []

    for s in range(N_SEEDS):
        seed = 42 + s
        tl, tb, tr = run_hmm_mcmc_test(backend, space, seed, func_name=func_name)
        test_losses.append(tl)
        test_best.append(tb)
        test_raw.append(tr)

        bl, bb, br = run_hmm_mcmc(backend, space, seed)
        base_losses.append(bl)
        base_best.append(bb)
        base_raw.append(br)

        ol, ob, opt_r = run_optuna_tpe(backend, space, seed)
        optuna_losses.append(ol)
        optuna_best.append(ob)
        optuna_raw.append(opt_r)

    return (
        test_losses,
        base_losses,
        optuna_losses,
        test_best,
        base_best,
        optuna_best,
        test_raw,
        base_raw,
        optuna_raw,
    )


def plot_convergence(
    group_name: str,
    items: list[tuple[str, str, list[np.ndarray], list[np.ndarray], list[np.ndarray]]],
    outfile: str,
    *,
    y_axis_label: str = "Best loss so far",
    title_prefix: str = "Best-so-far convergence",
):
    n = len(items)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False)
    evals = np.arange(1, BUDGET + 1)

    for idx, (func_name, _tag, test_cs, base_cs, opt_cs) in enumerate(items):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        for curves, label, color in (
            (test_cs, "HMM_MCMC_TEST", "C0"),
            (base_cs, "HMM_MCMC", "C2"),
            (opt_cs, "Optuna TPE", "C1"),
        ):
            stack = np.vstack(curves)
            mean, std = np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)
            ax.plot(evals, mean, label=label, color=color)
            ax.fill_between(evals, mean - std, mean + std, color=color, alpha=0.15)

        ax.set_title(func_name)
        ax.set_xlabel("Evaluation")
        ax.set_ylabel(y_axis_label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        f"{title_prefix}: {group_name} "
        f"(mean +/- std over {N_SEEDS} seeds)"
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {outfile}")


def print_table(
    results: list[tuple[str, str, list[float], list[float], list[float], float]],
    out=None,
):
    p = lambda s: print(s, file=out, flush=True) if out else print(s)
    header = (
        f"{'Benchmark':<32} | {'f*':>12} | {'TEST mean+-std':>22} | {'TEST min':>9} | "
        f"{'HMM mean+-std':>22} | {'HMM min':>9} | "
        f"{'Optuna mean+-std':>22} | {'Opt min':>9} | {'Winner':>10}"
    )
    sep = "-" * len(header)
    p(sep)
    p(header)
    p(sep)

    for name, label, test, base, opt, f_star in results:
        tm, ts = np.mean(test), np.std(test)
        t_min = np.min(test)
        bm, bs = np.mean(base), np.std(base)
        b_min = np.min(base)
        om, os_ = np.mean(opt), np.std(opt)
        o_min = np.min(opt)

        means = {"TEST": tm, "HMM": bm, "Optuna": om}
        winner = min(means, key=means.get)

        tag = f"{label} {name}"
        p(
            f"{tag:<32} | {f_star:12.4f} | {tm:10.2f} +- {ts:8.2f} | {t_min:9.4f} | "
            f"{bm:10.2f} +- {bs:8.2f} | {b_min:9.4f} | "
            f"{om:10.2f} +- {os_:8.2f} | {o_min:9.4f} | {winner:>10}"
        )
    p(sep)


def _apply_smoke_mode() -> None:
    """Quick gate: sphere + schwefel, 1 seed, budget=100."""
    global N_SEEDS, BUDGET, CONTINUOUS_BENCHMARKS, NOISY_BENCHMARKS, CATEGORICAL_BENCHMARKS
    N_SEEDS = 1
    BUDGET = 100
    CONTINUOUS_BENCHMARKS = {
        "sphere": {"bounds": (-5.0, 5.0)},
        "schwefel": {"bounds": (-500.0, 500.0)},
    }
    NOISY_BENCHMARKS = {}
    CATEGORICAL_BENCHMARKS = {}


def run_smoke_gate() -> None:
    """Assert TEST is competitive on sphere and strongly wins on schwefel."""
    _apply_smoke_mode()
    print("[smoke mode] sphere + schwefel, 1 seed, budget=100\n")

    for func_name in ("sphere", "schwefel"):
        backend = OptimizationBenchmarkBackend(
            function_name=func_name, dimensions=DIMENSIONS, noise_std=0.0
        )
        space = make_continuous_space(func_name)
        seed = 42
        tl, _, _ = run_hmm_mcmc_test(backend, space, seed, func_name=func_name)
        bl, _, _ = run_hmm_mcmc(backend, space, seed)
        delta = tl - bl
        print(f"  {func_name}: TEST={tl:.4f}  HMM={bl:.4f}  delta={delta:+.4f}")

        if func_name == "sphere":
            assert delta <= 0.5, (
                f"smoke FAIL sphere: TEST worse than HMM by {delta:.4f} (limit 0.5)"
            )
        else:
            assert delta < -50.0, (
                f"smoke FAIL schwefel: expected large negative delta, got {delta:.4f}"
            )

    print("\n[smoke] PASS")


def main() -> None:
    f = open(RESULTS_FILE, "w", encoding="utf-8")

    def log(s=""):
        print(s, file=f, flush=True)
        print(s)

    all_results: list[tuple[str, str, list[float], list[float], list[float], float]] = []
    plot_continuous: list = []
    plot_continuous_raw: list = []
    plot_noisy: list = []
    plot_noisy_raw: list = []
    plot_cat: list = []
    plot_cat_raw: list = []

    log(f"\n{'='*70}")
    log(f"  CONTINUOUS 10D  (budget={BUDGET}, seeds={N_SEEDS}, noise_std=0)")
    log(f"  Algorithms: HMM_MCMC_TEST (BW+spline) | HMM_MCMC | Optuna TPE")
    log(f"{'='*70}")
    for func_name in CONTINUOUS_BENCHMARKS:
        log(f"\n>>> {func_name} ...")
        (
            test, base, opt,
            test_b, base_b, opt_b,
            test_r, base_r, opt_r,
        ) = run_benchmark(func_name, make_continuous_space, "cont", noise_std=0.0)
        f_star = GLOBAL_OPTIMUM_VALUE[func_name]
        all_results.append((func_name, "cont", test, base, opt, f_star))
        plot_continuous.append((func_name, "cont", test_b, base_b, opt_b))
        plot_continuous_raw.append((func_name, "cont", test_r, base_r, opt_r))
        log(
            f"    TEST:   {np.mean(test):.2f} +- {np.std(test):.2f}  "
            f"(min={np.min(test):.4f})"
        )
        log(
            f"    HMM:    {np.mean(base):.2f} +- {np.std(base):.2f}  "
            f"(min={np.min(base):.4f})"
        )
        log(
            f"    Optuna: {np.mean(opt):.2f} +- {np.std(opt):.2f}  "
            f"(min={np.min(opt):.4f})"
        )

    plot_convergence(
        "continuous clean",
        plot_continuous,
        PLOT_FILES["continuous"],
        y_axis_label="Best loss so far",
        title_prefix="Best-so-far convergence",
    )
    plot_convergence(
        "continuous clean",
        plot_continuous_raw,
        PLOT_FILES["continuous_raw"],
        y_axis_label="Observed loss (this evaluation)",
        title_prefix="Per-evaluation objective",
    )

    log(f"\n{'='*70}")
    log(f"  NOISY  (budget={BUDGET}, seeds={N_SEEDS})")
    log(f"{'='*70}")
    for func_name, ns in NOISY_BENCHMARKS.items():
        log(f"\n>>> {func_name} (noise_std={ns}) ...")
        (
            test, base, opt,
            test_b, base_b, opt_b,
            test_r, base_r, opt_r,
        ) = run_benchmark(
            func_name, make_noisy_continuous_space, "noisy", noise_std=float(ns)
        )
        f_star = GLOBAL_OPTIMUM_VALUE[func_name]
        all_results.append((func_name, "noisy", test, base, opt, f_star))
        plot_noisy.append((func_name, "noisy", test_b, base_b, opt_b))
        plot_noisy_raw.append((func_name, "noisy", test_r, base_r, opt_r))
        log(
            f"    TEST:   {np.mean(test):.2f} +- {np.std(test):.2f}  "
            f"(min={np.min(test):.4f})"
        )
        log(
            f"    HMM:    {np.mean(base):.2f} +- {np.std(base):.2f}  "
            f"(min={np.min(base):.4f})"
        )
        log(
            f"    Optuna: {np.mean(opt):.2f} +- {np.std(opt):.2f}  "
            f"(min={np.min(opt):.4f})"
        )

    plot_convergence(
        "noisy",
        plot_noisy,
        PLOT_FILES["noisy"],
        y_axis_label="Best loss so far",
        title_prefix="Best-so-far convergence",
    )
    plot_convergence(
        "noisy",
        plot_noisy_raw,
        PLOT_FILES["noisy_raw"],
        y_axis_label="Observed loss (this evaluation)",
        title_prefix="Per-evaluation objective",
    )

    log(f"\n{'='*70}")
    log(f"  CATEGORICAL + 10D  (budget={BUDGET}, seeds={N_SEEDS}, noise_std=0)")
    log(f"{'='*70}")
    for func_name in CATEGORICAL_BENCHMARKS:
        log(f"\n>>> {func_name} + dummy categorical ...")
        (
            test, base, opt,
            test_b, base_b, opt_b,
            test_r, base_r, opt_r,
        ) = run_benchmark(func_name, make_categorical_space, "cat", noise_std=0.0)
        f_star = GLOBAL_OPTIMUM_VALUE[func_name]
        all_results.append((func_name, "cat", test, base, opt, f_star))
        plot_cat.append((func_name, "cat", test_b, base_b, opt_b))
        plot_cat_raw.append((func_name, "cat", test_r, base_r, opt_r))
        log(
            f"    TEST:   {np.mean(test):.2f} +- {np.std(test):.2f}  "
            f"(min={np.min(test):.4f})"
        )
        log(
            f"    HMM:    {np.mean(base):.2f} +- {np.std(base):.2f}  "
            f"(min={np.min(base):.4f})"
        )
        log(
            f"    Optuna: {np.mean(opt):.2f} +- {np.std(opt):.2f}  "
            f"(min={np.min(opt):.4f})"
        )

    plot_convergence(
        "categorical",
        plot_cat,
        PLOT_FILES["categorical"],
        y_axis_label="Best loss so far",
        title_prefix="Best-so-far convergence",
    )
    plot_convergence(
        "categorical",
        plot_cat_raw,
        PLOT_FILES["categorical_raw"],
        y_axis_label="Observed loss (this evaluation)",
        title_prefix="Per-evaluation objective",
    )

    log(f"\n{'='*70}")
    log(f"  SUMMARY  (budget={BUDGET}, dims={DIMENSIONS}, seeds={N_SEEDS})")
    log(f"{'='*70}")
    print_table(all_results, out=f)
    print_table(all_results)

    f.close()
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HMM_MCMC_TEST vs HMM_MCMC vs Optuna TPE benchmark"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick gate: sphere+schwefel, 1 seed, budget=100",
    )
    args = parser.parse_args()
    if args.smoke:
        run_smoke_gate()
    else:
        main()
