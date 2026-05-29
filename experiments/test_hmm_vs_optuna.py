import numpy as np
import sys
import os
import optuna
import matplotlib.pyplot as plt

from hpo_rl.baselines.HMM_MCMC import HMM_MCMC
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
    "continuous": "convergence_continuous.png",
    "continuous_raw": "per_eval_continuous.png",
    "noisy": "convergence_noisy.png",
    "noisy_raw": "per_eval_noisy.png",
    "categorical": "convergence_categorical.png",
    "categorical_raw": "per_eval_categorical.png",
}

RESULTS_FILE = "hmm_vs_optuna_results.txt"


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


def run_hmm_mcmc(
    backend, space: dict, seed: int
) -> tuple[float, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    alg = HMM_MCMC(
        objective_func=backend.evaluate,
        budget=BUDGET,
        dict_to_optimize=space,
        **HMM_PARAMS,
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

    hmm_losses = []
    optuna_losses = []
    hmm_best: list[np.ndarray] = []
    optuna_best: list[np.ndarray] = []
    hmm_raw: list[np.ndarray] = []
    optuna_raw: list[np.ndarray] = []

    for s in range(N_SEEDS):
        seed = 42 + s
        hl, hb, hr = run_hmm_mcmc(backend, space, seed)
        hmm_losses.append(hl)
        hmm_best.append(hb)
        hmm_raw.append(hr)
        ol, ob, opt_r = run_optuna_tpe(backend, space, seed)
        optuna_losses.append(ol)
        optuna_best.append(ob)
        optuna_raw.append(opt_r)

    return (
        hmm_losses,
        optuna_losses,
        hmm_best,
        optuna_best,
        hmm_raw,
        optuna_raw,
    )


def plot_convergence(
    group_name: str,
    items: list[tuple[str, str, list[np.ndarray], list[np.ndarray]]],
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

    for idx, (func_name, _tag, hmm_cs, opt_cs) in enumerate(items):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        hmm_stack = np.vstack(hmm_cs)
        opt_stack = np.vstack(opt_cs)
        hmm_m, hmm_s = np.nanmean(hmm_stack, axis=0), np.nanstd(hmm_stack, axis=0)
        opt_m, opt_s = np.nanmean(opt_stack, axis=0), np.nanstd(opt_stack, axis=0)

        ax.plot(evals, hmm_m, label="HMM_MCMC", color="C0")
        ax.fill_between(evals, hmm_m - hmm_s, hmm_m + hmm_s, color="C0", alpha=0.2)
        ax.plot(evals, opt_m, label="Optuna TPE", color="C1")
        ax.fill_between(evals, opt_m - opt_s, opt_m + opt_s, color="C1", alpha=0.2)
        ax.set_title(func_name)
        ax.set_xlabel("Evaluation")
        ax.set_ylabel(y_axis_label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        f"{title_prefix}: {group_name} "
        f"(mean ± std over {N_SEEDS} seeds)"
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {outfile}")


def print_table(
    results: list[tuple[str, str, list[float], list[float], float]],
    out=None,
):
    p = lambda s: print(s, file=out, flush=True) if out else print(s)
    header = (
        f"{'Benchmark':<32} | {'f*':>12} | {'HMM mean±std':>22} | {'HMM min':>9} | "
        f"{'Optuna mean±std':>22} | {'Opt min':>9} | {'Winner':>8} | {'Δ%':>8}"
    )
    sep = "-" * len(header)
    p(sep)
    p(header)
    p(sep)

    for name, label, hmm, opt, f_star in results:
        hm, hs = np.mean(hmm), np.std(hmm)
        h_min = np.min(hmm)
        om, os_ = np.mean(opt), np.std(opt)
        o_min = np.min(opt)
        delta = (hm - om) / (abs(om) + 1e-30) * 100
        if hm < om:
            winner = "HMM"
        elif om < hm:
            winner = "Optuna"
        else:
            winner = "tie"
        tag = f"{label} {name}"
        p(
            f"{tag:<32} | {f_star:12.4f} | {hm:10.2f} ± {hs:8.2f} | {h_min:9.4f} | "
            f"{om:10.2f} ± {os_:8.2f} | {o_min:9.4f} | {winner:>8} | {delta:+7.1f}%"
        )
    p(sep)


if __name__ == "__main__":
    f = open(RESULTS_FILE, "w", encoding="utf-8")

    def log(s=""):
        print(s, file=f, flush=True)
        print(s)

    all_results: list[tuple[str, str, list[float], list[float], float]] = []
    plot_continuous: list[tuple[str, str, list[np.ndarray], list[np.ndarray]]] = []
    plot_continuous_raw: list[tuple[str, str, list[np.ndarray], list[np.ndarray]]] = []
    plot_noisy: list[tuple[str, str, list[np.ndarray], list[np.ndarray]]] = []
    plot_noisy_raw: list[tuple[str, str, list[np.ndarray], list[np.ndarray]]] = []
    plot_cat: list[tuple[str, str, list[np.ndarray], list[np.ndarray]]] = []
    plot_cat_raw: list[tuple[str, str, list[np.ndarray], list[np.ndarray]]] = []

    log(f"\n{'='*70}")
    log(f"  CONTINUOUS 10D  (budget={BUDGET}, seeds={N_SEEDS}, noise_std=0)")
    log(f"{'='*70}")
    for func_name in CONTINUOUS_BENCHMARKS:
        log(f"\n>>> {func_name} ...")
        hmm, opt, hb, ob, hr, o_r = run_benchmark(
            func_name, make_continuous_space, "cont", noise_std=0.0
        )
        f_star = GLOBAL_OPTIMUM_VALUE[func_name]
        all_results.append((func_name, "cont", hmm, opt, f_star))
        plot_continuous.append((func_name, "cont", hb, ob))
        plot_continuous_raw.append((func_name, "cont", hr, o_r))
        log(f"    HMM_MCMC: {np.mean(hmm):.2f} ± {np.std(hmm):.2f}  (min={np.min(hmm):.4f})")
        log(f"    Optuna:   {np.mean(opt):.2f} ± {np.std(opt):.2f}  (min={np.min(opt):.4f})")

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
        hmm, opt, hb, ob, hr, o_r = run_benchmark(
            func_name, make_noisy_continuous_space, "noisy", noise_std=float(ns)
        )
        f_star = GLOBAL_OPTIMUM_VALUE[func_name]
        all_results.append((func_name, "noisy", hmm, opt, f_star))
        plot_noisy.append((func_name, "noisy", hb, ob))
        plot_noisy_raw.append((func_name, "noisy", hr, o_r))
        log(f"    HMM_MCMC: {np.mean(hmm):.2f} ± {np.std(hmm):.2f}  (min={np.min(hmm):.4f})")
        log(f"    Optuna:   {np.mean(opt):.2f} ± {np.std(opt):.2f}  (min={np.min(opt):.4f})")

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
        hmm, opt, hb, ob, hr, o_r = run_benchmark(
            func_name, make_categorical_space, "cat", noise_std=0.0
        )
        f_star = GLOBAL_OPTIMUM_VALUE[func_name]
        all_results.append((func_name, "cat", hmm, opt, f_star))
        plot_cat.append((func_name, "cat", hb, ob))
        plot_cat_raw.append((func_name, "cat", hr, o_r))
        log(f"    HMM_MCMC: {np.mean(hmm):.2f} ± {np.std(hmm):.2f}  (min={np.min(hmm):.4f})")
        log(f"    Optuna:   {np.mean(opt):.2f} ± {np.std(opt):.2f}  (min={np.min(opt):.4f})")

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
