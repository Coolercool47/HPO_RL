"""Benchmark HMM_MCMC_BW, HMM_MCMC_DREAM, and Optuna TPE on classic test functions.

Three suites (same structure as test_hmm_vs_optuna.py, rewritten here):
  1. Continuous 10D (clean)
  2. Noisy continuous
  3. Continuous + dummy categorical hyperparameters

Run:
  python experiments/test_bw_dream_tpe_benchmarks.py
  python experiments/test_bw_dream_tpe_benchmarks.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import optuna

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hpo_rl.backends.function import OptimizationBenchmarkBackend
from hpo_rl.baselines.HMM_MCMC_BW import HMM_MCMC_BW
from hpo_rl.baselines.HMM_MCMC_DREAM import HMM_MCMC_DREAM

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = REPO_ROOT / "experiments" / "benchmark_results"

ALGORITHMS = ("bw", "dream", "tpe")
ALGO_LABELS = {
    "bw": "HMM_MCMC_BW",
    "dream": "HMM_MCMC_DREAM",
    "tpe": "Optuna TPE",
}
ALGO_COLORS = {
    "bw": "#1f77b4",
    "dream": "#2ca02c",
    "tpe": "#ff7f0e",
}

# Shared HPO settings tuned for synthetic benchmarks (not lcbench JSON).
BENCHMARK_SHARED = dict(
    n_init=32,
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
    rejection_streak=10,
    show_progress=False,
    verbose_history=False,
)

BW_PARAMS = dict(
    **BENCHMARK_SHARED,
    n_chains=1,
    orchestrate_every=1000,
    use_baum_welch=True,
    bw_refit_every=5,
    bw_min_obs=12,
    bw_n_em_iters=3,
    bw_prior_strength=25.0,
    bw_exploit_prior_scale=3.0,
    bw_max_len=64,
)

DREAM_PARAMS = dict(
    **BENCHMARK_SHARED,
    n_chains=2,
    orchestrate_every=16,
    orchestrate_patience=15,
    p_dream=0.5,
    dream_n_pairs=1,
    dream_cr=0.9,
    dream_gamma1_prob=0.1,
    dream_eps=1e-3,
    dream_min_pop=4,
    dream_diversity_frac=0.25,
    bw_refit_every=5,
    bw_min_obs=12,
    bw_n_em_iters=3,
    bw_prior_strength=25.0,
    bw_exploit_prior_scale=3.0,
    bw_max_len=64,
)

TPE_STARTUP_TRIALS = BENCHMARK_SHARED["n_init"]

# ---------------------------------------------------------------------------
# Benchmark suites (rewritten for this experiment)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    bounds: tuple[float, float]
    global_optimum_value: float | None = None


def _styblinski_optimum(dims: int) -> float:
    return round(-39.16617 * dims, 5)


def _michalewicz_optimum(dims: int) -> float:
    return {1: -0.8013, 2: -1.8013, 5: -4.687658, 10: -9.66015}.get(dims, float("nan"))


def continuous_suite(dims: int) -> dict[str, BenchmarkSpec]:
    return {
        "sphere": BenchmarkSpec("sphere", (-5.0, 5.0), 0.0),
        "rosenbrock": BenchmarkSpec("rosenbrock", (-1.0, 1.0), 0.0),
        "rastrigin": BenchmarkSpec("rastrigin", (-5.12, 5.12), 0.0),
        "ackley": BenchmarkSpec("ackley", (-32.768, 32.768), 0.0),
        "griewank": BenchmarkSpec("griewank", (-600.0, 600.0), 0.0),
        "schwefel": BenchmarkSpec("schwefel", (-500.0, 500.0), 0.0),
        "levy": BenchmarkSpec("levy", (-10.0, 10.0), 0.0),
        # 10-D Michalewicz (m=10) global minimum is -9.66015, not 0 (fixed for the rebuttal)
        "michalewicz": BenchmarkSpec("michalewicz", (0.0, float(np.pi)), _michalewicz_optimum(dims)),
        "styblinski_tang": BenchmarkSpec(
            "styblinski_tang", (-5.0, 5.0), _styblinski_optimum(dims)
        ),
    }


def noisy_suite(dims: int) -> dict[str, tuple[BenchmarkSpec, float]]:
    """func_name -> (spec, noise_std). Subset of continuous benchmarks."""
    base = continuous_suite(dims)
    noise_levels = {
        "sphere": 0.5,
        "rastrigin": 8.0,
        "ackley": 2.0,
        "schwefel": 300.0,
        "levy": 1.5,
    }
    return {name: (base[name], noise_std) for name, noise_std in noise_levels.items()}


def categorical_suite(dims: int) -> dict[str, BenchmarkSpec]:
    """Continuous core + dummy categorical knobs (optimizer, activation, scheduler)."""
    return {
        name: spec
        for name, spec in continuous_suite(dims).items()
        if name in {"sphere", "rastrigin", "ackley", "schwefel", "levy"}
    }


DUMMY_CATEGORIES = {
    "optimizer": {"values": ["adam", "sgd", "rmsprop", "adamw"], "type": "categorical"},
    "activation": {"values": ["relu", "tanh", "gelu", "silu"], "type": "categorical"},
    "scheduler": {"values": ["cosine", "step", "plateau"], "type": "categorical"},
}


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

def make_continuous_space(spec: BenchmarkSpec, dims: int) -> dict:
    lo, hi = spec.bounds
    return {
        f"x{i}": {"values": [float(lo), float(hi)], "type": "float", "log": False}
        for i in range(dims)
    }


def make_categorical_space(spec: BenchmarkSpec, dims: int) -> dict:
    space = make_continuous_space(spec, dims)
    space.update(DUMMY_CATEGORIES)
    return space


# ---------------------------------------------------------------------------
# Curve helpers
# ---------------------------------------------------------------------------

def scores_to_best_curve(scores: list[float]) -> np.ndarray:
    return np.minimum.accumulate(np.asarray(scores, dtype=float))


def pad_curve(curve: np.ndarray, budget: int) -> np.ndarray:
    if len(curve) >= budget:
        return curve[:budget]
    if len(curve) == 0:
        return np.full(budget, np.nan)
    tail = np.full(budget - len(curve), curve[-1])
    return np.concatenate([curve, tail])


@dataclass
class RunOutput:
    best_loss: float
    best_curve: np.ndarray
    raw_curve: np.ndarray


# ---------------------------------------------------------------------------
# Algorithm runners
# ---------------------------------------------------------------------------

def _silence_hmm_run(main_loop_fn: Callable[[], tuple]) -> tuple:
    with open(os.devnull, "w") as devnull:
        saved_out, saved_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            return main_loop_fn()
        finally:
            sys.stdout, sys.stderr = saved_out, saved_err


def _extract_hmm_curves(data: list, budget: int) -> RunOutput:
    scores = [float(s) for _, s in data]
    raw = pad_curve(np.asarray(scores, dtype=float), budget)
    best = pad_curve(scores_to_best_curve(scores), budget)
    return RunOutput(best_loss=float(min(scores)), best_curve=best, raw_curve=raw)


def run_bw(
    backend: OptimizationBenchmarkBackend,
    space: dict,
    seed: int,
    budget: int,
) -> RunOutput:
    np.random.seed(seed)

    def _run():
        alg = HMM_MCMC_BW(
            objective_func=backend.evaluate,
            budget=budget,
            dict_to_optimize=space,
            **BW_PARAMS,
        )
        alg.main_loop()
        return alg

    alg = _silence_hmm_run(_run)
    return _extract_hmm_curves(alg.data, budget)


def run_dream(
    backend: OptimizationBenchmarkBackend,
    space: dict,
    seed: int,
    budget: int,
) -> RunOutput:
    np.random.seed(seed)

    def _run():
        alg = HMM_MCMC_DREAM(
            objective_func=backend.evaluate,
            budget=budget,
            dict_to_optimize=space,
            **DREAM_PARAMS,
        )
        alg.main_loop()
        return alg

    alg = _silence_hmm_run(_run)
    return _extract_hmm_curves(alg.data, budget)


def run_tpe(
    backend: OptimizationBenchmarkBackend,
    space: dict,
    seed: int,
    budget: int,
) -> RunOutput:
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
            else:
                raise ValueError(f"Unsupported param type: {info['type']}")
        return backend.evaluate(config)

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=TPE_STARTUP_TRIALS)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)

    scores = [
        float(t.value)
        for t in sorted(study.trials, key=lambda tr: tr.number)
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    raw = pad_curve(np.asarray(scores, dtype=float), budget)
    best = pad_curve(scores_to_best_curve(scores), budget)
    return RunOutput(best_loss=float(study.best_value), best_curve=best, raw_curve=raw)


RUNNERS = {
    "bw": run_bw,
    "dream": run_dream,
    "tpe": run_tpe,
}


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

@dataclass
class SuiteResult:
    suite: str
    benchmark: str
    tag: str
    f_star: float
    best_losses: dict[str, list[float]]
    best_curves: dict[str, list[np.ndarray]]
    raw_curves: dict[str, list[np.ndarray]]


def run_single_benchmark(
    suite_name: str,
    benchmark_name: str,
    spec: BenchmarkSpec,
    *,
    dims: int,
    budget: int,
    n_seeds: int,
    noise_std: float,
    categorical: bool,
) -> SuiteResult:
    backend = OptimizationBenchmarkBackend(
        function_name=benchmark_name,
        dimensions=dims,
        noise_std=noise_std,
    )
    space = (
        make_categorical_space(spec, dims)
        if categorical
        else make_continuous_space(spec, dims)
    )

    best_losses = {algo: [] for algo in ALGORITHMS}
    best_curves = {algo: [] for algo in ALGORITHMS}
    raw_curves = {algo: [] for algo in ALGORITHMS}

    for s in range(n_seeds):
        seed = 42 + s
        for algo in ALGORITHMS:
            out = RUNNERS[algo](backend, space, seed, budget)
            best_losses[algo].append(out.best_loss)
            best_curves[algo].append(out.best_curve)
            raw_curves[algo].append(out.raw_curve)

    tag = suite_name
    if noise_std > 0:
        tag = f"{suite_name}_noise{noise_std:g}"
    if categorical:
        tag = f"{tag}_cat"

    f_star = float(spec.global_optimum_value if spec.global_optimum_value is not None else np.nan)
    return SuiteResult(
        suite=suite_name,
        benchmark=benchmark_name,
        tag=tag,
        f_star=f_star,
        best_losses=best_losses,
        best_curves=best_curves,
        raw_curves=raw_curves,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def winner_label(means: dict[str, float]) -> str:
    best = min(means, key=means.get)
    tied = [a for a, m in means.items() if np.isclose(m, means[best], rtol=0, atol=1e-9)]
    if len(tied) > 1:
        return "tie"
    return ALGO_LABELS[best]


def print_summary_table(results: list[SuiteResult], out) -> None:
    header = (
        f"{'Benchmark':<36} | {'f*':>10} | "
        f"{'BW mean±std':>18} | {'DREAM mean±std':>18} | {'TPE mean±std':>18} | "
        f"{'Winner':>14}"
    )
    sep = "-" * len(header)
    print(sep, file=out)
    print(header, file=out)
    print(sep, file=out)

    for row in results:
        means = {a: float(np.mean(row.best_losses[a])) for a in ALGORITHMS}
        stds = {a: float(np.std(row.best_losses[a])) for a in ALGORITHMS}
        label = f"{row.tag} {row.benchmark}"
        print(
            f"{label:<36} | {row.f_star:10.4f} | "
            f"{means['bw']:8.2f}±{stds['bw']:6.2f} | "
            f"{means['dream']:8.2f}±{stds['dream']:6.2f} | "
            f"{means['tpe']:8.2f}±{stds['tpe']:6.2f} | "
            f"{winner_label(means):>14}",
            file=out,
        )
    print(sep, file=out)


def save_csv(results: list[SuiteResult], path: Path, n_seeds: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["suite", "benchmark", "tag", "f_star", "algorithm", "seed", "best_loss"]
        )
        for row in results:
            for algo in ALGORITHMS:
                for seed_idx, loss in enumerate(row.best_losses[algo]):
                    writer.writerow(
                        [
                            row.suite,
                            row.benchmark,
                            row.tag,
                            row.f_star,
                            algo,
                            42 + seed_idx,
                            loss,
                        ]
                    )


def plot_suite(
    results: list[SuiteResult],
    *,
    curve_key: str,
    y_label: str,
    title: str,
    outfile: Path,
    budget: int,
    n_seeds: int,
) -> None:
    if not results:
        return

    n = len(results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False)
    evals = np.arange(1, budget + 1)

    for idx, row in enumerate(results):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        curves_by_algo = row.best_curves if curve_key == "best" else row.raw_curves

        for algo in ALGORITHMS:
            stack = np.vstack(curves_by_algo[algo])
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
            color = ALGO_COLORS[algo]
            ax.plot(evals, mean, label=ALGO_LABELS[algo], color=color, lw=1.8)
            ax.fill_between(evals, mean - std, mean + std, color=color, alpha=0.18)

        ax.set_title(row.benchmark)
        ax.set_xlabel("Evaluation")
        ax.set_ylabel(y_label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"{title} (mean ± std over {n_seeds} seeds)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HMM_MCMC_BW, HMM_MCMC_DREAM, and Optuna TPE on synthetic benchmarks"
    )
    parser.add_argument("--smoke", action="store_true", help="Quick run: 2 seeds, budget 80, 3 functions")
    parser.add_argument("--budget", type=int, default=500, help="Evaluation budget per run")
    parser.add_argument("--dims", type=int, default=10, help="Continuous benchmark dimensionality")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for plots and CSV/text results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budget = 80 if args.smoke else int(args.budget)
    dims = 10 if args.smoke else int(args.dims)
    n_seeds = 2 if args.smoke else int(args.seeds)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[SuiteResult] = []

    def log(msg: str = "", file=None) -> None:
        print(msg)
        if file is not None:
            print(msg, file=file)

    results_path = args.output_dir / "bw_dream_tpe_benchmarks.txt"
    csv_path = args.output_dir / "bw_dream_tpe_benchmarks.csv"

    with results_path.open("w", encoding="utf-8") as report:
        log(f"BW / DREAM / TPE benchmark comparison", report)
        log(f"budget={budget}, dims={dims}, seeds={n_seeds}, smoke={args.smoke}", report)
        log("", report)

        # --- Continuous clean ---
        cont = continuous_suite(dims)
        if args.smoke:
            cont = {k: cont[k] for k in ["sphere", "rastrigin", "ackley"]}

        log("=" * 72, report)
        log(f"CONTINUOUS {dims}D (noise_std=0)", report)
        log("=" * 72, report)
        cont_results: list[SuiteResult] = []
        for name, spec in cont.items():
            log(f"\n>>> {name} ...", report)
            row = run_single_benchmark(
                "continuous",
                name,
                spec,
                dims=dims,
                budget=budget,
                n_seeds=n_seeds,
                noise_std=0.0,
                categorical=False,
            )
            cont_results.append(row)
            all_results.append(row)
            for algo in ALGORITHMS:
                losses = row.best_losses[algo]
                log(
                    f"    {ALGO_LABELS[algo]:16s}  "
                    f"{np.mean(losses):.2f} ± {np.std(losses):.2f}  (min={np.min(losses):.4f})",
                    report,
                )

        plot_suite(
            cont_results,
            curve_key="best",
            y_label="Best loss so far",
            title=f"Continuous {dims}D — best-so-far",
            outfile=args.output_dir / "convergence_continuous.png",
            budget=budget,
            n_seeds=n_seeds,
        )
        plot_suite(
            cont_results,
            curve_key="raw",
            y_label="Observed loss",
            title=f"Continuous {dims}D — per evaluation",
            outfile=args.output_dir / "per_eval_continuous.png",
            budget=budget,
            n_seeds=n_seeds,
        )

        if not args.smoke:
            # --- Noisy ---
            log("\n" + "=" * 72, report)
            log("NOISY CONTINUOUS", report)
            log("=" * 72, report)
            noisy_results: list[SuiteResult] = []
            for name, (spec, noise_std) in noisy_suite(dims).items():
                log(f"\n>>> {name} (noise_std={noise_std}) ...", report)
                row = run_single_benchmark(
                    "noisy",
                    name,
                    spec,
                    dims=dims,
                    budget=budget,
                    n_seeds=n_seeds,
                    noise_std=float(noise_std),
                    categorical=False,
                )
                noisy_results.append(row)
                all_results.append(row)
                for algo in ALGORITHMS:
                    losses = row.best_losses[algo]
                    log(
                        f"    {ALGO_LABELS[algo]:16s}  "
                        f"{np.mean(losses):.2f} ± {np.std(losses):.2f}  (min={np.min(losses):.4f})",
                        report,
                    )

            plot_suite(
                noisy_results,
                curve_key="best",
                y_label="Best loss so far",
                title="Noisy — best-so-far",
                outfile=args.output_dir / "convergence_noisy.png",
                budget=budget,
                n_seeds=n_seeds,
            )
            plot_suite(
                noisy_results,
                curve_key="raw",
                y_label="Observed loss",
                title="Noisy — per evaluation",
                outfile=args.output_dir / "per_eval_noisy.png",
                budget=budget,
                n_seeds=n_seeds,
            )

            # --- Categorical + continuous ---
            log("\n" + "=" * 72, report)
            log(f"CATEGORICAL + {dims}D CONTINUOUS", report)
            log("=" * 72, report)
            cat_results: list[SuiteResult] = []
            for name, spec in categorical_suite(dims).items():
                log(f"\n>>> {name} + dummy categorical ...", report)
                row = run_single_benchmark(
                    "categorical",
                    name,
                    spec,
                    dims=dims,
                    budget=budget,
                    n_seeds=n_seeds,
                    noise_std=0.0,
                    categorical=True,
                )
                cat_results.append(row)
                all_results.append(row)
                for algo in ALGORITHMS:
                    losses = row.best_losses[algo]
                    log(
                        f"    {ALGO_LABELS[algo]:16s}  "
                        f"{np.mean(losses):.2f} ± {np.std(losses):.2f}  (min={np.min(losses):.4f})",
                        report,
                    )

            plot_suite(
                cat_results,
                curve_key="best",
                y_label="Best loss so far",
                title=f"Categorical + {dims}D — best-so-far",
                outfile=args.output_dir / "convergence_categorical.png",
                budget=budget,
                n_seeds=n_seeds,
            )
            plot_suite(
                cat_results,
                curve_key="raw",
                y_label="Observed loss",
                title=f"Categorical + {dims}D — per evaluation",
                outfile=args.output_dir / "per_eval_categorical.png",
                budget=budget,
                n_seeds=n_seeds,
            )

        log("\n" + "=" * 72, report)
        log("SUMMARY", report)
        log("=" * 72, report)
        print_summary_table(all_results, report)
        print_summary_table(all_results, out=None)

    save_csv(all_results, csv_path, n_seeds)
    print(f"\nSaved report: {results_path}")
    print(f"Saved CSV:    {csv_path}")
    print(f"Saved plots:  {args.output_dir}")


if __name__ == "__main__":
    main()
