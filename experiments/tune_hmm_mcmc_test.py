"""Grid search hyperparameters for HMM_MCMC_TEST on underperforming benchmarks.

Tunes sharpness (T_mcmc, sigma_fraction), spline (spline_min_archive, spline_floor),
and Baum-Welch (bw_prior_strength) on functions where TEST lost to plain HMM_MCMC
in hmm_test_vs_optuna_results.txt: schwefel, griewank, sphere, ackley.

Run from repo root::

    python experiments/tune_hmm_mcmc_test.py

Plots and results are saved to experiments/tune_hmm_mcmc_test_plots/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from hpo_rl.backends.function import OptimizationBenchmarkBackend
from hpo_rl.baselines.HMM_MCMC import HMM_MCMC
from hpo_rl.baselines.HMM_MCMC_TEST import HMM_MCMC_TEST

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIMENSIONS = 10
BUDGET = 300
SEEDS = [42, 43, 44]

FUNCTIONS: dict[str, tuple[float, float]] = {
    "schwefel": (-500.0, 500.0),
    "griewank": (-600.0, 600.0),
    "sphere": (-5.0, 5.0),
    "ackley": (-32.768, 32.768),
}

GLOBAL_OPTIMUM: dict[str, float] = {
    "schwefel": 0.0,
    "griewank": 0.0,
    "sphere": 0.0,
    "ackley": 0.0,
}

GRID: dict[str, list] = {
    "T_mcmc": [0.01, 0.1],
    "sigma_fraction": [0.0055],
    "spline_min_archive": [10, 25, 50],
    "spline_floor": [0.02, 0.05, 0.10],
    "bw_prior_strength": [5.0, 25.0, 50.0],
    "spline_knots": [10, 60],
    "spline_mix_scale": [0.5, 1.0, 2.0],
}

HMM_BASELINE_PARAMS: dict = dict(
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

FIXED_TEST_PARAMS: dict = dict(
    n_init=32,
    n_chains=1,
    orchestrate_every=1000,
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
    use_baum_welch=True,
    use_spline_proposal=True,
    bw_exploit_prior_scale=3.0,
    locality_sigma_fraction=0.08,
    bw_refit_every=5,
    bw_min_obs=12,
    bw_n_em_iters=3,
    bw_max_len=64,
    verbose_history=False,
    show_progress=True,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "tune_hmm_mcmc_test_plots"
RESULTS_FILE = OUTPUT_DIR / "tune_hmm_mcmc_test_results.txt"

_FS_LABEL = 16
_FS_TITLE = 18
_FS_TICK = 14
_FS_ANNOT = 13
_FS_LEGEND = 11


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_search_space(lo: float, hi: float, dims: int) -> dict:
    return {
        f"x{i}": {"values": [float(lo), float(hi)], "type": "float", "log": False}
        for i in range(dims)
    }


def best_so_far(data: list) -> np.ndarray:
    scores = np.array([s for _, s in data], dtype=float)
    return np.minimum.accumulate(scores)


def pad_curve(curve: np.ndarray, budget: int) -> np.ndarray:
    if len(curve) >= budget:
        return curve[:budget]
    if len(curve) == 0:
        return np.full(budget, np.nan)
    pad = np.full(budget - len(curve), curve[-1])
    return np.concatenate([curve, pad])


def cfg_tag(cfg: dict) -> str:
    return "  ".join(f"{k}={v}" for k, v in cfg.items())


def run_test_config(
    varied: dict,
    search_space: dict,
    backend: OptimizationBenchmarkBackend,
    seed: int,
    func_name: str,
    cfg_idx: int,
) -> tuple[float, np.ndarray]:
    np.random.seed(seed)
    desc = f"TUNE {func_name} #{cfg_idx} s={seed}"
    alg = HMM_MCMC_TEST(
        objective_func=backend.evaluate,
        budget=BUDGET,
        dict_to_optimize=search_space,
        progress_desc=desc,
        **{**FIXED_TEST_PARAMS, **varied},
    )
    _best_cfg, best_loss = alg.main_loop()
    bsf = pad_curve(best_so_far(alg.data), BUDGET)
    return float(best_loss), bsf


def run_plain_hmm(
    search_space: dict,
    backend: OptimizationBenchmarkBackend,
    seed: int,
    func_name: str,
) -> tuple[float, np.ndarray]:
    np.random.seed(seed)
    alg = HMM_MCMC(
        objective_func=backend.evaluate,
        budget=BUDGET,
        dict_to_optimize=search_space,
        **HMM_BASELINE_PARAMS,
    )
    with open(os.devnull, "w") as devnull:
        _saved_out, _saved_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            _best_cfg, best_loss = alg.main_loop()
        finally:
            sys.stdout, sys.stderr = _saved_out, _saved_err
    bsf = pad_curve(best_so_far(alg.data), BUDGET)
    return float(best_loss), bsf


def plot_convergence(
    func_name: str,
    results: list[tuple[dict, float, np.ndarray]],
    baseline_curve: np.ndarray,
    baseline_loss: float,
    out_dir: Path,
) -> None:
    total = len(results)
    fig, ax = plt.subplots(figsize=(13, 7))
    palette = cm.viridis(np.linspace(0.0, 0.9, max(total, 1)))

    best_cfg, best_loss, best_curve = results[0]
    for (cfg, loss, curve), color in zip(results, palette):
        label = (
            f"T={cfg['T_mcmc']}, sf={cfg['sigma_fraction']}, "
            f"sma={cfg['spline_min_archive']}, fl={cfg['spline_floor']}, "
            f"bw={cfg['bw_prior_strength']}, kn={cfg['spline_knots']}, "
            f"mix={cfg['spline_mix_scale']}  -> {loss:.2f}"
        )
        ax.plot(curve, color=color, linewidth=1.0, alpha=0.65, label=label)

    ax.plot(
        best_curve,
        color="red",
        linewidth=2.5,
        zorder=10,
        label=f"BEST tuned  ->  {best_loss:.4f}",
    )
    ax.plot(
        baseline_curve,
        color="C2",
        linewidth=2.0,
        linestyle="--",
        zorder=9,
        label=f"plain HMM_MCMC  ->  {baseline_loss:.4f}",
    )
    f_star = GLOBAL_OPTIMUM.get(func_name, 0.0)
    ax.axhline(
        f_star,
        color="gray",
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
        label=f"Global optimum = {f_star}",
    )

    ax.set_xlabel("Evaluation", fontsize=_FS_LABEL)
    ax.set_ylabel("Best-so-far $f(x)$", fontsize=_FS_LABEL)
    ax.set_title(
        f"HMM_MCMC_TEST Grid Search -- {func_name} {DIMENSIONS}D",
        fontsize=_FS_TITLE,
    )
    ax.tick_params(axis="both", labelsize=_FS_TICK)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    conv_path = out_dir / f"convergence_{func_name}.png"
    plt.savefig(conv_path, dpi=150, bbox_inches="tight")

    handles, labels = ax.get_legend_handles_labels()
    fig_leg, ax_leg = plt.subplots(figsize=(14, max(4, 0.22 * len(labels))))
    ax_leg.axis("off")
    ax_leg.legend(
        handles,
        labels,
        fontsize=_FS_LEGEND,
        ncol=2,
        frameon=True,
        framealpha=0.9,
        loc="center",
    )
    leg_path = out_dir / f"convergence_{func_name}_legend.png"
    fig_leg.savefig(leg_path, dpi=150, bbox_inches="tight")
    plt.close(fig_leg)
    plt.close()
    print(f"  Convergence plot: {conv_path}")
    print(f"  Legend:           {leg_path}")


def plot_top5_bar(
    func_name: str,
    results: list[tuple[dict, float, np.ndarray]],
    baseline_loss: float,
    out_dir: Path,
) -> None:
    top5 = results[:5]
    bar_labels = [
        (
            f"#{rank}\n"
            f"T={c['T_mcmc']}\nsf={c['sigma_fraction']}\n"
            f"sma={c['spline_min_archive']}\nfl={c['spline_floor']}\n"
            f"bw={c['bw_prior_strength']}"
        )
        for rank, (c, _, _) in enumerate(top5, start=1)
    ]
    bar_values = [loss for _, loss, _ in top5]
    bar_colors = ["gold"] + ["steelblue"] * (len(top5) - 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(bar_labels, bar_values, color=bar_colors, edgecolor="black")
    for bar, val in zip(bars, bar_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() * 1.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=_FS_ANNOT,
        )
    ax.axhline(
        baseline_loss,
        color="C2",
        linestyle="--",
        linewidth=1.5,
        label=f"plain HMM_MCMC = {baseline_loss:.2f}",
    )
    ax.set_ylabel("Best $f(x)$", fontsize=_FS_LABEL)
    ax.set_title(f"Top-5 Tuned Configs -- {func_name}", fontsize=_FS_TITLE)
    ax.tick_params(axis="both", labelsize=_FS_TICK)
    ax.legend(fontsize=_FS_LEGEND)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    bar_path = out_dir / f"top5_bar_{func_name}.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Top-5 bar chart:  {bar_path}")


def plot_heatmap(
    func_name: str,
    results: list[tuple[dict, float, np.ndarray]],
    out_dir: Path,
) -> None:
    sf_vals = sorted(set(c["sigma_fraction"] for c, _, _ in results))
    sma_vals = sorted(set(c["spline_min_archive"] for c, _, _ in results))
    heat_data = np.full((len(sf_vals), len(sma_vals)), np.nan)

    for cfg, loss, _ in results:
        row = sf_vals.index(cfg["sigma_fraction"])
        col = sma_vals.index(cfg["spline_min_archive"])
        if np.isnan(heat_data[row, col]):
            heat_data[row, col] = loss
        else:
            heat_data[row, col] = min(heat_data[row, col], loss)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(heat_data, aspect="auto", cmap="plasma_r", origin="lower")
    ax.set_xticks(range(len(sma_vals)))
    ax.set_xticklabels([str(v) for v in sma_vals], fontsize=_FS_TICK)
    ax.set_yticks(range(len(sf_vals)))
    ax.set_yticklabels([str(v) for v in sf_vals], fontsize=_FS_TICK)
    ax.set_xlabel("spline_min_archive", fontsize=_FS_LABEL)
    ax.set_ylabel("sigma_fraction", fontsize=_FS_LABEL)
    ax.set_title(
        f"Best f(x) heatmap -- {func_name} (min over other dims)",
        fontsize=_FS_TITLE,
    )
    for row in range(len(sf_vals)):
        for col in range(len(sma_vals)):
            val = heat_data[row, col]
            if not np.isnan(val):
                ax.text(
                    col,
                    row,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=_FS_ANNOT,
                    color="black",
                )
    cbar = plt.colorbar(im, ax=ax, label="Best f(x)")
    cbar.ax.tick_params(labelsize=_FS_TICK)
    cbar.set_label("Best f(x)", fontsize=_FS_LABEL)
    plt.tight_layout()

    heat_path = out_dir / f"heatmap_{func_name}.png"
    plt.savefig(heat_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap:          {heat_path}")


def write_function_results(
    fh,
    func_name: str,
    results: list[tuple[dict, float, np.ndarray]],
    baseline_loss: float,
) -> None:
    best_cfg, best_loss, _ = results[0]
    delta = best_loss - baseline_loss
    improved = best_loss < baseline_loss

    fh.write("=" * 70 + "\n")
    fh.write(f"FUNCTION: {func_name} ({DIMENSIONS}D, budget={BUDGET})\n")
    fh.write("=" * 70 + "\n")
    fh.write(f"Best tuned f(x)     : {best_loss:.6f}\n")
    fh.write(f"plain HMM_MCMC f(x) : {baseline_loss:.6f}\n")
    fh.write(f"Delta (tuned - HMM) : {delta:+.6f}\n")
    fh.write(f"Improved over HMM   : {'YES' if improved else 'NO'}\n\n")

    fh.write("Best tuned varied parameters:\n")
    for k, v in best_cfg.items():
        fh.write(f"    {k:<26}: {v}\n")
    fh.write("\nFixed TEST parameters:\n")
    for k, v in FIXED_TEST_PARAMS.items():
        if k not in ("show_progress", "verbose_history"):
            fh.write(f"    {k:<26}: {v}\n")
    fh.write("\nFull best config (JSON):\n")
    fh.write(json.dumps({**best_cfg, **FIXED_TEST_PARAMS}, indent=2, default=str))
    fh.write("\n\n")

    fh.write("ALL RESULTS (sorted best -> worst)\n")
    fh.write("-" * 70 + "\n")
    header = (
        f"{'Rank':>4}  {'T_mcmc':>7}  {'sigma_frac':>10}  "
        f"{'spline_min':>10}  {'spline_fl':>9}  {'bw_prior':>8}  "
        f"{'knots':>5}  {'mix':>5}  {'best_f(x)':>12}\n"
    )
    fh.write(header)
    fh.write("-" * len(header.rstrip()) + "\n")
    for rank, (cfg, loss, _) in enumerate(results, start=1):
        fh.write(
            f"{rank:>4}  {cfg['T_mcmc']:>7.3f}  "
            f"{cfg['sigma_fraction']:>10.4f}  "
            f"{cfg['spline_min_archive']:>10d}  "
            f"{cfg['spline_floor']:>9.3f}  "
            f"{cfg['bw_prior_strength']:>8.1f}  "
            f"{cfg['spline_knots']:>5d}  "
            f"{cfg['spline_mix_scale']:>5.1f}  "
            f"{loss:>12.6f}\n"
        )
    fh.write("\n")


def tune_function(func_name: str, lo: float, hi: float) -> dict:
    print(f"\n{'='*60}")
    print(f"  Tuning HMM_MCMC_TEST on {func_name} {DIMENSIONS}D")
    print(f"{'='*60}")

    search_space = make_search_space(lo, hi, DIMENSIONS)
    backend = OptimizationBenchmarkBackend(
        function_name=func_name,
        dimensions=DIMENSIONS,
        noise_std=0.0,
    )

    print("\n  Running plain HMM_MCMC baseline ...")
    baseline_losses: list[float] = []
    baseline_curves: list[np.ndarray] = []
    for seed in SEEDS:
        loss, curve = run_plain_hmm(search_space, backend, seed, func_name)
        baseline_losses.append(loss)
        baseline_curves.append(curve)
        print(f"    seed={seed}: loss={loss:.4f}")
    baseline_loss = float(np.mean(baseline_losses))
    baseline_curve = np.nanmean(np.vstack(baseline_curves), axis=0)
    print(f"  Baseline mean loss: {baseline_loss:.4f}")

    grid_keys = list(GRID.keys())
    grid_vals = list(GRID.values())
    combos = list(itertools.product(*grid_vals))
    total = len(combos)

    print(f"\n  Grid search: {total} configs x {len(SEEDS)} seeds x {BUDGET} evals")
    results: list[tuple[dict, float, np.ndarray]] = []

    for idx, combo in enumerate(combos, start=1):
        varied = dict(zip(grid_keys, combo))
        print(f"\n  [{idx:3d}/{total}]  {cfg_tag(varied)}")

        seed_losses: list[float] = []
        seed_curves: list[np.ndarray] = []
        for seed in SEEDS:
            loss, curve = run_test_config(
                varied, search_space, backend, seed, func_name, idx
            )
            seed_losses.append(loss)
            seed_curves.append(curve)
            print(f"      seed={seed}: loss={loss:.4f}")

        mean_loss = float(np.mean(seed_losses))
        mean_curve = np.nanmean(np.vstack(seed_curves), axis=0)
        results.append((varied, mean_loss, mean_curve))
        print(f"    -> mean loss = {mean_loss:.4f}")

    results.sort(key=lambda r: r[1])

    plot_convergence(func_name, results, baseline_curve, baseline_loss, OUTPUT_DIR)
    plot_top5_bar(func_name, results, baseline_loss, OUTPUT_DIR)
    plot_heatmap(func_name, results, OUTPUT_DIR)

    best_cfg, best_loss, _ = results[0]
    return {
        "func_name": func_name,
        "best_cfg": best_cfg,
        "best_loss": best_loss,
        "baseline_loss": baseline_loss,
        "improved": best_loss < baseline_loss,
        "results": results,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid_keys = list(GRID.keys())
    grid_vals = list(GRID.values())
    total_configs = len(list(itertools.product(*grid_vals)))

    print(f"\n{'='*60}")
    print("  HMM_MCMC_TEST Hyperparameter Grid Search")
    print(f"  Functions : {list(FUNCTIONS.keys())}")
    print(f"  Dims      : {DIMENSIONS}")
    print(f"  Budget    : {BUDGET} evals / run")
    print(f"  Seeds     : {SEEDS}")
    print(f"  Grid size : {total_configs} configs / function")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"{'='*60}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summaries: list[dict] = []

    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        fh.write("HMM_MCMC_TEST Hyperparameter Grid Search\n")
        fh.write(f"Date       : {timestamp}\n")
        fh.write(f"Dimensions : {DIMENSIONS}\n")
        fh.write(f"Budget     : {BUDGET} evals / run\n")
        fh.write(f"Seeds      : {SEEDS}\n")
        fh.write(f"Grid size  : {total_configs} configurations / function\n")
        fh.write(f"Grid keys  : {grid_keys}\n")
        fh.write(f"Functions  : {list(FUNCTIONS.keys())}\n\n")

        for func_name, (lo, hi) in FUNCTIONS.items():
            summary = tune_function(func_name, lo, hi)
            summaries.append(summary)
            write_function_results(
                fh, func_name, summary["results"], summary["baseline_loss"]
            )

        fh.write("=" * 70 + "\n")
        fh.write("FINAL SUMMARY\n")
        fh.write("=" * 70 + "\n")
        fh.write(
            f"{'Function':<14} {'Best tuned':>12} {'plain HMM':>12} "
            f"{'Delta':>10} {'Improved':>10}\n"
        )
        fh.write("-" * 60 + "\n")
        for s in summaries:
            delta = s["best_loss"] - s["baseline_loss"]
            fh.write(
                f"{s['func_name']:<14} {s['best_loss']:12.4f} "
                f"{s['baseline_loss']:12.4f} {delta:+10.4f} "
                f"{'YES' if s['improved'] else 'NO':>10}\n"
            )

    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Function':<14} {'Best tuned':>12} {'plain HMM':>12} {'Improved':>10}")
    print("-" * 50)
    for s in summaries:
        print(
            f"{s['func_name']:<14} {s['best_loss']:12.4f} "
            f"{s['baseline_loss']:12.4f} "
            f"{'YES' if s['improved'] else 'NO':>10}"
        )

    print(f"\nResults file: {RESULTS_FILE}")
    print("Done.")


def _apply_smoke_mode() -> None:
    """Reduce grid/budget for a quick sanity check."""
    global DIMENSIONS, BUDGET, SEEDS, FUNCTIONS, GRID
    DIMENSIONS = 2
    BUDGET = 40
    SEEDS = [42]
    FUNCTIONS = {"sphere": (-5.0, 5.0)}
    GRID = {
        "T_mcmc": [0.01],
        "sigma_fraction": [0.0055],
        "spline_min_archive": [10, 50],
        "spline_floor": [0.02],
        "bw_prior_strength": [5.0, 25.0],
        "spline_knots": [10, 60],
        "spline_mix_scale": [1.0],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune HMM_MCMC_TEST hyperparameters")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick sanity run: sphere 2D, budget=40, reduced grid",
    )
    args = parser.parse_args()
    if args.smoke:
        _apply_smoke_mode()
        print("[smoke mode] Reduced grid/budget for quick verification\n")
    main()
