from __future__ import annotations

import itertools
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from hpo_rl.backends.function import OptimizationBenchmarkBackend
from hpo_rl.baselines.HMM_MCMC import HMM_MCMC
from hpo_rl.controller.plot import plot_and_save


DIMENSIONS = 2       
BUDGET     = 200    
SEED       = 42

GRID: dict[str, list] = {
    "T_mcmc":              [0.01, 0.1, 0.5],
    "sigma_fraction":      [0.05, 0.0055, 0.006],
    "n_chains":            [1, 2, 5],
    "wide_sigma_fraction": [0.20, 0.40, 0.60],
}

FIXED_PARAMS: dict = dict(
    n_init=1,
    orchestrate_every=5,
    temperature=0.60,
    hmm_window=8,
    hmm_obs_epsilon=1e-8,
    hmm_lambda_noise=0.01,
    clone_noise=0.05,
    burnin_fraction=0.10,
    p_cat_step=0.0,
    kde_tau=0.05,
    anneal_T=True,
)

OUTPUT_FILE = "tune_schwefel_results.txt"
PLOTS_DIR   = Path("tune_schwefel_plots")

def make_search_space(dims: int) -> dict:
    return {
        f"x{i}": {"values": [-500.0, 500.0], "type": "float"}
        for i in range(dims)
    }


def run_config(
    varied: dict,
    search_space: dict,
    backend: OptimizationBenchmarkBackend,
) -> tuple[float, list]:
    np.random.seed(SEED)

    optimizer = HMM_MCMC(
        objective_func=backend.evaluate,
        budget=BUDGET,
        dict_to_optimize=search_space,
        **{**FIXED_PARAMS, **varied},
    )
    _best_config, best_loss = optimizer.main_loop()
    return float(best_loss), optimizer.data


def best_so_far(data: list) -> np.ndarray:
    scores = np.array([s for _, s in data], dtype=float)
    return np.minimum.accumulate(scores)


def cfg_tag(cfg: dict) -> str:
    return "  ".join(f"{k}={v}" for k, v in cfg.items())

def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)

    search_space = make_search_space(DIMENSIONS)
    backend = OptimizationBenchmarkBackend(
        function_name="schwefel",
        dimensions=DIMENSIONS,
        maximize=False,
    )

    grid_keys  = list(GRID.keys())
    grid_vals  = list(GRID.values())
    combos     = list(itertools.product(*grid_vals))
    total      = len(combos)

    print(f"\n{'='*60}")
    print(f"  HMM-MCMC-FMP grid search on Schwefel-{DIMENSIONS}D")
    print(f"  {total} configurations × {BUDGET} evals each")
    print(f"{'='*60}\n")

    results: list[tuple[dict, float, list]] = []

    for idx, combo in enumerate(combos, start=1):
        varied = dict(zip(grid_keys, combo))
        print(f"\n[{idx:3d}/{total}]  {cfg_tag(varied)}")

        best_loss, data = run_config(varied, search_space, backend)
        results.append((varied, best_loss, data))

        suffix = (
            f"_T{varied['T_mcmc']}"
            f"_sf{varied['sigma_fraction']}"
            f"_K{varied['n_chains']}"
            f"_wsf{varied['wide_sigma_fraction']}"
        )
        best_entry = min(data, key=lambda t: t[1])
        plotter = plot_and_save(
            history=data,
            best_result=best_entry,
            save_path=PLOTS_DIR,
            backend=backend,
            experiment_number=idx,
        )
        plotter.plot_trajectory(suffix=suffix)
        if DIMENSIONS == 2:
            try:
                plotter.plot_3d(suffix=suffix)
            except Exception as exc:
                print(f"  plot_3d skipped: {exc}")
        plotter.save_history(as_latex=False, suffix=suffix)

        print(f"  → best loss = {best_loss:.4f}")

    results.sort(key=lambda r: r[1])
    best_cfg, best_loss, best_data = results[0]

    fig, ax = plt.subplots(figsize=(13, 7))
    palette = cm.viridis(np.linspace(0.0, 0.9, total))

    for (cfg, loss, data), color in zip(results, palette):
        bsf = best_so_far(data)
        label = (
            f"T={cfg['T_mcmc']}, σ={cfg['sigma_fraction']}, "
            f"K={cfg['n_chains']}, σw={cfg['wide_sigma_fraction']}  "
            f"→ {loss:.2f}"
        )
        ax.plot(bsf, color=color, linewidth=1.0, alpha=0.65, label=label)

    ax.plot(
        best_so_far(best_data),
        color="red", linewidth=2.5, zorder=10,
        label=f"BEST  →  {best_loss:.4f}",
    )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Global optimum = 0")

    ax.set_xlabel("Evaluation", fontsize=12)
    ax.set_ylabel("Best-so-far $f(x)$", fontsize=12)
    ax.set_title(f"HMM-MCMC-FMP Grid Search — Schwefel {DIMENSIONS}D", fontsize=14)
    ax.legend(fontsize=6, loc="upper right", ncol=2, framealpha=0.8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    conv_path = PLOTS_DIR / "convergence_all.png"
    plt.savefig(conv_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nConvergence comparison plot: {conv_path}")

    top5 = results[:5]
    bar_labels = [
        f"T={c['T_mcmc']}\nσ={c['sigma_fraction']}\nK={c['n_chains']}\nσw={c['wide_sigma_fraction']}"
        for c, _, _ in top5
    ]
    bar_values = [loss for _, loss, _ in top5]
    bar_colors = ["gold"] + ["steelblue"] * (len(top5) - 1)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bars = ax2.bar(bar_labels, bar_values, color=bar_colors, edgecolor="black")
    for bar, val in zip(bars, bar_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() * 1.01,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax2.set_ylabel("Best $f(x)$", fontsize=12)
    ax2.set_title("Top-5 Configs — Schwefel", fontsize=13)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    bar_path = PLOTS_DIR / "top5_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Top-5 bar chart: {bar_path}")

    T_vals     = sorted(set(c["T_mcmc"]              for c, _, _ in results))
    sf_vals    = sorted(set(c["sigma_fraction"]      for c, _, _ in results))
    heat_data  = np.full((len(sf_vals), len(T_vals)), np.nan)

    for cfg, loss, _ in results:
        row = sf_vals.index(cfg["sigma_fraction"])
        col = T_vals.index(cfg["T_mcmc"])
        if np.isnan(heat_data[row, col]):
            heat_data[row, col] = loss
        else:
            heat_data[row, col] = min(heat_data[row, col], loss)

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    im = ax3.imshow(heat_data, aspect="auto", cmap="plasma_r", origin="lower")
    ax3.set_xticks(range(len(T_vals)))
    ax3.set_xticklabels([str(v) for v in T_vals])
    ax3.set_yticks(range(len(sf_vals)))
    ax3.set_yticklabels([str(v) for v in sf_vals])
    ax3.set_xlabel("T_mcmc", fontsize=12)
    ax3.set_ylabel("sigma_fraction", fontsize=12)
    ax3.set_title("Best f(x) heatmap (min over n_chains, wide_σ)", fontsize=12)
    for row in range(len(sf_vals)):
        for col in range(len(T_vals)):
            ax3.text(col, row, f"{heat_data[row, col]:.1f}",
                     ha="center", va="center", fontsize=9, color="white")
    plt.colorbar(im, ax=ax3, label="Best f(x)")
    plt.tight_layout()

    heat_path = PLOTS_DIR / "heatmap_T_sigma.png"
    plt.savefig(heat_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap: {heat_path}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        fh.write("HMM-MCMC-FMP Grid Search on Schwefel\n")
        fh.write(f"Date       : {timestamp}\n")
        fh.write(f"Dimensions : {DIMENSIONS}\n")
        fh.write(f"Budget     : {BUDGET} evals / run\n")
        fh.write(f"Grid size  : {total} configurations\n")
        fh.write(f"PRNG seed  : {SEED}\n")
        fh.write("\n")

        fh.write("=" * 60 + "\n")
        fh.write("BEST CONFIGURATION\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"Best f(x)  : {best_loss:.6f}\n\n")
        fh.write("Varied parameters:\n")
        for k, v in best_cfg.items():
            fh.write(f"    {k:<26}: {v}\n")
        fh.write("\nFixed parameters:\n")
        for k, v in FIXED_PARAMS.items():
            fh.write(f"    {k:<26}: {v}\n")
        fh.write("\nFull config (JSON):\n")
        fh.write(json.dumps({**best_cfg, **FIXED_PARAMS}, indent=2))
        fh.write("\n\n")

        fh.write("=" * 60 + "\n")
        fh.write("ALL RESULTS (sorted best → worst)\n")
        fh.write("=" * 60 + "\n")
        header = (
            f"{'Rank':>4}  {'T_mcmc':>7}  {'sigma_frac':>10}  "
            f"{'n_chains':>8}  {'wide_sigma':>10}  {'best_f(x)':>12}\n"
        )
        fh.write(header)
        fh.write("-" * (len(header) - 1) + "\n")
        for rank, (cfg, loss, _) in enumerate(results, start=1):
            fh.write(
                f"{rank:>4}  {cfg['T_mcmc']:>7.3f}  "
                f"{cfg['sigma_fraction']:>10.3f}  "
                f"{cfg['n_chains']:>8d}  "
                f"{cfg['wide_sigma_fraction']:>10.3f}  "
                f"{loss:>12.6f}\n"
            )

    print(f"\nResults file: {OUTPUT_FILE}")
    print(f"\nBest config  : {best_cfg}")
    print(f"Best f(x)    : {best_loss:.6f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
