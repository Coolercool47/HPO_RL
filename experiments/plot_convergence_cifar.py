"""
Convergence plots for compare_hmm_mcmc_cifar_history.csv

Figures produced:
  1. Individual method figures  – 3 subplots each, with stat lines
        (a) current_loss per trial
        (b) best_loss_so_far per trial  (convergence curve)
        (c) current vs best on the same axes
  2. Combined figure             – all methods on two shared axes
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend – no window
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import make_interp_spline

# ── data ────────────────────────────────────────────────────────────────────
CSV_PATH = "compare_hmm_mcmc_cifar_history.csv"
df = pd.read_csv(CSV_PATH)

methods   = list(df["method"].unique())
PALETTE   = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
color_map = dict(zip(methods, PALETTE))

# ── helpers ──────────────────────────────────────────────────────────────────
def smooth(x, y, n=300):
    """Return a smooth spline curve (quadratic for <=3 pts, cubic otherwise)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    k = min(3, len(x) - 1)           # spline degree capped by num points
    spl = make_interp_spline(x, y, k=k)
    xs  = np.linspace(x[0], x[-1], n)
    return xs, spl(xs)


def add_stat_lines(ax, vals, x_min, x_max):
    """Draw horizontal lines for min, median, mode, max and return legend handles."""
    mode_val = float(stats.mode(vals, keepdims=True).mode[0])
    stat_defs = [
        ("Min",    float(np.min(vals)),    "#2ca02c",  (6, 2)),
        ("Median", float(np.median(vals)), "#1f77b4",  (4, 2)),
        ("Mode",   mode_val,               "#9467bd",  (2, 2)),
        ("Max",    float(np.max(vals)),    "#d62728",  (1, 1)),
    ]
    handles = []
    for label, val, color, dashes in stat_defs:
        line, = ax.plot(
            [x_min, x_max], [val, val],
            color=color, linewidth=1.4, alpha=0.85,
            linestyle=(0, dashes),
            label=f"{label}: {val:.5f}",
        )
        handles.append(line)
    return handles


# ═══════════════════════════════════════════════════════════════════════════
# 1.  One figure per method  –  3 subplots  +  stat lines
# ═══════════════════════════════════════════════════════════════════════════
for method in methods:
    mdf    = df[df["method"] == method].reset_index(drop=True)
    c      = color_map[method]
    trials = mdf["trial"].values
    t_min, t_max = trials[0], trials[-1]

    xs_cur,  ys_cur  = smooth(trials, mdf["current_loss"].values)
    xs_best, ys_best = smooth(trials, mdf["best_loss_so_far"].values)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f"Convergence analysis: {method}", fontsize=14, fontweight="bold")

    # --- subplot 1: current loss per trial -----------------------------------
    ax = axes[0]
    ax.scatter(trials, mdf["current_loss"], color=c, zorder=5, s=50)
    main_line, = ax.plot(xs_cur, ys_cur, color=c, linewidth=2.2, label="Current Loss")
    stat_handles = add_stat_lines(ax, mdf["current_loss"].values, t_min, t_max)
    ax.set_title("Current Loss per Trial")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(handles=[main_line] + stat_handles, fontsize=8)

    # --- subplot 2: convergence curve ----------------------------------------
    ax = axes[1]
    ax.scatter(trials, mdf["best_loss_so_far"], color=c, zorder=5, s=50, marker="s")
    main_line, = ax.plot(xs_best, ys_best, color=c, linewidth=2.2, label="Best Loss So Far")
    stat_handles = add_stat_lines(ax, mdf["best_loss_so_far"].values, t_min, t_max)
    ax.set_title("Best Loss So Far (Convergence)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(handles=[main_line] + stat_handles, fontsize=8)

    # --- subplot 3: current vs best ------------------------------------------
    ax = axes[2]
    ax.scatter(trials, mdf["current_loss"],    color="#aaaaaa", zorder=5, s=50)
    ax.scatter(trials, mdf["best_loss_so_far"], color=c,        zorder=5, s=50, marker="s")
    cur_line,  = ax.plot(xs_cur,  ys_cur,  color="#888888", linewidth=1.8,
                         linestyle="--", alpha=0.8, label="Current Loss")
    best_line, = ax.plot(xs_best, ys_best, color=c,         linewidth=2.2,
                         label="Best Loss So Far")
    # stat lines on best_loss_so_far column
    stat_handles = add_stat_lines(ax, mdf["best_loss_so_far"].values, t_min, t_max)
    ax.set_title("Current vs Best Loss")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Loss")
    ax.legend(handles=[cur_line, best_line] + stat_handles, fontsize=8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(f"convergence_{method}.png", dpi=150, bbox_inches="tight")
    print(f"Saved  convergence_{method}.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Combined figure  –  all methods together
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Convergence — All Methods", fontsize=14, fontweight="bold")

for method in methods:
    mdf    = df[df["method"] == method]
    c      = color_map[method]
    trials = mdf["trial"].values

    xs_cur,  ys_cur  = smooth(trials, mdf["current_loss"].values)
    xs_best, ys_best = smooth(trials, mdf["best_loss_so_far"].values)

    axes[0].scatter(trials, mdf["current_loss"],    color=c, zorder=5, s=40)
    axes[0].plot(xs_cur,  ys_cur,  color=c, linewidth=2.2, label=method)
    axes[1].scatter(trials, mdf["best_loss_so_far"], color=c, zorder=5, s=40, marker="s")
    axes[1].plot(xs_best, ys_best, color=c, linewidth=2.2, label=method)

for ax, title, ylabel in zip(
    axes,
    ["Current Loss per Trial", "Best Loss So Far (Convergence)"],
    ["Current Loss", "Best Loss So Far"],
):
    ax.set_title(title)
    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig("convergence_all_methods.png", dpi=150, bbox_inches="tight")
print("Saved  convergence_all_methods.png")
plt.close()
