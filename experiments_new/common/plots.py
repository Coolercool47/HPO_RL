"""Figures: convergence panels, state trajectories, observation histograms vs
emissions, transition-matrix evolution, sensitivity curves."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STATE_COLORS = {"EXPLOIT": "#1f77b4", "EXPLORE": "#ff7f0e", "TRAPPED": "#d62728", "INIT": "#7f7f7f",
                "rescue": "#9467bd", "reseed": "#8c564b"}
METHOD_COLORS = {"RS": "#7f7f7f", "TPE": "#ff7f0e", "GP": "#2ca02c", "CMAES": "#9467bd", "TPE_HB": "#bcbd22",
                 "SMAC": "#17becf", "FMP": "#1f77b4", "FMP_DREAM": "#d62728", "FMP_MC": "#3b6fb6",
                 "FMP_MC_NOSUB": "#6f9fd8", "FMP_SOFT": "#0b3d91"}


def _color(m):
    return METHOD_COLORS.get(m, None)


def plot_convergence(curves: dict, tasks: list[str], methods: list[str], path: Path, *, band: str = "sem",
                     ylabel: str = "best value so far", log_y: bool = False, transform=None, ncols: int = 3,
                     xlabel: str = "evaluation", title: str = "", x_of=None) -> None:
    """curves: {(task, method): seeds x T}. band: 'sem' | 'std' | 'iqr'.
    x_of(task, T) -> x values (default 1..T)."""
    n = len(tasks)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    for ax, task in zip(axes.ravel(), tasks):
        for m in methods:
            c = curves.get((task, m))
            if c is None:
                continue
            y = c if transform is None else transform(c, task)
            x = np.arange(1, y.shape[1] + 1) if x_of is None else np.asarray(x_of(task, y.shape[1]))
            if band == "iqr":
                mid = np.nanmedian(y, axis=0)
                lo, hi = np.nanpercentile(y, 25, axis=0), np.nanpercentile(y, 75, axis=0)
            else:
                mid = np.nanmean(y, axis=0)
                sd = np.nanstd(y, axis=0, ddof=1) if y.shape[0] > 1 else np.zeros_like(mid)
                half = sd / math.sqrt(y.shape[0]) if band == "sem" else sd
                lo, hi = mid - half, mid + half
            ax.plot(x, mid, label=f"{m} (n={y.shape[0]})", color=_color(m), lw=1.4)
            ax.fill_between(x, lo, hi, alpha=0.15, color=_color(m))
        ax.set_title(task, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), fontsize=8, bbox_to_anchor=(0.5, -0.01))
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_state_trajectory(rec: dict, path: Path, chain: int | None = None, title: str = "") -> None:
    """Decoded state per evaluation (colour band), loss, temperature, posterior for one run."""
    hist = rec["fmp"]["history"]
    rows = [r for r in hist if r["chain"] >= 0 and (chain is None or r["chain"] == chain)]
    if not rows:
        return
    ev = np.array([r["eval"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    T = np.array([r["T"] for r in rows])
    post = np.array([[r["post_exploit"], r["post_explore"], r["post_trapped"]] for r in rows])
    states = [r["state"] for r in rows]
    acc = np.array([r["accepted"] for r in rows])
    best = np.minimum.accumulate(np.array([r["loss"] for r in hist]))
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1, 1]})
    ax = axes[0]
    ax.plot(np.arange(1, len(best) + 1), best, color="k", lw=1.2, label="best so far")
    ax.scatter(ev[acc], loss[acc], s=10, c=[STATE_COLORS[s] for s, a in zip(states, acc) if a], label="accepted")
    ax.scatter(ev[~acc], loss[~acc], s=10, marker="x", c=[STATE_COLORS[s] for s, a in zip(states, acc) if not a], alpha=0.6, label="rejected")
    for r in hist:
        if r["kernel"] in ("rescue", "reseed"):
            ax.axvline(r["eval"], color=STATE_COLORS[r["kernel"]], lw=0.8, alpha=0.6)
    if rec.get("maximize_raw"):
        ax.set_ylabel("-accuracy")
    else:
        ax.set_ylabel("loss")
        if np.all(loss > 0):
            ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title or f"{rec['task']} / {rec['method']} / seed {rec['seed']}", fontsize=10)
    ax = axes[1]
    for k, nm in enumerate(("EXPLOIT", "EXPLORE", "TRAPPED")):
        ax.fill_between(ev, 0, post[:, k], where=None, alpha=0.0)
    ax.stackplot(ev, post.T, colors=[STATE_COLORS[s] for s in ("EXPLOIT", "EXPLORE", "TRAPPED")], alpha=0.8)
    ax.set_ylabel("posterior")
    ax.set_ylim(0, 1)
    ax = axes[2]
    for nm in ("EXPLOIT", "EXPLORE", "TRAPPED"):
        idx = [i for i, s in enumerate(states) if s == nm]
        ax.scatter(ev[idx], np.full(len(idx), {"EXPLOIT": 0, "EXPLORE": 1, "TRAPPED": 2}[nm]), s=6, color=STATE_COLORS[nm])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["EXPLOIT", "EXPLORE", "TRAPPED"], fontsize=8)
    ax.set_ylabel("decoded")
    ax = axes[3]
    ax.plot(ev, T, color="k", lw=1)
    ax.set_yscale("log")
    ax.set_ylabel("T")
    ax.set_xlabel("evaluation")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_observation_histogram(records: list[dict], path: Path, emission_mu=(-0.01, 0.10, 0.015),
                               emission_sigma=(0.06, 0.25, 0.02), xlim=(-1.5, 1.5), title: str = "") -> None:
    """Histogram of HMM observations O_t coloured by decoded state, with the hand-set
    emission densities and a 3-component Gaussian mixture fitted offline to the same data."""
    obs_by_state = {s: [] for s in ("EXPLOIT", "EXPLORE", "TRAPPED")}
    all_obs = []
    for rec in records:
        for r in rec["fmp"]["history"]:
            if r["chain"] >= 0 and np.isfinite(r["obs"]):
                obs_by_state.setdefault(r["state"], []).append(r["obs"])
                all_obs.append(r["obs"])
    all_obs = np.array(all_obs)
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(xlim[0], xlim[1], 121)
    ax.hist([obs_by_state[s] for s in ("EXPLOIT", "EXPLORE", "TRAPPED")], bins=bins, stacked=True, density=True,
            color=[STATE_COLORS[s] for s in ("EXPLOIT", "EXPLORE", "TRAPPED")], label=["EXPLOIT", "EXPLORE", "TRAPPED"], alpha=0.7)
    x = np.linspace(xlim[0], xlim[1], 600)
    for s, mu, sd in zip(("EXPLOIT", "EXPLORE", "TRAPPED"), emission_mu, emission_sigma):
        ax.plot(x, np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)), color=STATE_COLORS[s], lw=1.5, ls="--",
                label=f"hand-set b_{s.lower()} (mu={mu}, sigma={sd})")
    try:
        from sklearn.mixture import GaussianMixture

        clipped = all_obs[(all_obs > xlim[0]) & (all_obs < xlim[1])].reshape(-1, 1)
        if len(clipped) > 30:
            gm = GaussianMixture(3, random_state=0).fit(clipped)
            for k in np.argsort(gm.means_.ravel()):
                mu, sd, w = gm.means_[k, 0], math.sqrt(gm.covariances_[k, 0, 0]), gm.weights_[k]
                ax.plot(x, w * np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)), color="k", lw=1, ls=":",
                        label=f"offline GMM comp (mu={mu:.3f}, sigma={sd:.3f}, w={w:.2f})")
    except Exception:  # noqa: BLE001
        pass
    n_out = int(np.sum((all_obs <= xlim[0]) | (all_obs >= xlim[1])))
    ax.set_xlim(*xlim)
    ax.set_xlabel("normalized increment O_t")
    ax.set_ylabel("density")
    ax.set_title(title or f"O_t distribution ({len(all_obs)} obs, {n_out} outside plot range)", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_transition_evolution(records: list[dict], path: Path, title: str = "") -> None:
    """Entries of A over evaluations (all chains, all seeds) vs the prior."""
    fig, axes = plt.subplots(3, 3, figsize=(9, 7), sharex=True, sharey=True)
    names = ("EXPLOIT", "EXPLORE", "TRAPPED")
    prior = None
    for rec in records:
        p = rec.get("params", {}).get("transition_prior")
        if p:
            prior = np.array(p)
        for snap in rec["fmp"].get("A_history", []):
            A = np.array(snap["A"])
            for i in range(3):
                for j in range(3):
                    axes[i, j].plot(snap["eval"], A[i, j], ".", ms=3, alpha=0.4, color="#1f77b4")
    if prior is None:
        prior = np.array([[0.6, 0.4, 0.0], [0.5, 0.5, 0.0], [0.6, 0.2, 0.2]])
    for i in range(3):
        for j in range(3):
            axes[i, j].axhline(prior[i, j], color="r", lw=1, ls="--")
            axes[i, j].set_title(f"A[{names[i]} -> {names[j]}]", fontsize=8)
            axes[i, j].set_ylim(0, 1)
    for ax in axes[-1]:
        ax.set_xlabel("evaluation")
    fig.suptitle(title or "Baum-Welch transition estimates (dots) vs prior (dashed)", fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sensitivity(df, knobs: list[str], path: Path, value_col: str = "nregret", tasks: list[str] | None = None,
                     title: str = "", base_value: float = 1.0) -> None:
    """df columns: knob, level (multiplier or value), task, value_col (per seed). One panel per knob."""
    tasks = tasks or sorted(df["task"].unique())
    n = len(knobs)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), squeeze=False)
    for ax, knob in zip(axes.ravel(), knobs):
        d = df[df["knob"] == knob]
        for task in tasks:
            dt = d[d["task"] == task].groupby("level")[value_col].agg(["mean", "sem"]).reset_index().sort_values("level")
            if dt.empty:
                continue
            ax.errorbar(dt["level"], dt["mean"], yerr=dt["sem"], marker="o", ms=3, capsize=2, label=task, lw=1)
        ax.set_title(knob, fontsize=9)
        ax.axhline(base_value, color="k", lw=0.8, ls="--")
        if (d["level"].astype(float) > 0).all():
            ax.set_xscale("log")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), fontsize=8)
    fig.suptitle(title or f"One-at-a-time sensitivity ({value_col})", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
