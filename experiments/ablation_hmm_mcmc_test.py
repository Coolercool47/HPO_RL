"""Ablation study: why HMM_MCMC_TEST regresses vs plain HMM_MCMC on smooth benchmarks.

Experiments:
  A) Component ablation (BW / spline / both / neither vs plain HMM_MCMC)
  B) Spline resolution sweep (spline_knots)
  C) BW direction: state usage, step-size histograms, final A matrix

Run from repo root::

    python experiments/ablation_hmm_mcmc_test.py
    python experiments/ablation_hmm_mcmc_test.py --smoke
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from hpo_rl.backends.function import OptimizationBenchmarkBackend
from hpo_rl.baselines.HMM_MCMC import HMM_MCMC, HMMState, MCMCChain
from hpo_rl.baselines.HMM_MCMC_TEST import BaumWelchHMMController, HMM_MCMC_TEST

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DIMENSIONS = 10
BUDGET = 300
SEEDS = [42, 43, 44]

FUNCTIONS: dict[str, tuple[float, float]] = {
    "sphere": (-5.0, 5.0),
    "griewank": (-600.0, 600.0),
    "ackley": (-32.768, 32.768),
    "schwefel": (-500.0, 500.0),
}

STATE_NAMES = [s.name for s in HMMState]
KNOTS_SWEEP = [10, 30, 60, 120]

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

BASE_TEST_PARAMS = dict(
    **HMM_PARAMS,
    bw_refit_every=5,
    bw_min_obs=12,
    bw_n_em_iters=3,
    bw_prior_strength=5.0,
    bw_max_len=64,
    spline_knots=10,
    spline_floor=0.05,
    spline_min_archive=5,
    locality_sigma_fraction=0.08,
    verbose_history=False,
    show_progress=True,
)

ABLATION_VARIANTS: dict[str, dict] = {
    "plain_HMM": {"_use_hmm": True},
    "TEST_neither": {"use_baum_welch": False, "use_spline_proposal": False},
    "TEST_BW_only": {"use_baum_welch": True, "use_spline_proposal": False},
    "TEST_spline_only": {"use_baum_welch": False, "use_spline_proposal": True},
    "TEST_full": {"use_baum_welch": True, "use_spline_proposal": True},
}

OUTPUT_DIR = Path(__file__).resolve().parent / "ablation_hmm_mcmc_test_plots"
RESULTS_FILE = OUTPUT_DIR / "ablation_results.txt"

# Global step log for experiment C (monkey-patch)
_STEP_LOG: list[dict] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_space(lo: float, hi: float, dims: int) -> dict:
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


def l2_step(x_from: dict, x_to: dict, space: dict) -> float:
    sq = 0.0
    for name, info in space.items():
        if info["type"] == "float":
            d = float(x_to[name]) - float(x_from[name])
            lo, hi = info["values"]
            span = max(float(hi) - float(lo), 1e-8)
            sq += (d / span) ** 2
    return float(np.sqrt(sq / max(sum(1 for v in space.values() if v["type"] == "float"), 1)))


def run_plain_hmm(
    backend: OptimizationBenchmarkBackend,
    space: dict,
    seed: int,
    show_progress: bool = False,
) -> tuple[float, list, np.ndarray]:
    np.random.seed(seed)
    alg = HMM_MCMC(
        objective_func=backend.evaluate,
        budget=BUDGET,
        dict_to_optimize=space,
        **HMM_PARAMS,
    )
    if not show_progress:
        with open(os.devnull, "w") as devnull:
            saved_out, saved_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = devnull
            try:
                _best_cfg, best_loss = alg.main_loop()
            finally:
                sys.stdout, sys.stderr = saved_out, saved_err
    else:
        _best_cfg, best_loss = alg.main_loop()
    bsf = pad_curve(best_so_far(alg.data), BUDGET)
    return float(best_loss), alg.data, bsf


def run_test_variant(
    backend: OptimizationBenchmarkBackend,
    space: dict,
    seed: int,
    variant_params: dict,
    label: str = "",
    show_progress: bool = False,
) -> tuple[float, list, np.ndarray]:
    np.random.seed(seed)
    params = {**BASE_TEST_PARAMS, **variant_params}
    params.pop("show_progress", None)
    desc = f"ABL {label} s={seed}" if label else None
    alg = HMM_MCMC_TEST(
        objective_func=backend.evaluate,
        budget=BUDGET,
        dict_to_optimize=space,
        progress_desc=desc,
        show_progress=show_progress,
        **params,
    )
    _best_cfg, best_loss = alg.main_loop()
    bsf = pad_curve(best_so_far(alg.data), BUDGET)
    return float(best_loss), alg.data, bsf


def run_variant_multi_seed(
    backend: OptimizationBenchmarkBackend,
    space: dict,
    variant_name: str,
    variant_params: dict,
    show_progress: bool = False,
) -> tuple[float, float, np.ndarray]:
    losses: list[float] = []
    curves: list[np.ndarray] = []
    for seed in SEEDS:
        if variant_params.get("_use_hmm"):
            loss, _, bsf = run_plain_hmm(backend, space, seed, show_progress=show_progress)
        else:
            loss, _, bsf = run_test_variant(
                backend, space, seed, variant_params,
                label=variant_name, show_progress=show_progress,
            )
        losses.append(loss)
        curves.append(bsf)
    return float(np.mean(losses)), float(np.std(losses)), np.nanmean(np.vstack(curves), axis=0)


def state_fractions(history_table: list[dict]) -> dict[str, float]:
    if not history_table:
        return {s: 0.0 for s in STATE_NAMES}
    counts = Counter(row["State"] for row in history_table)
    total = sum(counts.values())
    return {s: counts.get(s, 0) / max(total, 1) for s in STATE_NAMES}


def _patch_mcmc_step(space: dict) -> None:
    global _STEP_LOG
    _STEP_LOG = []
    _orig = MCMCChain.step

    def patched_step(self, objective_func, progress=0.0, is_burnin=False):
        x_before = copy.deepcopy(self.current_x)
        result = _orig(self, objective_func, progress, is_burnin)
        x_prime, loss_prime, accepted = result
        _STEP_LOG.append({
            "state": self.state.name,
            "step_size": l2_step(x_before, x_prime, space),
            "accepted": bool(accepted),
        })
        return result

    MCMCChain.step = patched_step


def _unpatch_mcmc_step() -> None:
    MCMCChain.step = MCMCChain.__wrapped_step__  # type: ignore


# Store original step for unpatch
if not hasattr(MCMCChain, "__wrapped_step__"):
    MCMCChain.__wrapped_step__ = MCMCChain.step  # type: ignore


def run_instrumented_test(
    backend: OptimizationBenchmarkBackend,
    space: dict,
    seed: int,
    use_bw: bool,
    label: str,
) -> tuple[float, list[dict], np.ndarray, dict]:
    """Run TEST with step logging; return loss, step_log, final A, state fracs."""
    np.random.seed(seed)
    _patch_mcmc_step(space)
    try:
        params = {
            **BASE_TEST_PARAMS,
            "use_baum_welch": use_bw,
            "use_spline_proposal": True,
        }
        params.pop("show_progress", None)
        alg = HMM_MCMC_TEST(
            objective_func=backend.evaluate,
            budget=BUDGET,
            dict_to_optimize=space,
            progress_desc=f"INST {label} s={seed}",
            show_progress=False,
            **params,
        )
        _best_cfg, best_loss = alg.main_loop()
        step_log = list(_STEP_LOG)
        state_fracs = state_fractions(alg.history_table)
        A_final = alg._chains[0].hmm.A.copy() if alg._chains else np.eye(3) * np.nan
    finally:
        MCMCChain.step = MCMCChain.__wrapped_step__  # type: ignore

    return float(best_loss), step_log, A_final, state_fracs


# ---------------------------------------------------------------------------
# Experiment A: component ablation
# ---------------------------------------------------------------------------

def experiment_a(
    func_name: str,
    lo: float,
    hi: float,
    show_progress: bool = False,
) -> dict[str, tuple[float, float, np.ndarray]]:
    print(f"\n  [Exp A] {func_name}")
    backend = OptimizationBenchmarkBackend(func_name, DIMENSIONS, noise_std=0.0)
    space = make_space(lo, hi, DIMENSIONS)
    results: dict[str, tuple[float, float, np.ndarray]] = {}

    for vname, vparams in ABLATION_VARIANTS.items():
        print(f"    {vname} ...", end=" ", flush=True)
        mean_l, std_l, curve = run_variant_multi_seed(
            backend, space, vname, vparams, show_progress=show_progress,
        )
        results[vname] = (mean_l, std_l, curve)
        print(f"loss={mean_l:.4f} +- {std_l:.4f}")

    _plot_ablation_curves(func_name, results, OUTPUT_DIR / f"ablation_components_{func_name}.png")
    return results


def _plot_ablation_curves(
    func_name: str,
    results: dict[str, tuple[float, float, np.ndarray]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {
        "plain_HMM": "C2",
        "TEST_neither": "gray",
        "TEST_BW_only": "C1",
        "TEST_spline_only": "C0",
        "TEST_full": "red",
    }
    styles = {
        "plain_HMM": "--",
        "TEST_neither": ":",
        "TEST_BW_only": "-.",
        "TEST_spline_only": "-",
        "TEST_full": "-",
    }
    for vname, (mean_l, std_l, curve) in results.items():
        ax.plot(
            curve,
            label=f"{vname} ({mean_l:.2f} +- {std_l:.2f})",
            color=colors.get(vname, "black"),
            linestyle=styles.get(vname, "-"),
            linewidth=2.0 if vname in ("plain_HMM", "TEST_full") else 1.2,
        )
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Best-so-far loss")
    ax.set_title(f"Component ablation -- {func_name} {DIMENSIONS}D")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


# ---------------------------------------------------------------------------
# Experiment B: spline_knots sweep
# ---------------------------------------------------------------------------

def experiment_b(
    func_name: str,
    lo: float,
    hi: float,
    show_progress: bool = False,
) -> dict[int, tuple[float, float]]:
    print(f"\n  [Exp B] {func_name} -- spline_knots sweep")
    backend = OptimizationBenchmarkBackend(func_name, DIMENSIONS, noise_std=0.0)
    space = make_space(lo, hi, DIMENSIONS)
    results: dict[int, tuple[float, float]] = {}

    for knots in KNOTS_SWEEP:
        print(f"    knots={knots} ...", end=" ", flush=True)
        losses = []
        for seed in SEEDS:
            np.random.seed(seed)
            extra = {k: v for k, v in BASE_TEST_PARAMS.items()
                     if k not in ("spline_knots", "show_progress", "verbose_history")}
            alg = HMM_MCMC_TEST(
                objective_func=backend.evaluate,
                budget=BUDGET,
                dict_to_optimize=space,
                use_baum_welch=True,
                use_spline_proposal=True,
                spline_knots=knots,
                progress_desc=f"KNOTS {func_name} k={knots} s={seed}",
                show_progress=show_progress,
                verbose_history=False,
                **extra,
            )
            _, best_loss = alg.main_loop()
            losses.append(float(best_loss))
        mean_l, std_l = float(np.mean(losses)), float(np.std(losses))
        results[knots] = (mean_l, std_l)
        print(f"loss={mean_l:.4f} +- {std_l:.4f}")

    _plot_knots_sweep(func_name, results, OUTPUT_DIR / f"spline_knots_sweep_{func_name}.png")
    return results


def _plot_knots_sweep(
    func_name: str,
    results: dict[int, tuple[float, float]],
    out_path: Path,
) -> None:
    knots = sorted(results.keys())
    means = [results[k][0] for k in knots]
    stds = [results[k][1] for k in knots]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(knots, means, yerr=stds, marker="o", capsize=4, linewidth=1.5)
    ax.set_xlabel("spline_knots")
    ax.set_ylabel("Final best loss (mean +- std)")
    ax.set_title(f"Spline resolution sweep -- {func_name} {DIMENSIONS}D")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


# ---------------------------------------------------------------------------
# Experiment C: BW direction + step sizes
# ---------------------------------------------------------------------------

def experiment_c(
    func_name: str,
    lo: float,
    hi: float,
    show_progress: bool = False,
) -> dict:
    print(f"\n  [Exp C] {func_name} -- instrumentation")
    backend = OptimizationBenchmarkBackend(func_name, DIMENSIONS, noise_std=0.0)
    space = make_space(lo, hi, DIMENSIONS)

    bw_logs: list[dict] = []
    no_bw_logs: list[dict] = []
    bw_states: list[dict] = []
    no_bw_states: list[dict] = []
    bw_As: list[np.ndarray] = []
    no_bw_As: list[np.ndarray] = []

    gauss_step_logs: list[dict] = []

    for seed in SEEDS:
        loss_bw, slog_bw, A_bw, sf_bw = run_instrumented_test(
            backend, space, seed, use_bw=True, label=f"{func_name}_BW",
        )
        loss_nobw, slog_nobw, A_nobw, sf_nobw = run_instrumented_test(
            backend, space, seed, use_bw=False, label=f"{func_name}_noBW",
        )
        bw_logs.extend(slog_bw)
        no_bw_logs.extend(slog_nobw)
        bw_states.append(sf_bw)
        no_bw_states.append(sf_nobw)
        bw_As.append(A_bw)
        no_bw_As.append(A_nobw)
        print(f"    seed={seed}: BW loss={loss_bw:.4f}, noBW loss={loss_nobw:.4f}")

        # Gaussian baseline step sizes (plain HMM, one seed instrumentation)
        _patch_mcmc_step(space)
        try:
            np.random.seed(seed)
            alg = HMM_MCMC(
                objective_func=backend.evaluate,
                budget=BUDGET,
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
            gauss_step_logs.extend(list(_STEP_LOG))
        finally:
            MCMCChain.step = MCMCChain.__wrapped_step__  # type: ignore

    mean_bw_state = {
        s: float(np.mean([d[s] for d in bw_states])) for s in STATE_NAMES
    }
    mean_nobw_state = {
        s: float(np.mean([d[s] for d in no_bw_states])) for s in STATE_NAMES
    }
    A_bw_mean = np.mean(bw_As, axis=0)
    A_nobw_mean = np.mean(no_bw_As, axis=0)

    _plot_state_usage(
        func_name, mean_bw_state, mean_nobw_state,
        OUTPUT_DIR / f"bw_state_usage_{func_name}.png",
    )
    _plot_step_histograms(
        func_name, bw_logs, gauss_step_logs,
        OUTPUT_DIR / f"step_size_hist_{func_name}.png",
    )

    return {
        "mean_bw_state": mean_bw_state,
        "mean_nobw_state": mean_nobw_state,
        "A_bw_mean": A_bw_mean,
        "A_nobw_mean": A_nobw_mean,
        "exploit_exploit_bw": float(A_bw_mean[0, 0]),
        "exploit_exploit_nobw": float(A_nobw_mean[0, 0]),
    }


def _plot_state_usage(
    func_name: str,
    bw_states: dict,
    nobw_states: dict,
    out_path: Path,
) -> None:
    x = np.arange(len(STATE_NAMES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, [bw_states[s] for s in STATE_NAMES], width, label="BW on", color="C0")
    ax.bar(x + width / 2, [nobw_states[s] for s in STATE_NAMES], width, label="BW off", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES)
    ax.set_ylabel("Fraction of steps")
    ax.set_title(f"HMM state usage -- {func_name}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


def _plot_step_histograms(
    func_name: str,
    spline_logs: list[dict],
    gauss_logs: list[dict],
    out_path: Path,
) -> None:
    spline_steps = [e["step_size"] for e in spline_logs]
    gauss_steps = [e["step_size"] for e in gauss_logs]
    if not spline_steps and not gauss_steps:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(max(spline_steps, default=0), max(gauss_steps, default=0), 0.01), 40)
    if gauss_steps:
        ax.hist(gauss_steps, bins=bins, alpha=0.55, label="plain HMM (Gaussian)", color="C2", density=True)
    if spline_steps:
        ax.hist(spline_steps, bins=bins, alpha=0.55, label="TEST spline", color="C0", density=True)
    ax.set_xlabel("Normalized step size |x'-x| / range")
    ax.set_ylabel("Density")
    ax.set_title(f"Proposal step size -- {func_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


# ---------------------------------------------------------------------------
# Results writer + hypothesis evaluation
# ---------------------------------------------------------------------------

def evaluate_hypotheses(
    exp_a_all: dict,
    exp_b_all: dict,
    exp_c_all: dict,
) -> dict[str, str]:
    """Return verdict strings for H1, H2, H3."""
    verdicts: dict[str, str] = {}

    # H1: higher knots -> lower loss on smooth functions
    smooth = ["sphere", "griewank", "ackley"]
    h1_votes = 0
    for fn in smooth:
        if fn not in exp_b_all:
            continue
        losses = [exp_b_all[fn][k][0] for k in sorted(exp_b_all[fn])]
        if len(losses) >= 2 and losses[-1] < losses[0]:
            h1_votes += 1
    verdicts["H1"] = (
        f"CONFIRMED ({h1_votes}/{len(smooth)} smooth functions improve with more knots)"
        if h1_votes >= 2 else
        f"INCONCLUSIVE ({h1_votes}/{len(smooth)} smooth functions improve)"
    )

    # H2: BW-only worse than neither; BW lowers EXPLOIT fraction
    h2_bw_worse = 0
    h2_exploit_lower = 0
    for fn in FUNCTIONS:
        if fn not in exp_a_all:
            continue
        a = exp_a_all[fn]
        if "TEST_BW_only" in a and "TEST_neither" in a:
            if a["TEST_BW_only"][0] > a["TEST_neither"][0]:
                h2_bw_worse += 1
        if fn in exp_c_all:
            c = exp_c_all[fn]
            if c["mean_bw_state"]["EXPLOIT"] < c["mean_nobw_state"]["EXPLOIT"]:
                h2_exploit_lower += 1
    verdicts["H2"] = (
        f"CONFIRMED (BW-only worse on {h2_bw_worse}/{len(FUNCTIONS)} fns; "
        f"EXPLOIT fraction lower with BW on {h2_exploit_lower}/{len(FUNCTIONS)})"
        if h2_bw_worse >= 2 and h2_exploit_lower >= 2 else
        f"PARTIAL (BW-only worse: {h2_bw_worse}/{len(FUNCTIONS)}, "
        f"EXPLOIT lower: {h2_exploit_lower}/{len(FUNCTIONS)})"
    )

    # H3: full TEST has largest delta vs plain
    h3_full_worst = 0
    for fn in smooth:
        if fn not in exp_a_all:
            continue
        a = exp_a_all[fn]
        plain = a["plain_HMM"][0]
        deltas = {
            v: a[v][0] - plain
            for v in ("TEST_BW_only", "TEST_spline_only", "TEST_full")
            if v in a
        }
        if deltas and max(deltas, key=deltas.get) == "TEST_full":
            h3_full_worst += 1
    verdicts["H3"] = (
        f"CONFIRMED (full TEST has largest regression on {h3_full_worst}/{len(smooth)} smooth fns)"
        if h3_full_worst >= 2 else
        f"INCONCLUSIVE ({h3_full_worst}/{len(smooth)} smooth fns)"
    )

    return verdicts


def write_results(
    fh,
    exp_a_all: dict,
    exp_b_all: dict,
    exp_c_all: dict,
    verdicts: dict[str, str],
) -> None:
    fh.write("=" * 70 + "\n")
    fh.write("EXPERIMENT A: Component ablation\n")
    fh.write("=" * 70 + "\n")
    fh.write(f"{'Function':<12} {'Variant':<18} {'Mean':>10} {'Std':>10} {'Delta':>10}\n")
    fh.write("-" * 62 + "\n")
    for fn, results in exp_a_all.items():
        plain = results["plain_HMM"][0]
        for vname, (mean_l, std_l, _) in results.items():
            delta = mean_l - plain
            fh.write(f"{fn:<12} {vname:<18} {mean_l:10.4f} {std_l:10.4f} {delta:+10.4f}\n")
        fh.write("\n")

    fh.write("=" * 70 + "\n")
    fh.write("EXPERIMENT B: spline_knots sweep (full TEST)\n")
    fh.write("=" * 70 + "\n")
    for fn, knot_res in exp_b_all.items():
        fh.write(f"\n{fn}:\n")
        plain_loss = exp_a_all.get(fn, {}).get("plain_HMM", (np.nan,))[0]
        for k in sorted(knot_res):
            mean_l, std_l = knot_res[k]
            fh.write(f"  knots={k:>4}: {mean_l:10.4f} +- {std_l:.4f}  (delta plain: {mean_l - plain_loss:+.4f})\n")

    fh.write("\n" + "=" * 70 + "\n")
    fh.write("EXPERIMENT C: BW instrumentation\n")
    fh.write("=" * 70 + "\n")
    for fn, c in exp_c_all.items():
        fh.write(f"\n{fn}:\n")
        fh.write(f"  EXPLOIT fraction  BW on: {c['mean_bw_state']['EXPLOIT']:.4f}  "
                 f"BW off: {c['mean_nobw_state']['EXPLOIT']:.4f}\n")
        fh.write(f"  A[EXPLOIT,EXPLOIT]  BW on: {c['exploit_exploit_bw']:.4f}  "
                 f"BW off: {c['exploit_exploit_nobw']:.4f}\n")
        fh.write(f"  Final A (BW on):\n")
        for i, sn in enumerate(STATE_NAMES):
            row = " ".join(f"{c['A_bw_mean'][i,j]:.4f}" for j in range(3))
            fh.write(f"    {sn}: {row}\n")

    fh.write("\n" + "=" * 70 + "\n")
    fh.write("HYPOTHESIS VERDICTS\n")
    fh.write("=" * 70 + "\n")
    for h, v in verdicts.items():
        fh.write(f"  {h}: {v}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*60}")
    print("  HMM_MCMC_TEST Ablation Study")
    print(f"  Functions : {list(FUNCTIONS.keys())}")
    print(f"  Dims      : {DIMENSIONS}, Budget: {BUDGET}, Seeds: {SEEDS}")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"{'='*60}")

    exp_a_all: dict = {}
    exp_b_all: dict = {}
    exp_c_all: dict = {}

    show_progress = False

    for func_name, (lo, hi) in FUNCTIONS.items():
        print(f"\n{'='*60}")
        print(f"  {func_name}")
        print(f"{'='*60}")
        exp_a_all[func_name] = experiment_a(func_name, lo, hi, show_progress)
        exp_b_all[func_name] = experiment_b(func_name, lo, hi, show_progress)
        exp_c_all[func_name] = experiment_c(func_name, lo, hi, show_progress)

    verdicts = evaluate_hypotheses(exp_a_all, exp_b_all, exp_c_all)

    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        fh.write("HMM_MCMC_TEST Ablation Study\n")
        fh.write(f"Date       : {timestamp}\n")
        fh.write(f"Dimensions : {DIMENSIONS}\n")
        fh.write(f"Budget     : {BUDGET}\n")
        fh.write(f"Seeds      : {SEEDS}\n\n")
        write_results(fh, exp_a_all, exp_b_all, exp_c_all, verdicts)

    print(f"\n{'='*60}")
    print("  HYPOTHESIS VERDICTS")
    print(f"{'='*60}")
    for h, v in verdicts.items():
        print(f"  {h}: {v}")
    print(f"\nResults: {RESULTS_FILE}")
    print("Done.")


def _apply_smoke_mode() -> None:
    global DIMENSIONS, BUDGET, SEEDS, FUNCTIONS, KNOTS_SWEEP
    DIMENSIONS = 2
    BUDGET = 40
    SEEDS = [42]
    FUNCTIONS = {"sphere": (-5.0, 5.0)}
    KNOTS_SWEEP = [10, 30]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM_MCMC_TEST ablation study")
    parser.add_argument("--smoke", action="store_true", help="Quick sanity run")
    args = parser.parse_args()
    if args.smoke:
        _apply_smoke_mode()
        print("[smoke mode] Reduced scope for quick verification\n")
    main()
