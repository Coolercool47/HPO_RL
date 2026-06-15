from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from hpo_rl.backends.function import OptimizationBenchmarkBackend
from hpo_rl.baselines.HMM_MCMC import HMM_MCMC, HMMState
from hpo_rl.baselines.HMM_MCMC_TEST import (
    BaumWelchHMMController,
    HMM_MCMC_TEST,
)

N_SEEDS = 3
BUDGET = 300
DIMENSIONS = 10
SEEDS = [42, 43, 44]

FUNCTIONS: dict[str, tuple[float, float]] = {
    "griewank": (-600.0, 600.0),
    "sphere": (-5.0, 5.0),
    "ackley": (-32.768, 32.768),
}

STATE_NAMES = [s.name for s in HMMState]

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
    bw_prior_strength=5.0,
    bw_max_len=64,
    use_spline_proposal=True,
    spline_knots=10,
    spline_floor=0.05,
    spline_min_archive=5,
    locality_sigma_fraction=0.08,
    verbose_history=False,
    show_progress=True,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "hmm_transition_matrix_logs"
RESULTS_FILE = OUTPUT_DIR / "transition_matrix_comparison.txt"


class LoggingBaumWelchHMMController(BaumWelchHMMController):
    """Baum-Welch controller that records A after each refit."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A_history: list[dict] = []
        self._record_A(event="init")

    def reset(self):
        super().reset()
        self.A_history = []
        self._record_A(event="reset")

    def _record_A(self, event: str) -> None:
        self.A_history.append({
            "event": event,
            "step": int(self._step_count),
            "buffer_len": len(self._bw_buffer),
            "A": self.A.copy().tolist(),
        })

    def _fit_transitions(self, obs_seq: list[float]) -> None:
        super()._fit_transitions(obs_seq)
        self._record_A(event="refit")


class LoggingHMM_MCMC_TEST(HMM_MCMC_TEST):
    """HMM_MCMC_TEST that uses LoggingBaumWelchHMMController."""

    def _make_hmm_controller(self) -> LoggingBaumWelchHMMController:
        return LoggingBaumWelchHMMController(
            window=self._hmm_window,
            obs_epsilon=self._hmm_obs_epsilon,
            lambda_noise=self._hmm_lambda_noise,
            refit_every=self.bw_refit_every,
            min_obs=self.bw_min_obs,
            n_em_iters=self.bw_n_em_iters,
            prior_strength=self.bw_prior_strength,
            bw_max_len=self.bw_max_len,
        )


def make_space(lo: float, hi: float, dims: int) -> dict:
    return {
        f"x{i}": {"values": [float(lo), float(hi)], "type": "float", "log": False}
        for i in range(dims)
    }


def matrix_to_table(A: np.ndarray, title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
    header = f"{'from\\to':>12}" + "".join(f"{s:>12}" for s in STATE_NAMES)
    lines.append(header)
    lines.append("-" * len(header))
    for i, row_name in enumerate(STATE_NAMES):
        row = f"{row_name:>12}" + "".join(f"{A[i, j]:12.6f}" for j in range(3))
        lines.append(row)
    return "\n".join(lines)


def extract_hmm_A(alg: HMM_MCMC) -> dict:
    """Fixed A from plain HMM_MCMC chains."""
    chains_A = []
    for chain in alg._chains:
        A = chain.hmm.A.copy()
        chains_A.append({
            "chain_id": chain.chain_id,
            "A": A.tolist(),
            "A_default": getattr(chain.hmm, "_A_default", chain.hmm.A).copy().tolist(),
        })
    return {
        "type": "fixed",
        "chains": chains_A,
        "A_final_mean": np.mean([c["A"] for c in chains_A], axis=0).tolist()
        if chains_A else [],
    }


def extract_test_A(alg: LoggingHMM_MCMC_TEST) -> dict:
    """Evolving A history from HMM_MCMC_TEST chains."""
    chains_data = []
    for chain in alg._chains:
        hmm: LoggingBaumWelchHMMController = chain.hmm
        chains_data.append({
            "chain_id": chain.chain_id,
            "A_default": hmm._A_default.copy().tolist(),
            "A_final": hmm.A.copy().tolist(),
            "A_history": hmm.A_history,
            "n_refits": sum(1 for h in hmm.A_history if h["event"] == "refit"),
        })
    finals = [c["A_final"] for c in chains_data]
    return {
        "type": "baum_welch",
        "chains": chains_data,
        "A_final_mean": np.mean(finals, axis=0).tolist() if finals else [],
        "A_default_mean": np.mean(
            [c["A_default"] for c in chains_data], axis=0
        ).tolist() if chains_data else [],
    }


def run_hmm_mcmc(
    backend: OptimizationBenchmarkBackend,
    space: dict,
    seed: int,
    func_name: str,
) -> tuple[float, dict]:
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
            _best_cfg, best_loss = alg.main_loop()
        finally:
            sys.stdout, sys.stderr = saved_out, saved_err
    A_info = extract_hmm_A(alg)
    return float(best_loss), A_info


def run_hmm_mcmc_test(
    backend: OptimizationBenchmarkBackend,
    space: dict,
    seed: int,
    func_name: str,
) -> tuple[float, dict]:
    np.random.seed(seed)
    desc = f"TEST {func_name} seed={seed}"
    alg = LoggingHMM_MCMC_TEST(
        objective_func=backend.evaluate,
        budget=BUDGET,
        dict_to_optimize=space,
        progress_desc=desc,
        **HMM_TEST_PARAMS,
    )
    _best_cfg, best_loss = alg.main_loop()
    A_info = extract_test_A(alg)
    return float(best_loss), A_info


def plot_matrix(
    A: np.ndarray,
    title: str,
    out_path: Path,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(A, cmap="Blues", vmin=vmin, vmax=vmax, origin="upper")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(STATE_NAMES, fontsize=9)
    ax.set_yticklabels(STATE_NAMES, fontsize=9)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title(title, fontsize=11)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_A_evolution(
    A_history: list[dict],
    title: str,
    out_path: Path,
) -> None:
    refits = [h for h in A_history if h["event"] in ("init", "refit", "reset")]
    if not refits:
        return
    n = len(refits)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5), squeeze=False)
    for idx, entry in enumerate(refits):
        ax = axes[0, idx]
        A = np.array(entry["A"])
        im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1, origin="upper")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(STATE_NAMES, fontsize=7, rotation=45)
        ax.set_yticklabels(STATE_NAMES, fontsize=7)
        ax.set_title(
            f"{entry['event']}\nstep={entry['step']}",
            fontsize=8,
        )
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_side_by_side(
    A_hmm: np.ndarray,
    A_test: np.ndarray,
    A_default: np.ndarray,
    func_name: str,
    seed: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    mats = [
        (A_default, "Default A (prior)"),
        (A_hmm, "HMM_MCMC (fixed)"),
        (A_test, "HMM_MCMC_TEST (learned)"),
    ]
    for ax, (A, title) in zip(axes, mats):
        im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1, origin="upper")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(STATE_NAMES, fontsize=8)
        ax.set_yticklabels(STATE_NAMES, fontsize=8)
        ax.set_title(title, fontsize=9)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{A[i,j]:.3f}", ha="center", va="center", fontsize=8)
    fig.suptitle(f"{func_name} seed={seed}: Transition matrix comparison", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def matrix_delta(A1: np.ndarray, A2: np.ndarray) -> float:
    return float(np.max(np.abs(A1 - A2)))


def run_comparison(func_name: str, lo: float, hi: float) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"  {func_name} ({DIMENSIONS}D, budget={BUDGET})")
    print(f"{'='*60}")

    space = make_space(lo, hi, DIMENSIONS)
    backend = OptimizationBenchmarkBackend(
        function_name=func_name,
        dimensions=DIMENSIONS,
        noise_std=0.0,
    )

    func_dir = OUTPUT_DIR / func_name
    func_dir.mkdir(parents=True, exist_ok=True)

    seed_results: list[dict] = []

    for seed in SEEDS[:N_SEEDS]:
        print(f"\n  seed={seed} ...")

        hmm_loss, hmm_A_info = run_hmm_mcmc(backend, space, seed, func_name)
        test_loss, test_A_info = run_hmm_mcmc_test(backend, space, seed, func_name)

        A_hmm = np.array(hmm_A_info["A_final_mean"])
        A_test = np.array(test_A_info["A_final_mean"])
        A_default = np.array(test_A_info["A_default_mean"])

        delta_test_vs_hmm = matrix_delta(A_test, A_hmm)
        delta_test_vs_default = matrix_delta(A_test, A_default)

        print(f"    HMM_MCMC      loss={hmm_loss:.4f}")
        print(f"    HMM_MCMC_TEST loss={test_loss:.4f}")
        print(f"    |A_test - A_hmm|_inf = {delta_test_vs_hmm:.6f}")
        print(f"    |A_test - A_default|_inf = {delta_test_vs_default:.6f}")
        print(f"\n{matrix_to_table(A_hmm, title='HMM_MCMC A (fixed):')}")
        print(f"\n{matrix_to_table(A_test, title='HMM_MCMC_TEST A (final):')}")

        seed_tag = f"seed{seed}"
        json_path = func_dir / f"{seed_tag}_matrices.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump({
                "func_name": func_name,
                "seed": seed,
                "budget": BUDGET,
                "dimensions": DIMENSIONS,
                "hmm_loss": hmm_loss,
                "test_loss": test_loss,
                "delta_test_vs_hmm": delta_test_vs_hmm,
                "delta_test_vs_default": delta_test_vs_default,
                "hmm_A": hmm_A_info,
                "test_A": test_A_info,
            }, fh, indent=2)

        txt_path = func_dir / f"{seed_tag}_matrices.txt"
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(f"Function: {func_name}, seed={seed}\n")
            fh.write(f"HMM_MCMC loss:      {hmm_loss:.6f}\n")
            fh.write(f"HMM_MCMC_TEST loss: {test_loss:.6f}\n")
            fh.write(f"|A_test - A_hmm|_inf:     {delta_test_vs_hmm:.6f}\n")
            fh.write(f"|A_test - A_default|_inf: {delta_test_vs_default:.6f}\n\n")
            fh.write(matrix_to_table(A_default, title="Default A (prior):"))
            fh.write("\n\n")
            fh.write(matrix_to_table(A_hmm, title="HMM_MCMC A (fixed):"))
            fh.write("\n\n")
            fh.write(matrix_to_table(A_test, title="HMM_MCMC_TEST A (final mean):"))
            fh.write("\n\n")
            fh.write("HMM_MCMC_TEST A evolution (chain 0):\n")
            if test_A_info["chains"]:
                for entry in test_A_info["chains"][0]["A_history"]:
                    fh.write(
                        f"\n--- {entry['event']} step={entry['step']} "
                        f"buffer={entry['buffer_len']} ---\n"
                    )
                    fh.write(
                        matrix_to_table(
                            np.array(entry["A"]),
                        )
                    )
                    fh.write("\n")

        plot_side_by_side(
            A_hmm, A_test, A_default, func_name, seed,
            func_dir / f"{seed_tag}_comparison.png",
        )
        plot_matrix(
            A_test,
            f"HMM_MCMC_TEST final A -- {func_name} seed={seed}",
            func_dir / f"{seed_tag}_test_final.png",
        )
        if test_A_info["chains"]:
            plot_A_evolution(
                test_A_info["chains"][0]["A_history"],
                f"A evolution -- {func_name} seed={seed} (chain 0)",
                func_dir / f"{seed_tag}_test_evolution.png",
            )

        print(f"    Saved: {json_path}")
        print(f"    Saved: {txt_path}")

        seed_results.append({
            "seed": seed,
            "hmm_loss": hmm_loss,
            "test_loss": test_loss,
            "delta_test_vs_hmm": delta_test_vs_hmm,
            "delta_test_vs_default": delta_test_vs_default,
            "A_hmm": A_hmm.tolist(),
            "A_test": A_test.tolist(),
            "A_default": A_default.tolist(),
        })

    return seed_results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*60}")
    print("  HMM Transition Matrix Comparison")
    print(f"  Functions: {list(FUNCTIONS.keys())}")
    print(f"  Output:    {OUTPUT_DIR}")
    print(f"{'='*60}")

    all_summaries: dict[str, list[dict]] = {}

    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        fh.write("HMM_MCMC vs HMM_MCMC_TEST -- Transition Matrix Comparison\n")
        fh.write(f"Date       : {timestamp}\n")
        fh.write(f"Dimensions : {DIMENSIONS}\n")
        fh.write(f"Budget     : {BUDGET}\n")
        fh.write(f"Seeds      : {SEEDS[:N_SEEDS]}\n")
        fh.write(f"Functions  : {list(FUNCTIONS.keys())}\n\n")

        for func_name, (lo, hi) in FUNCTIONS.items():
            results = run_comparison(func_name, lo, hi)
            all_summaries[func_name] = results

            fh.write("=" * 70 + "\n")
            fh.write(f"FUNCTION: {func_name}\n")
            fh.write("=" * 70 + "\n")
            for r in results:
                fh.write(f"\nseed={r['seed']}\n")
                fh.write(f"  HMM_MCMC loss      : {r['hmm_loss']:.6f}\n")
                fh.write(f"  HMM_MCMC_TEST loss : {r['test_loss']:.6f}\n")
                fh.write(f"  |A_test - A_hmm|_inf     : {r['delta_test_vs_hmm']:.6f}\n")
                fh.write(f"  |A_test - A_default|_inf : {r['delta_test_vs_default']:.6f}\n")
                fh.write(f"\n{matrix_to_table(np.array(r['A_hmm']), title='  HMM_MCMC A:')}\n")
                fh.write(f"\n{matrix_to_table(np.array(r['A_test']), title='  TEST final A:')}\n")
            fh.write("\n")

        fh.write("=" * 70 + "\n")
        fh.write("AGGREGATE SUMMARY\n")
        fh.write("=" * 70 + "\n")
        fh.write(
            f"{'Function':<12} {'HMM mean':>10} {'TEST mean':>10} "
            f"{'dA inf mean':>12} {'TEST wins':>10}\n"
        )
        fh.write("-" * 56 + "\n")
        for func_name, results in all_summaries.items():
            hmm_mean = np.mean([r["hmm_loss"] for r in results])
            test_mean = np.mean([r["test_loss"] for r in results])
            da_mean = np.mean([r["delta_test_vs_hmm"] for r in results])
            test_wins = sum(1 for r in results if r["test_loss"] < r["hmm_loss"])
            fh.write(
                f"{func_name:<12} {hmm_mean:10.4f} {test_mean:10.4f} "
                f"{da_mean:12.6f} {test_wins:>10}\n"
            )

    print(f"\n{'='*60}")
    print("  AGGREGATE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Function':<12} {'HMM mean':>10} {'TEST mean':>10} {'dA inf':>10}")
    print("-" * 44)
    for func_name, results in all_summaries.items():
        hmm_mean = np.mean([r["hmm_loss"] for r in results])
        test_mean = np.mean([r["test_loss"] for r in results])
        da_mean = np.mean([r["delta_test_vs_hmm"] for r in results])
        print(f"{func_name:<12} {hmm_mean:10.4f} {test_mean:10.4f} {da_mean:10.4f}")

    print(f"\nResults: {RESULTS_FILE}")
    print("Done.")


def _apply_smoke_mode() -> None:
    global BUDGET, DIMENSIONS, SEEDS, N_SEEDS, FUNCTIONS
    BUDGET = 40
    DIMENSIONS = 2
    SEEDS = [42]
    N_SEEDS = 1
    FUNCTIONS = {"sphere": (-5.0, 5.0)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare HMM transition matrices")
    parser.add_argument("--smoke", action="store_true", help="Quick sanity run")
    args = parser.parse_args()
    if args.smoke:
        _apply_smoke_mode()
        print("[smoke mode] Reduced budget/dims for quick verification\n")
    main()
