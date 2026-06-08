"""Compare HMM_MCMC_TEST vs Optuna TPE vs BOHB on a real CIFAR-100 CNN objective.

Uses the same training pipeline as compare_hmm_mcmc_vs_tpe.py and HMM_MCMC_TEST
hyperparameters from test_hmm_mcmc_test_vs_optuna.py (post-fix defaults).

Run from repo root::

    python experiments/compare_hmm_mcmc_test_vs_tpe.py
    python experiments/compare_hmm_mcmc_test_vs_tpe.py --smoke
    python experiments/compare_hmm_mcmc_test_vs_tpe.py --trials 20 --seeds 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from hpo_rl.baselines.HMM_MCMC_TEST import HMM_MCMC_TEST

# Reuse CIFAR-100 objective stack from compare_hmm_mcmc_vs_tpe.py
from compare_hmm_mcmc_vs_tpe import (
    HMM_SPACE,
    EPOCHS_PER_TRIAL,
    N_SEEDS,
    N_TRIALS,
    RNG_BASE,
    TRAIN_SUBSET,
    VAL_SUBSET,
    _seed_all,
    _train_indices_for_seed,
    build_hmm_objective,
    pad_curve,
    run_optuna_bohb,
    run_optuna_tpe,
    scores_to_best_curve,
    write_csv,
)

from test_hmm_mcmc_test_vs_optuna import _get_hmm_test_params

OUT_DIR = ROOT / "logs" / "compare_hmm_mcmc_test_vs_tpe"
RESULTS_CSV = str(OUT_DIR / "compare_hmm_mcmc_test_cifar_history.csv")
RESULTS_TXT = str(OUT_DIR / "compare_hmm_mcmc_test_cifar_results.txt")
PLOT_FILE = str(OUT_DIR / "compare_hmm_mcmc_test_cifar_convergence.png")

METHODS = ("HMM_MCMC_TEST", "TPE", "BOHB")


def run_hmm_mcmc_test(
    seed: int,
    budget: int,
    device: torch.device,
    data_root: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    epochs: int,
    *,
    show_progress: bool = True,
) -> tuple[list[float], list[tuple[float, float]]]:
    _seed_all(seed)
    np.random.seed(seed)
    metrics: list[tuple[float, float]] = []
    obj = build_hmm_objective(
        device, data_root, train_idx, val_idx, epochs, metrics
    )
    params = _get_hmm_test_params()
    params.pop("show_progress", None)
    params.pop("verbose_history", None)
    alg = HMM_MCMC_TEST(
        objective_func=obj,
        budget=budget,
        dict_to_optimize=HMM_SPACE,
        show_progress=show_progress,
        progress_desc=f"TEST seed={seed}",
        verbose_history=False,
        **params,
    )
    alg.main_loop()
    return [float(s) for _, s in alg.data], metrics


def plot_convergence(
    curves: dict[str, list[np.ndarray]],
    n_trials: int,
    n_seeds: int,
    outfile: str,
) -> None:
    evals = np.arange(1, n_trials + 1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    styles = [
        ("HMM_MCMC_TEST", "C0", "-"),
        ("TPE", "C1", "-"),
        ("BOHB", "C2", "-"),
    ]
    for label, color, ls in styles:
        runs = curves.get(label, [])
        if not runs:
            continue
        stack = np.vstack([pad_curve(r, n_trials) for r in runs])
        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0)
        ax.plot(evals, mean, label=label, color=color, linestyle=ls, linewidth=2)
        if len(runs) > 1:
            ax.fill_between(evals, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Best validation loss so far")
    ax.set_title(
        f"CIFAR-100 HMM_MCMC_TEST vs TPE vs BOHB "
        f"(n_trials={n_trials}, mean +/- std over {n_seeds} seeds)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {outfile}")


def print_summary_table(
    finals_loss: dict[str, list[float]],
    finals_acc: dict[str, list[float]],
    out_stream=None,
) -> None:
    p = lambda s="": print(s, file=out_stream, flush=True)
    hdr = (
        f"{'Method':<16} | {'loss mean+-std':>20} | {'l_min':>8} | {'l_max':>8} "
        f"| {'acc% mean+-std':>16} | {'a_min%':>7} | {'a_max%':>7}"
    )
    sep = "-" * len(hdr)
    p(sep)
    p(hdr)
    p(sep)
    for method in METHODS:
        xs = finals_loss.get(method, [])
        accs = finals_acc.get(method, [])
        if not xs or not accs:
            p(
                f"{method:<16} | {'n/a':>20} | {'n/a':>8} | {'n/a':>8} "
                f"| {'n/a':>16} | {'n/a':>7} | {'n/a':>7}"
            )
            continue
        m, s = float(np.mean(xs)), float(np.std(xs))
        am = float(np.mean(accs) * 100.0)
        astd = float(np.std(accs) * 100.0)
        amin = float(min(accs) * 100.0)
        amax = float(max(accs) * 100.0)
        p(
            f"{method:<16} | {m:9.4f} +- {s:7.4f} | {min(xs):8.4f} | {max(xs):8.4f} | "
            f"{am:7.2f}% +- {astd:5.2f}% | {amin:7.2f} | {amax:7.2f}"
        )
    p(sep)


def _append_csv_rows(
    csv_rows: list[dict],
    method: str,
    seed: int,
    losses: list[float],
    accs: list[float],
) -> None:
    best_curve = scores_to_best_curve(losses)
    best_acc_so_far = np.maximum.accumulate(np.asarray(accs, dtype=np.float64))
    for i, (cur, best, cur_acc, ba_sf) in enumerate(
        zip(losses, best_curve.tolist(), accs, best_acc_so_far.tolist()),
        start=1,
    ):
        csv_rows.append(
            {
                "method": method,
                "seed": seed,
                "trial": i,
                "best_loss_so_far": best,
                "current_loss": cur,
                "trial_val_acc": cur_acc,
                "best_val_acc_so_far": ba_sf,
            }
        )


def _record_finals(
    methods_curves: dict[str, list[np.ndarray]],
    finals_loss: dict[str, list[float]],
    finals_acc: dict[str, list[float]],
    method: str,
    losses: list[float],
    accs: list[float],
    n_trials: int,
) -> None:
    curve = scores_to_best_curve(losses)
    methods_curves[method].append(pad_curve(curve, n_trials))
    finals_loss[method].append(float(curve[-1]) if len(curve) else float("nan"))
    if losses and accs:
        j = int(np.argmin(losses))
        finals_acc[method].append(float(accs[j]))
    else:
        finals_acc[method].append(float("nan"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS_PER_TRIAL)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--train-subset", type=int, default=TRAIN_SUBSET)
    parser.add_argument("--val-subset", type=int, default=VAL_SUBSET)
    parser.add_argument("--csv", type=str, default=RESULTS_CSV)
    parser.add_argument("--txt", type=str, default=RESULTS_TXT)
    parser.add_argument("--plot", type=str, default=PLOT_FILE)
    parser.add_argument(
        "--quiet-optuna",
        action="store_true",
        help="Disable Optuna tqdm progress bar per study",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick run: 1 seed, 3 trials, smaller subsets",
    )
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 1
        args.trials = 3
        args.train_subset = 2000
        args.val_subset = 500
        args.epochs = 2
        print("[smoke mode] 1 seed, 3 trials, train=2000, val=500, epochs=2\n")

    optuna_prog = not args.quiet_optuna

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_params = _get_hmm_test_params()
    print(f"Device: {device}")
    print("HMM_MCMC_TEST params:")
    for k, v in sorted(test_params.items()):
        print(f"  {k}: {v}")
    print()

    methods_curves: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
    finals_loss: dict[str, list[float]] = {m: [] for m in METHODS}
    finals_acc: dict[str, list[float]] = {m: [] for m in METHODS}
    csv_rows: list[dict] = []

    for s in range(args.seeds):
        seed = RNG_BASE + s
        train_idx, val_idx = _train_indices_for_seed(
            seed, args.train_subset, args.val_subset
        )

        print(f"=== seed {seed} ===")

        print("  HMM_MCMC_TEST ...")
        test_losses, test_metrics = run_hmm_mcmc_test(
            seed,
            args.trials,
            device,
            args.data_root,
            train_idx,
            val_idx,
            args.epochs,
            show_progress=True,
        )
        test_accs = [m[1] for m in test_metrics]
        _record_finals(
            methods_curves, finals_loss, finals_acc,
            "HMM_MCMC_TEST", test_losses, test_accs, args.trials,
        )
        _append_csv_rows(csv_rows, "HMM_MCMC_TEST", seed, test_losses, test_accs)
        print(f"    best loss={min(test_losses):.4f}")

        print("  Optuna TPE ...")
        tpe_losses, tpe_accs = run_optuna_tpe(
            seed,
            args.trials,
            device,
            args.data_root,
            train_idx,
            val_idx,
            args.epochs,
            show_progress_bar=optuna_prog,
        )
        _record_finals(
            methods_curves, finals_loss, finals_acc,
            "TPE", tpe_losses, tpe_accs, args.trials,
        )
        _append_csv_rows(csv_rows, "TPE", seed, tpe_losses, tpe_accs)
        print(f"    best loss={min(tpe_losses):.4f}")

        print("  BOHB (TPE + Hyperband) ...")
        bohb_losses, bohb_accs = run_optuna_bohb(
            seed,
            args.trials,
            device,
            args.data_root,
            train_idx,
            val_idx,
            args.epochs,
            show_progress_bar=optuna_prog,
        )
        _record_finals(
            methods_curves, finals_loss, finals_acc,
            "BOHB", bohb_losses, bohb_accs, args.trials,
        )
        _append_csv_rows(csv_rows, "BOHB", seed, bohb_losses, bohb_accs)
        print(f"    best loss={min(bohb_losses):.4f}")
        print()

    plot_convergence(methods_curves, args.trials, args.seeds, args.plot)
    write_csv(args.csv, csv_rows)

    print()
    print_summary_table(finals_loss, finals_acc)
    summary = {
        "n_trials": args.trials,
        "n_seeds": args.seeds,
        "epochs_per_trial": args.epochs,
        "train_subset": args.train_subset,
        "val_subset": args.val_subset,
        "device": str(device),
        "hmm_mcmc_test_params": test_params,
        "final_best_val_loss_per_method": finals_loss,
        "final_val_acc_at_best_loss_trial_per_method": finals_acc,
    }
    with open(args.txt, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2, default=str))
        f.write("\n\n")
        print_summary_table(finals_loss, finals_acc, out_stream=f)
    print(f"\nWrote summary: {args.txt}")


if __name__ == "__main__":
    main()
