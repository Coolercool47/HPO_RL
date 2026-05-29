"""
CNN HPO comparison: HMM_MCMC vs Optuna (TPE, BOHB-style).

Compares three methods on CIFAR-100 with a small CNN (500 HPO trials each by default).
Plots best-so-far validation loss vs trial index; saves CSV and a summary table.
Prints validation loss and accuracy after each training epoch.

Usage:
    python compare_hmm_mcmc_vs_tpe.py
    python compare_hmm_mcmc_vs_tpe.py --seeds 3 --trials 500 --epochs 5

BOHB here = TPESampler + HyperbandPruner (Optuna has no dedicated BOHB sampler).
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
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from hpo_rl.baselines.HMM_MCMC import HMM_MCMC

# ── Defaults ──────────────────────────────────────────────────────────
N_TRIALS = 1
N_SEEDS = 3
TRAIN_SUBSET = 10_000
VAL_SUBSET = 2_000
EPOCHS_PER_TRIAL = 5
RNG_BASE = 42

OUT_DIR = Path(__file__).parent.parent / "logs" / "compare_hmm_mcmc_vs_tpe"
RESULTS_CSV = str(OUT_DIR / "compare_hmm_mcmc_cifar_history.csv")
RESULTS_TXT = str(OUT_DIR / "compare_hmm_mcmc_cifar_results.txt")
PLOT_FILE = str(OUT_DIR / "compare_hmm_mcmc_cifar_convergence.png")

# HMM_MCMC hyperparameters (aligned with test_hmm_vs_optuna.py style)
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

# Search space for HMM_MCMC (values + type schema)
HMM_SPACE = {
    "learning_rate": {"values": [1e-5, 1e-1], "type": "float", "log": False},
    "weight_decay": {"values": [1e-6, 1e-2], "type": "float", "log": False},
    "dropout": {"values": [0.0, 0.7], "type": "float", "log": False},
    "batch_size": {"values": [32, 64, 128, 256], "type": "categorical"},
    "optimizer": {"values": ["adam", "sgd", "adamw"], "type": "categorical"},
}


class SmallCNN(nn.Module):
    """Light CNN for CIFAR-100 (32x32)."""

    def __init__(self, num_classes: int = 100, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.adapt = nn.AdaptiveAvgPool2d((2, 2))
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adapt(x)
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def prepare_data(
    data_root: str,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    tf_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),
        ]
    )
    tf_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),
        ]
    )
    train_set = datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=tf_train
    )
    val_set = datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=tf_val
    )
    train_loader = DataLoader(
        Subset(train_set, train_indices.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_set, val_indices.tolist()),
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Returns (mean cross-entropy loss, accuracy in [0, 1])."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item()
            total += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
    n = max(total, 1)
    return loss_sum / n, correct / n


def train_one_trial(
    config: dict,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    trial: optuna.Trial | None = None,
    metrics_out: list[tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """
    Train SmallCNN; optionally report per-epoch val loss for pruning.
    Returns (best validation loss over epochs, val accuracy [0,1] at that epoch).
    If metrics_out is given, appends exactly one (loss, acc) per call (incl. pruned).
    """
    lr = float(config["learning_rate"])
    wd = float(config["weight_decay"])
    dropout = float(config["dropout"])
    opt_name = str(config["optimizer"])

    model = SmallCNN(num_classes=100, dropout=dropout).to(device)
    optim = _build_optimizer(opt_name, model.parameters(), lr, wd)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()

        val_loss, val_acc = evaluate(model, val_loader, device)
        if val_loss < best_val:
            best_val = val_loss
            best_acc = val_acc

        if trial is not None:
            print(
                f"[optuna trial {trial.number}] epoch {epoch + 1}/{epochs} "
                f"val_loss={val_loss:.4f} val_acc={100.0 * val_acc:.2f}%",
                flush=True,
            )
        else:
            print(
                f"epoch {epoch + 1}/{epochs} val_loss={val_loss:.4f} "
                f"val_acc={100.0 * val_acc:.2f}%",
                flush=True,
            )

        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                if metrics_out is not None:
                    metrics_out.append((float(best_val), float(best_acc)))
                raise optuna.TrialPruned()

    if metrics_out is not None:
        metrics_out.append((float(best_val), float(best_acc)))
    return float(best_val), float(best_acc)


def _train_indices_for_seed(
    base_seed: int, train_subset: int, val_subset: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(base_seed)
    all_idx = np.arange(50_000)
    rng.shuffle(all_idx)
    train_idx = all_idx[:train_subset]
    val_idx = all_idx[train_subset : train_subset + val_subset]
    return train_idx, val_idx


def scores_to_best_curve(scores: list[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return np.array([], dtype=np.float64)
    return np.minimum.accumulate(arr)


def pad_curve(y: np.ndarray, length: int) -> np.ndarray:
    if len(y) >= length:
        return y[:length].astype(np.float64)
    if len(y) == 0:
        return np.full(length, np.nan, dtype=np.float64)
    pad = np.full(length - len(y), y[-1], dtype=np.float64)
    return np.concatenate([y.astype(np.float64), pad])


def build_hmm_objective(
    device: torch.device,
    data_root: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    epochs: int,
    metrics_out: list[tuple[float, float]],
):
    """Returns objective_func(config) -> val_loss; appends (loss, acc) per eval."""

    def objective_func(cfg: dict) -> float:
        bs = int(cfg["batch_size"])
        train_loader, val_loader = prepare_data(
            data_root, train_idx, val_idx, batch_size=bs
        )
        loss, acc = train_one_trial(
            cfg, device, train_loader, val_loader, epochs, trial=None, metrics_out=None
        )
        metrics_out.append((loss, acc))
        return loss

    return objective_func


def run_hmm_mcmc(
    seed: int,
    budget: int,
    device: torch.device,
    data_root: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    epochs: int,
) -> tuple[list[float], list[tuple[float, float]]]:
    _seed_all(seed)
    np.random.seed(seed)
    metrics: list[tuple[float, float]] = []
    obj = build_hmm_objective(
        device, data_root, train_idx, val_idx, epochs, metrics
    )
    alg = HMM_MCMC(
        objective_func=obj,
        budget=budget,
        dict_to_optimize=HMM_SPACE,
        **HMM_PARAMS,
    )
    alg.main_loop()

    return [float(s) for _, s in alg.data], metrics


def suggest_config(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 1e-1, log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-6, 1e-2, log=True
        ),
        "dropout": trial.suggest_float("dropout", 0.0, 0.7),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd", "adamw"]),
    }


def run_optuna_study(
    seed: int,
    n_trials: int,
    device: torch.device,
    data_root: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    epochs: int,
    sampler: optuna.samplers.BaseSampler,
    pruner: optuna.pruners.BasePruner | None,
    *,
    show_progress_bar: bool = True,
) -> tuple[list[float], list[float]]:
    _seed_all(seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    per_trial_by_id: dict[int, tuple[float, float]] = {}

    if pruner is None:

        def objective(trial: optuna.Trial) -> float:
            cfg = suggest_config(trial)
            bs = int(cfg["batch_size"])
            train_loader, val_loader = prepare_data(
                data_root, train_idx, val_idx, batch_size=bs
            )
            loss, acc = train_one_trial(
                cfg, device, train_loader, val_loader, epochs, trial=None
            )
            per_trial_by_id[trial.number] = (loss, acc)
            return loss

        study = optuna.create_study(direction="minimize", sampler=sampler)
    else:

        def objective_prune(trial: optuna.Trial) -> float:
            cfg = suggest_config(trial)
            bs = int(cfg["batch_size"])
            train_loader, val_loader = prepare_data(
                data_root, train_idx, val_idx, batch_size=bs
            )
            buf: list[tuple[float, float]] = []
            try:
                loss, acc = train_one_trial(
                    cfg,
                    device,
                    train_loader,
                    val_loader,
                    epochs,
                    trial=trial,
                    metrics_out=buf,
                )
            except optuna.TrialPruned:
                loss, acc = buf[0]
            per_trial_by_id[trial.number] = (loss, acc)
            return loss

        objective = objective_prune
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )

    losses: list[float] = []
    accs: list[float] = []
    for t in study.trials:
        loss_i, acc_i = per_trial_by_id.get(
            t.number, (float("inf"), float("nan"))
        )
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
            losses.append(float(t.value))
            accs.append(float(acc_i))
        elif t.state == optuna.trial.TrialState.PRUNED:
            vals = []
            for iv in t.intermediate_values.values():
                try:
                    vals.append(float(iv))
                except (TypeError, ValueError):
                    pass
            losses.append(float(min(vals)) if vals else float("inf"))
            accs.append(float(acc_i))
        else:
            losses.append(float("inf"))
            accs.append(float("nan"))

    return losses, accs


def run_optuna_tpe(
    seed: int,
    n_trials: int,
    device: torch.device,
    data_root: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    epochs: int,
    *,
    show_progress_bar: bool = True,
) -> tuple[list[float], list[float]]:
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=32)
    return run_optuna_study(
        seed,
        n_trials,
        device,
        data_root,
        train_idx,
        val_idx,
        epochs,
        sampler,
        None,
        show_progress_bar=show_progress_bar,
    )


def run_optuna_bohb(
    seed: int,
    n_trials: int,
    device: torch.device,
    data_root: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    epochs: int,
    *,
    show_progress_bar: bool = True,
) -> tuple[list[float], list[float]]:
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=32)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=epochs,
        reduction_factor=3,
    )
    return run_optuna_study(
        seed,
        n_trials,
        device,
        data_root,
        train_idx,
        val_idx,
        epochs,
        sampler,
        pruner,
        show_progress_bar=show_progress_bar,
    )


def plot_convergence(
    curves: dict[str, list[np.ndarray]],
    n_trials: int,
    n_seeds: int,
    outfile: str,
) -> None:
    evals = np.arange(1, n_trials + 1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    styles = [
        ("HMM_MCMC", "C0", "-"),
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
        f"CIFAR-100 small CNN HPO (n_trials={n_trials}, mean ± std over {n_seeds} seeds)"
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
        f"{'Method':<14} | {'loss mean±std':>20} | {'l_min':>8} | {'l_max':>8} "
        f"| {'acc% mean±std':>16} | {'a_min%':>7} | {'a_max%':>7}"
    )
    sep = "-" * len(hdr)
    p(sep)
    p(hdr)
    p(sep)
    for method in ("HMM_MCMC", "TPE", "BOHB"):
        xs = finals_loss.get(method, [])
        accs = finals_acc.get(method, [])
        if not xs or not accs:
            p(f"{method:<14} | {'n/a':>20} | {'n/a':>8} | {'n/a':>8} | {'n/a':>16} | {'n/a':>7} | {'n/a':>7}")
            continue
        m, s = float(np.mean(xs)), float(np.std(xs))
        am = float(np.mean(accs) * 100.0)
        astd = float(np.std(accs) * 100.0)
        amin = float(min(accs) * 100.0)
        amax = float(max(accs) * 100.0)
        p(
            f"{method:<14} | {m:9.4f} ± {s:7.4f} | {min(xs):8.4f} | {max(xs):8.4f} | "
            f"{am:7.2f}% ± {astd:5.2f}% | {amin:7.2f} | {amax:7.2f}"
        )
    p(sep)


def write_csv(
    path: str,
    rows: list[dict],
) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote CSV: {path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS_PER_TRIAL)
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root for CIFAR-100 download",
    )
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
    args = parser.parse_args()
    optuna_prog = not args.quiet_optuna

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    methods_curves: dict[str, list[np.ndarray]] = {
        "HMM_MCMC": [],
        "TPE": [],
        "BOHB": [],
    }
    finals_loss: dict[str, list[float]] = {k: [] for k in methods_curves}
    finals_acc: dict[str, list[float]] = {k: [] for k in methods_curves}
    csv_rows: list[dict] = []

    for s in range(args.seeds):
        seed = RNG_BASE + s
        train_idx, val_idx = _train_indices_for_seed(
            seed, args.train_subset, args.val_subset
        )

        # HMM_MCMC
        hmm_losses, hmm_metrics = run_hmm_mcmc(
            seed,
            args.trials,
            device,
            args.data_root,
            train_idx,
            val_idx,
            args.epochs,
        )
        hmm_accs = [m[1] for m in hmm_metrics]
        hmm_best = scores_to_best_curve(hmm_losses)
        methods_curves["HMM_MCMC"].append(pad_curve(hmm_best, args.trials))
        finals_loss["HMM_MCMC"].append(
            float(hmm_best[-1]) if len(hmm_best) else float("nan")
        )
        if hmm_losses and hmm_accs:
            _j = int(np.argmin(hmm_losses))
            finals_acc["HMM_MCMC"].append(float(hmm_accs[_j]))
        else:
            finals_acc["HMM_MCMC"].append(float("nan"))
        best_acc_so_far_hmm = np.maximum.accumulate(
            np.asarray(hmm_accs, dtype=np.float64)
        )
        for i, (cur, best, cur_acc, ba_sf) in enumerate(
            zip(
                hmm_losses,
                hmm_best.tolist(),
                hmm_accs,
                best_acc_so_far_hmm.tolist(),
            ),
            start=1,
        ):
            csv_rows.append(
                {
                    "method": "HMM_MCMC",
                    "seed": seed,
                    "trial": i,
                    "best_loss_so_far": best,
                    "current_loss": cur,
                    "trial_val_acc": cur_acc,
                    "best_val_acc_so_far": ba_sf,
                }
            )

        # TPE
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
        curve = scores_to_best_curve(tpe_losses)
        methods_curves["TPE"].append(pad_curve(curve, args.trials))
        finals_loss["TPE"].append(float(curve[-1]) if len(curve) else float("nan"))
        if tpe_losses and tpe_accs:
            _j = int(np.argmin(tpe_losses))
            finals_acc["TPE"].append(float(tpe_accs[_j]))
        else:
            finals_acc["TPE"].append(float("nan"))
        best_so_far = np.minimum.accumulate(np.asarray(tpe_losses, dtype=np.float64))
        best_acc_so_far_tpe = np.maximum.accumulate(
            np.asarray(tpe_accs, dtype=np.float64)
        )
        for i in range(1, len(tpe_losses) + 1):
            csv_rows.append(
                {
                    "method": "TPE",
                    "seed": seed,
                    "trial": i,
                    "best_loss_so_far": float(best_so_far[i - 1]),
                    "current_loss": float(tpe_losses[i - 1]),
                    "trial_val_acc": float(tpe_accs[i - 1]),
                    "best_val_acc_so_far": float(best_acc_so_far_tpe[i - 1]),
                }
            )

        # BOHB (TPE + HyperbandPruner)
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
        curve_b = scores_to_best_curve(bohb_losses)
        methods_curves["BOHB"].append(pad_curve(curve_b, args.trials))
        finals_loss["BOHB"].append(
            float(curve_b[-1]) if len(curve_b) else float("nan")
        )
        if bohb_losses and bohb_accs:
            _j = int(np.argmin(bohb_losses))
            finals_acc["BOHB"].append(float(bohb_accs[_j]))
        else:
            finals_acc["BOHB"].append(float("nan"))
        best_so_far_b = np.minimum.accumulate(
            np.asarray(bohb_losses, dtype=np.float64)
        )
        best_acc_so_far_b = np.maximum.accumulate(
            np.asarray(bohb_accs, dtype=np.float64)
        )
        for i in range(1, len(bohb_losses) + 1):
            csv_rows.append(
                {
                    "method": "BOHB",
                    "seed": seed,
                    "trial": i,
                    "best_loss_so_far": float(best_so_far_b[i - 1]),
                    "current_loss": float(bohb_losses[i - 1]),
                    "trial_val_acc": float(bohb_accs[i - 1]),
                    "best_val_acc_so_far": float(best_acc_so_far_b[i - 1]),
                }
            )

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
        "final_best_val_loss_per_method": finals_loss,
        "final_val_acc_at_best_loss_trial_per_method": finals_acc,
    }
    with open(args.txt, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2))
        f.write("\n\n")
        print_summary_table(finals_loss, finals_acc, out_stream=f)
    print(f"\nWrote summary: {args.txt}")


if __name__ == "__main__":
    main()
