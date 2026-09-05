"""Loading per-run JSON results into pandas / numpy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def iter_records(out_dir: Path, methods: list[str] | None = None, tasks: list[str] | None = None,
                 light: bool = True):
    """Yield result dicts. light=True drops the heavy per-eval lists except best-so-far."""
    out_dir = Path(out_dir)
    for p in sorted(out_dir.glob("*/*/seed_*.json")):
        task_key, method = p.parts[-3], p.parts[-2]
        if methods and method not in methods:
            continue
        if tasks and not any(t in task_key for t in tasks):
            continue
        with open(p, encoding="utf-8") as fh:
            rec = json.load(fh)
        if light:
            rec = {k: v for k, v in rec.items() if k not in ("configs", "fmp")}
        yield rec


def load_results(out_dir: Path, methods=None, tasks=None) -> pd.DataFrame:
    """One row per run with scalar summaries (+ FMP summary fields when present)."""
    rows = []
    for p in sorted(Path(out_dir).glob("*/*/seed_*.json")):
        task_key, method = p.parts[-3], p.parts[-2]
        if methods and method not in methods:
            continue
        if tasks and not any(t in task_key for t in tasks):
            continue
        with open(p, encoding="utf-8") as fh:
            rec = json.load(fh)
        row = {
            "task": rec["task"], "suite": rec["suite"], "method": rec["method"], "seed": rec["seed"],
            "budget": rec["budget"], "n_evals": rec["n_evals"], "best_value": rec["best_value"],
            "best_true_value": rec["best_true_value"], "best_raw": rec["best_raw"], "f_star": rec.get("f_star"),
            "maximize_raw": rec.get("maximize_raw", False), "time_total": rec["time_total"],
            "time_objective": rec["time_objective"], "wall_time": rec.get("wall_time", np.nan),
            "cost_total": rec["costs"][-1] if rec["costs"] else np.nan,
        }
        row["time_overhead"] = row["time_total"] - row["time_objective"]
        if "fmp" in rec and "summary" in rec["fmp"]:
            for k, v in rec["fmp"]["summary"].items():
                row[f"fmp_{k}"] = v
            row["fmp_params"] = json.dumps(rec.get("params", {}), sort_keys=True)
        # regret helpers
        if rec.get("f_star") is not None:
            row["regret"] = row["best_true_value"] - rec["f_star"]
        if row["maximize_raw"]:
            row["best_metric"] = -row["best_true_value"]      # accuracy
        else:
            row["best_metric"] = row["best_true_value"]
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def best_so_far_curves(out_dir: Path, methods=None, tasks=None, use_true: bool = True,
                       on_cost: bool = False, grid: int | None = None) -> dict:
    """{(task, method): 2-D array seeds x budget of best-so-far values (minimize orientation)}.

    on_cost=True resamples on a cumulative-cost grid (LCBench epochs) so that
    multi-fidelity runs (TPE_HB) are comparable with full-fidelity ones.
    """
    curves: dict = {}
    for rec in iter_records(out_dir, methods, tasks, light=True):
        vals = np.asarray(rec["true_values"] if (use_true and rec.get("true_values")) else rec["values"], dtype=float)
        fid = np.asarray(rec["fidelity"], dtype=float)
        full = fid >= rec["eval_cost"] - 1e-9
        if on_cost:
            cost = np.asarray(rec["costs"], dtype=float)
            total = rec["budget"] * rec["eval_cost"]
            g = np.arange(1, int(total) + 1) if grid is None else np.linspace(1, total, grid)
            v = np.where(full, vals, np.inf)
            curve = np.full(len(g), np.nan)
            best = np.inf
            j = 0
            for gi, c in enumerate(g):
                while j < len(cost) and cost[j] <= c + 1e-9:
                    best = min(best, v[j])
                    j += 1
                curve[gi] = best if np.isfinite(best) else np.nan
        else:
            v = vals[full]
            curve = np.minimum.accumulate(v) if v.size else np.array([])
            if curve.size < rec["budget"]:
                curve = np.concatenate([curve, np.full(rec["budget"] - curve.size, curve[-1] if curve.size else np.nan)])
            curve = curve[: rec["budget"]]
        curves.setdefault((rec["task"], rec["method"]), []).append(curve)
    return {k: np.vstack(v) for k, v in curves.items()}


def load_fmp_histories(out_dir: Path, methods=None, tasks=None) -> list[dict]:
    """Full records including FMP per-eval history (heavy)."""
    out = []
    for p in sorted(Path(out_dir).glob("*/*/seed_*.json")):
        task_key, method = p.parts[-3], p.parts[-2]
        if methods and method not in methods:
            continue
        if tasks and not any(t in task_key for t in tasks):
            continue
        with open(p, encoding="utf-8") as fh:
            rec = json.load(fh)
        if "fmp" in rec:
            out.append(rec)
    return out
