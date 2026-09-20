"""Anytime performance from stored runs (no new runs): best accuracy reached within the
first b full-fidelity evaluations (multi-fidelity methods: within b * eval_cost cost units).

    python experiments_new/common/anytime.py experiments_new/lcbench/results [--budgets 50 100 200]
Writes <figures dir>/anytime.csv (mean accuracy and average rank per method and budget).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--budgets", nargs="*", type=int, default=[50, 100, 200])
    a = ap.parse_args()
    res = Path(a.results)
    figs = res.parent / res.name.replace("results", "figures")
    rows = []
    for p in sorted(res.glob("*/*/seed_*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        vals = np.asarray(rec["values"], dtype=float)
        tv = np.asarray(rec["true_values"], dtype=float) if rec.get("true_values") else vals
        costs = np.asarray(rec["costs"], dtype=float) if rec.get("costs") else np.arange(1, len(vals) + 1) * rec.get("eval_cost", 1.0)
        full = np.asarray(rec["fidelity"], dtype=float) >= 0.999 * max(rec["fidelity"]) if rec.get("fidelity") else np.ones(len(vals), bool)
        for b in a.budgets:
            m = (costs <= b * float(rec.get("eval_cost", 1.0)) + 1e-9) & full
            if not m.any():
                continue
            i = int(np.argmin(np.where(m, vals, np.inf)))
            rows.append(dict(suite=rec["suite"], task=rec["task"], method=rec["method"], seed=rec["seed"], budget=b, acc=-tv[i] if rec.get("maximize_raw") else tv[i]))
    d = pd.DataFrame(rows)
    out = []
    for (suite, b), g in d.groupby(["suite", "budget"]):
        t = g.groupby(["task", "method"])["acc"].mean().unstack()
        hib = bool(g["acc"].mean() > 0 and suite in ("lcbench",) or str(suite).startswith("rbv2"))
        ranks = t.rank(axis=1, ascending=not hib).mean()
        for m in t.columns:
            out.append(dict(suite=suite, budget=b, method=m, mean=t[m].mean(), avg_rank=ranks[m], n_tasks=int(t[m].notna().sum())))
    o = pd.DataFrame(out)
    figs.mkdir(parents=True, exist_ok=True)
    o.to_csv(figs / "anytime.csv", index=False)
    for suite, g in o.groupby("suite"):
        print(f"== {suite}: mean (avg rank)")
        print(g.assign(cell=g.apply(lambda r: f"{r['mean']:.2f} ({r['avg_rank']:.2f})", axis=1)).pivot(index="method", columns="budget", values="cell").to_string())


if __name__ == "__main__":
    sys.exit(main())
