"""E6 analysis: one-at-a-time curves (normalized to the base config per task) and
fANOVA importances from the random-configuration part.

    python experiments_new/sensitivity/analyze.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments_new.common import io as IO  # noqa: E402
from experiments_new.common import plots as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--figs", default=str(HERE / "figures"))
    a = ap.parse_args()
    res, figs = Path(a.results), Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    with open(res / "_meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    df = IO.load_results(res)
    if df.empty:
        print("no results")
        return
    df["knob"] = df["method"].map(lambda m: meta.get(m, {}).get("knob"))
    df["level"] = df["method"].map(lambda m: meta.get(m, {}).get("level"))
    # score relative to base config: for synthetic, regret ratio; for lcbench, accuracy difference
    rows = []
    for task, g in df.groupby("task"):
        base = g[g["method"] == "S__base"]
        if base.empty:
            continue
        if g["f_star"].notna().all():
            f_star = float(g["f_star"].iloc[0])
            b = float(np.median(base["best_true_value"] - f_star))
            g = g.assign(score=(g["best_true_value"] - f_star) / max(b, 1e-12))   # 1 = base; lower better
        else:
            b = float(base["best_true_value"].mean())
            g = g.assign(score=g["best_true_value"] - b)                           # 0 = base; lower better
        rows.append(g)
    d = pd.concat(rows)
    d.to_csv(figs / "sensitivity_runs.csv", index=False)
    oat = d[d["knob"].notna() & (d["knob"] != "random") & (d["knob"] != "base")].copy()
    if not oat.empty:
        oat["level"] = oat["level"].astype(float)
        knobs = sorted(oat["knob"].unique())
        # synthetic and lcbench on separate panels because the score has different meaning
        for suite_name, mask in (("synthetic", oat["f_star"].notna()), ("lcbench", oat["f_star"].isna())):
            sub = oat[mask]
            if sub.empty:
                continue
            P.plot_sensitivity(sub, knobs, figs / f"oat_{suite_name}.png", value_col="score",
                               title=f"OAT sensitivity ({suite_name}): " + ("regret / base regret (dashed = base config)" if suite_name == "synthetic" else "accuracy - base accuracy (dashed = base config)"),
                               base_value=1.0 if suite_name == "synthetic" else 0.0)
        summ = oat.groupby(["knob", "level", "task"])["score"].agg(["mean", "sem", "count"]).reset_index()
        summ.to_csv(figs / "oat_summary.csv", index=False)
        # knob influence: range of task-averaged score across levels
        infl = summ.groupby(["knob", "level"])["mean"].mean().groupby("knob").agg(lambda s: s.max() - s.min()).sort_values(ascending=False)
        infl.to_csv(figs / "oat_influence.csv")
        print("OAT influence (range of mean score across levels):\n", infl.round(3).to_string())
    rnd = d[d["knob"] == "random"]
    if len(rnd) > 20:
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            dists = {}
            from experiments_new.sensitivity.run import INT_KNOBS, RANDOM_SPACE

            for k, (lo, hi, log) in RANDOM_SPACE.items():
                dists[k] = (optuna.distributions.IntDistribution(int(lo), int(hi), log=log) if k in INT_KNOBS
                            else optuna.distributions.FloatDistribution(float(lo), float(hi), log=log))
            study = optuna.create_study(direction="minimize")
            agg = rnd.groupby("method")["score"].mean()
            for m, v in agg.items():
                sample = meta[m]["sample"]
                study.add_trial(optuna.trial.create_trial(params=sample, distributions=dists, value=float(np.log10(max(v, 1e-9)))))
            imp = optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))
            pd.Series(imp).to_csv(figs / "fanova_importance.csv")
            print("fANOVA importances:\n", pd.Series(imp).round(3).to_string())
        except Exception as e:  # noqa: BLE001
            print("fANOVA failed:", e)


if __name__ == "__main__":
    main()
