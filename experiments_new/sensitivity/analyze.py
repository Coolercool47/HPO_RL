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

from experiments_new.common.logging_util import start_log  # noqa: E402
from experiments_new.common import io as IO  # noqa: E402
from experiments_new.common import plots as P  # noqa: E402


def main():
    start_log(Path(__file__).resolve().parent, "analyze")
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
            g = g.assign(score=b - g["best_true_value"])     # accuracy gain over base in points; 0 = base; HIGHER better
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
                               title=f"OAT sensitivity ({suite_name}): " + ("regret / base regret, lower is better (dashed = base config)" if suite_name == "synthetic" else "accuracy - base accuracy in points, higher is better (dashed = base config)"),
                               base_value=1.0 if suite_name == "synthetic" else 0.0)
        summ = oat.groupby(["knob", "level", "task"])["score"].agg(["mean", "sem", "count"]).reset_index()
        summ.to_csv(figs / "oat_summary.csv", index=False)
        # knob influence per suite (the two scores have different units): range across levels of
        # the task-averaged score, plus the level that is best on average and per task
        summ["suite"] = np.where(summ["task"].str.startswith("lcbench"), "lcbench", "synthetic")
        rows_i = []
        for (suite, knob), g in summ.groupby(["suite", "knob"]):
            m = g.groupby("level")["mean"].mean()
            better = m.idxmax() if suite == "lcbench" else m.idxmin()
            per_task = g.loc[(g.groupby("task")["mean"].idxmax() if suite == "lcbench" else g.groupby("task")["mean"].idxmin())]
            rows_i.append({"suite": suite, "knob": knob, "range": m.max() - m.min(), "best_level": better,
                           "worst_level": m.idxmin() if suite == "lcbench" else m.idxmax(),
                           "best_level_per_task": "; ".join(f"{t.replace('lcbench_', '').replace('_10d', '')}={l:g}" for t, l in zip(per_task["task"], per_task["level"]))})
        infl = pd.DataFrame(rows_i).sort_values(["suite", "range"], ascending=[True, False])
        infl.to_csv(figs / "oat_influence.csv", index=False)
        print("OAT influence per suite (synthetic: regret ratio, lcbench: accuracy points):\n", infl.round(3).to_string(index=False))
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
            imps = {}
            for task, rt in rnd.groupby("task"):
                lc = str(task).startswith("lcbench")
                study = optuna.create_study(direction="minimize")
                agg = rt.groupby("method")["score"].mean()
                for m, v in agg.items():
                    # minimise log regret ratio (synthetic) or negative accuracy gain (lcbench)
                    val = -float(v) if lc else float(np.log10(max(v, 1e-9)))
                    study.add_trial(optuna.trial.create_trial(params=meta[m]["sample"], distributions=dists, value=val))
                imps[task] = optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))
            imp = pd.DataFrame(imps)
            imp["mean_synthetic"] = imp[[c for c in imps if not c.startswith("lcbench")]].mean(axis=1)
            imp["mean_lcbench"] = imp[[c for c in imps if c.startswith("lcbench")]].mean(axis=1)
            imp = imp.sort_values("mean_lcbench", ascending=False)
            imp.to_csv(figs / "fanova_importance.csv")
            print("fANOVA importances per task:\n", imp.round(3).to_string())
            # spread of random configurations = how much a bad configuration costs
            sp = rnd.groupby(["task", "method"])["score"].mean().groupby("task").describe(percentiles=[0.1, 0.5, 0.9])
            sp.to_csv(figs / "random_config_spread.csv")
            print("score of 200 random configurations (per task):\n", sp.round(3).to_string())
        except Exception as e:  # noqa: BLE001
            print("fANOVA failed:", e)


if __name__ == "__main__":
    main()
