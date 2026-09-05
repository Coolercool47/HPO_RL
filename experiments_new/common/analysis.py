"""Generic analysis for a result directory: tables, tests, ranks, CD diagram, curves.

    analyze(out_dir, fig_dir, ref_method="TPE", methods=None, exp_name="")

Outputs (in fig_dir):
    runs.csv                one row per run
    summary_<col>.csv       per task x method mean/std/sem/median/IQR
    tests_vs_<ref>.csv      per-task Mann-Whitney (Holm-corrected) for each method vs ref
    cross_task.json         per-suite Wilcoxon (each method vs ref) + Friedman/Nemenyi ranks
    cd_<suite>.png          critical-difference diagrams
    convergence_<suite>.png mean +- SEM best-so-far curves (log regret for synthetic)
    table_<suite>.tex       LaTeX tables with bold best and significance stars
    overhead.csv            wall-clock per evaluation per method
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments_new.common import io as IO
from experiments_new.common import plots as P
from experiments_new.common import stats as S


def analyze(out_dir: Path, fig_dir: Path, ref_method: str = "TPE", methods: list[str] | None = None,
            exp_name: str = "", curve_ncols: int = 3, min_seeds: int = 2) -> dict:
    out_dir, fig_dir = Path(out_dir), Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = IO.load_results(out_dir, methods=methods)
    if df.empty:
        print(f"[analyze] no results in {out_dir}")
        return {}
    df = S.normalized_regret(df, ref_method="RS")
    df.to_csv(fig_dir / "runs.csv", index=False)
    methods = methods or sorted(df["method"].unique())
    if ref_method not in methods:
        ref_method = methods[0]
    report: dict = {"exp": exp_name, "n_runs": int(len(df)), "methods": methods, "ref": ref_method}

    # value column: synthetic -> true objective at incumbent; lcbench -> -accuracy
    df["value"] = df["best_true_value"]
    summ = S.summary(df, "value")
    summ.to_csv(fig_dir / "summary_value.csv", index=False)
    if df["nregret"].notna().any():
        S.summary(df[df["nregret"].notna()], "nregret").to_csv(fig_dir / "summary_nregret.csv", index=False)
    tests = S.per_task_tests(df, "value", ref_method)
    tests.to_csv(fig_dir / f"tests_vs_{ref_method}.csv", index=False)

    cross: dict = {}
    for suite, g in df.groupby("suite"):
        agg_col = "nregret" if g["nregret"].notna().all() else "value"
        table = S.task_method_table(g, agg_col, "mean")
        table = table[[m for m in methods if m in table.columns]]
        fn = S.friedman_nemenyi(table)
        wil = {m: S.cross_task_wilcoxon(table, m, ref_method) for m in table.columns if m != ref_method}
        cross[suite] = {"agg_col": agg_col, "n_tasks": int(table.shape[0]), "friedman_nemenyi": fn, "wilcoxon_vs_ref": wil}
        if "cd" in fn:
            S.cd_diagram(fn["avg_ranks"], fn["cd"], fig_dir / f"cd_{suite}.png", sig_matrix=fn.get("nemenyi_p"),
                         title=f"{exp_name} / {suite}: ranks on {agg_col}, N = {table.shape[0]} tasks")
        # LaTeX table
        sum_s = summ[summ["suite"] == suite]
        lower_better = True
        if suite == "lcbench":  # report accuracy = -value
            sum_s = sum_s.copy()
            sum_s["mean"] = -sum_s["mean"]
            lower_better = False
        tex = S.latex_table(sum_s, sig=tests[tests["suite"] == suite] if not tests.empty else None, ref_method=ref_method,
                            methods=methods, caption=f"{exp_name} {suite}: mean $\\pm$ SEM over seeds; * = Holm-corrected Mann--Whitney $p<0.05$ vs {ref_method}.",
                            label=f"tab:{exp_name}_{suite}", lower_better=lower_better)
        (fig_dir / f"table_{suite}.tex").write_text(tex, encoding="utf-8")
    with open(fig_dir / "cross_task.json", "w", encoding="utf-8") as fh:
        json.dump(cross, fh, indent=2, default=float)
    report["cross_task"] = cross

    # wins table vs ref
    if not tests.empty:
        wins = tests.groupby(["method"]).agg(n_tasks=("task", "count"), wins=("better", "sum"), sig_wins=("sig05", lambda s: int(((s) & tests.loc[s.index, "better"]).sum())),
                                            sig_losses=("sig05", lambda s: int(((s) & ~tests.loc[s.index, "better"]).sum())))
        wins.to_csv(fig_dir / f"wins_vs_{ref_method}.csv")
        report["wins"] = wins.to_dict()

    # overhead
    ov = df.groupby("method").agg(time_total=("time_total", "mean"), time_objective=("time_objective", "mean"),
                                  time_overhead=("time_overhead", "mean"), n_evals=("n_evals", "mean"))
    ov["overhead_per_eval_ms"] = 1000 * ov["time_overhead"] / ov["n_evals"]
    ov.to_csv(fig_dir / "overhead.csv")

    # curves
    for suite, g in df.groupby("suite"):
        tasks = sorted(g["task"].unique())
        on_cost = suite == "lcbench" and "TPE_HB" in methods
        curves = IO.best_so_far_curves(out_dir, methods=methods, tasks=tasks, on_cost=on_cost, grid=400 if on_cost else None)
        curves = {k: v for k, v in curves.items() if v.shape[0] >= min_seeds}
        if not curves:
            continue
        if suite == "lcbench":
            total = {t: float(g[g["task"] == t]["cost_total"].max()) for t in tasks}
            P.plot_convergence(curves, tasks, methods, fig_dir / f"convergence_{suite}.png", band="sem",
                               ylabel="best validation accuracy", transform=lambda c, t: -c, ncols=curve_ncols,
                               xlabel="cumulative training epochs" if on_cost else "evaluation", title=exp_name,
                               x_of=(lambda t, n: np.linspace(1, total[t], n)) if on_cost else None)
        else:
            fstar = {t: float(g[g["task"] == t]["f_star"].iloc[0]) for t in tasks}
            P.plot_convergence(curves, tasks, methods, fig_dir / f"convergence_{suite}.png", band="sem",
                               ylabel="log10 regret of incumbent", ncols=curve_ncols,
                               transform=lambda c, t: np.log10(np.maximum(c - fstar[t], 0) + 1e-12), title=exp_name)
    print(f"[analyze] {exp_name}: {len(df)} runs, {df['task'].nunique()} tasks, methods={methods} -> {fig_dir}")
    for suite, d in cross.items():
        fn = d["friedman_nemenyi"]
        print(f"  {suite}: N={d['n_tasks']} avg ranks {', '.join(f'{k}={v:.2f}' for k, v in fn.get('avg_ranks', {}).items())}"
              + (f"  Friedman p={fn['friedman_p']:.3g} CD={fn['cd']:.2f}" if "cd" in fn else ""))
        for m, w in d["wilcoxon_vs_ref"].items():
            print(f"     {m} vs {ref_method}: wins {w.get('wins_a')}/{w['n']}  Wilcoxon p={w['p']:.3g}" if w.get("p") == w.get("p") else f"     {m} vs {ref_method}: n={w['n']} (too few tasks)")
    return report
