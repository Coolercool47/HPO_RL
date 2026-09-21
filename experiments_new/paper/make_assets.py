"""Generate every table and figure of the revised paper from the stored results.

    python experiments_new/paper/make_assets.py --paper C:/path/to/HMM_MCMC [--with-smac]

Writes <paper>/generated/*.tex and copies / draws figures into <paper>/figures/.
Nothing in the paper is typed by hand: all numbers in the tables come from here, and
generated/numbers.tex defines the macros used in the running text.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
E = HERE.parent
sys.path.insert(0, str(E.parent))

from experiments_new.common import io as IO  # noqa: E402
from experiments_new.common import stats as S  # noqa: E402

NAME = {"GP": "GP-BO", "SMAC": "SMAC", "TPE": "TPE", "CMAES": "CMA-ES", "TPE_HB": "TPE+HB", "RS": "Random search",
        "FMP": "H-MCMC-FMP", "FMP_DREAM": "H-MCMC-FMP+DREAM", "L1_K1_VITERBI_SUB": "FMP-only (submitted)"}
OURS = {"FMP", "FMP_DREAM"}


def accdf(res: Path, methods: list[str]) -> pd.DataFrame:
    df = IO.load_results(res, methods=methods)
    df["acc"] = -df["best_true_value"]
    return df


def anytime(res: Path, methods: list[str], budgets=(50, 100)) -> pd.DataFrame:
    rows = []
    for p in sorted(res.glob("*/*/seed_*.json")):
        if p.parts[-2] not in methods:
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        vals = np.asarray(rec["values"], float)
        tv = np.asarray(rec["true_values"], float) if rec.get("true_values") else vals
        costs = np.asarray(rec["costs"], float)
        fid = np.asarray(rec["fidelity"], float) if rec.get("fidelity") else np.ones(len(vals))
        full = fid >= 0.999 * fid.max()
        for b in budgets:
            m = (costs <= b * float(rec.get("eval_cost", 1.0)) + 1e-9) & full
            if m.any():
                i = int(np.argmin(np.where(m, vals, np.inf)))
                rows.append(dict(task=rec["task"], method=rec["method"], seed=rec["seed"], budget=b, acc=-tv[i]))
    return pd.DataFrame(rows)


def boot_ci(per_task: pd.Series, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    v = per_task.to_numpy()
    bs = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def sig_counts(df: pd.DataFrame, ref="TPE") -> pd.DataFrame:
    d = df.copy()
    d["value"] = d["best_true_value"]
    t = S.per_task_tests(d, "value", ref)
    t["sw"] = t["sig05"] & t["better"]
    t["sl"] = t["sig05"] & ~t["better"]
    return t.groupby("method")[["better", "sw", "sl"]].sum()


def fmt_name(m):
    return (r"\textbf{" + NAME[m] + "}") if m in OURS else NAME[m]


def table_lcbench(out: Path, methods: list[str], macros: dict):
    res, res5 = E / "lcbench/results", E / "lcbench/results_table5"
    df = accdf(res, methods)
    tab = df.groupby(["task", "method"])["acc"].mean().unstack()
    ranks = tab.rank(axis=1, ascending=False).mean()
    fn = S.friedman_nemenyi(-tab)
    sc = sig_counts(df)
    at = anytime(res, methods)
    a = at.groupby(["budget", "task", "method"])["acc"].mean().groupby(["budget", "method"]).mean().unstack(0)
    ov = pd.read_csv(E / "paper_assets/lcbench/overhead.csv").set_index("method")["overhead_per_eval_ms"]
    d5 = accdf(res5, methods + ["L1_K1_VITERBI_SUB"])
    t5 = d5.groupby(["task", "method"])["acc"].mean().groupby("method").mean()
    n = tab.shape[0]
    order = tab.mean().sort_values(ascending=False).index.tolist()
    pp = df.pivot_table(index=["task", "seed"], columns="method", values="acc")

    def ci_cell(m, ref="TPE"):
        if m == ref:
            return "--"
        mean_, lo_, hi_ = boot_ci((pp[m] - pp[ref]).groupby(level=0).mean())
        return f"${mean_:+.2f}$ [${lo_:+.2f}$, ${hi_:+.2f}$]"

    seedstat = df.groupby(["task", "method"])["acc"].agg(["min", "max"]).groupby("method").mean()
    lines = [r"\begin{tabular}{@{}lcccccclccr@{}}", r"\hline",
             r" & \multicolumn{3}{c}{\textbf{mean accuracy (\%) at $b$}} & \multicolumn{2}{c}{\textbf{seeds, $b{=}200$}} & & \multicolumn{2}{c}{\textbf{vs.\ TPE at $b{=}200$}} & \textbf{default} & \textbf{ms /} \\",
             r"\textbf{Method} & $50$ & $100$ & $200$ & worst & best & \textbf{rank} & difference [95\% CI] & s$+$ / s$-$ & $b{=}200$ & eval \\", r"\hline"]
    for m in methods:
        macros["LcWorst" + m.replace("_", "")] = f"{seedstat.loc[m, 'min']:.2f}"
        macros["LcBest" + m.replace("_", "")] = f"{seedstat.loc[m, 'max']:.2f}"
    best = {c: max(a.loc[m, c] for m in order) for c in (50, 100)}
    for m in order:
        cells = []
        for b in (50, 100):
            v = a.loc[m, b]
            cells.append((r"\textbf{%.2f}" if abs(v - best[b]) < 1e-9 else "%.2f") % v)
        v = tab[m].mean()
        cells.append((r"\textbf{%.2f}" if m == order[0] else "%.2f") % v)
        vs = "--" if m == "TPE" else f"{int(sc.loc[m, 'sw'])} / {int(sc.loc[m, 'sl'])}"
        un = f"{t5[m]:.2f}" if m in t5.index else "--"
        wb = " & ".join((r"\textbf{%.2f}" if abs(seedstat.loc[m, c] - seedstat[c].max()) < 1e-9 else "%.2f") % seedstat.loc[m, c] for c in ("min", "max"))
        lines.append(f"{fmt_name(m)} & {cells[0]} & {cells[1]} & {cells[2]} & {wb} & {ranks[m]:.2f} & {ci_cell(m)} & {vs} & {un} & {ov[m]:.1f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (out / "table_lcbench.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # paired differences
    p = df.pivot_table(index=["task", "seed"], columns="method", values="acc")
    for a_, b_, key in [("FMP_DREAM", "TPE", "LcDreamTpe"), ("FMP_DREAM", "CMAES", "LcDreamCma"), ("FMP_DREAM", "FMP", "LcDreamFmp"),
                        ("GP", "FMP_DREAM", "LcGpDream"), ("FMP_DREAM", "RS", "LcDreamRs"), ("FMP", "TPE", "LcFmpTpe"), ("FMP", "CMAES", "LcFmpCma"), ("GP", "FMP", "LcGpFmp"), ("FMP", "RS", "LcFmpRs")] +             ([("SMAC", "FMP_DREAM", "LcSmacDream"), ("SMAC", "TPE", "LcSmacTpe"), ("SMAC", "FMP", "LcSmacFmp")] if "SMAC" in methods else []):
        pt = (p[a_] - p[b_]).groupby(level=0).mean()
        m_, lo, hi = boot_ci(pt)
        macros[key] = f"{m_:+.2f}"
        macros[key + "CI"] = f"[{lo:+.2f}, {hi:+.2f}]"
        macros[key + "Wins"] = f"{int((pt > 0).sum())}"
    macros["LcN"] = str(n)
    macros["LcCD"] = f"{fn['cd']:.2f}"
    macros["LcFriedman"] = f"{fn['friedman_p']:.0e}".replace("e-", r"\times 10^{-").replace("{-0", "{-") + "}"
    for m in methods:
        macros["LcAcc" + m.replace("_", "")] = f"{tab[m].mean():.2f}"
        macros["LcRank" + m.replace("_", "")] = f"{ranks[m]:.2f}"
        macros["LcOv" + m.replace("_", "")] = f"{ov[m]:.0f}" if ov[m] >= 10 else f"{ov[m]:.1f}"
        if m in t5.index:
            macros["LcUntuned" + m.replace("_", "")] = f"{t5[m]:.2f}"
        for b in (50, 100):
            macros[f"LcAcc{'Fifty' if b == 50 else 'Hundred'}" + m.replace("_", "")] = f"{a.loc[m, b]:.2f}"
    macros["LcUntunedLone"] = f"{t5['L1_K1_VITERBI_SUB']:.2f}"
    sc5 = sig_counts(d5)
    macros["LcUntunedDreamSigLoss"] = str(int(sc5.loc["FMP_DREAM", "sl"]))
    macros["LcUntunedLoneSigLoss"] = str(int(sc5.loc["L1_K1_VITERBI_SUB", "sl"]))
    macros["LcDreamSigLoss"] = str(int(sc.loc["FMP_DREAM", "sl"]))
    macros["LcFmpSigLoss"] = str(int(sc.loc["FMP", "sl"]))
    macros["LcFmpSigWin"] = str(int(sc.loc["FMP", "sw"]))
    macros["LcFmpBetter"] = str(int(sc.loc["FMP", "better"]))
    macros["LcUntunedFmpSigLoss"] = str(int(sc5.loc["FMP", "sl"])) if "FMP" in sc5.index else "--"
    macros["LcTpeBetterThanFmp"] = str(int(n - sc.loc["FMP", "better"]))
    macros["LcDreamBetter"] = str(int(sc.loc["FMP_DREAM", "better"]))
    macros["LcCmaSigLoss"] = str(int(sc.loc["CMAES", "sl"]))
    macros["LcGpSigWin"] = str(int(sc.loc["GP", "sw"]))
    if "SMAC" in methods:
        macros["LcSmacSigWin"] = str(int(sc.loc["SMAC", "sw"]))
        macros["LcSmacBetter"] = str(int(sc.loc["SMAC", "better"]))


def table_rbv2(out: Path, methods: list[str], macros: dict):
    df = accdf(E / "rbv2/results", methods)
    df["acc"] *= 100
    blocks = {}
    for suite, g in df.groupby("suite"):
        tab = g.groupby(["task", "method"])["acc"].mean().unstack()
        ranks = tab.rank(axis=1, ascending=False).mean()
        sc = sig_counts(g)
        fn = S.friedman_nemenyi(-tab)
        blocks[suite] = (tab, ranks, sc, fn)
        p = g.pivot_table(index=["task", "seed"], columns="method", values="acc")
        tag = "Svm" if "svm" in suite else "Xgb"
        for a_, b_, key in [("FMP_DREAM", "TPE", "DreamTpe"), ("FMP_DREAM", "CMAES", "DreamCma"), ("FMP_DREAM", "FMP", "DreamFmp"), ("FMP_DREAM", "RS", "DreamRs"), ("FMP", "TPE", "FmpTpe"), ("FMP", "CMAES", "FmpCma"), ("FMP", "RS", "FmpRs")] +                 ([("SMAC", "FMP_DREAM", "SmacDream"), ("SMAC", "FMP", "SmacFmp")] if "SMAC" in methods else []):
            pt = (p[a_] - p[b_]).groupby(level=0).mean()
            m_, lo, hi = boot_ci(pt)
            macros[tag + key] = f"{m_:+.2f}"
            macros[tag + key + "CI"] = f"[{lo:+.2f}, {hi:+.2f}]"
            macros[tag + key + "Wins"] = str(int((pt > 0).sum()))
        macros[tag + "CD"] = f"{fn['cd']:.2f}"
        macros[tag + "DreamSigLoss"] = str(int(sc.loc["FMP_DREAM", "sl"]))
        macros[tag + "FmpSigLoss"] = str(int(sc.loc["FMP", "sl"]))
        macros[tag + "FmpSigWin"] = str(int(sc.loc["FMP", "sw"]))
        for m in methods:
            macros[tag + "Acc" + m.replace("_", "")] = f"{tab[m].mean():.2f}"
            macros[tag + "Rank" + m.replace("_", "")] = f"{ranks[m]:.2f}"
    order = (blocks["rbv2_svm"][1] + blocks["rbv2_xgboost"][1]).sort_values().index.tolist()
    ov = pd.read_csv(E / "paper_assets/rbv2/overhead.csv").set_index("method")["overhead_per_eval_ms"]
    lines = [r"\begin{tabular}{@{}lccclcccclc@{}}", r"\hline",
             r" & \multicolumn{5}{c}{\textbf{SVM} (kernel + 2 conditional)} & \multicolumn{5}{c}{\textbf{XGBoost} (booster + 8 conditional)} \\",
             r"\cline{2-6}\cline{7-11}",
             r"\textbf{Method} & mean & worst & best & diff.\ to TPE [95\% CI] & s$+$/s$-$ & mean & worst & best & diff.\ to TPE [95\% CI] & s$+$/s$-$ \\", r"\hline"]
    seedstat = {su: g.groupby(["task", "method"])["acc"].agg(["min", "max"]).groupby("method").mean() for su, g in df.groupby("suite")}
    for su, tag_ in (("rbv2_svm", "Svm"), ("rbv2_xgboost", "Xgb")):
        for m in methods:
            macros[tag_ + "Worst" + m.replace("_", "")] = f"{seedstat[su].loc[m, 'min']:.2f}"
            macros[tag_ + "Best" + m.replace("_", "")] = f"{seedstat[su].loc[m, 'max']:.2f}"
    piv = {su: g.pivot_table(index=["task", "seed"], columns="method", values="acc") for su, g in df.groupby("suite")}
    for m in order:
        cells = []
        for suite in ("rbv2_svm", "rbv2_xgboost"):
            tab, ranks, sc, _ = blocks[suite]
            bestm = tab.mean().idxmax()
            v = tab[m].mean()
            if m == "TPE":
                ci_, vs = "--", "--"
            else:
                mean_, lo_, hi_ = boot_ci((piv[suite][m] - piv[suite]["TPE"]).groupby(level=0).mean())
                ci_ = f"${mean_:+.2f}$ [${lo_:+.2f}$, ${hi_:+.2f}$]"
                vs = f"{int(sc.loc[m, 'sw'])}/{int(sc.loc[m, 'sl'])}"
            st = seedstat[suite]
            wb = " & ".join((r"\textbf{%.2f}" if abs(st.loc[m, c] - st[c].max()) < 1e-9 else "%.2f") % st.loc[m, c] for c in ("min", "max"))
            cells.append(((r"\textbf{%.2f}" if m == bestm else "%.2f") % v) + f" & {wb} & {ci_} & {vs}")
        lines.append(f"{fmt_name(m)} & {cells[0]} & {cells[1]} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (out / "table_rbv2.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


LADDER = [("L1_K1_VITERBI_SUB", "(i) one chain, Viterbi, subsampling"), ("L2_K4_ORCH", "(ii) $+$ four chains, orchestrator"),
          ("L3_NOSUB", "(iii) $-$ subsampling"), ("L4_SOFT", "(iv) $+$ soft decoding"),
          ("L5_DREAM", "(v) $+$ DREAM, $p_{\\mathrm{dream}} = 0.5$"), ("L5_DREAM_LEGACYKERNEL", "(v') (v) with asymmetric DE move")]
CTRL = [("CTRL_HMM", "HMM (= iv)"), ("CTRL_HMM_NOBW", "HMM, no Baum--Welch"), ("CTRL_HMM_LEARNEMIS", "HMM, learned emissions"),
        ("CTRL_HMM_VITERBI", "HMM, Viterbi decoding"), ("CTRL_RULE", "threshold rule on $O_t$"),
        ("CTRL_RANDOM", "random switching"), ("CTRL_FIXED", "always \\textsc{Exploit}")]


def table_ablation(out: Path, macros: dict):
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\hline",
             r"\textbf{Variant} & \textbf{LCBench acc.\ (\%)} & \textbf{better than ref.} & \textbf{Wilcoxon $p$} & \textbf{synthetic cont.\ $\log_{10}$ regret} \\", r"\hline"]
    for exp, rows, ref, head in (("ablation_ladder", LADDER, "L1_K1_VITERBI_SUB", r"\multicolumn{5}{@{}l}{\textit{Single-factor ladder (reference: i)}} \\"),
                                 ("ablation_controller", CTRL, "CTRL_HMM", r"\multicolumn{5}{@{}l}{\textit{Controller, all on variant (iv) (reference: HMM)}} \\")):
        df = IO.load_results(E / exp / "results")
        lc = df[df["suite"] == "lcbench"].copy()
        lc["acc"] = -lc["best_true_value"]
        tab = lc.groupby(["task", "method"])["acc"].mean().unstack()
        co = df[df["suite"] == "cont"].copy()
        co["lr"] = np.log10(np.maximum(co["best_true_value"] - co["f_star"], 0) + 1e-12)
        clr = co.groupby(["task", "method"])["lr"].mean().groupby("method").mean()
        lines.append(head)
        for m, label in rows:
            if m == ref:
                w, pv = "--", "--"
            else:
                r = S.cross_task_wilcoxon(tab, m, ref)
                w = f"{int((tab[m] > tab[ref]).sum())} / {tab.shape[0]}"
                pv = f"{r['p']:.1e}" if r["p"] < 0.001 else (f"{r['p']:.3f}" if r["p"] < 0.01 else f"{r['p']:.2f}")
                pv = pv.replace("e-0", r"{\times}10^{-").replace("e-", r"{\times}10^{-")
                pv = f"${pv}" + ("}$" if "times" in pv else "$")
            lines.append(f"\\quad {label} & {tab[m].mean():.2f} & {w} & {pv} & {clr[m]:.2f} \\\\")
            key = "Abl" + "".join(ch for ch in m.title() if ch.isalpha())
            macros[key] = f"{tab[m].mean():.2f}"
            if m != ref:
                macros[key + "P"] = f"{r['p']:.3f}" if r["p"] >= 0.001 else "<0.001"
                macros[key + "Wins"] = str(int((tab[m] > tab[ref]).sum()))
                macros[key + "Diff"] = f"{tab[m].mean() - tab[ref].mean():+.2f}"
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    (out / "table_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # compact side-by-side version for the main text: variant, LCBench accuracy, Wilcoxon p vs the block reference
    blocks = []
    for exp, rows, ref in (("ablation_ladder", LADDER, "L1_K1_VITERBI_SUB"), ("ablation_controller", CTRL, "CTRL_HMM")):
        df = IO.load_results(E / exp / "results")
        lc = df[df["suite"] == "lcbench"].copy()
        lc["acc"] = -lc["best_true_value"]
        tab = lc.groupby(["task", "method"])["acc"].mean().unstack()
        cells = []
        for m, label in rows:
            if m == ref:
                pv = "ref."
            else:
                pr = S.cross_task_wilcoxon(tab, m, ref)["p"]
                pv = "$<0.001$" if pr < 0.001 else (f"${pr:.3f}$" if pr < 0.01 else f"${pr:.2f}$")
            cells.append(f"{label} & {tab[m].mean():.2f} & {pv}")
        blocks.append(cells)
    n_rows = max(len(b) for b in blocks)
    comp = [r"\begin{tabular}{@{}lcc@{\hspace{1.5em}}lcc@{}}", r"\hline",
            r"\textbf{Component ladder} & acc.\ (\%) & $p$ vs (i) & \textbf{Controller, on variant (iv)} & acc.\ (\%) & $p$ vs HMM \\", r"\hline"]
    for i in range(n_rows):
        left = blocks[0][i] if i < len(blocks[0]) else " & & "
        right = blocks[1][i] if i < len(blocks[1]) else " & & "
        comp.append(f"{left} & {right} \\\\")
    comp += [r"\hline", r"\end{tabular}"]
    (out / "table_ablation_compact.tex").write_text("\n".join(comp) + "\n", encoding="utf-8")


def table_synth(out: Path, macros: dict):
    methods = ["GP", "CMAES", "FMP", "FMP_DREAM", "TPE", "RS"]
    df = IO.load_results(E / "synt_functions/results", methods=methods)
    df = S.normalized_regret(df, "RS")
    t = df.groupby(["suite", "task", "method"])["nregret"].mean().unstack()
    suites = [("cont", "continuous (9)"), ("noisy", "noisy (5)"), ("cat", "irrelevant categorical (5)")]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(methods) + "@{}}", r"\hline",
             r"\textbf{Suite} & " + " & ".join(NAME[m].replace(" (ours)", "") for m in methods) + r" \\", r"\hline"]
    for s, label in suites:
        row = t.loc[s].mean()
        best = row.idxmin()
        lines.append(label + " & " + " & ".join((r"\textbf{%.3f}" if m == best else "%.3f") % row[m] for m in methods) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    (out / "table_synth.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    d = df.copy()
    d["value"] = d["best_true_value"]
    tt = S.per_task_tests(d, "value", "TPE")
    for m, key in (("FMP", "SynFmp"), ("FMP_DREAM", "SynDream")):
        g = tt[tt["method"] == m]
        macros[key + "Better"] = str(int(g["better"].sum()))
        macros[key + "SigWin"] = str(int((g["sig05"] & g["better"]).sum()))
        macros[key + "SigLoss"] = str(int((g["sig05"] & ~g["better"]).sum()))
    # per-function appendix table
    sig = tt.set_index(["task", "method"])
    fl = [r"\begin{tabular}{@{}ll" + "c" * len(methods) + "@{}}", r"\hline",
          r"\textbf{Function} & \textbf{Suite} & " + " & ".join(NAME[m].replace(" (ours)", "") for m in methods) + r" \\", r"\hline"]
    for s, _ in suites:
        for task in t.loc[s].index:
            row = t.loc[(s, task)]
            best = row.idxmin()
            cells = []
            for m in methods:
                c = (r"\textbf{%.3f}" if m == best else "%.3f") % row[m]
                if m != "TPE" and (task, m) in sig.index and bool(sig.loc[(task, m), "sig05"]):
                    c += r"$^{+}$" if bool(sig.loc[(task, m), "better"]) else r"$^{-}$"
                cells.append(c)
            fname = task.split("__")[1].replace("_10d", "").replace("_", "--").title()
            fl.append(f"{fname} & {s} & " + " & ".join(cells) + r" \\")
        fl.append(r"\hline")
    fl.append(r"\end{tabular}")
    (out / "table_synth_full.tex").write_text("\n".join(fl) + "\n", encoding="utf-8")


SHORT = {"GP": "GP-BO", "SMAC": "SMAC", "TPE": "TPE", "CMAES": "CMA-ES", "TPE_HB": "TPE+HB", "RS": "Random search",
         "FMP": "H-MCMC-FMP", "FMP_DREAM": "H-MCMC-FMP+DREAM"}


def fig_cd(tab: pd.DataFrame, path: Path, width: float = 6.0):
    """Compact critical-difference diagram (Demsar 2006). tab: tasks x methods, higher is better."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = tab.rank(axis=1, ascending=False).mean().sort_values()
    k, n = len(ranks), tab.shape[0]
    cd = S._q_alpha(k) * np.sqrt(k * (k + 1) / (6.0 * n))
    names, vals = list(ranks.index), ranks.to_numpy()
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    fig, ax = plt.subplots(figsize=(width, 0.34 * k + 0.9))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.2, hi + 0.2)
    ax.set_ylim(-(k / 2 + 1.2), 1.4)
    ax.hlines(0, lo, hi, color="k", lw=1)
    for t in range(lo, hi + 1):
        ax.vlines(t, 0, 0.12, color="k", lw=1)
        ax.text(t, 0.2, str(t), ha="center", va="bottom", fontsize=8)
    ax.hlines(1.0, lo, lo + cd, color="k", lw=1.5)
    ax.text(lo + cd / 2, 1.08, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)
    half = (k + 1) // 2
    for i, (m, v) in enumerate(zip(names, vals)):
        left = i < half
        y = -(0.9 + 0.5 * (i if left else k - 1 - i)) - 0.35 * len(cliques)
        xt = lo - 0.15 if left else hi + 0.15
        ax.plot([v, v, xt], [0, y, y], color="C3" if m in OURS else "0.35", lw=1)
        ax.text(xt, y, f"{SHORT[m]} ({v:.2f})" if left else f"({v:.2f}) {SHORT[m]}", ha="right" if left else "left", va="center",
                fontsize=8, fontweight="bold" if m in OURS else "normal")
    for c, (a, b) in enumerate(cliques):
        yy = -0.25 - 0.3 * c
        ax.hlines(yy, vals[a] - 0.04, vals[b] + 0.04, color="k", lw=3)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(path)
    plt.close(fig)


def _curves(res: Path, methods, on_cost: bool):
    """{(task, method): seeds x T} of best-so-far accuracy; on_cost resamples on an epoch grid (LCBench, TPE_HB)."""
    c = IO.best_so_far_curves(res, methods=methods, on_cost=on_cost, grid=200 if on_cost else None)
    return {k: -v for k, v in c.items() if v.shape[0] >= 3}


def fig_rank_over_budget(figdir: Path, lc_methods, rb_methods):
    """Average rank (over tasks) of the mean best accuracy as a function of the budget spent."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments_new.common.plots import METHOD_COLORS

    panels = [("LCBench (26 tasks)", E / "lcbench/results", lc_methods, True, None),
              ("SVM (20 tasks)", E / "rbv2/results", rb_methods, False, "rbv2_svm"),
              ("XGBoost (20 tasks)", E / "rbv2/results", rb_methods, False, "rbv2_xgboost")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=False)
    for ax, (title, res, methods, on_cost, prefix) in zip(axes, panels):
        cur = _curves(res, methods, on_cost)
        tasks = sorted({t for t, _ in cur if prefix is None or t.startswith(prefix)})
        T = min(v.shape[1] for (t, _), v in cur.items() if t in tasks)
        ranks = {m: np.zeros(T) for m in methods}
        for t in tasks:
            mat = np.vstack([np.nanmean(cur[(t, m)][:, :T], axis=0) for m in methods])      # methods x T
            mat = np.where(np.isnan(mat), -np.inf, mat)
            order = (-mat).argsort(axis=0).argsort(axis=0) + 1.0
            # average ranks for ties
            for j in range(T):
                col = mat[:, j]
                for v in np.unique(col):
                    idx = np.where(col == v)[0]
                    if len(idx) > 1:
                        order[idx, j] = order[idx, j].mean()
            for i, m in enumerate(methods):
                ranks[m] += order[i] / len(tasks)
        x = np.linspace(1, 200, T)
        for m in methods:
            ax.plot(x, ranks[m], color=METHOD_COLORS.get(m), lw=2.2 if m in OURS else 1.4, label=SHORT[m])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("evaluations" + (" (epochs / 52)" if on_cost else ""), fontsize=10)
        ax.set_xlim(8, 200)
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("average rank (1 = best)", fontsize=10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(figdir / "rank_over_budget.pdf")
    plt.close(fig)


def fig_per_task(figdir: Path, res: Path, methods, prefix, fname, on_cost: bool, ncols: int = 3):
    """Readable per-task grid: mean best accuracy per method, one legend, y-range zoomed to the informative part."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments_new.common.plots import METHOD_COLORS

    cur = _curves(res, methods, on_cost)
    tasks = sorted({t for t, _ in cur if t.startswith(prefix)}, key=lambda t: int(t.split("_")[-1]))
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.75 * ncols, 1.3 * nrows + 0.4), squeeze=False)
    scale = 1.0 if prefix.startswith("lcbench") else 100.0
    for ax, t in zip(axes.ravel(), tasks):
        finals = []
        for m in methods:
            y = np.nanmean(cur[(t, m)], axis=0) * scale
            x = np.linspace(1, 200, len(y))
            ax.plot(x, y, color=METHOD_COLORS.get(m), lw=1.8 if m in OURS else 1.0, label=SHORT[m])
            finals.append(y[-1])
            if m == "RS":
                lo_ref = y[min(len(y) - 1, int(0.15 * len(y)))]
        top = max(finals)
        ax.set_ylim(min(lo_ref, min(finals)) - 0.05 * (top - min(finals) + 1e-9), top + 0.08 * (top - min(lo_ref, min(finals)) + 1e-9))
        ax.set_xlim(1, 200)
        ax.set_title(t.split("_")[-1], fontsize=9, pad=2)
        ax.tick_params(labelsize=7, pad=1)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(tasks):]:
        ax.axis("off")
    for r in range(nrows):
        axes[r, 0].set_ylabel("accuracy (%)", fontsize=8)
    for c in range(ncols):
        axes[nrows - 1, c].set_xlabel("evaluations", fontsize=8)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.5, w_pad=0.5)
    fig.savefig(figdir / fname)
    plt.close(fig)


def fig_synth(figdir: Path):
    """Log regret on the nine continuous synthetic functions with display names."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments_new.common.plots import METHOD_COLORS

    methods = ["GP", "CMAES", "FMP", "FMP_DREAM", "TPE", "RS"]
    res = E / "synt_functions/results"
    cur = IO.best_so_far_curves(res, methods=methods)
    df = IO.load_results(res, methods=["RS"])
    fstar = df.groupby("task")["f_star"].first().to_dict()
    tasks = sorted(t for t in {k[0] for k in cur} if t.startswith("cont__"))
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 6.4), squeeze=False)
    for ax, t in zip(axes.ravel(), tasks):
        for m in methods:
            y = np.log10(np.maximum(np.nanmean(cur[(t, m)], axis=0) - fstar[t], 1e-12))
            ax.plot(np.arange(1, len(y) + 1), y, color=METHOD_COLORS.get(m), lw=2.0 if m in OURS else 1.2, label=SHORT[m])
        ax.set_title(t.split("__")[1].replace("_10d", "").replace("_", "-").title(), fontsize=10, pad=3)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    for r in range(3):
        axes[r, 0].set_ylabel(r"$\log_{10}$ regret", fontsize=9)
    for c in range(3):
        axes[2, c].set_xlabel("evaluations", fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(figdir / "convergence_synth_cont.pdf")
    plt.close(fig)


def fig_sensitivity(figdir: Path, macros: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.read_csv(E / "sensitivity/figures_tuned/oat_summary.csv")
    d = d[d["task"].str.startswith("lcbench")]
    knobs = [("p_dream", r"$p_{\mathrm{dream}}$", False), ("emission_scale", r"emission scale", True), ("T_mcmc", r"temperature $T_0$", True),
             ("n_chains", r"chains $K$", True), ("rejection_streak", r"$L_{\mathrm{stag}}$", True), ("kde_tau", r"$\tau_{\mathrm{kde}}$", True)]
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 3.5), sharey=True)
    for ax, (k, label, logx) in zip(axes.ravel(), knobs):
        g = d[d["knob"] == k]
        for _, gt in g.groupby("task"):
            ax.plot(gt["level"], gt["mean"], color="0.75", lw=0.8)
        m = g.groupby("level")["mean"].mean()
        ax.plot(m.index, m.values, color="C3", lw=2, marker="o", ms=3)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(label, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(-6.5, 2.5)
    for r in range(2):
        axes[r, 0].set_ylabel("accuracy $-$ tuned (points)", fontsize=9)
    fig.tight_layout(pad=0.4)
    fig.savefig(figdir / "sensitivity_lcbench.pdf")
    plt.close(fig)
    all_knobs = [("T_mcmc", r"temperature $T_0$", True), ("burnin_fraction", r"burn-in fraction $\beta$", False),
                 ("rejection_streak", r"stagnation threshold $L_{\mathrm{stag}}$", True), ("n_init", r"initial points $n_{\mathrm{init}}$", True),
                 ("sigma_fraction", r"local step $\sigma_{\mathrm{frac}}$", True), ("wide_sigma_fraction", r"wide step $\sigma_{w,\mathrm{frac}}$", True),
                 ("kde_tau", r"archive temperature $\tau_{\mathrm{kde}}$", True), ("n_chains", r"chains $K$", True),
                 ("hmm_window", r"HMM window $W$", True), ("bw_prior_strength", r"Baum--Welch prior strength $\rho$", True),
                 ("emission_scale", r"emission scale", True), ("p_dream", r"$p_{\mathrm{dream}}$", False)]
    fig, axes = plt.subplots(3, 4, figsize=(10.5, 6.6), sharey=True)
    for ax, (k, label, logx) in zip(axes.ravel(), all_knobs):
        g = d[d["knob"] == k]
        for _, gt in g.groupby("task"):
            ax.plot(gt["level"], gt["mean"], color="0.75", lw=0.8)
        m = g.groupby("level")["mean"].mean()
        ax.plot(m.index, m.values, color="C3", lw=2, marker="o", ms=3)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(label.replace("--", "-"), fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(-6.5, 2.5)
    for r in range(3):
        axes[r, 0].set_ylabel("accuracy $-$ tuned (points)", fontsize=9)
    fig.tight_layout(pad=0.5)
    fig.savefig(figdir / "sensitivity_lcbench_all.pdf")
    plt.close(fig)
    inf = pd.read_csv(E / "sensitivity/figures_tuned/oat_influence.csv")
    inf = inf[inf["suite"] == "lcbench"].set_index("knob")["range"]
    for k in inf.index:
        macros["Sens" + "".join(ch for ch in k.title() if ch.isalpha())] = f"{inf[k]:.2f}"
    lv = d.groupby(["knob", "level"])["mean"].mean()
    macros["SensPdreamZero"] = f"{lv[('p_dream', 0.0)]:+.2f}"
    macros["SensPdreamHalf"] = f"{lv[('p_dream', 0.5)]:+.2f}"
    macros["SensPdreamOne"] = f"{lv[('p_dream', 1.0)]:+.2f}"
    macros["SensEmisTen"] = f"{lv[('emission_scale', 10.0)]:+.2f}"
    t_levels = sorted(l for k_, l in lv.index if k_ == "T_mcmc")
    macros["SensTHot"] = f"{lv[('T_mcmc', t_levels[-1])]:+.2f}"
    r_levels = sorted(l for k_, l in lv.index if k_ == "rejection_streak")
    macros["SensStreakLow"] = f"{lv[('rejection_streak', r_levels[0])]:+.2f}"
    macros["SensStreakHigh"] = f"{lv[('rejection_streak', r_levels[-1])]:+.2f}"
    macros["SensStreakLowLevel"] = f"{r_levels[0]:g}"
    macros["SensStreakHighLevel"] = f"{r_levels[-1]:g}"
    others = inf.drop(["p_dream", "emission_scale"])
    macros["SensOthersMax"] = f"{others.max():.2f}"
    sp = pd.read_csv(E / "sensitivity/figures_tuned/random_config_spread.csv", index_col=0)
    sp = sp[sp.index.str.startswith("lcbench")]
    med = sp["50%"].sort_values()
    macros["RandMedEasyLo"] = f"{-med.iloc[2:].max():.1f}"
    macros["RandMedEasyHi"] = f"{-med.iloc[2:].min():.1f}"
    macros["RandMedHardA"] = f"{-med.iloc[1]:.1f}"
    macros["RandMedHardB"] = f"{-med.iloc[0]:.1f}"
    macros["RandWorst"] = f"{-sp['min'].min():.0f}"
    fa = pd.read_csv(E / "sensitivity/figures_tuned/fanova_importance.csv", index_col=0)["mean_lcbench"].sort_values(ascending=False)
    for i_, name_ in enumerate(("FanovaFirst", "FanovaSecond", "FanovaThird")):
        macros[name_] = f"{fa.iloc[i_]:.2f}"
        macros[name_ + "Name"] = fa.index[i_].replace("_", "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    ap.add_argument("--with-smac", action="store_true")
    ap.add_argument("--skip-analyze", action="store_true", help="reuse paper_assets/ from a previous call")
    a = ap.parse_args()
    paper = Path(a.paper)
    out, figdir = paper / "generated", paper / "figures"
    out.mkdir(exist_ok=True)
    figdir.mkdir(exist_ok=True)
    macros: dict = {}
    lc_methods = ["RS", "TPE", "GP", "CMAES", "TPE_HB", "FMP", "FMP_DREAM"] + (["SMAC"] if a.with_smac else [])
    rb_methods = ["RS", "TPE", "GP", "CMAES", "FMP", "FMP_DREAM"] + (["SMAC"] if a.with_smac else [])
    from experiments_new.common import plots as P
    from experiments_new.common.analysis import analyze

    P.LABELS = dict(SHORT)
    if not a.skip_analyze:
        analyze(E / "lcbench/results", E / "paper_assets/lcbench", ref_method="TPE", methods=lc_methods, exp_name="", curve_ncols=6)
        analyze(E / "rbv2/results", E / "paper_assets/rbv2", ref_method="TPE", methods=rb_methods, exp_name="", curve_ncols=5)
    lcd = accdf(E / "lcbench/results", lc_methods)
    fig_cd(lcd.groupby(["task", "method"])["acc"].mean().unstack(), figdir / "cd_lcbench.pdf", width=4.4)
    rbd = accdf(E / "rbv2/results", rb_methods)
    for sc in ("rbv2_svm", "rbv2_xgboost"):
        fig_cd(rbd[rbd["suite"] == sc].groupby(["task", "method"])["acc"].mean().unstack(), figdir / f"cd_{sc}.pdf", width=4.6)
    fig_rank_over_budget(figdir, lc_methods, rb_methods)
    fig_per_task(figdir, E / "lcbench/results", lc_methods, "lcbench_", "per_task_lcbench.pdf", on_cost=True)
    fig_per_task(figdir, E / "rbv2/results", rb_methods, "rbv2_svm_", "per_task_rbv2_svm.pdf", on_cost=False)
    fig_per_task(figdir, E / "rbv2/results", rb_methods, "rbv2_xgboost_", "per_task_rbv2_xgboost.pdf", on_cost=False)
    fig_synth(figdir)
    table_lcbench(out, lc_methods, macros)
    table_rbv2(out, rb_methods, macros)
    table_ablation(out, macros)
    table_synth(out, macros)
    fig_sensitivity(figdir, macros)
    occ = pd.read_csv(E / "diagnostics/figures/lcbench/occupancy_FMP.csv")
    lc = occ[occ["suite"] == "lcbench"].mean(numeric_only=True)
    macros["OccExploit"] = f"{100 * lc['frac_exploit']:.1f}"
    macros["OccExplore"] = f"{100 * lc['frac_explore']:.1f}"
    macros["OccTrapped"] = f"{100 * lc['frac_trapped']:.1f}"
    macros["AccRate"] = f"{lc['acceptance_rate']:.2f}"
    for src, dst in [(E / "paper_assets/lcbench/convergence_lcbench.png", "convergence_lcbench.png"),
                     (E / "paper_assets/rbv2/convergence_rbv2_svm.png", "convergence_rbv2_svm.png"),
                     (E / "paper_assets/rbv2/convergence_rbv2_xgboost.png", "convergence_rbv2_xgboost.png"),
                     (E / "diagnostics/figures/lcbench/obs_hist_lcbench_FMP.png", "hmm_obs_hist_lcbench.png"),
                     (E / "diagnostics/figures/lcbench/states_lcbench_7593_FMP_seed0.png", "hmm_states_lcbench_7593.png"),
                     (E / "diagnostics/figures/lcbench/A_evolution_lcbench_FMP.png", "hmm_A_evolution_lcbench.png"),
                     (E / "sensitivity/figures_tuned/oat_lcbench.png", "sensitivity_lcbench_all.png"),
                     (E / "synt_functions/figures/convergence_cont.png", "convergence_synth_cont.png")]:
        if src.is_file():
            shutil.copyfile(src, figdir / dst)
        else:
            print("missing", src)
    cfg = json.loads((E / "configs/fmp_tuned.json").read_text(encoding="utf-8"))
    from hpo_rl.baselines.HMM_MCMC_FMP import DEFAULT_EMISSION_MU

    def sig2(v):
        return f"{v:.2g}" if v < 1 else (f"{v:.1f}" if v < 10 else f"{v:.0f}")

    for key, name in (("T_mcmc", "Tmcmc"), ("sigma_fraction", "Sigma"), ("wide_sigma_fraction", "Wide"), ("kde_tau", "KdeTau"),
                      ("bw_prior_strength", "Rho"), ("burnin_fraction", "Burnin"), ("p_dream", "Pdream")):
        macros["Tuned" + name] = sig2(float(cfg[key]))
    for key, name in (("hmm_window", "Window"), ("rejection_streak", "Streak"), ("n_init", "Ninit"), ("n_chains", "Chains")):
        macros["Tuned" + name] = str(int(cfg[key]))
    macros["TunedEmisScale"] = f"{float(cfg['emission_mu'][1]) / float(DEFAULT_EMISSION_MU[1]):.2f}"
    macros["TunedScore"] = f"{float(cfg['_meta']['score']):.2f}"
    bt = json.loads((E / "configs/baselines_tuned.json").read_text(encoding="utf-8"))
    macros["TunedScoreTPE"] = f"{float(bt['_meta']['TPE']['score']):.2f}"
    macros["TunedScoreCMAES"] = f"{float(bt['_meta']['CMAES']['score']):.2f}"
    with open(out / "numbers.tex", "w", encoding="utf-8") as fh:
        fh.write("% generated by experiments_new/paper/make_assets.py -- do not edit\n")
        for k, v in sorted(macros.items()):
            body = f"\\ensuremath{{{v}}}" if (v[:1] in "+-[<" or "times" in v) else v
            fh.write(f"\\newcommand{{\\num{k}}}{{{body}}}\n")
    print(f"{len(macros)} macros; tables in {out}")


if __name__ == "__main__":
    main()
