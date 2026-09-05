"""Statistics for the rebuttal: per-task tests with Holm correction, cross-task
Wilcoxon signed-rank, Friedman + Nemenyi with a critical-difference diagram,
normalized regret, and LaTeX table emission.

Conventions: `value_col` is in *minimize* orientation (loss / negative accuracy).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def task_method_table(df: pd.DataFrame, value_col: str, agg: str = "mean") -> pd.DataFrame:
    """tasks x methods matrix of aggregated values."""
    return df.pivot_table(index="task", columns="method", values=value_col, aggfunc=agg)


def summary(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    g = df.groupby(["suite", "task", "method"])[value_col]
    out = g.agg(mean="mean", std="std", sem=lambda x: x.std(ddof=1) / math.sqrt(len(x)), median="median",
                q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75), n="count").reset_index()
    return out


def normalized_regret(df: pd.DataFrame, ref_method: str = "RS") -> pd.DataFrame:
    """Add `nregret` = (f - f*) / (median_ref(f) - f*) per task (synthetic, f* known) and
    `log_regret` = log10(f - f* + 1e-12). For tasks without f* (LCBench) `nregret` is NaN."""
    df = df.copy()
    df["nregret"] = np.nan
    df["log_regret"] = np.nan
    for task, g in df.groupby("task"):
        if g["f_star"].isna().all():
            continue
        f_star = float(g["f_star"].iloc[0])
        ref = g[g["method"] == ref_method]["best_true_value"]
        denom = (float(ref.median()) - f_star) if len(ref) else np.nan
        idx = g.index
        r = g["best_true_value"] - f_star
        df.loc[idx, "log_regret"] = np.log10(np.maximum(r, 0.0) + 1e-12)
        if np.isfinite(denom) and denom > 0:
            df.loc[idx, "nregret"] = r / denom
    return df


# ---------------------------------------------------------------------------
# per-task tests
# ---------------------------------------------------------------------------

def holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (n - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj


def per_task_tests(df: pd.DataFrame, value_col: str, ref_method: str, alternative: str = "two-sided") -> pd.DataFrame:
    """Mann-Whitney U per task, each method vs `ref_method`, Holm-corrected across tasks
    within each method. `win` = method better (lower) than ref by median."""
    rows = []
    for method in sorted(df["method"].unique()):
        if method == ref_method:
            continue
        for task, g in df.groupby("task"):
            a = g[g["method"] == method][value_col].dropna().values
            b = g[g["method"] == ref_method][value_col].dropna().values
            if len(a) < 2 or len(b) < 2:
                continue
            p = stats.mannwhitneyu(a, b, alternative=alternative).pvalue
            rows.append(dict(task=task, suite=g["suite"].iloc[0], method=method, ref=ref_method, n_a=len(a), n_b=len(b),
                             mean_a=a.mean(), mean_b=b.mean(), median_a=np.median(a), median_b=np.median(b),
                             better=bool(np.median(a) < np.median(b)), p=p))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = np.nan
    for method, idx in out.groupby("method").groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"].values)
    out["sig05"] = out["p_holm"] < 0.05
    return out


# ---------------------------------------------------------------------------
# cross-task tests
# ---------------------------------------------------------------------------

def cross_task_wilcoxon(table: pd.DataFrame, a: str, b: str) -> dict:
    """Wilcoxon signed-rank on the per-task aggregated values of methods a and b."""
    x = table[[a, b]].dropna()
    if len(x) < 5:
        return dict(a=a, b=b, n=len(x), p=np.nan, wins_a=int((x[a] < x[b]).sum()), note="n<5")
    try:
        p = stats.wilcoxon(x[a], x[b]).pvalue
    except ValueError:
        p = np.nan
    return dict(a=a, b=b, n=len(x), p=p, wins_a=int((x[a] < x[b]).sum()), wins_b=int((x[b] < x[a]).sum()))


def friedman_nemenyi(table: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Average ranks (1 = best, lower value better), Friedman test, Nemenyi CD."""
    t = table.dropna(axis=0, how="any")
    k = t.shape[1]
    n = t.shape[0]
    ranks = t.rank(axis=1, method="average")
    avg = ranks.mean(axis=0).sort_values()
    out = dict(n_tasks=n, n_methods=k, avg_ranks=avg.to_dict())
    if n >= 2 and k >= 3:
        out["friedman_p"] = float(stats.friedmanchisquare(*[t[c].values for c in t.columns]).pvalue)
        # Nemenyi critical difference (Demsar 2006), q_alpha from studentized range / sqrt(2)
        q = _q_alpha(k, alpha)
        out["cd"] = float(q * math.sqrt(k * (k + 1) / (6.0 * n)))
        try:
            import scikit_posthocs as sp

            long = t.reset_index().melt(id_vars="task", var_name="method", value_name="v")
            ph = sp.posthoc_nemenyi_friedman(long, y_col="v", block_col="task", group_col="method", melted=True)
            out["nemenyi_p"] = ph.to_dict()
        except Exception:  # noqa: BLE001
            pass
    return out


def _q_alpha(k: int, alpha: float = 0.05) -> float:
    """Critical values of the studentized range statistic / sqrt(2) for the Nemenyi test (alpha=0.05)."""
    table05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
               11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}
    table10 = {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920}
    tbl = table05 if alpha <= 0.05 else table10
    return tbl[min(max(k, 2), max(tbl))]


def cd_diagram(avg_ranks: dict, cd: float, path: Path, title: str = "", sig_matrix=None, alpha: float = 0.05) -> None:
    """Critical-difference diagram (Demsar 2006).

    Reading it: the axis is the average rank over tasks (1 = best). Each method hangs
    from its rank. A horizontal bar connects methods whose ranks differ by less than
    the critical difference CD (Nemenyi post-hoc), i.e. methods that are *not*
    significantly different at level alpha. With few tasks CD is large and everything
    is connected -- the diagram only becomes informative with >= ~10 tasks.
    Uses scikit-posthocs' implementation when available (sig_matrix = Nemenyi p-values).
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranks = pd.Series(avg_ranks).sort_values()
    k = len(ranks)
    try:
        import scikit_posthocs as sp

        if sig_matrix is None:
            names = list(ranks.index)
            sig_matrix = pd.DataFrame(1.0, index=names, columns=names)
            for a in names:
                for b in names:
                    if a != b and abs(ranks[a] - ranks[b]) >= cd:
                        sig_matrix.loc[a, b] = 0.0
        else:
            sig_matrix = pd.DataFrame(sig_matrix).loc[ranks.index, ranks.index]
        fig, ax = plt.subplots(figsize=(8, 0.45 * k + 1.8))
        sp.critical_difference_diagram(ranks, sig_matrix, ax=ax, label_fmt_left="{label} ({rank:.2f})  ",
                                       label_fmt_right="  {label} ({rank:.2f})", crossbar_props={"linewidth": 3})
        ax.set_title((title + "\n" if title else "")
                     + f"average rank (1 = best); bars join methods not significantly different (CD = {cd:.2f}, alpha = {alpha})",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    except Exception:  # noqa: BLE001  fallback: bar chart of ranks with CD reference
        pass
    fig, ax = plt.subplots(figsize=(7, 0.4 * k + 1.5))
    y = np.arange(k)
    ax.barh(y, ranks.values, color="#1f77b4")
    ax.set_yticks(y)
    ax.set_yticklabels(ranks.index)
    ax.invert_yaxis()
    ax.set_xlabel("average rank (1 = best)")
    ax.axvline(ranks.values[0] + cd, color="r", ls="--", lw=1, label=f"best + CD ({cd:.2f})")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

def latex_table(summary_df: pd.DataFrame, value_fmt: str = "{:.3g}", bold_best: bool = True,
                sig: pd.DataFrame | None = None, ref_method: str | None = None,
                methods: list[str] | None = None, caption: str = "", label: str = "",
                mean_col: str = "mean", err_col: str = "sem", lower_better: bool = True) -> str:
    """Rows = tasks, columns = methods, cells = mean +- err, best in bold, '*' when
    significantly different from `ref_method` after Holm correction."""
    methods = methods or sorted(summary_df["method"].unique())
    piv_m = summary_df.pivot(index="task", columns="method", values=mean_col)
    piv_e = summary_df.pivot(index="task", columns="method", values=err_col)
    sig_map = {}
    if sig is not None and not sig.empty:
        for _, r in sig.iterrows():
            sig_map[(r["task"], r["method"])] = bool(r["sig05"])
    lines = ["\\begin{tabular}{l" + "c" * len(methods) + "}", "\\hline",
             "Task & " + " & ".join(m.replace("_", "\\_") for m in methods) + " \\\\", "\\hline"]
    for task in piv_m.index:
        vals = piv_m.loc[task, methods]
        best = (vals.idxmin() if lower_better else vals.idxmax()) if bold_best else None
        cells = []
        for m in methods:
            v, e = piv_m.loc[task, m], piv_e.loc[task, m]
            if pd.isna(v):
                cells.append("--")
                continue
            s = (value_fmt + "$\\pm$" + value_fmt).format(v, e)
            if m == best:
                s = "\\textbf{" + s + "}"
            if sig_map.get((task, m)):
                s += "$^{*}$"
            cells.append(s)
        lines.append(task.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    body = "\n".join(lines)
    if caption:
        body = "\\begin{table}[t]\n\\caption{" + caption + "}\n" + (f"\\label{{{label}}}\n" if label else "") + \
               "\\begin{center}\\small\n" + body + "\n\\end{center}\n\\end{table}"
    return body
