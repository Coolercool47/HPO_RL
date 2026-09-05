"""E5: HMM controller diagnostics from stored FMP runs (no new workload needed).

    python experiments_new/diagnostics/run.py --results experiments_new/synt_functions/results --method FMP_DREAM
    python experiments_new/diagnostics/run.py --fresh cont__rastrigin_10d FMP_SOFT 0   # run one seed now and plot

Figures (diagnostics/figures/<results-name>/):
    states_<task>_<method>_seed<k>.png    decoded state / posterior / temperature trajectory
    obs_hist_<suite>_<method>.png         O_t histogram by state vs hand-set emissions + offline GMM
    A_evolution_<suite>_<method>.png      Baum-Welch A estimates vs prior
    occupancy_<method>.csv                state occupancy & acceptance per state per task
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments_new.common import io as IO  # noqa: E402
from experiments_new.common import plots as P  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO_ROOT / "experiments_new" / "synt_functions" / "results"))
    ap.add_argument("--method", default="FMP_DREAM")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--fresh", nargs=3, metavar=("TASK_KEY", "METHOD", "SEED"), default=None)
    ap.add_argument("--figs", default=None)
    a = ap.parse_args()
    res = Path(a.results)
    figs = Path(a.figs) if a.figs else HERE / "figures" / res.parent.name
    figs.mkdir(parents=True, exist_ok=True)

    if a.fresh:
        from experiments_new.common import methods as M
        from experiments_new.common.runner import load_fmp_base
        from experiments_new.common.spaces import build_task, synthetic_specs
        from experiments_new.lcbench import config as L

        key, method, seed = a.fresh[0], a.fresh[1], int(a.fresh[2])
        specs = {s if isinstance(s, str) else None: None for s in []}
        all_specs = synthetic_specs(10) + L.TASKS
        from experiments_new.common.spaces import spec_key

        spec = next(s for s in all_specs if spec_key(s) == key)
        task = build_task(spec, seed)
        rec = M.run_method(method, M.method_spec(method, load_fmp_base("table5")), task, seed, 500 if spec["kind"] == "synthetic" else 200)
        P.plot_state_trajectory(rec, figs / f"states_{key}_{method}_seed{seed}.png")
        P.plot_observation_histogram([rec], figs / f"obs_hist_{key}_{method}_seed{seed}.png")
        print("written", figs)
        return

    recs = IO.load_fmp_histories(res, methods=[a.method], tasks=a.tasks)
    recs = [r for r in recs if "history" in r["fmp"]]
    if not recs:
        print("no FMP histories found in", res)
        return
    by_suite: dict[str, list] = {}
    occ_rows = []
    for r in recs:
        by_suite.setdefault(r["suite"], []).append(r)
        s = r["fmp"]["summary"]
        occ_rows.append({"task": r["task"], "suite": r["suite"], "method": r["method"], "seed": r["seed"], **s})
        if r["seed"] in a.seeds:
            P.plot_state_trajectory(r, figs / f"states_{r['task']}_{r['method']}_seed{r['seed']}.png")
    pd.DataFrame(occ_rows).to_csv(figs / f"occupancy_{a.method}.csv", index=False)
    occ = pd.DataFrame(occ_rows).groupby("suite")[["frac_exploit", "frac_explore", "frac_trapped", "acc_exploit", "acc_explore",
                                                  "acc_trapped", "acceptance_rate", "n_bw_refits", "n_rescues", "n_reseeds"]].mean()
    print(occ.round(3).to_string())
    for suite, rs in by_suite.items():
        mu = rs[0]["params"].get("emission_mu", (-0.01, 0.10, 0.015))
        sd = rs[0]["params"].get("emission_sigma", (0.06, 0.25, 0.02))
        P.plot_observation_histogram(rs, figs / f"obs_hist_{suite}_{a.method}.png", emission_mu=mu, emission_sigma=sd,
                                     title=f"{suite} / {a.method}: O_t by decoded state ({len(rs)} runs)")
        P.plot_transition_evolution(rs, figs / f"A_evolution_{suite}_{a.method}.png", title=f"{suite} / {a.method}: Baum-Welch A vs prior")
    print("figures ->", figs)


if __name__ == "__main__":
    main()
