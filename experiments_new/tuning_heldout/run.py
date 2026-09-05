"""E0: held-out meta-tuning of ONE shared configuration per method on LCBench
instances that are disjoint from the test set (configs/lcbench_instances.json 'tune').

For each meta-trial the candidate configuration is run on every tuning instance
x TUNE_SEEDS (200 evaluations each); the meta-objective is the mean normalized
accuracy  (acc - RS_median) / (RS_best - RS_median)  using the survey statistics,
so instances with different accuracy scales contribute equally.

Methods tuned with the same meta-budget: FMP (ladder step iv config, DREAM
probability included as a knob), TPE, CMAES. GP-BO and RS have no knobs worth tuning
beyond n_startup and are left at defaults.

    python experiments_new/tuning_heldout/run.py --dry-run
    python experiments_new/tuning_heldout/run.py --smoke
    python experiments_new/tuning_heldout/run.py --method FMP --n-trials 100 --workers 10
Outputs configs/fmp_tuned.json and configs/baselines_tuned.json (merged).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments_new.common import methods as M  # noqa: E402
from experiments_new.common.runner import CONFIG_DIR, run_job  # noqa: E402
from experiments_new.lcbench import config as L  # noqa: E402
from experiments_new.sensitivity.run import INT_KNOBS, RANDOM_SPACE, apply_knob  # noqa: E402

HERE = Path(__file__).resolve().parent
TUNE_SEEDS = [1000, 1001, 1002]
BUDGET = 200


def _survey_stats() -> dict:
    with open(CONFIG_DIR / "lcbench_instances.json", encoding="utf-8") as fh:
        return json.load(fh)["survey"]


def _normalized(acc: float, inst: str, survey: dict) -> float:
    s = survey[inst]
    return (acc - s["median"]) / max(s["best"] - s["median"], 1e-6)


def suggest_fmp(trial) -> dict:
    p = {}
    for k, (lo, hi, log) in RANDOM_SPACE.items():
        if k in INT_KNOBS:
            p[k] = trial.suggest_int(k, int(lo), int(hi), log=log)
        else:
            p[k] = trial.suggest_float(k, float(lo), float(hi), log=log)
    return p


def suggest_tpe(trial) -> dict:
    return {"n_startup_trials": trial.suggest_int("n_startup_trials", 4, 64, log=True),
            "multivariate": trial.suggest_categorical("multivariate", [True, False]),
            "n_ei_candidates": trial.suggest_int("n_ei_candidates", 8, 64, log=True),
            "consider_endpoints": trial.suggest_categorical("consider_endpoints", [True, False])}


def suggest_cmaes(trial) -> dict:
    return {"n_startup_trials": trial.suggest_int("n_startup_trials", 4, 64, log=True),
            "sigma0": trial.suggest_float("sigma0", 0.05, 1.0, log=True),
            "popsize": trial.suggest_int("popsize", 4, 32, log=True)}


SUGGEST = {"FMP": suggest_fmp, "TPE": suggest_tpe, "CMAES": suggest_cmaes}


def make_spec(method: str, sample: dict, fmp_base: dict) -> dict:
    if method == "FMP":
        params = {**fmp_base, **M.FMP_VARIANTS["FMP_SOFT"]}
        for k, v in sample.items():
            params = apply_knob(params, k, v)
        return {"kind": "fmp", "params": params}
    spec = dict(M.BASELINES[method])
    spec["params"] = dict(sample)
    return spec


def meta_objective(method: str, sample: dict, tasks: list[dict], seeds: list[int], survey: dict, workers: int,
                   fmp_base: dict, out_dir: Path, trial_no: int) -> float:
    spec = make_spec(method, sample, fmp_base)
    jobs = [dict(exp=f"tune_{method}", task_spec=t, task_key=f"lcbench_{t['instance']}", method=f"trial_{trial_no:04d}",
                 method_spec=spec, seed=s, budget=BUDGET, n_startup=int(spec.get("params", {}).get("n_startup_trials", 16)),
                 keep_history=False) for t in tasks for s in seeds]
    if workers > 1:
        from joblib import Parallel, delayed

        res = Parallel(n_jobs=workers, backend="loky")(delayed(run_job)(j, out_dir, save=True) for j in jobs)
    else:
        res = [run_job(j, out_dir, save=True) for j in jobs]
    scores = []
    for j, r in zip(jobs, res):
        if not r.get("ok"):
            print("   inner run failed:", r.get("error"))
            scores.append(-1.0)
            continue
        acc = -r["best_true_value"]
        scores.append(_normalized(acc, j["task_spec"]["instance"], survey))
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="FMP", choices=list(SUGGEST))
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--n-startup", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="*", default=TUNE_SEEDS)
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--fmp-config", default="table5")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--meta-seed", type=int, default=7)
    a = ap.parse_args()

    from experiments_new.common.runner import load_fmp_base

    fmp_base = load_fmp_base(a.fmp_config)
    survey = _survey_stats()
    instances = a.instances or L.TUNE_INSTANCES
    if not instances:
        raise SystemExit("no tuning instances: run experiments_new/lcbench/select_instances.py first")
    assert not (set(instances) & set(L.TEST_INSTANCES)), "tuning instances overlap the test set!"
    tasks = [dict(kind="lcbench", instance=i) for i in instances]
    seeds = list(a.seeds)
    n_trials = a.n_trials
    global BUDGET
    if a.smoke:
        tasks, seeds, n_trials, BUDGET = tasks[:2], seeds[:1], 3, 16 + 5
        a.n_startup = 2
    out_dir = HERE / ("results_smoke" if a.smoke else "results") / a.method
    print(f"[tune {a.method}] instances={instances} seeds={seeds} trials={n_trials} budget={BUDGET} workers={a.workers}")
    print(f"[tune] test instances (must be disjoint): {L.TEST_INSTANCES}")
    if a.dry_run:
        import optuna

        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=a.meta_seed))
        tr = st.ask()
        sample = SUGGEST[a.method](tr)
        spec = make_spec(a.method, sample, fmp_base)
        print("[dry-run] example candidate:", json.dumps(sample, default=float))
        print("[dry-run] resolved spec params keys:", sorted(spec.get("params", {}).keys()))
        print(f"[dry-run] inner runs per trial = {len(tasks)} x {len(seeds)} = {len(tasks) * len(seeds)}; total = {n_trials * len(tasks) * len(seeds)}")
        return
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(out_dir / 'meta_study.db').as_posix()}"
    study = optuna.create_study(direction="maximize", study_name=f"tune_{a.method}", storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=a.meta_seed, n_startup_trials=a.n_startup))
    t0 = time.perf_counter()
    while len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) < n_trials:
        trial = study.ask()
        sample = SUGGEST[a.method](trial)
        v = meta_objective(a.method, sample, tasks, seeds, survey, a.workers, fmp_base, out_dir, trial.number)
        study.tell(trial, v)
        done = len(study.trials)
        print(f"[tune {a.method}] trial {trial.number}: score={v:.4f} best={study.best_value:.4f} "
              f"({(time.perf_counter() - t0) / done:.0f}s/trial)", flush=True)
    best = dict(study.best_params)
    print("[tune] best:", json.dumps(best, default=float), "score", study.best_value)
    if a.method == "FMP":
        params = {**fmp_base}
        for k, v in best.items():
            params = apply_knob(params, k, v)
        params = {k: v for k, v in params.items() if k not in M.FMP_VARIANTS["FMP_SOFT"] or k == "p_dream"}
        target = CONFIG_DIR / ("fmp_tuned_smoke.json" if a.smoke else "fmp_tuned.json")
        with open(target, "w", encoding="utf-8") as fh:
            json.dump({**params, "_meta": {"score": study.best_value, "instances": instances, "seeds": seeds, "n_trials": n_trials}}, fh, indent=2, default=float)
    else:
        target = CONFIG_DIR / ("baselines_tuned_smoke.json" if a.smoke else "baselines_tuned.json")
        cur = {}
        if target.is_file():
            with open(target, encoding="utf-8") as fh:
                cur = json.load(fh)
        cur[a.method] = best
        cur.setdefault("_meta", {})[a.method] = {"score": study.best_value, "instances": instances, "seeds": seeds, "n_trials": n_trials}
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(cur, fh, indent=2, default=float)
    print("[tune] written", target)


if __name__ == "__main__":
    main()
