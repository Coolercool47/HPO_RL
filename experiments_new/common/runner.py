"""Job-grid runner shared by all experiments_new/* experiments.

Result layout (one JSON per run, so methods/seeds/tasks can be added later without
re-running anything):

    <out_dir>/<task_key>/<method>/seed_<seed>.json

CLI (see `cli`):
    --dry-run     build the job grid, validate every method on a tiny quadratic
                  (determinism: same seed -> identical trace, different seed -> different),
                  print the table, write nothing
    --smoke       real workload but budget = n_init + 5 evaluations, 1 seed, first 2 tasks
    --workers N   parallel worker processes (joblib / loky)
    --seeds a b   explicit seeds;  --n-seeds N -> seeds 0..N-1
    --budget B    evaluations per run
    --methods ... subset of method names
    --tasks ...   subset of task keys (substring match)
    --fmp-config  "table5" (class defaults) | "tuned" (configs/fmp_tuned.json) | path
    --out DIR     result directory (default experiments_new/<exp>/results)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_new.common import methods as M          # noqa: E402
from experiments_new.common.seeding import derive_seed, seed_everything  # noqa: E402
from experiments_new.common.spaces import build_task, spec_key  # noqa: E402

CONFIG_DIR = REPO_ROOT / "experiments_new" / "configs"


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------

def load_fmp_base(name_or_path: str = "table5") -> dict:
    """Shared FMP hyperparameters. 'table5' = class defaults (empty override)."""
    if name_or_path in ("table5", "default", ""):
        return {}
    p = Path(name_or_path)
    if not p.is_file():
        p = CONFIG_DIR / f"fmp_{name_or_path}.json"
    if not p.is_file():
        raise FileNotFoundError(f"FMP config not found: {name_or_path} (looked at {p})")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def load_baseline_params(name_or_path: str = "default") -> dict:
    if name_or_path in ("default", ""):
        p = CONFIG_DIR / "baselines_default.json"
    else:
        p = Path(name_or_path)
        if not p.is_file():
            p = CONFIG_DIR / f"baselines_{name_or_path}.json"
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

def make_jobs(exp: str, task_specs: list[dict], method_names: list[str], seeds: list[int], budget: int,
              fmp_base: dict, baseline_params: dict | None = None, n_startup: int | None = None,
              overrides: dict | None = None, keep_history: bool = True,
              custom_specs: dict | None = None) -> list[dict]:
    """overrides: {method_name: {param: value}} applied on top of the resolved spec params.
    custom_specs: {method_name: spec} for names outside the registry (sensitivity sweeps)."""
    jobs = []
    for spec in task_specs:
        for m in method_names:
            if custom_specs and m in custom_specs:
                mspec = custom_specs[m]
            else:
                mspec = M.method_spec(m, fmp_base, baseline_params)
            if overrides and m in overrides:
                mspec = dict(mspec)
                mspec["params"] = {**mspec.get("params", {}), **overrides[m]}
            b = int(spec.get("budget", budget))   # per-task budget override (e.g. LCBench 200 vs synthetic 500)
            for s in seeds:
                jobs.append(dict(exp=exp, task_spec=spec, task_key=spec_key(spec), method=m, method_spec=mspec,
                                 seed=int(s), budget=b,
                                 n_startup=int(n_startup if n_startup is not None else fmp_n_init(mspec, fmp_base)),
                                 keep_history=keep_history))
    return jobs


def fmp_n_init(mspec: dict, fmp_base: dict) -> int:
    from hpo_rl.baselines.HMM_MCMC_FMP import HMM_MCMC_FMP
    import inspect

    default = inspect.signature(HMM_MCMC_FMP.__init__).parameters["n_init"].default
    if mspec.get("kind") == "fmp":
        return int(mspec["params"].get("n_init", default))
    return int(fmp_base.get("n_init", default))


def result_path(out_dir: Path, job: dict) -> Path:
    return Path(out_dir) / job["task_key"] / job["method"] / f"seed_{job['seed']:03d}.json"


def run_job(job: dict, out_dir: Path | None = None, save: bool = True) -> dict:
    """Execute one job; returns a short summary dict. Errors are captured, not raised."""
    t0 = time.perf_counter()
    run_seed = derive_seed(job["seed"], job["task_key"])
    seed_everything(run_seed)
    try:
        task = build_task(job["task_spec"], seed=run_seed)
        rec = M.run_method(job["method"], job["method_spec"], task, job["seed"], job["budget"],
                           n_startup=job["n_startup"], keep_history=job.get("keep_history", True))
        rec["exp"] = job["exp"]
        rec["run_seed_derived"] = run_seed
        rec["wall_time"] = time.perf_counter() - t0
        if save and out_dir is not None:
            _atomic_json_dump(rec, result_path(out_dir, job))
        return {"ok": True, "task": job["task_key"], "method": job["method"], "seed": job["seed"],
                "best_value": rec["best_value"], "best_true_value": rec["best_true_value"], "time": rec["wall_time"]}
    except Exception as e:  # noqa: BLE001
        err = {"ok": False, "task": job["task_key"], "method": job["method"], "seed": job["seed"],
               "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        if save and out_dir is not None:
            p = result_path(out_dir, job).with_suffix(".error.txt")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(err["traceback"], encoding="utf-8")
        return err


def _atomic_json_dump(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def pending_jobs(jobs: list[dict], out_dir: Path) -> list[dict]:
    return [j for j in jobs if not result_path(out_dir, j).is_file()]


def run_jobs(jobs: list[dict], out_dir: Path, workers: int = 1, resume: bool = True, verbose: bool = True) -> list[dict]:
    out_dir = Path(out_dir)
    todo = pending_jobs(jobs, out_dir) if resume else jobs
    if verbose:
        print(f"[runner] {len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run, workers={workers}")
    if not todo:
        return []
    t0 = time.perf_counter()
    results: list[dict] = []
    if workers <= 1:
        for i, j in enumerate(todo):
            r = run_job(j, out_dir)
            results.append(r)
            if verbose:
                _print_progress(i + 1, len(todo), r, t0)
    else:
        from joblib import Parallel, delayed

        for i, r in enumerate(Parallel(n_jobs=workers, backend="loky", return_as="generator")(
                delayed(run_job)(j, out_dir) for j in todo)):
            results.append(r)
            if verbose:
                _print_progress(i + 1, len(todo), r, t0)
    n_err = sum(1 for r in results if not r.get("ok"))
    if verbose:
        print(f"[runner] finished {len(results)} jobs in {time.perf_counter() - t0:.0f}s, errors={n_err}")
        for r in results:
            if not r.get("ok"):
                print(f"  ERROR {r['task']}/{r['method']}/seed{r['seed']}: {r['error']}")
    return results


def _print_progress(i, n, r, t0):
    el = time.perf_counter() - t0
    eta = el / i * (n - i)
    if r.get("ok"):
        print(f"[{i}/{n}] {r['task']:<28s} {r['method']:<24s} seed={r['seed']:<3d} best={r['best_true_value']:.4g} "
              f"({r['time']:.1f}s)  elapsed={el:.0f}s eta={eta:.0f}s", flush=True)
    else:
        print(f"[{i}/{n}] {r['task']}/{r['method']}/seed{r['seed']} FAILED: {r['error']}", flush=True)


# ---------------------------------------------------------------------------
# dry run: no workload, checks configs and randomisation
# ---------------------------------------------------------------------------

def dry_run(jobs: list[dict], fmp_base: dict, n_check_evals: int = 5) -> bool:
    """Validate every distinct method spec on a 3-D quadratic with a mixed space:
    * two runs with the same seed give identical traces
    * a different seed gives a different trace
    * FMP kwargs are accepted by the class (typos -> TypeError here, not after hours)
    * baseline samplers construct and suggest correctly
    Prints the job grid summary. Returns True if all checks pass."""
    from experiments_new.common.spaces import Task

    space = {"x0": {"type": "float", "values": [-3.0, 3.0]}, "x1": {"type": "float", "values": [1e-3, 1.0], "log": True},
             "k": {"type": "int", "values": [1, 9]}, "c": {"type": "categorical", "values": ["a", "b", "c"]}}

    def quad(cfg):
        return (cfg["x0"] - 1.0) ** 2 + (np.log10(cfg["x1"]) + 1.0) ** 2 + 0.1 * (cfg["k"] - 4) ** 2 + {"a": 0.0, "b": 0.3, "c": 0.6}[cfg["c"]]

    task = Task(key="dry__quad", spec={"kind": "synthetic", "function": "quad", "dims": 2, "noise_std": 0.0, "categorical": True},
                space=space, objective=quad, f_star=0.0)
    specs = {}
    for j in jobs:
        specs.setdefault(j["method"], (j["method_spec"], j["n_startup"]))
    ok = True
    print(f"[dry-run] {len(jobs)} jobs: {len({j['task_key'] for j in jobs})} tasks x {len(specs)} methods x "
          f"{len({j['seed'] for j in jobs})} seeds, budget={jobs[0]['budget'] if jobs else '-'}")
    print(f"[dry-run] FMP shared config: {fmp_base if fmp_base else 'Table-5 class defaults'}")
    for name, (spec, n_startup) in specs.items():
        if spec["kind"] == "smac":
            print(f"  {name:<24s} SMAC runner present; skipped in dry-run (not installed on this platform)")
            continue
        budget = n_startup + n_check_evals
        t = task
        if spec.get("pruner") == "hyperband":
            t = build_task({"kind": "lcbench", "instance": "3945"})
        try:
            r1 = M.run_method(name, spec, t, 0, budget, n_startup=n_startup, keep_history=False)
            r2 = M.run_method(name, spec, t, 0, budget, n_startup=n_startup, keep_history=False)
            r3 = M.run_method(name, spec, t, 1, budget, n_startup=n_startup, keep_history=False)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:<24s} FAILED: {type(e).__name__}: {e}")
            ok = False
            continue
        same = r1["values"] == r2["values"] and r1["configs"] == r2["configs"]
        diff = r1["values"] != r3["values"]
        if spec.get("pruner"):   # multi-fidelity: budget is counted in cost units, not evaluations
            n_ok = r1["costs"][-1] >= budget * r1["eval_cost"] - 1e-9
        else:
            n_ok = r1["n_evals"] == budget
        status = "OK " if (same and diff and n_ok) else "BAD"
        if status == "BAD":
            ok = False
        extra = ""
        if spec["kind"] == "fmp":
            s = r1["fmp"]["summary"]
            extra = f" states={s['frac_exploit']:.2f}/{s['frac_explore']:.2f}/{s['frac_trapped']:.2f} dream={s['kernel_fraction_dream']:.2f}"
        print(f"  {name:<24s} {status} deterministic={same} seed_sensitive={diff} n_evals={r1['n_evals']}/{budget}"
              f"{' (multi-fidelity, cost=%.0f/%.0f)' % (r1['costs'][-1], budget * r1['eval_cost']) if spec.get('pruner') else ''} "
              f"best={r1['best_value']:.3g}{extra}")
    print("[dry-run] " + ("all checks passed" if ok else "SOME CHECKS FAILED"))
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli(exp: str, task_specs: list[dict], methods: list[str], seeds: list[int], budget: int,
        out_dir: Path | None = None, overrides: dict | None = None, keep_history: bool = True,
        smoke_tasks: int = 2, description: str = "", custom_specs: dict | None = None,
        smoke_methods: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description or exp)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None, help="substring filters on task keys")
    parser.add_argument("--fmp-config", default="table5")
    parser.add_argument("--baseline-config", default="default")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-history", action="store_true", help="do not store per-eval FMP diagnostics")
    args = parser.parse_args()

    fmp_base = load_fmp_base(args.fmp_config)
    baseline_params = load_baseline_params(args.baseline_config)
    if args.seeds is not None:
        seeds = args.seeds
    elif args.n_seeds is not None:
        seeds = list(range(args.n_seeds))
    if args.budget is not None:
        budget = args.budget
    if args.methods:
        methods = args.methods
    if args.tasks:
        task_specs = [s for s in task_specs if any(f in spec_key(s) for f in args.tasks)]
    out = Path(args.out) if args.out else (out_dir or REPO_ROOT / "experiments_new" / exp / "results")
    if args.smoke:
        n_init = int(fmp_base.get("n_init", fmp_n_init({"kind": "x"}, fmp_base)))
        budget = n_init + 5
        seeds = seeds[:1] if args.seeds is None else seeds   # --seeds overrides the 1-seed default
        task_specs = [{k: v for k, v in t.items() if k != "budget"} for t in task_specs[:smoke_tasks]]
        if smoke_methods:
            methods = smoke_methods
        out = out.parent / (out.name + "_smoke")
        print(f"[smoke] budget={budget} (n_init={n_init}+5), seeds={seeds}, tasks={[spec_key(s) for s in task_specs]}")
    jobs = make_jobs(exp, task_specs, methods, seeds, budget, fmp_base, baseline_params, overrides=overrides,
                     keep_history=keep_history and not args.no_history, custom_specs=custom_specs)
    args.jobs = jobs
    args.out_dir = out
    args.fmp_base = fmp_base
    if args.dry_run:
        n_check = 5
        ok = dry_run(jobs, fmp_base, n_check_evals=n_check)
        args.dry_ok = ok
        return args
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "_grid.json", "w", encoding="utf-8") as fh:
        json.dump({"exp": exp, "budget": budget, "seeds": seeds, "methods": methods,
                   "tasks": [spec_key(s) for s in task_specs], "fmp_base": fmp_base,
                   "baseline_params": baseline_params, "overrides": overrides}, fh, indent=2)
    args.results = run_jobs(jobs, out, workers=args.workers, resume=not args.no_resume)
    return args
