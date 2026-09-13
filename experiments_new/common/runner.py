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
from experiments_new.common.logging_util import start_log  # noqa: E402
from experiments_new.common import memlog  # noqa: E402
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
        cfg = json.load(fh)
    # keys starting with "_" (e.g. "_meta" written by the tuning script) are documentation, not kwargs
    return {k: v for k, v in cfg.items() if not str(k).startswith("_")}


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
        cfg = json.load(fh)
    return {k: v for k, v in cfg.items() if not str(k).startswith("_")}


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


HEAVY_METHODS = {"GP", "SMAC"}          # torch / RF surrogates: ~0.5-2 GB per run, fresh process per job
HEAVY_GB_PER_WORKER = 2.0               # memory budget assumed per concurrent heavy job (measured peak ~0.75 GB; margin for WSL/laptops)


def _rss_gb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**30
    except Exception:  # noqa: BLE001
        return float("nan")


def _limit_worker_threads() -> None:
    """Called inside every worker before a job: one compute thread per process."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    if "torch" in sys.modules:
        try:
            sys.modules["torch"].set_num_threads(1)
        except Exception:  # noqa: BLE001
            pass


def _release_memory() -> None:
    """After a job: drop Python garbage and, on glibc, hand freed heap back to the OS."""
    import gc

    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:  # noqa: BLE001
            pass


def run_job(job: dict, out_dir: Path | None = None, save: bool = True) -> dict:
    """Execute one job; returns a short summary dict. Errors are captured, not raised."""
    _limit_worker_threads()
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
        wm = memlog.worker_memory()
        rec["worker_rss_gb"] = wm["rss_gb"]
        rec["worker_peak_rss_gb"] = wm["peak_rss_gb"]
        if save and out_dir is not None:
            _atomic_json_dump(rec, result_path(out_dir, job))
        out = {"ok": True, "task": job["task_key"], "method": job["method"], "seed": job["seed"],
               "best_value": rec["best_value"], "best_true_value": rec["best_true_value"], "time": rec["wall_time"],
               "rss_gb": rec["worker_rss_gb"], "peak_rss_gb": rec["worker_peak_rss_gb"]}
    except Exception as e:  # noqa: BLE001
        out = {"ok": False, "task": job["task_key"], "method": job["method"], "seed": job["seed"],
               "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(), "rss_gb": _rss_gb()}
        if save and out_dir is not None:
            p = result_path(out_dir, job).with_suffix(".error.txt")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(out["traceback"], encoding="utf-8")
    finally:
        _release_memory()
    max_gb = float(os.environ.get("EXP_MAX_WORKER_GB", "0") or 0)
    if max_gb > 0 and out.get("rss_gb", 0) > max_gb and os.environ.get("LOKY_PID") is not None:
        # this worker has grown too large: leave; loky replaces it with a fresh process
        out["worker_recycled"] = True
        sys.stdout.flush()
        os._exit(0)
    return out


def _atomic_json_dump(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def pending_jobs(jobs: list[dict], out_dir: Path) -> list[dict]:
    return [j for j in jobs if not result_path(out_dir, j).is_file()]


def default_workers(mem_per_worker_gb: float = 0.5) -> int:
    """CPU count - 2, capped by available RAM (a worker that has run GP-BO holds ~0.4 GB)."""
    n = max(1, (os.cpu_count() or 2) - 2)
    try:
        import psutil

        avail_gb = psutil.virtual_memory().available / 2**30
        n = max(1, min(n, int(avail_gb / mem_per_worker_gb)))
    except Exception:  # noqa: BLE001  psutil optional
        n = min(n, 8)
    return n


def heavy_workers(workers: int, gb_per_job: float = HEAVY_GB_PER_WORKER) -> int:
    """Concurrent GP/SMAC jobs allowed by available RAM (never more than `workers`)."""
    try:
        import psutil

        avail_gb = psutil.virtual_memory().available / 2**30
        return max(1, min(workers, int(avail_gb / gb_per_job)))
    except Exception:  # noqa: BLE001
        return max(1, min(workers, 4))


def _parallel_run(chunk: list[dict], out_dir: Path, workers: int, results: list, counter: list, n_total: int,
                  t0: float, verbose: bool) -> None:
    from joblib import Parallel, delayed, parallel_config
    from joblib.externals.loky import get_reusable_executor

    if sys.platform.startswith("linux"):
        # glibc: fewer arenas and eager trimming, so torch's many small frees are returned to the OS
        os.environ.setdefault("MALLOC_ARENA_MAX", "2")
        os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "131072")
        os.environ.setdefault("MALLOC_MMAP_THRESHOLD_", "131072")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        for r in Parallel(n_jobs=workers, return_as="generator")(delayed(run_job)(j, out_dir) for j in chunk):
            results.append(r)
            counter[0] += 1
            if verbose:
                _print_progress(counter[0], n_total, r, t0)
    get_reusable_executor().shutdown(wait=True)   # recycle worker processes


def run_jobs(jobs: list[dict], out_dir: Path, workers: int = 1, resume: bool = True, verbose: bool = True,
             chunk_size: int = 200, heavy_gb: float = HEAVY_GB_PER_WORKER, max_worker_gb: float = 0.0) -> list[dict]:
    """Execute pending jobs.

    Light jobs (everything but GP / SMAC) run `chunk_size` at a time in a loky pool of
    `workers` processes limited to one compute thread each; the pool is recycled after
    every chunk. Heavy jobs run one job per worker lifetime (the pool is recycled every
    `n_heavy` jobs) with `n_heavy` = min(workers, available RAM / heavy_gb), because a
    500-trial GP-BO run holds 0.5-2 GB that glibc does not always return."""
    out_dir = Path(out_dir)
    todo = pending_jobs(jobs, out_dir) if resume else jobs
    light = [j for j in todo if j["method"] not in HEAVY_METHODS]
    heavy = [j for j in todo if j["method"] in HEAVY_METHODS]
    n_heavy = heavy_workers(workers, heavy_gb) if heavy else 0
    if max_worker_gb > 0:
        os.environ["EXP_MAX_WORKER_GB"] = str(max_worker_gb)
    if verbose:
        try:
            import psutil

            vm = psutil.virtual_memory()
            print(f"[runner] memory visible to this process: total={vm.total / 2**30:.1f} GB, available={vm.available / 2**30:.1f} GB")
        except Exception:  # noqa: BLE001
            pass
        print(f"[runner] {len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run "
              f"({len(light)} light + {len(heavy)} heavy), workers={workers}, chunk={chunk_size}, "
              f"heavy concurrency={n_heavy} ({heavy_gb:.1f} GB/job budget), max_worker_gb={max_worker_gb or 'off'}")
    if not todo:
        return []
    t0 = time.perf_counter()
    results: list[dict] = []
    counter = [0]
    if workers <= 1:
        for j in todo:
            r = run_job(j, out_dir)
            results.append(r)
            counter[0] += 1
            if verbose:
                _print_progress(counter[0], len(todo), r, t0)
    else:
        for start in range(0, len(light), max(1, chunk_size)):
            _parallel_run(light[start:start + chunk_size], out_dir, workers, results, counter, len(todo), t0, verbose)
        i = 0
        batch = 0
        while i < len(heavy):
            # re-evaluate the budget before every heavy batch: other programs (editor, browser)
            # may have taken memory since the run started; wait while memory is critically low
            batch += 1
            d = memlog.snapshot(f"before heavy batch {batch}")
            waited = 0
            while d.get("available_gb", 1e9) < heavy_gb * 0.75 and waited < 600:
                print(f"[mem] available {d['available_gb']:.2f} GB < {0.75 * heavy_gb:.2f} GB: waiting 30s for memory to free up", flush=True)
                time.sleep(30)
                waited += 30
                d = memlog.snapshot("waiting")
            n_heavy = heavy_workers(workers, heavy_gb)
            print(f"[runner] heavy batch {batch}: {n_heavy} concurrent {sorted({j['method'] for j in heavy[i:i + n_heavy]})} job(s)", flush=True)
            _parallel_run(heavy[i:i + n_heavy], out_dir, n_heavy, results, counter, len(todo), t0, verbose)
            i += n_heavy
    n_err = sum(1 for r in results if not r.get("ok"))
    if verbose:
        rss = [r.get("rss_gb") for r in results if r.get("rss_gb") == r.get("rss_gb")]
        print(f"[runner] finished {len(results)} jobs in {time.perf_counter() - t0:.0f}s, errors={n_err}, "
              f"peak worker RSS={max(rss) if rss else float('nan'):.2f} GB")
        for r in results:
            if not r.get("ok"):
                print(f"  ERROR {r['task']}/{r['method']}/seed{r['seed']}: {r['error']}")
    return results


def _print_progress(i, n, r, t0):
    el = time.perf_counter() - t0
    eta = el / i * (n - i)
    if r.get("ok"):
        pk = r.get("peak_rss_gb", float("nan"))
        print(f"[{i}/{n}] {r['task']:<28s} {r['method']:<24s} seed={r['seed']:<3d} best={r['best_true_value']:.4g} "
              f"({r['time']:.1f}s, rss={r.get('rss_gb', float('nan')):.2f}GB" + (f" peak={pk:.2f}GB" if pk == pk else "") +
              f")  elapsed={el:.0f}s eta={eta:.0f}s", flush=True)
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
    parser.add_argument("--workers", type=int, default=None, help="default: CPU-2 capped by available RAM (0.5 GB/worker)")
    parser.add_argument("--chunk", type=int, default=200, help="light jobs per worker-pool lifetime (memory recycling)")
    parser.add_argument("--heavy-gb", type=float, default=HEAVY_GB_PER_WORKER,
                        help="RAM budget per concurrent GP/SMAC job; caps their concurrency (default 2.0)")
    parser.add_argument("--mem-interval", type=float, default=10.0, help="seconds between memory samples (logs/*_memory.csv)")
    parser.add_argument("--mem-report", type=float, default=120.0, help="seconds between memory summary lines in the log")
    parser.add_argument("--mem-warn-gb", type=float, default=1.5, help="warn when available memory drops below this")
    parser.add_argument("--max-worker-gb", type=float, default=3.0,
                        help="a worker exceeding this RSS after a job is replaced by a fresh process (0 = off)")
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
    if args.workers is None:
        args.workers = default_workers()
    start_log(REPO_ROOT / "experiments_new" / exp, "run" + ("_dry" if args.dry_run else "_smoke" if args.smoke else ""))

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
    memlog.environment_report()
    monitor = memlog.start_memory_monitor(REPO_ROOT / "experiments_new" / exp, interval=args.mem_interval,
                                          report_every=args.mem_report, warn_gb=args.mem_warn_gb)
    memlog.snapshot("start")
    with open(out / "_grid.json", "w", encoding="utf-8") as fh:
        json.dump({"exp": exp, "budget": budget, "seeds": seeds, "methods": methods,
                   "tasks": [spec_key(s) for s in task_specs], "fmp_base": fmp_base,
                   "baseline_params": baseline_params, "overrides": overrides}, fh, indent=2)
    try:
        args.results = run_jobs(jobs, out, workers=args.workers, resume=not args.no_resume, chunk_size=args.chunk,
                                heavy_gb=args.heavy_gb, max_worker_gb=args.max_worker_gb)
    except BaseException as e:  # noqa: BLE001  log the crash context, then re-raise
        print(f"[runner] ABORTED: {type(e).__name__}: {e}", flush=True)
        memlog.snapshot("at abort")
        memlog.kernel_oom_report()
        monitor.stop()
        raise
    memlog.snapshot("end")
    monitor.stop()
    memlog.kernel_oom_report()
    # machine-readable outcome of this invocation (appended, one entry per invocation)
    summ_path = out / "_run_summary.json"
    prev = json.loads(summ_path.read_text(encoding="utf-8")) if summ_path.is_file() else []
    prev.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv, "n_jobs": len(jobs),
                 "memory_peaks": monitor.peak, "workers": args.workers, "heavy_gb": args.heavy_gb,
                 "n_run": len(args.results), "n_errors": sum(1 for r in args.results if not r.get("ok")),
                 "errors": [r for r in args.results if not r.get("ok")][:50],
                 "pending_after": len(pending_jobs(jobs, out))})
    summ_path.write_text(json.dumps(prev, indent=1, default=str), encoding="utf-8")
    return args
