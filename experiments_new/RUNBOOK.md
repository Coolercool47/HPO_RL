# Runbook: rebuttal experiments in the right order

> Current status and the exact commands for the *next* launch are in `NEXT_RUNS.md`;
> this file is the general reference.

All commands run from the code repo root (`HPO_RL/HPO_RL`) with the project venv
(`.venv\Scripts\python.exe` on Windows, `.venv/bin/python` elsewhere). Every runner is
resumable: re-running the same command skips finished runs and fills gaps.

**Memory.** Under WSL2 the VM sees only the RAM allowed by `.wslconfig` (default half
of the host); the runner prints what it can see on its first line. GP-BO (Optuna `GPSampler`, torch) is the only memory-heavy method: a
500-trial run grows to 0.5 GB on Windows and considerably more on Linux, where glibc
does not return torch's freed memory to the OS. The runner therefore treats `GP` and
`SMAC` as *heavy* jobs: each runs in a fresh worker process, and their concurrency is
capped at `available RAM / --heavy-gb` (default 2 GB per job) independently of
`--workers`. Light jobs (everything else, < 0.3 GB per worker) run with the full
`--workers` in chunks of `--chunk` jobs, after which the pool is recycled. Every worker
is limited to one compute thread and one single-threaded ONNX session; on Linux the
workers run with `MALLOC_ARENA_MAX=2` and call `malloc_trim` after every job. A worker
whose RSS exceeds `--max-worker-gb` (default 3) after a job is replaced. The progress
line prints the worker RSS after each job and the final line prints the peak.
Every script writes a full log to `experiments_new/<exp>/logs/` and every runner
invocation appends its outcome (errors included, memory peaks) to `results/_run_summary.json`.

**Memory diagnostics (for out-of-memory kills).** Every run prints an `[env]` block at
start (platform, WSL/container detection, cgroup limit, RAM and swap, torch build, the
eight largest processes on the machine), samples memory every 10 s into
`logs/<ts>_memory_pid<N>.csv` (system, swap, cgroup, parent, workers, five largest other
processes such as an editor), prints a `[mem] periodic` line every 2 min and a `[mem]
WARNING` line when available memory drops below 1.5 GB, snapshots before every GP batch
and waits while memory is critically low, prints per-job worker RSS and peak RSS, and at
the end (or on abort) prints the peaks and any kernel OOM-kill messages from `dmesg`.
If a run dies, the `_memory.csv` shows what held the RAM at that moment.

## 0. Environment (new machine)

```
powershell -ExecutionPolicy Bypass -File experiments_new\setup_env.ps1     # Windows
bash experiments_new/setup_env.sh                                           # Linux / WSL / macOS
```

The script creates `.venv`, installs `experiments_new/requirements.txt`, installs
`yahpo-gym` without its outdated dependency pins, clones the LCBench surrogate data
(sparse checkout, ~150 MB) into `yahpo_data/`, and runs a dry-run. To move results
between machines copy the `experiments_new/*/results*/` directories and
`experiments_new/configs/`; nothing else is stateful.

## 1. Instance selection (once, ~1 min)

```
python experiments_new/lcbench/select_instances.py --n-test 10 --n-tune 8
```

Writes `configs/lcbench_instances.json`: 10 test instances (paper's 5 + 5 hardest by a
seeded random survey) and 8 disjoint tuning instances. Already done on this machine;
re-running reproduces the same file (seed 12345).

## 2. Sanity checks (5 min)

```
python experiments_new/synt_functions/run.py --dry-run
python experiments_new/lcbench/run.py --dry-run
python experiments_new/sensitivity/run.py --dry-run
python experiments_new/synt_functions/run.py --smoke
python experiments_new/lcbench/run.py --smoke
```

Dry-run validates every method (determinism per seed, seed sensitivity, kwargs) with
no workload. Smoke runs n_init + 5 evaluations on two tasks into `results_smoke/`.

## 3. E6 sensitivity at the Table-5 defaults (~25 min on 10 workers)

```
python experiments_new/sensitivity/run.py --workers 10
python experiments_new/sensitivity/analyze.py
```

Figures in `sensitivity/figures/`: `oat_synthetic.png`, `oat_lcbench.png` (one panel
per knob, dashed line = base configuration), `oat_influence.csv` (knob ranking),
`fanova_importance.csv`. Read this before tuning: knobs with flat curves can be
frozen at their defaults; the paper's sensitivity paragraph comes from here.

## 4. E0 held-out tuning (~1.5 h total)

```
python experiments_new/tuning_heldout/run.py --method FMP   --n-trials 100 --workers 10
python experiments_new/tuning_heldout/run.py --method TPE   --n-trials 100 --workers 10
python experiments_new/tuning_heldout/run.py --method CMAES --n-trials 100 --workers 10
```

Each meta-trial runs the candidate on the 8 tuning instances x 3 seeds (200 evals).
Outputs `configs/fmp_tuned.json` and `configs/baselines_tuned.json`. The study is
stored in `tuning_heldout/results/<method>/meta_study.db`, so an interrupted run
continues from where it stopped. The script refuses to run if tuning and test
instances overlap.

## 5. E1 synthetic and E2 LCBench, main comparison (~3 h with GP-BO)

Method definitions: `FMP` = soft filter + orchestrator + factorized proposals (p_dream = 0);
`FMP_DREAM` = the same + DREAM kernel. Both take every other knob, including n_chains and
p_dream, from `--fmp-config` (Table-5 class defaults or `configs/fmp_tuned.json`), so the
only difference between them is the DREAM kernel. The submitted paper's one-chain Viterbi
arm lives in the ladder as `L1_K1_VITERBI_SUB`.

Run the untuned reference first (cheap, separate output directory), then the tuned
configuration used in the paper tables:

```
python experiments_new/synt_functions/run.py --workers 10 --out experiments_new/synt_functions/results_table5
python experiments_new/lcbench/run.py        --workers 10 --out experiments_new/lcbench/results_table5

python experiments_new/synt_functions/run.py --workers 10 --fmp-config tuned --baseline-config tuned
python experiments_new/lcbench/run.py        --workers 10 --fmp-config tuned --baseline-config tuned

python experiments_new/synt_functions/analyze.py            # -> synt_functions/figures/
python experiments_new/lcbench/analyze.py                   # -> lcbench/figures/
python experiments_new/synt_functions/analyze.py --results experiments_new/synt_functions/results_table5
python experiments_new/lcbench/analyze.py        --results experiments_new/lcbench/results_table5
```

GP-BO dominates the run time (about 150 s per 500-evaluation run). To get everything
else first: `--methods RS TPE CMAES FMP FMP_DREAM` now and `--methods GP` later; the
analysis merges whatever is present. Outputs per suite: `table_<suite>.tex` (mean ±
SEM, bold best, * = Holm-corrected Mann-Whitney p < 0.05 vs TPE), `tests_vs_TPE.csv`,
`wins_vs_TPE.csv`, `cross_task.json` (Wilcoxon over tasks, Friedman, average ranks,
Nemenyi CD), `cd_<suite>.png`, `convergence_<suite>.png`, `overhead.csv`.

## 6. E3 ablation ladder and E4 controller ablation (~30 min)

```
python experiments_new/ablation_ladder/run.py     --workers 10 --fmp-config tuned
python experiments_new/ablation_controller/run.py --workers 10 --fmp-config tuned
python experiments_new/ablation_ladder/analyze.py
python experiments_new/ablation_controller/analyze.py
```

Ladder reference = `L1_K1_VITERBI_SUB` (step i); controller reference = `CTRL_HMM`. Decision
gates from the plan: if `L5_DREAM` is not significantly better than the best earlier
step, the paper is reframed around H-MCMC-FMP; if `CTRL_RULE` or `CTRL_RANDOM`
match `CTRL_HMM`, the HMM contribution is not supported.

## 7. E5 diagnostics (1 min, no new runs)

```
python experiments_new/diagnostics/run.py --results experiments_new/synt_functions/results --method FMP_DREAM --seeds 0
python experiments_new/diagnostics/run.py --results experiments_new/lcbench/results        --method FMP_DREAM --seeds 0
python experiments_new/diagnostics/run.py --results experiments_new/ablation_controller/results --method CTRL_HMM --seeds 0
```

Produces state-trajectory plots, O_t histograms with the hand-set emission densities
and an offline 3-component GMM, Baum-Welch transition-matrix evolution, and
`occupancy_<method>.csv` (state occupancy, acceptance per state, rescues).

## 8. Adding SMAC later (Linux, WSL or Docker)

SMAC3 has no Windows build. On Windows use Docker Desktop (it runs on the WSL2 backend, so
no separate WSL setup is needed). From the repository root:

```
docker build -f experiments_new/docker/Dockerfile.smac -t hpo-smac experiments_new
docker run --rm -v "${PWD}:/work" -w /work hpo-smac python experiments_new/lcbench/run.py        --methods SMAC --workers 4 --fmp-config tuned --baseline-config tuned
docker run --rm -v "${PWD}:/work" -w /work hpo-smac python experiments_new/synt_functions/run.py --methods SMAC --workers 4 --fmp-config tuned --baseline-config tuned
```

(Git Bash: prefix with `MSYS_NO_PATHCONV=1` and give the Windows path, e.g.
`-v "C:/path/to/HPO_RL:/work"`.) Results are written into the mounted working tree, the run
is resumable, and SMAC is scheduled as a heavy job like GP. Measured in the container:
about 0.3 s per evaluation on synthetic tasks and 2 s on LCBench at small budgets, 0.5 GB
peak per worker; expect roughly 10-15 h for both suites with 4 workers. The container sees
only the memory Docker Desktop is allowed (Settings > Resources, or `.wslconfig`).

On native Linux / WSL instead: `pip install "smac>=2.1"` and run the same two commands
without Docker.

Only the SMAC runs are executed; re-running the analysis scripts merges them into the
tables and figures.

## Reading the critical-difference diagram

The axis is the average rank over tasks (1 = best). Each method hangs from its rank.
A thick bar joins methods whose ranks differ by less than the critical difference
(Nemenyi test at alpha = 0.05), i.e. methods that are **not** significantly different.
With only 2 to 5 tasks the CD is larger than the whole axis and every method is joined;
the diagram is informative from about 10 tasks upward (19 synthetic configurations,
10 LCBench instances).
