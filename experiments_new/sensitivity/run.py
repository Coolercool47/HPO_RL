"""E6: hyperparameter sensitivity of H-MCMC-FMP (multi-chain, soft; ladder step iv).

Part A - one-at-a-time: every knob in KNOBS is set to 5 levels (others at the shared
         config), 5 representative tasks, 10 seeds. Method names: S__<knob>__<level>.
Part B - random configurations for fANOVA importances: N_RANDOM joint samples of all
         knobs on all 9 tasks, 5 seeds. Method names: R__<i>.

    python experiments_new/sensitivity/run.py --dry-run | --smoke | --workers 10
    python experiments_new/sensitivity/run.py --part A            # only the OAT sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments_new.common import methods as M  # noqa: E402
from experiments_new.common.runner import cli, load_fmp_base  # noqa: E402

HERE = Path(__file__).resolve().parent
BASE_VARIANT = "L4_SOFT"

# knob -> levels (absolute values). Table-5 default is the middle entry.
KNOBS: dict[str, list] = {
    "T_mcmc": [0.1, 0.3, 1.0, 3.0, 10.0],
    "sigma_fraction": [0.01, 0.03, 0.1, 0.3, 1.0],
    "wide_sigma_fraction": [0.04, 0.12, 0.4, 1.2, 4.0],
    "kde_tau": [0.005, 0.015, 0.05, 0.15, 0.5],
    "hmm_window": [2, 4, 8, 16, 32],
    "rejection_streak": [3, 5, 10, 20, 40],
    "bw_prior_strength": [2.5, 7.5, 25.0, 75.0, 250.0],
    "burnin_fraction": [0.0, 0.05, 0.1, 0.2, 0.4],
    "n_init": [4, 8, 16, 32, 64],
    "n_chains": [1, 2, 4, 8, 16],
    "p_dream": [0.0, 0.25, 0.5, 0.75, 1.0],
    "emission_scale": [0.1, 0.3, 1.0, 3.0, 10.0],     # multiplies emission mu and sigma jointly
}
# ranges for the random (fANOVA) part: (low, high, log?)
RANDOM_SPACE = {
    "T_mcmc": (0.05, 20.0, True), "sigma_fraction": (0.005, 1.0, True), "wide_sigma_fraction": (0.04, 4.0, True),
    "kde_tau": (0.005, 0.5, True), "hmm_window": (2, 32, True), "rejection_streak": (3, 40, True),
    "bw_prior_strength": (1.0, 250.0, True), "burnin_fraction": (0.0, 0.4, False), "n_init": (4, 64, True),
    "n_chains": (1, 16, True), "p_dream": (0.0, 1.0, False), "emission_scale": (0.1, 10.0, True),
}
INT_KNOBS = {"hmm_window", "rejection_streak", "n_init", "n_chains"}
N_RANDOM = 200

TASKS_A = [
    dict(kind="synthetic", function="rastrigin", dims=10, noise_std=0.0, categorical=False),
    dict(kind="synthetic", function="schwefel", dims=10, noise_std=0.0, categorical=False),
    dict(kind="synthetic", function="ackley", dims=10, noise_std=0.0, categorical=True),
    # LCBench at the main protocol's budget (200 full-fidelity evaluations); two instances from
    # the original hard set and two from the added ones
    dict(kind="lcbench", instance="7593", budget=200),
    dict(kind="lcbench", instance="168329", budget=200),
    dict(kind="lcbench", instance="167185", budget=200),
    dict(kind="lcbench", instance="126026", budget=200),
    dict(kind="lcbench", instance="167181", budget=200),
    dict(kind="lcbench", instance="189908", budget=200),
]
# knobs swept multiplicatively (x0.25, x0.5, x1, x2, x4) when the sweep is centred on a tuned config
MULT_KNOBS = ["T_mcmc", "sigma_fraction", "wide_sigma_fraction", "kde_tau", "hmm_window", "rejection_streak",
              "bw_prior_strength", "n_init"]
KNOB_CAPS = {"sigma_fraction": 1.0, "wide_sigma_fraction": 4.0, "n_init": 64}


def centred_knobs(base: dict) -> dict[str, list]:
    """Levels around the values of `base` (tuned config); absolute grids for the bounded knobs."""
    out = dict(KNOBS)
    for k in MULT_KNOBS:
        if k not in base:
            continue
        lv = [float(base[k]) * m for m in (0.25, 0.5, 1.0, 2.0, 4.0)]
        lv = [min(v, KNOB_CAPS.get(k, float("inf"))) for v in lv]
        lv = [max(2, int(round(v))) for v in lv] if k in INT_KNOBS else [float(f"{v:.4g}") for v in lv]
        out[k] = sorted(set(lv))
    return out
TASKS_B = TASKS_A   # random configurations (fANOVA) on every task, not on one function


def apply_knob(params: dict, knob: str, value) -> dict:
    p = dict(params)
    if knob == "emission_scale":
        from hpo_rl.baselines.HMM_MCMC_FMP import DEFAULT_EMISSION_MU, DEFAULT_EMISSION_SIGMA

        mu = p.get("emission_mu", DEFAULT_EMISSION_MU)
        sd = p.get("emission_sigma", DEFAULT_EMISSION_SIGMA)
        p["emission_mu"] = [float(m) * value for m in mu]
        p["emission_sigma"] = [float(s) * value for s in sd]
    else:
        p[knob] = int(value) if knob in INT_KNOBS else value
    return p


def build_specs(fmp_base: dict, part: str) -> tuple[list[str], dict, dict]:
    if fmp_base:   # tuned config: sweep around the reported method (n_chains / p_dream from the config)
        base = M.method_spec("FMP_DREAM", fmp_base)["params"]
        knobs = centred_knobs(base)
    else:          # Table-5 defaults: multi-chain soft FMP, p_dream = 0 (ladder step iv)
        base = {**fmp_base, **M.FMP_VARIANTS[BASE_VARIANT]}
        knobs = KNOBS
    names, specs, meta = [], {}, {}
    if part in ("A", "AB"):
        names.append("S__base")
        specs["S__base"] = {"kind": "fmp", "params": dict(base)}
        meta["S__base"] = {"knob": "base", "level": None}
        for knob, levels in knobs.items():
            for lv in levels:
                n = f"S__{knob}__{lv}"
                names.append(n)
                specs[n] = {"kind": "fmp", "params": apply_knob(base, knob, lv)}
                meta[n] = {"knob": knob, "level": lv}
    if part in ("B", "AB"):
        rng = np.random.default_rng(2026)
        for i in range(N_RANDOM):
            p = dict(base)
            sample = {}
            for knob, (lo, hi, log) in RANDOM_SPACE.items():
                v = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if log else float(rng.uniform(lo, hi))
                if knob in INT_KNOBS:
                    v = int(round(v))
                sample[knob] = v
                p = apply_knob(p, knob, v)
            n = f"R__{i:03d}"
            names.append(n)
            specs[n] = {"kind": "fmp", "params": p}
            meta[n] = {"knob": "random", "level": None, "sample": sample}
    return names, specs, meta


if __name__ == "__main__":
    # peel the --part flag before the generic CLI
    part = "AB"
    if "--part" in sys.argv:
        i = sys.argv.index("--part")
        part = sys.argv[i + 1].upper()
        del sys.argv[i:i + 2]
    fmp_cfg = "table5"
    if "--fmp-config" in sys.argv:
        fmp_cfg = sys.argv[sys.argv.index("--fmp-config") + 1]
    fmp_base = load_fmp_base(fmp_cfg)
    names, specs, meta = build_specs(fmp_base, part)
    tasks = TASKS_A if part == "A" else (TASKS_B if part == "B" else TASKS_A)
    res_dir = HERE / ("results_smoke" if "--smoke" in sys.argv else "results")
    if "--out" in sys.argv:
        res_dir = Path(sys.argv[sys.argv.index("--out") + 1])
    if part == "AB":
        # A on TASKS_A, B on TASKS_B: run as two grids under the same result dir
        names_a = [n for n in names if n.startswith("S__")]
        names_b = [n for n in names if n.startswith("R__")]
        res_dir.mkdir(parents=True, exist_ok=True)
        with open(res_dir / "_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)
        argv = list(sys.argv)
        cli("sensitivity", TASKS_A, names_a, list(range(10)), 500, custom_specs=specs, keep_history=False,
            smoke_methods=["S__base", "S__T_mcmc__0.1", "S__emission_scale__10.0", "S__n_chains__1"], smoke_tasks=2)
        sys.argv = argv
        cli("sensitivity", TASKS_B, names_b, list(range(5)), 500, custom_specs=specs, keep_history=False,
            smoke_methods=["R__000", "R__001"], smoke_tasks=1)
    else:
        res_dir.mkdir(parents=True, exist_ok=True)
        with open(res_dir / "_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)
        cli("sensitivity", tasks, names, list(range(10 if part == "A" else 5)), 500, custom_specs=specs,
            keep_history=False, smoke_methods=names[:3], smoke_tasks=2)
