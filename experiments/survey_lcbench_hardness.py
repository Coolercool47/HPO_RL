"""Survey all 34 yahpo lcbench instances for landscape hardness proxies."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA_PATH = ROOT / "yahpo_data"
os.environ["YAHPO_DATA_PATH"] = str(DATA_PATH)

import ConfigSpace as CS

if not hasattr(CS.ConfigurationSpace, "_sort_hyperparameters"):
    CS.ConfigurationSpace._sort_hyperparameters = lambda self: None

from yahpo_gym import benchmark_set, local_config

local_config.init_config() if not local_config.settings_path.exists() else None
local_config.set_data_path(str(DATA_PATH))

N_SAMPLES = 256
rows: list[dict] = []

instances = benchmark_set.BenchmarkSet("lcbench").instances
for inst in instances:
    bench = benchmark_set.BenchmarkSet("lcbench")
    bench.set_instance(inst)
    bench.check = False
    target = "val_accuracy"
    fidelity = "epoch"
    f_hp = bench.get_fidelity_space()[fidelity]
    max_epoch = int(round(float(f_hp.upper)))
    for fp in bench.config.fidelity_params:
        if fp != fidelity:
            fpi = bench.get_fidelity_space()[fp]
            val = int(round(float(fpi.upper)))
            bench.set_constant(fp, val)
    opt_space = bench.get_opt_space(drop_fidelity_params=True)
    accs = []
    for _ in range(N_SAMPLES):
        cfg = opt_space.sample_configuration().get_dictionary()
        cfg[fidelity] = max_epoch
        accs.append(float(bench.objective_function(cfg)[0][target]))
    accs = np.asarray(accs, dtype=float)
    rows.append(
        {
            "instance": inst,
            "rand_best": float(accs.max()),
            "rand_p90": float(np.percentile(accs, 90)),
            "rand_median": float(np.median(accs)),
            "rand_mean": float(accs.mean()),
            "rand_worst": float(accs.min()),
            "rand_spread": float(accs.max() - accs.min()),
            "rand_std": float(accs.std(ddof=0)),
        }
    )
    print(
        f"done {inst}: best={accs.max():.4f} median={np.median(accs):.4f} "
        f"spread={accs.max() - accs.min():.4f}",
        flush=True,
    )

df = pd.DataFrame(rows)
df["hardness"] = (
    (1.0 - (df["rand_best"] - df["rand_best"].min()) / (df["rand_best"].max() - df["rand_best"].min() + 1e-9))
    + 0.5
    * ((df["rand_spread"] - df["rand_spread"].min()) / (df["rand_spread"].max() - df["rand_spread"].min() + 1e-9))
    + 0.3 * ((df["rand_std"] - df["rand_std"].min()) / (df["rand_std"].max() - df["rand_std"].min() + 1e-9))
)
df = df.sort_values("hardness", ascending=False)

out = ROOT / "experiments" / "yahpo_results" / "lcbench_hardness_survey.csv"
df.to_csv(out, index=False)
print(f"\nSaved {out}")

print("\n=== HARDEST (low random-best + high spread) ===")
print(df.head(12).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n=== EASIEST (high random-best) ===")
print(
    df.sort_values("rand_best", ascending=False)
    .head(8)[["instance", "rand_best", "rand_median", "rand_spread", "rand_std"]]
    .to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

print("\n=== YOUR CURRENT INSTANCES ===")
cur = ["3945", "7593", "126026"]
print(
    df[df["instance"].isin(cur)]
    .sort_values("hardness", ascending=False)
    .to_string(index=False, float_format=lambda x: f"{x:.4f}")
)
