"""Choose LCBench test / held-out-tuning instances by a documented, seeded hardness survey.

For every one of the 34 instances, evaluate N random configurations at the full
52 epochs and compute
    hardness = (1 - norm(best))   + 0.5 * norm(p90 - median) + 0.3 * norm(std)
(higher = harder: low best accuracy from random search, wide spread).
The paper's original 5 instances are always kept in the test set; the remaining test
instances are the hardest ones, and the tuning instances are the next hardest
(disjoint from the test set). Writes experiments_new/configs/lcbench_instances.json.

    python experiments_new/lcbench/select_instances.py --n-test 10 --n-tune 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments_new.common.yahpo import ALL_LCBENCH_INSTANCES, evaluate_at_epoch, lcbench_space  # noqa: E402
from hpo_rl.baselines.HMM_MCMC_FMP import SobolInitializer, _param_record, decode_config  # noqa: E402

PAPER_INSTANCES = ["7593", "189873", "168329", "167185", "167152"]
OUT = REPO_ROOT / "experiments_new" / "configs" / "lcbench_instances.json"


def survey(n_samples: int, seed: int) -> dict:
    rows = {}
    for inst in ALL_LCBENCH_INSTANCES:
        space, _, max_e = lcbench_space(inst)
        params = [_param_record(k, v) for k, v in space.items()]
        cfgs = SobolInitializer(params).generate(n_samples, seed=seed)
        accs = np.array([evaluate_at_epoch(inst, decode_config(c, params), int(max_e)) for c in cfgs])
        rows[inst] = dict(best=float(accs.max()), median=float(np.median(accs)), p90=float(np.percentile(accs, 90)),
                          std=float(accs.std()), spread=float(np.percentile(accs, 90) - np.median(accs)))
    keys = list(rows)

    def norm(v):
        v = np.asarray(v, dtype=float)
        return (v - v.min()) / (v.max() - v.min() + 1e-12)

    h = (1 - norm([rows[k]["best"] for k in keys])) + 0.5 * norm([rows[k]["spread"] for k in keys]) + 0.3 * norm([rows[k]["std"] for k in keys])
    for k, hv in zip(keys, h):
        rows[k]["hardness"] = float(hv)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--n-tune", type=int, default=8)
    args = ap.parse_args()
    rows = survey(args.n_samples, args.seed)
    ranked = sorted(rows, key=lambda k: -rows[k]["hardness"])
    test = list(PAPER_INSTANCES)
    for k in ranked:
        if len(test) >= args.n_test:
            break
        if k not in test:
            test.append(k)
    tune = [k for k in ranked if k not in test][: args.n_tune]
    out = dict(seed=args.seed, n_samples=args.n_samples, rule="paper 5 + hardest by survey; tuning = next hardest, disjoint",
               test=test, tune=tune, survey=rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("test instances:", test)
    print("tuning instances:", tune)
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
