"""E2 analysis: python experiments_new/lcbench/analyze.py [--results DIR] [--ref TPE]"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments_new.common.analysis import analyze  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--figs", default=None)
    ap.add_argument("--ref", default="TPE")
    ap.add_argument("--methods", nargs="*", default=None)
    a = ap.parse_args()
    analyze(Path(a.results), Path(a.figs or (Path(a.results).parent / (Path(a.results).name.replace("results", "figures")))),
            ref_method=a.ref, methods=a.methods, exp_name="lcbench", curve_ncols=5)
