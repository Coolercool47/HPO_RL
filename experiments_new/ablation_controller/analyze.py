"""E4 analysis; reference = the HMM controller."""

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
    ap.add_argument("--ref", default="FMP_CTRL_HMM")
    a = ap.parse_args()
    analyze(Path(a.results), Path(a.figs or (Path(a.results).parent / (Path(a.results).name.replace("results", "figures")))),
            ref_method=a.ref, exp_name="controller", curve_ncols=4)
