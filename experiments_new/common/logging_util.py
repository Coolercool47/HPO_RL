"""Tee stdout/stderr of a script into a timestamped log file (full execution log).

    from experiments_new.common.logging_util import start_log
    start_log(Path("experiments_new/lcbench"), "run")     # -> experiments_new/lcbench/logs/<ts>_run.log

Worker processes (joblib/loky) do not inherit the tee; their per-run outcomes are
reported by the parent (progress lines, `.error.txt` tracebacks, `_run_summary.json`).
"""

from __future__ import annotations

import atexit
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
            except Exception:  # noqa: BLE001  never let logging kill a run
                pass

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:  # noqa: BLE001
                pass

    def isatty(self):
        return False

    def fileno(self):
        return self._streams[0].fileno()

    @property
    def encoding(self):
        return getattr(self._streams[0], "encoding", "utf-8")


_ACTIVE: list[Path] = []


def start_log(exp_dir: Path, name: str, log_dir: Path | None = None) -> Path:
    """Start teeing. Idempotent per process (returns the existing log path if already active)."""
    if _ACTIVE:
        return _ACTIVE[0]
    log_dir = Path(log_dir) if log_dir else Path(exp_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"{ts}_{name}_pid{os.getpid()}.log"
    fh = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        rev = "?"
    print(f"[log] {path}")
    print(f"[log] started {ts}  cwd={os.getcwd()}  git={rev}  python={sys.version.split()[0]}")
    print(f"[log] argv: {' '.join(sys.argv)}")
    _ACTIVE.append(path)

    def _close():
        print(f"[log] finished {_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        fh.close()

    atexit.register(_close)
    return path
