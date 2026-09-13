"""Memory diagnostics for the experiment runner (to explain out-of-memory kills).

start_memory_monitor(exp_dir)  -> background thread that every `interval` seconds appends
    a row to <exp_dir>/logs/<ts>_memory.csv with: system total/available/used, swap,
    cgroup limit/usage (WSL2 / containers), this process' RSS, the runner's worker RSS
    (sum, max, count), and the five largest other processes on the machine (name, RSS).
    Every `report_every` seconds a one-line summary is also printed into the main log,
    and a WARNING line whenever available memory falls below `warn_gb`.
snapshot(tag)                  -> one such line, printed immediately (start/end, heavy batches).
environment_report()           -> platform, WSL/container detection, cgroup limit, swap, top
    processes; printed once at start.
kernel_oom_report()            -> on Linux, dmesg / journalctl lines about OOM kills (best effort).
worker_memory()                -> RSS and peak RSS of the calling worker (VmHWM on Linux).
"""

from __future__ import annotations

import csv
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path


def _psutil():
    try:
        import psutil

        return psutil
    except Exception:  # noqa: BLE001
        return None


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except Exception:  # noqa: BLE001
        return None


def cgroup_memory() -> dict:
    """cgroup v2 / v1 memory limit and usage (Linux; WSL2 and containers expose these)."""
    out: dict = {}
    for lim, use, key in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current", "v2"),
                          ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes", "v1")):
        a, b = _read(lim), _read(use)
        if a is not None:
            out["cgroup"] = key
            out["cgroup_limit_gb"] = None if a in ("max", "") or int(a) > 2**60 else int(a) / 2**30
            out["cgroup_usage_gb"] = int(b) / 2**30 if b and b.isdigit() else None
            break
    return out


def is_wsl() -> bool:
    v = _read("/proc/version") or ""
    return "microsoft" in v.lower() or "wsl" in v.lower()


def worker_memory() -> dict:
    ps = _psutil()
    out = {"rss_gb": float("nan"), "peak_rss_gb": float("nan")}
    if ps is not None:
        try:
            out["rss_gb"] = ps.Process().memory_info().rss / 2**30
        except Exception:  # noqa: BLE001
            pass
    st = _read("/proc/self/status")
    if st:
        for line in st.splitlines():
            if line.startswith("VmHWM"):
                out["peak_rss_gb"] = int(line.split()[1]) / 2**20
    return out


def _top_processes(ps, n: int = 5, exclude_pids: set | None = None) -> list[tuple[str, int, float]]:
    rows = []
    for p in ps.process_iter(["pid", "name", "memory_info"]):
        try:
            if exclude_pids and p.info["pid"] in exclude_pids:
                continue
            mi = p.info["memory_info"]
            if mi is None:
                continue
            rows.append((p.info["name"] or "?", p.info["pid"], mi.rss / 2**30))
        except Exception:  # noqa: BLE001
            continue
    rows.sort(key=lambda r: -r[2])
    return rows[:n]


def _family(ps):
    """(parent process, list of children) of the current process."""
    me = ps.Process()
    try:
        kids = me.children(recursive=True)
    except Exception:  # noqa: BLE001
        kids = []
    return me, kids


def snapshot_dict() -> dict:
    ps = _psutil()
    d: dict = {"time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if ps is None:
        return d
    try:
        vm = ps.virtual_memory()
        sw = ps.swap_memory()
        d.update(total_gb=vm.total / 2**30, available_gb=vm.available / 2**30, used_gb=vm.used / 2**30,
                 percent=vm.percent, swap_total_gb=sw.total / 2**30, swap_used_gb=sw.used / 2**30)
    except Exception:  # noqa: BLE001
        pass
    d.update(cgroup_memory())
    try:
        me, kids = _family(ps)
        d["parent_rss_gb"] = me.memory_info().rss / 2**30
        krss = []
        for k in kids:
            try:
                krss.append(k.memory_info().rss / 2**30)
            except Exception:  # noqa: BLE001
                pass
        d.update(n_workers=len(krss), workers_rss_sum_gb=sum(krss), workers_rss_max_gb=max(krss) if krss else 0.0)
        fam = {me.pid, *[k.pid for k in kids]}
        d["top_other"] = "; ".join(f"{n}[{pid}]={r:.2f}GB" for n, pid, r in _top_processes(ps, 5, fam))
    except Exception as e:  # noqa: BLE001
        d["error"] = repr(e)
    return d


def format_snapshot(d: dict, tag: str = "") -> str:
    cg = f" cgroup={d.get('cgroup_usage_gb', float('nan')):.2f}/{d['cgroup_limit_gb']:.2f}GB" if d.get("cgroup_limit_gb") else ""
    return (f"[mem]{(' ' + tag) if tag else ''} avail={d.get('available_gb', float('nan')):.2f}/{d.get('total_gb', float('nan')):.2f}GB "
            f"used={d.get('percent', float('nan')):.0f}% swap={d.get('swap_used_gb', float('nan')):.2f}/{d.get('swap_total_gb', float('nan')):.2f}GB{cg} | "
            f"runner: parent={d.get('parent_rss_gb', float('nan')):.2f}GB workers={d.get('n_workers', 0)} "
            f"sum={d.get('workers_rss_sum_gb', 0.0):.2f}GB max={d.get('workers_rss_max_gb', 0.0):.2f}GB | "
            f"top other: {d.get('top_other', '?')}")


def snapshot(tag: str = "") -> dict:
    d = snapshot_dict()
    print(format_snapshot(d, tag), flush=True)
    return d


def environment_report() -> None:
    ps = _psutil()
    print(f"[env] platform={platform.platform()} python={sys.version.split()[0]} cpus={os.cpu_count()} "
          f"wsl={is_wsl()} container={'yes' if Path('/.dockerenv').exists() else 'no'}")
    cg = cgroup_memory()
    if cg:
        print(f"[env] cgroup {cg.get('cgroup')} limit={cg.get('cgroup_limit_gb')} GB usage={cg.get('cgroup_usage_gb')} GB")
    if is_wsl():
        cfg = Path(os.path.expandvars("/mnt/c/Users")) if Path("/mnt/c/Users").exists() else None
        print("[env] WSL2 detected: the VM sees only the memory allowed by %UserProfile%\\.wslconfig "
              "(default: 50% of host RAM); 'free -h' inside WSL shows the effective limit."
              + (f" Host user dirs: {[p.name for p in cfg.iterdir()][:6]}" if cfg else ""))
    if ps is not None:
        try:
            vm, sw = ps.virtual_memory(), ps.swap_memory()
            print(f"[env] RAM total={vm.total / 2**30:.2f} GB available={vm.available / 2**30:.2f} GB; "
                  f"swap total={sw.total / 2**30:.2f} GB used={sw.used / 2**30:.2f} GB")
            print("[env] largest processes now: " + "; ".join(f"{n}[{pid}]={r:.2f}GB" for n, pid, r in _top_processes(ps, 8)))
        except Exception as e:  # noqa: BLE001
            print(f"[env] psutil error: {e!r}")
    try:
        import torch  # noqa: F401  (only to report the build; GP baseline imports it anyway)

        print(f"[env] torch {torch.__version__} cuda_build={torch.version.cuda} threads={torch.get_num_threads()}")
    except Exception:  # noqa: BLE001
        pass


def kernel_oom_report(max_lines: int = 20) -> None:
    """Best-effort: print recent kernel OOM-kill messages (Linux). Needs dmesg permission."""
    if not sys.platform.startswith("linux"):
        return
    for cmd in (["dmesg", "-T"], ["journalctl", "-k", "--no-pager", "-n", "2000"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
        except Exception:  # noqa: BLE001
            continue
        hits = [ln for ln in out.splitlines() if "killed process" in ln.lower() or "out of memory" in ln.lower()
                or "oom-kill" in ln.lower() or "oom_reaper" in ln.lower()]
        if hits:
            print(f"[oom] kernel messages via {cmd[0]} (last {min(len(hits), max_lines)}):")
            for ln in hits[-max_lines:]:
                print("   " + ln)
            return
        if out:
            print(f"[oom] no OOM-kill messages in {cmd[0]} output")
            return
    print("[oom] could not read kernel messages (dmesg/journalctl unavailable or not permitted)")


class MemoryMonitor(threading.Thread):
    def __init__(self, csv_path: Path, interval: float = 10.0, report_every: float = 120.0, warn_gb: float = 1.5):
        super().__init__(daemon=True, name="memory-monitor")
        self.csv_path = Path(csv_path)
        self.interval = interval
        self.report_every = report_every
        self.warn_gb = warn_gb
        self._stop_event = threading.Event()
        self.peak: dict = {}

    def run(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["time", "total_gb", "available_gb", "used_gb", "percent", "swap_total_gb", "swap_used_gb", "cgroup",
                  "cgroup_limit_gb", "cgroup_usage_gb", "parent_rss_gb", "n_workers", "workers_rss_sum_gb",
                  "workers_rss_max_gb", "top_other", "error"]
        new = not self.csv_path.exists()
        last_report = 0.0
        warned = False
        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            while not self._stop_event.is_set():
                d = snapshot_dict()
                w.writerow(d)
                fh.flush()
                for k in ("workers_rss_sum_gb", "workers_rss_max_gb", "used_gb", "swap_used_gb"):
                    if k in d:
                        self.peak[k] = max(self.peak.get(k, 0.0), d[k])
                if "available_gb" in d:
                    self.peak["min_available_gb"] = min(self.peak.get("min_available_gb", 1e9), d["available_gb"])
                now = time.time()
                avail = d.get("available_gb")
                if avail is not None and avail < self.warn_gb:
                    if not warned:
                        print(f"[mem] WARNING available memory {avail:.2f} GB < {self.warn_gb} GB", flush=True)
                        print(format_snapshot(d, "low-memory"), flush=True)
                        warned = True
                elif avail is not None and avail > 2 * self.warn_gb:
                    warned = False
                if now - last_report >= self.report_every:
                    print(format_snapshot(d, "periodic"), flush=True)
                    last_report = now
                self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=5)
        if self.peak:
            print("[mem] peaks during this run: " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(self.peak.items())), flush=True)


def start_memory_monitor(exp_dir: Path, interval: float = 10.0, report_every: float = 120.0, warn_gb: float = 1.5) -> MemoryMonitor:
    ts = time.strftime("%Y%m%d_%H%M%S")
    mon = MemoryMonitor(Path(exp_dir) / "logs" / f"{ts}_memory_pid{os.getpid()}.csv", interval, report_every, warn_gb)
    mon.start()
    print(f"[mem] sampling every {interval:.0f}s -> {mon.csv_path}", flush=True)
    return mon
