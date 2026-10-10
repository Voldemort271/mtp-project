"""OOM rehearsal: measure peak RSS while running the heaviest experiment job.

Runs black-4 @ uniform cap 600 through the same run_variant function used by
exp_sampling, sampling *system-wide* python+uv RSS at 1 Hz so we can see the
worst-case footprint of a single parallel slot.
"""

from __future__ import annotations

import sys
import threading
import time
import subprocess

sys.path.insert(0, "/home/shubi/work/code/mtp")

peak = {"sys_mb": 0.0, "single_mb": 0.0}
stop = threading.Event()


def monitor() -> None:
    while not stop.wait(1.0):
        try:
            out = subprocess.run(
                ["ps", "-eo", "rss,args", "--no-headers"], capture_output=True, text=True
            ).stdout.splitlines()
            total_py = 0.0
            largest = 0.0
            for line in out:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                rss_kb, args = parts
                if "python" in args or "uv" in args or "pytest" in args:
                    rss = float(rss_kb)
                    total_py += rss
                    largest = max(largest, rss)
            peak["sys_mb"] = max(peak["sys_mb"], total_py / 1024)
            peak["single_mb"] = max(peak["single_mb"], largest / 1024)
        except Exception:
            pass


def main() -> int:
    th = threading.Thread(target=monitor, daemon=True)
    th.start()

    from scripts.build_bugs import BUGS
    from scripts.exp_sampling import run_variant

    spec = next(s for s in BUGS if s["project"] == "black" and s["bug"] == 4)
    job = {"spec": spec, "mode": "uniform", "cap": 600, "tag": "black_4_rehearse"}

    t0 = time.perf_counter()
    r = run_variant(job)
    dt = time.perf_counter() - t0
    stop.set()
    th.join()

    print(f"REHEARSE: execs={r['mutants']} secs={dt:.0f}s")
    print(f"peak all-python RSS during job: {peak['sys_mb']:.0f} MB")
    print(f"peak single-process RSS:        {peak['single_mb']:.0f} MB")
    print("(1GB per-child cap is active; this is one parallel slot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())