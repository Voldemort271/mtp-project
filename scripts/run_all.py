"""Full end-to-end run: training, evaluation, baselines, and timing benchmark.

Usage:
    python scripts/run_all.py [--rebuild]

Stages: build the BugsInPy corpus (skipped if present unless --rebuild),
train and save a model, evaluate leave-one-bug-out localization (max and
two-stage), report pair-model metrics, compare against SBFL/MBFL/hybrid, and
run the exact wall-clock benchmark.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _common import PROJECT_ROOT

SCRIPTS = PROJECT_ROOT / "scripts"
DATA = PROJECT_ROOT / "data"


def _run(script: str, *args: str, echo: bool = True) -> str:
    command = [sys.executable, str(SCRIPTS / script), *args]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    output = completed.stdout
    if echo:
        print(output, end="")
    if completed.returncode != 0:
        print(f"[run_all] ERROR in {script}: {completed.stderr[-2000:]}", file=sys.stderr)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="rebuild the corpus")
    parser.add_argument("--skip-benchmark", action="store_true", help="skip timing benchmark")
    args = parser.parse_args()

    import time as _time
    _stage_t0 = _time.monotonic()

    def _stage(num: str, label: str) -> None:
        print(f"\n[{num}] {label} ...", flush=True)

    def _stage_done(num: str) -> None:
        print(
            f"[{num}] done ({_time.monotonic() - _stage_t0:.0f}s wall since start)",
            flush=True,
        )

    print("=" * 72)
    print("ML-PMT full run (38-bug BugsInPy slice)")
    print("=" * 72)

    # Stage 1: corpus
    _stage_t0 = _time.monotonic()
    if args.rebuild or not (DATA / "pairs.csv").exists():
        _stage("1/6", "Building the BugsInPy corpus")
        _run("build_bugs.py", "--workers", "8")
    else:
        _stage("1/6", "Corpus present (use --rebuild to regenerate)")
    _stage_done("1/6")

    # Stage 2: training
    _stage_t0 = _time.monotonic()
    _stage("2/6", "Training the outcome-change model on the full corpus")
    _run("train_model.py", "--pairs", str(DATA / "pairs.csv"),
         "--output", str(PROJECT_ROOT / "artifacts" / "model.joblib"))
    _stage_done("2/6")

    # Stage 3: localization
    _stage_t0 = _time.monotonic()
    _stage("3/6", "Leave-one-bug-out localization (max formula)")
    _run("evaluate_localization.py")
    _stage("3/6", "Leave-one-bug-out localization (two-stage)")
    _run("evaluate_localization.py", "--scoring", "two-stage")
    _stage_done("3/6")

    # Stage 4: pair model
    _stage_t0 = _time.monotonic()
    _stage("4/6", "Pair-prediction model metrics")
    _run("evaluate_model.py", "--pairs", str(DATA / "pairs.csv"))
    _stage_done("4/6")

    # Stage 5: baselines + hybrid
    _stage_t0 = _time.monotonic()
    _stage("5/6", "Baselines and hybrid comparison")
    baselines = _run("evaluate_baselines.py", echo=False)
    print(baselines.split("=== Fault-line rank")[0])  # skip per-variant rank table
    # print the aggregate + cost sections
    if "=== Aggregate metrics ===" in baselines:
        print(baselines.split("=== Aggregate metrics ===")[1])
    _stage_done("5/6")

    # Stage 6: timing benchmark
    _stage_t0 = _time.monotonic()
    if not args.skip_benchmark:
        _stage("6/6", "Exact wall-clock benchmark")
        _run("benchmark_time.py")
        _stage_done("6/6")
    else:
        print("\n[6/6] Benchmark skipped.")

    print("\n" + "=" * 72)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())