"""Exactly time each method's real work per bug with time.perf_counter.

MBFL-real is timed by actually running every mutant's test suite (the cost it
is known for). ML-PMT / Hybrid / SBFL are timed on their actual baseline,
coverage, mutant-generation, and prediction steps. Bugs are processed in
parallel workers, but each bug's reported MBFL time is the serial sum of its
mutant runs — the honest single-threaded cost.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.pytest_runner import coverage_by_test_file, run_pytest
from ml_pmt import OutcomeChangeModel
from ml_pmt.ranker import line_aggregates
from mutation_engine.generator import generate_mutants
from scripts.build_bugs import (
    BUGS,
    BUGS_ROOT,
    WORK,
    fault_lines,
    make_sandbox,
    select_tests,
)
from scripts.evaluate_baselines import variant_environment

LIMIT = 60


def _time_bug(spec: dict) -> dict:
    project, bug = spec["project"], spec["bug"]
    variant_id = f"{project}-{bug}"
    checkout, python, packages = variant_environment(variant_id)
    bug_dir = BUGS_ROOT / project / "bugs" / str(bug)
    sandbox = make_sandbox(checkout, f"{project}_{bug}_timed", project)
    tests, _failing = select_tests(sandbox, bug_dir, python)
    fault_file = next(iter(fault_lines((bug_dir / "bug_patch.txt").read_text())))
    path = sandbox / fault_file
    original = path.read_text()

    t0 = time.perf_counter()
    mutants = generate_mutants(
        original, limit=LIMIT
    )
    t_generate = time.perf_counter() - t0

    t0 = time.perf_counter()
    run_pytest(sandbox, tests, python)
    t_baseline = time.perf_counter() - t0

    t0 = time.perf_counter()
    coverage_by_test_file(sandbox, tests, python, source_packages=packages)
    t_coverage = time.perf_counter() - t0

    # MBFL-real: actually execute every mutant's test suite, summing wall-clock.
    t_mbfl = 0.0
    for mutant in mutants:
        path.write_text(mutant["source"])
        t0 = time.perf_counter()
        try:
            run_pytest(sandbox, tests, python)
        finally:
            path.write_text(original)
        t_mbfl += time.perf_counter() - t0

    # Model predict + rank for ML-PMT (trained on the other bugs).
    pairs = pd.read_csv(ROOT / "data" / "pairs.csv")
    candidates = pairs[pairs["source_version"] == variant_id]
    train = pairs[
        ~pairs["source_version"].str.removesuffix("-fixed").eq(
            variant_id.removesuffix("-fixed")
        )
    ]
    t0 = time.perf_counter()
    model = OutcomeChangeModel(random_state=42).fit(train)
    probs = model.predict_probabilities(candidates)
    line_aggregates(candidates, probs)
    t_predict = time.perf_counter() - t0

    return {
        "variant": variant_id,
        "n_mutants": len(mutants),
        "t_generate_s": t_generate,
        "t_baseline_s": t_baseline,
        "t_coverage_s": t_coverage,
        "t_predict_s": t_predict,
        "t_mbfl_s": t_baseline + t_mbfl,
    }


def main() -> int:
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(_time_bug, BUGS))

    print("=== Exact per-bug wall-clock (serial-equivalent for MBFL) ===")
    for r in results:
        mlpmt = r["t_baseline_s"] + r["t_coverage_s"] + r["t_generate_s"] + r["t_predict_s"]
        hybrid = mlpmt
        sbfl = r["t_coverage_s"]
        print(
            f"  {r['variant']:12s} mutants={r['n_mutants']:3d} | "
            f"MBFL {r['t_mbfl_s']:7.1f}s | ML-PMT {mlpmt:6.1f}s | "
            f"Hybrid {hybrid:6.1f}s | SBFL {sbfl:5.1f}s"
        )
        r.update(mlpmt=mlpmt, hybrid=hybrid, sbfl=sbfl)

    mean = lambda key: float(np.mean([r[key] for r in results]))
    print(
        f"\nMean per bug: MBFL {mean('t_mbfl_s'):.1f}s | ML-PMT {mean('mlpmt'):.1f}s | "
        f"Hybrid {mean('hybrid'):.1f}s | SBFL {mean('sbfl'):.1f}s"
    )
    print(f"ML-PMT speedup vs MBFL-real: {mean('t_mbfl_s')/mean('mlpmt'):.0f}x")
    print(
        f"(of which mutant generation: {mean('t_generate_s'):.1f}s shared; "
        f"exclude for a generation-free comparison)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())