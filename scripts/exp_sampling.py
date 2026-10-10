"""Small experiment: uniform@capN vs stratified_floor@capN expressibility.

Runs real mutant execution on a couple of bugs, comparing uniform vs
stratified_floor at cap 300 and 600.

Runs the (variant x config) combinations in parallel across workers, prints
in-mutant progress (unbuffered), and reports, per fault line, whether a mutant
flips a *FAILING* baseline test (the MBFL-expressible definition).
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_bugs import BUGS, BUGS_ROOT, WORK, fault_lines, make_sandbox, select_tests


def run_variant(job: dict) -> dict:
    spec = job["spec"]
    mode = job["mode"]
    cap = job["cap"]
    tag = job["tag"]
    project, bug = spec["project"], spec["bug"]
    python = WORK / spec["venv"] / "venv" / "bin" / "python"
    bug_dir = BUGS_ROOT / project / "bugs" / str(bug)
    # unique sandbox per job so parallel workers never collide
    sandbox = make_sandbox(
        WORK / f"{project}_{bug}" / project, f"{project}_{bug}_{tag}", project
    )
    tests, _failing = select_tests(sandbox, bug_dir, python)

    from execution.pytest_runner import run_pytest
    from mutation_engine.generator import generate_mutants

    faults = fault_lines((bug_dir / "bug_patch.txt").read_text())

    t0 = time.perf_counter()
    print(f"[{tag}] baseline suite running ...", flush=True)
    baseline = run_pytest(sandbox, tests, python)
    failing_tests = {t for t, r in baseline.items() if r == "fail"}
    if not failing_tests:
        print(f"[{tag}] WARN no failing baseline test", flush=True)
    print(
        f"[{tag}] baseline done: {len(tests)} tests, "
        f"{len(failing_tests)} failing, {time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    covered_lines: set[int] = set()
    fault_lines_seen: set[int] = set()
    expressible: set[int] = set()
    n_exec = 0
    flips_fail = 0
    errors = 0
    t_run = time.perf_counter()

    for rel_file, flines in faults.items():
        path = sandbox / rel_file
        if not path.exists():
            print(f"[{tag}] skip missing {rel_file}", flush=True)
            continue
        original = path.read_text()
        mutants = generate_mutants(original, limit=cap, seed=42, mode=mode)
        n_mut = len(mutants)
        print(f"[{tag}] {rel_file}: generated {n_mut} mutants (mode={mode} cap={cap})", flush=True)

        for i, mutant in enumerate(mutants, 1):
            covered_lines.add(mutant["line"])
            path.write_text(mutant["source"])
            try:
                mres = run_pytest(sandbox, tests, python)
                n_exec += 1
            except Exception as exc:  # noqa: BLE001
                errors += 1
                mres = {"_suite_error": "fail"}
            finally:
                path.write_text(original)

            flipped = "_suite_error" not in mres and any(
                mres.get(t) != r for t, r in baseline.items() if t in failing_tests
            )
            if flipped:
                flips_fail += 1
                if mutant["line"] in flines:
                    expressible.add(mutant["line"])
            if mutant["line"] in flines:
                fault_lines_seen.add(mutant["line"])

            if i % 25 == 0 or i == n_mut:
                rate = i / max(time.perf_counter() - t_run, 1e-9)
                print(
                    f"[{tag}]   {i}/{n_mut} mutants | lines={len(covered_lines)} "
                    f"fault_seen={len(fault_lines_seen)} expr={len(expressible)} "
                    f"flips={flips_fail} @{rate:.2f}/s",
                    flush=True,
                )

    return {
        "tag": tag,
        "variant": f"{project}-{bug}",
        "mode": mode,
        "cap": cap,
        "mutants": n_exec,
        "covered_lines": len(covered_lines),
        "fault_lines_seen": sorted(fault_lines_seen),
        "expressible": sorted(expressible),
        "flips_fail": flips_fail,
        "errors": errors,
        "seconds": time.perf_counter() - t0,
    }


def main() -> int:
    specs = [
        s for s in BUGS
        if (s["project"], s["bug"]) in {
            ("tqdm", 1),      # SVR: arg reposition
            ("tqdm", 5),      # guards/assigns
            ("tqdm", 6),      # LVR: self.total
            ("black", 4),     # baseline big-file
            ("black", 5),     # EXM
            ("black", 6),     # bitwise-heavy (BOR)
            ("black", 10),    # EXM
            ("black", 13),    # EXM
            ("black", 20),    # ATTR: src.name
            ("thefuck", 6),   # regex SCP
            ("thefuck", 25),  # regex SCP
            ("thefuck", 28),  # regex SCP
        }
    ]
    modes_caps = (
        ("uniform", 300),
        ("stratified_floor", 300),
    )
    jobs = []
    for spec in specs:
        for mode, cap in modes_caps:
            tag = f"{spec['project']}_{spec['bug']}_{mode}_{cap}"
            jobs.append({"spec": spec, "mode": mode, "cap": cap, "tag": tag})

    summary_path = ROOT / "data" / "exp_sampling.out"
    with summary_path.open("w") as summary:
        summary.write(
            "variant     mode               cap   execs  lines  fault_seen              EXPRESSIBLE              flips errors  seconds\n"
        )
        with ProcessPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(run_variant, j): j["tag"] for j in jobs}
            for fut in as_completed(futures):
                r = fut.result()
                line = (
                    f"{r['variant']:10s} {r['mode']:16s} cap={r['cap']:<4d} "
                    f"{r['mutants']:5d} {r['covered_lines']:5d} "
                    f"{str(r['fault_lines_seen']):24s} {str(r['expressible']):24s} "
                    f"{r['flips_fail']:5d} {r['errors']:4d} {r['seconds']:8.0f}"
                )
                print(line, flush=True)
                summary.write(line + "\n")
    print("done. summary at data/exp_sampling.out", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())