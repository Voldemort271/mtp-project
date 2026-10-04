"""Build a historical training corpus from a small BugsInPy slice."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.pytest_runner import coverage_by_test, run_pytest
from features.extractor import node_metadata
from mutation_engine.generator import generate_mutants

WORK = ROOT / "benchmarks" / "work"
BUGS_ROOT = ROOT / "benchmarks" / "BugsInPy" / "projects"

BUGS = [
    {"project": "tqdm", "bug": 1, "venv": "tqdm_1", "packages": ["tqdm"]},
    {"project": "tqdm", "bug": 2, "venv": "tqdm_1", "packages": ["tqdm"]},
    {"project": "tqdm", "bug": 4, "venv": "tqdm_1", "packages": ["tqdm"]},
    {"project": "tqdm", "bug": 5, "venv": "tqdm_1", "packages": ["tqdm"]},
    {"project": "tqdm", "bug": 9, "venv": "tqdm_1", "packages": ["tqdm"]},
    {"project": "thefuck", "bug": 1, "venv": "thefuck_5", "packages": ["thefuck"]},
    {"project": "thefuck", "bug": 2, "venv": "thefuck_5", "packages": ["thefuck"]},
    {"project": "thefuck", "bug": 5, "venv": "thefuck_5", "packages": ["thefuck"]},
    {"project": "thefuck", "bug": 6, "venv": "thefuck_5", "packages": ["thefuck"]},
    {"project": "thefuck", "bug": 7, "venv": "thefuck_5", "packages": ["thefuck"]},
    {"project": "thefuck", "bug": 8, "venv": "thefuck_5", "packages": ["thefuck"]},
    {"project": "thefuck", "bug": 25, "venv": "thefuck_5", "packages": ["thefuck"]},
    {"project": "thefuck", "bug": 26, "venv": "thefuck_5", "packages": ["thefuck"]},
    {"project": "thefuck", "bug": 27, "venv": "thefuck_5", "packages": ["thefuck"]},
    {"project": "thefuck", "bug": 28, "venv": "thefuck_5", "packages": ["thefuck"]},
    {"project": "thefuck", "bug": 30, "venv": "thefuck_5", "packages": ["thefuck"]},
    {"project": "thefuck", "bug": 32, "venv": "thefuck_5", "packages": ["thefuck"]},
    {"project": "black", "bug": 4, "venv": "black_4", "packages": ["black", "blib2to3"]},
    {"project": "black", "bug": 5, "venv": "black_4", "packages": ["black", "blib2to3"]},
    {"project": "black", "bug": 6, "venv": "black_4", "packages": ["black", "blib2to3"]},
    {"project": "black", "bug": 7, "venv": "black_4", "packages": ["black", "blib2to3"]},
    {"project": "black", "bug": 8, "venv": "black_4", "packages": ["black", "blib2to3"]},
    {"project": "black", "bug": 10, "venv": "black_4", "packages": ["black", "blib2to3"]},
    {"project": "black", "bug": 11, "venv": "black_4", "packages": ["black", "blib2to3"]},
    {"project": "black", "bug": 12, "venv": "black_4", "packages": ["black", "blib2to3"]},
    {"project": "black", "bug": 13, "venv": "black_4", "packages": ["black", "blib2to3"]},
    {"project": "black", "bug": 15, "venv": "black_4", "packages": ["black", "blib2to3"]},
    {"project": "black", "bug": 16, "venv": "black_4", "packages": ["black", "blib2to3"]},
    {"project": "black", "bug": 17, "venv": "black_4", "packages": ["black", "blib2to3"]},
    {"project": "black", "bug": 18, "venv": "black_4", "packages": ["black", "blib2to3"]},
    {"project": "black", "bug": 21, "venv": "black_4", "packages": ["black", "blib2to3"]},
    {"project": "black", "bug": 23, "venv": "black_4", "packages": ["black", "blib2to3"]},
]

# Test-infrastructure patches required to run old test suites on modern pytest.
CONFTEST_PATCH = "request.node.get_marker('functional')"
CONFTEST_REPLACEMENT = "request.node.get_closest_marker('functional')"

HUNK_RE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
TARGET_RE = re.compile(r"::([\w]+)")


def prepare_checkout(checkout: Path, project: str) -> None:
    """Apply any test-infra patches needed to run the suite on this pytest."""
    if project == "thefuck":
        conftest = checkout / "tests" / "conftest.py"
        if conftest.exists():
            text = conftest.read_text()
            if CONFTEST_PATCH in text:
                conftest.write_text(text.replace(CONFTEST_PATCH, CONFTEST_REPLACEMENT))


def make_sandbox(checkout: Path, name: str, project: str) -> Path:
    """Copy a checkout into a disposable sandbox for in-place mutant runs.

    The source checkout is never mutated, so a killed build cannot corrupt it.
    """
    target = WORK / "_sandbox" / name
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns(
        ".git",
        ".coverage",
        ".pytest_cache",
        "__pycache__",
        "*.pyc",
        "venv",
        ".venv",
    )
    shutil.copytree(
        checkout,
        target,
        ignore=ignore,
        symlinks=True,
    )
    prepare_checkout(target, project)
    return target


def read_info(bug_dir: Path) -> dict:
    info = {}
    for line in (bug_dir / "bug.info").read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            info[key.strip()] = value.strip().strip('"')
    return info


def fault_lines(patch_text: str) -> dict[str, list[int]]:
    """Map file -> buggy line numbers the fix removes or changes."""
    result: dict[str, list[int]] = {}
    current_file = None
    old_line = None
    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            match = re.match(r"diff --git a/(\S+) b/", line)
            current_file = match.group(1)
            result[current_file] = []
        elif line.startswith("@@"):
            match = HUNK_RE.match(line)
            old_line = int(match.group(1))
        elif line.startswith("-") and not line.startswith("---"):
            if current_file is not None and old_line is not None:
                result[current_file].append(old_line)
                old_line += 1
        elif line.startswith("+") and not line.startswith("+++"):
            pass
        elif line.startswith(" ") or line.startswith("\\"):
            if old_line is not None:
                old_line += 1
    return {f: lines for f, lines in result.items() if lines}


def select_tests(
    checkout: Path,
    bug_dir: Path,
    python: Path,
) -> tuple[list[str], list[str]]:
    """Return concrete test nodeids and which of them fail on the buggy version.

    ``test_file`` may be semicolon-separated (a test module plus data files);
    only non-``/data/`` ``.py`` modules are run. Targets from ``run_test.sh``
    may name a parametrized test (``foo[case]``) or a dotted unittest test
    (``module.Class.test_name``); both are matched by the test method name.
    """
    info = read_info(bug_dir)
    test_modules = _test_modules(info["test_file"])
    if not test_modules:
        raise RuntimeError(f"No test module in test_file for {bug_dir}")
    results: dict[str, str] = {}
    for module in test_modules:
        results.update(run_pytest(checkout, [module], python))
    if "_suite_error" in results:
        raise RuntimeError(f"Test collection failed for {checkout}")
    targets = target_tests((bug_dir / "run_test.sh").read_text(), test_modules[0])

    failing = [
        test_id
        for target in targets
        for test_id, result in results.items()
        if result == "fail" and matches(test_id, target)
    ]
    if not failing:
        raise RuntimeError(f"No target test fails on {checkout}")
    passing = [tid for tid, result in results.items() if result == "pass"][:6]
    return failing + passing, failing


def _test_modules(test_file: str) -> list[str]:
    """Return the test modules from a possibly semicolon-separated test_file."""
    return [
        part.strip()
        for part in test_file.split(";")
        if part.strip().endswith(".py") and "/data/" not in part
    ]


def matches(test_id: str, target: str) -> bool:
    """True when a nodeid corresponds to a (possibly parametrized) target."""
    return (
        test_id == target
        or test_id.endswith("::" + target)
        or test_id.startswith(target + "[")
    )


def target_tests(run_test_sh: str, test_file: str) -> list[str]:
    """Extract target test names from pytest or unittest run commands."""
    targets = []
    for line in run_test_sh.splitlines():
        if "::" in line:
            name = TARGET_RE.search(line)
            if name:
                targets.append(f"{test_file}::{name.group(1)}")
        elif "unittest" in line:
            identifiers = re.findall(r"[A-Za-z_]\w*", line)
            if identifiers:
                targets.append(identifiers[-1])
    return targets


def _clean_sandbox_state(checkout: Path) -> None:
    """Remove per-run state so it cannot accumulate across mutants."""
    import shutil as _shutil

    for pattern in (".thefuck", ".coverage", ".pytest_cache", "__pycache__"):
        for match in checkout.glob(pattern):
            if match.is_dir():
                _shutil.rmtree(match, ignore_errors=True)
            else:
                match.unlink(missing_ok=True)
    for match in checkout.rglob("*.pyc"):
        match.unlink(missing_ok=True)


def _stable_results(
    checkout: Path,
    tests: list[str],
    python: Path,
    max_attempts: int = 3,
) -> dict[str, str]:
    """Run a mutant's tests until two runs agree; drop per-test flakiness.

    Returns only the tests whose outcome is stable across attempts. A test that
    keeps disagreeing is omitted so it never enters the corpus.
    """
    attempts = []
    for _ in range(max_attempts):
        result = run_pytest(checkout, tests, python)
        attempts.append(result)
        if len(attempts) >= 2 and attempts[-1] == attempts[-2]:
            return result
    stable = {}
    for test_id in tests:
        votes = Counter(attempt.get(test_id) for attempt in attempts)
        outcome, count = votes.most_common(1)[0]
        if count >= 2:
            stable[test_id] = outcome
    return stable


def version_pairs(
    checkout: Path,
    relative_file: str,
    *,
    fault_lines_set: set[int],
    tests: list[str],
    python: Path,
    source_packages: list[str],
    max_mutants: int,
    version_name: str,
) -> tuple[list[dict], int]:
    """Run all selected mutants of one file and label every test–mutant pair."""
    path = checkout / relative_file
    original = path.read_text()
    mutants = generate_mutants(
        original,
        limit=max_mutants,
        priority_lines=frozenset(fault_lines_set),
    )

    baseline_results = run_pytest(checkout, tests, python)
    coverage = coverage_by_test(
        checkout,
        tests,
        python,
        source_packages=source_packages,
    )

    rows: list[dict] = []
    flaky_cells = 0
    for index, mutant in enumerate(mutants):
        path.write_text(mutant["source"])
        try:
            mutant_results = _stable_results(checkout, tests, python)
        finally:
            path.write_text(original)
            _clean_sandbox_state(checkout)
        flaky_cells += sum(
            1 for test_id in tests if test_id not in mutant_results
        )
        metadata = node_metadata(
            original,
            mutant["line"],
            mutant["mutation_type"],
            mutant["detail"],
        )
        for test_id in tests:
            if test_id not in mutant_results:
                continue
            baseline_result = baseline_results.get(test_id, "fail")
            mutant_result = mutant_results[test_id]
            rows.append(
                {
                    "source_version": version_name,
                    "mutant_id": f"{version_name}::{relative_file}::{mutant['mutant_id']}",
                    "line_number": mutant["line"],
                    "file": relative_file,
                    "test_id": test_id,
                    "mutation_type": mutant["mutation_type"],
                    "mutated_operator": mutant["operator"],
                    "replacement_operator": mutant["detail"],
                    "ast_depth": metadata["ast_depth"],
                    "parent_node_type": metadata["parent_node_type"],
                    "test_covers_mutation": int(
                        mutant["line"] in coverage.get(test_id, set())
                    ),
                    "baseline_test_result": baseline_result,
                    "outcome_changed": int(mutant_result != baseline_result),
                }
            )
    return rows, flaky_cells


def _select_mutants(
    mutants: list[dict],
    fault_lines_set: set[int],
    max_mutants: int,
) -> list[dict]:
    """Prioritize mutants on the fault lines, then sample deterministically.

    Retained for tests and the synthetic corpus; ``version_pairs`` now passes
    the limit and priority lines straight to ``generate_mutants``.
    """
    ordered = sorted(
        mutants,
        key=lambda m: (m["line"] not in fault_lines_set, m["line"], m["mutant_id"]),
    )
    return ordered[:max_mutants]


def _version_task(spec: dict, tests: list[str], is_fixed: bool, task_index: int):
    """Build one source version's rows in its own sandbox (a parallel unit)."""
    project = spec["project"]
    bug = spec["bug"]
    max_mutants = spec.get("max_mutants", 60)
    python = WORK / spec["venv"] / "venv" / "bin" / "python"
    source_packages = spec["packages"]
    bug_dir = BUGS_ROOT / project / "bugs" / str(bug)
    checkout = WORK / f"{project}_{bug}" / project
    fixed_checkout = WORK / f"{project}_{bug}_fixed" / project
    name = f"{project}_{bug}_{'f' if is_fixed else 'b'}_{task_index}"
    sandbox = make_sandbox(
        fixed_checkout if is_fixed else checkout, name, project
    )
    faults = fault_lines((bug_dir / "bug_patch.txt").read_text())
    version_name = f"{project}-{bug}" + ("-fixed" if is_fixed else "")

    rows: list[dict] = []
    fault_rows: list[dict] = []
    flaky = 0
    for relative_file, lines in faults.items():
        version_rows, version_flaky = version_pairs(
            sandbox,
            relative_file,
            fault_lines_set=set(lines) if not is_fixed else set(),
            tests=tests,
            python=python,
            source_packages=source_packages,
            max_mutants=max_mutants,
            version_name=version_name,
        )
        rows.extend(version_rows)
        flaky += version_flaky
        if not is_fixed:
            fault_rows.extend(
                {"source_version": version_name, "fault_line": line}
                for line in lines
            )
    return rows, fault_rows, flaky


def build_corpus(
    out_dir: Path = ROOT / "data",
    bugs: list[dict] | None = None,
    *,
    workers: int = 1,
) -> dict:
    rows: list[dict] = []
    fault_rows: list[dict] = []
    total_flaky = 0

    tasks: list[tuple] = []
    for spec in bugs or BUGS:
        project = spec["project"]
        bug = spec["bug"]
        python = WORK / spec["venv"] / "venv" / "bin" / "python"
        bug_dir = BUGS_ROOT / project / "bugs" / str(bug)
        checkout = WORK / f"{project}_{bug}" / project
        sandbox = make_sandbox(checkout, f"{project}_{bug}_select", project)
        tests, failing = select_tests(sandbox, bug_dir, python)
        faults = fault_lines((bug_dir / "bug_patch.txt").read_text())
        print(
            f"  {project} bug {bug}: {len(tests)} tests "
            f"({len(failing)} failing: {[t.rsplit('::', 1)[-1] for t in failing]}), "
            f"fault files {list(faults)}"
        )
        tasks.append((spec, tests, False, len(tasks)))
        tasks.append((spec, tests, True, len(tasks)))

    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_version_task, *task) for task in tasks]
            for future in futures:
                version_rows, version_faults, flaky = future.result()
                rows.extend(version_rows)
                fault_rows.extend(version_faults)
                total_flaky += flaky
    else:
        for task in tasks:
            version_rows, version_faults, flaky = _version_task(*task)
            rows.extend(version_rows)
            fault_rows.extend(version_faults)
            total_flaky += flaky

    if total_flaky:
        print(f"Dropped {total_flaky} flaky (mutant, test) cells.")

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.DataFrame(rows)
    pairs.to_csv(out_dir / "pairs.csv", index=False)
    faults = pd.DataFrame(fault_rows)
    faults.to_csv(out_dir / "fault_labels.csv", index=False)
    return {
        "pairs_path": str(out_dir / "pairs.csv"),
        "faults_path": str(out_dir / "fault_labels.csv"),
        "num_rows": len(pairs),
        "num_variants": len(pairs["source_version"].unique()) if len(pairs) else 0,
        "num_faults": len(faults),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data",
        help="directory to write pairs.csv and fault_labels.csv",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel workers for the mutant executions (0 = auto)",
    )
    args = parser.parse_args()

    workers = args.workers or min(os.cpu_count() or 4, 8)
    summary = build_corpus(out_dir=args.out, workers=workers)
    print(
        f"Wrote {summary['num_rows']} pair rows across "
        f"{summary['num_variants']} source versions and "
        f"{summary['num_faults']} known fault lines."
    )
    print(f"Pairs:  {summary['pairs_path']}")
    print(f"Faults: {summary['faults_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())