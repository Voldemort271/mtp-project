"""Analyze BugsInPy bugs: fault lines, mutant coverage of fault lines, test outcomes."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.pytest_runner import run_pytest
from mutation_engine.generator import generate_mutants

WORK = ROOT / "benchmarks" / "work"
BUGS = ROOT / "benchmarks" / "BugsInPy" / "projects" / "tqdm" / "bugs"
VENV_PYTHON = WORK / "tqdm_1" / "venv" / "bin" / "python"

HUNK_RE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def faulty_lines(patch_text: str) -> dict[str, list[int]]:
    """Map file -> buggy line numbers that the fix removes/changes."""
    result: dict[str, list[int]] = {}
    current_file = None
    old_line = None
    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            current_file = re.match(r"diff --git a/(\S+) b/", line).group(1)
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


def main() -> int:
    for bug_id in range(1, 10):
        bug_dir = BUGS / str(bug_id)
        info = {}
        for line in (bug_dir / "bug.info").read_text().splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                info[key.strip()] = value.strip().strip('"')
        patch = (bug_dir / "bug_patch.txt").read_text()
        faults = faulty_lines(patch)
        checkout = WORK / f"tqdm_{bug_id}" / "tqdm"

        print(f"\n=== tqdm bug {bug_id} ===")
        print(f"  test_file: {info.get('test_file')}")
        for file, lines in faults.items():
            src_path = checkout / file
            if not src_path.exists():
                print(f"  fault file {file}: NOT PRESENT in checkout")
                continue
            source = src_path.read_text()
            mutants = generate_mutants(source)
            mutant_lines = {m["line"] for m in mutants}
            findable = sorted(set(lines) & mutant_lines)
            print(
                f"  {file}: fault lines {sorted(lines)} | "
                f"mutants on fault lines: {findable} | "
                f"total mutants {len(mutants)}"
            )

        # run the test file (whole file) and report pass/fail summary
        test_file = info.get("test_file", "")
        if test_file:
            results = run_pytest(
                checkout,
                [test_file],
                VENV_PYTHON,
            )
            failing = [tid for tid, res in results.items() if res == "fail"]
            print(f"  tests: {len(results)} total, {len(failing)} failing -> {failing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())