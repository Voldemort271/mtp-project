"""Triage BugsInPy bugs for the corpus: reproduce, find fault lines, check
that the fault lines are operator-mutatable. Prints a keep/drop table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.pytest_runner import run_pytest
from mutation_engine.generator import generate_mutants
from scripts.build_bugs import BUGS_ROOT, WORK, fault_lines, prepare_checkout, select_tests


def triage(project: str, bug: int, venv: str) -> dict:
    python = WORK / venv / "venv" / "bin" / "python"
    bug_dir = BUGS_ROOT / project / "bugs" / str(bug)
    checkout = WORK / f"{project}_{bug}" / project
    fixed = WORK / f"{project}_{bug}_fixed" / project
    prepare_checkout(checkout, project)
    prepare_checkout(fixed, project)
    patch = (bug_dir / "bug_patch.txt").read_text()
    faults = fault_lines(patch)

    try:
        tests, failing = select_tests(checkout, bug_dir, python)
    except RuntimeError:
        return {"project": project, "bug": bug, "keep": False,
                "mutatable_fault_lines": {},
                "reason": "select_tests failed (no failing target or collection error)"}
    fixed_res = run_pytest(fixed, tests, python)
    fixed_pass = sum(1 for v in fixed_res.values() if v == "pass")

    mutatable: dict[str, list[int]] = {}
    for file, lines in faults.items():
        src_path = checkout / file
        if not src_path.exists():
            mutatable[file] = []
            continue
        mutants = generate_mutants(
            src_path.read_text(), limit=60, priority_lines=frozenset(lines)
        )
        mutatable[file] = sorted(
            set(lines) & {m["line"] for m in mutants}
        )

    any_mutatable = any(mutatable.values())
    keep = bool(failing) and fixed_pass > 0 and any_mutatable
    return {
        "project": project,
        "bug": bug,
        "keep": keep,
        "reason": None,
        "failing": [t.rsplit("::", 1)[-1] for t in failing],
        "fixed_passing": fixed_pass,
        "mutatable_fault_lines": mutatable,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="thefuck")
    parser.add_argument("--venv", default="thefuck_5")
    parser.add_argument("--bugs", nargs="+", type=int, required=True)
    args = parser.parse_args()

    for bug in args.bugs:
        info = triage(args.project, bug, args.venv)
        mut = "; ".join(
            f"{f}:{','.join(map(str, ls))}" for f, ls in info["mutatable_fault_lines"].items()
        )
        if info.get("reason"):
            print(f"bug {bug:>2} | keep=False | reason={info['reason']}")
        else:
            print(
                f"bug {bug:>2} | keep={info['keep']!s:5} | "
                f"failing={info['failing']} fixed_pass={info['fixed_passing']} | "
                f"mutatable={mut or 'none'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())