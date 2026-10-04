"""Provision the BugsInPy benchmark environment for the pipeline.

Automates the README setup steps: installs the project deps, clones the BugsInPy
framework, creates the per-project test venvs at the right Python version,
installs each project's test dependencies, and checks out every bug (buggy +
fixed). Idempotent: completed steps are skipped. The corpus build is run only
with ``--build``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUGSINPY_URL = "https://github.com/soarsmu/BugsInPy.git"

# Per-project test environment: Python version and required test deps.
PROJECTS = {
    "tqdm": {
        "python": "3.11",
        "deps": ["pytest", "pytest-cov", "coverage", "numpy", "nose"],
    },
    "thefuck": {
        "python": "3.11",
        "deps": [
            "pytest", "pytest-cov", "coverage", "mock", "pytest-mock",
            "colorama", "decorator", "psutil", "pyte", "six", "wcwidth",
        ],
    },
    "black": {
        "python": "3.11",
        "deps": [
            "pytest", "pytest-cov", "coverage", "click", "appdirs",
            "pathspec", "toml", "regex", "attrs", "typed-ast",
        ],
    },
}


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(str(c) for c in command))
    return subprocess.run(command, cwd=ROOT, check=check)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build", action="store_true", help="also run the corpus build afterwards"
    )
    args = parser.parse_args()

    from scripts.build_bugs import BUGS, WORK

    # 1. Install the project's own deps.
    if not (ROOT / ".venv").exists():
        print("[1/4] Installing project deps (uv sync)")
        run(["uv", "sync"])
    else:
        print("[1/4] Project deps present")

    # 2. Clone the BugsInPy framework.
    bugs_dir = ROOT / "benchmarks" / "BugsInPy"
    print("[2/4] BugsInPy framework")
    if not (bugs_dir / "framework").exists():
        run(["git", "clone", "--depth", "1", BUGSINPY_URL, str(bugs_dir)])
    else:
        print("  framework already present")

    # 3. Create per-project venvs and install their test deps.
    print("[3/4] Per-project test venvs")
    venv_to_project: dict[str, str] = {}
    for spec in BUGS:
        venv_to_project.setdefault(spec["venv"], spec["project"])
    for venv_name, project in venv_to_project.items():
        cfg = PROJECTS[project]
        python_bin = WORK / venv_name / "venv" / "bin" / "python"
        if not python_bin.exists():
            run(["uv", "venv", str(WORK / venv_name / "venv"), "--python", cfg["python"]])
        run(["uv", "pip", "install", "--python", str(python_bin), *cfg["deps"]])

    # 4. Check out every bug (buggy + fixed).
    print("[4/4] Checking out bugs")
    checkout_script = bugs_dir / "framework" / "bin" / "bugsinpy-checkout"
    for spec in BUGS:
        project, bug = spec["project"], spec["bug"]
        for version, label in (
            (0, f"{project}_{bug}"),
            (1, f"{project}_{bug}_fixed"),
        ):
            target = WORK / label
            if (target / project).exists():
                print(f"  {label} present")
                continue
            run(
                [
                    "bash",
                    str(checkout_script),
                    "-p", project,
                    "-i", str(bug),
                    "-v", str(version),
                    "-w", str(target),
                ]
            )

    print("\nBenchmark environment ready.")
    print("Next: uv run python scripts/build_bugs.py --workers 8")
    if args.build:
        run(["uv", "run", "python", "scripts/build_bugs.py", "--workers", "8"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())