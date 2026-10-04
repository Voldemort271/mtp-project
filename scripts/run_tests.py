"""Run the project's pytest suite from the repository root."""

from __future__ import annotations

import argparse
import subprocess
import sys

from _common import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _, pytest_args = parser.parse_known_args()
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *pytest_args],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
