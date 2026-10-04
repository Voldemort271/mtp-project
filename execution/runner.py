"""Execute baseline and mutant modules against the shared test oracle."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import coverage

SUBJECT_FILENAME = "subject.py"


def load_module(
    source_text: str,
    module_name: str,
) -> tuple[object, Path]:
    """Write ``source_text`` to a temp module and import it."""
    directory = Path(tempfile.mkdtemp(prefix="mlpmt_"))
    path = directory / SUBJECT_FILENAME
    path.write_text(source_text)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def unload_module(module_name: str) -> None:
    sys.modules.pop(module_name, None)


def baseline_results(
    source_text: str,
    tests: list,
    module_name: str = "__mlpmt_baseline__",
) -> tuple[dict[str, str], dict[str, list[int]]]:
    """Return per-test results and per-test covered line numbers."""
    module, path = load_module(source_text, module_name)
    results: dict[str, str] = {}
    coverage_by_test: dict[str, list[int]] = {}
    try:
        for test_id, test_fn in tests:
            cov = coverage.Coverage(data_file=None)
            cov.start()
            try:
                test_fn(module)
                results[test_id] = "pass"
            except Exception:
                results[test_id] = "fail"
            cov.stop()
            data = cov.get_data()
            covered: list[int] = []
            for filename in data.measured_files():
                if filename.endswith(SUBJECT_FILENAME):
                    covered = sorted(data.lines(filename) or [])
            coverage_by_test[test_id] = covered
    finally:
        unload_module(module_name)
    return results, coverage_by_test


def mutant_results(
    mutant_source: str,
    tests: list,
    module_name: str,
) -> dict[str, str]:
    """Return per-test pass/fail outcomes for one mutant module."""
    module, path = load_module(mutant_source, module_name)
    results: dict[str, str] = {}
    try:
        for test_id, test_fn in tests:
            try:
                test_fn(module)
                results[test_id] = "pass"
            except Exception:
                results[test_id] = "fail"
    finally:
        unload_module(module_name)
    return results