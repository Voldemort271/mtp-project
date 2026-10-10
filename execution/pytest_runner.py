"""Subprocess pytest runner for real multi-file projects."""

from __future__ import annotations

import os
import resource
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import coverage

# Cap a single mutant's address space so a pathological mutant (e.g. a
# MOR-induced ``x ** huge`` producing an astronomical bignum, or a runaway
# string) is killed with MemoryError instead of exhausting the whole machine.
# Without this, one mutant can allocate >10 GB in seconds and trigger the OS
# OOM killer before the wall-clock timeout can intervene. A normal pytest run
# for these suites peaks at ~120 MB VSZ, so 1 GB is ~8x headroom and never
# affects legitimate mutants, while 4 workers stay well within 14 GB RAM.
MUTANT_MEMORY_LIMIT_BYTES = 1 * 1024**3


def _limit_address_space() -> None:
    """``preexec_fn`` that caps the child's address space."""
    try:
        resource.setrlimit(
            resource.RLIMIT_AS,
            (MUTANT_MEMORY_LIMIT_BYTES, MUTANT_MEMORY_LIMIT_BYTES),
        )
    except (ValueError, OSError):
        pass


def run_pytest(
    project_dir: Path,
    test_args: list[str],
    python: Path | str,
) -> dict[str, str]:
    """Run pytest and return ``{node_id: 'pass'|'fail'}`` from junitxml.

    Any collected test that fails or errors is reported as ``fail``. If the
    whole run errors before collecting tests, ``{'_suite_error': 'fail'}`` is
    returned so callers can treat collection/import breakage as a failure.

    The child is memory- and time-bounded (see ``MUTANT_MEMORY_LIMIT_BYTES``
    and ``timeout``) and its stdout/stderr are discarded, so a single runaway
    mutant cannot accumulate gigabytes of buffered output or exhaust system RAM.
    """
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as handle:
        xml_path = Path(handle.name)
    try:
        command = [
            str(python),
            "-m",
            "pytest",
            "-c",
            "/dev/null",
            "--rootdir",
            str(project_dir),
            *test_args,
            "-q",
            "--no-header",
            "--tb=no",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            "--junitxml",
            str(xml_path),
        ]
        completed = subprocess.run(
            command,
            cwd=project_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_subprocess_env(project_dir),
            timeout=300,
            preexec_fn=_limit_address_space,
        )
        return _parse_junitxml(xml_path, collected_ok=completed.returncode < 5)
    except subprocess.TimeoutExpired:
        return {"_suite_error": "fail"}
    finally:
        xml_path.unlink(missing_ok=True)


def coverage_by_test(
    project_dir: Path,
    test_args: list[str],
    python: Path | str,
    *,
    source_packages: list[str],
) -> dict[str, set[int]]:
    """Return merged per-test covered line numbers using coverage contexts."""
    per_file = coverage_by_test_file(
        project_dir, test_args, python, source_packages=source_packages
    )
    return {
        test_id: set().union(*(lines.values() if lines else []))
        for test_id, lines in per_file.items()
    }


def coverage_by_test_file(
    project_dir: Path,
    test_args: list[str],
    python: Path | str,
    *,
    source_packages: list[str],
) -> dict[str, dict[str, set[int]]]:
    """Return ``{test_id: {file_path: set(lines)}}`` via coverage contexts."""
    coverage_file = project_dir / ".coverage"
    if coverage_file.exists():
        coverage_file.unlink()
    command = [
        str(python),
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "--rootdir",
        str(project_dir),
        *test_args,
        "-q",
        "--no-header",
        "--tb=no",
        "-p",
        "no:cacheprovider",
        "--cov=" + ",".join(source_packages),
        "--cov-context=test",
    ]
    subprocess.run(
        command,
        cwd=project_dir,
        capture_output=True,
        text=True,
        env=_subprocess_env(project_dir),
    )

    if not coverage_file.exists():
        return {}
    cov = coverage.Coverage(data_file=str(coverage_file))
    cov.load()
    data = cov.get_data()
    result: dict[str, dict[str, set[int]]] = {}
    for filename in data.measured_files():
        contexts_by_line = data.contexts_by_lineno(filename)
        for line, contexts in contexts_by_line.items():
            for context in contexts:
                test_id = context.removesuffix("|run")
                result.setdefault(test_id, {}).setdefault(filename, set()).add(line)
    return result


def _subprocess_env(project_dir: Path) -> dict:
    """Deterministic environment: fixed hash seed and per-sandbox HOME.

    A fixed ``PYTHONHASHSEED`` removes hash-randomization variance; isolating
    ``HOME`` to the sandbox stops user-config state (e.g. ``~/.thefuck``) from
    leaking between builds.
    """
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env["HOME"] = str(project_dir)
    return env


def _parse_junitxml(xml_path: Path, *, collected_ok: bool) -> dict[str, str]:
    if not xml_path.exists() or not collected_ok:
        return {"_suite_error": "fail"}
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return {"_suite_error": "fail"}
    results: dict[str, str] = {}
    for case in root.iter("testcase"):
        classname = case.get("classname") or ""
        name = case.get("name") or ""
        results[_nodeid(classname, name)] = (
            "fail"
            if case.find("failure") is not None or case.find("error") is not None
            else "pass"
        )
    if not results:
        return {"_suite_error": "fail"}
    return results


def _nodeid(classname: str, name: str) -> str:
    """Reconstruct a pytest nodeid that pytest can re-collect.

    junitxml's ``classname`` is the dotted module for plain pytest tests
    (``tqdm.tests.tests_tqdm``) but includes the test class for unittest
    (``tests.test_black.BlackTestCase``). A trailing CamelCase segment marks a
    unittest class; the nodeid then carries ``File::ClassName::test``.
    """
    parts = [part for part in classname.split(".") if part]
    if not parts:
        return name
    if parts[-1][0].isupper():
        file_path = "/".join(parts[:-1]) + ".py"
        return f"{file_path}::{parts[-1]}::{name}"
    file_path = "/".join(parts) + ".py"
    return f"{file_path}::{name}"