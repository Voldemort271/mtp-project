"""Streamlined entry point for running the ML-PMT pipeline in its modes.

Usage:
    python scripts/mlpmt.py build        # build the BugsInPy historical corpus
    python scripts/mlpmt.py status       # show corpus/model status
    python scripts/mlpmt.py train        # fit and save the model
    python scripts/mlpmt.py evaluate     # leave-one-bug-out localization metrics
    python scripts/mlpmt.py pairs        # pair-prediction metrics (AUC/AP/Brier)
    python scripts/mlpmt.py baselines    # compare vs SBFL and real-MBFL
    python scripts/mlpmt.py compare      # compare model configurations
    python scripts/mlpmt.py localize ... # rank lines for a new buggy program
    python scripts/mlpmt.py demo         # synthetic smoke demo
    python scripts/mlpmt.py test         # run the project test suite

Each mode delegates to the focused script that implements it, so the
fine-grained scripts remain the source of truth.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _common import PROJECT_ROOT

SCRIPTS = PROJECT_ROOT / "scripts"
DATA = PROJECT_ROOT / "data"
ARTIFACTS = PROJECT_ROOT / "artifacts"
DEFAULT_MODEL = ARTIFACTS / "model.joblib"


def _run(script: str, *args: str) -> int:
    command = [sys.executable, str(SCRIPTS / script), *args]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def mode_build(args) -> int:
    extra = ["--out", str(args.out)] if args.out else []
    if args.no_clean:
        extra.append("--no-clean")
    if args.workers:
        extra += ["--workers", str(args.workers)]
    return _run("build_bugs.py", *extra)


def mode_status(args) -> int:
    pairs = DATA / "pairs.csv"
    faults = DATA / "fault_labels.csv"
    if pairs.exists() and faults.exists():
        import pandas as pd

        pairs_frame = pd.read_csv(pairs)
        faults_frame = pd.read_csv(faults)
        print(f"pairs.csv:        {len(pairs_frame)} rows, "
              f"{pairs_frame['source_version'].nunique()} source versions")
        print(f"fault_labels.csv: {len(faults_frame)} known fault lines across "
              f"{faults_frame['source_version'].nunique()} bugs")
        if DEFAULT_MODEL.exists():
            print(f"model.joblib:     saved ({DEFAULT_MODEL.stat().st_size / 1024:.0f} KiB)")
        else:
            print("model.joblib:     not trained yet (run 'mlpmt.py train')")
        return 0
    print("No corpus found at data/pairs.csv. Run 'python scripts/mlpmt.py build'.")
    return 1


def mode_train(args) -> int:
    command = [
        "train_model.py",
        "--pairs",
        str(args.pairs),
        "--output",
        str(args.output),
    ]
    if args.scoring:
        command += ["--scoring", args.scoring, "--faults", str(args.faults)]
    return _run(*command)


def mode_evaluate(args) -> int:
    command = [
        "evaluate_localization.py",
        "--pairs",
        str(args.pairs),
        "--faults",
        str(args.faults),
        "--seed",
        str(args.seed),
    ]
    if args.scoring:
        command += ["--scoring", args.scoring]
    return _run(*command)


def mode_pairs(args) -> int:
    return _run(
        "evaluate_model.py",
        "--pairs",
        str(args.pairs),
        "--seed",
        str(args.seed),
    )


def mode_baselines(args) -> int:
    return _run(
        "evaluate_baselines.py",
        "--pairs",
        str(args.pairs),
        "--faults",
        str(args.faults),
        "--seed",
        str(args.seed),
    )


def mode_compare(args) -> int:
    return _run(
        "compare_models.py",
        "--pairs",
        str(args.pairs),
        "--seed",
        str(args.seed),
    )


def mode_localize(args) -> int:
    return _run(
        "localize.py",
        "--model",
        str(args.model),
        "--candidates",
        str(args.candidates),
        "--baseline-tests",
        str(args.baseline_tests),
        *(["--output", str(args.output)] if args.output else []),
    )


def mode_demo(args) -> int:
    return _run("demo.py")


def mode_test(args) -> int:
    return _run("run_tests.py", *args.pytest_args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    def add(name: str, help_text: str, handler, options=None):
        child = sub.add_parser(name, help=help_text, description=help_text)
        child.set_defaults(handler=handler)
        for option in options or ():
            option(child)
        return child

    add("build", "Build the BugsInPy historical corpus.", mode_build, [
        lambda p: p.add_argument("--out", type=Path, help="output directory"),
        lambda p: p.add_argument("--no-clean", action="store_true",
                                 help="exclude the clean base program"),
        lambda p: p.add_argument("--workers", type=int,
                                 help="parallel workers for the mutant executions (0 = auto)"),
    ])
    add("status", "Show corpus and model status.", mode_status)
    add("train", "Fit and save a model from the corpus.", mode_train, [
        lambda p: p.add_argument("--pairs", type=Path, default=DATA / "pairs.csv"),
        lambda p: p.add_argument("--faults", type=Path, default=DATA / "fault_labels.csv"),
        lambda p: p.add_argument("--scoring", choices=["max", "two-stage"],
                                 help="scoring architecture (default: max)"),
        lambda p: p.add_argument("--output", type=Path, default=DEFAULT_MODEL),
    ])
    add("evaluate", "Leave-one-bug-out localization metrics.", mode_evaluate, [
        lambda p: p.add_argument("--pairs", type=Path, default=DATA / "pairs.csv"),
        lambda p: p.add_argument("--faults", type=Path, default=DATA / "fault_labels.csv"),
        lambda p: p.add_argument("--seed", type=int, default=42),
        lambda p: p.add_argument("--scoring", choices=["max", "two-stage"],
                                 help="line scoring rule (default: max)"),
    ])
    add("pairs", "Pair-prediction metrics (AUC/AP/Brier).", mode_pairs, [
        lambda p: p.add_argument("--pairs", type=Path, default=DATA / "pairs.csv"),
        lambda p: p.add_argument("--seed", type=int, default=42),
    ])
    add("baselines", "Compare against SBFL and real-execution MBFL.", mode_baselines, [
        lambda p: p.add_argument("--pairs", type=Path, default=DATA / "pairs.csv"),
        lambda p: p.add_argument("--faults", type=Path, default=DATA / "fault_labels.csv"),
        lambda p: p.add_argument("--seed", type=int, default=42),
    ])
    add("compare", "Compare model configurations on leave-one-bug-out.", mode_compare, [
        lambda p: p.add_argument("--pairs", type=Path, default=DATA / "pairs.csv"),
        lambda p: p.add_argument("--seed", type=int, default=42),
    ])
    localize = add("localize", "Rank lines for a new buggy program with a saved model.",
                   mode_localize, [
                       lambda p: p.add_argument("--model", type=Path, default=DEFAULT_MODEL),
                       lambda p: p.add_argument("--candidates", type=Path, required=True),
                       lambda p: p.add_argument("--baseline-tests", type=Path, required=True),
                       lambda p: p.add_argument("--output", type=Path),
                   ])
    add("demo", "Run the synthetic smoke demo.", mode_demo)
    test = add("test", "Run the project test suite.", mode_test)
    test.add_argument(
        "pytest_args", nargs=argparse.REMAINDER,
        help="arguments passed through to pytest (e.g. -q)",
    )

    args = parser.parse_args()
    pytest_args = getattr(args, "pytest_args", None)
    if pytest_args and pytest_args[0] == "--":
        args.pytest_args = pytest_args[1:]
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())