"""Rank source lines from candidate test–mutant feature rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from _common import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="saved joblib model")
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="CSV with line_number, test_id, and the five model features",
    )
    parser.add_argument(
        "--baseline-tests",
        type=Path,
        required=True,
        help="CSV with test_id and result columns (result is pass or fail)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional output CSV; rankings are printed to stdout by default",
    )
    args = parser.parse_args()

    model_path = _resolve(args.model)
    candidates_path = _resolve(args.candidates)
    baseline_path = _resolve(args.baseline_tests)
    model = joblib.load(model_path)
    candidates = pd.read_csv(candidates_path)
    baseline_frame = pd.read_csv(baseline_path)
    required = {"test_id", "result"}
    missing = required.difference(baseline_frame.columns)
    if missing:
        parser.error(f"baseline CSV is missing columns: {', '.join(sorted(missing))}")
    if baseline_frame["test_id"].duplicated().any():
        parser.error("baseline CSV must contain one row per test_id")

    baseline = dict(zip(baseline_frame["test_id"], baseline_frame["result"]))
    if hasattr(model, "predict_and_rank"):
        ranking = model.predict_and_rank(candidates, baseline)
    else:
        from ml_pmt import predict_and_rank

        ranking = predict_and_rank(model, candidates, baseline)
    if args.output:
        output_path = _resolve(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(output_path, index=False)
        print(f"Saved {len(ranking)} ranked lines to {output_path}")
    else:
        print(ranking.to_string(index=False))
    return 0


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
