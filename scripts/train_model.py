"""Train and save an outcome-change model from pair-level CSV data.

``--scoring max`` saves the single-stage pair model; ``--scoring two-stage``
additionally trains the learned line ranker, so localization uses the
two-stage architecture.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

from _common import PROJECT_ROOT, load_labeled_pairs

ROOT = PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help="training CSV with model features and labels or mutant test results",
    )
    parser.add_argument(
        "--faults",
        type=Path,
        default=ROOT / "data" / "fault_labels.csv",
        help="fault-label CSV (required for --scoring two-stage)",
    )
    parser.add_argument(
        "--scoring",
        choices=["max", "two-stage"],
        default="max",
        help="line scoring architecture to train (default: max)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "outcome_change_model.joblib",
        help="where to save the fitted model",
    )
    args = parser.parse_args()

    pairs_path = _resolve(args.pairs)
    output_path = _resolve(args.output)
    pairs = load_labeled_pairs(pairs_path)
    if args.scoring == "two-stage":
        from ml_pmt.ranker import TwoStageRanker

        faults_path = _resolve(args.faults)
        faults = _read_csv(faults_path)
        model = TwoStageRanker().fit(pairs, faults)
        label = "two-stage (pair model + learned line ranker)"
    else:
        from ml_pmt import OutcomeChangeModel

        model = OutcomeChangeModel().fit(pairs)
        label = "single-stage pair model"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    positives = int(pairs["outcome_changed"].astype(int).sum())
    print(f"Trained {label} on {len(pairs)} test–mutant pairs ({positives} outcome changes).")
    print(f"Saved model: {output_path}")
    print("Use scripts/evaluate_model.py to measure held-out source-version performance.")
    return 0


def _read_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
