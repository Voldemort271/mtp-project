"""Evaluate pair prediction with a source-version-held-out split."""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from _common import PROJECT_ROOT, load_labeled_pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True, help="labeled pair CSV")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from ml_pmt import OutcomeChangeModel, grouped_train_test_split

    pairs = load_labeled_pairs(_resolve(args.pairs))
    train, test = grouped_train_test_split(
        pairs,
        test_size=args.test_size,
        random_state=args.seed,
    )
    model = OutcomeChangeModel(random_state=args.seed).fit(train)
    labels = test["outcome_changed"].astype(int)
    probabilities = model.predict_probabilities(test)

    print(f"Train rows: {len(train)} across {train['source_version'].nunique()} source versions")
    print(f"Held-out rows: {len(test)} across {test['source_version'].nunique()} source versions")
    print(f"Positive-class rate in held-out rows: {labels.mean():.4f}")
    print(f"Brier score: {brier_score_loss(labels, probabilities):.4f}")
    if labels.nunique() == 2:
        print(f"ROC AUC: {roc_auc_score(labels, probabilities):.4f}")
        print(f"Average precision: {average_precision_score(labels, probabilities):.4f}")
    else:
        print("ROC AUC and average precision need both classes in the held-out split.")
    return 0


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
