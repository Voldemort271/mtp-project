"""Evaluate fault-localization ranking on the bundled synthetic faults."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_pmt import OutcomeChangeModel, predict_and_rank
from ml_pmt.ranker import LINE_FEATURE_COLUMNS, TwoStageRanker, line_aggregates
from sklearn.ensemble import RandomForestClassifier


def rank_metrics(ranking: pd.DataFrame, fault_lines: list[int]) -> dict:
    """Compute ranking metrics for known fault lines (best rank wins)."""
    lines = ranking["line_number"].tolist()
    ranks = [lines.index(line) + 1 for line in fault_lines if line in lines]
    if ranks:
        rank = min(ranks)
    else:
        rank = len(lines) + 1
    exam = rank / max(len(lines), 1)
    return {
        "fault_lines": fault_lines,
        "rank": rank,
        "top1": rank == 1,
        "top5": rank <= 5,
        "top10": rank <= 10,
        "mrr": 1.0 / rank,
        "exam": exam,
        "exam_at_10": exam <= 0.1,
        "exam_at_20": exam <= 0.2,
    }


def evaluate_variant(
    pairs: pd.DataFrame,
    faults: pd.DataFrame,
    variant_id: str,
    fault_lines: list[int],
    *,
    random_state: int = 42,
    scoring: str = "max",
) -> dict:
    """Train on all other bugs, rank the held-out variant's candidates.

    The variant's own buggy and fixed versions are both excluded from training
    so the same code base cannot leak across the split. ``scoring`` is ``max``
    (fixed Metallaxis-style formula) or ``two-stage`` (learned line ranker).
    """
    bug_key = variant_id.removesuffix("-fixed")
    train = pairs[
        ~pairs["source_version"].str.removesuffix("-fixed").eq(bug_key)
    ]
    leaked = train["source_version"].str.removesuffix("-fixed").eq(bug_key).any()
    if leaked:
        raise RuntimeError(f"Leakage: held-out bug {bug_key} present in training data")
    candidates = pairs[pairs["source_version"] == variant_id]
    baseline = dict(
        zip(candidates["test_id"], candidates["baseline_test_result"], strict=True)
    )
    if scoring == "two-stage":
        ranking = honest_two_stage_ranking(pairs, faults, variant_id, random_state)
    else:
        model = OutcomeChangeModel(random_state=random_state).fit(train)
        ranking = predict_and_rank(model, candidates, baseline)
    metrics = rank_metrics(ranking, fault_lines)
    metrics["source_version"] = variant_id
    return metrics


def honest_two_stage_ranking(
    pairs: pd.DataFrame,
    faults: pd.DataFrame,
    variant_id: str,
    random_state: int,
) -> pd.DataFrame:
    """Two-stage ranking with honest leave-one-bug-out line-rank training.

    Each training bug's line features come from a pair model that has NOT seen
    that bug (nested), so the ranker never trains on in-sample pair predictions.
    """
    bug_key = variant_id.removesuffix("-fixed")

    def train_without(*excluded: str) -> OutcomeChangeModel:
        subset = pairs[
            ~pairs["source_version"].str.removesuffix("-fixed").isin(excluded)
        ]
        leaked = subset["source_version"].str.removesuffix("-fixed").isin(excluded).any()
        if leaked:
            raise RuntimeError(
                f"Leakage: excluded bugs {set(excluded)} present in training data"
            )
        return OutcomeChangeModel(random_state=random_state).fit(subset)

    holdout_model = train_without(bug_key)
    held = pairs[pairs["source_version"] == variant_id]
    held_features = line_aggregates(held, holdout_model.predict_probabilities(held))

    training_rows = []
    for other in pairs["source_version"].unique():
        other_key = str(other).removesuffix("-fixed")
        if other_key == bug_key:
            continue
        other_pairs = pairs[pairs["source_version"] == other]
        model = train_without(bug_key, other_key)
        features = line_aggregates(other_pairs, model.predict_probabilities(other_pairs))
        fault_lines = set(
            faults[faults["source_version"] == other]["fault_line"].astype(int)
        )
        features["is_fault"] = features["line_number"].isin(fault_lines).astype(int)
        training_rows.append(features)
    training = pd.concat(training_rows, ignore_index=True)

    ranker = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=random_state
    )
    ranker.fit(training[LINE_FEATURE_COLUMNS], training["is_fault"])
    proba = ranker.predict_proba(held_features[LINE_FEATURE_COLUMNS])
    held_features["suspiciousness"] = (
        proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    )
    return held_features.loc[:, ["line_number", "suspiciousness"]].sort_values(
        ["suspiciousness", "line_number"], ascending=[False, True], ignore_index=True
    )


def evaluate_corpus(
    pairs_path: Path,
    faults_path: Path,
    *,
    random_state: int = 42,
    scoring: str = "max",
) -> pd.DataFrame:
    pairs = pd.read_csv(pairs_path)
    faults = pd.read_csv(faults_path)
    records = []
    for variant_id, group in faults.groupby("source_version"):
        records.append(
            evaluate_variant(
                pairs,
                faults,
                str(variant_id),
                sorted(int(line) for line in group["fault_line"]),
                random_state=random_state,
                scoring=scoring,
            )
        )
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=ROOT / "data" / "pairs.csv",
        help="training pair CSV produced by build_corpus.py",
    )
    parser.add_argument(
        "--faults",
        type=Path,
        default=ROOT / "data" / "fault_labels.csv",
        help="fault-label CSV produced by build_corpus.py",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scoring",
        choices=["max", "two-stage"],
        default="max",
        help="line scoring rule: fixed Metallaxis formula or learned two-stage ranker",
    )
    args = parser.parse_args()

    results = evaluate_corpus(
        args.pairs, args.faults, random_state=args.seed, scoring=args.scoring
    )
    print(results.to_string(index=False))
    print(f"\nTop-1: {results['top1'].mean():.2f}")
    print(f"Top-5: {results['top5'].mean():.2f}")
    print(f"Top-10: {results['top10'].mean():.2f}")
    print(f"Mean MRR: {results['mrr'].mean():.3f}")
    print(f"Mean rank: {results['rank'].mean():.2f}")
    print(f"Mean EXAM: {results['exam'].mean():.3f}")
    print(f"EXAM@10%: {results['exam_at_10'].mean():.2f}")
    print(f"EXAM@20%: {results['exam_at_20'].mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())