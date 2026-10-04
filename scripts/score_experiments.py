"""Experiments on line-scoring/aggregation for the predicted model.

Evaluates leave-one-bug-out ranking under different ways of turning per-test
pair probabilities into a line score:
  max       - current: per (line,test) take the max mutant probability
  mean      - per (line,test) take the mean mutant probability
  failonly  - score depends only on failing-test flips (drop passing term)
  weighted  - passing-test flips enter the denominator with weight < 1
  two-stage - a learned line-level ranker consumes the aggregate features
              instead of the hand-fixed Metallaxis formula
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_pmt import OutcomeChangeModel
from ml_pmt.ranker import line_aggregates
from scripts.evaluate_localization import rank_metrics


def fold_pair_model(pairs: pd.DataFrame, held_out: str, random_state: int):
    """Fit the pair model on all bugs except the held-out one."""
    bug_key = held_out.removesuffix("-fixed")
    train = pairs[~pairs["source_version"].str.removesuffix("-fixed").eq(bug_key)]
    return OutcomeChangeModel(random_state=random_state).fit(train)


def score_variant(aggregates: pd.DataFrame, name: str, *, alpha: float | None = None) -> pd.DataFrame:
    """Turn aggregate features into a ranked line table for a scoring rule."""
    f = aggregates["f_killed"]
    p = aggregates["p_killed"]
    F = aggregates["n_failing"].replace(0, 1)  # total failing tests per line proxy
    if name == "failonly":
        score = f / np.sqrt(F * f)
    elif name == "weighted":
        weight = alpha if alpha is not None else 0.25
        score = f / np.sqrt(F * (f + weight * p))
    else:  # max / mean
        score = f / np.sqrt(F * (f + p))
    return pd.DataFrame(
        {"line_number": aggregates["line_number"], "suspiciousness": score}
    ).sort_values(
        ["suspiciousness", "line_number"], ascending=[False, True], ignore_index=True
    )


def evaluate_aggregation_variants(
    pairs: pd.DataFrame,
    faults: pd.DataFrame,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    records = []
    for variant_id in map(str, faults["source_version"].unique()):
        fault_lines = sorted(
            int(x) for x in faults[faults["source_version"] == variant_id]["fault_line"]
        )
        candidates = pairs[pairs["source_version"] == variant_id]
        model = fold_pair_model(pairs, variant_id, random_state)
        probs = model.predict_probabilities(candidates)

        aggregates_max = line_aggregates(candidates, probs, collapse="max")
        aggregates_mean = line_aggregates(candidates, probs, collapse="mean")
        rules = (
            ("agg-max", aggregates_max, "max"),
            ("agg-mean", aggregates_mean, "mean"),
            ("failonly", aggregates_max, "failonly"),
            ("weighted", aggregates_max, "weighted"),
        )
        for variant_name, aggregates, rule in rules:
            ranking = score_variant(aggregates, rule)
            metrics = rank_metrics(ranking, fault_lines)
            metrics["variant"] = variant_name
            metrics["source_version"] = variant_id
            records.append(metrics)
    return pd.DataFrame(records)


def two_stage_features(
    pairs: pd.DataFrame,
    variant_id: str,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    candidates = pairs[pairs["source_version"] == variant_id]
    aggregates = line_aggregates(candidates, probabilities, collapse="max")
    faults = pd.read_csv(ROOT / "data" / "fault_labels.csv")
    fault_lines = set(
        faults[faults["source_version"] == variant_id]["fault_line"].astype(int)
    )
    aggregates["is_fault"] = aggregates["line_number"].isin(fault_lines).astype(int)
    return aggregates


def evaluate_two_stage(
    pairs: pd.DataFrame,
    faults: pd.DataFrame,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    feature_columns = [
        "f_killed",
        "p_killed",
        "n_mutants",
        "n_failing_cover",
        "n_passing_cover",
        "max_failing_prob",
        "mean_failing_prob",
        "frac_failing_high",
        "n_failing",
        "n_passing",
    ]
    records = []
    variants = [v for v in faults["source_version"].unique()]
    for variant_id in map(str, variants):
        fault_lines = sorted(
            int(x) for x in faults[faults["source_version"] == variant_id]["fault_line"]
        )
        bug_key = variant_id.removesuffix("-fixed")

        # Held-out features: pair model trained without this bug.
        model_holdout = fold_pair_model(pairs, variant_id, random_state)
        held = pairs[pairs["source_version"] == variant_id]
        held_features = two_stage_features(
            pairs, variant_id, model_holdout.predict_probabilities(held)
        )

        # Training features for the line ranker: for each other bug Y, use a
        # pair model that has not seen Y either.
        training_rows = []
        for other in map(str, variants):
            if other == variant_id:
                continue
            other_key = other.removesuffix("-fixed")
            pair_train = pairs[
                ~pairs["source_version"].str.removesuffix("-fixed").isin(
                    {bug_key, other_key}
                )
            ]
            model = OutcomeChangeModel(random_state=random_state).fit(pair_train)
            other_pairs = pairs[pairs["source_version"] == other]
            other_features = two_stage_features(
                pairs, other, model.predict_probabilities(other_pairs)
            )
            training_rows.append(other_features)

        training = pd.concat(training_rows, ignore_index=True)
        ranker = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=random_state
        )
        ranker.fit(training[feature_columns], training["is_fault"])
        held_features["score"] = ranker.predict_proba(held_features[feature_columns])[:, 1]
        ranking = held_features.loc[:, ["line_number", "score"]].rename(
            columns={"score": "suspiciousness"}
        ).sort_values(
            ["suspiciousness", "line_number"], ascending=[False, True], ignore_index=True
        )
        metrics = rank_metrics(ranking, fault_lines)
        metrics["variant"] = "two-stage"
        metrics["source_version"] = variant_id
        records.append(metrics)
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=ROOT / "data" / "pairs.csv")
    parser.add_argument("--faults", type=Path, default=ROOT / "data" / "fault_labels.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs)
    faults = pd.read_csv(args.faults)

    aggregation = evaluate_aggregation_variants(
        pairs, faults, random_state=args.seed
    )
    two_stage = evaluate_two_stage(pairs, faults, random_state=args.seed)
    all_results = pd.concat([aggregation, two_stage], ignore_index=True)

    summary = all_results.groupby("variant").agg(
        top1=("top1", "mean"),
        top5=("top5", "mean"),
        mrr=("mrr", "mean"),
        exam=("exam", "mean"),
    ).round(3)
    print("=== Scoring/aggregation experiments (leave-one-bug-out) ===")
    print(summary.sort_values("mrr", ascending=False).to_string())
    print()
    print("=== Per-bug rank (two-stage vs current max) ===")
    pivot = all_results.pivot_table(
        index="source_version",
        columns="variant",
        values="rank",
        aggfunc="first",
    )
    for col in ("agg-max", "two-stage"):
        if col in pivot:
            print(f"\n{col}:")
            print(pivot[col].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())