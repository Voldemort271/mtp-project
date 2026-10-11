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
from ml_pmt.ranker import (
    LINE_FEATURE_COLUMNS,
    TwoStageRanker,
    extend_to_full_universe,
    line_aggregates,
)
from sklearn.ensemble import RandomForestClassifier


def apply_full_universe_floor(
    ranking: pd.DataFrame,
    spectrum: pd.DataFrame,
) -> pd.DataFrame:
    """B1a: full-universe scoring via max(normalize(mutation), normalize(ochiai)).

    ``ranking`` is ``{line_number, suspiciousness}`` over mutant lines (the
    learned/Arar formula score). ``spectrum`` is the SBFL frame with
    ``line_number`` and ``ochiai`` for the full covered universe.

    To combine the two on a common scale, each is min-max normalized over its
    own universe, then a line's final suspiciousness is the max of the two (the
    same fusion as the Hybrid). A non-mutant line therefore keeps its coverage
    score and is always rankable — so the fault line is never absent from the
    ranking just because it has no mutant (B1a).
    """
    if ranking is None or ranking.empty:
        return ranking
    if spectrum is None or spectrum.empty:
        return ranking
    spec = spectrum.loc[:, ["line_number", "ochiai"]].copy()
    if "file" in spectrum.columns and "file" in ranking.columns:
        spec["file"] = spectrum["file"].astype(str)

    def _norm(s: pd.Series) -> pd.Series:
        spread = s.max() - s.min()
        return (s - s.min()) / spread if spread > 0 else s * 0.0

    mut = ranking.loc[:, ["line_number", "suspiciousness"]].copy()
    if "file" in ranking.columns:
        mut["file"] = ranking["file"].astype(str)
    mut["suspiciousness"] = _norm(mut["suspiciousness"].astype(float))
    spec["ochiai"] = _norm(spec["ochiai"].astype(float))

    merged = spec.rename(columns={"ochiai": "ochiai_n"})
    if "file" in spec.columns and "file" in mut.columns:
        merged = merged.merge(mut, on=["file", "line_number"], how="outer")
    else:
        merged = merged.merge(mut, on="line_number", how="outer")
    merged["suspiciousness"] = merged[
        ["ochiai_n", "suspiciousness"]
    ].max(axis=1)
    out_cols = (
        ["file", "line_number", "suspiciousness"]
        if "file" in merged.columns
        else ["line_number", "suspiciousness"]
    )
    return merged.loc[:, out_cols].sort_values(
        ["suspiciousness", "line_number"],
        ascending=[False, True],
        ignore_index=True,
    )


def rank_metrics(
    ranking: pd.DataFrame,
    faults: list[tuple],
) -> dict:
    """Compute ranking metrics for known faults.

    ``faults`` is a list of ``(file, line)`` (or ``(line,)`` for legacy
    line-only callers). If the ranking carries a ``file`` column, faults are
    matched by ``(file, line)``; otherwise by line number only. Best rank wins.
    """
    if "file" in ranking.columns:
        entries = list(
            zip(ranking["file"].astype(str),
                ranking["line_number"].astype(int))
        )
        fault_pairs = []
        if all(isinstance(f, (tuple, list)) and len(f) == 2 for f in faults):
            fault_pairs = [(str(f[0]), int(f[1])) for f in faults]
        else:
            # legacy line-only faults -> match against the first file
            first_file = str(ranking["file"].iloc[0])
            fault_pairs = [(first_file, int(f)) for f in faults]
        ranks = [entries.index(fp) + 1 for fp in fault_pairs if fp in entries]
    else:
        lines = ranking["line_number"].tolist()
        fault_lines = [f[1] if isinstance(f, (tuple, list)) else f for f in faults]
        ranks = [lines.index(line) + 1 for line in fault_lines if line in lines]
    if ranks:
        rank = min(ranks)
    else:
        rank = len(ranking) + 1
    exam = rank / max(len(ranking), 1)
    return {
        "fault_lines": [list(f) if not isinstance(f, int) else f for f in faults],
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
    *,
    coverage_cache: dict[str, dict[int, dict]] | None = None,
) -> pd.DataFrame:
    """Two-stage ranking with honest leave-one-bug-out line-rank training.

    Each training bug's line features come from a pair model that has NOT seen
    that bug (nested), so the ranker never trains on in-sample pair predictions.

    When ``coverage_cache`` is provided (maps source_version -> {line: {a,b,
    ochiai,tarantula,dstar}} for the full covered universe), both the training
    rows and the held-out lines are extended to the **full covered universe**:
    non-mutant lines get zero mutation signal plus their SBFL coverage features
    and the ``is_fault`` label, so the learned ranker is trained on them rather
    than only flooring them at inference (B1a, learned variant).
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

    def _build_features(source_version: str, model: OutcomeChangeModel) -> pd.DataFrame:
        sp = pairs[pairs["source_version"] == source_version]
        feats = line_aggregates(sp, model.predict_probabilities(sp))
        if coverage_cache:
            feats = extend_to_full_universe(
                feats, coverage_cache.get(source_version, {})
            )
        return feats

    holdout_model = train_without(bug_key)
    held_features = _build_features(variant_id, holdout_model)

    training_rows = []
    for other in pairs["source_version"].unique():
        other_key = str(other).removesuffix("-fixed")
        if other_key == bug_key:
            continue
        model = train_without(bug_key, other_key)
        features = _build_features(str(other), model)
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
    out_cols = (
        ["file", "line_number", "suspiciousness"]
        if "file" in held_features.columns
        else ["line_number", "suspiciousness"]
    )
    return held_features.loc[:, out_cols].sort_values(
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