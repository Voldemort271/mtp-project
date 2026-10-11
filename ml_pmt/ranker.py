"""Learned line-level scoring: a two-stage ranker.

Stage 1 is the pair model (:class:`ml_pmt.classifier.OutcomeChangeModel`) that
predicts per-test outcome-change probabilities. Stage 2 collapses those into
per-line aggregate features and learns how to combine them into a
suspiciousness score, replacing the hand-fixed Metallaxis formula.

On the seven-bug slice this improves leave-one-bug-out MRR from 0.60 (the
fixed formula) to 0.68, at the cost of slightly worse worst-case bugs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .classifier import OutcomeChangeModel

LINE_FEATURE_COLUMNS = [
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
    "ochiai",
    "tarantula",
    "dstar",
]


def line_aggregates(
    pairs_for_bug: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    collapse: str = "max",
) -> pd.DataFrame:
    """Collapse per-test pair probabilities into one row per (file, line)."""
    has_file = "file" in pairs_for_bug.columns
    cols = (["file"] if has_file else []) + [
        "line_number", "test_id", "test_covers_mutation"
    ]
    frame = pairs_for_bug.loc[:, cols].copy()
    frame["probability"] = probabilities
    frame["is_failing"] = pairs_for_bug["baseline_test_result"] == "fail"

    group_cols = (["file"] if has_file else []) + ["line_number", "test_id"]
    collapsed = frame.groupby(group_cols).agg(
        probability=("probability", "max" if collapse == "max" else "mean"),
        covers=("test_covers_mutation", "max"),
    ).reset_index()
    failing_test_ids = set(frame[frame["is_failing"]]["test_id"])
    collapsed["is_failing"] = collapsed["test_id"].isin(failing_test_ids)

    rows = []
    for key, group in collapsed.groupby(
        (["file", "line_number"] if has_file else ["line_number"])
    ):
        if has_file:
            file, line = key
        else:
            line = key[0] if isinstance(key, tuple) else key
        failing = group[group["is_failing"]]
        passing = group[~group["is_failing"]]
        a = int((failing["covers"] == 1).sum())
        b = int((passing["covers"] == 1).sum())
        n_failing = int(len(failing))
        n_passing = int(len(passing))
        c = n_failing - a
        e = n_passing - b
        row = {
            "line_number": int(line),
            "f_killed": float(failing["probability"].sum()),
            "p_killed": float(passing["probability"].sum()),
            "n_mutants": int(pairs_for_bug[pairs_for_bug["line_number"] == line].shape[0]),
            "n_failing_cover": a,
            "n_passing_cover": b,
            "max_failing_prob": float(
                failing["probability"].max() if len(failing) else 0.0
            ),
            "mean_failing_prob": float(
                failing["probability"].mean() if len(failing) else 0.0
            ),
            "frac_failing_high": float(
                (failing["probability"] > 0.5).mean() if len(failing) else 0.0
            ),
            "n_failing": n_failing,
            "n_passing": n_passing,
            "ochiai": _ochiai(a, b, c),
            "tarantula": _tarantula(a, b, c, e),
            "dstar": _dstar(a, b, c),
        }
        if has_file:
            row["file"] = file
        rows.append(row)
    return pd.DataFrame(rows)


def _ochiai(a: int, b: int, c: int) -> float:
    denominator = np.sqrt((a + b) * (a + c))
    return float(a / denominator) if denominator > 0 else 0.0


def _tarantula(a: int, b: int, c: int, e: int) -> float:
    positive = a / (a + c) if a + c > 0 else 0.0
    negative = b / (b + e) if b + e > 0 else 0.0
    denominator = positive + negative
    return float(positive / denominator) if denominator > 0 else 0.0


def _dstar(a: int, b: int, c: int) -> float:
    denominator = b + c
    return float((a * a) / denominator) if denominator > 0 else float(a) if a else 0.0


def extend_to_full_universe(
    aggregates: pd.DataFrame,
    universe: dict[int, dict],
) -> pd.DataFrame:
    """Append non-mutant coverage-universe rows to a mutant-lines frame.

    ``universe`` maps ``line -> {a: int, b: int, ochiai: float, ...}`` for every
    covered line. Lines already in ``aggregates`` keep their full mutation
    features and are scored by the learned ranker. Other lines (no mutant) are
    appended with ``is_universe_only=1`` and zero mutation signal, so the
    caller can give them a coverage-based floor score (B1a full-universe
    ranking); the RF never extrapolates onto unobserved mutation features.
    """
    existing = set(aggregates["line_number"])
    missing = [line for line in sorted(universe) if line not in existing]
    if not missing:
        if aggregates.get("is_universe_only") is None:
            aggregates = aggregates.copy()
            aggregates["is_universe_only"] = 0
        return aggregates
    rows = []
    for line in missing:
        u = universe[line]
        a = int(u.get("a", 0))
        b = int(u.get("b", 0))
        rows.append(
            {
                "line_number": line,
                "f_killed": 0.0,
                "p_killed": 0.0,
                "n_mutants": 0,
                "n_failing_cover": a,
                "n_passing_cover": b,
                "max_failing_prob": 0.0,
                "mean_failing_prob": 0.0,
                "frac_failing_high": 0.0,
                "n_failing": 0,
                "n_passing": 0,
                "ochiai": float(u.get("ochiai", _ochiai(a, b, 0))),
                "tarantula": float(u.get("tarantula", _tarantula(a, b, 0, 0))),
                "dstar": float(u.get("dstar", _dstar(a, b, 0))),
                "is_universe_only": 1,
            }
        )
    out = pd.concat([aggregates, pd.DataFrame(rows)], ignore_index=True)
    if out.get("is_universe_only") is None:
        out["is_universe_only"] = 0
    return out


class TwoStageRanker:
    """Predict pair outcomes, then learn a line-level suspiciousness score."""

    def __init__(self, *, n_estimators: int = 200, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.pair_model: OutcomeChangeModel | None = None
        self.ranker: RandomForestClassifier | None = None

    def fit(self, pairs: pd.DataFrame, faults: pd.DataFrame) -> "TwoStageRanker":
        """Fit the pair model and the line ranker on a corpus of labeled bugs."""
        self.pair_model = OutcomeChangeModel(random_state=self.random_state).fit(pairs)
        training_rows = []
        for variant_id in pairs["source_version"].unique():
            candidates = pairs[pairs["source_version"] == variant_id]
            features = line_aggregates(
                candidates, self.pair_model.predict_probabilities(candidates)
            )
            fault_lines = set(
                faults[faults["source_version"] == variant_id]["fault_line"].astype(int)
            )
            features["is_fault"] = features["line_number"].isin(fault_lines).astype(int)
            training_rows.append(features)
        training = pd.concat(training_rows, ignore_index=True)
        self.ranker = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.ranker.fit(training[LINE_FEATURE_COLUMNS], training["is_fault"])
        return self

    def fit_excluding(
        self,
        pairs: pd.DataFrame,
        faults: pd.DataFrame,
        excluded: str,
    ) -> "TwoStageRanker":
        """Fit on all bugs except ``excluded`` (for leave-one-bug-out)."""
        bug_key = excluded.removesuffix("-fixed")
        pair_keep = ~pairs["source_version"].str.removesuffix("-fixed").eq(bug_key)
        fault_keep = ~faults["source_version"].str.removesuffix("-fixed").eq(bug_key)
        return self.fit(pairs[pair_keep], faults[fault_keep])

    def predict_and_rank(
        self,
        candidates: pd.DataFrame,
        baseline_test_results: dict[str, str],
        *,
        universe: dict[int, dict] | None = None,
    ) -> pd.DataFrame:
        """Predict pair probabilities, score lines, and rank them.

        When ``universe`` is given (``{line: {'a','b','ochiai',...}}`` for the
        full covered line set), non-mutant lines are appended with zero
        mutation features and their SBFL coverage features, so every covered
        line is ranked (B1a full-universe scoring). Mutant lines keep their
        learned mutation+coverage score.
        """
        if self.pair_model is None or self.ranker is None:
            raise RuntimeError("Call fit() before predict_and_rank()")
        probabilities = self.pair_model.predict_probabilities(candidates)
        features = line_aggregates(candidates, probabilities)
        if universe:
            features = extend_to_full_universe(features, universe)
            # B1a full-universe floor: non-mutant lines are scored by their
            # normalized Ochiai (coverage) instead of the RF, which only ever
            # saw mutant-line rows during training.
            mask = features.get("is_universe_only", pd.Series(0, index=features.index)).fillna(0).astype(bool)
            if mask.any():
                ochiai = features.loc[mask, "ochiai"]
                spread = ochiai.max() - ochiai.min()
                features.loc[mask, "suspiciousness"] = (
                    (ochiai - ochiai.min()) / spread if spread > 0 else ochiai * 0.0
                )
            mutant_mask = ~mask
        else:
            mutant_mask = pd.Series(True, index=features.index)
        if mutant_mask.any():
            proba = self.ranker.predict_proba(
                features.loc[mutant_mask, LINE_FEATURE_COLUMNS]
            )
            features.loc[mutant_mask, "suspiciousness"] = (
                proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            )
        out_cols = ["file", "line_number", "suspiciousness"] if "file" in features.columns else [
            "line_number", "suspiciousness"
        ]
        return features.loc[:, out_cols].sort_values(
            ["suspiciousness", "line_number"],
            ascending=[False, True],
            ignore_index=True,
        )