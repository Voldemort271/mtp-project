"""Feature and label helpers for test–mutant training rows."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

FEATURE_COLUMNS = (
    "mutation_type",
    "ast_depth",
    "parent_node_type",
    "test_covers_mutation",
    "baseline_test_result",
    "mutated_operator",
    "replacement_operator",
)
LABEL_COLUMN = "outcome_changed"
TEST_RESULTS = frozenset({"pass", "fail"})


def add_outcome_change_labels(pairs: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with labels set when a test's pass/fail result flips.

    Input rows must contain ``baseline_test_result`` and
    ``mutant_test_result`` columns, each containing only ``pass`` or ``fail``.
    Infrastructure errors and skipped/undetermined results should be filtered
    before calling this helper.
    """
    required = {"baseline_test_result", "mutant_test_result"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"Missing result columns: {', '.join(sorted(missing))}")

    result = pairs.copy()
    baseline = _normalize_results(result["baseline_test_result"], "baseline_test_result")
    mutant = _normalize_results(result["mutant_test_result"], "mutant_test_result")
    result["baseline_test_result"] = baseline
    result["mutant_test_result"] = mutant
    result[LABEL_COLUMN] = baseline.ne(mutant).astype("int8")
    return result


def grouped_train_test_split(
    pairs: pd.DataFrame,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    group_column: str = "source_version",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by source version so related mutants cannot cross the split."""
    if group_column not in pairs:
        raise ValueError(f"Missing grouping column: {group_column}")
    if pairs[group_column].isna().any():
        raise ValueError(f"Grouping column {group_column!r} contains missing values")
    if pairs[group_column].nunique() < 2:
        raise ValueError("At least two distinct source groups are required")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_indices, test_indices = next(
        splitter.split(pairs, groups=pairs[group_column])
    )
    return pairs.iloc[train_indices].copy(), pairs.iloc[test_indices].copy()


def _normalize_results(results: pd.Series, column: str) -> pd.Series:
    normalized = results.astype("string").str.strip().str.lower()
    invalid = normalized.isna() | ~normalized.isin(TEST_RESULTS)
    if invalid.any():
        bad_values = sorted({str(value) for value in results[invalid].tolist()})
        raise ValueError(
            f"{column} must contain only 'pass' or 'fail'; got {bad_values}"
        )
    return normalized.astype(str)
