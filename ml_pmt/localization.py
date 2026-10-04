"""Ochiai-style ranking from predicted test–mutant outcome changes."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def rank_suspicious_lines(
    pair_predictions: pd.DataFrame,
    baseline_test_results: Mapping[str, str],
) -> pd.DataFrame:
    """Rank source lines using expected per-test outcome-change probabilities.

    ``pair_predictions`` must have ``line_number``, ``test_id``, and
    ``probability`` columns. If multiple mutants target one line, the maximum
    probability per (line, test) is used, as specified for version 1 in the
    project design. Tests absent from a line's prediction rows contribute 0.
    """
    required = {"line_number", "test_id", "probability"}
    missing = required.difference(pair_predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {', '.join(sorted(missing))}")
    if not baseline_test_results:
        raise ValueError("At least one baseline test result is required")

    baseline = {
        str(test_id): str(result).strip().lower()
        for test_id, result in baseline_test_results.items()
    }
    invalid_results = {
        test_id: result
        for test_id, result in baseline.items()
        if result not in {"pass", "fail"}
    }
    if invalid_results:
        raise ValueError(f"Baseline results must be 'pass' or 'fail': {invalid_results}")
    failing_count = sum(result == "fail" for result in baseline.values())
    if failing_count == 0:
        raise ValueError("At least one baseline test must fail to localize a fault")

    predictions = pair_predictions.loc[:, ["line_number", "test_id", "probability"]].copy()
    if predictions.empty:
        return pd.DataFrame(
            columns=["line_number", "f_killed", "p_killed", "suspiciousness"]
        )
    predictions["test_id"] = predictions["test_id"].astype(str)
    unknown_tests = set(predictions["test_id"]).difference(baseline)
    if unknown_tests:
        raise ValueError(f"Predictions reference unknown tests: {sorted(unknown_tests)}")

    line_numbers = pd.to_numeric(predictions["line_number"], errors="coerce")
    if line_numbers.isna().any() or not line_numbers.ge(1).all():
        raise ValueError("line_number must contain positive integers")
    if not np.equal(line_numbers, np.floor(line_numbers)).all():
        raise ValueError("line_number must contain positive integers")
    predictions["line_number"] = line_numbers.astype(int)

    probabilities = pd.to_numeric(predictions["probability"], errors="coerce")
    if probabilities.isna().any() or not np.isfinite(probabilities).all():
        raise ValueError("probability must contain finite numeric values")
    if not probabilities.between(0, 1).all():
        raise ValueError("probability values must be between 0 and 1")
    predictions["probability"] = probabilities

    # Version 1: one test contributes at most once to a source line.
    collapsed = predictions.groupby(["line_number", "test_id"], as_index=False)[
        "probability"
    ].max()
    rows: list[dict[str, float | int]] = []
    for line_number, line_predictions in collapsed.groupby("line_number"):
        probabilities_by_test = line_predictions.set_index("test_id")["probability"]
        failing_mass = sum(
            float(probabilities_by_test.get(test_id, 0.0))
            for test_id, result in baseline.items()
            if result == "fail"
        )
        passing_mass = sum(
            float(probabilities_by_test.get(test_id, 0.0))
            for test_id, result in baseline.items()
            if result == "pass"
        )
        denominator = np.sqrt(failing_count * (failing_mass + passing_mass))
        score = failing_mass / denominator if denominator > 0 else 0.0
        rows.append(
            {
                "line_number": int(line_number),
                "f_killed": failing_mass,
                "p_killed": passing_mass,
                "suspiciousness": float(score),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["suspiciousness", "line_number"],
        ascending=[False, True],
        ignore_index=True,
    )
