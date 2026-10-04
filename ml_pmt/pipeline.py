"""Composition of pair prediction and suspicious-line ranking."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .classifier import OutcomeChangeModel
from .localization import rank_suspicious_lines


def predict_and_rank(
    model: OutcomeChangeModel,
    candidate_pairs: pd.DataFrame,
    baseline_test_results: Mapping[str, str],
) -> pd.DataFrame:
    """Predict candidate pair probabilities and return a ranked line table.

    Candidate rows must include the model feature columns plus ``line_number``
    and ``test_id``. The function does not execute mutants.
    """
    required = {"line_number", "test_id"}
    missing = required.difference(candidate_pairs.columns)
    if missing:
        raise ValueError(f"Missing candidate columns: {', '.join(sorted(missing))}")

    predictions = candidate_pairs.loc[:, ["line_number", "test_id"]].copy()
    predictions["probability"] = model.predict_probabilities(candidate_pairs)
    return rank_suspicious_lines(predictions, baseline_test_results)
