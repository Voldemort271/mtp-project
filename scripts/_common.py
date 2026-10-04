"""Utilities shared by the command-line scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_labeled_pairs(path: str | Path) -> pd.DataFrame:
    """Load pair rows, deriving outcome labels when raw test results exist."""
    from ml_pmt import add_outcome_change_labels

    pairs = pd.read_csv(path)
    if "outcome_changed" not in pairs:
        pairs = add_outcome_change_labels(pairs)
    return pairs
