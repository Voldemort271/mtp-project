"""Random-forest model for predicting test–mutant outcome changes."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from .features import FEATURE_COLUMNS, LABEL_COLUMN

CATEGORICAL_FEATURES = (
    "mutation_type",
    "parent_node_type",
    "baseline_test_result",
    "mutated_operator",
    "replacement_operator",
)
NUMERIC_FEATURES = ("ast_depth", "test_covers_mutation")


class OutcomeChangeModel:
    """Estimate P(test pass/fail changes | mutant, test, source features).

    Training data must contain one row per source-version/mutant/test tuple.
    Use :func:`ml_pmt.grouped_train_test_split` when evaluating generalization
    so mutants from one source version do not appear in both data partitions.

    ``categorical_encoding`` is ``"onehot"`` or ``"ordinal"``. Ordinal encoding
    is usually better here: with one-hot, an unseen ``(mutated_operator,
    replacement_operator)`` pair produces an all-zero vector in leave-one-bug-out,
    whereas ordinal lets the trees split on each operator independently.
    ``estimator`` is ``"rf"`` (RandomForest) or ``"gbt"`` (HistGradientBoosting
    with inverse-class-frequency sample weights).
    """

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        random_state: int = 42,
        max_depth: int | None = None,
        categorical_encoding: str = "ordinal",
        estimator: str = "gbt",
    ) -> None:
        if categorical_encoding == "onehot":
            categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        elif categorical_encoding == "ordinal":
            categorical = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
        else:
            raise ValueError(f"Unknown categorical_encoding {categorical_encoding!r}")
        features = ColumnTransformer(
            transformers=[
                ("categorical", categorical, list(CATEGORICAL_FEATURES)),
                ("numeric", "passthrough", list(NUMERIC_FEATURES)),
            ],
            remainder="drop",
        )
        if estimator == "rf":
            classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced_subsample",
                random_state=random_state,
            )
        elif estimator == "gbt":
            classifier = HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.08,
                max_depth=3,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown estimator {estimator!r}")
        self._estimator = estimator
        self._categorical_encoding = categorical_encoding
        self._pipeline = Pipeline(
            [("features", features), ("classifier", classifier)]
        )
        self._is_fitted = False

    def fit(self, pairs: pd.DataFrame) -> Self:
        """Fit from labeled test–mutant rows; requires both label classes."""
        missing = set(FEATURE_COLUMNS + (LABEL_COLUMN,)).difference(pairs.columns)
        if missing:
            raise ValueError(f"Missing training columns: {', '.join(sorted(missing))}")

        labels = pd.to_numeric(pairs[LABEL_COLUMN], errors="coerce")
        if labels.isna().any() or not labels.isin((0, 1)).all():
            raise ValueError(f"{LABEL_COLUMN} must contain only binary 0/1 labels")
        if labels.nunique() < 2:
            raise ValueError("Training requires both outcome_changed label classes")
        labels = labels.astype(int)

        if self._estimator == "gbt":
            weights = self._sample_weights(labels)
            self._pipeline.fit(
                self._prepare_features(pairs),
                labels,
                classifier__sample_weight=weights,
            )
        else:
            self._pipeline.fit(self._prepare_features(pairs), labels)
        self._is_fitted = True
        return self

    def predict_probabilities(self, pairs: pd.DataFrame) -> np.ndarray:
        """Return one outcome-change probability for each input pair row."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_probabilities()")
        missing = set(FEATURE_COLUMNS).difference(pairs.columns)
        if missing:
            raise ValueError(f"Missing prediction columns: {', '.join(sorted(missing))}")

        probabilities = self._pipeline.predict_proba(self._prepare_features(pairs))
        classes = self._pipeline.named_steps["classifier"].classes_
        positive_class = np.flatnonzero(classes == 1)
        if positive_class.size == 0:
            return np.zeros(len(pairs), dtype=float)
        return probabilities[:, positive_class[0]]

    @staticmethod
    def _sample_weights(labels: np.ndarray) -> np.ndarray:
        """Inverse-class-frequency weights for gradient boosting."""
        counts = np.bincount(labels)
        total = labels.size
        return np.array([total / (2 * counts[label]) for label in labels])

    @staticmethod
    def _prepare_features(pairs: pd.DataFrame) -> pd.DataFrame:
        features = pairs.loc[:, FEATURE_COLUMNS].copy()
        for column in CATEGORICAL_FEATURES:
            features[column] = (
                features[column]
                .astype("string")
                .fillna("__missing__")
                .astype(str)
            )

        for column in NUMERIC_FEATURES:
            features[column] = pd.to_numeric(features[column], errors="coerce")
            if features[column].isna().any():
                raise ValueError(f"Feature {column!r} must contain numeric values")
            if not np.isfinite(features[column].to_numpy(dtype=float)).all():
                raise ValueError(f"Feature {column!r} must contain finite values")

        if not features["test_covers_mutation"].isin((0, 1, False, True)).all():
            raise ValueError("test_covers_mutation must contain only 0/1 values")
        if not features["ast_depth"].ge(0).all():
            raise ValueError("ast_depth must be non-negative")
        if not features["baseline_test_result"].isin(("pass", "fail")).all():
            raise ValueError("baseline_test_result must contain 'pass' or 'fail'")
        return features
