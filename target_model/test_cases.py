"""Deterministic test oracle for the Markowitz target subject.

Each test is a ``callable(model) -> None`` that raises on failure. Tests are
executed against whichever module is loaded, so the same cases are used for the
baseline program and every mutant.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

COV = np.array([[0.04, 0.01], [0.01, 0.09]])


def _test_expected_return_1(model):
    value = model.expected_return(np.array([0.5, 0.5]), np.array([0.1, 0.2]))
    assert abs(value - 0.15) < 1e-9, f"expected 0.15, got {value}"


def _test_expected_return_2(model):
    value = model.expected_return(np.array([1.0, 0.0]), np.array([0.3, 0.4]))
    assert abs(value - 0.3) < 1e-9, f"expected 0.3, got {value}"


def _test_portfolio_variance_1(model):
    value = model.portfolio_variance(np.array([0.5, 0.5]), COV)
    assert abs(value - 0.0375) < 1e-9, f"expected 0.0375, got {value}"


def _test_portfolio_variance_2(model):
    value = model.portfolio_variance(np.array([1.0, 0.0]), COV)
    assert abs(value - 0.04) < 1e-9, f"expected 0.04, got {value}"


def _test_normalize_weights_1(model):
    weights = model.normalize_weights(np.array([2.0, 6.0]))
    assert np.allclose(weights, [0.25, 0.75]), f"got {weights}"


def _test_normalize_weights_2(model):
    weights = model.normalize_weights(np.array([1.0, 1.0, 2.0]))
    assert np.allclose(weights, [0.25, 0.25, 0.5]), f"got {weights}"


def _test_markowitz_weights(model):
    weights = model.markowitz_weights(np.array([0.1, 0.2]), COV)
    assert np.allclose(weights, [0.5, 0.5]), f"got {weights}"


def _test_transpose_returns(model):
    array = np.array([[1, 2], [3, 4], [5, 6]])
    result = model.transpose_returns(array)
    assert result.shape == (3, 2), f"got shape {result.shape}"
    assert np.array_equal(result, array)


TESTS: list[tuple[str, Callable]] = [
    ("T_er1", _test_expected_return_1),
    ("T_er2", _test_expected_return_2),
    ("T_pv1", _test_portfolio_variance_1),
    ("T_pv2", _test_portfolio_variance_2),
    ("T_nw1", _test_normalize_weights_1),
    ("T_nw2", _test_normalize_weights_2),
    ("T_mw1", _test_markowitz_weights),
    ("T_tr1", _test_transpose_returns),
]