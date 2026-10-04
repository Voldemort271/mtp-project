"""Deterministic NumPy Markowitz mean-variance optimization subject."""

import numpy as np


def expected_return(weights, returns):
    return float(np.sum(weights * returns))


def portfolio_variance(weights, cov_matrix):
    return float(weights @ cov_matrix @ weights)


def normalize_weights(weights):
    total = np.sum(weights)
    return weights / total


def markowitz_weights(expected_returns, cov_matrix):
    inverse = np.linalg.inv(cov_matrix)
    raw = inverse @ expected_returns
    return normalize_weights(raw)


def transpose_returns(returns_2d):
    n_rows = returns_2d.shape[0]
    return returns_2d.reshape(n_rows, -1)