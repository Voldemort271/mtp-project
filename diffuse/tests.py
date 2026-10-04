"""Deterministic test oracle for the diffuse modules.

Every test exercises the module's shared core, so coverage of the fault line is
identical between failing and passing tests (diffuse coverage). Tests with a
nonzero bias/penalty parameter fail on the seeded fault; zero-parameter tests
still cover the line but pass.
"""

from __future__ import annotations

from collections.abc import Callable


def _test_weighted_mean_fail1(module):
    value = module.weighted_mean([1, 2], [3, 4], bias=1.0)
    assert abs(value - 11 / 7) < 1e-9, f"expected {11/7}, got {value}"


def _test_weighted_sum_fail(module):
    value = module.weighted_sum([1, 2], [3, 4], bias=1.0)
    assert abs(value - 11.0) < 1e-9, f"expected 11, got {value}"


def _test_weighted_pass1(module):
    assert module.weighted_sum([1, 2], [3, 4]) == 11.0


def _test_weighted_pass2(module):
    assert module.weighted_mean([0, 0], [1, 1]) == 0.0


def _test_weighted_pass3(module):
    assert abs(module.weighted_mean([1, 2], [3, 4]) - 11 / 7) < 1e-9


def _test_smooth_fail(module):
    result = module.smooth([1.0, 2.0], alpha=0.5, anchor=1.0)
    assert result == [0.5, 0.75], f"got {result}"


def _test_alpha_fail(module):
    rate = module.apply_alpha([1.0, 2.0], 0.5, anchor=1.0)
    assert abs(rate - 0.25) < 1e-9, f"got {rate}"


def _test_smooth_pass1(module):
    assert module.smooth([1.0, 2.0], alpha=0.5) == [0.5, 0.75]


def _test_smooth_pass2(module):
    assert module.smooth([0.0, 0.0], alpha=0.5) == [0.0, 0.0]


def _test_scale_rows_fail(module):
    assert module.scale_rows([2, 4], 3, margin=1.0) == [6.0, 12.0]


def _test_scale_total_fail(module):
    assert module.scale_total([2, 4], 3, margin=1.0) == 12.0


def _test_scale_pass1(module):
    assert module.scale_rows([2, 4], 3) == [6.0, 12.0]


def _test_scale_pass2(module):
    assert module.scale_total([0, 0], 5) == 0.0


def _test_average_fail(module):
    assert abs(module.moving_average([1, 2, 3], 2, offset=1.0) - 4.0) < 1e-9


def _test_window_fail(module):
    assert abs(module.window_sum([1, 2, 3], 2, offset=1.0) - 12.0) < 1e-9


def _test_average_pass1(module):
    assert abs(module.moving_average([1, 2, 3], 2) - 4.0) < 1e-9


def _test_average_pass2(module):
    assert module.window_sum([0, 0], 5) == 0.0


def _test_discount_fail(module):
    assert abs(module.discounted_value(100, 0.1, penalty=1.0, periods=2) - 20.0) < 1e-9


def _test_future_fail(module):
    assert abs(module.future_value(100, 0.1, penalty=1.0, periods=2) - 30.0) < 1e-9


def _test_discount_pass1(module):
    assert abs(module.discounted_value(100, 0.1, periods=2) - 20.0) < 1e-9


def _test_discount_pass2(module):
    assert module.future_value(0, 0.1, periods=2) == 0.0


TESTS: list[tuple[str, Callable]] = [
    ("T_wmean_fail1", _test_weighted_mean_fail1),
    ("T_wsum_fail", _test_weighted_sum_fail),
    ("T_wmean_pass1", _test_weighted_pass1),
    ("T_wmean_pass2", _test_weighted_pass2),
    ("T_wmean_pass3", _test_weighted_pass3),
    ("T_smooth_fail", _test_smooth_fail),
    ("T_alpha_fail", _test_alpha_fail),
    ("T_smooth_pass1", _test_smooth_pass1),
    ("T_smooth_pass2", _test_smooth_pass2),
    ("T_scale_rows_fail", _test_scale_rows_fail),
    ("T_scale_total_fail", _test_scale_total_fail),
    ("T_scale_pass1", _test_scale_pass1),
    ("T_scale_pass2", _test_scale_pass2),
    ("T_average_fail", _test_average_fail),
    ("T_window_fail", _test_window_fail),
    ("T_average_pass1", _test_average_pass1),
    ("T_average_pass2", _test_average_pass2),
    ("T_discount_fail", _test_discount_fail),
    ("T_future_fail", _test_future_fail),
    ("T_discount_pass1", _test_discount_pass1),
    ("T_discount_pass2", _test_discount_pass2),
]