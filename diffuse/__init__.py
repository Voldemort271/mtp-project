"""Diffuse-coverage benchmark modules.

Each module has a core helper whose lines are executed by every test (failing
and passing alike), so coverage cannot discriminate the fault line. A seeded
operator fault sits on one of those shared lines. These are engineered faults
on realistic pure-Python logic, used to demonstrate that mutation-based
localization finds faults SBFL's coverage signal cannot.
"""

from .weighted import weighted_mean, weighted_sum
from .smoothing import smooth, apply_alpha
from .scaling import scale_rows, scale_total
from .averaging import moving_average, window_sum
from .discount import discounted_value, future_value

__all__ = [
    "weighted_mean",
    "weighted_sum",
    "smooth",
    "apply_alpha",
    "scale_rows",
    "scale_total",
    "moving_average",
    "window_sum",
    "discounted_value",
    "future_value",
]