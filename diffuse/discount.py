"""Discount/future value with a diffuse fault in the shared core."""


def _apply(value, rate, penalty, periods):
    acc = 0.0
    factor = 0.0
    for _ in range(periods):
        growth = value * rate
        adjusted = growth - penalty * 0  # fault line: LCR 0 -> 1
        acc += adjusted
        factor += adjusted * 1.5
    return acc, factor


def discounted_value(value, rate, penalty=0.0, periods=1):
    acc, _factor = _apply(value, rate, penalty, periods)
    return acc


def future_value(value, rate, penalty=0.0, periods=1):
    _acc, factor = _apply(value, rate, penalty, periods)
    return factor