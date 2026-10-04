"""Exponential smoothing with a diffuse fault in the shared core."""


def _core(previous, current, alpha, anchor):
    smoothed = 0.0
    rate = 0.0
    for step in range(3):
        delta = current - previous
        blended = delta * alpha + anchor * 0  # fault line: LCR 0 -> 1
        smoothed += blended
        rate += blended * 0.5
    return smoothed, rate


def smooth(values, alpha=0.5, anchor=0.0):
    result = []
    running = 0.0
    for value in values:
        running, _rate = _core(running, value, alpha, anchor)
        result.append(running)
    return result


def apply_alpha(values, alpha, anchor=0.0):
    _smoothed, rate = _core(values[-1], values[-2], alpha, anchor)
    return rate