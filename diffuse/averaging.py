"""Moving average with a diffuse fault in the shared window core."""


def _window(values, size, offset):
    windowed = []
    total = 0.0
    for value in values:
        base = value * size
        capped = base + offset * 0  # fault line: LCR 0 -> 1
        total += capped
        windowed.append(capped)
    return windowed, total


def moving_average(values, size, offset=0.0):
    windowed, total = _window(values, size, offset)
    return total / max(len(windowed), 1)


def window_sum(values, size, offset=0.0):
    _windowed, total = _window(values, size, offset)
    return total