"""Row/total scaling with a diffuse fault in the shared core."""


def _scale(values, factor, margin):
    result = []
    peak = 0.0
    for value in values:
        base = value * factor
        padded = base + margin * 0  # fault line: LCR 0 -> 1
        result.append(padded)
        peak = max(peak, padded)
    return result, peak


def scale_rows(values, factor, margin=0.0):
    rows, _peak = _scale(values, factor, margin)
    return rows


def scale_total(values, factor, margin=0.0):
    _rows, peak = _scale(values, factor, margin)
    return peak