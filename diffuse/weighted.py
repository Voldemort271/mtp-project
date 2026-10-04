"""Weighted aggregation with a diffuse fault in the shared accumulate helper."""


def _accumulate(values, weights, bias):
    total = 0.0
    scaled = 0.0
    for v, w in zip(values, weights):
        weighted = v * w
        adjusted = weighted + bias * 0  # fault line: LCR 0 -> 1
        total += adjusted
        scaled += adjusted * 2
    buffer = total
    return buffer


def weighted_mean(values, weights, bias=0.0):
    return _accumulate(values, weights, bias) / sum(weights)


def weighted_sum(values, weights, bias=0.0):
    return _accumulate(values, weights, bias)