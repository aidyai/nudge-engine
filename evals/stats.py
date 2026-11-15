"""The few statistics the benchmarks and evals need. No dependencies."""

from __future__ import annotations

import math
from collections.abc import Sequence


def pct(values: Sequence[float], p: float) -> float:
    """The p-th percentile (0 to 100), by linear interpolation between the closest values."""
    if not values:
        raise ValueError("no values")
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(values: Sequence[float]) -> dict:
    """n, p50, p95, p99, mean and max of a list of measurements."""
    if not values:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "mean": None, "max": None}
    return {
        "n": len(values),
        "p50": pct(values, 50),
        "p95": pct(values, 95),
        "p99": pct(values, 99),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a pass rate. Honest with few runs: 10 of 10 is not '100%', it is 72% to 100%."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
