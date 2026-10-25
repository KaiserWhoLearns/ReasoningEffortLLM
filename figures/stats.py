"""The two interval estimates the tables and figures use.

`wilson` for a resolve rate (k of n tasks passed): the interval stays inside [0, 1] even at k=0
or k=n.

`boot_ci` for a per-unit mean (dollars, LM calls, tokens on the easy set). These distributions
are right-skewed and the easy set can be small, so a percentile bootstrap is used rather than a
symmetric interval. The bootstrap is seeded, so unchanged data reproduces the same interval.
"""
from __future__ import annotations

import math

import numpy as np

Z = 1.96
N_BOOT = 10_000
SEED = 0


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float, float]:
    """(rate, lo, hi) as PERCENTAGES — Wilson score interval, 95% at the default `z`."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * p, 100 * max(centre - half, 0.0), 100 * min(centre + half, 1.0)


def boot_ci(values) -> tuple[float | None, float | None, float | None]:
    """(mean, lo, hi) of a per-unit mean — 95% percentile bootstrap over the units.

    Returns (None, None, None) with nothing to average, and (mean, None, None) from a single unit.
    """
    xs = np.asarray([v for v in values if v is not None], dtype=float)
    if xs.size == 0:
        return None, None, None
    mean = float(xs.mean())
    if xs.size < 2:
        return mean, None, None
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, xs.size, size=(N_BOOT, xs.size))
    means = xs[draws].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return mean, float(lo), float(hi)
