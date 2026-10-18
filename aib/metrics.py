"""Metrics and statistics (task 5.1 completes this module; Req 6).

`wilson_interval` is the 95% Wilson score interval for a proportion, used where cases are independent
(the CI smoke subset and per-output filter statistics in the calibration check).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

Z95 = 1.959964


@dataclass(frozen=True)
class RateCI:
    point: float
    lower: float
    upper: float
    method: str
    n: int  # the denominator

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class NotComputed:
    cause: str

    def to_dict(self) -> dict:
        return {"not_computed": True, "cause": self.cause}


def wilson_interval(successes: int, n: int, z: float = Z95) -> RateCI:
    if n <= 0:
        raise ValueError("wilson_interval needs n > 0")
    if not 0 <= successes <= n:
        raise ValueError("successes must lie in [0, n]")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return RateCI(point=p, lower=max(0.0, centre - half), upper=min(1.0, centre + half), method="wilson", n=n)


def rate_or_not_computed(successes: int, n: int, cause: str) -> RateCI | NotComputed:
    return wilson_interval(successes, n) if n > 0 else NotComputed(cause)
