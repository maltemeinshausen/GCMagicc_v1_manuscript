# SPDX-License-Identifier: Apache-2.0
"""Small, audited scientific kernels used by release verification."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def modified_hargreaves_monthly_mm(
    temperature_c: float,
    temperature_min_c: float,
    temperature_max_c: float,
    rsds_mj_m2_day: float,
    days: int,
) -> float:
    """Modified Hargreaves radiation proxy with MJ-to-mm factor 0.408."""
    spread = max(0.0, temperature_max_c - temperature_min_c)
    daily = 0.0023 * 0.408 * rsds_mj_m2_day * (temperature_c + 17.8) * math.sqrt(spread)
    return max(0.0, daily * days)


def area_weighted_mean(values: Sequence[float], latitudes_deg: Sequence[float]) -> float:
    if len(values) != len(latitudes_deg) or not values:
        raise ValueError("values and latitudes must have equal, non-zero length")
    weights = [math.cos(math.radians(lat)) for lat in latitudes_deg]
    total = sum(weights)
    if total <= 0:
        raise ValueError("area weights sum to zero")
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total


def corrected_tas_predictor(normalized: Sequence[float], annual_delta_c: Sequence[float]) -> list[float]:
    """Apply the pass-1 annual temperature deltas used by the single pass-2 rerun."""
    if len(normalized) != len(annual_delta_c):
        raise ValueError("predictor and delta arrays must have the same length")
    return [value - delta / 10.0 for value, delta in zip(normalized, annual_delta_c, strict=True)]


def moving_block_resample(values: Sequence[float], block: int, seed: int) -> list[float]:
    """Deterministic circular moving-block resample used by the smoke check."""
    if block <= 0 or block > len(values):
        raise ValueError("invalid block length")
    rng = random.Random(seed)
    out: list[float] = []
    while len(out) < len(values):
        start = rng.randrange(len(values))
        out.extend(values[(start + offset) % len(values)] for offset in range(block))
    return out[: len(values)]
