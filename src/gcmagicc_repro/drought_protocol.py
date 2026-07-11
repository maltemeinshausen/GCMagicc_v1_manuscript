# SPDX-License-Identifier: Apache-2.0
"""Locked event-selection and hierarchical-bootstrap protocol for SPEI-48."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskInterval:
    probability: float
    lower: float
    upper: float
    one_sided: bool


def december_series(years: Sequence[int], months: Sequence[int], values: Sequence[float]) -> dict[int, float]:
    if not (len(years) == len(months) == len(values)):
        raise ValueError("years, months, and values must have equal length")
    selected: dict[int, float] = {}
    for year, month, value in zip(years, months, values, strict=True):
        if month != 12 or not math.isfinite(value):
            continue
        if year in selected:
            raise ValueError(f"multiple December values for {year}")
        selected[year] = float(value)
    return selected


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def hierarchical_probability_interval(
    member_december_values: Sequence[Sequence[float]],
    threshold: float,
    *,
    replicates: int = 10_000,
    block_years: int = 5,
    seed: int = 20260711,
) -> RiskInterval:
    """Resample members and circular moving blocks of annual December values."""
    if not member_december_values or replicates <= 0:
        raise ValueError("members and positive replicate count are required")
    members = [list(map(float, member)) for member in member_december_values]
    if any(len(member) < block_years for member in members):
        raise ValueError("every member must contain at least one complete block")
    pooled = [value for member in members for value in member]
    estimate = sum(value <= threshold for value in pooled) / len(pooled)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(replicates):
        sample: list[float] = []
        for _member_slot in members:
            member = members[rng.randrange(len(members))]
            generated: list[float] = []
            while len(generated) < len(member):
                start = rng.randrange(len(member))
                generated.extend(member[(start + offset) % len(member)] for offset in range(block_years))
            sample.extend(generated[: len(member)])
        draws.append(sum(value <= threshold for value in sample) / len(sample))
    one_sided = estimate == 0.0
    if one_sided:
        return RiskInterval(estimate, 0.0, _percentile(draws, 0.95), True)
    return RiskInterval(estimate, _percentile(draws, 0.025), _percentile(draws, 0.975), False)
