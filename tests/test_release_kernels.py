# SPDX-License-Identifier: Apache-2.0
import math

from gcmagicc_repro.kernels import area_weighted_mean, corrected_tas_predictor, modified_hargreaves_monthly_mm, moving_block_resample
from gcmagicc_repro.drought_protocol import december_series, hierarchical_probability_interval


def test_hargreaves_includes_energy_to_water_conversion() -> None:
    got = modified_hargreaves_monthly_mm(20.0, 15.0, 27.0, 18.0, 30)
    expected = 0.0023 * 0.408 * 18.0 * 37.8 * math.sqrt(12.0) * 30
    assert got == expected


def test_two_pass_changes_only_temperature_predictor_values() -> None:
    assert corrected_tas_predictor([1.0, 2.0], [0.2, -0.1]) == [0.98, 2.01]


def test_area_weighting_uses_cosine_latitude() -> None:
    got = area_weighted_mean([1.0, 3.0], [0.0, 60.0])
    assert math.isclose(got, (1.0 + 1.5) / 1.5)


def test_moving_block_resample_is_seeded() -> None:
    a = moving_block_resample(list(range(20)), 5, 20260711)
    b = moving_block_resample(list(range(20)), 5, 20260711)
    assert a == b
    assert len(a) == 20


def test_event_selection_keeps_one_december_value_per_year() -> None:
    got = december_series([2024, 2024, 2025, 2025], [11, 12, 11, 12], [-1.0, -1.2, -1.5, -2.0])
    assert got == {2024: -1.2, 2025: -2.0}


def test_zero_event_bootstrap_reports_one_sided_bound() -> None:
    got = hierarchical_probability_interval([[0.0] * 10, [0.5] * 10], -1.0, replicates=200, seed=9)
    assert got.probability == 0.0
    assert got.lower == 0.0
    assert got.one_sided is True
