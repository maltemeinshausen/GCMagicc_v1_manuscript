# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib

import numpy as np


workflow = importlib.import_module("gcmagicc_eval.workflows.1090_drought_common_protocol")


def test_bootstrap_is_deterministic_and_bounded() -> None:
    values = np.asarray([[-2.0, -1.0, 0.0, 1.0, 2.0], [-3.0, -2.0, -1.0, 0.0, 1.0]])
    first = workflow._bootstrap_probabilities(values, -1.0, 200, 42)
    second = workflow._bootstrap_probabilities(values, -1.0, 200, 42)
    np.testing.assert_array_equal(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_shared_fit_applies_factual_calibration_to_target() -> None:
    years = np.arange(1980, 2021)
    factual = np.stack([years, years + 1.0])[:, :, None].astype(float)
    target = np.stack([years + 5.0, years + 6.0])[:, :, None].astype(float)
    transformed = workflow._transform_shared(factual, years, target, (1991, 2010))
    assert transformed.shape == target.shape
    assert np.isfinite(transformed).all()
    assert float(np.nanmean(transformed)) > 0.0


def test_window_requires_one_december_value_per_year() -> None:
    years = np.arange(1990, 2021)
    values = np.zeros((3, years.size))
    selected = workflow._window(values, years, (1995, 2014))
    assert selected.shape == (3, 20)


def test_centered_rsds_smoothing_preserves_constant_field() -> None:
    values = np.full((86, 4), 123.5)
    smoothed = workflow._rolling_mean_centered_nan(values)
    np.testing.assert_allclose(smoothed, values)


def test_zero_event_probability_gets_one_sided_block_bound() -> None:
    values = np.ones((10, 5))
    row, draws = workflow._probability_row(
        "test", "natural", "thornthwaite", "adjusted", (1991, 2010),
        "gridcell-spei-area-mean", (2021, 2025), values, -1.0, 100,
    )
    assert row["estimate"] == 0.0
    assert row["one_sided"] is True
    assert 0.0 < row["upper"] < 1.0
    assert np.all(draws == 0.0)
