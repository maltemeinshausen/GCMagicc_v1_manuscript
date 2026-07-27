from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "gcmagicc_eval"
    / "workflows"
    / "1060_emergent_constraints.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("emergent_constraints_1060", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def emergent():
    return _load_module()


def test_paired_difference_standard_deviation_uses_finite_points_and_ddof_one(emergent):
    points = {
        "model-a": {"x": [1.0, 2.0, np.nan], "y": [1.1, 1.8, 9.0]},
        "model-b": {"x": [3.0, 4.0], "y": [3.3, np.inf]},
    }

    standard_deviation, n_points = emergent._paired_difference_standard_deviation(points)

    assert n_points == 3
    assert standard_deviation == pytest.approx(np.std([0.1, -0.2, 0.3], ddof=1))


@pytest.mark.parametrize(
    ("points", "expected_n"),
    [
        ({}, 0),
        ({"model-a": {"x": [1.0], "y": [1.2]}}, 1),
        ({"model-a": {"x": [np.nan], "y": [1.2]}}, 0),
    ],
)
def test_paired_difference_standard_deviation_requires_two_finite_pairs(
    emergent,
    points,
    expected_n,
):
    standard_deviation, n_points = emergent._paired_difference_standard_deviation(points)

    assert standard_deviation is None
    assert n_points == expected_n


def test_panel_a_standard_deviation_bounds_are_clipped_and_annotated(emergent):
    fig, ax = plt.subplots()
    ax.set_xlim(0.0, 7.5)
    ax.set_ylim(0.0, 7.5)
    standard_deviation = 0.24

    lines, annotations = emergent._draw_panel_a_standard_deviation(ax, standard_deviation)

    assert len(lines) == 2
    offsets = []
    for line in lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        assert np.all((0.0 <= x) & (x <= 7.5))
        assert np.all((0.0 <= y) & (y <= 7.5))
        offsets.append(float(np.mean(y - x)))
    assert offsets == pytest.approx([-standard_deviation, standard_deviation])

    assert len(annotations) == 2
    assert annotations[0].get_text() == (
        "±1 standard deviation\nof paired difference = 0.24 °C"
    )
    assert annotations[1].get_text() == ""
    assert annotations[0].arrow_patch is not None
    assert annotations[1].arrow_patch is not None
    arrow_offsets = sorted(float(y - x) for x, y in (annotation.xy for annotation in annotations))
    assert arrow_offsets == pytest.approx([-standard_deviation, standard_deviation])
    plt.close(fig)


def test_scatter_alpha_is_separate_from_legend_alpha(emergent):
    fig, ax = plt.subplots()
    handles, x, y = emergent._plot_scatter_from_points(
        ax=ax,
        points_by_version={"model-a": {"x": [1.0, 1.0], "y": [1.0, 1.0]}},
        model_styles={"model-a": {"marker": "o", "color": "#336699"}},
        point_size=20.0,
        point_alpha=emergent.DEFAULT_POINT_ALPHA,
    )

    assert x.size == y.size == 2
    assert ax.collections[0].get_alpha() == pytest.approx(0.12)
    assert handles[0].get_alpha() == pytest.approx(emergent.LEGEND_POINT_ALPHA)
    plt.close(fig)


def test_composite_payload_records_spread_and_compositing_metadata(emergent):
    top_points = {
        "model-a": {"x": [1.0, 2.0, 3.0], "y": [1.1, 1.8, 3.3]},
    }
    payload = emergent._build_composite_payload(
        scenarios=["ssp126"],
        model_styles={"model-a": {"marker": "o", "color": "#336699"}},
        top_points=top_points,
        bottom_points=top_points,
        era_by_scen={"ssp126": np.asarray([1.5])},
        qdf=pd.DataFrame(
            {"ssp126": [1.0, 1.1, 1.2, 1.5, 1.8, 1.9, 2.0]},
            index=emergent.REQUIRED_QUANTILES,
        ),
        point_size=20.0,
        point_alpha=0.12,
        dpi=320,
        output_base="figure",
        formats=["pdf", "png"],
    )

    assert payload["schema"] == "emergent_constraint_composite_v1"
    assert payload["panel_a_spread"] == {
        "definition": "sample_standard_deviation_of_y_minus_x",
        "degrees_of_freedom": 1,
        "units": "degC",
        "value": pytest.approx(np.std([0.1, -0.2, 0.3], ddof=1)),
        "n_points": 3,
    }
    assert payload["render_settings"]["point_alpha"] == pytest.approx(0.12)
    assert (
        payload["render_settings"]["point_compositing_strategy"]
        == emergent.POINT_COMPOSITING_STRATEGY
    )


def test_default_data_root_uses_release_native_provenance_layout(emergent):
    assert emergent._resolve_default_data_root(SCRIPT_PATH.parent) == (
        SCRIPT_PATH.parents[3]
        / "data"
        / "derived"
        / "emergent_constraints"
    )
