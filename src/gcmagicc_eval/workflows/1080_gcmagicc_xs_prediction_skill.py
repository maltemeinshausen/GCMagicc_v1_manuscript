#!/usr/bin/env python3
"""
1080_gcmagicc_xs_prediction_skill
=======================================

Create one multi-panel observed-vs-predicted scatter figure per variable from
GCMagicc-XS monthly prediction-skill output.

Default invocation:

  python notebooks/1080_gcmagicc_xs_prediction_skill.py
"""

from __future__ import annotations

import argparse
import math
import os
import re
import string
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "matplotlib"),
)

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter


# ---------------------------------------------------------------------------
# User-editable defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT_CSV = _REPO_ROOT / "data" / "GCMagicc-XS_data" / "full_monthly_results.csv"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "GCMagicc-XS_data" / "prediction_skill"
DEFAULT_OUTPUT_PREFIX = "gcmagicc_xs_prediction_skill"

DEFAULT_EXCLUDE_SCENARIOS = ("abrupt-2xCO2", "abrupt-4xCO2")
DEFAULT_VARIABLES: tuple[str, ...] | None = None  # None means discover all variables.
DEFAULT_FORMATS = ("pdf", "png")
DEFAULT_DPI = 220
DEFAULT_CHUNKSIZE = 1_000_000
DEFAULT_MAX_ROWS: int | None = None

DEFAULT_PANEL_SIZE_INCH = 1.35
DEFAULT_LEFT_MARGIN_INCH = 0.72
DEFAULT_RIGHT_MARGIN_INCH = 0.62
DEFAULT_TOP_MARGIN_INCH = 0.52
DEFAULT_BOTTOM_MARGIN_INCH = 1.10
DEFAULT_PANEL_LABEL_FONTSIZE = 6.5
DEFAULT_TICK_FONTSIZE = 5.6
DEFAULT_AXIS_LABEL_FONTSIZE = 8.0
DEFAULT_LEGEND_FONTSIZE = 5.8
DEFAULT_LEGEND_NCOL = 6

DEFAULT_TRAINING_COLOR = "#e66101"
DEFAULT_POINT_SIZE = 2.0
DEFAULT_POINT_ALPHA = 0.18
DEFAULT_HOLDOUT_COLOR = "#56B4E9"
DEFAULT_TEST_RING_SIZE = 10.0
DEFAULT_TEST_RING_ALPHA = 0.58
DEFAULT_TEST_RING_LINEWIDTH = 0.35
DEFAULT_TEST_INTERVAL_LINEWIDTH = 0.22
DEFAULT_TEST_INTERVAL_ALPHA = 0.48
DEFAULT_DIAGONAL_LINEWIDTH = 0.55
DEFAULT_SPINE_LINEWIDTH = 0.45
DEFAULT_AXIS_PADDING_FRACTION = 0.006
DEFAULT_N_AXIS_TICKS = 4
DEFAULT_GREY_SCENARIOS = ("ssp585",)
DEFAULT_VARIABLE_UNITS = {
    "hurs": "%",
    "huss": r"kg kg$^{-1}$",
    "pr": r"kg m$^{-2}$ s$^{-1}$",
    "psl": "Pa",
    "rsds": r"W m$^{-2}$",
    "sfcWind": r"m s$^{-1}$",
    "tas": "K",
    "tasmax": "K",
    "tasmin": "K",
    "ts": "K",
}
DEFAULT_VARIABLE_TITLES = {
    "hurs": "Near-Surface Relative Humidity (hurs)",
    "huss": "Near-Surface Specific Humidity (huss)",
    "pr": "Precipitation (pr)",
    "psl": "Sea Level Pressure (psl)",
    "rsds": "Surface Downwelling Shortwave Radiation (rsds)",
    "sfcWind": "Near-Surface Wind Speed (sfcWind)",
    "tas": "Surface Air Temperature (tas)",
    "tasmax": "Daily Maximum Near-Surface Air Temperature (tasmax)",
    "tasmin": "Daily Minimum Near-Surface Air Temperature (tasmin)",
    "ts": "Surface Temperature (ts)",
}

DEFAULT_SCENARIO_ORDER_HINTS = (
    "historical",
    "hist-nat",
    "hist-aer",
    "hist-GHG",
    "ssp119",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp585",
)

DEFAULT_SCENARIO_COLOR_OVERRIDES = {
    "historical": "#4b4b4b",
    "hist-nat": "#2ca25f",
    "hist-aer": "#3182bd",
    "hist-GHG": "#de2d26",
    "ssp119": "#756bb1",
    "ssp126": "#31a354",
    "ssp245": "#fdae61",
    "ssp370": "#e6550d",
    "ssp585": "#9e0142",
}

REQUIRED_COLUMNS = (
    "model",
    "scenario",
    "is_test",
    "variable",
    "obs",
    "pred_mean",
    "pred_p10",
    "pred_p90",
)
RANGE_COLUMNS = ("obs", "pred_mean", "pred_p10", "pred_p90")
POINT_COLUMNS = ("obs", "pred_mean", "pred_p10", "pred_p90")


@dataclass
class VariableSummary:
    value_min: float = math.inf
    value_max: float = -math.inf
    models: set[str] = field(default_factory=set)
    scenarios: set[str] = field(default_factory=set)
    training_scenarios: set[str] = field(default_factory=set)
    test_scenarios: set[str] = field(default_factory=set)
    rows: int = 0
    test_rows: int = 0


@dataclass
class CsvSummary:
    variables: dict[str, VariableSummary] = field(default_factory=dict)
    rows_seen: int = 0
    rows_after_filter: int = 0
    rows_dropped_malformed: int = 0
    rows_excluded_scenario: int = 0
    file_changed_during_read: bool = False


PointArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
PointStore = dict[tuple[str, str, str, bool], list[PointArrays]]


def panel_letter(idx: int) -> str:
    letters = string.ascii_lowercase
    if idx < len(letters):
        return letters[idx]
    q, r = divmod(idx, len(letters))
    return f"{letters[q - 1]}{letters[r]}"


def natural_sort_key(value: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def _parse_csv_list(raw: str | Iterable[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    out: list[str] = []
    for item in raw:
        out.extend(part.strip() for part in str(item).split(",") if part.strip())
    return tuple(out)


def _parse_optional_csv_list(raw: str | None) -> tuple[str, ...] | None:
    parsed = _parse_csv_list(raw)
    return parsed if parsed else None


def _normalise_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    norm = series.astype("string").str.strip().str.lower()
    return norm.isin(("true", "t", "1", "yes", "y"))


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _iter_csv_chunks(
    path: Path,
    *,
    chunksize: int,
    max_rows: int | None,
) -> Iterable[pd.DataFrame]:
    rows_left = max_rows
    reader = pd.read_csv(
        path,
        usecols=list(REQUIRED_COLUMNS),
        chunksize=int(chunksize),
        on_bad_lines="warn",
    )
    for chunk in reader:
        if rows_left is not None:
            if rows_left <= 0:
                break
            chunk = chunk.iloc[:rows_left].copy()
            rows_left -= len(chunk)
        yield chunk


def _clean_and_filter_chunk(
    chunk: pd.DataFrame,
    *,
    exclude_scenarios: set[str],
    variables: set[str] | None,
) -> tuple[pd.DataFrame, int, int]:
    missing_mask = chunk.loc[:, list(REQUIRED_COLUMNS)].isna().any(axis=1)
    malformed_rows = int(missing_mask.sum())
    if malformed_rows:
        chunk = chunk.loc[~missing_mask].copy()
    else:
        chunk = chunk.copy()

    for col in RANGE_COLUMNS:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    numeric_bad_mask = chunk.loc[:, list(RANGE_COLUMNS)].isna().any(axis=1)
    numeric_bad_rows = int(numeric_bad_mask.sum())
    if numeric_bad_rows:
        chunk = chunk.loc[~numeric_bad_mask].copy()

    chunk["model"] = chunk["model"].astype(str)
    chunk["scenario"] = chunk["scenario"].astype(str)
    chunk["variable"] = chunk["variable"].astype(str)
    chunk["is_test_bool"] = _normalise_bool_series(chunk["is_test"]).to_numpy(dtype=bool)

    excluded_rows = 0
    if exclude_scenarios:
        excluded_mask = chunk["scenario"].isin(exclude_scenarios)
        excluded_rows = int(excluded_mask.sum())
        chunk = chunk.loc[~excluded_mask].copy()

    if variables is not None:
        chunk = chunk.loc[chunk["variable"].isin(variables)].copy()

    return chunk, malformed_rows + numeric_bad_rows, excluded_rows


def _scan_csv(
    input_csv: Path,
    *,
    chunksize: int,
    max_rows: int | None,
    exclude_scenarios: set[str],
    variables: set[str] | None,
) -> CsvSummary:
    before = _file_signature(input_csv)
    summary = CsvSummary()

    for chunk_idx, raw_chunk in enumerate(
        _iter_csv_chunks(input_csv, chunksize=chunksize, max_rows=max_rows), start=1
    ):
        summary.rows_seen += len(raw_chunk)
        chunk, malformed_rows, excluded_rows = _clean_and_filter_chunk(
            raw_chunk,
            exclude_scenarios=exclude_scenarios,
            variables=variables,
        )
        summary.rows_dropped_malformed += malformed_rows
        summary.rows_excluded_scenario += excluded_rows
        summary.rows_after_filter += len(chunk)

        if chunk.empty:
            print(f"scan chunk {chunk_idx}: kept 0 rows", flush=True)
            continue

        for variable, group in chunk.groupby("variable", sort=False):
            variable = str(variable)
            var_summary = summary.variables.setdefault(variable, VariableSummary())
            vals = group.loc[:, list(RANGE_COLUMNS)]
            value_min = float(vals.min(skipna=True).min())
            value_max = float(vals.max(skipna=True).max())
            if math.isfinite(value_min):
                var_summary.value_min = min(var_summary.value_min, value_min)
            if math.isfinite(value_max):
                var_summary.value_max = max(var_summary.value_max, value_max)
            var_summary.models.update(map(str, group["model"].unique()))
            var_summary.scenarios.update(map(str, group["scenario"].unique()))
            train_group = group.loc[~group["is_test_bool"]]
            if not train_group.empty:
                var_summary.training_scenarios.update(map(str, train_group["scenario"].unique()))
            test_group = group.loc[group["is_test_bool"]]
            if not test_group.empty:
                var_summary.test_scenarios.update(map(str, test_group["scenario"].unique()))
            var_summary.rows += len(group)
            var_summary.test_rows += int(group["is_test_bool"].sum())

        print(
            f"scan chunk {chunk_idx}: kept {len(chunk):,} rows "
            f"(total kept {summary.rows_after_filter:,})",
            flush=True,
        )

    after = _file_signature(input_csv)
    summary.file_changed_during_read = before != after
    return summary


def _collect_points(
    input_csv: Path,
    *,
    chunksize: int,
    max_rows: int | None,
    exclude_scenarios: set[str],
    variables: set[str],
) -> tuple[PointStore, int, int]:
    points: PointStore = defaultdict(list)
    dropped = 0
    kept = 0

    for chunk_idx, raw_chunk in enumerate(
        _iter_csv_chunks(input_csv, chunksize=chunksize, max_rows=max_rows), start=1
    ):
        chunk, malformed_rows, _excluded_rows = _clean_and_filter_chunk(
            raw_chunk,
            exclude_scenarios=exclude_scenarios,
            variables=variables,
        )
        dropped += malformed_rows
        kept += len(chunk)

        if not chunk.empty:
            for (variable, model, scenario, is_test), group in chunk.groupby(
                ["variable", "model", "scenario", "is_test_bool"],
                sort=False,
                observed=True,
            ):
                x = group["obs"].to_numpy(dtype=np.float32, copy=True)
                y = group["pred_mean"].to_numpy(dtype=np.float32, copy=True)
                p10 = group["pred_p10"].to_numpy(dtype=np.float32, copy=True)
                p90 = group["pred_p90"].to_numpy(dtype=np.float32, copy=True)
                points[(str(variable), str(model), str(scenario), bool(is_test))].append((x, y, p10, p90))

        print(
            f"collect chunk {chunk_idx}: kept {len(chunk):,} rows "
            f"(total kept {kept:,})",
            flush=True,
        )

    return points, kept, dropped


def _scenario_sort_key(scenario: str) -> tuple[int, tuple[object, ...]]:
    try:
        idx = DEFAULT_SCENARIO_ORDER_HINTS.index(scenario)
    except ValueError:
        idx = len(DEFAULT_SCENARIO_ORDER_HINTS)
    return idx, natural_sort_key(scenario)


def _assign_scenario_colors(scenarios: Iterable[str]) -> dict[str, str]:
    ordered = sorted(set(scenarios), key=_scenario_sort_key)
    fallback_names = (
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    )
    colors: dict[str, str] = {}
    fallback_idx = 0
    for scenario in ordered:
        if scenario in DEFAULT_SCENARIO_COLOR_OVERRIDES:
            colors[scenario] = DEFAULT_SCENARIO_COLOR_OVERRIDES[scenario]
        else:
            colors[scenario] = fallback_names[fallback_idx % len(fallback_names)]
            fallback_idx += 1
    return colors


def _nice_number(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10**exponent)
    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5.0:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return nice_fraction * (10**exponent)


def _nice_limits(
    value_min: float,
    value_max: float,
    *,
    padding_fraction: float,
    n_ticks: int,
) -> tuple[float, float]:
    if not (math.isfinite(value_min) and math.isfinite(value_max)):
        return 0.0, 1.0

    if value_min == value_max:
        pad = abs(value_min) * 0.05 if value_min else 1.0
        value_min -= pad
        value_max += pad

    span = value_max - value_min
    pad = span * padding_fraction
    if pad == 0.0:
        pad = max(abs(value_min) * padding_fraction, 1e-12)
    return float(value_min - pad), float(value_max + pad)


def _axis_ticks(limits: tuple[float, float], *, n_ticks: int) -> np.ndarray:
    locator = MaxNLocator(nbins=max(2, n_ticks), steps=[1, 2, 2.5, 5, 10])
    ticks = locator.tick_values(limits[0], limits[1])
    eps = (limits[1] - limits[0]) * 1e-9
    inner_ticks = ticks[(ticks >= limits[0] - eps) & (ticks < limits[1] - eps)]
    if inner_ticks.size:
        return inner_ticks
    return np.array([(limits[0] + limits[1]) / 2.0])


def _safe_variable_for_filename(variable: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", variable).strip("_") or "variable"


def _variable_with_unit(variable: str) -> str:
    unit = DEFAULT_VARIABLE_UNITS.get(variable)
    if not unit:
        return variable
    return f"{variable} ({unit})"


def _variable_title(variable: str) -> str:
    return DEFAULT_VARIABLE_TITLES.get(variable, variable)


def _concat_point_chunks(
    segments: list[PointArrays] | None,
) -> PointArrays | None:
    if not segments:
        return None
    if len(segments) == 1:
        return segments[0]
    return (
        np.concatenate([segment[0] for segment in segments]),
        np.concatenate([segment[1] for segment in segments]),
        np.concatenate([segment[2] for segment in segments]),
        np.concatenate([segment[3] for segment in segments]),
    )


def _format_outer_axis(ax: plt.Axes, limits: tuple[float, float], ticks: np.ndarray) -> None:
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_aspect("equal", adjustable="box")

    formatter_x = ScalarFormatter(useMathText=True)
    formatter_x.set_powerlimits((-3, 4))
    formatter_x.set_useOffset(False)
    formatter_y = ScalarFormatter(useMathText=True)
    formatter_y.set_powerlimits((-3, 4))
    formatter_y.set_useOffset(False)
    ax.xaxis.set_major_formatter(formatter_x)
    ax.yaxis.set_major_formatter(formatter_y)

    for spine in ax.spines.values():
        spine.set_linewidth(DEFAULT_SPINE_LINEWIDTH)
        spine.set_color("#4a4a4a")


def _legend_handles(
    *,
    training_scenarios: Iterable[str],
    test_scenarios: Iterable[str],
) -> tuple[list[object], list[str], dict[type, HandlerTuple]]:
    handles: list[object] = []
    labels: list[str] = []
    training_scenarios = sorted(set(training_scenarios), key=_scenario_sort_key)
    if training_scenarios:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=DEFAULT_TRAINING_COLOR,
                markeredgecolor="none",
                alpha=DEFAULT_POINT_ALPHA,
                markersize=4.2,
            )
        )
        labels.append(f"Training scenario ({', '.join(training_scenarios)})")

    test_scenarios = sorted(set(test_scenarios), key=_scenario_sort_key)
    if test_scenarios:
        interval = Line2D(
            [0],
            [0],
            marker="|",
            linestyle="none",
            markeredgecolor=DEFAULT_HOLDOUT_COLOR,
            markeredgewidth=0.9,
            markersize=8.5,
        )
        ring = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=DEFAULT_HOLDOUT_COLOR,
            markeredgewidth=DEFAULT_TEST_RING_LINEWIDTH * 2.0,
            alpha=DEFAULT_TEST_RING_ALPHA,
            markersize=6.4,
        )
        handles.append((interval, ring))
        labels.append("Hold-out scenario (SSP5-8.5), mean (circle) and p10-p90 (bar)")
    return handles, labels, {tuple: HandlerTuple(ndivide=1)}


def plot_variable(
    *,
    variable: str,
    summary: VariableSummary,
    points: PointStore,
    output_dir: Path,
    output_prefix: str,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    models = sorted(summary.models, key=natural_sort_key)
    scenarios = sorted(summary.scenarios, key=_scenario_sort_key)

    if not models:
        raise ValueError(f"No models found for variable {variable!r}.")

    n_models = len(models)
    ncols = int(math.ceil(math.sqrt(n_models)))
    nrows = int(math.ceil(n_models / ncols))

    limits = _nice_limits(
        summary.value_min,
        summary.value_max,
        padding_fraction=DEFAULT_AXIS_PADDING_FRACTION,
        n_ticks=DEFAULT_N_AXIS_TICKS,
    )
    ticks = _axis_ticks(limits, n_ticks=DEFAULT_N_AXIS_TICKS)

    grid_w = ncols * DEFAULT_PANEL_SIZE_INCH
    grid_h = nrows * DEFAULT_PANEL_SIZE_INCH
    fig_w = DEFAULT_LEFT_MARGIN_INCH + grid_w + DEFAULT_RIGHT_MARGIN_INCH
    fig_h = DEFAULT_BOTTOM_MARGIN_INCH + grid_h + DEFAULT_TOP_MARGIN_INCH

    left = DEFAULT_LEFT_MARGIN_INCH / fig_w
    right = (DEFAULT_LEFT_MARGIN_INCH + grid_w) / fig_w
    bottom = DEFAULT_BOTTOM_MARGIN_INCH / fig_h
    top = (DEFAULT_BOTTOM_MARGIN_INCH + grid_h) / fig_h

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=0.0,
        hspace=0.0,
    )

    axes: list[plt.Axes] = []
    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)
        if idx >= n_models:
            ax.set_visible(False)
            continue

        model = models[idx]
        _format_outer_axis(ax, limits, ticks)
        ax.plot(
            limits,
            limits,
            color="#b8b8b8",
            lw=DEFAULT_DIAGONAL_LINEWIDTH,
            zorder=0,
        )

        for scenario in scenarios:
            test_segments = points.get((variable, model, scenario, True))
            test_xy = _concat_point_chunks(test_segments)

            if test_xy is not None:
                x_test, y_test, p10_test, p90_test = test_xy
                interval_lines = ax.vlines(
                    x_test,
                    p10_test,
                    p90_test,
                    color=DEFAULT_HOLDOUT_COLOR,
                    linewidth=DEFAULT_TEST_INTERVAL_LINEWIDTH,
                    alpha=DEFAULT_TEST_INTERVAL_ALPHA,
                    zorder=1.25,
                )
                interval_lines.set_rasterized(True)
                ax.scatter(
                    x_test,
                    y_test,
                    s=DEFAULT_TEST_RING_SIZE,
                    facecolors="none",
                    edgecolors=DEFAULT_HOLDOUT_COLOR,
                    linewidths=DEFAULT_TEST_RING_LINEWIDTH,
                    alpha=DEFAULT_TEST_RING_ALPHA,
                    zorder=1.4,
                    rasterized=True,
                )

            train_segments = points.get((variable, model, scenario, False))
            train_xy = _concat_point_chunks(train_segments)
            if train_xy is not None:
                x_train, y_train, _p10_train, _p90_train = train_xy
                ax.scatter(
                    x_train,
                    y_train,
                    s=DEFAULT_POINT_SIZE,
                    color=DEFAULT_TRAINING_COLOR,
                    alpha=DEFAULT_POINT_ALPHA,
                    edgecolors="none",
                    linewidths=0.0,
                    zorder=2.0,
                    rasterized=True,
                )

        is_left = col == 0
        is_right = col == ncols - 1
        is_top = row == 0
        is_bottom = row == nrows - 1
        ax.tick_params(
            axis="x",
            which="major",
            top=is_top,
            labeltop=is_top,
            bottom=is_bottom,
            labelbottom=is_bottom,
            labelsize=DEFAULT_TICK_FONTSIZE,
            direction="out",
            length=2.2,
            pad=1.5,
        )
        ax.tick_params(
            axis="x",
            which="minor",
            top=is_top,
            bottom=is_bottom,
            labeltop=False,
            labelbottom=False,
            direction="out",
            length=1.25,
        )
        ax.tick_params(
            axis="y",
            which="major",
            left=is_left,
            labelleft=is_left,
            right=is_right,
            labelright=is_right,
            labelsize=DEFAULT_TICK_FONTSIZE,
            direction="out",
            length=2.2,
            pad=1.5,
        )
        ax.tick_params(
            axis="y",
            which="minor",
            left=is_left,
            right=is_right,
            labelleft=False,
            labelright=False,
            direction="out",
            length=1.25,
        )
        ax.xaxis.get_offset_text().set_size(DEFAULT_TICK_FONTSIZE)
        ax.yaxis.get_offset_text().set_size(DEFAULT_TICK_FONTSIZE)

        label = panel_letter(idx)
        ax.text(
            0.025,
            0.975,
            rf"$\mathbf{{{label}}}$ {model}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=DEFAULT_PANEL_LABEL_FONTSIZE,
            color="#111111",
        )

    fig.text(
        left,
        min(0.985, top + 0.045),
        _variable_title(variable),
        ha="left",
        va="top",
        fontsize=DEFAULT_AXIS_LABEL_FONTSIZE,
        fontweight="bold",
        color="#111111",
    )
    fig.supxlabel(
        f"Observed {_variable_with_unit(variable)}",
        x=(left + right) / 2.0,
        y=max(0.02, bottom - 0.052),
        fontsize=DEFAULT_AXIS_LABEL_FONTSIZE,
    )
    fig.supylabel(
        f"Predicted mean {_variable_with_unit(variable)}",
        x=0.018,
        y=(bottom + top) / 2.0,
        fontsize=DEFAULT_AXIS_LABEL_FONTSIZE,
    )

    handles, labels, handler_map = _legend_handles(
        training_scenarios=summary.training_scenarios,
        test_scenarios=summary.test_scenarios,
    )
    if handles:
        fig.legend(
            handles=handles,
            labels=labels,
            loc="lower left",
            bbox_to_anchor=(left, 0.018),
            frameon=False,
            ncol=min(DEFAULT_LEGEND_NCOL, len(handles)),
            fontsize=DEFAULT_LEGEND_FONTSIZE,
            handletextpad=0.32,
            borderaxespad=0.0,
            labelspacing=0.22,
            columnspacing=0.65,
            handler_map=handler_map,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_variable = _safe_variable_for_filename(variable)
    output_base = output_dir / f"{output_prefix}_{safe_variable}"

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["pdf.compression"] = 9

    saved: list[Path] = []
    for fmt in formats:
        out = output_base.with_suffix(f".{fmt.lower()}")
        fig.savefig(
            out,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.035,
            facecolor="white",
        )
        saved.append(out)
    plt.close(fig)
    return saved


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "--input-csv", dest="input_csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--exclude-scenarios",
        default=",".join(DEFAULT_EXCLUDE_SCENARIOS),
        help="Comma-separated scenarios to exclude before range calculation and plotting.",
    )
    parser.add_argument(
        "--variables",
        default=",".join(DEFAULT_VARIABLES) if DEFAULT_VARIABLES else "",
        help="Comma-separated variables to plot. Empty means discover all variables.",
    )
    parser.add_argument("--formats", default=",".join(DEFAULT_FORMATS))
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="Optional debug limit on rows read from the CSV.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    input_csv = args.input_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    exclude_scenarios = set(_parse_csv_list(args.exclude_scenarios))
    requested_variables = _parse_optional_csv_list(args.variables)
    requested_variable_set = set(requested_variables) if requested_variables else None
    formats = tuple(fmt.lower().lstrip(".") for fmt in _parse_csv_list(args.formats))

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not formats:
        raise ValueError("At least one output format is required.")

    print(f"Input CSV: {input_csv}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Excluded scenarios: {sorted(exclude_scenarios) if exclude_scenarios else 'none'}", flush=True)

    summary = _scan_csv(
        input_csv,
        chunksize=int(args.chunksize),
        max_rows=args.max_rows,
        exclude_scenarios=exclude_scenarios,
        variables=requested_variable_set,
    )

    if summary.file_changed_during_read:
        print(
            "WARNING: input CSV size or mtime changed during the scan. "
            "Results may reflect a file that was still being written.",
            file=sys.stderr,
            flush=True,
        )

    variables = sorted(summary.variables, key=natural_sort_key)
    if requested_variables:
        missing = [v for v in requested_variables if v not in summary.variables]
        if missing:
            raise ValueError(f"Requested variables not found after filtering: {missing}")
        variables = list(requested_variables)

    if not variables:
        raise ValueError("No variables remain after filtering.")

    print(
        f"Rows seen: {summary.rows_seen:,}; kept: {summary.rows_after_filter:,}; "
        f"excluded by scenario: {summary.rows_excluded_scenario:,}; "
        f"dropped malformed: {summary.rows_dropped_malformed:,}",
        flush=True,
    )
    print(f"Variables: {', '.join(variables)}", flush=True)

    points, kept_for_points, dropped_for_points = _collect_points(
        input_csv,
        chunksize=int(args.chunksize),
        max_rows=args.max_rows,
        exclude_scenarios=exclude_scenarios,
        variables=set(variables),
    )
    print(
        f"Point pass kept {kept_for_points:,} rows "
        f"and dropped {dropped_for_points:,} malformed rows.",
        flush=True,
    )

    for variable in variables:
        var_summary = summary.variables[variable]
        print(
            f"Plotting {variable}: {var_summary.rows:,} rows, "
            f"{len(var_summary.models)} models, {len(var_summary.scenarios)} scenarios, "
            f"{var_summary.test_rows:,} test rows.",
            flush=True,
        )
        saved = plot_variable(
            variable=variable,
            summary=var_summary,
            points=points,
            output_dir=output_dir,
            output_prefix=str(args.output_prefix),
            formats=formats,
            dpi=int(args.dpi),
        )
        for path in saved:
            print(f"Saved {path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
