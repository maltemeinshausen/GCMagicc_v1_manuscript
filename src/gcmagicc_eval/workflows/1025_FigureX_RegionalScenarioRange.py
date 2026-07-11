#!/usr/bin/env python3
"""
1025_FigureX_RegionalScenarioRange.py
=====================================

Build a regional scenario-range figure from 815 percentiles outputs.

The figure repeats the first two-row pattern from 1021_Figure2_ERAspliced:
for each selected variable, draw a scenario time-series row followed by a
future-minus-baseline native-unit delta bar row.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIG1021 = importlib.import_module("notebooks.1021_Figure2_ERAspliced")

LOG = logging.getLogger("1025_FigureX_RegionalScenarioRange")


DEFAULT_REGION = "BRA"
DEFAULT_VARIABLES: Tuple[str, ...] = ("tas", "pr", "hurs")
DEFAULT_SEASON = "annual"
DEFAULT_VERSION_TAG = "v100"
DEFAULT_BASELINE = (1995, 2014)
DEFAULT_FUTURE = (2081, 2100)
DEFAULT_OUTDIR = REPO_ROOT / "data" / "manuscript_figures" / "Figure3A_RegionalProjections"
DEFAULT_DPI = 300
DEFAULT_FIG_WIDTH = 7.2
TIME_ROW_HEIGHT = 1.0
BAR_ROW_HEIGHT = 1.12
PAIR_HSPACE = 0.46
GROUP_WSPACE = 0.05
BAR_XTICK_FONTSIZE = 5.0
BAR_LINEWIDTH = 0.55
BAR_ALPHA = 0.78
GCMAGICC_DELTA_OFFSET = -0.16
CMIP6_DELTA_OFFSET = 0.16
DELTA_CAP_HALF_WIDTH = 0.085


@dataclass(frozen=True)
class DeltaStats:
    scenario: str
    p5: float
    p50: float
    p95: float


@dataclass(frozen=True)
class VariablePanelSpec:
    variable: str
    season: str
    region: str
    label: str
    region_label: str
    units: str
    series: Mapping[str, object]
    delta_stats: Mapping[str, DeltaStats]
    cmip6_delta_stats: Mapping[str, DeltaStats]
    source_paths: Mapping[str, str]
    cmip6_source_paths: Mapping[str, str]
    cmip6_member_counts: Mapping[str, int]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_timetag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_csv_list(raw: str, *, field_name: str) -> Tuple[str, ...]:
    values = tuple(item.strip() for item in str(raw).split(",") if item.strip())
    if not values:
        raise ValueError(f"{field_name} must contain at least one value")
    return values


def _parse_period(raw: str, *, field_name: str) -> Tuple[int, int]:
    token = str(raw).strip()
    if "-" not in token:
        raise ValueError(f"{field_name} must have form YYYY-YYYY, got {raw!r}")
    left, right = token.split("-", 1)
    try:
        start = int(left)
        end = int(right)
    except ValueError as exc:
        raise ValueError(f"{field_name} must have integer years, got {raw!r}") from exc
    if start > end:
        raise ValueError(f"{field_name} start year must be <= end year, got {raw!r}")
    return start, end


def _safe_token(value: str) -> str:
    token = str(value).strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in token) or "value"


def _scenario_list(raw: Optional[str]) -> Tuple[str, ...]:
    if raw is None or str(raw).strip() == "":
        return tuple(FIG1021.PLOT_SCENARIOS)
    return _parse_csv_list(raw, field_name="scenarios")


def _candidate_815_run_roots(version_tag: str, timetag: Optional[str]) -> List[Path]:
    roots: List[Path] = []
    roots.extend(FIG1021._candidate_815_publish_roots(version_tag))
    roots.extend(FIG1021._candidate_815_run_roots(version_tag, timetag))
    return FIG1021._dedupe_paths(roots)


def _missing_artifacts(
    roots: Sequence[Path],
    *,
    variables: Sequence[str],
    season: str,
    region: str,
    scenarios: Sequence[str],
) -> Dict[str, List[str]]:
    missing: Dict[str, List[str]] = {}
    for scenario in scenarios:
        missing_specs: List[str] = []
        for variable in variables:
            path = FIG1021._find_scenario_file(roots, variable, season, region, scenario)
            if path is None:
                missing_specs.append(f"{variable}/{season}/{region}")
        if missing_specs:
            missing[str(scenario)] = missing_specs
    return missing


def _format_missing_message(
    roots: Sequence[Path],
    *,
    variables: Sequence[str],
    season: str,
    region: str,
    scenarios: Sequence[str],
    missing: Mapping[str, Sequence[str]],
) -> str:
    lines = [
        "Missing required 815 percentiles outputs for regional scenario-range figure.",
        f"Region: {region}",
        f"Season: {season}",
        f"Variables: {', '.join(variables)}",
        f"Scenarios requested: {', '.join(scenarios)}",
        "Missing:",
    ]
    for scenario in scenarios:
        specs = list(missing.get(str(scenario), []))
        if specs:
            lines.append(f"  - {scenario}: {', '.join(specs)}")
    lines.append("Candidate roots searched:")
    lines.extend([f"  - {path}" for path in roots] or ["  - (none)"])
    lines.extend(
        [
            "",
            "For the default BRA/tas-pr-hurs annual v100 figure, generate the missing inputs with:",
            "  pixi run python notebooks/815_simple_plot_SSPprojections.py --figure1025-defaults --resume",
            "Then re-run:",
            "  pixi run python notebooks/1025_FigureX_RegionalScenarioRange.py",
        ]
    )
    return "\n".join(lines)


def _load_payload_metadata(path: Path) -> Dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, Mapping):
        return {}
    meta = payload.get("meta")
    meta_map = meta if isinstance(meta, Mapping) else {}
    out = {
        "variable": str(payload.get("variable") or meta_map.get("variable") or ""),
        "season": str(payload.get("season") or meta_map.get("season") or ""),
        "region": str(payload.get("region") or meta_map.get("region") or ""),
        "scenario": str(payload.get("scenario") or meta_map.get("scenario") or ""),
        "units": str(payload.get("units") or payload.get("source_units") or ""),
        "long_name": str(payload.get("long_name") or payload.get("long_variable") or ""),
        "long_region": str(payload.get("long_region") or ""),
    }
    return {key: value for key, value in out.items() if value}


def _panel_label_from_metadata(variable: str, metadata: Mapping[str, str]) -> str:
    label = str(metadata.get("long_variable") or metadata.get("long_name") or "").strip()
    if label:
        return label
    return str(variable).strip()


def _region_label_from_metadata(region: str, metadata: Mapping[str, str]) -> str:
    label = str(metadata.get("long_region") or "").strip()
    return label or str(region).strip()


def _units_from_series(series_map: Mapping[str, object], metadata: Mapping[str, str]) -> str:
    units = str(metadata.get("units") or "").strip()
    if units:
        return units
    for series in series_map.values():
        value = str(getattr(series, "units", "") or "").strip()
        if value:
            return value
    return ""


def _compute_delta_stats(
    series_map: Mapping[str, object],
    *,
    baseline: Tuple[int, int],
    future: Tuple[int, int],
) -> Dict[str, DeltaStats]:
    stats: Dict[str, DeltaStats] = {}
    for scenario, series in series_map.items():
        years = np.asarray(getattr(series, "years"), dtype=int)
        low = np.asarray(getattr(series, "low"), dtype=float)
        med = np.asarray(getattr(series, "median"), dtype=float)
        high = np.asarray(getattr(series, "high"), dtype=float)

        base_low = FIG1021._period_mean(low, years, baseline)
        base_med = FIG1021._period_mean(med, years, baseline)
        base_high = FIG1021._period_mean(high, years, baseline)
        fut_low = FIG1021._period_mean(low, years, future)
        fut_med = FIG1021._period_mean(med, years, future)
        fut_high = FIG1021._period_mean(high, years, future)

        p5 = fut_low - base_low
        p50 = fut_med - base_med
        p95 = fut_high - base_high
        if p5 > p95:
            p5, p95 = p95, p5
        stats[str(scenario)] = DeltaStats(scenario=str(scenario), p5=float(p5), p50=float(p50), p95=float(p95))
    return stats


def _delta_stats_from_values(scenario: str, values: Sequence[float]) -> Optional[DeltaStats]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    p5, p50, p95 = np.percentile(arr, [5, 50, 95])
    return DeltaStats(scenario=str(scenario), p5=float(p5), p50=float(p50), p95=float(p95))


def _compute_member_delta_values(
    years: Sequence[int],
    members: Sequence[Mapping[str, object]],
    *,
    baseline: Tuple[int, int],
    future: Tuple[int, int],
) -> List[float]:
    year_arr = np.asarray(years, dtype=int)
    deltas: List[float] = []
    for member in members:
        values = np.asarray(member.get("values", []), dtype=float)
        if values.shape != year_arr.shape:
            continue
        delta = FIG1021._period_mean(values, year_arr, future) - FIG1021._period_mean(values, year_arr, baseline)
        if np.isfinite(delta):
            deltas.append(float(delta))
    return deltas


def _load_cmip6_delta_stats(
    percentiles_path: Path,
    scenario: str,
    *,
    baseline: Tuple[int, int],
    future: Tuple[int, int],
) -> Tuple[Optional[DeltaStats], Optional[Path], int]:
    sidecar_path = Path(percentiles_path).with_name("cmip6_members.json")
    if not sidecar_path.exists():
        return None, None, 0
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("Could not read CMIP6 sidecar %s (%s)", sidecar_path, exc)
        return None, sidecar_path, 0
    if not isinstance(payload, Mapping):
        return None, sidecar_path, 0
    members = payload.get("members")
    years = payload.get("years")
    if not isinstance(members, Sequence) or isinstance(members, (str, bytes)) or not isinstance(years, Sequence):
        return None, sidecar_path, 0
    member_maps = [m for m in members if isinstance(m, Mapping)]
    deltas = _compute_member_delta_values(years, member_maps, baseline=baseline, future=future)
    return _delta_stats_from_values(scenario, deltas), sidecar_path, len(deltas)


def _load_variable_panel_spec(
    roots: Sequence[Path],
    *,
    variable: str,
    season: str,
    region: str,
    scenarios: Sequence[str],
    baseline: Tuple[int, int],
    future: Tuple[int, int],
) -> VariablePanelSpec:
    series_map: Dict[str, object] = {}
    source_paths: Dict[str, str] = {}
    cmip6_delta_stats: Dict[str, DeltaStats] = {}
    cmip6_source_paths: Dict[str, str] = {}
    cmip6_member_counts: Dict[str, int] = {}
    first_metadata: Dict[str, str] = {}
    for scenario in scenarios:
        path = FIG1021._find_scenario_file(roots, variable, season, region, scenario)
        if path is None:
            raise FileNotFoundError(f"Missing {variable}/{season}/{region}/{scenario}/percentiles.json")
        if not first_metadata:
            first_metadata = _load_payload_metadata(path)
        series_map[str(scenario)] = FIG1021._load_series(path, str(scenario))
        source_paths[str(scenario)] = str(path)
        cmip6_stats, cmip6_path, cmip6_member_count = _load_cmip6_delta_stats(
            path,
            str(scenario),
            baseline=baseline,
            future=future,
        )
        if cmip6_path is not None:
            cmip6_source_paths[str(scenario)] = str(cmip6_path)
        if cmip6_stats is not None:
            cmip6_delta_stats[str(scenario)] = cmip6_stats
            cmip6_member_counts[str(scenario)] = cmip6_member_count

    label = _panel_label_from_metadata(variable, first_metadata)
    region_label = _region_label_from_metadata(region, first_metadata)
    units = _units_from_series(series_map, first_metadata)
    return VariablePanelSpec(
        variable=str(variable),
        season=str(season),
        region=str(region),
        label=label,
        region_label=region_label,
        units=units,
        series=series_map,
        delta_stats=_compute_delta_stats(series_map, baseline=baseline, future=future),
        cmip6_delta_stats=cmip6_delta_stats,
        source_paths=source_paths,
        cmip6_source_paths=cmip6_source_paths,
        cmip6_member_counts=cmip6_member_counts,
    )


def _timeseries_ylim(series_map: Mapping[str, object], scenarios: Sequence[str], xlim: Tuple[int, int]) -> Tuple[float, float]:
    present = [scenario for scenario in scenarios if scenario in series_map]
    if not present:
        return (0.0, 1.0)
    return FIG1021._timeseries_ylim(series_map, present, xlim)


def _row_timeseries_ylim(series_map: Mapping[str, object], scenarios: Sequence[str]) -> Tuple[float, float]:
    bounds: List[Tuple[float, float]] = []
    selected = {str(scenario) for scenario in scenarios}
    for group in FIG1021.ROW1_GROUPS:
        group_scenarios = [str(s) for s in group["scenarios"] if str(s) in selected]
        if not group_scenarios:
            continue
        bounds.append(_timeseries_ylim(series_map, group_scenarios, tuple(group["xlim"])))
    if not bounds:
        return (0.0, 1.0)
    return (min(bound[0] for bound in bounds), max(bound[1] for bound in bounds))


def _delta_ylim(
    stats: Mapping[str, DeltaStats],
    scenarios: Sequence[str],
    *,
    extra_stats: Sequence[Mapping[str, DeltaStats]] = (),
) -> Tuple[float, float]:
    values: List[float] = [0.0]
    for stats_map in (stats, *extra_stats):
        for scenario in scenarios:
            stat = stats_map.get(str(scenario))
            if stat is None:
                continue
            values.extend([stat.p5, stat.p50, stat.p95])
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (-1.0, 1.0)
    low = float(np.nanmin(arr))
    high = float(np.nanmax(arr))
    span = high - low
    pad = 0.10 * span if span > 0 else max(0.2, abs(high) * 0.10)
    return low - pad, high + pad


def _var_ylabel(spec: VariablePanelSpec) -> str:
    return f"{spec.label} ({spec.units})" if spec.units else spec.label


def _delta_ylabel(spec: VariablePanelSpec) -> str:
    return f"Delta {spec.variable} ({spec.units})" if spec.units else f"Delta {spec.variable}"


def _load_618_module():
    mod618 = FIG1021._load_module(FIG1021.SCRIPT_618, "mod618_for_1025")
    FIG1021._patch_618_colors(mod618)
    return mod618


def _draw_delta_whisker_dataset(
    ax: plt.Axes,
    *,
    stats: Mapping[str, DeltaStats],
    scenarios: Sequence[str],
    offset: float,
    marker: str,
    linewidth: float,
    alpha: float,
    mod618,
    zorder: float,
) -> None:
    for index, scenario in enumerate(scenarios):
        stat = stats.get(str(scenario))
        if stat is None:
            continue
        xpos = float(index) + float(offset)
        color = FIG1021._scenario_color(str(scenario), mod618)
        ax.vlines(xpos, stat.p5, stat.p95, color=color, lw=linewidth, alpha=alpha, zorder=zorder)
        ax.hlines(
            stat.p50,
            xpos - DELTA_CAP_HALF_WIDTH,
            xpos + DELTA_CAP_HALF_WIDTH,
            colors=color,
            lw=max(1.0, linewidth * 0.42),
            alpha=min(1.0, alpha + 0.15),
            zorder=zorder + 0.05,
        )
        if marker == "x":
            ax.scatter(xpos, stat.p50, marker=marker, color=color, s=17.0, linewidths=0.7, alpha=0.88, zorder=zorder + 0.1)
        else:
            ax.scatter(
                xpos,
                stat.p50,
                marker=marker,
                color=color,
                s=18.0,
                edgecolors="#ffffff",
                linewidths=0.35,
                alpha=0.95,
                zorder=zorder + 0.1,
            )


def _plot_delta_bar_panel(
    ax: plt.Axes,
    *,
    title: str,
    spec: VariablePanelSpec,
    scenarios: Sequence[str],
    baseline: Tuple[int, int],
    future: Tuple[int, int],
    mod618,
) -> None:
    _draw_delta_whisker_dataset(
        ax,
        stats=spec.delta_stats,
        scenarios=scenarios,
        offset=GCMAGICC_DELTA_OFFSET,
        marker="o",
        linewidth=4.0,
        alpha=0.62,
        mod618=mod618,
        zorder=2.0,
    )
    _draw_delta_whisker_dataset(
        ax,
        stats=spec.cmip6_delta_stats,
        scenarios=scenarios,
        offset=CMIP6_DELTA_OFFSET,
        marker="x",
        linewidth=2.4,
        alpha=0.55,
        mod618=mod618,
        zorder=2.4,
    )
    ax.axhline(0.0, color="#333333", linewidth=0.7, zorder=1)
    FIG1021._set_panel_title(ax, title)
    ax.set_ylabel(_delta_ylabel(spec))
    ax.set_xlim(-0.8, len(scenarios) - 0.2)
    ax.set_ylim(*_delta_ylim(spec.delta_stats, scenarios, extra_stats=(spec.cmip6_delta_stats,)))
    ax.grid(True, axis="y", alpha=0.28)
    FIG1021._apply_scenario_xtick_labels(
        ax,
        scenarios,
        mod618=mod618,
        rotation=90.0,
        fontsize=BAR_XTICK_FONTSIZE,
        horizontalalignment="right",
    )
    for tick in ax.get_xticklabels():
        tick.set_verticalalignment("top")
    ax.tick_params(axis="x", length=0, pad=2.0)
    handles = [
        Line2D([0], [0], marker="o", color="#666666", markerfacecolor="#666666", lw=0, markersize=4.0, label="GCMAGICC n20"),
    ]
    if spec.cmip6_delta_stats:
        handles.append(Line2D([0], [0], marker="x", color="#666666", lw=0, markersize=4.2, label="CMIP6"))
    handles.append(Line2D([0], [0], color="#666666", lw=1.4, marker="_", markersize=6.0, label="median + p5-p95"))
    ax.legend(
        handles=handles,
        loc="upper left",
        ncol=min(3, len(handles)),
        fontsize=5.2,
        frameon=False,
        handlelength=1.0,
        columnspacing=0.5,
        handletextpad=0.28,
        borderaxespad=0.15,
    )
    subtitle = f"{future[0]}-{future[1]} minus {baseline[0]}-{baseline[1]}"
    ax.text(
        1.0,
        1.01,
        subtitle,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.2,
        color="#555555",
    )


def _rebuild_timeseries_legend(ax: plt.Axes, *, scenarios: Sequence[str]) -> None:
    handles = [
        Patch(facecolor="#bfbfbf", edgecolor="none", alpha=0.35, label="p5-p95"),
        Line2D([0], [0], color="#7f7f7f", linewidth=FIG1021.LINEWIDTH_MAIN, label="median"),
    ]
    if any(
        getattr(series, "era5_years", None) is not None and getattr(series, "era5_values", None) is not None
        for series in getattr(ax, "_scenario_series_map", {}).values()
    ):
        handles.append(Line2D([0], [0], color="#111111", linewidth=FIG1021.LINEWIDTH_EMPHASIS, label="ERA5"))
    _ = scenarios
    ax.legend(
        handles=handles,
        loc="upper left",
        ncol=len(handles),
        fontsize=5.4,
        frameon=False,
        handlelength=1.0,
        columnspacing=0.5,
        handletextpad=0.25,
    )


def _draw_empty_group_axis(ax: plt.Axes, title: str) -> None:
    FIG1021._set_panel_title(ax, title)
    ax.set_axis_off()
    ax.text(
        0.5,
        0.5,
        "No selected scenarios",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.0,
        color="#666666",
    )


def _build_figure(
    specs: Sequence[VariablePanelSpec],
    *,
    scenarios: Sequence[str],
    baseline: Tuple[int, int],
    future: Tuple[int, int],
    dpi: int,
    mod618,
) -> Tuple[plt.Figure, Dict[str, object]]:
    row_count = 2 * len(specs)
    height_ratios: List[float] = []
    for _spec in specs:
        height_ratios.extend([TIME_ROW_HEIGHT, BAR_ROW_HEIGHT])
    fig_height = max(6.0, 2.05 * len(specs) + 3.8)
    fig = plt.figure(figsize=(DEFAULT_FIG_WIDTH, fig_height), dpi=dpi)
    gs = fig.add_gridspec(row_count, 1, height_ratios=height_ratios, hspace=PAIR_HSPACE)

    panel_meta: Dict[str, object] = {
        "variables": [],
        "baseline": list(baseline),
        "future": list(future),
        "scenario_order": list(scenarios),
    }
    letters = "abcdefghijklmnopqrstuvwxyz"

    for spec_index, spec in enumerate(specs):
        time_row = 2 * spec_index
        delta_row = time_row + 1
        time_panel_letter = letters[time_row] if time_row < len(letters) else f"r{time_row + 1}"
        delta_panel_letter = letters[delta_row] if delta_row < len(letters) else f"r{delta_row + 1}"

        row_ylim = _row_timeseries_ylim(spec.series, scenarios)
        row1_widths = [int(group["xlim"][1]) - int(group["xlim"][0]) for group in FIG1021.ROW1_GROUPS]
        row_gs = gs[time_row, 0].subgridspec(1, len(FIG1021.ROW1_GROUPS), width_ratios=row1_widths, wspace=GROUP_WSPACE)
        row_axes: List[plt.Axes] = []
        legend_added = spec_index != 0
        for group_index, group in enumerate(FIG1021.ROW1_GROUPS):
            group_scenarios = [str(s) for s in group["scenarios"] if str(s) in scenarios]
            ax = fig.add_subplot(row_gs[0, group_index], sharey=row_axes[0] if row_axes else None)
            row_axes.append(ax)
            group_title = f"{time_panel_letter}{group_index + 1} {spec.region_label} {spec.variable} {group['key']}"
            if not group_scenarios:
                _draw_empty_group_axis(ax, group_title)
                continue
            FIG1021._plot_timeseries_panel(
                ax,
                title=group_title,
                ylabel=_var_ylabel(spec),
                series_map=spec.series,
                scenario_subset=group_scenarios,
                xlim=tuple(group["xlim"]),
                show_ylabel=(group_index == 0),
                show_legend=False,
                ylim=row_ylim,
                show_y_grid=True,
                y_grid_positions=None,
                legend_style="summary_range",
                background_band=None,
                background_band_label=None,
                mod618=mod618,
            )
            ax.set_xlabel("")
            if group_index > 0:
                ax.tick_params(axis="y", left=False, labelleft=False, length=0.0)
                FIG1021._blank_xtick_label(ax, float(group["xlim"][0]))
            if not legend_added:
                setattr(ax, "_scenario_series_map", spec.series)
                _rebuild_timeseries_legend(ax, scenarios=group_scenarios)
                legend_added = True

        ax_delta = fig.add_subplot(gs[delta_row, 0])
        _plot_delta_bar_panel(
            ax_delta,
            title=f"{delta_panel_letter} {spec.region_label} {spec.variable} change",
            spec=spec,
            scenarios=scenarios,
            baseline=baseline,
            future=future,
            mod618=mod618,
        )

        panel_meta["variables"].append(
            {
                "variable": spec.variable,
                "season": spec.season,
                "region": spec.region,
                "label": spec.label,
                "region_label": spec.region_label,
                "units": spec.units,
                "source_paths": dict(spec.source_paths),
                "cmip6_sidecars": {
                    "source_paths": dict(spec.cmip6_source_paths),
                    "member_counts": dict(spec.cmip6_member_counts),
                    "available_scenarios": [str(s) for s in scenarios if str(s) in spec.cmip6_delta_stats],
                    "missing_scenarios": [str(s) for s in scenarios if str(s) not in spec.cmip6_delta_stats],
                },
                "delta_stats": {
                    scenario: {
                        "p5": stat.p5,
                        "p50": stat.p50,
                        "p95": stat.p95,
                    }
                    for scenario, stat in spec.delta_stats.items()
                },
                "cmip6_delta_stats": {
                    scenario: {
                        "p5": stat.p5,
                        "p50": stat.p50,
                        "p95": stat.p95,
                    }
                    for scenario, stat in spec.cmip6_delta_stats.items()
                },
            }
        )

    fig.subplots_adjust(left=0.12, right=0.95, top=0.985, bottom=0.08)
    return fig, panel_meta


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a regional multi-variable scenario range figure from 815 outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--region", default=DEFAULT_REGION, help="ISO3 or existing 815/IPCC AR6 region path token.")
    parser.add_argument("--variables", default=",".join(DEFAULT_VARIABLES), help="Comma-separated variable list.")
    parser.add_argument("--season", default=DEFAULT_SEASON, help="Season token used in 815 output paths.")
    parser.add_argument("--version-tag", default=DEFAULT_VERSION_TAG, help="Versioned 815 output family to read.")
    parser.add_argument("--timetag", default=None, help="Optional 815 timetag to prefer.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory.")
    parser.add_argument("--baseline", default=f"{DEFAULT_BASELINE[0]}-{DEFAULT_BASELINE[1]}", help="Baseline period.")
    parser.add_argument("--future", default=f"{DEFAULT_FUTURE[0]}-{DEFAULT_FUTURE[1]}", help="Future period.")
    parser.add_argument("--scenarios", default=None, help="Comma-separated scenarios; default uses 1021 scenario order.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Output raster DPI.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    started = perf_counter()

    variables = _parse_csv_list(args.variables, field_name="variables")
    scenarios = _scenario_list(args.scenarios)
    baseline = _parse_period(args.baseline, field_name="baseline")
    future = _parse_period(args.future, field_name="future")
    region = str(args.region).strip()
    season = str(args.season).strip()
    version_tag = str(args.version_tag).strip()
    if not region:
        raise ValueError("--region must not be empty")
    if not season:
        raise ValueError("--season must not be empty")
    if not version_tag:
        raise ValueError("--version-tag must not be empty")

    roots = _candidate_815_run_roots(version_tag, args.timetag)
    missing = _missing_artifacts(
        roots,
        variables=variables,
        season=season,
        region=region,
        scenarios=scenarios,
    )
    if missing:
        raise FileNotFoundError(
            _format_missing_message(
                roots,
                variables=variables,
                season=season,
                region=region,
                scenarios=scenarios,
                missing=missing,
            )
        )

    LOG.info("Using 815 roots: %s", [str(path) for path in roots])
    specs = [
        _load_variable_panel_spec(
            roots,
            variable=variable,
            season=season,
            region=region,
            scenarios=scenarios,
            baseline=baseline,
            future=future,
        )
        for variable in variables
    ]

    mod618 = _load_618_module()
    fig, panel_meta = _build_figure(
        specs,
        scenarios=scenarios,
        baseline=baseline,
        future=future,
        dpi=int(args.dpi),
        mod618=mod618,
    )

    outdir = Path(args.outdir).expanduser().resolve(strict=False)
    outdir.mkdir(parents=True, exist_ok=True)
    timetag = _default_timetag()
    stem = "FigureX_RegionalScenarioRange_{region}_{vars}_{version}_{timetag}".format(
        region=_safe_token(region),
        vars="-".join(_safe_token(v) for v in variables),
        version=_safe_token(version_tag),
        timetag=timetag,
    )
    png_path = outdir / f"{stem}.png"
    pdf_path = outdir / f"{stem}.pdf"
    meta_path = outdir / f"{stem}.json"
    fig.savefig(png_path, dpi=int(args.dpi))
    fig.savefig(pdf_path)
    plt.close(fig)

    metadata = {
        "generated_at_utc": _utc_now_iso(),
        "script": str(HERE),
        "version_tag": version_tag,
        "timetag_preference": args.timetag,
        "region": region,
        "variables": list(variables),
        "season": season,
        "baseline": list(baseline),
        "future": list(future),
        "scenario_order": list(scenarios),
        "roots": [str(path) for path in roots],
        "missing_artifacts": missing,
        "panel_meta": panel_meta,
        "outputs": {
            "png": str(png_path),
            "pdf": str(pdf_path),
            "metadata": str(meta_path),
        },
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    LOG.info("Wrote %s", png_path)
    LOG.info("Wrote %s", pdf_path)
    LOG.info("Wrote %s", meta_path)


if __name__ == "__main__":
    main()
