#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the seven-panel corrected Figure 5 drought synthesis.

The layout restores the useful visual grammar of the original 1040 workflow
(bordered map, separate factual/natural SPEI-48 series, and recent/future
histograms) while taking every plotted time series and attribution statistic
from the corrected common-protocol release.  The two former risk panels are
replaced by the corrected GCMagicc event-probability and three-SMILE panels.

With ``--extract-era5``, the script freezes the small ERA5 event-map and
Natural Earth boundary artifact required for standalone figure reproduction.
The normal plotting path reads only files committed to this repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data" / "derived" / "iran_drought_attribution"
DEFAULT_SERIES = DATA_ROOT / "drought_common_protocol_series_GCMagicc_v100_and_SMILEs.csv"
DEFAULT_SUMMARY = DATA_ROOT / "drought_common_protocol_summary_GCMagicc_v100_and_SMILEs.csv"
DEFAULT_MAP = DATA_ROOT / "era5_irn_penman_monteith_spei48_map.json"
DEFAULT_CMIP6_SIDECAR = DATA_ROOT / "cmip6_irn_penman_monteith_spei48_sidecar.json"
DEFAULT_OUTPUT = ROOT / "figures" / "archive" / "iran_drought_attribution"
STEM = "iran_drought_attribution_synthesis_common_protocol"

PET_METHODS = ("thornthwaite", "hargreaves", "penman-monteith")
PET_LABELS = ("Thornthwaite", "Modified\nHargreaves", "Penman–\nMonteith")
MODELS = ("CanESM5", "MIROC6", "GISS-E2-1-G")
RECENT = (2021, 2025)
FUTURE = (2041, 2060)
PRIMARY_BASELINE = (1991, 2010)
SMILE_WINDOW = (1995, 2014)

FACTUAL = "#D18F35"
NATURAL = "#4C90C0"
ERA5 = "#111111"
MODEL_COLORS = {"CanESM5": "#228833", "MIROC6": "#CCBB44", "GISS-E2-1-G": "#EE6677"}
CMIP6_SIDECAR = "#A31849"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: object) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except (TypeError, ValueError):
        return np.nan


def grid_edges(centres: np.ndarray) -> np.ndarray:
    centres = np.asarray(centres, dtype=float)
    if centres.size < 2:
        return np.array([centres[0] - 0.5, centres[0] + 0.5])
    mids = 0.5 * (centres[:-1] + centres[1:])
    return np.concatenate(([centres[0] - (mids[0] - centres[0])], mids, [centres[-1] + (centres[-1] - mids[-1])]))


def geometry_lines(geometry: object) -> list[list[list[float]]]:
    """Return exterior coordinate arrays from Polygon-like geometries."""
    if getattr(geometry, "is_empty", True):
        return []
    if hasattr(geometry, "geoms"):
        lines: list[list[list[float]]] = []
        for part in geometry.geoms:
            lines.extend(geometry_lines(part))
        return lines
    exterior = getattr(geometry, "exterior", None)
    if exterior is None:
        return []
    return [[[float(x), float(y)] for x, y in exterior.coords]]


def extract_map_artifact(era5_file: Path, output: Path) -> None:
    """Freeze the corrected ERA5 map plus clipped Natural Earth boundaries."""
    workflow = importlib.import_module("gcmagicc_eval.workflows.1090_drought_common_protocol")
    region = workflow.build_region(era5_file)
    years, grid_raw, _regional_raw = workflow._era5_raw(era5_file, region, "unadjusted", "penman-monteith")
    grid = workflow._transform_era5(grid_raw[0], years, PRIMARY_BASELINE)
    regional = workflow._aggregate_grid(grid[None, ...], region.weights)[0]
    event_indices = np.where(years == 2025)[0]
    if event_indices.size != 1:
        raise RuntimeError("ERA5 December 2025 event is missing")
    event_index = int(event_indices[0])

    import regionmask
    from shapely.geometry import box

    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
    bbox = box(float(region.lon.min()) - 4.0, float(region.lat.min()) - 3.0, float(region.lon.max()) + 4.0, float(region.lat.max()) + 3.0)
    iran_index = int(countries.map_keys("IRN"))
    boundaries: list[dict[str, object]] = []
    for index, polygon in enumerate(countries.polygons):
        clipped = polygon.intersection(bbox)
        lines = geometry_lines(clipped)
        if lines:
            boundaries.append({"iso3": "IRN" if index == iran_index else "", "lines": lines})

    event_grid = np.full(region.mask.shape, np.nan, dtype=float)
    event_grid[region.mask] = grid[event_index]
    source_stat = era5_file.stat()
    payload = {
        "schema": "gcmagicc-era5-irn-event-map/v1",
        "method": "Penman–Monteith SPEI-48; area-weighted mean of grid-cell-standardized SPEI",
        "baseline": list(PRIMARY_BASELINE),
        "event": "December 2025",
        "region": "IRN",
        "region_mask": "Natural Earth v5.0.0 countries_110; ISO3 IRN",
        "source": {
            "filename": era5_file.name,
            "bytes": source_stat.st_size,
            "sha256": sha256(era5_file),
        },
        "lat": region.lat.astype(float).tolist(),
        "lon": region.lon.astype(float).tolist(),
        "mask": region.mask.astype(int).tolist(),
        "spei48_december_2025": [[None if not np.isfinite(value) else float(value) for value in row] for row in event_grid],
        "area_weighted_series": {
            "years": years.astype(int).tolist(),
            "values": [None if not np.isfinite(value) else float(value) for value in regional],
        },
        "event_threshold": float(regional[event_index]),
        "boundaries": boundaries,
        "extraction": {
            "workflow": "src/gcmagicc_eval/workflows/1090_drought_common_protocol.py",
            "command": "python src/gcmagicc_eval/workflows/1130_drought_attribution_synthesis.py --extract-era5 /path/to/ERA5.nc",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


def extract_cmip6_sidecar(source: Path, output: Path) -> None:
    """Reduce the former 1040 CMIP6 panel-I sidecar to regional lines only."""
    payload = json.loads(source.read_text(encoding="utf-8"))

    def reduce_series(items: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
        reduced: list[dict[str, object]] = []
        for item in items:
            values = np.asarray(item["values"], dtype=float)
            regional = values if values.ndim == 1 else np.nanmean(values, axis=1)
            times = np.asarray(item.get("time", item.get("years", [])), dtype=float)
            if regional.size != times.size:
                raise ValueError(f"CMIP6 sidecar length mismatch for {item.get('label')}")
            reduced.append(
                {
                    "label": str(item.get("label", "CMIP6")),
                    "time": times.tolist(),
                    "regional_mean": [None if not np.isfinite(value) else float(value) for value in regional],
                }
            )
        return reduced

    series = payload["series"]
    artifact = {
        "schema": "gcmagicc-cmip6-drought-sidecar/v1",
        "purpose": "thin contextual CMIP6 lines in Figure 5 panels b and c; not used in corrected attribution statistics",
        "pet_method": payload["pet_method"],
        "region": payload["region"],
        "source": {
            "path_tail": "/".join(source.parts[-7:]),
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
            "generated_at": payload.get("generated_at"),
            "timetag": payload.get("timetag"),
            "baseline_strategy": "historical:per_member (legacy 1040 visual context)",
        },
        "factual": reduce_series([*series.get("cmip6_hist", []), *series.get("cmip6_ssp245", [])]),
        "natural": reduce_series(series.get("cmip6_hist_nat", [])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


def load_series(path: Path) -> dict[str, dict[int, list[tuple[int, float]]]]:
    groups: dict[str, dict[int, list[tuple[int, float]]]] = {
        "factual": defaultdict(list),
        "natural": defaultdict(list),
    }
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["source"] != "GCMagicc" or row["pet_method"] != "penman-monteith":
                continue
            forcing = row["forcing"]
            if forcing not in groups:
                continue
            groups[forcing][int(row["member_index"])].append((int(row["year"]), float(row["december_spei48"])))
    for members in groups.values():
        for points in members.values():
            points.sort()
    return groups


def load_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def primary_ratio_rows(rows: Iterable[Mapping[str, str]]) -> list[Mapping[str, str]]:
    return [
        row
        for row in rows
        if row["record_type"] == "probability_ratio"
        and int(row["baseline_start"]) == PRIMARY_BASELINE[0]
        and row["rsds_treatment"] == "adjusted"
        and row["aggregation"] == "gridcell-spei-area-mean"
        and int(row["window_start"]) in {RECENT[0], SMILE_WINDOW[0]}
    ]


def ensemble_matrix(members: Mapping[int, list[tuple[int, float]]]) -> tuple[np.ndarray, np.ndarray]:
    years = np.array(sorted(set(year for points in members.values() for year, _ in points)), dtype=int)
    index = {year: position for position, year in enumerate(years)}
    matrix = np.full((len(members), len(years)), np.nan, dtype=float)
    for row_index, member in enumerate(sorted(members)):
        for year, value in members[member]:
            matrix[row_index, index[year]] = value
    return years, matrix


def panel_label(ax: plt.Axes, label: str, *, inside: bool = False) -> None:
    if inside:
        ax.text(
            0.02,
            0.94,
            label,
            transform=ax.transAxes,
            fontsize=10.0,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
            zorder=10,
        )
    else:
        ax.text(-0.055, 1.04, label, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="top", ha="left")


def plot_map(ax: plt.Axes, cax: plt.Axes, payload: Mapping[str, object]) -> None:
    lat = np.asarray(payload["lat"], dtype=float)
    lon = np.asarray(payload["lon"], dtype=float)
    values = np.array([[np.nan if value is None else value for value in row] for row in payload["spei48_december_2025"]], dtype=float)
    mesh = ax.pcolormesh(grid_edges(lon), grid_edges(lat), values, cmap="BrBG", vmin=-3, vmax=3, shading="flat", rasterized=True)
    for boundary in payload["boundaries"]:
        highlight = boundary["iso3"] == "IRN"
        for line in boundary["lines"]:
            coords = np.asarray(line, dtype=float)
            ax.plot(coords[:, 0], coords[:, 1], color="#111111" if highlight else "#666666", lw=1.15 if highlight else 0.45, zorder=4)
    ax.text(53.5, 32.2, "Iran", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#111111", zorder=5)
    ax.set_facecolor("#EAF4F8")
    ax.set_xlim(float(lon.min()) - 3.2, float(lon.max()) + 3.2)
    ax.set_ylim(float(lat.min()) - 2.2, float(lat.max()) + 2.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylabel("Latitude")
    ax.set_title("ERA5 December 2025\nPenman–Monteith SPEI-48", loc="left", fontsize=9.5, pad=4)
    cbar = plt.colorbar(mesh, cax=cax, orientation="horizontal")
    cbar.set_label("SPEI-48", labelpad=1)
    cbar.ax.tick_params(labelsize=7)


def plot_series_panel(
    ax: plt.Axes,
    members: Mapping[int, list[tuple[int, float]]],
    *,
    forcing_label: str,
    color: str,
    era5_years: np.ndarray,
    era5_values: np.ndarray,
    threshold: float,
    cmip6_sidecar: Iterable[Mapping[str, object]],
) -> None:
    years, matrix = ensemble_matrix(members)
    for row in matrix:
        ax.plot(years, row, color=color, alpha=0.055, lw=0.35, rasterized=True)
    q05, q50, q95 = np.nanquantile(matrix, [0.05, 0.50, 0.95], axis=0)
    ax.fill_between(years, q05, q95, color=color, alpha=0.16, linewidth=0)
    ax.plot(years, q50, color=color, lw=1.5, label=f"GCMagicc median ({forcing_label})")
    first_cmip6 = True
    for item in cmip6_sidecar:
        sidecar_time = np.asarray(item["time"], dtype=float)
        sidecar_values = np.asarray(
            [np.nan if value is None else value for value in item["regional_mean"]],
            dtype=float,
        )
        ax.plot(
            sidecar_time,
            sidecar_values,
            color=CMIP6_SIDECAR,
            alpha=0.24,
            lw=0.28,
            label="CMIP6 sidecar" if first_cmip6 else None,
            zorder=3,
        )
        first_cmip6 = False
    ax.plot(era5_years, era5_values, color=ERA5, lw=1.25, label="ERA5", zorder=5)
    ax.axhline(threshold, color=ERA5, lw=0.9, ls=":", label="ERA5 Dec 2025 threshold")
    ax.axvline(2025, color="0.55", lw=0.55, ls="--")
    ax.set_xlim(1940, 2100)
    ax.set_ylim(-4.2, 4.2)
    ax.set_ylabel("December SPEI-48")
    ax.set_title(f"GCMagicc {forcing_label}", loc="left", fontsize=10.2, pad=5)
    ax.grid(axis="y", color="0.88", lw=0.5)
    ax.legend(loc="upper left", ncol=4, fontsize=6.8, frameon=False, handlelength=2.2, columnspacing=1.0)


def window_values(members: Mapping[int, list[tuple[int, float]]], period: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        [value for points in members.values() for year, value in points if period[0] <= year <= period[1]],
        dtype=float,
    )


def plot_histogram(ax: plt.Axes, groups: Mapping[str, Mapping[int, list[tuple[int, float]]]], period: tuple[int, int], threshold: float) -> None:
    bins = np.arange(-4.2, 3.61, 0.22)
    factual = window_values(groups["factual"], period)
    natural = window_values(groups["natural"], period)
    ax.hist(natural, bins=bins, density=True, color=NATURAL, alpha=0.48, label="SSP2-4.5-nat")
    ax.hist(factual, bins=bins, density=True, histtype="step", color=FACTUAL, lw=1.25, label="SSP2-4.5")
    ax.axvline(threshold, color=ERA5, lw=1.15, ls=":", label="ERA5 Dec 2025")
    ax.set_xlim(-4.2, 3.5)
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution, {period[0]}–{period[1]}", loc="left", fontsize=9.0, pad=4)
    ax.grid(axis="y", color="0.9", lw=0.45)


def plot_probabilities(
    ax: plt.Axes,
    rows: list[Mapping[str, str]],
    all_summary_rows: list[Mapping[str, str]],
) -> None:
    gcm = [row for row in rows if row["source"] == "GCMagicc"]
    x = np.arange(len(PET_METHODS))
    factual = [finite(next(row["factual_probability"] for row in gcm if row["pet_method"] == method)) for method in PET_METHODS]
    natural = [finite(next(row["natural_probability"] for row in gcm if row["pet_method"] == method)) for method in PET_METHODS]
    natural_upper = [
        finite(
            next(
                row["upper"]
                for row in all_summary_rows
                if row["record_type"] == "probability"
                and row["source"] == "GCMagicc"
                and row["forcing"] == "natural"
                and row["pet_method"] == method
                and int(row["baseline_start"]) == PRIMARY_BASELINE[0]
                and row["rsds_treatment"] == "adjusted"
                and row["aggregation"] == "gridcell-spei-area-mean"
                and int(row["window_start"]) == RECENT[0]
            )
        )
        for method in PET_METHODS
    ]
    width = 0.36
    ax.bar(x - width / 2, factual, width=width, color="#4477AA", label="Factual")
    ax.bar(x + width / 2, natural, width=width, color="#BBBBBB", label="Natural-only")
    ax.scatter(x + width / 2, natural_upper, marker="v", color="black", s=24, label="95% upper bound (zero count)", zorder=4)
    ax.set_xticks(x, PET_LABELS)
    ax.set_ylabel("Probability")
    ax.set_title(r"$\mathbf{f}$   GCMagicc event probability, 2021–2025", loc="left", fontsize=10.2, pad=5)
    ax.legend(fontsize=7.2, frameon=False)
    ax.grid(axis="y", color="0.9", lw=0.5)


def plot_smile_ratios(ax: plt.Axes, rows: list[Mapping[str, str]]) -> None:
    x = np.arange(len(PET_METHODS))
    width = 0.24
    maximum = 0.0
    for model_index, model in enumerate(MODELS):
        model_rows = [row for row in rows if row["source"] == model]
        values: list[float] = []
        bounded: list[bool] = []
        for method in PET_METHODS:
            row = next(row for row in model_rows if row["pet_method"] == method)
            estimate = finite(row["estimate"])
            bound = finite(row.get("one_sided_point_ratio_bound"))
            values.append(estimate if np.isfinite(estimate) else bound)
            bounded.append(not np.isfinite(estimate) and np.isfinite(bound))
        positions = x + (model_index - 1) * width
        ax.bar(positions, values, width=width, label=model, color=MODEL_COLORS[model])
        for xpos, value, one_sided in zip(positions, values, bounded, strict=True):
            if one_sided:
                ax.scatter([xpos], [value], marker="^", color="black", s=20, zorder=4)
        maximum = max(maximum, max(value for value in values if np.isfinite(value)))
    ax.scatter([], [], marker="^", color="black", s=20, label="Finite point bound (zero natural events)")
    ax.axhline(1.0, color="0.3", lw=0.8, ls=":")
    ax.set_xticks(x, PET_LABELS)
    ax.set_ylim(0, maximum * 1.18)
    ax.set_ylabel("Probability ratio")
    ax.set_title(r"$\mathbf{g}$   CMIP6 SMILE comparison, 1995–2014", loc="left", fontsize=10.2, pad=5)
    ax.legend(fontsize=7.0, frameon=False, ncol=2)
    ax.grid(axis="y", color="0.9", lw=0.5)


def make_figure(
    series_path: Path,
    summary_path: Path,
    map_path: Path,
    cmip6_sidecar_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    groups = load_series(series_path)
    summary_rows = load_summary(summary_path)
    ratios = primary_ratio_rows(summary_rows)
    map_payload = json.loads(map_path.read_text(encoding="utf-8"))
    cmip6_sidecar = json.loads(cmip6_sidecar_path.read_text(encoding="utf-8"))
    era5_years = np.asarray(map_payload["area_weighted_series"]["years"], dtype=int)
    era5_values = np.asarray([np.nan if value is None else value for value in map_payload["area_weighted_series"]["values"]], dtype=float)
    threshold = float(map_payload["event_threshold"])

    mpl.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(13.2, 8.45))
    outer = GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[1.0, 1.0, 0.92],
        width_ratios=[1.16, 1.0, 1.0, 1.0],
        left=0.065,
        right=0.99,
        bottom=0.075,
        top=0.965,
        wspace=0.34,
        hspace=0.29,
    )
    map_grid = outer[0, 0].subgridspec(2, 1, height_ratios=[1.0, 0.075], hspace=0.23)
    ax_map = fig.add_subplot(map_grid[0, 0])
    cax_map = fig.add_subplot(map_grid[1, 0])
    ax_factual = fig.add_subplot(outer[0, 1:])
    hist_grid = outer[1, 0].subgridspec(2, 1, hspace=0.50)
    ax_hist_recent = fig.add_subplot(hist_grid[0, 0])
    ax_hist_future = fig.add_subplot(hist_grid[1, 0])
    ax_natural = fig.add_subplot(outer[1, 1:])
    ax_probability = fig.add_subplot(outer[2, :2])
    ax_smile = fig.add_subplot(outer[2, 2:])

    plot_map(ax_map, cax_map, map_payload)
    plot_series_panel(
        ax_factual,
        groups["factual"],
        forcing_label="SSP2-4.5",
        color=FACTUAL,
        era5_years=era5_years,
        era5_values=era5_values,
        threshold=threshold,
        cmip6_sidecar=cmip6_sidecar["factual"],
    )
    ax_factual.tick_params(axis="x", labelbottom=False)
    plot_series_panel(
        ax_natural,
        groups["natural"],
        forcing_label="SSP2-4.5-nat",
        color=NATURAL,
        era5_years=era5_years,
        era5_values=era5_values,
        threshold=threshold,
        cmip6_sidecar=cmip6_sidecar["natural"],
    )
    ax_natural.set_xlabel("Year")
    plot_histogram(ax_hist_recent, groups, RECENT, threshold)
    plot_histogram(ax_hist_future, groups, FUTURE, threshold)
    ax_hist_recent.tick_params(axis="x", labelbottom=False)
    ax_hist_future.set_xlabel("December SPEI-48")
    handles, labels = ax_hist_recent.get_legend_handles_labels()
    ax_hist_recent.legend(handles, labels, fontsize=6.3, frameon=False, loc="upper right")
    plot_probabilities(ax_probability, ratios, summary_rows)
    plot_smile_ratios(ax_smile, ratios)

    for label, ax in zip("abc", (ax_map, ax_factual, ax_natural), strict=True):
        panel_label(ax, label)
    panel_label(ax_hist_recent, "d", inside=True)
    panel_label(ax_hist_future, "e", inside=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{STEM}.pdf"
    png = output_dir / f"{STEM}.png"
    metadata = {
        "Title": "Corrected Iranian drought attribution synthesis",
        "Author": "Malte Meinshausen and GCMagicc evaluation suite contributors",
        "Creator": "GCMagicc v1.0.1 reproducibility release",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04, metadata=metadata)
    fig.savefig(png, dpi=220, bbox_inches="tight", pad_inches=0.04, metadata={"Software": metadata["Creator"]})
    plt.close(fig)

    sidecar: dict[str, object] = {
        "schema": "gcmagicc-drought-attribution-synthesis/v2",
        "design_provenance": {
            "former_layout": "gcmmagicc/notebooks/1040_drought_attribution_example_country.py",
            "former_layout_revision": "2b9bd0a",
            "scientific_results": "corrected common-protocol release generated by 1090_drought_common_protocol.py",
        },
        "panels": {
            "a": "ERA5 December 2025 Penman–Monteith SPEI-48 map with Natural Earth country boundaries",
            "b": "GCMagicc ssp245 ensemble plus legacy CMIP6 sidecar context",
            "c": "GCMagicc ssp245-nat ensemble plus legacy CMIP6 hist-nat sidecar context",
            "d": "GCMagicc factual/natural distribution during 2021–2025",
            "e": "GCMagicc factual/natural distribution during 2041–2060",
            "f": "corrected GCMagicc event probabilities by PET method",
            "g": "corrected three-SMILE probability ratios by PET method",
        },
        "method": {
            "pet": "Penman–Monteith for panels a–e; three PET methods in panels f–g",
            "baseline": list(PRIMARY_BASELINE),
            "aggregation": "area-weighted mean of grid-cell-standardized December SPEI-48",
            "threshold": threshold,
        },
        "inputs": [
            {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (series_path, summary_path, map_path, cmip6_sidecar_path)
        ],
        "outputs": {pdf.name: sha256(pdf), png.name: sha256(png)},
    }
    sidecar_path = output_dir / f"{STEM}.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(sidecar_path)
    return sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, default=DEFAULT_SERIES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--map-json", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--cmip6-sidecar", type=Path, default=DEFAULT_CMIP6_SIDECAR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--extract-era5", type=Path, help="Extract and freeze the small map artifact from this ERA5 NetCDF")
    parser.add_argument("--extract-cmip6-sidecar", type=Path, help="Reduce the former 1040 panel-I CMIP6 sidecar")
    args = parser.parse_args()
    if args.extract_era5:
        extract_map_artifact(args.extract_era5, args.map_json)
        return 0
    if args.extract_cmip6_sidecar:
        extract_cmip6_sidecar(args.extract_cmip6_sidecar, args.cmip6_sidecar)
        return 0
    make_figure(args.series, args.summary, args.map_json, args.cmip6_sidecar, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
