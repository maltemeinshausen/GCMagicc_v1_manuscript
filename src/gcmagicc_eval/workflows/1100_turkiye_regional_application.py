#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the 3x3 Türkiye multi-variable scenario application.

Rows show tas, pr, and hurs. Columns separate CMIP6 SSPs; NDC,
SSP2-com, and current-policy pathways; and CMIP7 scenarios. The input files
are frozen annual regional percentiles from the v1.0.1 projection workflow.
This release-native plotter deliberately imports nothing from the development
repositories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
INPUT_ROOT = ROOT / "data" / "derived" / "turkiye_regional_application" / "inputs"
DEFAULT_OUTPUT = ROOT / "figures" / "turkiye_regional_application"

SCENARIO_GROUPS = {
    "cmip6_ssps": (
        "ssp119",
        "ssp126",
        "ssp245",
        "ssp370",
        "ssp434",
        "ssp460",
        "ssp534-over",
        "ssp585",
    ),
    "ndc_ssp2com_current_policy": (
        "SSP2-com",
        "NDC-submitted-low",
        "NDC-submitted-high",
        "NDC-PA-allNDCs-30June26-lower",
        "NDC-PA-allNDCs-30June26-upper",
        "NDC-PA-wo-USA-30June26-lower",
        "NDC-PA-wo-USA-30June26-upper",
        "Current-Policies-GCAM",
        "Current-Policies-MESSAGE",
        "Current-Policies-REMIND",
    ),
    "cmip7": ("VL", "LN", "L", "ML", "M", "HL", "H"),
}
GROUP_TITLES = {
    "cmip6_ssps": "CMIP6 SSP scenarios",
    "ndc_ssp2com_current_policy": "NDC, SSP2-com and current policies",
    "cmip7": "CMIP7 scenarios",
}
COLORS = {
    "ssp119": "#00a9cf",
    "ssp126": "#003466",
    "ssp245": "#f69320",
    "ssp370": "#df0000",
    "ssp434": "#2274a5",
    "ssp460": "#8f6bb3",
    "ssp534-over": "#9d5a6c",
    "ssp585": "#980002",
    "SSP2-com": "#4d4d4d",
    "NDC-submitted-low": "#1b9e77",
    "NDC-submitted-high": "#66a61e",
    "NDC-PA-allNDCs-30June26-lower": "#1f78b4",
    "NDC-PA-allNDCs-30June26-upper": "#6baed6",
    "NDC-PA-wo-USA-30June26-lower": "#6a3d9a",
    "NDC-PA-wo-USA-30June26-upper": "#cab2d6",
    "Current-Policies-GCAM": "#e31a1c",
    "Current-Policies-MESSAGE": "#ff7f00",
    "Current-Policies-REMIND": "#b15928",
    "VL": "#2166ac",
    "LN": "#4393c3",
    "L": "#1b9e77",
    "ML": "#a6d96a",
    "M": "#d8b365",
    "HL": "#f46d43",
    "H": "#a50026",
}
LABELS = {
    "ssp119": "SSP1-1.9",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp434": "SSP4-3.4",
    "ssp460": "SSP4-6.0",
    "ssp534-over": "SSP5-3.4-OS",
    "ssp585": "SSP5-8.5",
    "SSP2-com": "SSP2-com",
    "NDC-submitted-low": "NDC submitted, low",
    "NDC-submitted-high": "NDC submitted, high",
    "NDC-PA-allNDCs-30June26-lower": "NDC PA all, lower",
    "NDC-PA-allNDCs-30June26-upper": "NDC PA all, upper",
    "NDC-PA-wo-USA-30June26-lower": "NDC PA w/o USA, lower",
    "NDC-PA-wo-USA-30June26-upper": "NDC PA w/o USA, upper",
    "Current-Policies-GCAM": "Current policies, GCAM",
    "Current-Policies-MESSAGE": "Current policies, MESSAGE",
    "Current-Policies-REMIND": "Current policies, REMIND",
    "VL": "VL",
    "LN": "LN",
    "L": "L",
    "ML": "ML",
    "M": "M",
    "HL": "HL",
    "H": "H",
}
LINESTYLES = {
    "SSP2-com": ":",
    "Current-Policies-GCAM": "--",
    "Current-Policies-MESSAGE": "--",
    "Current-Policies-REMIND": "--",
}
VARIABLES = {
    "tas": ("Near-surface air temperature", "K", "K"),
    "pr": ("Precipitation", "mm month$^{-1}$", "mm month$^{-1}$"),
    "hurs": ("Near-surface relative humidity", "%", "percentage points"),
}
BASELINE = (1995, 2014)
FUTURE = (2081, 2100)
PLOT_START = 1940
PLOT_END = 2100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(variable: str, scenario: str) -> tuple[Path, dict]:
    path = INPUT_ROOT / variable / f"{scenario}.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def period_mean(years: np.ndarray, values: np.ndarray, period: tuple[int, int]) -> float:
    mask = (years >= period[0]) & (years <= period[1])
    return float(np.nanmean(values[mask]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    group_items = tuple(SCENARIO_GROUPS.items())
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(11.4, 8.8),
        sharex=True,
        sharey="row",
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.235, wspace=0.10, hspace=0.16)
    summary: dict[str, object] = {
        "schema": "gcmagicc-turkiye-application/v2",
        "layout": "3 rows (tas, pr, hurs) x 3 scenario-family columns",
        "region": "Türkiye (ISO3: TUR)",
        "season": "annual",
        "baseline": list(BASELINE),
        "future": list(FUTURE),
        "scenario_groups": {key: list(value) for key, value in group_items},
        "variables": {},
        "era5": {},
        "inputs": [],
    }
    legend_payloads: list[tuple[list[object], list[str]]] = []

    for row, (variable, (row_title, unit, change_unit)) in enumerate(VARIABLES.items()):
        variable_summary: dict[str, object] = {}
        era5_reference: tuple[np.ndarray, np.ndarray] | None = None
        for col, (group_key, scenarios) in enumerate(group_items):
            ax = axes[row, col]
            for scenario in scenarios:
                path, payload = load(variable, scenario)
                years = np.asarray(payload["years"], dtype=int)
                p05 = np.asarray(payload["percentiles"]["p05"], dtype=float)
                p50 = np.asarray(payload["percentiles"]["p50"], dtype=float)
                p95 = np.asarray(payload["percentiles"]["p95"], dtype=float)
                keep = (years >= PLOT_START) & (years <= PLOT_END)
                color = COLORS[scenario]
                ax.plot(
                    years[keep],
                    p50[keep],
                    color=color,
                    linestyle=LINESTYLES.get(scenario, "-"),
                    lw=1.25,
                    label=LABELS[scenario],
                    zorder=3,
                )
                ax.fill_between(
                    years[keep],
                    p05[keep],
                    p95[keep],
                    color=color,
                    alpha=0.035,
                    linewidth=0,
                    zorder=1,
                )
                baseline = period_mean(years, p50, BASELINE)
                future = period_mean(years, p50, FUTURE)
                variable_summary[scenario] = {
                    "scenario_group": group_key,
                    "median_baseline": baseline,
                    "median_future": future,
                    "median_change": future - baseline,
                    "change_unit": change_unit,
                    "members": int(payload["meta"]["n_members"]),
                }
                summary["inputs"].append(
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "sha256": sha256(path),
                        "source_tag": payload["source_tags"],
                    }
                )

                era5_years = np.asarray(payload["era5"]["years"], dtype=int)
                era5_values = np.asarray(payload["era5"]["values"], dtype=float)
                if era5_reference is None:
                    era5_reference = (era5_years, era5_values)
                elif not (
                    np.array_equal(era5_reference[0], era5_years)
                    and np.allclose(era5_reference[1], era5_values, equal_nan=True)
                ):
                    raise ValueError(f"ERA5 series differs between frozen {variable} scenario inputs")

            assert era5_reference is not None
            era5_keep = (era5_reference[0] >= PLOT_START) & (era5_reference[0] <= PLOT_END)
            ax.plot(
                era5_reference[0][era5_keep],
                era5_reference[1][era5_keep],
                color="black",
                lw=2.0,
                label="ERA5",
                zorder=10,
            )
            ax.set_xlim(PLOT_START, PLOT_END)
            ax.grid(axis="y", color="0.86", linewidth=0.55)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8.5)
            if row == 0:
                ax.set_title(GROUP_TITLES[group_key], fontsize=11.5, fontweight="bold", pad=9)
                handles, labels = ax.get_legend_handles_labels()
                legend_payloads.append((handles, labels))
            if row == 2:
                ax.set_xlabel("Year")

        summary["variables"][variable] = variable_summary
        assert era5_reference is not None
        summary["era5"][variable] = {
            "plotted": True,
            "colour": "black",
            "start_year": int(era5_reference[0].min()),
            "end_year": int(era5_reference[0].max()),
            "n_annual_values": int(era5_reference[0].size),
            "source": "embedded in every frozen annual-percentile input",
        }
        axes[row, 0].set_ylabel(unit)

    row_centres = (0.79, 0.545, 0.30)
    for centre, (_, (row_title, _, _)) in zip(row_centres, VARIABLES.items()):
        fig.text(0.018, centre, row_title, rotation=90, va="center", ha="center", fontsize=10.5, fontweight="bold")

    legend_x = (0.18, 0.50, 0.82)
    for x, (handles, labels) in zip(legend_x, legend_payloads):
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(x, 0.018),
            ncol=2,
            frameon=False,
            fontsize=8.0,
            handlelength=2.4,
            columnspacing=0.8,
            labelspacing=0.42,
        )
    fig.suptitle("Annual regional projections for Türkiye", fontsize=14, fontweight="bold")

    pdf = args.output_dir / "Figure3_Turkiye_tas_pr_hurs.pdf"
    png = args.output_dir / "Figure3_Turkiye_tas_pr_hurs.png"
    metadata = {"CreationDate": None, "ModDate": None, "Creator": "GCMagicc v1.0.1 reproducibility release"}
    fig.savefig(pdf, metadata=metadata, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(
        png,
        dpi=220,
        metadata={"Software": "GCMagicc v1.0.1 reproducibility release"},
        bbox_inches="tight",
        pad_inches=0.06,
    )
    plt.close(fig)

    summary["outputs"] = {pdf.name: sha256(pdf), png.name: sha256(png)}
    summary_path = args.output_dir / "Figure3_Turkiye_tas_pr_hurs.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
