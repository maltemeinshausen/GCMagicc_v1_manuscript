#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the compact Türkiye multi-variable scenario application.

The input files are frozen annual regional percentiles from the v1.0.1
projection workflow.  This release-native plotter deliberately has no imports
from the model-development repositories.
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
SCENARIOS = ("ssp119", "ssp245", "ssp585", "VL", "M", "H")
COLORS = {
    "ssp119": "#2b83ba",
    "ssp245": "#fdae61",
    "ssp585": "#b2182b",
    "VL": "#5ab4ac",
    "M": "#d8b365",
    "H": "#8c510a",
}
LABELS = {
    "ssp119": "SSP1-1.9",
    "ssp245": "SSP2-4.5",
    "ssp585": "SSP5-8.5",
    "VL": "CMIP7 VL",
    "M": "CMIP7 M",
    "H": "CMIP7 H",
}
VARIABLES = {
    "tas": ("Near-surface air temperature", "K", "K"),
    "pr": ("Precipitation", "mm month$^{-1}$", "mm month$^{-1}$"),
    "hurs": ("Near-surface relative humidity", "%", "percentage points"),
}
BASELINE = (1995, 2014)
FUTURE = (2081, 2100)


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

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 10.2), sharex=True, constrained_layout=True)
    summary: dict[str, object] = {
        "schema": "gcmagicc-turkiye-application/v1",
        "region": "Türkiye (ISO3: TUR)",
        "season": "annual",
        "baseline": list(BASELINE),
        "future": list(FUTURE),
        "scenarios": list(SCENARIOS),
        "variables": {},
        "inputs": [],
    }

    for ax, (variable, (title, unit, change_unit)) in zip(axes, VARIABLES.items()):
        variable_summary: dict[str, object] = {}
        for scenario in SCENARIOS:
            path, payload = load(variable, scenario)
            years = np.asarray(payload["years"], dtype=int)
            p05 = np.asarray(payload["percentiles"]["p05"], dtype=float)
            p50 = np.asarray(payload["percentiles"]["p50"], dtype=float)
            p95 = np.asarray(payload["percentiles"]["p95"], dtype=float)
            keep = years >= 1990
            color = COLORS[scenario]
            ax.plot(years[keep], p50[keep], color=color, lw=1.7, label=LABELS[scenario])
            ax.fill_between(years[keep], p05[keep], p95[keep], color=color, alpha=0.08, linewidth=0)
            baseline = period_mean(years, p50, BASELINE)
            future = period_mean(years, p50, FUTURE)
            variable_summary[scenario] = {
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
        summary["variables"][variable] = variable_summary
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(unit)
        ax.grid(axis="y", color="0.85", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Year")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Annual regional projections for Türkiye", fontweight="bold")

    pdf = args.output_dir / "Figure3_Turkiye_tas_pr_hurs.pdf"
    png = args.output_dir / "Figure3_Turkiye_tas_pr_hurs.png"
    metadata = {"CreationDate": None, "ModDate": None, "Creator": "GCMagicc v1.0.1 reproducibility release"}
    fig.savefig(pdf, metadata=metadata)
    fig.savefig(png, dpi=220, metadata={"Software": "GCMagicc v1.0.1 reproducibility release"})
    plt.close(fig)

    summary["outputs"] = {
        pdf.name: sha256(pdf),
        png.name: sha256(png),
    }
    summary_path = args.output_dir / "Figure3_Turkiye_tas_pr_hurs.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
