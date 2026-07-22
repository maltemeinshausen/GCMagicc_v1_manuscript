#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the publication workflow schematic for GCMagicc v1.0.1.

The diagram deliberately separates training from inference and represents the
global-temperature correction as exactly two emulator passes, not as an
iterative convergence loop.  It is fully vector-native and has no external
data dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "figures" / "main" / "training_inference_workflow"

NAVY = "#17324D"
MUTED = "#52677B"
LINE = "#8CA0B3"
PANEL = "#F5F8FA"
BLUE = "#DDEFF8"
BLUE_EDGE = "#237AA5"
GOLD = "#FFF0C8"
GOLD_EDGE = "#B97800"
TEAL = "#D9F1EC"
TEAL_EDGE = "#198B7A"
PURPLE = "#ECE5F7"
PURPLE_EDGE = "#7255A5"
CORAL = "#FCE4DC"
CORAL_EDGE = "#C95A3B"
GREEN = "#E2F2DE"
GREEN_EDGE = "#4E8B45"
ORANGE = "#FFF0E3"
ORANGE_EDGE = "#D56A22"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    face: str,
    edge: str,
    number: int | None = None,
    title_size: float = 10.0,
    body_size: float = 8.3,
    title_y: float = 0.73,
    linewidth: float = 1.4,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.13",
        linewidth=linewidth,
        facecolor=face,
        edgecolor=edge,
        zorder=2,
    )
    ax.add_patch(patch)
    if number is not None:
        ax.text(
            x + 0.19,
            y + height - 0.20,
            str(number),
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "circle,pad=0.20", "facecolor": edge, "edgecolor": "none"},
            zorder=5,
        )
        title_x = x + 0.39
        horizontal = "left"
    else:
        title_x = x + width / 2
        horizontal = "center"
    ax.text(
        title_x,
        y + height * title_y,
        title,
        ha=horizontal,
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=NAVY,
        zorder=4,
    )
    ax.text(
        x + width / 2,
        y + height * 0.36,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=NAVY,
        linespacing=1.18,
        zorder=4,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    connectionstyle: str = "arc3",
    linestyle: str = "-",
    width: float = 1.7,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=width,
            color=color,
            linestyle=linestyle,
            connectionstyle=connectionstyle,
            shrinkA=3,
            shrinkB=3,
            zorder=3,
        )
    )


def panel(ax: plt.Axes, y: float, height: float, label: str, title: str, subtitle: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0.22, y),
            15.56,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            linewidth=0.9,
            edgecolor="#C9D4DD",
            facecolor=PANEL,
            zorder=0,
        )
    )
    ax.text(0.52, y + height - 0.34, label, fontsize=12.5, fontweight="bold", color=BLUE_EDGE, va="center")
    ax.text(0.91, y + height - 0.34, title, fontsize=12.5, fontweight="bold", color=NAVY, va="center")
    ax.text(15.45, y + height - 0.34, subtitle, fontsize=8.5, color=MUTED, va="center", ha="right")


def variant_chip(ax: plt.Axes, x: float, y: float, width: float, name: str, detail: str, *, reduced: bool = False) -> None:
    edge = ORANGE_EDGE if reduced else BLUE_EDGE
    face = ORANGE if reduced else BLUE
    patch = FancyBboxPatch(
        (x, y),
        width,
        0.58,
        boxstyle="round,pad=0.01,rounding_size=0.10",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + 0.13, y + 0.39, name, ha="left", va="center", fontsize=8.4, fontweight="bold", color=edge)
    ax.text(x + 0.13, y + 0.16, detail, ha="left", va="center", fontsize=7.25, color=NAVY)


def build_figure(output_dir: Path) -> dict[str, object]:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "gcmagicc-v1.0.1-workflow",
        }
    )
    fig, ax = plt.subplots(figsize=(16, 9.4))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.4)
    ax.axis("off")

    ax.text(0.25, 9.16, "GCMagicc v1.0.1: train once, run many scenarios", fontsize=18, fontweight="bold", color=NAVY)
    ax.text(
        15.75,
        9.16,
        "Monthly stochastic fields • 10 variables • 1° or 0.25° output",
        fontsize=9.5,
        color=MUTED,
        ha="right",
        va="center",
    )

    # Panel a: training and evaluation.
    panel(ax, 5.20, 3.65, "a", "TRAIN AND EVALUATE", "Training fields are used here—not when a new scenario is run")
    rounded_box(
        ax,
        0.55,
        6.55,
        2.55,
        1.50,
        "Training targets",
        "32 CMIP6 models + ERA5\nmonthly fields • 10 variables\nHEALPix hierarchy",
        face=BLUE,
        edge=BLUE_EDGE,
        number=1,
    )
    rounded_box(
        ax,
        3.55,
        6.55,
        3.15,
        1.50,
        "Training predictors",
        "model_index + month/season\ndecomposed ERF components\nLOWESS tas/rtmt for full variants",
        face=GOLD,
        edge=GOLD_EDGE,
        number=2,
    )
    rounded_box(
        ax,
        7.15,
        6.40,
        4.15,
        1.80,
        "Shared multi-resolution emulator",
        "nside 1 → 2 → 4 → 8 → … → 64 / 256\nbias + optional common-effect + stochastic generator\ncoarse fields condition each finer level",
        face=TEAL,
        edge=TEAL_EDGE,
        number=3,
        title_y=0.78,
    )
    rounded_box(
        ax,
        11.75,
        6.40,
        3.65,
        1.80,
        "Held-out evaluation",
        "ssp245 held out for all variants\nMIROC & UKESM future runs withheld\nXS additionally: ssp585 + abrupt-2x/4xCO₂",
        face=PURPLE,
        edge=PURPLE_EDGE,
        number=4,
        title_y=0.78,
        body_size=7.9,
    )
    arrow(ax, (3.10, 7.30), (3.55, 7.30))
    arrow(ax, (6.70, 7.30), (7.15, 7.30))
    arrow(ax, (11.30, 7.30), (11.75, 7.30), linestyle="--", color=PURPLE_EDGE)

    variant_chip(ax, 0.55, 5.48, 3.52, "GCMagicc", "full predictors • default projections")
    variant_chip(ax, 4.32, 5.48, 3.52, "GCMagicc-CE", "full predictors • common-effect sensitivity")
    variant_chip(ax, 8.09, 5.48, 3.52, "GCMagicc-PM", "reduced predictors • historical retraining", reduced=True)
    variant_chip(ax, 11.86, 5.48, 3.52, "GCMagicc-XS", "reduced predictors • extrapolation test", reduced=True)

    # Panel b: default production inference path, shown as a numbered serpentine.
    panel(ax, 0.20, 4.70, "b", "RUN A NEW SCENARIO", "Default GCMagicc path • pretrained weights • no retraining")
    rounded_box(
        ax,
        0.55,
        3.05,
        2.05,
        1.18,
        "New pathway",
        "emissions / concentrations\n+ natural forcings",
        face=BLUE,
        edge=BLUE_EDGE,
        number=1,
        title_size=9.2,
        body_size=7.7,
    )
    rounded_box(
        ax,
        3.00,
        3.05,
        2.05,
        1.18,
        "MAGICC v7.5.3",
        "600 probabilistic global\nclimate members",
        face=GOLD,
        edge=GOLD_EDGE,
        number=2,
        title_size=9.2,
        body_size=7.7,
    )
    rounded_box(
        ax,
        5.45,
        3.05,
        2.35,
        1.18,
        "Monthly predictors",
        "tas • rtmt • ERFs\nsmooth + subsample to 20 / 100",
        face=GOLD,
        edge=GOLD_EDGE,
        number=3,
        title_size=9.2,
        body_size=7.6,
    )
    rounded_box(
        ax,
        8.20,
        3.05,
        2.40,
        1.18,
        "Conditioning",
        "ERA5 splice • model_index\nmonth/season • stochastic seed",
        face=BLUE,
        edge=BLUE_EDGE,
        number=4,
        title_size=9.2,
        body_size=7.5,
    )
    rounded_box(
        ax,
        11.00,
        3.05,
        2.05,
        1.18,
        "GCMagicc pass 1",
        "pretrained weights\nmonthly 10-variable fields",
        face=TEAL,
        edge=TEAL_EDGE,
        number=5,
        title_size=9.2,
        body_size=7.5,
    )
    rounded_box(
        ax,
        13.45,
        3.05,
        1.95,
        1.18,
        "tas diagnostic",
        "global area mean\nannual + centred 21 y",
        face=CORAL,
        edge=CORAL_EDGE,
        number=6,
        title_size=9.0,
        body_size=7.4,
    )
    arrow(ax, (2.60, 3.64), (3.00, 3.64))
    arrow(ax, (5.05, 3.64), (5.45, 3.64))
    arrow(ax, (7.80, 3.64), (8.20, 3.64))
    arrow(ax, (10.60, 3.64), (11.00, 3.64))
    arrow(ax, (13.05, 3.64), (13.45, 3.64))

    rounded_box(
        ax,
        12.60,
        0.77,
        2.80,
        1.38,
        "Correct one predictor",
        "Δᵧ = output tas − predictor tas\ntas_smoothed ← tas_smoothed − Δᵧ/10\nno other predictor changes",
        face=CORAL,
        edge=CORAL_EDGE,
        number=7,
        title_size=9.2,
        body_size=7.35,
        title_y=0.76,
    )
    rounded_box(
        ax,
        9.55,
        0.77,
        2.55,
        1.38,
        "GCMagicc pass 2",
        "same seed • rerun once\nno convergence loop",
        face=TEAL,
        edge=TEAL_EDGE,
        number=8,
        title_size=9.2,
        body_size=7.7,
        title_y=0.76,
    )
    rounded_box(
        ax,
        6.35,
        0.77,
        2.70,
        1.38,
        "Final ensemble",
        "monthly gridded output\n10 variables • 1° or 0.25°",
        face=GREEN,
        edge=GREEN_EDGE,
        number=9,
        title_size=9.2,
        body_size=7.7,
        title_y=0.76,
    )
    rounded_box(
        ax,
        1.95,
        0.77,
        3.90,
        1.38,
        "Evaluation and applications",
        "held-out fidelity • scenario projections\nregional impacts • event attribution",
        face=PURPLE,
        edge=PURPLE_EDGE,
        number=10,
        title_size=9.2,
        body_size=7.7,
        title_y=0.76,
    )
    arrow(ax, (14.43, 3.05), (14.00, 2.15), color=CORAL_EDGE, connectionstyle="arc3,rad=-0.18")
    arrow(ax, (12.60, 1.46), (12.10, 1.46), color=CORAL_EDGE)
    arrow(ax, (9.55, 1.46), (9.05, 1.46))
    arrow(ax, (6.35, 1.46), (5.85, 1.46), color=GREEN_EDGE)

    ax.text(
        0.58,
        2.70,
        "At inference, no CMIP6/ERA5 gridded fields, prescribed SSTs, or ocean-state fields are supplied.",
        fontsize=8.4,
        fontweight="bold",
        color=NAVY,
        va="center",
    )
    ax.text(
        15.38,
        2.38,
        "PM / XS: ERFs + model/season only (no tas/rtmt) → one emulator pass; steps 6–8 do not apply.",
        fontsize=7.9,
        color=ORANGE_EDGE,
        va="center",
        ha="right",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / "gcmagicc_training_inference_workflow.pdf"
    png = output_dir / "gcmagicc_training_inference_workflow.png"
    svg = output_dir / "gcmagicc_training_inference_workflow.svg"
    metadata = {
        "Title": "GCMagicc v1.0.1 workflow",
        "Author": "Malte Meinshausen and GCMagicc evaluation suite contributors",
        "Subject": "Training, evaluation, and scenario inference workflow",
        "Creator": "GCMagicc v1.0.1 reproducibility release",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05, metadata=metadata)
    fig.savefig(png, dpi=220, bbox_inches="tight", pad_inches=0.05, metadata={"Software": metadata["Creator"]})
    fig.savefig(
        svg,
        bbox_inches="tight",
        pad_inches=0.05,
        metadata={"Title": metadata["Title"], "Creator": metadata["Creator"], "Date": None},
    )
    plt.close(fig)

    script = Path(__file__).resolve()
    summary: dict[str, object] = {
        "schema": "gcmagicc-training-inference-workflow/v2",
        "generator": {"path": script.relative_to(ROOT).as_posix(), "sha256": sha256(script)},
        "training": {
            "targets": "32 CMIP6 models plus ERA5; monthly fields for 10 variables",
            "shared_architecture": "multi-resolution HEALPix stochastic emulator",
            "holdouts": {
                "all_variants": ["ssp245"],
                "model_families_with_future_runs_withheld": ["MIROC", "UKESM"],
                "GCMagicc-XS_additional": ["ssp585", "abrupt-2xCO2", "abrupt-4xCO2"],
            },
        },
        "inference": {
            "requires_gridded_esm_or_reanalysis_fields": False,
            "requires_prescribed_sst_or_ocean_state": False,
            "full_predictor_variants": ["GCMagicc", "GCMagicc-CE"],
            "reduced_predictor_variants": ["GCMagicc-PM", "GCMagicc-XS"],
            "correction": {
                "passes": 2,
                "changed_predictors": ["tas_smoothed"],
                "unchanged_seed": True,
                "iterative_convergence_loop": False,
            },
        },
        "outputs": {},
    }
    summary["outputs"] = {path.name: sha256(path) for path in (pdf, png, svg)}
    summary_path = output_dir / "gcmagicc_training_inference_workflow.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_figure(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
