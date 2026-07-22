#!/usr/bin/env python3
"""
1060_emergent_constraints
================================

Stacked emergent-constraint figure based on sample_emergent2.py:
1) Top panel  : observed vs observed model warming
2) Bottom panel: estimated vs observed model warming

Additions:
- Scenario-resolved ERA5 estimates for all SSPs below the lower x-axis (offset rows).
- Scenario-colored fading vertical plumes in the lower panel towards the x=y diagonal.
- Lower-right whisker/band panel from quantiles_by_scenario.csv
  (2.5-97.5, 5-95, 10-90 + median).
- Upper-right dedicated legend panel for model markers.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import ceil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scr.validation_helpers.helper_fonts import apply_sans_font_rcparams


# IPCC-consistent SSP colors (same base palette as notebooks/810_plot_SSPprojections.py)
SCENARIO_COLORS: dict[str, str] = {
    "ssp585": "#951b1e",
    "ssp370": "#e71d25",
    "ssp245": "#f79420",
    "ssp126": "#173c66",
    "ssp119": "#00addf",
    "ssp434": "#b32f4c",
    "ssp460": "#d65f2e",
    "ssp534-over": "#7c2d6f",
}

SCENARIO_ORDER = [
    "ssp119",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp434",
    "ssp460",
    "ssp534-over",
    "ssp585",
]

REQUIRED_QUANTILES = ["2.5%", "5%", "10%", "50%", "90%", "95%", "97.5%"]
PANEL_AXIS_MIN = 0.0
PANEL_AXIS_MAX = 7.5
PANEL_AXIS_SPAN = PANEL_AXIS_MAX - PANEL_AXIS_MIN
ERA_BARCODE_DEPTH_EQUIV_DEG = 2.0
ERA_BARCODE_DEPTH_AXES = ERA_BARCODE_DEPTH_EQUIV_DEG / PANEL_AXIS_SPAN
SSP_LABEL_FONTSIZE = 6.1
PANEL_LABEL_FONTSIZE = 8.2
SHINE_THROUGH_ALPHA = 0.30


def _resolve_default_data_root(script_dir: Path) -> Path:
    candidates = [
        script_dir.parent / "data" / "nicolaiplots" / "plots" / "plots_emergent_constraint",
        Path.cwd() / "data" / "nicolaiplots" / "plots" / "plots_emergent_constraint",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _set_style(savefig_dpi: int) -> None:
    apply_sans_font_rcparams(
        rc_updates={
            "figure.dpi": 150,
            "savefig.dpi": int(savefig_dpi),
            "font.size": 8.6,
            "axes.labelsize": 9.4,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "axes.edgecolor": "#3f3f3f",
            "axes.linewidth": 0.82,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d4d4d0",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.45,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.1,
            "ytick.major.size": 3.1,
            "legend.frameon": False,
        }
    )


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _to_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    m = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def _near_diag_mask(x: np.ndarray, y: np.ndarray, rel_tol: float) -> np.ndarray:
    scale = np.maximum.reduce([np.abs(x), np.abs(y), np.ones_like(x)])
    return np.abs(y - x) <= (rel_tol * scale)


def _filter_distinct_points(df: pd.DataFrame, x_col: str, y_col: str, rel_tol: float) -> pd.DataFrame:
    x = _to_numeric(df[x_col])
    y = _to_numeric(df[y_col])
    m = _finite_mask(x, y)
    x = x[m]
    y = y[m]
    keep = ~_near_diag_mask(x, y, rel_tol=rel_tol)
    idx_finite = np.flatnonzero(m)
    idx_keep = idx_finite[np.flatnonzero(keep)]
    return df.iloc[idx_keep].copy()


def _compute_limits(values: np.ndarray, pad_frac: float = 0.06) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = hi - lo
    pad = pad_frac * (span if span > 0 else 1.0)
    return (lo - pad, hi + pad)


def _draw_diag(ax: plt.Axes) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], linestyle=(0, (4, 2)), linewidth=0.95, color="#222222", alpha=0.62, zorder=0.5)


def _build_model_styles(versions: list[str]) -> dict[str, dict[str, object]]:
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h", "H", "p", "8", "d"]
    cmap = plt.get_cmap("tab20")
    out: dict[str, dict[str, object]] = {}
    for i, version in enumerate(versions):
        out[version] = {
            "marker": markers[i % len(markers)],
            "color": cmap(i % cmap.N),
        }
    return out


def _unique_handles(handles: list[Line2D]) -> list[Line2D]:
    seen: set[str] = set()
    out: list[Line2D] = []
    for h in handles:
        label = str(h.get_label())
        if label in seen:
            continue
        seen.add(label)
        out.append(h)
    return out


def _legend_layout_for_height(
    n_items: int,
    available_points: float,
    base_fontsize: float = 7.4,
    min_fontsize: float = 5.2,
    max_cols: int = 3,
    line_factor: float = 1.17,
) -> tuple[int, float]:
    if n_items <= 0:
        return (1, base_fontsize)
    for ncol in range(1, max_cols + 1):
        rows = int(ceil(n_items / ncol))
        required = rows * line_factor * base_fontsize
        if required <= available_points:
            return (ncol, base_fontsize)
        fs_fit = available_points / (rows * line_factor)
        if fs_fit >= min_fontsize:
            return (ncol, fs_fit)
    return (max_cols, min_fontsize)


def _serialize_model_styles(model_styles: dict[str, dict[str, object]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for version, style in model_styles.items():
        marker = str(style.get("marker", "o"))
        color = style.get("color", "#6f6f6f")
        out[version] = {"marker": marker, "color": to_hex(color)}
    return out


def _deserialize_model_styles(raw: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for version, style in raw.items():
        out[str(version)] = {
            "marker": str(style.get("marker", "o")),
            "color": str(style.get("color", "#6f6f6f")),
        }
    return out


def _extract_scatter_points_by_model(
    df_ssp: pd.DataFrame,
    x_col: str,
    y_col: str,
    rel_tol: float,
    filter_distinct_points: bool,
) -> dict[str, dict[str, list[float]]]:
    if filter_distinct_points:
        dfp = _filter_distinct_points(df_ssp, x_col=x_col, y_col=y_col, rel_tol=rel_tol)
    else:
        dfp = df_ssp.copy()
    out: dict[str, dict[str, list[float]]] = {}

    for version in sorted(dfp["version"].astype(str).unique()):
        g = dfp[dfp["version"].astype(str) == version]
        x = _to_numeric(g[x_col])
        y = _to_numeric(g[y_col])
        m = _finite_mask(x, y)
        x = x[m]
        y = y[m]
        if x.size == 0:
            continue
        out[version] = {
            "x": [float(v) for v in x],
            "y": [float(v) for v in y],
        }
    return out


def _plot_scatter_from_points(
    ax: plt.Axes,
    points_by_version: dict[str, dict[str, list[float]]],
    model_styles: dict[str, dict[str, object]],
    point_size: float,
    point_alpha: float,
) -> tuple[list[Line2D], np.ndarray, np.ndarray]:
    handles: list[Line2D] = []
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for version in sorted(points_by_version.keys()):
        pts = points_by_version[version]
        x = np.asarray(pts.get("x", []), dtype=float)
        y = np.asarray(pts.get("y", []), dtype=float)
        if x.size == 0 or y.size == 0:
            continue
        m = _finite_mask(x, y)
        x = x[m]
        y = y[m]
        if x.size == 0:
            continue

        st = model_styles.get(version, {"marker": "o", "color": "#6f6f6f"})
        ax.scatter(
            x,
            y,
            s=point_size,
            marker=st["marker"],
            facecolors=st["color"],
            edgecolors="#111111",
            linewidths=0.24,
            alpha=point_alpha,
            zorder=2.8,
        )
        handles.append(
            Line2D(
                [],
                [],
                linestyle="None",
                marker=st["marker"],
                markersize=np.sqrt(point_size),
                markerfacecolor=st["color"],
                markeredgecolor="#111111",
                markeredgewidth=0.34,
                alpha=point_alpha,
                label=version,
            )
        )
        xs.append(x)
        ys.append(y)

    if xs:
        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
    else:
        x_all = np.array([], dtype=float)
        y_all = np.array([], dtype=float)
    return handles, x_all, y_all


def _plot_scatter_by_model(
    ax: plt.Axes,
    df_ssp: pd.DataFrame,
    x_col: str,
    y_col: str,
    model_styles: dict[str, dict[str, object]],
    rel_tol: float,
    point_size: float,
    point_alpha: float,
    filter_distinct_points: bool = True,
) -> tuple[list[Line2D], np.ndarray, np.ndarray]:
    if filter_distinct_points:
        dfp = _filter_distinct_points(df_ssp, x_col=x_col, y_col=y_col, rel_tol=rel_tol)
    else:
        dfp = df_ssp.copy()
    handles: list[Line2D] = []
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for version in sorted(dfp["version"].astype(str).unique()):
        g = dfp[dfp["version"].astype(str) == version]
        x = _to_numeric(g[x_col])
        y = _to_numeric(g[y_col])
        m = _finite_mask(x, y)
        x = x[m]
        y = y[m]
        if x.size == 0:
            continue

        st = model_styles.get(version, {"marker": "o", "color": "#6f6f6f"})
        ax.scatter(
            x,
            y,
            s=point_size,
            marker=st["marker"],
            facecolors=st["color"],
            edgecolors="#111111",
            linewidths=0.24,
            alpha=point_alpha,
            zorder=2.8,
        )
        handles.append(
            Line2D(
                [],
                [],
                linestyle="None",
                marker=st["marker"],
                markersize=np.sqrt(point_size),
                markerfacecolor=st["color"],
                markeredgecolor="#111111",
                markeredgewidth=0.34,
                alpha=point_alpha,
                label=version,
            )
        )
        xs.append(x)
        ys.append(y)

    if xs:
        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
    else:
        x_all = np.array([], dtype=float)
        y_all = np.array([], dtype=float)
    return handles, x_all, y_all


def _scenario_sort_key(name: str) -> tuple[int, int | str]:
    try:
        idx = SCENARIO_ORDER.index(name)
        return (0, idx)
    except ValueError:
        return (1, name)


def _parse_scenarios(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def _str2bool(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    val = str(raw).strip().lower()
    if val in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw!r}")


def _read_quantiles(path: Path) -> pd.DataFrame:
    df = _clean_columns(pd.read_csv(path))
    if "quantile" in df.columns:
        out = df.set_index("quantile")
    else:
        out = df.set_index(df.columns[0])
    out.index = [str(i).strip() for i in out.index]
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _discover_scenarios_from_template(era_template: str) -> list[str]:
    if "{scen}" not in era_template:
        return []
    pre, post = era_template.split("{scen}")
    matches = []
    for p in sorted(Path().glob(era_template.replace("{scen}", "*"))):
        s = str(p)
        if not (s.startswith(pre) and s.endswith(post)):
            continue
        scen = s[len(pre): len(s) - len(post) if len(post) > 0 else None]
        if scen:
            matches.append(scen)
    return matches


def _read_era_values(era_template: str, scenarios: Iterable[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    era_by_scen: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for scen in scenarios:
        p = Path(era_template.format(scen=scen))
        if not p.exists():
            missing.append(scen)
            continue
        df = _clean_columns(pd.read_csv(p))
        if "trend_yera_hat" not in df.columns:
            missing.append(scen)
            continue
        vals = _to_numeric(df["trend_yera_hat"])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            missing.append(scen)
            continue
        era_by_scen[scen] = vals.astype(float)
    return era_by_scen, missing


def _draw_era_rows_below_axis(
    ax: plt.Axes,
    scenarios: list[str],
    era_by_scen: dict[str, np.ndarray],
    scenario_colors: dict[str, str],
    depth_axes: float = ERA_BARCODE_DEPTH_AXES,
) -> float:
    if not scenarios:
        return float(depth_axes)

    n = len(scenarios)
    top_gap = 0.035
    bottom_gap = 0.020
    usable = max(depth_axes - top_gap - bottom_gap, 0.12)
    row_step = usable / max(n, 1)
    row_height = float(np.clip(0.68 * row_step, 0.014, 0.026))
    row_start = -(top_gap + row_height)

    for i, scen in enumerate(scenarios):
        vals = era_by_scen.get(scen)
        if vals is None or vals.size == 0:
            continue
        color = scenario_colors.get(scen, "#666666")
        y0 = row_start - i * row_step
        y1 = y0 + row_height

        ax.vlines(
            vals,
            ymin=y0,
            ymax=y1,
            transform=ax.get_xaxis_transform(),
            color=color,
            linewidth=0.65,
            alpha=0.82,
            zorder=4,
            clip_on=False,
        )
        med = float(np.median(vals))
        ax.vlines(
            med,
            ymin=y0 - 0.008,
            ymax=y1 + 0.008,
            transform=ax.get_xaxis_transform(),
            color=color,
            linewidth=1.4,
            alpha=0.95,
            zorder=4.2,
            clip_on=False,
        )
        # Scenario labels left of each row's line/rug cluster, in data-x + axes-y coordinates.
        x_lbl = float(np.nanpercentile(vals, 8) - 0.15)
        x_lbl = max(x_lbl, ax.get_xlim()[0] + 0.05)
        ax.text(
            x_lbl,
            y0 + 0.5 * row_height,
            scen.upper(),
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="center",
            fontsize=SSP_LABEL_FONTSIZE,
            color=color,
            fontweight=600,
            clip_on=False,
        )

    # Place section label just above the moved bottom x-axis; include panel label "d".
    y_era_label = -(depth_axes + 0.006)  # lifted by ~half-line additional offset
    ax.text(
        0.0,
        y_era_label,
        "d",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        color="#111111",
        fontweight=700,
        clip_on=False,
    )
    ax.text(
        0.04,
        y_era_label,
        "ERA5 scenario estimates",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        color="#222222",
        fontweight=400,
        clip_on=False,
    )

    return float(depth_axes)


def _move_bottom_xaxis_below_barcodes(ax: plt.Axes, barcode_depth_axes: float, extra_pad_axes: float = 0.045) -> None:
    """
    Move the x-axis spine/ticks beneath the scenario barcode rows while keeping y-limits at 0..7.
    """
    y_pos = -(barcode_depth_axes + extra_pad_axes)
    ax.spines["bottom"].set_position(("axes", y_pos))
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_label_position("bottom")
    ax.tick_params(axis="x", which="both", direction="out", length=3.2, width=0.72, pad=2.4)
    ax.xaxis.labelpad = 6.0


def _draw_era_fading_plumes(
    ax: plt.Axes,
    scenarios: list[str],
    era_by_scen: dict[str, np.ndarray],
    scenario_colors: dict[str, str],
    n_segments: int = 18,
    max_per_scenario: int = 90,
) -> None:
    y0, y1 = ax.get_ylim()
    for scen in scenarios:
        vals = era_by_scen.get(scen)
        if vals is None or vals.size == 0:
            continue
        color = scenario_colors.get(scen, "#666666")

        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        if arr.size > max_per_scenario:
            idx = np.linspace(0, arr.size - 1, max_per_scenario).astype(int)
            arr = np.sort(arr)[idx]

        for x in arr:
            y_top = min(float(x), y1)
            if y_top <= y0:
                continue
            ys = np.linspace(y0, y_top, n_segments + 1)
            for j in range(n_segments):
                frac = (j + 1) / n_segments
                alpha = 0.23 * (1.0 - frac) ** 1.45
                if alpha <= 0:
                    continue
                ax.plot(
                    [x, x],
                    [ys[j], ys[j + 1]],
                    color=color,
                    linewidth=0.72,
                    alpha=alpha,
                    zorder=1.15,
                    solid_capstyle="round",
                )


def _draw_quantile_whisker_panel(
    ax: plt.Axes,
    quantiles: pd.DataFrame,
    scenarios: list[str],
    scenario_colors: dict[str, str],
) -> None:
    valid = [s for s in scenarios if s in quantiles.columns]
    if not valid:
        ax.axis("off")
        ax.text(0.0, 1.0, "No quantile columns for selected scenarios", transform=ax.transAxes, ha="left", va="top")
        return

    q = quantiles.loc[REQUIRED_QUANTILES, valid]
    xpos = np.arange(len(valid), dtype=float)

    bar_w = 0.14
    cap_w = 0.10
    median_w = 0.09

    for i, scen in enumerate(valid):
        x = xpos[i]
        c = scenario_colors.get(scen, "#6a6a6a")
        q2 = float(q.loc["2.5%", scen])
        q5 = float(q.loc["5%", scen])
        q10 = float(q.loc["10%", scen])
        q50 = float(q.loc["50%", scen])
        q90 = float(q.loc["90%", scen])
        q95 = float(q.loc["95%", scen])
        q97 = float(q.loc["97.5%", scen])

        ax.add_patch(Rectangle((x - bar_w / 2.0, q2), bar_w, q97 - q2, facecolor=c, edgecolor="none", alpha=0.10, zorder=1))
        ax.add_patch(Rectangle((x - bar_w / 2.0, q5), bar_w, q95 - q5, facecolor=c, edgecolor="none", alpha=0.16, zorder=2))
        ax.add_patch(Rectangle((x - bar_w / 2.0, q10), bar_w, q90 - q10, facecolor=c, edgecolor="none", alpha=0.26, zorder=3))

        ax.vlines(x, q2, q97, color=c, linewidth=1.05, alpha=0.82, zorder=4)
        ax.hlines([q2, q97], x - cap_w / 2.0, x + cap_w / 2.0, color=c, linewidth=1.0, alpha=0.86, zorder=4)
        ax.hlines(q50, x - median_w / 2.0, x + median_w / 2.0, color=c, linewidth=1.20, alpha=0.96, zorder=5)

        # Small data labels for central quantiles (10/50/90), fixed to the right of each whisker.
        x_txt = x + (bar_w * 0.72)
        ha = "left"
        fs = 5.0
        txt_kw = dict(fontsize=fs, color=c, alpha=0.94, zorder=6, ha=ha, va="center")
        ax.text(x_txt, q10, f"{q10:.1f}", **txt_kw)
        ax.text(x_txt, q50, f"{q50:.1f}", **txt_kw)
        ax.text(x_txt, q90, f"{q90:.1f}", **txt_kw)

        # Scenario labels near whiskers:
        #   - below by default
        #   - above for SSP370 and SSP585
        place_above = scen.lower() in {"ssp370", "ssp585"}
        if place_above:
            y_lbl = float(min(q97 + 0.14, PANEL_AXIS_MAX - 0.08))
            va_lbl = "bottom"
            ha_lbl = "right"
        else:
            y_lbl = float(max(q2 - 0.14, PANEL_AXIS_MIN + 0.08))
            va_lbl = "top"
            ha_lbl = "left"
        ax.text(
            x + bar_w * 0.95,
            y_lbl,
            scen.upper(),
            fontsize=SSP_LABEL_FONTSIZE,
            color=c,
            fontweight=600,
            ha=ha_lbl,
            va=va_lbl,
            zorder=6,
        )

    # Small panel label only; no title/subtitle requested.
    ax.text(
        0.02,
        0.98,
        "b",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        color="#111111",
        fontweight=700,
    )

    ax.set_xlim(-0.65, float(len(valid)) - 0.35)
    ax.set_ylim(PANEL_AXIS_MIN, PANEL_AXIS_MAX)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.set_ylabel("Derived Model Warming (°C)")
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    ax.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False)
    ax.grid(axis="y", alpha=0.26, linewidth=0.5)
    ax.grid(axis="x", visible=False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _align_bottom_row_with_top_right(fig: plt.Figure, ax_top: plt.Axes, ax_bottom: plt.Axes, ax_q: plt.Axes) -> None:
    """
    Enforce:
      - Right panels (top and bottom) share exact x-span.
      - Lower-left whisker panel is directly adjacent to lower-right panel.
      - Whisker-panel width is tied to ERA barcode depth (~2/7 of panel-B height).
    """
    fig.canvas.draw()

    p_top = ax_top.get_position()
    p_bottom = ax_bottom.get_position()

    # Match bottom-right panel geometry to top-right panel geometry in x (and size for consistency).
    new_bottom = [p_top.x0, p_bottom.y0, p_top.width, p_top.height]
    ax_bottom.set_position(new_bottom)

    desired_q_width = new_bottom[3] * ERA_BARCODE_DEPTH_AXES
    max_q_width = max(0.06, new_bottom[0] - 0.03)
    q_width = min(desired_q_width, max_q_width)
    q_x0 = new_bottom[0] - q_width
    ax_q.set_position([q_x0, new_bottom[1], q_width, new_bottom[3]])


def _save_figure_files(fig: plt.Figure, output_base: str | Path, formats: list[str], dpi: int) -> list[Path]:
    out_base = Path(output_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    stem = out_base.name.rsplit(".", 1)[0] if "." in out_base.name else out_base.name
    out_dir = out_base.parent

    saved: list[Path] = []
    for fmt in formats:
        fmt_clean = str(fmt).strip().lstrip(".").lower()
        if not fmt_clean:
            continue
        out = out_dir / f"{stem}.{fmt_clean}"
        fig.savefig(out, dpi=int(dpi), bbox_inches="tight", pad_inches=0.01, format=fmt_clean)
        saved.append(out)
    plt.close(fig)
    return saved


def _build_quantile_map(qdf: pd.DataFrame, scenarios: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for scen in scenarios:
        if scen not in qdf.columns:
            continue
        out[scen] = {}
        for q in REQUIRED_QUANTILES:
            if q not in qdf.index:
                continue
            val = float(qdf.loc[q, scen])
            if np.isfinite(val):
                out[scen][q] = val
    return out


def _quantile_map_to_df(quantile_map: dict[str, dict[str, float]], scenarios: list[str]) -> pd.DataFrame:
    cols = {scen: quantile_map.get(scen, {}) for scen in scenarios}
    qdf = pd.DataFrame(cols)
    qdf = qdf.reindex(REQUIRED_QUANTILES)
    return qdf


def _build_composite_payload(
    scenarios: list[str],
    model_styles: dict[str, dict[str, object]],
    top_points: dict[str, dict[str, list[float]]],
    bottom_points: dict[str, dict[str, list[float]]],
    era_by_scen: dict[str, np.ndarray],
    qdf: pd.DataFrame,
    point_size: float,
    point_alpha: float,
    dpi: int,
    output_base: str | Path,
    formats: list[str],
) -> dict[str, object]:
    payload = {
        "schema": "emergent_constraint_composite_v1",
        "scenarios": scenarios,
        "required_quantiles": REQUIRED_QUANTILES,
        "panel_axis": {"min": PANEL_AXIS_MIN, "max": PANEL_AXIS_MAX},
        "render_settings": {
            "point_size": float(point_size),
            "point_alpha": float(point_alpha),
            "dpi": int(dpi),
            "formats": [str(f) for f in formats],
        },
        "default_output_base": str(output_base),
        "model_styles": _serialize_model_styles(model_styles),
        "scatter_points": {"top": top_points, "bottom": bottom_points},
        "era_by_scenario": {sc: [float(v) for v in np.asarray(vals, dtype=float)] for sc, vals in era_by_scen.items()},
        "quantiles_by_scenario": _build_quantile_map(qdf, scenarios),
    }
    return payload


def _write_composite_json(path: str | Path, payload: dict[str, object]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _read_composite_json(path: str | Path) -> dict[str, object]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Composite JSON not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _render_prepared_figure(
    scenarios: list[str],
    model_styles: dict[str, dict[str, object]],
    top_points: dict[str, dict[str, list[float]]],
    bottom_points: dict[str, dict[str, list[float]]],
    era_by_scen: dict[str, np.ndarray],
    qdf: pd.DataFrame,
    output_base: str | Path,
    formats: list[str],
    dpi: int,
    point_size: float,
    point_alpha: float,
) -> list[Path]:
    _set_style(int(dpi))

    fig = plt.figure(figsize=(9.5, 8.3))
    gs_outer = fig.add_gridspec(nrows=2, ncols=1, hspace=0.20)
    gs_top = gs_outer[0].subgridspec(nrows=1, ncols=2, width_ratios=[0.56, 1.0], wspace=0.03)
    gs_bottom = gs_outer[1].subgridspec(nrows=1, ncols=2, width_ratios=[0.21, 1.0], wspace=0.0)

    ax_leg = fig.add_subplot(gs_top[0, 0])
    ax_top = fig.add_subplot(gs_top[0, 1])
    ax_bottom = fig.add_subplot(gs_bottom[0, 1])
    ax_q = fig.add_subplot(gs_bottom[0, 0], sharey=ax_bottom)

    handles_top, _, _ = _plot_scatter_from_points(
        ax=ax_top,
        points_by_version=top_points,
        model_styles=model_styles,
        point_size=float(point_size),
        point_alpha=float(point_alpha),
    )
    handles_bottom, _, _ = _plot_scatter_from_points(
        ax=ax_bottom,
        points_by_version=bottom_points,
        model_styles=model_styles,
        point_size=float(point_size),
        point_alpha=float(point_alpha),
    )

    ax_top.set_xlim(PANEL_AXIS_MIN, PANEL_AXIS_MAX)
    ax_top.set_ylim(PANEL_AXIS_MIN, PANEL_AXIS_MAX)
    ax_bottom.set_xlim(PANEL_AXIS_MIN, PANEL_AXIS_MAX)
    ax_bottom.set_ylim(PANEL_AXIS_MIN, PANEL_AXIS_MAX)
    ax_top.set_box_aspect(1.0)
    ax_bottom.set_box_aspect(1.0)

    _align_bottom_row_with_top_right(fig=fig, ax_top=ax_top, ax_bottom=ax_bottom, ax_q=ax_q)

    int_ticks = np.arange(int(PANEL_AXIS_MIN), int(PANEL_AXIS_MAX) + 1, 1)
    ax_top.set_xticks(int_ticks)
    ax_top.set_yticks(int_ticks)
    ax_bottom.set_xticks(int_ticks)
    ax_bottom.set_yticks(int_ticks)

    _draw_diag(ax_top)
    _draw_diag(ax_bottom)
    ax_bottom.axhline(PANEL_AXIS_MIN, color="#2f2f2f", linewidth=0.75, alpha=0.65, zorder=3.05)

    _draw_era_fading_plumes(ax_bottom, scenarios=scenarios, era_by_scen=era_by_scen, scenario_colors=SCENARIO_COLORS)
    barcode_depth_axes = _draw_era_rows_below_axis(
        ax_bottom,
        scenarios=scenarios,
        era_by_scen=era_by_scen,
        scenario_colors=SCENARIO_COLORS,
        depth_axes=ERA_BARCODE_DEPTH_AXES,
    )
    _move_bottom_xaxis_below_barcodes(ax_bottom, barcode_depth_axes=barcode_depth_axes, extra_pad_axes=0.045)

    ax_top.set_xlabel("Observed Model Warming (°C)")
    ax_top.set_ylabel("Observed Model Warming (°C)")
    ax_bottom.set_xlabel("Estimated Model Warming (°C)")
    ax_bottom.set_ylabel("Observed Model Warming (°C)")
    ax_bottom.yaxis.set_label_position("right")
    ax_bottom.yaxis.tick_right()
    ax_bottom.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
    ax_bottom.spines["left"].set_visible(False)
    ax_bottom.spines["right"].set_visible(True)

    ax_top.text(
        0.02,
        0.98,
        "a",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        color="#111111",
        fontweight=700,
    )
    ax_top.text(
        0.07,
        0.98,
        "Observed vs Observed",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        color="#222222",
        fontweight=400,
    )
    ax_bottom.text(
        0.02,
        0.98,
        "c",
        transform=ax_bottom.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        color="#111111",
        fontweight=700,
    )
    ax_bottom.text(
        0.07,
        0.98,
        "Estimated vs Observed",
        transform=ax_bottom.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        color="#222222",
        fontweight=400,
    )

    ax_top.grid(alpha=0.33)
    ax_bottom.grid(alpha=0.33)

    _draw_quantile_whisker_panel(ax_q, quantiles=qdf, scenarios=scenarios, scenario_colors=SCENARIO_COLORS)
    ax_q.set_yticks(int_ticks)

    leg_handles = _unique_handles(handles_top + handles_bottom)
    ax_leg.axis("off")
    ax_leg.set_title("CMIP6 Model Legend", loc="left", y=0.94, fontsize=8.0, fontweight=400, color="#222222", pad=2.0)
    fig_h_in = fig.get_figheight()
    available_points = ax_leg.get_position().height * fig_h_in * 72.0
    ncol, fs = _legend_layout_for_height(
        n_items=len(leg_handles),
        available_points=available_points,
        base_fontsize=7.4,
        min_fontsize=5.2,
        max_cols=3,
    )
    if leg_handles:
        ax_leg.legend(
            handles=leg_handles,
            labels=[h.get_label() for h in leg_handles],
            loc="upper left",
            bbox_to_anchor=(0.0, 0.93),
            frameon=False,
            ncol=ncol,
            fontsize=fs,
            handletextpad=0.48,
            columnspacing=0.85,
            labelspacing=0.35,
            borderaxespad=0.0,
        )

    return _save_figure_files(fig, output_base=output_base, formats=formats, dpi=int(dpi))


def replot(
    composite_json: str | Path,
    output_base: str | Path | None = None,
    formats: list[str] | None = None,
    dpi: int | None = None,
) -> list[Path]:
    payload = _read_composite_json(composite_json)
    if payload.get("schema") != "emergent_constraint_composite_v1":
        raise ValueError("Unsupported composite JSON schema.")

    scenarios = [str(s) for s in payload.get("scenarios", [])]
    if not scenarios:
        raise ValueError("Composite JSON does not contain scenarios.")

    model_styles = _deserialize_model_styles(payload.get("model_styles", {}))
    scatter = payload.get("scatter_points", {})
    top_points = scatter.get("top", {})
    bottom_points = scatter.get("bottom", {})

    era_raw = payload.get("era_by_scenario", {})
    era_by_scen: dict[str, np.ndarray] = {}
    for sc in scenarios:
        vals = era_raw.get(sc, [])
        era_by_scen[sc] = np.asarray(vals, dtype=float)

    q_map = payload.get("quantiles_by_scenario", {})
    qdf = _quantile_map_to_df(q_map, scenarios)
    missing_rows = [q for q in REQUIRED_QUANTILES if q not in qdf.index]
    if missing_rows:
        raise ValueError(f"Composite JSON missing quantile rows: {missing_rows}")

    render_settings = payload.get("render_settings", {})
    point_size = float(render_settings.get("point_size", 20.0))
    point_alpha = float(render_settings.get("point_alpha", 0.78))
    dpi_use = int(dpi if dpi is not None else render_settings.get("dpi", 320))
    fmts = formats if formats is not None else list(render_settings.get("formats", ["pdf", "png"]))
    out_base = output_base if output_base is not None else payload.get("default_output_base")
    if out_base is None:
        out_base = Path(composite_json).with_suffix("").as_posix() + "_replot"

    return _render_prepared_figure(
        scenarios=scenarios,
        model_styles=model_styles,
        top_points=top_points,
        bottom_points=bottom_points,
        era_by_scen=era_by_scen,
        qdf=qdf,
        output_base=out_base,
        formats=fmts,
        dpi=dpi_use,
        point_size=point_size,
        point_alpha=point_alpha,
    )


def _build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    data_root = _resolve_default_data_root(here)

    p = argparse.ArgumentParser(description="Stacked emergent-constraint figure with scenario ERA/quantile overlays.")
    p.add_argument(
        "--data_ssp",
        type=str,
        default=str(data_root / "figchangeeraSsmall_1" / "trends_ssp_True_6.csv"),
        help="CSV with model-level trends (must include version/trend columns).",
    )
    p.add_argument(
        "--data_era_template",
        type=str,
        default=str(data_root / "figchangeforceeraSsmall_1" / "trends_{scen}_True_6.csv"),
        help="ERA CSV template with {scen} placeholder.",
    )
    p.add_argument(
        "--quantiles_csv",
        type=str,
        default=str(data_root / "quantiles_by_scenario.csv"),
        help="Quantiles table CSV with rows as quantiles and columns as scenarios.",
    )
    p.add_argument(
        "--scenarios",
        type=str,
        default="",
        help="Comma-separated scenarios. Default: inferred from quantiles/ERA files.",
    )

    p.add_argument("--left_x_col", type=str, default="trend_y_truesecond")
    p.add_argument("--left_y_col", type=str, default="trend_y_true")
    p.add_argument("--right_x_col", type=str, default="trend_y_hat")
    p.add_argument("--right_y_col", type=str, default="trend_y_true")

    p.add_argument("--diag_rel_tol", type=float, default=1e-2, help="Relative tolerance for filtering near x=y points.")
    p.add_argument(
        "--filter_distinct_points",
        type=_str2bool,
        default=False,
        help="Toggle near-diagonal filtering (TRUE/FALSE). FALSE plots all points with shine-through alpha.",
    )
    p.add_argument("--point_size", type=float, default=20.0)
    p.add_argument("--point_alpha", type=float, default=0.78)

    p.add_argument(
        "--output_base",
        type=str,
        default=str(data_root / "emergent_constraints"),
        help="Output path without extension (or with extension; extension will be ignored).",
    )
    p.add_argument(
        "--composite_json",
        type=str,
        default=None,
        help="Path to write composite plotted-data JSON (default: <output_base>_composite.json).",
    )
    p.add_argument(
        "--replot_json",
        type=str,
        default=None,
        help="If set, recreate the figure from this composite JSON (CSV inputs are ignored).",
    )
    p.add_argument("--formats", nargs="+", default=["pdf", "png"], help="Output formats, e.g. pdf png.")
    p.add_argument("--dpi", type=int, default=320)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.replot_json:
        try:
            saved = replot(
                composite_json=args.replot_json,
                output_base=args.output_base,
                formats=list(args.formats),
                dpi=int(args.dpi),
            )
        except Exception as exc:
            print(f"ERROR during replot: {exc}", file=sys.stderr)
            return 2

        if not saved:
            print("ERROR: no outputs were written because --formats was empty.", file=sys.stderr)
            return 2

        print("Saved figure files:")
        for p in saved:
            print(f"  - {p}")
        print(f"Replotted from composite JSON: {args.replot_json}")
        return 0

    try:
        df_ssp = _clean_columns(pd.read_csv(args.data_ssp))
    except Exception as exc:
        print(f"ERROR reading --data_ssp: {exc}", file=sys.stderr)
        return 2

    needed = [args.left_x_col, args.left_y_col, args.right_x_col, args.right_y_col, "version"]
    for col in needed:
        if col not in df_ssp.columns:
            print(f"ERROR: data_ssp missing '{col}'. Found: {list(df_ssp.columns)}", file=sys.stderr)
            return 2

    q_path = Path(args.quantiles_csv)
    if not q_path.exists():
        print(f"ERROR: quantiles CSV not found: {q_path}", file=sys.stderr)
        return 2
    try:
        qdf = _read_quantiles(q_path)
    except Exception as exc:
        print(f"ERROR reading quantiles CSV: {exc}", file=sys.stderr)
        return 2

    missing_rows = [q for q in REQUIRED_QUANTILES if q not in qdf.index]
    if missing_rows:
        print(f"ERROR: quantiles CSV missing rows: {missing_rows}", file=sys.stderr)
        return 2

    scenarios = _parse_scenarios(args.scenarios)
    if not scenarios:
        scenarios = [str(c).strip() for c in qdf.columns if str(c).strip()]
    if not scenarios:
        scenarios = _discover_scenarios_from_template(args.data_era_template)
    scenarios = sorted(dict.fromkeys(scenarios), key=_scenario_sort_key)

    era_by_scen, missing_era = _read_era_values(args.data_era_template, scenarios)
    if missing_era:
        print(f"Warning: skipped scenarios with missing ERA input: {missing_era}", file=sys.stderr)

    scenarios = [s for s in scenarios if s in era_by_scen and s in qdf.columns]
    if not scenarios:
        print("ERROR: no scenarios available with both ERA values and quantiles.", file=sys.stderr)
        return 2

    versions = sorted(df_ssp["version"].astype(str).unique())
    model_styles = _build_model_styles(versions)
    filter_distinct_points = bool(args.filter_distinct_points)
    point_alpha = float(args.point_alpha) if filter_distinct_points else SHINE_THROUGH_ALPHA

    top_points = _extract_scatter_points_by_model(
        df_ssp=df_ssp,
        x_col=args.left_x_col,
        y_col=args.left_y_col,
        rel_tol=float(args.diag_rel_tol),
        filter_distinct_points=filter_distinct_points,
    )
    bottom_points = _extract_scatter_points_by_model(
        df_ssp=df_ssp,
        x_col=args.right_x_col,
        y_col=args.right_y_col,
        rel_tol=float(args.diag_rel_tol),
        filter_distinct_points=filter_distinct_points,
    )

    saved = _render_prepared_figure(
        scenarios=scenarios,
        model_styles=model_styles,
        top_points=top_points,
        bottom_points=bottom_points,
        era_by_scen=era_by_scen,
        qdf=qdf,
        output_base=args.output_base,
        formats=list(args.formats),
        dpi=int(args.dpi),
        point_size=float(args.point_size),
        point_alpha=point_alpha,
    )
    if not saved:
        print("ERROR: no outputs were written because --formats was empty.", file=sys.stderr)
        return 2

    out_base = Path(args.output_base)
    stem = out_base.name.rsplit(".", 1)[0] if "." in out_base.name else out_base.name
    composite_path = Path(args.composite_json) if args.composite_json else (out_base.parent / f"{stem}_composite.json")
    payload = _build_composite_payload(
        scenarios=scenarios,
        model_styles=model_styles,
        top_points=top_points,
        bottom_points=bottom_points,
        era_by_scen=era_by_scen,
        qdf=qdf,
        point_size=float(args.point_size),
        point_alpha=point_alpha,
        dpi=int(args.dpi),
        output_base=args.output_base,
        formats=list(args.formats),
    )
    _write_composite_json(composite_path, payload)

    print("Saved figure files:")
    for p in saved:
        print(f"  - {p}")
    print(f"Saved composite JSON: {composite_path}")
    print("Scenarios plotted:")
    print(f"  - {', '.join(scenarios)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
