#!/usr/bin/env python3
"""
Create two-panel scatter figure from SSP/ERA trend CSVs with a journal-style aesthetic,
and sample an ERA+residual distribution to report selected quantiles.

Inputs (CSV; column names are trimmed for whitespace):
  - data_ssp must contain:
      version, trend_y_true, trend_y_truesecond, trend_y_hat
  - data_era must contain:
      trend_yera_hat

Figure behavior:
  - Different colors and filled markers per model (version).
  - No model names written into the plot; use a legend on the RIGHT side (no frame/border).
  - Legend is sized/column-split to fit the available vertical space.
  - Right panel shows ERA values as short vertical "rug" ticks along the x-axis.
  - Axis labels:
      Left  x: "Observed Model Warming"
      Left  y: "Observed Model Warming"
      Right x: "Estimated Model Warming"
      Right y: "Observed Model Warming"
  - Exclude points close to x=y (within a relative tolerance; default 1e-3 = 0.1 percent).
  - Save three PNGs:
      (1) combined two-panel figure
      (2) left panel only
      (3) right panel only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from math import ceil

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


# -----------------------------
# User-configurable defaults
# -----------------------------
maxlag = 6
depend = True
dir = "Ssmall"

# ["ssp119", "ssp126", "ssp370", "ssp434", "ssp460", "ssp534", "ssp585", "ssp534-over"]
scen = "ssp585"
data_ssp: str = f"figchangeera{dir}_1/trends_ssp_{depend}_{maxlag}.csv"
data_era: str = f"./figchangeforceera{dir}_1/trends_{scen}_{depend}_{maxlag}.csv"

# Sampling defaults
nsim: int = 2000
seed: int | None = 0

# Figure defaults
output_png: str = "warming_panels.png"   # combined output; left/right are derived from this
dpi: int = 300
point_size: float = 22.0
point_alpha: float = 0.85
diag_rel_tol: float = 1e-2  # 0.1 percent

# ERA rug on right panel
rug_alpha: float = 0.55
rug_lw: float = 1.0
rug_height: float = 0.06  # in axes coords (fraction of axis height)

# Legend formatting
legend_fontsize: float = 7.5
legend_min_fontsize: float = 5.5
legend_max_cols: int = 3
legend_xpad: float = 0.02  # padding to the right of axes, in axes fraction via bbox_to_anchor


# -----------------------------
# Helpers
# -----------------------------
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _to_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    m = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def _near_diag_mask(x: np.ndarray, y: np.ndarray, rel_tol: float) -> np.ndarray:
    """
    True where points are "too close" to the diagonal x=y.
    Relative tolerance with stabilizer for near-zero values.
    """
    scale = np.maximum.reduce([np.abs(x), np.abs(y), np.ones_like(x)])
    return np.abs(y - x) <= (rel_tol * scale)


def _set_journal_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": dpi,
        "font.size": 8.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "mathtext.default": "regular",
    })


def read_inputs(data_ssp_path: str | Path, data_era_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_ssp = _clean_columns(pd.read_csv(data_ssp_path))
    df_era = _clean_columns(pd.read_csv(data_era_path))
    return df_ssp, df_era


def simulate_distribution(df_ssp: pd.DataFrame, df_era: pd.DataFrame, nsim: int, seed: int | None) -> np.ndarray:
    """
    Sample nsim values by picking a random ERA value (trend_yera_hat) and adding a random residual:
      residual = trend_y_true - trend_y_hat (from df_ssp).
    """
    if nsim <= 0:
        raise ValueError("nsim must be a positive integer.")

    for col in ("trend_y_true", "trend_y_hat"):
        if col not in df_ssp.columns:
            raise KeyError(f"data_ssp missing required column '{col}'. Found: {list(df_ssp.columns)}")

    if "trend_yera_hat" not in df_era.columns:
        raise KeyError(f"data_era missing required column 'trend_yera_hat'. Found: {list(df_era.columns)}")

    y_true = _to_numeric(df_ssp["trend_y_true"])
    y_hat = _to_numeric(df_ssp["trend_y_hat"])
    m = _finite_mask(y_true, y_hat)
    residuals = (y_true[m] - y_hat[m]).astype(float)
    if residuals.size == 0:
        raise ValueError("No finite residuals could be computed from data_ssp (trend_y_true - trend_y_hat).")

    era_vals = _to_numeric(df_era["trend_yera_hat"])
    era_vals = era_vals[np.isfinite(era_vals)]
    if era_vals.size == 0:
        raise ValueError("No finite numeric values found for column 'trend_yera_hat' in data_era.")

    rng = np.random.default_rng(seed)
    base = rng.choice(era_vals, size=nsim, replace=True)
    res = rng.choice(residuals, size=nsim, replace=True)
    return base + res


def compute_quantiles(samples: np.ndarray) -> dict[str, float]:
    qs = np.array([0.025, 0.05, 0.10, 0.90, 0.95, 0.975], dtype=float)
    vals = np.quantile(samples, qs)
    return {
        "2.5%": float(vals[0]),
        "5%": float(vals[1]),
        "10%": float(vals[2]),
        "90%": float(vals[3]),
        "95%": float(vals[4]),
        "97.5%": float(vals[5]),
    }


def _draw_diag(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=0.9, alpha=0.55)  # no label

def _add_rug_ticks(ax: plt.Axes, x_vals: np.ndarray, height: float, lw: float, alpha: float) -> None:
    x_vals = np.asarray(x_vals, dtype=float)
    x_vals = x_vals[np.isfinite(x_vals)]
    if x_vals.size == 0:
        return

    ax.vlines(
        x_vals,
        ymin=0.0,
        ymax=height,
        transform=ax.get_xaxis_transform(),
        linewidth=lw,
        alpha=alpha,
        zorder=1,
    )
    x0 = float(np.median(x_vals))  # or np.mean(x_vals), or np.min(x_vals) if you prefer left-aligned
    ax.text(
        x0,
        height + 0.01,
        "ERA5 ssp585",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8.0,
        alpha=0.9,
        zorder=3,
    )
   

def _filter_distinct_points(df: pd.DataFrame, x_col: str, y_col: str, rel_tol: float) -> pd.DataFrame:
    x = _to_numeric(df[x_col])
    y = _to_numeric(df[y_col])
    m = _finite_mask(x, y)
    x = x[m]
    y = y[m]

    near = _near_diag_mask(x, y, rel_tol=rel_tol)
    keep = ~near

    idx_finite = np.flatnonzero(m)
    idx_keep = idx_finite[np.flatnonzero(keep)]
    return df.iloc[idx_keep].copy()


def _build_model_styles(versions: list[str]) -> dict[str, dict[str, object]]:
    """
    Assign each model (version) a distinct filled marker + color.
    Only filled markers to avoid edgecolor/facecolor warnings.
    """
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h", "H", "p", "8", "d"]
    cmap = plt.get_cmap("tab20")

    styles: dict[str, dict[str, object]] = {}
    for i, v in enumerate(versions):
        styles[v] = {"marker": markers[i % len(markers)], "color": cmap(i % cmap.N)}
    return styles


def _panel_scatter_by_model(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    model_styles: dict[str, dict[str, object]],
    rel_tol: float,
    s: float,
    alpha: float,
    show_rug: bool,
    rug_vals: np.ndarray | None,
    rug_height: float,
    rug_lw: float,
    rug_alpha: float,
) -> list[plt.Line2D]:
    dfp = _filter_distinct_points(df, x_col, y_col, rel_tol=rel_tol)
    handles: list[plt.Line2D] = []

    if show_rug and rug_vals is not None:
        _add_rug_ticks(ax, rug_vals, height=rug_height, lw=rug_lw, alpha=rug_alpha)

    for version in sorted(dfp["version"].astype(str).unique()):
        g = dfp[dfp["version"].astype(str) == version]
        x = _to_numeric(g[x_col])
        y = _to_numeric(g[y_col])
        m = _finite_mask(x, y)
        x = x[m]
        y = y[m]
        if x.size == 0:
            continue

        st = model_styles.get(version, {"marker": "o", "color": None})

        ax.scatter(
            x, y,
            s=s,
            alpha=alpha,
            marker=st["marker"],
            facecolors=st["color"],
            edgecolors="none",
            zorder=2,
        )

        h = plt.Line2D(
            [], [],
            linestyle="None",
            marker=st["marker"],
            markersize=np.sqrt(s) * 0.6,
            markerfacecolor=st["color"],
            markeredgecolor="none",
            alpha=alpha,
            label=version,
        )
        handles.append(h)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    x_all = _to_numeric(dfp[x_col])
    y_all = _to_numeric(dfp[y_col])
    m_all = _finite_mask(x_all, y_all)
    x_all = x_all[m_all]
    y_all = y_all[m_all]
    if x_all.size > 0 and y_all.size > 0:
        xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
        ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
        xr = xmax - xmin
        yr = ymax - ymin
        pad_x = 0.06 * (xr if xr > 0 else 1.0)
        pad_y = 0.06 * (yr if yr > 0 else 1.0)
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

    _draw_diag(ax, ax.get_xlim(), ax.get_ylim())
    ax.minorticks_on()
    return handles


def _unique_handles(handles: list[plt.Line2D]) -> list[plt.Line2D]:
    seen = set()
    out: list[plt.Line2D] = []
    for h in handles:
        lab = h.get_label()
        if lab in seen:
            continue
        seen.add(lab)
        out.append(h)
    return out


def _legend_layout_for_height(
    n_items: int,
    available_points: float,
    base_fontsize: float,
    min_fontsize: float,
    max_cols: int,
    line_factor: float = 1.15,
) -> tuple[int, float]:
    """
    Choose (ncol, fontsize) so legend fits within available_points (vertical).
    Approximate each row height as line_factor * fontsize points.
    """
    if n_items <= 0:
        return 1, base_fontsize

    # Try increasing columns to reduce rows; then reduce fontsize if needed.
    best_ncol = 1
    best_fs = base_fontsize

    for ncol in range(1, max_cols + 1):
        rows = int(ceil(n_items / ncol))
        required = rows * line_factor * base_fontsize
        if required <= available_points:
            return ncol, base_fontsize  # fits at base size

        # If not fitting at base fontsize, compute required fontsize to fit.
        fs_fit = available_points / (rows * line_factor)
        if fs_fit >= min_fontsize:
            # pick the first (smallest columns) that can fit with >= min fontsize
            return ncol, fs_fit

        # Track best attempt (largest fontsize) among failures, as fallback
        if fs_fit > best_fs:
            best_fs = fs_fit
            best_ncol = ncol

    # If nothing fits even at min_fontsize, use max_cols and clamp fontsize to min
    return max_cols, min_fontsize


def _add_right_side_legend(
    fig: plt.Figure,
    ax_ref: plt.Axes,
    handles: list[plt.Line2D],
    base_fontsize: float,
    min_fontsize: float,
    max_cols: int,
    xpad: float,
) -> None:
    """
    Place a legend on the right side, vertically centered, sized to fit within the axes' height.
    """
    handles = _unique_handles(handles)
    if not handles:
        return

    # Available vertical space in points, based on the axes height in the figure.
    bbox = ax_ref.get_position()  # figure fraction
    fig_h_in = fig.get_figheight()
    available_points = bbox.height * fig_h_in * 72.0

    ncol, fs = _legend_layout_for_height(
        n_items=len(handles),
        available_points=available_points,
        base_fontsize=base_fontsize,
        min_fontsize=min_fontsize,
        max_cols=max_cols,
    )

    # Place legend outside to the right of the reference axes.
    ax_ref.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="center left",
        bbox_to_anchor=(1.0 + xpad, 0.5),
        frameon=False,
        fontsize=fs,
        ncol=ncol,
        borderaxespad=0.0,
        handletextpad=0.5,
        columnspacing=0.9,
        labelspacing=0.35,
    )


def save_three_figures(
    df_ssp: pd.DataFrame,
    df_era: pd.DataFrame,
    out_png: str | Path,
    x1_col: str,
    y1_col: str,
    x2_col: str,
    y2_col: str,
    rel_tol: float,
    s: float,
    alpha: float,
    rug_height: float,
    rug_lw: float,
    rug_alpha: float,
    dpi: int,
    legend_fontsize: float,
    legend_min_fontsize: float,
    legend_max_cols: int,
    legend_xpad: float,
) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    stem = out_png.name
    if stem.lower().endswith(".png"):
        base = stem[:-4]
    else:
        base = stem
        out_png = out_png.with_name(base + ".png")

    out_left = out_png.with_name(base + "_left.png")
    out_right = out_png.with_name(base + "_right.png")

    era_vals = _to_numeric(df_era["trend_yera_hat"]) if "trend_yera_hat" in df_era.columns else np.array([])

    versions = sorted(df_ssp["version"].astype(str).unique()) if "version" in df_ssp.columns else []
    model_styles = _build_model_styles(versions)

    # --- Combined figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2))

    handles1 = _panel_scatter_by_model(
        ax=ax1,
        df=df_ssp,
        x_col=x1_col,
        y_col=y1_col,
        x_label="Observed Model Warming",
        y_label="Observed Model Warming",
        model_styles=model_styles,
        rel_tol=rel_tol,
        s=s,
        alpha=alpha,
        show_rug=False,
        rug_vals=None,
        rug_height=rug_height,
        rug_lw=rug_lw,
        rug_alpha=rug_alpha,
    )

    handles2 = _panel_scatter_by_model(
        ax=ax2,
        df=df_ssp,
        x_col=x2_col,
        y_col=y2_col,
        x_label="Estimated Model Warming",
        y_label="Observed Model Warming",
        model_styles=model_styles,
        rel_tol=rel_tol,
        s=s,
        alpha=alpha,
        show_rug=True,
        rug_vals=era_vals,
        rug_height=rug_height,
        rug_lw=rug_lw,
        rug_alpha=rug_alpha,
    )

    # Leave room on the right for the legend
    fig.subplots_adjust(left=0.10, right=0.80, bottom=0.18, top=0.98, wspace=0.22)

    # Legend on the right side, sized to fit vertical space of ax2
    _add_right_side_legend(
        fig=fig,
        ax_ref=ax2,
        handles=handles1 + handles2,
        base_fontsize=legend_fontsize,
        min_fontsize=legend_min_fontsize,
        max_cols=legend_max_cols,
        xpad=legend_xpad,
    )

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # --- Left-only ---
    figL, axL = plt.subplots(1, 1, figsize=(4.0, 3.2))
    handlesL = _panel_scatter_by_model(
        ax=axL,
        df=df_ssp,
        x_col=x1_col,
        y_col=y1_col,
        x_label="Observed Model Warming",
        y_label="Observed Model Warming",
        model_styles=model_styles,
        rel_tol=rel_tol,
        s=s,
        alpha=alpha,
        show_rug=False,
        rug_vals=None,
        rug_height=rug_height,
        rug_lw=rug_lw,
        rug_alpha=rug_alpha,
    )
    figL.subplots_adjust(left=0.16, right=0.78, bottom=0.18, top=0.98)
    _add_right_side_legend(
        fig=figL,
        ax_ref=axL,
        handles=handlesL,
        base_fontsize=legend_fontsize,
        min_fontsize=legend_min_fontsize,
        max_cols=legend_max_cols,
        xpad=legend_xpad,
    )
    figL.savefig(out_left, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(figL)

    # --- Right-only ---
    figR, axR = plt.subplots(1, 1, figsize=(4.0, 3.2))
    handlesR = _panel_scatter_by_model(
        ax=axR,
        df=df_ssp,
        x_col=x2_col,
        y_col=y2_col,
        x_label="Estimated Model Warming",
        y_label="Observed Model Warming",
        model_styles=model_styles,
        rel_tol=rel_tol,
        s=s,
        alpha=alpha,
        show_rug=True,
        rug_vals=era_vals,
        rug_height=rug_height,
        rug_lw=rug_lw,
        rug_alpha=rug_alpha,
    )
    figR.subplots_adjust(left=0.16, right=0.78, bottom=0.18, top=0.98)
    _add_right_side_legend(
        fig=figR,
        ax_ref=axR,
        handles=handlesR,
        base_fontsize=legend_fontsize,
        min_fontsize=legend_min_fontsize,
        max_cols=legend_max_cols,
        xpad=legend_xpad,
    )
    figR.savefig(out_right, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(figR)

    print(f"Saved combined figure: {out_png}")
    print(f"Saved left panel:      {out_left}")
    print(f"Saved right panel:     {out_right}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-panel warming comparison figure + ERA+residual sampling quantiles."
    )
    p.add_argument("--data_ssp", type=str, default=data_ssp, help="Path to SSP CSV file.")
    p.add_argument("--data_era", type=str, default=data_era, help="Path to ERA CSV file.")
    p.add_argument("--nsim", type=int, default=nsim, help="Number of simulations/samples for quantiles.")
    p.add_argument("--seed", type=int, default=seed, help="Random seed (set to -1 for no seed).")

    p.add_argument("--output_png", type=str, default=output_png, help="Output PNG path for combined figure.")
    p.add_argument("--dpi", type=int, default=dpi, help="DPI for saved PNGs.")

    p.add_argument("--point_size", type=float, default=point_size, help="Scatter point size.")
    p.add_argument("--point_alpha", type=float, default=point_alpha, help="Scatter point alpha.")

    # Avoid '%' in argparse help strings (or escape as '%%') to prevent formatting errors.
    p.add_argument(
        "--diag_rel_tol",
        type=float,
        default=diag_rel_tol,
        help="Relative tolerance for removing near-diagonal points (default 1e-3 = 0.1 percent).",
    )

    p.add_argument("--rug_height", type=float, default=rug_height, help="Height of ERA rug ticks (axes fraction).")
    p.add_argument("--rug_lw", type=float, default=rug_lw, help="Line width of ERA rug ticks.")
    p.add_argument("--rug_alpha", type=float, default=rug_alpha, help="Alpha of ERA rug ticks.")

    p.add_argument("--legend_fontsize", type=float, default=legend_fontsize, help="Base legend font size.")
    p.add_argument("--legend_min_fontsize", type=float, default=legend_min_fontsize, help="Minimum legend font size.")
    p.add_argument("--legend_max_cols", type=int, default=legend_max_cols, help="Maximum legend columns.")
    p.add_argument("--legend_xpad", type=float, default=legend_xpad, help="Legend x padding to the right.")

    # Column mapping knobs
    p.add_argument("--left_x_col", type=str, default="trend_y_truesecond",
                   help="Column used as x for left panel (observed).")
    p.add_argument("--left_y_col", type=str, default="trend_y_true",
                   help="Column used as y for left panel (observed).")
    p.add_argument("--right_x_col", type=str, default="trend_y_hat",
                   help="Column used as x for right panel (estimated).")
    p.add_argument("--right_y_col", type=str, default="trend_y_true",
                   help="Column used as y for right panel (observed).")

    return p.parse_args()


def main() -> int:
    _set_journal_style()
    args = _parse_args()

    seed_arg: int | None = None if args.seed == -1 else int(args.seed)

    try:
        df_ssp, df_era = read_inputs(args.data_ssp, args.data_era)

        for col in (args.left_x_col, args.left_y_col, args.right_x_col, args.right_y_col, "version"):
            if col not in df_ssp.columns:
                raise KeyError(f"data_ssp missing required column '{col}'. Found: {list(df_ssp.columns)}")
        if "trend_yera_hat" not in df_era.columns:
            raise KeyError(f"data_era missing required column 'trend_yera_hat'. Found: {list(df_era.columns)}")

        samples = simulate_distribution(df_ssp=df_ssp, df_era=df_era, nsim=args.nsim, seed=seed_arg)
        q = compute_quantiles(samples)

        save_three_figures(
            df_ssp=df_ssp,
            df_era=df_era,
            out_png=args.output_png,
            x1_col=args.left_x_col,
            y1_col=args.left_y_col,
            x2_col=args.right_x_col,
            y2_col=args.right_y_col,
            rel_tol=args.diag_rel_tol,
            s=args.point_size,
            alpha=args.point_alpha,
            rug_height=args.rug_height,
            rug_lw=args.rug_lw,
            rug_alpha=args.rug_alpha,
            dpi=args.dpi,
            legend_fontsize=args.legend_fontsize,
            legend_min_fontsize=args.legend_min_fontsize,
            legend_max_cols=args.legend_max_cols,
            legend_xpad=args.legend_xpad,
        )

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print("Quantiles for sampled ERA+residual distribution:")
    for k in ["2.5%", "5%", "10%", "90%", "95%", "97.5%"]:
        print(f"  {k:>5} : {q[k]:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
