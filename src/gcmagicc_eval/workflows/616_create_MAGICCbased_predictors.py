# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MAGICC predictors with optional ERA5 splicing (lean)
#
# This stripped-down notebook script rebuilds the ERA5-spliced predictor
# generation from the `616_*` series without importing those files. The steps
# are intentionally compact:
# 1. Load ERA5 predictor series from a vetted NetCDF.
# 2. Load MAGICC scenario-by-scenario parquet files across resampled packages
#    (AR6/AR7, runmode `all`, `natural`, and `aerosols`) and turn them into
#    monthly predictor time series.
# 3. Optionally sample a handful of CMIP6 NetCDF files to plot alongside.
# 4. Splice MAGICC predictors to ERA5 for the configured runmodes using the
#    same shift/scale/guarded-scale logic as `616_*`, plot raw vs. spliced series,
#    and export the predictors
#    predictors in a CSV/HDF5 layout consumable by
#    `320_run_probabilistic_segments_gcmagicc.py` and
#    `321_run_probabilistic_ensembles_gcmagicc.py`.
#
# Every major block below is separated by a cell to keep the workflow readable.

# %%
from __future__ import annotations

import os
import re
import random
import fnmatch
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# %% [markdown]
# ## Configuration
# Defaults mirror the prior `616_*` scripts but live entirely in this file.
# Paths and toggles can be overridden via environment variables to keep the
# workflow reproducible.

# %%
# Core paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "data" / "newscenario_inputs"))
_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_predictor_output_root_env = os.environ.get("PREDICTOR_OUTPUT_ROOT", "").strip()
PREDICTOR_OUTPUT_ROOT = (
    Path(_predictor_output_root_env)
    if _predictor_output_root_env
    else OUTPUT_DIR / f"magicc_based_predictors_{_timestamp}"
)
COMPARISON_PLOTS_SUBDIR = "comparison_plots"

ETH_PROJECTS_ROOT = Path("data/site_eth/projects")
GUS_PROJECTS_ROOT = Path("data/site_gus/projects")
ETH_MAGICC_RESAMPLED_ROOT = ETH_PROJECTS_ROOT / "2025magicc"
GUS_MAGICC_RESAMPLED_ROOT = GUS_PROJECTS_ROOT / "2025magicc"
ETH_ERA5_DIR = Path("data/site_eth/out_ERA5_4July2025_1degree_vetted")
GUS_ERA5_DIR = Path("data/archive/ERA5/processed/out_ERA5_19Feb2026_1degree_vetted")


def _path_under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _normalise_site(value: str) -> Optional[str]:
    site = value.strip().lower()
    if site in {"eth", "gus"}:
        return site
    return None


def _detect_site() -> str:
    env_site = _normalise_site(os.environ.get("GCMAGICC_SITE", ""))
    if env_site:
        return env_site
    if _path_under(PROJECT_ROOT, GUS_PROJECTS_ROOT):
        return "gus"
    if _path_under(PROJECT_ROOT, ETH_PROJECTS_ROOT):
        return "eth"
    hostname = os.environ.get("HOSTNAME", "").strip().lower()
    if hostname.startswith("gus"):
        return "gus"
    return "eth"


def _default_magicc_resampled_root() -> Path:
    return GUS_MAGICC_RESAMPLED_ROOT if _detect_site() == "gus" else ETH_MAGICC_RESAMPLED_ROOT


def _resolve_magicc_resampled_root() -> Path:
    root = os.environ.get("MAGICC_RESAMPLED_ROOT", "").strip()
    return Path(root).expanduser() if root else _default_magicc_resampled_root()


# Input data
MAGICC_RESAMPLED_ROOT = _resolve_magicc_resampled_root()
MAGICC_RESAMPLED_PREFIX = os.environ.get("MAGICC_RESAMPLED_PREFIX", "output_resampled_")
_RESAMPLE_SIZES_RAW = os.environ.get("MAGICC_RESAMPLE_SIZES", "2,10,20,50,100")
RESAMPLE_SIZES: List[int] = []
for _part in _RESAMPLE_SIZES_RAW.split(","):
    _part = _part.strip()
    if not _part:
        continue
    try:
        RESAMPLE_SIZES.append(int(_part))
    except ValueError as exc:
        raise ValueError(f"Invalid MAGICC_RESAMPLE_SIZES entry: {_part!r}") from exc

WORKFLOWS: List[str] = [w.strip().upper() for w in os.environ.get("MAGICC_WORKFLOWS", "AR6,AR7").split(",") if w.strip()]
RUNMODES: List[str] = [m.strip().lower() for m in os.environ.get("MAGICC_RUNMODES", "all,natural,aerosols").split(",") if m.strip()]
RUNMODES_SPLICED: List[str] = [
    m.strip().lower()
    for m in os.environ.get("MAGICC_RUNMODES_SPLICED", "all").split(",")
    if m.strip()
]
SCENARIO_WHITELIST_DEFAULT: List[str] = [] # ["SSP2-com*",] # "H_*", "HL_*", "L_*", "LN_*", "M_*", "VL_*", "Current-Policies*"]
_SCENARIO_WHITELIST_RAW = os.environ.get("SCENARIO_WHITELIST", "")
if _SCENARIO_WHITELIST_RAW.strip():
    _parts = [p.strip() for p in re.split(r"[;,]", _SCENARIO_WHITELIST_RAW) if p.strip()]
    if len(_parts) == 1 and _parts[0].lower() in {"*", "all", "none", "off"}:
        SCENARIO_WHITELIST: List[str] = []
    else:
        SCENARIO_WHITELIST = _parts
else:
    SCENARIO_WHITELIST = list(SCENARIO_WHITELIST_DEFAULT)

ERA5_FILENAME = (
    "DAT_ERA5_historical-ERA5_r1i1p1f1_clt-day-hurs-huss-month-pr-psl-rlut-rsds-rsdt-rsnt-rtmt-"
    "sfcWind-tas-tasmax-tasmin-ts-year.nc"
)


def _default_era5_dir() -> Path:
    return GUS_ERA5_DIR if _detect_site() == "gus" else ETH_ERA5_DIR


def _resolve_era5_file() -> Path:
    era5_file = os.environ.get("ERA5_FILE", "").strip()
    if era5_file:
        return Path(era5_file).expanduser()
    era5_dir = os.environ.get("ERA5_DIR", "").strip()
    return (Path(era5_dir).expanduser() if era5_dir else _default_era5_dir()) / ERA5_FILENAME


ERA5_FILE = _resolve_era5_file()

CMIP6_DIR = Path(os.environ.get("CMIP6_DIR", "data/site_eth/out_ETHFOG_10June2025_vetted"))
CMIP6_FILENAME_FILTER: List[str] = [s.strip() for s in os.environ.get("CMIP6_FILENAME_FILTER", "ssp,historical").split(",") if s.strip()]
N_CMIP6_RANDOM = int(os.environ.get("N_CMIP6_RANDOM", 25))
CMIP6_RANDOM_SEED = int(os.environ.get("CMIP6_RANDOM_SEED", 2025))

# Time windows
START_YEAR = int(os.environ.get("START_YEAR", 1850))
END_YEAR = int(os.environ.get("END_YEAR", 2100))
BASELINE_PERIOD: Tuple[int, int] = (
    int(os.environ.get("BASELINE_START", 1995)),
    int(os.environ.get("BASELINE_END", 2014)),
)

# ERA5 splice settings (mirrors 616_* behaviour)
ERA5_NORMALISATION_START_PERIOD = int(os.environ.get("ERA5_NORMALISATION_START_PERIOD", 5))  # years fading into ERA5
ERA5_NORMALISATION_END_PERIOD = int(os.environ.get("ERA5_NORMALISATION_END_PERIOD", 1))  # years fading out of ERA5
ERA5_NORMALISATION_START_YR = os.environ.get("ERA5_NORMALISATION_START_YR", 1945)
ERA5_NORMALISATION_END_YR = os.environ.get("ERA5_NORMALISATION_END_YR", 2023)
ERA5_NORMALISATION_START_YR = int(ERA5_NORMALISATION_START_YR) if ERA5_NORMALISATION_START_YR else None
ERA5_NORMALISATION_END_YR = int(ERA5_NORMALISATION_END_YR) if ERA5_NORMALISATION_END_YR else None

SPLICE_GUARDED_SCALE_MIN = float(os.environ.get("SPLICE_GUARDED_SCALE_MIN", -3.0))
SPLICE_GUARDED_SCALE_MAX = float(os.environ.get("SPLICE_GUARDED_SCALE_MAX", 3.0))
SPLICE_METHOD_BY_PRED: Dict[str, str] = {
    "tas_smoothed": os.environ.get("SPLICE_MODE_TAS_SMOOTHED", "shift"),
    "rtmt_smoothed": os.environ.get("SPLICE_MODE_RTMT_SMOOTHED", "shift"),
    "GHG_ERF": os.environ.get("SPLICE_MODE_GHG_ERF", "shift"),
    "CO2_ERF": os.environ.get("SPLICE_MODE_CO2_ERF", "shift"),
    "aer_ERF": os.environ.get("SPLICE_MODE_AER_ERF", "guarded_scale"),
    "totalO3_ERF": os.environ.get("SPLICE_MODE_TOTALO3_ERF", "guarded_scale"),
    "stratO3_ERF": os.environ.get("SPLICE_MODE_STRATO3_ERF", "guarded_scale"),
    "sol_ERF": os.environ.get("SPLICE_MODE_SOL_ERF", "shift"),
    "volc_ERF": os.environ.get("SPLICE_MODE_VOLC_ERF", "shift"),
    "other_ERF": os.environ.get("SPLICE_MODE_OTHER_ERF", "shift"),
}

# Predictors to plot/export (order matches GCMagicc meta['variables_X'] expectations)
PREDICTOR_COLUMNS_ORDER: List[str] = [
    "model_index",
    "month",
    "sin_time",
    "cos_time",
    "tas_smoothed",
    "rtmt_smoothed",
    "stratO3_ERF",
    "sol_ERF",
    "other_ERF",
    "volc_ERF",
    "nat_ERF",
    "totalO3_ERF",
    "GHG_ERF",
    "aer_ERF",
    "CO2_ERF",
]
VISUAL_TARGETS = [
    "tas_smoothed",
    "rtmt_smoothed",
    "stratO3_ERF",
    "sol_ERF",
    "other_ERF",
    "volc_ERF",
    "nat_ERF",
    "totalO3_ERF",
    "GHG_ERF",
    "aer_ERF",
    "CO2_ERF",
]

# Cache for tas anchoring offsets keyed by (scenario_label_clean, run_id_str)
TAS_ANCHOR_OFFSETS: Dict[Tuple[str, str], float] = {}

# Region filtering for MAGICC parquet (global only)
MAGICC_REGION_PATS: List[str] = ["World", "Global", "GLO"]

# Model index to stamp into exported predictor arrays (0 = ERA5)
PREDICTOR_MODEL_INDEX_DEFAULT = int(os.environ.get("PREDICTOR_MODEL_INDEX", 0))

# Parallelization
N_WORKERS = int(os.environ.get("N_WORKERS", min(80, os.cpu_count() or 1)))

# Plot styling
ERA5_STYLE = dict(color="black", lw=2.5, alpha=0.5, label="ERA5 (historical)", zorder=10)
MAGICC_STYLE = dict(color="#1b6ef3", lw=2.5, alpha=0.5, label="MAGICC raw", zorder=3)
SPLICED_STYLE = dict(color="#0a9237", lw=2.5, alpha=0.5, label="MAGICC (ERA5 spliced)", zorder=4)
CMIP6_STYLE = dict(color="#c44e52", lw=0.8, alpha=0.5, label="CMIP6 sample", zorder=2)

# Ensemble member colors (lighter versions of mean colors)
MAGICC_RAW_ENSEMBLE_COLOR = "#7db3f0"  # lighter blue
SPLICED_ENSEMBLE_COLOR = "#5cc97a"  # lighter green

# tas_smoothed period statistics
TAS_PERIOD_WINDOWS: Dict[str, Tuple[int, int]] = {
    "1995-2014": (1995, 2014),
    "2081-2100": (2081, 2100),
}
TAS_DIFF_PLUS_OFFSET = 0.85  # IPCC reference adjustment
TAS_DIFF_COLOR = "#7b2cbf"
TAS_PERIOD_TEXT_FONTSIZE = 10

# %% [markdown]
# ## Helper utilities
# Only the minimum utilities needed for the splice and aggregation logic are
# kept here. Everything is self contained to avoid coupling to archived files.

# %%
_WS_RX = re.compile(r"\s+")
_BADCHARS_RX = re.compile(r"[^A-Za-z0-9._-]+")


def scenario_safe_name(s: str) -> str:
    """
    Make a human-readable label safe for filesystem paths (similar to 620_*).
    """
    name = _WS_RX.sub("-", str(s).strip())
    name = name.replace("/", "-").replace("\\", "-")
    name = _BADCHARS_RX.sub("-", name)
    return name.strip("-")


def scenario_allowed(scenario_name: str, scenario_stem: Optional[str] = None) -> bool:
    if not SCENARIO_WHITELIST:
        return True
    for pattern in SCENARIO_WHITELIST:
        if fnmatch.fnmatchcase(scenario_name, pattern):
            return True
        if scenario_stem and scenario_stem != scenario_name and fnmatch.fnmatchcase(scenario_stem, pattern):
            return True
    return False


def monthly_index(start_year: int, end_year: int) -> pd.DatetimeIndex:
    """Return a month-start index spanning start_year..end_year inclusive."""
    return pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")


def _index_to_fractional_year(idx) -> np.ndarray:
    """Convert a datetime-like index to fractional years (yyyy.fraction)."""
    if hasattr(idx, "year") and hasattr(idx, "month"):
        years = np.asarray(idx.year, dtype=np.float64)
        months = np.asarray(idx.month, dtype=np.float64)
        days = np.asarray(getattr(idx, "day", np.ones_like(years)), dtype=np.float64)
    else:
        vals = list(idx)
        years = np.array([getattr(t, "year", np.nan) for t in vals], dtype=np.float64)
        months = np.array([getattr(t, "month", 1.0) for t in vals], dtype=np.float64)
        days = np.array([getattr(t, "day", 1.0) for t in vals], dtype=np.float64)
    return years + (months - 1.0) / 12.0 + (days - 1.0) / 365.0


def _coerce_datetime_index(series: pd.Series) -> pd.Series:
    """
    Ensure a series index is a DatetimeIndex. Falls back to constructing
    datetimes from year/month/day attributes when needed.
    """
    if series is None or series.empty:
        return series
    if isinstance(series.index, pd.DatetimeIndex):
        return series
    try:
        coerced = pd.to_datetime(series.index)
        series = series.copy()
        series.index = coerced
        return series
    except Exception:
        def _to_dt(val):
            y = getattr(val, "year", None)
            m = getattr(val, "month", 1) or 1
            d = getattr(val, "day", 1) or 1
            if y is None:
                return pd.NaT
            if y == 0:
                y = 1
            return datetime(int(y), int(m), int(d))
        new_idx = pd.DatetimeIndex([_to_dt(v) for v in series.index])
        series = series.copy()
        series.index = new_idx
        return series


def _align_to_month_start(series: pd.Series) -> pd.Series:
    """
    Snap index to month starts and collapse duplicates (mean). Prevents double
    rows when sources use different intra-month anchors.
    """
    if series is None or series.empty:
        return series
    s = _coerce_datetime_index(series).sort_index()
    month_idx = s.index.to_period("M")
    collapsed = s.groupby(month_idx).mean()
    collapsed.index = collapsed.index.to_timestamp(how="start")
    return collapsed.astype("float32")


def _trend_extend(series: pd.Series, years: int) -> pd.Series:
    """Extend a series backward/forward with a linear trend over `years`."""
    if series is None or series.empty:
        return series
    s = series.sort_index()
    start_year = int(s.index.min().year)
    end_year = int(s.index.max().year)
    if end_year - start_year < 1:
        return s

    # Start slope
    start_cutoff = start_year + years - 1
    start_window = s[s.index.year <= start_cutoff].dropna()
    if len(start_window) >= 2:
        x0 = _index_to_fractional_year(start_window.index)
        y0 = start_window.values.astype("float64")
        m0, b0 = np.polyfit(x0, y0, 1)
        extra_years = np.arange(start_year - years, start_year, dtype=int)
        extra_idx_start = pd.date_range(f"{extra_years[0]}-01-01", f"{extra_years[-1]}-12-01", freq="MS")
        extra_x_start = _index_to_fractional_year(extra_idx_start)
        extra_y_start = m0 * extra_x_start + b0
        s = pd.concat([pd.Series(extra_y_start.astype("float32"), index=extra_idx_start), s])
    elif len(start_window) == 1:
        extra_years = np.arange(start_year - years, start_year, dtype=int)
        extra_idx_start = pd.date_range(f"{extra_years[0]}-01-01", f"{extra_years[-1]}-12-01", freq="MS")
        extra_y_start = np.full(len(extra_idx_start), float(start_window.iloc[0]), dtype="float32")
        s = pd.concat([pd.Series(extra_y_start, index=extra_idx_start), s])

    # End slope
    end_cutoff = end_year - years + 1
    end_window = s[s.index.year >= end_cutoff].dropna()
    if len(end_window) >= 2:
        x1 = _index_to_fractional_year(end_window.index)
        y1 = end_window.values.astype("float64")
        m1, b1 = np.polyfit(x1, y1, 1)
        extra_years = np.arange(end_year + 1, end_year + years + 1, dtype=int)
        extra_idx_end = pd.date_range(f"{extra_years[0]}-01-01", f"{extra_years[-1]}-12-01", freq="MS")
        extra_x_end = _index_to_fractional_year(extra_idx_end)
        extra_y_end = m1 * extra_x_end + b1
        s = pd.concat([s, pd.Series(extra_y_end.astype("float32"), index=extra_idx_end)])
    elif len(end_window) == 1:
        extra_years = np.arange(end_year + 1, end_year + years + 1, dtype=int)
        extra_idx_end = pd.date_range(f"{extra_years[0]}-01-01", f"{extra_years[-1]}-12-01", freq="MS")
        extra_y_end = np.full(len(extra_idx_end), float(end_window.iloc[-1]), dtype="float32")
        s = pd.concat([s, pd.Series(extra_y_end, index=extra_idx_end)])

    return s.sort_index()


def _running_mean_extend(series: pd.Series, window_years: int = 20) -> pd.Series:
    """Centered running mean with trend extension to keep a fixed window."""
    if series is None or series.empty:
        return series
    s_ext = _trend_extend(series, window_years)
    win = window_years * 12  # months
    smoothed = s_ext.rolling(window=win, center=True, min_periods=win).mean()
    smoothed = smoothed.loc[series.index.min():series.index.max()]
    return smoothed.astype("float32")


def _mean_offset(src: pd.Series, tgt: pd.Series, window_start: pd.Timestamp, window_end: pd.Timestamp) -> float:
    """Return tgt-src mean difference over a window; 0 if insufficient data."""
    if window_start is None or window_end is None or window_start > window_end:
        return 0.0
    mask = (src.index >= window_start) & (src.index <= window_end)
    if not mask.any():
        return 0.0
    src_mean = src.loc[mask].mean()
    tgt_mean = tgt.loc[mask].mean()
    if pd.isna(src_mean) or pd.isna(tgt_mean):
        return 0.0
    return float(tgt_mean - src_mean)


def _mean_scale(src: pd.Series, tgt: pd.Series, window_start: pd.Timestamp, window_end: pd.Timestamp) -> float:
    """Return multiplicative factor to align src mean to tgt mean over window."""
    if window_start is None or window_end is None or window_start > window_end:
        return 1.0
    mask = (src.index >= window_start) & (src.index <= window_end)
    if not mask.any():
        return 1.0
    src_mean = src.loc[mask].mean()
    tgt_mean = tgt.loc[mask].mean()
    if pd.isna(src_mean) or pd.isna(tgt_mean):
        return 1.0
    if abs(src_mean) < 1e-8:
        return 1.0
    return float(tgt_mean / src_mean)


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _fade_weights(idx: pd.Index) -> np.ndarray:
    """Return weights in [0,1] spanning the provided index (0 at start, 1 at end)."""
    if len(idx) == 0:
        return np.array([], dtype=np.float32)
    if len(idx) == 1:
        return np.array([1.0], dtype=np.float32)
    years = _index_to_fractional_year(idx)
    span = years.max() - years.min()
    if span == 0:
        return np.linspace(0.0, 1.0, num=len(idx), dtype=np.float32)
    return ((years - years.min()) / span).astype(np.float32)


def _period_mean(series: Optional[pd.Series], start_year: int, end_year: int) -> Optional[float]:
    """Return the mean over a year window for a series (nan-safe)."""
    if series is None or series.empty:
        return None
    s = _coerce_datetime_index(series)
    mask = (s.index.year >= start_year) & (s.index.year <= end_year)
    if not mask.any():
        return None
    vals = pd.to_numeric(s[mask], errors="coerce")
    if not vals.notna().any():
        return None
    return float(vals.mean(skipna=True))


def _summary_quantiles(values: Sequence[Optional[float]]) -> Optional[Dict[str, float]]:
    """Compute 5/50/95% stats for a list of values, skipping nan/None."""
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return None
    return {
        "p05": float(np.nanpercentile(arr, 5)),
        "median": float(np.nanmedian(arr)),
        "p95": float(np.nanpercentile(arr, 95)),
        "count": int(arr.size),
    }


def _period_and_delta_stats(
    series_dict: Dict[str, pd.Series],
    windows: Dict[str, Tuple[int, int]],
    base_key: str,
    future_key: str,
) -> Tuple[Dict[str, Optional[Dict[str, float]]], Optional[Dict[str, float]]]:
    """
    Compute per-period quantiles (5/50/95) and warming deltas between two
    named periods for the provided series.
    """
    per_period: Dict[str, List[float]] = {name: [] for name in windows}
    diffs: List[float] = []

    for _, series in series_dict.items():
        base_mean: Optional[float] = None
        future_mean: Optional[float] = None

        for name, (start, end) in windows.items():
            m = _period_mean(series, start, end)
            if m is None:
                continue
            per_period[name].append(m)
            if name == base_key:
                base_mean = m
            if name == future_key:
                future_mean = m

        if base_mean is not None and future_mean is not None:
            diffs.append(future_mean - base_mean)

    period_stats = {name: _summary_quantiles(vals) for name, vals in per_period.items()}
    diff_stats = _summary_quantiles(diffs)
    return period_stats, diff_stats


def _is_ssp_like(label: str) -> bool:
    """Heuristic: scenario names starting with ssp*/clean_ssp* (case-insensitive)."""
    low = (label or "").lower()
    return low.startswith("ssp") or low.startswith("clean_ssp") or "|ssp" in low or " ssp" in low


def _select_end_year_for_label(label: str, override: Optional[int], era5_end_year: int) -> int:
    """
    Pick the end-year for the ERA5 fade-out window:
    - explicit override if provided
    - else 2015 for SSP-like scenarios
    - else 2024
    """
    if override is not None:
        return min(override, era5_end_year)
    if _is_ssp_like(label):
        return min(2015, era5_end_year)
    return min(2024, era5_end_year)


def _clean_scenario_label(model_name: Optional[str], scenario_label: Optional[str], scenario_from_path: Optional[str]) -> str:
    """
    Prefer a human-readable scenario label without model prefixes.
    Example: model='NDCtool', scenario_label='NDCtool_NDC-Trump-low',
    scenario_from_path='NDC-Trump-low' -> 'NDC-Trump-low'.
    """
    label = str(scenario_label or "").strip()
    model = str(model_name or "").strip()
    scen_path = str(scenario_from_path or "").strip()

    def _strip_prefix(val: str, prefix: str) -> str:
        if not val or not prefix:
            return val
        for sep in ("_", "-", " "):
            pref = f"{prefix}{sep}"
            if val.lower().startswith(pref.lower()):
                return val[len(pref):]
        return val

    label = _strip_prefix(label, model)
    # If the parquet filename already encodes the scenario, prefer that suffix.
    if scen_path:
        if label.lower().endswith(scen_path.lower()):
            label = scen_path
        elif not label:
            label = scen_path

    if not label:
        label = scen_path or "unknown_scenario"
    return label


def _pred_units_adjusted_series(pred_key: str, series: Optional[pd.Series]) -> Optional[pd.Series]:
    """
    Convert predictor units for blending/plotting:
    - tas_smoothed: Kelvin -> degC, then divide by 10 (emulator scale)
    - others: passthrough
    """
    if series is None:
        return None
    if pred_key == "tas_smoothed":
        return ((series - 273.15) / 10.0).astype("float32")
    return series.astype("float32")


def _standardize_df(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """
    Convert MAGICC parquet wide format (year columns) into a tidy long DataFrame
    with columns: year, value, variable, region, scenario, model, unit, run_id.
    """
    d = df.copy()
    if isinstance(d.index, pd.MultiIndex):
        d = d.reset_index()
    year_cols = [c for c in d.columns if re.fullmatch(r"\d{4}", str(c))]
    if year_cols:
        id_cols = [c for c in d.columns if c not in year_cols]
        d = d.melt(id_vars=id_cols, value_vars=year_cols, var_name="year", value_name="value")

    def _first_col(names: Sequence[str]) -> Optional[str]:
        low = {str(c).lower(): c for c in d.columns}
        for n in names:
            key = str(n).lower()
            if key in low:
                return low[key]
        return None

    mapping = {
        "year": _first_col(["year", "time"]),
        "value": _first_col(["value", "val"]),
        "variable": _first_col(["variable", "var"]),
        "region": _first_col(["region", "area"]),
        "scenario": _first_col(["scenario", "scenario (name)"]),
        "model": _first_col(["model", "model (name)"]),
        "unit": _first_col(["unit", "units"]),
        "run_id": _first_col(["run_id", "run"]),
    }
    for std, orig in mapping.items():
        if orig and orig != std:
            d = d.rename(columns={orig: std})

    if "model" not in d.columns:
        d["model"] = Path(file_name).stem.split("_", 1)[0]
    if "scenario" not in d.columns:
        stem = Path(file_name).stem
        parts = stem.split("_", 1)
        d["scenario"] = parts[1] if len(parts) > 1 else stem

    d["source"] = file_name
    if "year" in d.columns:
        d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
    if "value" in d.columns:
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
    if "run_id" in d.columns:
        d["run_id"] = pd.to_numeric(d["run_id"], errors="coerce").astype("Int64")
    return d


def _run_ids_from_parquet(pq_path: Path) -> Set[str]:
    """Read only the run_id column (if present) from a parquet and return as strings."""
    try:
        df = pd.read_parquet(pq_path, columns=["run_id"])
    except Exception:
        try:
            df = pd.read_parquet(pq_path)
        except Exception:
            return {"single"}
    # If run_id lives in index (common for these parquets), pull it out
    if isinstance(df.index, pd.MultiIndex):
        if "run_id" in df.index.names:
            df = df.reset_index("run_id")
        else:
            df = df.reset_index()
    if "run_id" not in df.columns:
        return {"single"}
    vals = df["run_id"].dropna().unique()
    if len(vals) == 0:
        return {"single"}
    return {str(int(v)) for v in vals}


# %% [markdown]
# ## Data loading helpers
# These routines keep I/O minimal: read only what we need, align to a monthly
# index, and ensure every series is a `pd.Series` with a DatetimeIndex.

# %%
MAGICC_VARMAP = {
    "ERF_total": ("Effective Radiative Forcing|Total", "Effective Radiative Forcing"),
    "ERF_aer": ("Effective Radiative Forcing|Aerosols",),
    "ERF_co2": ("Effective Radiative Forcing|CO2",),
    "ERF_ghg": ("Effective Radiative Forcing|Greenhouse Gases",),
    "ERF_o3": ("Effective Radiative Forcing|Ozone",),
    "ERF_o3_str": ("Effective Radiative Forcing|Stratospheric Ozone",),
    "ERF_o3_tro": ("Effective Radiative Forcing|Tropospheric Ozone",),
    "ERF_sol": ("Effective Radiative Forcing|Solar",),
    "ERF_volc": ("Effective Radiative Forcing|Volcanic",),
    "SAT_change": ("Surface Air Temperature Change",),
    "HeatUptake": ("Heat Uptake",),
}

PRED_TO_MAGICC_KEY = {
    "tas_smoothed": "SAT_change",
    "rtmt_smoothed": "HeatUptake",
    "GHG_ERF": "ERF_ghg",
    "CO2_ERF": "ERF_co2",
    "aer_ERF": "ERF_aer",
    "totalO3_ERF": "ERF_o3",
    "stratO3_ERF": "ERF_o3_str",
    "sol_ERF": "ERF_sol",
    "volc_ERF": "ERF_volc",
    "other_ERF": "ERF_total",  # computed as total - components
}


def load_era5_predictors(nc_path: Path, start_year: int, end_year: int) -> Dict[str, pd.Series]:
    """
    Load ERA5 predictors from the vetted NetCDF and return as {name: Series}.
    """
    if not nc_path.exists():
        raise FileNotFoundError(f"ERA5 file not found: {nc_path}")
    with xr.open_dataset(nc_path) as ds:
        ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-30"))
        out: Dict[str, pd.Series] = {}
        for key in ["tas_smoothed", "rtmt_smoothed"]:
            if key in ds:
                out[key] = ds[key].to_series().astype("float32")
        erf_vars = ["stratO3_ERF", "sol_ERF", "other_ERF", "volc_ERF", "nat_ERF", "totalO3_ERF", "GHG_ERF", "aer_ERF", "CO2_ERF"]
        for var in erf_vars:
            if var in ds:
                out[var] = ds[var].to_series().astype("float32")
    return out


def load_cmip6_predictors(nc_path: Path, start_year: int, end_year: int) -> Dict[str, pd.Series]:
    """
    Load CMIP6 predictor series (global means) from a single NetCDF file.
    """
    with xr.open_dataset(nc_path) as ds:
        ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-30"))
        out: Dict[str, pd.Series] = {}
        for key in ["tas_smoothed", "rtmt_smoothed"]:
            if key in ds:
                out[key] = ds[key].to_series().astype("float32")
        erf_vars = ["stratO3_ERF", "sol_ERF", "other_ERF", "volc_ERF", "nat_ERF", "totalO3_ERF", "GHG_ERF", "aer_ERF", "CO2_ERF"]
        for var in erf_vars:
            if var in ds and "tas_smoothed" in out:
                out[var] = ds[var].to_series().reindex(out["tas_smoothed"].index).astype("float32")
    return out


def _list_cmip6_files(root: Path, filters: Sequence[str]) -> List[Path]:
    """
    Return NetCDF files whose filename contains any substring in `filters`.
    """
    files = sorted(root.rglob("*.nc"))
    if not filters:
        return files
    fl = [f.lower() for f in filters]
    return [p for p in files if any(tok in p.name.lower() for tok in fl)]


def pick_cmip6_samples(root: Path, filters: Sequence[str], n: int, seed: int) -> List[Path]:
    """Pick up to `n` random CMIP6 NetCDF files once for plotting overlays."""
    candidates = _list_cmip6_files(root, filters)
    if n < 0 or n >= len(candidates):
        return candidates
    random.seed(seed)
    return random.sample(candidates, n)


def _var_options(var_spec) -> tuple:
    return (var_spec,) if isinstance(var_spec, str) else tuple(var_spec)


def build_monthly_from_annual(df_rows: pd.DataFrame, start_year: int, end_year: int) -> pd.Series:
    """
    Convert annual values to monthly by interpolation.
    If multiple values exist per year (e.g., duplicates), they are averaged.
    For ensemble members processed separately, this typically yields a single value per year.
    """
    if df_rows.empty:
        return pd.Series(index=monthly_index(start_year, end_year), dtype="float32")
    annual = df_rows.groupby("year")["value"].mean().sort_index().astype(float)
    annual = annual[np.isfinite(annual.values)]
    m_idx = monthly_index(start_year, end_year)
    if annual.empty:
        return pd.Series(index=m_idx, dtype="float32")
    monthly_vals = np.interp(
        (m_idx.year + (m_idx.month - 1) / 12.0),
        annual.index.values.astype("float64"),
        annual.values.astype("float64"),
    )
    return pd.Series(monthly_vals, index=m_idx, dtype="float32")


def build_monthly_temperature_normalized(
    df_rows: pd.DataFrame,
    start_year: int,
    end_year: int,
    era5_baseline_temp: float,
    baseline_start: int = 1950,
    baseline_end: int = 2020,
    tas_offset: Optional[float] = None,
    return_offset: bool = False,
) -> Union[pd.Series, Tuple[pd.Series, float]]:
    """
    Normalize MAGICC tas to the ERA5 baseline (keeps Kelvin scale).
    If `tas_offset` is provided, it is applied directly; otherwise the offset is
    computed from the MAGICC baseline over the given period.
    When `return_offset` is True, the applied offset is returned alongside the series.
    """
    if df_rows.empty:
        empty = pd.Series(index=monthly_index(start_year, end_year), dtype="float32")
        return (empty, float("nan")) if return_offset else empty

    monthly_series = build_monthly_from_annual(df_rows, start_year, end_year)
    if tas_offset is None:
        baseline_mask = (monthly_series.index.year >= baseline_start) & (monthly_series.index.year <= baseline_end)
        magicc_baseline = monthly_series[baseline_mask].mean()
        tas_offset = era5_baseline_temp - magicc_baseline

    normalized = (monthly_series + tas_offset).astype("float32")
    if return_offset:
        return normalized, float(tas_offset)
    return normalized


def read_magicc_scenario(parquet_path: Path) -> pd.DataFrame:
    """
    Read one MAGICC parquet and return a tidy long dataframe filtered to global regions.
    """
    df0 = pd.read_parquet(parquet_path)
    d = _standardize_df(df0, parquet_path.name)
    if "region" in d.columns:
        rx = re.compile("|".join(MAGICC_REGION_PATS), re.IGNORECASE)
        d = d[d["region"].astype(str).str.contains(rx, na=False)]
    d = d.dropna(subset=["year", "value", "variable", "scenario"])
    d["year"] = d["year"].astype(int)
    return d


def scenario_predictors_from_long(
    exp_df: pd.DataFrame,
    start_year: int,
    end_year: int,
    era5_baseline_temp: float,
    tas_offset: Optional[float] = None,
    return_tas_offset: bool = False,
) -> Union[Dict[str, pd.Series], Tuple[Dict[str, pd.Series], float]]:
    """
    Build predictor series for one MAGICC experiment (model+scenario) using
    the same recipe as 616_*: annual -> monthly, tas normalized to ERA5,
    rtmt/tas smoothed with a 20y running mean, and ERF components combined.

    tas_offset: optional precomputed TAS offset (ERA5 baseline minus MAGICC baseline)
                to enforce consistent anchoring across runmodes/ensemble members.
    return_tas_offset: when True, return (predictor_dict, tas_offset_used).
    """
    r = {k: exp_df[exp_df["variable"].isin(_var_options(var_name))] for k, var_name in MAGICC_VARMAP.items()}

    erf_total = build_monthly_from_annual(r["ERF_total"][["year", "value"]], start_year, end_year)
    aer = build_monthly_from_annual(r["ERF_aer"][["year", "value"]], start_year, end_year)
    ghg = build_monthly_from_annual(r["ERF_ghg"][["year", "value"]], start_year, end_year)
    co2 = build_monthly_from_annual(r["ERF_co2"][["year", "value"]], start_year, end_year)
    o3 = build_monthly_from_annual(r["ERF_o3"][["year", "value"]], start_year, end_year)
    o3_str = build_monthly_from_annual(r["ERF_o3_str"][["year", "value"]], start_year, end_year)
    o3_tro = build_monthly_from_annual(r["ERF_o3_tro"][["year", "value"]], start_year, end_year)
    sol = build_monthly_from_annual(r["ERF_sol"][["year", "value"]], start_year, end_year)
    volc = build_monthly_from_annual(r["ERF_volc"][["year", "value"]], start_year, end_year)

    tas_norm_res = build_monthly_temperature_normalized(
        r["SAT_change"][["year", "value"]],
        start_year,
        end_year,
        era5_baseline_temp,
        baseline_start=BASELINE_PERIOD[0],
        baseline_end=BASELINE_PERIOD[1],
        tas_offset=tas_offset,
        return_offset=True,
    )
    tas_normalized, tas_offset_used = tas_norm_res
    tas_sm = _running_mean_extend(tas_normalized, window_years=20)

    hup = build_monthly_from_annual(r["HeatUptake"][["year", "value"]], start_year, end_year)
    rtmt_sm = _running_mean_extend(hup, window_years=20)

    total_o3 = o3 if not o3.isna().all() else (o3_str.fillna(0) + o3_tro.fillna(0))
    other = erf_total - (ghg.fillna(0) + total_o3.fillna(0) + aer.fillna(0) + sol.fillna(0) + volc.fillna(0))

    result = {
        "stratO3_ERF": o3_str.astype("float32"),
        "sol_ERF": sol.astype("float32"),
        "other_ERF": other.astype("float32"),
        "volc_ERF": volc.astype("float32"),
        "nat_ERF": (sol + volc).astype("float32"),
        "totalO3_ERF": total_o3.astype("float32"),
        "GHG_ERF": ghg.astype("float32"),
        "aer_ERF": aer.astype("float32"),
        "CO2_ERF": co2.astype("float32"),
        "tas_smoothed": tas_sm.astype("float32"),
        "rtmt_smoothed": rtmt_sm.astype("float32"),
    }
    if return_tas_offset:
        return result, tas_offset_used
    return result


# %% [markdown]
# ## ERA5 splice and export helpers
# The splice function mirrors the original `616_*` logic (shift/scale/guarded
# scale with start/end fade windows). Predictors are converted into emulator
# units (tas in degC/10) before splicing so the exports drop directly into the
# probabilistic runners.

# %%
def _splice_with_era5(
    base_series: pd.Series,
    era5_series: pd.Series,
    normalisation_start_years: int,
    normalisation_end_years: int,
    start_year_override: Optional[int] = None,
    end_year_override: Optional[int] = None,
    scenario_label: Optional[str] = None,
    mode: str = "shift",
) -> Optional[pd.Series]:
    """
    Combine a model series with ERA5 using crossfades at the start and end of
    the ERA5 window. Pre-ERA5: base aligned to ERA5 mean over initial window.
    Post-ERA5: base aligned to ERA5 mean over final window. Mid ERA5 window
    uses ERA5 values directly.
    """
    base_series = _align_to_month_start(base_series)
    era5_series = _align_to_month_start(era5_series)
    if base_series is None or era5_series is None or base_series.empty or era5_series.empty:
        return None

    era5_start = era5_series.index.min()
    era5_end = era5_series.index.max()
    if pd.isna(era5_start) or pd.isna(era5_end):
        return None

    start_years = max(int(normalisation_start_years or 0), 0)
    end_years = max(int(normalisation_end_years or 0), 0)

    idx_union = base_series.index.union(era5_series.index).sort_values()
    base_full = base_series.sort_index().reindex(idx_union).astype("float32")
    base_full = base_full.interpolate(limit_direction="both")
    era5_full = era5_series.sort_index().reindex(idx_union)
    era5_full = era5_full.interpolate(limit_direction="both", limit_area="inside")

    start_year = start_year_override if start_year_override is not None else int(era5_start.year)
    start_year = max(start_year, int(era5_start.year))
    start_window_start = pd.Timestamp(f"{start_year}-01-01")
    if start_years == 0:
        start_window_end = start_window_start - pd.DateOffset(months=1)
    else:
        start_window_end = start_window_start + pd.DateOffset(years=start_years) - pd.DateOffset(months=1)
    start_window_end = min(start_window_end, era5_end)

    end_year = _select_end_year_for_label(scenario_label or "", end_year_override, int(era5_end.year))
    end_year = max(end_year, start_window_start.year)
    end_window_end = pd.Timestamp(f"{end_year}-12-01")
    if end_window_end > era5_end:
        end_window_end = era5_end
    if end_years == 0:
        end_window_start = end_window_end + pd.DateOffset(months=1)
    else:
        end_window_start = end_window_end - pd.DateOffset(years=end_years) + pd.DateOffset(months=1)
    end_window_start = max(end_window_start, start_window_start)

    m = (mode or "shift").strip().lower()
    if m == "scale":
        start_factor = _mean_scale(base_full, era5_full, start_window_start, start_window_end)
        end_factor = _mean_scale(base_full, era5_full, end_window_start, end_window_end)
        base_start = (base_full * start_factor).astype("float32")
        base_end = (base_full * end_factor).astype("float32")
    elif m == "guarded_scale":
        raw_start_factor = _mean_scale(base_full, era5_full, start_window_start, start_window_end)
        raw_end_factor = _mean_scale(base_full, era5_full, end_window_start, end_window_end)
        start_factor = _clamp(raw_start_factor, SPLICE_GUARDED_SCALE_MIN, SPLICE_GUARDED_SCALE_MAX)
        end_factor = _clamp(raw_end_factor, SPLICE_GUARDED_SCALE_MIN, SPLICE_GUARDED_SCALE_MAX)
        base_start_scaled = (base_full * start_factor).astype("float32")
        base_end_scaled = (base_full * end_factor).astype("float32")
        start_offset = _mean_offset(base_start_scaled, era5_full, start_window_start, start_window_end)
        end_offset = _mean_offset(base_end_scaled, era5_full, end_window_start, end_window_end)
        base_start = (base_start_scaled + start_offset).astype("float32")
        base_end = (base_end_scaled + end_offset).astype("float32")
    else:
        start_offset = _mean_offset(base_full, era5_full, start_window_start, start_window_end)
        end_offset = _mean_offset(base_full, era5_full, end_window_start, end_window_end)
        base_start = (base_full + start_offset).astype("float32")
        base_end = (base_full + end_offset).astype("float32")

    result = pd.Series(index=idx_union, dtype="float32")

    pre_mask = idx_union < start_window_start
    result.loc[pre_mask] = base_start.loc[pre_mask]

    if start_years > 0 and start_window_end >= start_window_start:
        fade_idx = idx_union[(idx_union >= start_window_start) & (idx_union <= start_window_end)]
        if len(fade_idx):
            w = _fade_weights(fade_idx)
            base_vals = base_start.loc[fade_idx].astype("float32").to_numpy()
            era_vals = era5_full.loc[fade_idx].astype("float32").to_numpy()
            blended = base_vals * (np.float32(1.0) - w) + era_vals * w
            result.loc[fade_idx] = blended.astype("float32")

    mid_mask = (idx_union > start_window_end) & (idx_union < end_window_start)
    result.loc[mid_mask] = era5_full.loc[mid_mask].astype("float32")

    if end_years > 0 and end_window_start <= end_window_end:
        fade_idx = idx_union[(idx_union >= end_window_start) & (idx_union <= end_window_end)]
        if len(fade_idx):
            w = _fade_weights(fade_idx)
            era_vals = era5_full.loc[fade_idx].astype("float32").to_numpy()
            base_vals = base_end.loc[fade_idx].astype("float32").to_numpy()
            blended = era_vals * (np.float32(1.0) - w) + base_vals * w
            result.loc[fade_idx] = blended.astype("float32")

    post_mask = idx_union > end_window_end
    result.loc[post_mask] = base_end.loc[post_mask]

    result = result.fillna(era5_full).fillna(base_end)
    return result.astype("float32")


def _write_spliced_predictors(
    label: str,
    scenario_label: str,
    workflow: str,
    runmodus: str,
    predictors: Dict[str, pd.Series],
    output_dir: Path,
    model_index: Optional[int] = None,
    formats: Sequence[str] = ("h5", "csv"),
) -> None:
    """
    Persist one MAGICC-based predictor set in a minimal CSV/HDF5 layout.
    Layout matches the `PREDICTOR_COLUMNS_ORDER` used by GCMagicc meta files.
    
    Directory structure: {output_dir}/{scenario}/
    (output_dir already points at .../n_X/ARY/runmode_Z/predictors)
    """
    if not predictors:
        return

    aligned_preds: Dict[str, pd.Series] = {}
    for key, series in predictors.items():
        aligned = _align_to_month_start(series)
        if aligned is None or aligned.empty:
            continue
        aligned_preds[key] = aligned
    if not aligned_preds:
        return

    ref = aligned_preds.get("tas_smoothed", next(iter(aligned_preds.values())))
    ref = _coerce_datetime_index(ref).sort_index()
    idx = ref.index
    years = idx.year.astype("int32")
    months = idx.month.astype("int32")
    month_f32 = months.astype("float32")
    sin_time = np.sin((month_f32 - 1.0) / 12.0 * 2 * np.pi).astype("float32")
    cos_time = np.cos((month_f32 - 1.0) / 12.0 * 2 * np.pi).astype("float32")
    model_idx_value = float(model_index if model_index is not None else PREDICTOR_MODEL_INDEX_DEFAULT)
    model_idx_arr = np.full(len(idx), model_idx_value, dtype="float32")

    def _series_for(name: str) -> pd.Series:
        base = aligned_preds.get(name)
        if base is None:
            return pd.Series(np.zeros(len(idx), dtype="float32"), index=idx)
        return _coerce_datetime_index(base).reindex(idx).astype("float32")

    ordered_cols: Dict[str, pd.Series] = {}
    for col in PREDICTOR_COLUMNS_ORDER:
        if col == "model_index":
            ordered_cols[col] = pd.Series(model_idx_arr, index=idx)
        elif col == "month":
            ordered_cols[col] = pd.Series(month_f32, index=idx)
        elif col == "sin_time":
            ordered_cols[col] = pd.Series(sin_time, index=idx)
        elif col == "cos_time":
            ordered_cols[col] = pd.Series(cos_time, index=idx)
        else:
            ordered_cols[col] = _series_for(col)

    df = pd.DataFrame(ordered_cols, index=idx)

    safe_scen = scenario_safe_name(scenario_label or "unknown_scenario")
    safe_label = scenario_safe_name(label)
    
    # Directory structure: {output_dir}/{scenario}/
    scen_dir = output_dir / safe_scen
    scen_dir.mkdir(parents=True, exist_ok=True)
    
    # source_name for metadata and filenames (backward compatibility)
    source_name = f"{workflow}_{runmodus}"

    formats_set = {fmt.lower() for fmt in formats}

    if "h5" in formats_set or "hdf5" in formats_set:
        try:
            import h5py  # type: ignore
        except Exception as exc:
            print(f"[io] Skipping HDF5 write for {label!r} (h5py not available: {exc})")
        else:
            h5_path = scen_dir / f"predictors_{source_name}_{safe_label}.h5"
            with h5py.File(h5_path, "w") as h5:
                h5.create_dataset("year", data=years)
                for col in df.columns:
                    h5.create_dataset(col, data=df[col].to_numpy(dtype="float32"))
                meta_grp = h5.create_group("meta")
                meta_grp.attrs["label"] = label
                meta_grp.attrs["scenario_label"] = scenario_label
                meta_grp.attrs["source_name"] = source_name
                meta_grp.attrs["created_by"] = "616_create_MAGICCbased_predictors.py"
                meta_grp.attrs["era5_normalisation_start_period"] = int(ERA5_NORMALISATION_START_PERIOD)
                meta_grp.attrs["era5_normalisation_end_period"] = int(ERA5_NORMALISATION_END_PERIOD)
                if ERA5_NORMALISATION_START_YR is not None:
                    meta_grp.attrs["era5_normalisation_start_year"] = int(ERA5_NORMALISATION_START_YR)
                if ERA5_NORMALISATION_END_YR is not None:
                    meta_grp.attrs["era5_normalisation_end_year"] = int(ERA5_NORMALISATION_END_YR)
            print(f"[io] Wrote MAGICC-based predictors (HDF5): {h5_path}")

    if "csv" in formats_set:
        csv_path = scen_dir / f"predictors_{source_name}_{safe_label}.csv"
        df_csv = df.copy()
        df_csv.insert(0, "year", years)
        df_csv.to_csv(csv_path, index=False)
        print(f"[io] Wrote MAGICC-based predictors (CSV): {csv_path}")


# %% [markdown]
# ## Plotting
# Each predictor gets its own plot comparing ERA5, raw MAGICC, ERA5-spliced
# MAGICC, and a small overlay of CMIP6 samples. Units:
# - tas_smoothed plotted in degC (converted from degC/10 emulator scale for visualization)
# - others in native units (W/m²).

# %%
def _plot_predictor(
    pred_key: str,
    era5_series: Optional[pd.Series],
    magicc_raw: Optional[pd.Series],
    magicc_spliced: Optional[pd.Series],
    magicc_raw_ensemble: Optional[Dict[str, pd.Series]] = None,
    magicc_spliced_ensemble: Optional[Dict[str, pd.Series]] = None,
    cmip6_series: Dict[str, pd.Series] = None,
    scenario_label: str = "",
    workflow: str = "",
    runmodus: str = "",
    figs_dir: Path = None,
    splice_point: Optional[pd.Timestamp] = None,
) -> None:
    """
    Plot predictor comparison including individual ensemble members and their mean.
    
    Parameters:
    -----------
    magicc_raw_ensemble : Dict[str, pd.Series]
        Dictionary mapping run_id to raw MAGICC series for individual ensemble members
    magicc_spliced_ensemble : Dict[str, pd.Series]
        Dictionary mapping run_id to spliced MAGICC series for individual ensemble members
    """
    if era5_series is None and magicc_raw is None:
        return
    if cmip6_series is None:
        cmip6_series = {}

    figs_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    def _plot_series(series: Optional[pd.Series], style: dict, label: str):
        if series is None or series.empty:
            return
        s = _align_to_month_start(series)
        # Convert tas_smoothed from degC/10 to degC for plotting
        if pred_key == "tas_smoothed":
            plot_values = (s.values * 10.0).astype("float32")
        else:
            plot_values = s.values
        style_no_label = {k: v for k, v in style.items() if k != "label"}
        ax.plot(_index_to_fractional_year(s.index), plot_values, label=label, **style_no_label)

    # Collect all series for tas_smoothed period statistics
    all_series_for_stats: Dict[str, pd.Series] = {}
    
    # Plot individual ensemble members (raw) - lighter color, lower zorder
    if magicc_raw_ensemble:
        ensemble_style = dict(color=MAGICC_RAW_ENSEMBLE_COLOR, lw=0.8, alpha=0.5, zorder=1)
        for i, (run_id, ser) in enumerate(sorted(magicc_raw_ensemble.items())):
            if ser is None or ser.empty:
                continue
            s = _align_to_month_start(ser)
            if pred_key == "tas_smoothed":
                all_series_for_stats[f"raw_r{run_id}"] = s
            # Convert tas_smoothed from degC/10 to degC for plotting
            if pred_key == "tas_smoothed":
                plot_values = (s.values * 10.0).astype("float32")
            else:
                plot_values = s.values
            if i == 0:
                ax.plot(_index_to_fractional_year(s.index), plot_values, label="MAGICC raw (ensemble members)", **ensemble_style)
            else:
                ax.plot(_index_to_fractional_year(s.index), plot_values, **ensemble_style)
    
    # Plot mean raw (if provided and different from ensemble)
    if magicc_raw is not None:
        _plot_series(magicc_raw, MAGICC_STYLE, MAGICC_STYLE["label"])
        if pred_key == "tas_smoothed":
            all_series_for_stats["raw_mean"] = _align_to_month_start(magicc_raw)
    
    # Plot individual ensemble members (spliced) - lighter color, lower zorder
    if magicc_spliced_ensemble:
        ensemble_spliced_style = dict(color=SPLICED_ENSEMBLE_COLOR, lw=0.8, alpha=0.5, zorder=2)
        for i, (run_id, ser) in enumerate(sorted(magicc_spliced_ensemble.items())):
            if ser is None or ser.empty:
                continue
            s = _align_to_month_start(ser)
            if pred_key == "tas_smoothed":
                all_series_for_stats[f"spliced_r{run_id}"] = s
            # Convert tas_smoothed from degC/10 to degC for plotting
            if pred_key == "tas_smoothed":
                plot_values = (s.values * 10.0).astype("float32")
            else:
                plot_values = s.values
            if i == 0:
                ax.plot(_index_to_fractional_year(s.index), plot_values, label="MAGICC spliced (ensemble members)", **ensemble_spliced_style)
            else:
                ax.plot(_index_to_fractional_year(s.index), plot_values, **ensemble_spliced_style)
    
    # Plot mean spliced
    if magicc_spliced is not None:
        _plot_series(magicc_spliced, SPLICED_STYLE, SPLICED_STYLE["label"])
        if pred_key == "tas_smoothed":
            all_series_for_stats["spliced_mean"] = _align_to_month_start(magicc_spliced)

    if cmip6_series:
        for i, (name, ser) in enumerate(sorted(cmip6_series.items())):
            style = {k: v for k, v in CMIP6_STYLE.items() if k != "label"}
            if i == 0:
                lbl = CMIP6_STYLE["label"]
            else:
                lbl = "_nolegend_"
            s_aligned = _align_to_month_start(ser)
            # Convert tas_smoothed from degC/10 to degC for plotting
            if pred_key == "tas_smoothed":
                plot_values = (s_aligned.values * 10.0).astype("float32")
            else:
                plot_values = s_aligned.values
            ax.plot(_index_to_fractional_year(s_aligned.index), plot_values, label=lbl, **style)
            if pred_key == "tas_smoothed":
                all_series_for_stats[f"cmip6_{name}"] = s_aligned

    # Plot ERA5 on top (highest zorder)
    if era5_series is not None:
        _plot_series(era5_series, ERA5_STYLE, ERA5_STYLE["label"])
        if pred_key == "tas_smoothed":
            all_series_for_stats["ERA5"] = _align_to_month_start(era5_series)

    if splice_point is not None:
        spx = float(splice_point.year) + (float(splice_point.month) - 1.0) / 12.0
        ax.axvline(spx, color="#888888", lw=1.2, ls="--", alpha=0.7, label="splice point", zorder=5)

    # Special handling for tas_smoothed: plot period difference statistics
    # Only use ERA5-spliced MAGICC series (the ones written to predictor files)
    if pred_key == "tas_smoothed" and all_series_for_stats:
        # Filter to only include spliced ensemble members (keys starting with "spliced_r")
        spliced_only = {k: s for k, s in all_series_for_stats.items() if k.startswith("spliced_r")}
        
        if not spliced_only:
            # Fallback: if no individual members, try spliced_mean
            if "spliced_mean" in all_series_for_stats:
                spliced_only = {"spliced_mean": all_series_for_stats["spliced_mean"]}
        
        if spliced_only:
            # Convert from degC/10 to degC for period statistics
            series_for_stats = {k: (s * 10.0).astype("float32") for k, s in spliced_only.items()}
            
            period_stats, diff_stats = _period_and_delta_stats(
                series_for_stats, TAS_PERIOD_WINDOWS, "1995-2014", "2081-2100"
            )
        else:
            diff_stats = None
        
        if diff_stats:
            baseperiod_rel_preind = 0.85
            # Display in text box below legend in top left corner
            diff_text = (
                f"Delta 2081-2100 vs 1995-2014:\n"
                f"  median = {(diff_stats['median']+baseperiod_rel_preind):.2f}°C (incl. +0.85)\n"
                f"  5-95% range: {(diff_stats['p05']+baseperiod_rel_preind):.2f} - {(diff_stats['p95']+baseperiod_rel_preind):.2f}°C\n"
                f"  n = {diff_stats['count']}"
            )
            
            ax.text(
                0.02, 0.70,
                diff_text,
                transform=ax.transAxes,
                color=TAS_DIFF_COLOR,
                fontsize=TAS_PERIOD_TEXT_FONTSIZE,
                va="top",
                ha="left",
                bbox={"facecolor": "white", "edgecolor": TAS_DIFF_COLOR, "alpha": 0.85, "boxstyle": "round,pad=0.5"},
                zorder=15,
            )

    ax.set_title(f"{pred_key} | {scenario_label} | {workflow} / {runmodus}")
    ax.set_xlabel("Year")
    ax.set_ylabel("degC" if pred_key == "tas_smoothed" else "W m$^{-2}$")
    ax.grid(True, ls=":", alpha=0.3)
    # Position legend in upper left to make room for tas_smoothed statistics box below it
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()

    safe_pred = scenario_safe_name(pred_key)
    safe_scen = scenario_safe_name(scenario_label)
    fname_base = figs_dir / f"{workflow.lower()}_{runmodus}_{safe_scen}_{safe_pred}"
    fig.savefig(fname_base.with_suffix(".png"), dpi=150)
    fig.savefig(fname_base.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[plot] Saved {pred_key} -> {fname_base.with_suffix('.pdf')}")


# %% [markdown]
# ## Main driver
# 1. Load ERA5 once (and compute baseline temp for tas).
# 2. Sample CMIP6 files once for consistent overlays across scenarios.
# 3. Loop over resampled MAGICC packages, workflows/runmodes, splice, plot, and export predictors.

# %%
def _process_single_scenario(
    pq_path: Path,
    workflow: str,
    runmodus: str,
    output_dir: Path,
    figs_dir: Path,
    do_splice: bool,
    era5_tas_baseline: float,
    era5_for_splice: Dict[str, pd.Series],
    cmip6_overlay: Dict[str, Dict[str, pd.Series]],
    splice_point: Optional[pd.Timestamp],
    tas_anchor_cache: Optional[Dict[Tuple[str, str], float]] = None,
    runid_label_map: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[Tuple[str, str], float], Set[str]]:
    """
    Process a single MAGICC scenario parquet file.
    This function is designed to be called in parallel and writes to the
    provided output/plot directories.

    Returns
    -------
    (offsets, run_ids_used)
        offsets: mapping of (scenario_label_clean, run_id_str) -> tas_offset (only for runmode_all)
        run_ids_used: set of run_id_str processed for this scenario
    """
    scenario_name = pq_path.stem.split("_", 1)[0]
    print(f"\n[scenario] {scenario_name} ({pq_path.name})")
    
    scenario_df = read_magicc_scenario(pq_path)
    if scenario_df.empty:
        print(f"[skip] No usable rows in {pq_path.name}")
        return {}, set()

    # group by experiment (model+scenario) though files already per-scenario
    if "model" in scenario_df.columns and "scenario" in scenario_df.columns:
        scenario_df["experiment_id"] = scenario_df["model"].astype(str) + "_" + scenario_df["scenario"].astype(str)
    else:
        scenario_df["experiment_id"] = scenario_name

    exp_ids = sorted(scenario_df["experiment_id"].astype(str).unique())
    tas_offsets_local: Dict[Tuple[str, str], float] = {}
    run_ids_used: Set[str] = set()

    for exp_id in exp_ids:
        exp_df_all = scenario_df[scenario_df["experiment_id"] == exp_id]
        model_name, scen_label = (exp_id.split("_", 1) + ["unknown"])[:2]
        scen_label_clean = _clean_scenario_label(model_name, scen_label, scenario_name)

        # Get unique run_ids (ensemble members)
        if "run_id" in exp_df_all.columns:
            run_ids = sorted(exp_df_all["run_id"].dropna().unique())
            if len(run_ids) == 0:
                run_ids = [None]
        else:
            run_ids = [None]

        if runid_label_map is not None:
            filtered = []
            for r in run_ids:
                rid_str = str(int(r)) if r is not None else "single"
                if rid_str in runid_label_map:
                    filtered.append(r)
            run_ids = filtered
            if not run_ids:
                # Positional fallback: map available members in order
                mapped_labels = list(runid_label_map.values())
                run_ids = [None] * min(len(mapped_labels), len(runid_label_map))
                print(
                    f"[warn] {scenario_name} / {runmodus}: run_ids differ; falling back to positional mapping "
                    f"available -> {mapped_labels}"
                )

        # Process each ensemble member separately
        all_raw_predictors: Dict[str, Dict[str, pd.Series]] = {}  # run_id -> pred_key -> series
        all_spliced_predictors: Dict[str, Dict[str, pd.Series]] = {}  # run_id -> pred_key -> series

        for run_id in run_ids:
            # Filter to this ensemble member
            if run_id is not None:
                exp_df = exp_df_all[exp_df_all["run_id"] == run_id].copy()
                run_id_str = str(int(run_id)) if pd.notna(run_id) else "unknown"
            else:
                exp_df = exp_df_all.copy()
                run_id_str = "single"

            if exp_df.empty:
                continue

            run_id_label = runid_label_map.get(run_id_str, run_id_str) if runid_label_map else run_id_str
            run_ids_used.add(run_id_label)

            offset_key = (scen_label_clean, run_id_label)

            if runmodus == "all":
                scenario_pred_raw, tas_offset_used = scenario_predictors_from_long(
                    exp_df,
                    START_YEAR,
                    END_YEAR,
                    era5_baseline_temp=era5_tas_baseline,
                    return_tas_offset=True,
                )
                tas_offsets_local[offset_key] = tas_offset_used
                # Also store under scenario_name to allow cross-runmode matching when labels differ
                tas_offsets_local[(scenario_name, run_id_label)] = tas_offset_used
            else:
                cached_offset = None
                if tas_anchor_cache is not None:
                    for key in (
                        offset_key,
                        (scenario_name, run_id_label),
                        (scen_label_clean, "single"),
                        (scenario_name, "single"),
                    ):
                        if key in tas_anchor_cache:
                            cached_offset = tas_anchor_cache[key]
                            break
                if cached_offset is None:
                    raise RuntimeError(
                        f"TAS anchor offset missing for scenario={scenario_name}, run_id={run_id_str}, "
                        f"runmode={runmodus}. Ensure runmode 'all' was processed first for the same member."
                    )
                scenario_pred_raw = scenario_predictors_from_long(
                    exp_df,
                    START_YEAR,
                    END_YEAR,
                    era5_baseline_temp=era5_tas_baseline,
                    tas_offset=cached_offset,
                    return_tas_offset=False,
                )
            scenario_pred = {k: _pred_units_adjusted_series(k, v) for k, v in scenario_pred_raw.items()}

            # Fill nat_ERF if missing
            if "nat_ERF" not in scenario_pred and "sol_ERF" in scenario_pred and "volc_ERF" in scenario_pred:
                scenario_pred["nat_ERF"] = (scenario_pred["sol_ERF"] + scenario_pred["volc_ERF"]).astype("float32")

            # ERA5 splice only for selected runmodes; others keep raw predictors
            if do_splice:
                spliced: Dict[str, Optional[pd.Series]] = {}
                for key in VISUAL_TARGETS:
                    base_series = scenario_pred.get(key)
                    era_series = era5_for_splice.get(key)
                    if base_series is None or era_series is None:
                        continue
                    mode = SPLICE_METHOD_BY_PRED.get(key, "shift")
                    spliced_series = _splice_with_era5(
                        base_series,
                        era_series,
                        ERA5_NORMALISATION_START_PERIOD,
                        ERA5_NORMALISATION_END_PERIOD,
                        start_year_override=ERA5_NORMALISATION_START_YR,
                        end_year_override=ERA5_NORMALISATION_END_YR,
                        scenario_label=scen_label_clean,
                        mode=mode,
                    )
                    if spliced_series is not None:
                        spliced[key] = spliced_series.astype("float32")
                combined_for_export = spliced
            else:
                combined_for_export = scenario_pred
                spliced = {}

            # Store this ensemble member's predictors
            all_raw_predictors[run_id_label] = scenario_pred
            all_spliced_predictors[run_id_label] = spliced

            # Export each ensemble member separately
            # member_label = f"{exp_id}_r{run_id_str}" if run_id is not None else exp_id
            member_label = f"{scen_label_clean}_r{run_id_label}" if run_id is not None else scen_label_clean
            _write_spliced_predictors(
                label=member_label,
                scenario_label=scen_label_clean,
                workflow=workflow,
                runmodus=runmodus,
                predictors=combined_for_export,
                output_dir=output_dir,
                model_index=PREDICTOR_MODEL_INDEX_DEFAULT,
                formats=("h5", "csv"),
            )

        # Calculate mean across ensemble members for plotting
        mean_raw_predictors: Dict[str, pd.Series] = {}
        mean_spliced_predictors: Dict[str, pd.Series] = {}

    for pred_key in VISUAL_TARGETS:
        # Mean of raw predictors
        raw_series_list = []
        for run_id_str, pred_dict in all_raw_predictors.items():
            if pred_key in pred_dict and pred_dict[pred_key] is not None:
                    raw_series_list.append(_align_to_month_start(pred_dict[pred_key]))
            if raw_series_list:
                # Align all series to union of all indices
                all_indices = raw_series_list[0].index
                for s in raw_series_list[1:]:
                    all_indices = all_indices.union(s.index)
                common_idx = all_indices.sort_values()
                aligned_raw = [s.reindex(common_idx) for s in raw_series_list]
                mean_raw = pd.concat(aligned_raw, axis=1).mean(axis=1)
                mean_raw_predictors[pred_key] = mean_raw.astype("float32")

            if do_splice:
                # Mean of spliced predictors
                spliced_series_list = []
                for run_id_str, pred_dict in all_spliced_predictors.items():
                    if pred_key in pred_dict and pred_dict[pred_key] is not None:
                        spliced_series_list.append(_align_to_month_start(pred_dict[pred_key]))
                if spliced_series_list:
                    # Align all series to union of all indices
                    all_indices = spliced_series_list[0].index
                    for s in spliced_series_list[1:]:
                        all_indices = all_indices.union(s.index)
                    common_idx = all_indices.sort_values()
                    aligned_spliced = [s.reindex(common_idx) for s in spliced_series_list]
                    mean_spliced = pd.concat(aligned_spliced, axis=1).mean(axis=1)
                    mean_spliced_predictors[pred_key] = mean_spliced.astype("float32")

        # Prepare ensemble dictionaries for plotting (individual members)
        raw_ensemble_dict: Dict[str, Dict[str, pd.Series]] = {}
        spliced_ensemble_dict: Dict[str, Dict[str, pd.Series]] = {}
        for run_id_str in all_raw_predictors.keys():
            raw_ensemble_dict[run_id_str] = {k: v for k, v in all_raw_predictors[run_id_str].items() if k in VISUAL_TARGETS}
        if do_splice:
            for run_id_str in all_spliced_predictors.keys():
                spliced_ensemble_dict[run_id_str] = {
                    k: v for k, v in all_spliced_predictors[run_id_str].items() if k in VISUAL_TARGETS
                }

        # Plot comparisons per predictor (with individual members and mean)
        for pred_key in VISUAL_TARGETS:
            era5_series = era5_for_splice.get(pred_key)
            cmip6_overlay_for_key = {name: series_map[pred_key] for name, series_map in cmip6_overlay.items() if pred_key in series_map}
            
            # Extract individual ensemble members for this predictor
            raw_ensemble_for_key = {run_id: pred_dict[pred_key] for run_id, pred_dict in raw_ensemble_dict.items() if pred_key in pred_dict}
            spliced_ensemble_for_key = {run_id: pred_dict[pred_key] for run_id, pred_dict in spliced_ensemble_dict.items() if pred_key in pred_dict}
            
            _plot_predictor(
                pred_key,
                era5_series=era5_series,
                magicc_raw=mean_raw_predictors.get(pred_key),
                magicc_spliced=mean_spliced_predictors.get(pred_key) if do_splice else None,
                magicc_raw_ensemble=raw_ensemble_for_key if len(raw_ensemble_for_key) > 1 else None,
                magicc_spliced_ensemble=(
                    spliced_ensemble_for_key if do_splice and len(spliced_ensemble_for_key) > 1 else None
                ),
                cmip6_series=cmip6_overlay_for_key,
                scenario_label=scen_label_clean,
                workflow=workflow,
                runmodus=runmodus,
                figs_dir=figs_dir,
                splice_point=splice_point if do_splice else None,
            )

    return tas_offsets_local, run_ids_used

def build_and_plot() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTOR_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[config] MAGICC resampled root : {MAGICC_RESAMPLED_ROOT}")
    print(f"[config] MAGICC resampled prefix: {MAGICC_RESAMPLED_PREFIX}")
    print(f"[config] Resample sizes        : {RESAMPLE_SIZES}")
    print(f"[config] ERA5 file : {ERA5_FILE}")
    print(f"[config] CMIP6 dir : {CMIP6_DIR} (filters={CMIP6_FILENAME_FILTER}, n={N_CMIP6_RANDOM})")
    print(f"[config] Workflows            : {WORKFLOWS}")
    print(f"[config] Runmodes             : {RUNMODES}")
    print(f"[config] Runmodes spliced     : {RUNMODES_SPLICED}")
    print(
        f"[config] Scenario whitelist  : "
        f"{SCENARIO_WHITELIST if SCENARIO_WHITELIST else 'ALL (no filter)'}"
    )
    print(f"[config] Output root          : {PREDICTOR_OUTPUT_ROOT}")

    # 1) ERA5
    era5_raw = load_era5_predictors(ERA5_FILE, START_YEAR, END_YEAR)
    if "tas_smoothed" not in era5_raw or era5_raw["tas_smoothed"].empty:
        raise RuntimeError("ERA5 tas_smoothed missing; cannot proceed with splicing.")
    era5_tas_baseline = (
        era5_raw["tas_smoothed"]
        .loc[(era5_raw["tas_smoothed"].index.year >= BASELINE_PERIOD[0]) & (era5_raw["tas_smoothed"].index.year <= BASELINE_PERIOD[1])]
        .mean()
    )
    era5_for_splice = {k: _pred_units_adjusted_series(k, v) for k, v in era5_raw.items()}
    splice_point = _coerce_datetime_index(era5_raw["tas_smoothed"]).index.max()

    # 2) CMIP6 sample overlays
    cmip6_files = pick_cmip6_samples(CMIP6_DIR, CMIP6_FILENAME_FILTER, N_CMIP6_RANDOM, CMIP6_RANDOM_SEED)
    cmip6_overlay: Dict[str, Dict[str, pd.Series]] = {}
    for path in cmip6_files:
        cmip6_raw = load_cmip6_predictors(path, START_YEAR, END_YEAR)
        cmip6_overlay[path.stem] = {k: _pred_units_adjusted_series(k, v) for k, v in cmip6_raw.items()}
    print(f"[load] CMIP6 overlay files: {len(cmip6_files)}")

    # 3) MAGICC scenarios across resampled packages
    runid_label_cache: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for resample_size in RESAMPLE_SIZES:
        resample_tag = f"n_{resample_size}"
        resample_root = MAGICC_RESAMPLED_ROOT / f"{MAGICC_RESAMPLED_PREFIX}{resample_size}"
        if not resample_root.exists():
            print(f"[warn] Missing resampled root: {resample_root}")
            continue

        for workflow in WORKFLOWS:
            # Always process 'all' first so its offsets are available to other runmodes
            runmodes_order = sorted(RUNMODES, key=lambda x: (x != "all", x))

            # Pre-scan run_ids per scenario across runmodes to choose aligned set
            all_runmode_dirs = {
                rm: resample_root / workflow / f"runmode_{rm}"
                for rm in runmodes_order
                if (resample_root / workflow / f"runmode_{rm}").exists()
            }
            scenario_runids: Dict[str, Dict[str, Set[str]]] = {}
            # use runmode_all as baseline of scenarios to consider
            base_dir = all_runmode_dirs.get("all")
            if base_dir is None:
                print(f"[warn] Missing runmode_all for {workflow} {resample_tag}; skipping workflow.")
                continue
            for pq_path in sorted(base_dir.glob("*.parquet")):
                scen_name = pq_path.stem.split("_", 1)[0]
                if not scenario_allowed(scen_name, pq_path.stem):
                    continue
                scenario_runids.setdefault(scen_name, {})
                scenario_runids[scen_name]["all"] = _run_ids_from_parquet(pq_path)
            # collect run_ids for other runmodes where files exist
            for rm, rm_dir in all_runmode_dirs.items():
                if rm == "all":
                    continue
                for pq_path in rm_dir.glob("*.parquet"):
                    scen_name = pq_path.stem.split("_", 1)[0]
                    if not scenario_allowed(scen_name, pq_path.stem):
                        continue
                    if scen_name not in scenario_runids:
                        # skip scenarios absent in all
                        continue
                    scenario_runids[scen_name][rm] = _run_ids_from_parquet(pq_path)

            # derive aligned run_ids per scenario: positional alignment anchored to runmode_all
            for scen_name, by_rm in scenario_runids.items():
                all_ids_sorted = sorted(by_rm.get("all", {"single"}))
                other_lists = {rm: sorted(ids) for rm, ids in by_rm.items() if rm != "all"}
                min_len = min([len(all_ids_sorted)] + [len(lst) for lst in other_lists.values()]) if other_lists else len(all_ids_sorted)
                if min_len == 0:
                    continue
                aligned_labels = all_ids_sorted[:min_len]
                # runmode_all mapping (identity for aligned slice)
                runid_label_cache[(resample_tag, workflow, "all", scen_name)] = {
                    src_id: aligned_labels[i] for i, src_id in enumerate(all_ids_sorted[:min_len])
                }
                for rm, ids_sorted in other_lists.items():
                    map_len = min(min_len, len(ids_sorted))
                    mapping = {ids_sorted[i]: aligned_labels[i] for i in range(map_len)}
                    runid_label_cache[(resample_tag, workflow, rm, scen_name)] = mapping
                    if map_len < min_len:
                        print(
                            f"[warn] {workflow} {resample_tag} {scen_name} runmode={rm} has only {map_len} members; "
                            f"aligned to first {map_len} of runmode_all."
                        )
            for runmodus in runmodes_order:
                run_dir = resample_root / workflow / f"runmode_{runmodus}"
                if not run_dir.exists():
                    print(f"[warn] Missing run directory: {run_dir}")
                    continue

                parquet_files = sorted(run_dir.glob("*.parquet"))
                if SCENARIO_WHITELIST:
                    parquet_files = [
                        pq_path
                        for pq_path in parquet_files
                        if scenario_allowed(pq_path.stem.split("_", 1)[0], pq_path.stem)
                    ]
                if not parquet_files:
                    if SCENARIO_WHITELIST:
                        print(f"[warn] No parquet files in {run_dir} after scenario whitelist filtering")
                    else:
                        print(f"[warn] No parquet files in {run_dir}")
                    continue

                do_splice = runmodus in RUNMODES_SPLICED
                output_runmode_dir = PREDICTOR_OUTPUT_ROOT / resample_tag / workflow / f"runmode_{runmodus}"
                output_runmode_dir.mkdir(parents=True, exist_ok=True)
                figs_dir = output_runmode_dir / COMPARISON_PLOTS_SUBDIR
                figs_dir.mkdir(parents=True, exist_ok=True)
                predictors_dir = output_runmode_dir / "predictors"
                predictors_dir.mkdir(parents=True, exist_ok=True)

                print(
                    f"\n[workflow] n={resample_size} {workflow} / {runmodus} -> {len(parquet_files)} files "
                    f"(splice={do_splice})"
                )
                
                # Process scenarios in parallel using ProcessPoolExecutor
                # Each scenario (parquet file) is processed independently, making this
                # an ideal candidate for parallelization. The shared data (ERA5, CMIP6)
                # is passed to each worker, which will pickle/unpickle it.
                # Set N_WORKERS=1 or N_WORKERS=0 to disable parallelization.
                if N_WORKERS > 1 and len(parquet_files) > 1:
                    print(f"[parallel] Processing {len(parquet_files)} scenarios with {N_WORKERS} workers")
                    cache_snapshot = dict(TAS_ANCHOR_OFFSETS)
                    task_args = []
                    for pq_path in parquet_files:
                        scen_name = pq_path.stem.split("_", 1)[0]
                        runid_map = runid_label_cache.get((resample_tag, workflow, runmodus, scen_name))
                        task_args.append(
                            (
                                pq_path,
                                workflow,
                                runmodus,
                                predictors_dir,
                                figs_dir,
                                do_splice,
                                era5_tas_baseline,
                                era5_for_splice,
                                cmip6_overlay,
                                splice_point,
                                cache_snapshot,
                                runid_map,
                            )
                        )

                    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                        futures = {
                            executor.submit(
                                _process_single_scenario,
                                pq,
                                wf,
                                rm,
                                outdir,
                                figdir,
                                splice_flag,
                                tas_base,
                                era5_fs,
                                cmip6_ovl,
                                sp_point,
                                cache_snap,
                                runid_map,
                            ): pq
                            for (
                                pq,
                                wf,
                                rm,
                                outdir,
                                figdir,
                                splice_flag,
                                tas_base,
                                era5_fs,
                                cmip6_ovl,
                                sp_point,
                                cache_snap,
                                runid_map,
                            ) in task_args
                        }
                        for future in as_completed(futures):
                            pq_path = futures[future]
                            try:
                                offsets, run_ids_used = future.result()
                                if offsets:
                                    TAS_ANCHOR_OFFSETS.update(offsets)
                                if runmodus == "all":
                                    scen_name = pq_path.stem.split("_", 1)[0]
                                    runid_label_cache[(resample_tag, workflow, runmodus, scen_name)] = {
                                        rid: rid for rid in run_ids_used
                                    }
                            except Exception as exc:
                                print(f"[error] Scenario {pq_path.name} failed: {exc}")
                else:
                    # Sequential processing (for debugging or single file)
                    for pq_path in parquet_files:
                        cache_snapshot = dict(TAS_ANCHOR_OFFSETS)
                        scen_name = pq_path.stem.split("_", 1)[0]
                        runid_map = runid_label_cache.get((resample_tag, workflow, runmodus, scen_name))
                        offsets, run_ids_used = _process_single_scenario(
                            pq_path,
                            workflow=workflow,
                            runmodus=runmodus,
                            output_dir=predictors_dir,
                            figs_dir=figs_dir,
                            do_splice=do_splice,
                            era5_tas_baseline=era5_tas_baseline,
                            era5_for_splice=era5_for_splice,
                            cmip6_overlay=cmip6_overlay,
                            splice_point=splice_point,
                            tas_anchor_cache=cache_snapshot,
                            runid_label_map=runid_map,
                        )
                        if offsets:
                            TAS_ANCHOR_OFFSETS.update(offsets)
                        if runmodus == "all":
                            runid_label_cache[(resample_tag, workflow, runmodus, scen_name)] = {
                                rid: rid for rid in run_ids_used
                            }


if __name__ == "__main__":
    build_and_plot()
