#!/usr/bin/env python3
"""
758_GapFiller_SPEI (rev3 - plot from precomputed SPEIx segments)

This script plots SPEI{scale} diagnostics for a target AR6 region (default IRN),
using the precomputed SPEI segment stores written by:

  754_add_SPEI_to_ensemble_outputs.py

Inputs (defaults as requested)
------------------------------
- SCENARIO1 GCMagicc ensembles root:
  data/site_eth/GCMAGICCoutput/ERA5splicedS3/v101/ssp245/AR6/all/n_100/original/run_<latest_verified>

- SCENARIO2 GCMagicc ensembles root:
  data/site_eth/GCMAGICCoutput/ERA5splicedS3/v101/ssp245/AR6/nat/n_100/original/run_<latest_verified>

- ERA5 file:
  data/site_eth/out_ERA5_4July2025_1degree_vetted/DAT_ERA5_historical-ERA5_r1i1p1f1_clt-day-hurs-huss-month-pr-psl-rlut-rsds-rsdt-rsnt-rtmt-sfcWind-tas-tasmax-tasmin-ts-year.nc

The script expects that 754 has created segment stores at:
  <ROOT>/data_derivatives/SPEIx/segments.zarr

Plot layout
-----------
1) Map triptych for 2020–2024:
   - ERA5 SPEI{scale} spatial mean (region grid)
   - SCENARIO1 ensemble member with highest mean SPEI{scale} over 2020–2024
   - SCENARIO1 ensemble member with lowest mean SPEI{scale} over 2020–2024

2) Time-series rows (default 1975–2024):
   - ERA5
   - SCENARIO1 ensembles
   - SCENARIO2 ensembles

3) Histogram matrix (each row same height as above rows):
  - Row A: Month SPEI{scale} histograms (current 2021–2025 and future 2041–2060, normalized frequencies)
   - Row B: Return-frequency risk lines by month-year (1960–2100) for Scenario 1 and Scenario 2
   - Row C: Two summary tables (return periods and matched-return SPEI values)
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import string
import os
import re
import shutil
import sys
import gzip
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Iterator, Any, Set
import math

import numpy as np

try:
    import regionmask  # type: ignore
except Exception:
    regionmask = None

try:
    import xarray as xr
except Exception as exc:
    xr = None
    XR_IMPORT_ERROR = exc
else:
    XR_IMPORT_ERROR = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.dates as mdates
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MultipleLocator
except Exception as exc:
    plt = None
    GridSpec = None
    mdates = None
    LinearSegmentedColormap = None
    ListedColormap = None
    Line2D = None
    FuncFormatter = None
    MultipleLocator = None
    MPL_IMPORT_ERROR = exc
else:
    MPL_IMPORT_ERROR = None

try:
    import cartopy.crs as ccrs  # type: ignore
    import cartopy.feature as cfeature  # type: ignore
except Exception:
    ccrs = None
    cfeature = None

try:
    import pycountry  # type: ignore
except Exception:
    pycountry = None

# Resolve repository root in script/notebook mode so helper imports stay stable.
try:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:  # pragma: no cover - notebook mode
    _cwd = Path.cwd()
    _REPO_ROOT = _cwd.parent if _cwd.name == "notebooks" else _cwd
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_helper_path_utils_file = _REPO_ROOT / "scr" / "validation_helpers" / "helper_path_utils.py"
_helper_path_utils_spec = importlib.util.spec_from_file_location(
    "_gcmagicc_helper_path_utils",
    _helper_path_utils_file,
)
if _helper_path_utils_spec is None or _helper_path_utils_spec.loader is None:  # pragma: no cover
    raise ImportError(f"Failed to load helper_path_utils from {_helper_path_utils_file}")
_helper_path_utils = importlib.util.module_from_spec(_helper_path_utils_spec)
_helper_path_utils_spec.loader.exec_module(_helper_path_utils)
from scr.validation_helpers import helper_overlay_cache as _overlay_cache
try:
    from scr.validation_helpers.region_registry_815 import (
        build_region_mask as _build_canonical_region_mask,
    )
except Exception:
    _build_canonical_region_mask = None

get_cmip6_vetted_path = _helper_path_utils.get_cmip6_vetted_path
get_era5spliced_root = _helper_path_utils.get_era5spliced_root
build_era5spliced_dataset_path = _helper_path_utils.build_era5spliced_dataset_path
resolve_canonical_dataset_root = _helper_path_utils.resolve_canonical_dataset_root
resolve_derivatives_root = _helper_path_utils.resolve_derivatives_root
resolve_speix_root = _helper_path_utils.resolve_speix_root
get_era5_main_file = _helper_path_utils.get_era5_main_file
get_repo_path = _helper_path_utils.get_repo_path
get_version_default = _helper_path_utils.get_version_default
get_storage_access_default = _helper_path_utils.get_storage_access_default
normalize_storage_access = _helper_path_utils.normalize_storage_access
convert_local_path_to_s3_uri = _helper_path_utils.convert_local_path_to_s3_uri
DERIVATIVES_LAYOUT_CHOICES = _helper_path_utils.DERIVATIVES_LAYOUT_CHOICES
DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE = _helper_path_utils.DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE
DERIVATIVES_LAYOUT_INPLACE = _helper_path_utils.DERIVATIVES_LAYOUT_INPLACE
DEFAULT_DERIVATIVES_RUN_SUFFIX = _helper_path_utils.DEFAULT_DERIVATIVES_RUN_SUFFIX
STORAGE_ACCESS_CHOICES = _helper_path_utils.STORAGE_ACCESS_CHOICES
STORAGE_ACCESS_MOUNT = _helper_path_utils.STORAGE_ACCESS_MOUNT
STORAGE_ACCESS_S3_DIRECT = _helper_path_utils.STORAGE_ACCESS_S3_DIRECT

DEFAULT_DERIVATIVES_LAYOUT = (
    os.environ.get("GCMAGICC_DERIVATIVES_LAYOUT", DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE).strip().lower()
    or DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE
)
_DEFAULT_DERIVATIVES_RUN_SUFFIX = (
    os.environ.get("GCMAGICC_DERIVATIVES_RUN_SUFFIX", DEFAULT_DERIVATIVES_RUN_SUFFIX).strip()
    or DEFAULT_DERIVATIVES_RUN_SUFFIX
)
_ACTIVE_STORAGE_ACCESS = get_storage_access_default()

_CMIPCRUNCHER_ROOT = get_repo_path("cmipcruncher_firefly")
_DEFAULT_VERSION_TAG = get_version_default()
_ERA5SPLICED_ROOT = get_era5spliced_root()


def _prefer_existing_path(*paths: Path) -> Path:
    for candidate in paths:
        p = Path(candidate).expanduser().resolve(strict=False)
        if p.exists():
            return p
    return Path(paths[0]).expanduser().resolve(strict=False)


def _canonical_original_root(
    *,
    experiment_id: str,
    arx: str = "AR6",
    runmodus: str = "all",
    n_ensemble: str = "n_20",
) -> Path:
    return build_era5spliced_dataset_path(
        version=_DEFAULT_VERSION_TAG,
        experiment_id=experiment_id,
        arx=arx,
        runmodus=runmodus,
        n_ensemble=n_ensemble,
        kind="original",
        run_instance=None,
        root=_ERA5SPLICED_ROOT,
    )


def _resolve_latest_dataset_root(
    *,
    experiment_id: str,
    arx: str = "AR6",
    runmodus: str = "all",
    n_ensemble: str = "n_20",
    kind: str = "original",
) -> Path:
    return resolve_canonical_dataset_root(
        version=_DEFAULT_VERSION_TAG,
        experiment_id=experiment_id,
        arx=arx,
        runmodus=runmodus,
        n_ensemble=n_ensemble,
        kind=kind,
        root=_ERA5SPLICED_ROOT,
    )



# debiasloop_100NDClow_20260207-0548/debias for NDC-Trump-low/
# 
# -----------------------------------------------------------------------------
# Defaults (as requested)
# -----------------------------------------------------------------------------
DEFAULT_SSP245PLUSNAT_100_ROOT = _resolve_latest_dataset_root(
    experiment_id="ssp245",
    arx="AR6",
    runmodus="all",
    n_ensemble="n_100",
    kind="original",
)

# data/site_eth/GCMAGICCoutput/ERA5splicedS3/v101/ssp245/AR6/all/n_100/dataderivatives/run_<latest_verified>/data_derivatives/SPEIx/<tag>
PRESET_100SSP245PLUSNAT_20260204_ROOT = (
    _resolve_latest_dataset_root(
        experiment_id="ssp245",
        arx="AR6",
        runmodus="all",
        n_ensemble="n_100",
        kind="dataderivatives",
    )
)

DEFAULT_GCMAGICC_SCENARIO1_ROOT = Path(
    os.environ.get(
        "GCMAGICC_758_SCENARIO1_ROOT",
        str(
            _prefer_existing_path(
                DEFAULT_SSP245PLUSNAT_100_ROOT,
                _canonical_original_root(
                    experiment_id="ssp245",
                    arx="AR6",
                    n_ensemble="n_100",
                    runmodus="all",
                ),
                _canonical_original_root(
                    experiment_id="ssp245",
                    arx="AR6",
                    runmodus="all",
                    n_ensemble="n_20",
                ),
                _resolve_latest_dataset_root(
                    experiment_id="ssp245",
                    arx="AR6",
                    runmodus="all",
                    n_ensemble="n_20",
                    kind="original",
                ),
            )
        ),
    )
).expanduser().resolve(strict=False)
DEFAULT_GCMAGICC_ALL_ROOT = DEFAULT_GCMAGICC_SCENARIO1_ROOT  # legacy alias
DEFAULT_SCENARIO1_LABEL = "SSP2-4.5" # "SSP2-4.5" # "NDC-Trump-low" # "SSP2-4.5" # "SSP2-4.5"



# 100 NDCs.. debiasloop_100NDClow_20260207-0548
# 100 SSP245 + nat: debiasloop_100ssp245plusnat_20260204-0448
# 20 SSP245 + nat: debiasloop_20ssp245plusnat_20260124-0401


DEFAULT_GCMAGICC_SCENARIO2_ROOT = Path(
    os.environ.get(
        "GCMAGICC_758_SCENARIO2_ROOT",
        str(
            _prefer_existing_path(
                DEFAULT_SSP245PLUSNAT_100_ROOT,
                _canonical_original_root(
                    experiment_id="ssp245",
                    arx="AR6",
                    n_ensemble="n_100",
                    runmodus="nat",
                ),
                _canonical_original_root(
                    experiment_id="ssp245",
                    arx="AR6",
                    runmodus="nat",
                    n_ensemble="n_20",
                ),
                _resolve_latest_dataset_root(
                    experiment_id="ssp245",
                    arx="AR6",
                    runmodus="nat",
                    n_ensemble="n_20",
                    kind="original",
                ),
            )
        ),
    )
).expanduser().resolve(strict=False)

DEFAULT_GCMAGICC_NAT_ROOT = DEFAULT_GCMAGICC_SCENARIO2_ROOT  # legacy alias

DEFAULT_SCENARIO2_LABEL = "SSP2-4.5-nat" # "SSP2-4.5-nat" # "NDC-submitted-low" # "SSP2-4.5-nat" # "NDCs+Trump-low" # "SSP2-4.5-nat"
DEFAULT_SCENARIO2_SUFFIX = "-nat"# "-nat"  # "-nat" ""; used when explicit --scenario2-tag is not provided


DEFAULT_ERA5_FILE = Path(
    os.environ.get("GCMAGICC_ERA5_FILE", str(get_era5_main_file()))
).expanduser().resolve(strict=False)

DEFAULT_CMIP6_ROOT = Path(
    os.environ.get("GCMAGICC_CMIP6_ROOT", str(get_cmip6_vetted_path()))
).expanduser().resolve(strict=False)
DEFAULT_OVERLAY_SPEIX_TAG = (
    os.environ.get("GCMAGICC_OVERLAY_SPEIX_TAG", _overlay_cache.OVERLAY_CANONICAL_TAG).strip()
    or _overlay_cache.OVERLAY_CANONICAL_TAG
)
DEFAULT_ERA5_OVERLAY_ROOT = Path(
    os.environ.get(
        "GCMAGICC_ERA5_OVERLAY_ROOT",
        str(
            _overlay_cache.get_era5_overlay_speix_root(
                derivatives_layout=DEFAULT_DERIVATIVES_LAYOUT,
                derivatives_run_suffix=_DEFAULT_DERIVATIVES_RUN_SUFFIX,
            )
        ),
    )
).expanduser().resolve(strict=False)
DEFAULT_CMIP6_OVERLAY_ROOT = Path(
    os.environ.get(
        "GCMAGICC_CMIP6_OVERLAY_ROOT",
        str(
            _overlay_cache.get_cmip6_overlay_speix_root(
                derivatives_layout=DEFAULT_DERIVATIVES_LAYOUT,
                derivatives_run_suffix=_DEFAULT_DERIVATIVES_RUN_SUFFIX,
            )
        ),
    )
).expanduser().resolve(strict=False)
DEFAULT_CMIP6_HISTORICAL_SCENARIO = "historical"
DEFAULT_CMIP6_HISTNAT_SCENARIO = "hist-nat"
DEFAULT_CMIP6_SSP245_SCENARIO = "ssp245"
DEFAULT_INCLUDE_CMIP6 = True
DEFAULT_SHOW_CMIP6_HISTNAT = True
PLOT_ERA5DROUGHT_KEUNEETAL = True

DEFAULT_ERA5DROUGHT_KEUNEETAL_FILE = Path(
    os.environ.get(
        "GCMAGICC_ERA5DROUGHT_FILE",
        str(
            _CMIPCRUNCHER_ROOT
            / "data/raw/ERA5/13Jan2026/drought/regridded_1x1_lon360/ERA5Drought_spei48_1943-2025_1x1_lon360.nc"
        ),
    )
).expanduser().resolve(strict=False)
DEFAULT_REGIONMASK_ROOT = Path(
    os.environ.get(
        "GCMAGICC_REGIONMASK_ROOT",
        str(
            _prefer_existing_path(
                _REPO_ROOT / "data" / "regionmasks",
                get_repo_path("gcmagicc_ensemble_runner") / "data" / "regionmasks",
            )
        ),
    )
).expanduser().resolve(strict=False)
APPLY_LANDMASK_IPCCAR6REGIONS = True


DEFAULT_REGION = "SYR"
DEFAULT_SCENARIO1 ="ssp245" # "NDC-Trump-low" # "ssp245"  # "NDC-submitted-low" # "ssp245"
DEFAULT_SCENARIO2 = "ssp245-nat" # "NDC-submitted-low" # "ssp245-nat" # "NDC-Trump-low" #"ssp245-nat"
# Keep in sync with 754 default unless overridden via --scale.
DEFAULT_SCALE = 48
DEFAULT_PET_METHOD = "thornthwaite"
DEFAULT_SPEIX_TAG: Optional[str] = None

PET_METHOD_CHOICES = (
    "penman-monteith",
    "hargreaves",
    "thornthwaite",
    "pet-penman-monteith",
    "pet-hargreaves",
    "pet-thornthwaite",
)

# Default plotting windows (match 754 defaults)
PLOT_START_YEAR = 1975
PLOT_END_YEAR = 2024
MAP_AGG_START = 2021
MAP_AGG_END = 2024
HIST_START = 2021
HIST_END = 2025
# Limit for plotting to keep figures readable
DEFAULT_LIMIT_ENSEMBLES: Optional[int] = 100
MAX_GRID_TRACES = 10
ERA5DROUGHT_LINESTYLES = ("-", "--", ":", "-.")
ERA5DROUGHT_OVERLAY_COLOR = "#9980bf"
ERA5DROUGHT_PANEL_H_LEGEND = "Keune et al, 2025, ERA5-Drought, SPEI48 Penman-Monteith"
ERA5DROUGHT_MAP_TITLE = "Keune et al., ERA5-Drought"
ERA5DROUGHT_NAN_COLOR = "#bfbfbf"

ROW_COLORS = {
    # base colors for general traces / hist bars
    "all": "#dba25c",
    "nat": "#5c9edb",
    "era5": "#2d8069",
    "cmip6_hist": "#b7473a",
    "cmip6_hist_nat": "#3a78b7",
    "cmip6_ssp245": "#2f8f46",
}

# Highlight colors for bold mean/median overlays
ROW_MEAN_COLORS = {
    "all": "#ab6a1b",
    "nat": "#18658c",
    "era5": "#0e3329",
    "cmip6_hist": "#7b2e25",
    "cmip6_hist_nat": "#6cd5e6",
}

CMIP6_PANEL_COLOR = "#a31849"
CMIP6_PANEL_LW = 0.5
CMIP6_PANEL_ALPHA = 0.2

def _spei_cmap() -> LinearSegmentedColormap:
    """Continuous diverging colormap roughly matching provided palette."""
    # Anchors: strong reds for negative, green at zero, blues for positive
    colors = [
        (-3.0, "#5c0000"),
        (-2.33, "#a01212"),
        (-1.65, "#d02b2b"),
        (-1.28, "#f04c1a"),
        (-0.84, "#f0a000"),
        (-0.2, "#a5d66d"),
        (0.0, "#b9e980"),
        (0.2, "#6de1e8"),
        (0.84, "#00c3ff"),
        (1.28, "#1b8be0"),
        (1.65, "#1164c1"),
        (2.33, "#0a2570"),
        (3.0, "#081547"),
    ]
    xs, cs = zip(*colors)
    # Normalize positions 0..1
    x_min, x_max = min(xs), max(xs)
    pos = [(x - x_min) / (x_max - x_min) for x in xs]
    return LinearSegmentedColormap.from_list("spei_diverging", list(zip(pos, cs)))

SPEI_CMAP = _spei_cmap()


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    """Place a small bold panel label in the top-left corner."""
    ax.text(0.01, 0.99, label, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")


# -----------------------------------------------------------------------------
# Small containers
# -----------------------------------------------------------------------------
@dataclass
class SPEISeries:
    label: str
    source: str
    # time stored as fractional years (e.g., 2020.5 ~ mid-year); keeps us off pandas limits.
    time: np.ndarray  # float years
    years: np.ndarray  # int years
    months: np.ndarray  # int months 1-12
    values: np.ndarray  # shape (time, points)
    lat: Optional[np.ndarray] = None  # shape (points,)
    lon: Optional[np.ndarray] = None  # shape (points,)
    pet_method: Optional[str] = None
    baseline_source: Optional[str] = None
    baseline_pooling: Optional[str] = None
    baseline_strategy: Optional[str] = None
    baseline_start_year: Optional[int] = None
    baseline_end_year: Optional[int] = None
    baseline_fit_file: Optional[str] = None

    def spatial_mean(self) -> np.ndarray:
        if self.values.ndim == 1:
            return self.values
        return np.nanmean(self.values, axis=1)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def _require_xarray() -> None:
    if xr is None:
        raise RuntimeError("xarray is required for this script but could not be imported.") from XR_IMPORT_ERROR


def _s3_storage_options() -> Dict[str, object]:
    opts: Dict[str, object] = {}
    endpoint = (
        os.environ.get("AWS_ENDPOINT_URL_S3")
        or os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("GCMAGICC_S3_ENDPOINT_URL")
    )
    if endpoint:
        opts["client_kwargs"] = {"endpoint_url": endpoint}
    if os.environ.get("AWS_NO_SIGN_REQUEST", "").strip().lower() in {"1", "true", "yes"}:
        opts["anon"] = True
    force_path_style = os.environ.get("GCMAGICC_S3_FORCE_PATH_STYLE", "1").strip().lower()
    if force_path_style not in {"0", "false", "no"}:
        opts["config_kwargs"] = {"s3": {"addressing_style": "path"}}
    return opts


def _looks_like_derivative_path(path: Path) -> bool:
    derivative_tokens = {"data_derivatives", "data_derivatives_archive", "dataderivatives"}
    return any(part in derivative_tokens for part in path.parts)


def _resolve_dataset_open_target(path: Path | str) -> tuple[Path | str, Dict[str, object] | None]:
    local_path = Path(path).expanduser().resolve(strict=False)
    if _ACTIVE_STORAGE_ACCESS != STORAGE_ACCESS_S3_DIRECT:
        return local_path, None
    if _looks_like_derivative_path(local_path):
        return local_path, None
    s3_uri = convert_local_path_to_s3_uri(local_path)
    if not s3_uri:
        return local_path, None
    return s3_uri, _s3_storage_options()


def _open_dataset_safe(path: Path | str, **kwargs) -> xr.Dataset:
    _require_xarray()
    target, storage_options = _resolve_dataset_open_target(path)
    errors: List[Exception] = []
    is_s3_target = isinstance(target, str) and target.startswith("s3://")

    if is_s3_target:
        try:
            import s3fs  # type: ignore  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "storage-access=s3_direct requires the 's3fs' package in the environment."
            ) from exc
        for extra in (
            {"storage_options": storage_options} if storage_options else {},
            {"backend_kwargs": {"storage_options": storage_options}} if storage_options else {},
            {},
        ):
            try:
                return xr.open_dataset(target, **kwargs, **extra)  # type: ignore[arg-type]
            except Exception as exc:
                errors.append(exc)
        raise errors[-1]

    return xr.open_dataset(target, **kwargs)  # type: ignore[arg-type]


def _time_arrays(time_da: xr.DataArray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert xarray time coordinate to fractional years (float), years (int), months (int)
    without relying on pandas DateTimeIndex to avoid calendar and range issues.
    """
    time_idx = None
    try:
        time_idx = time_da.indexes.get("time")
    except Exception:
        pass

    # Prefer index values; fall back to raw values
    try:
        raw = np.asarray(time_idx) if time_idx is not None else np.asarray(time_da.values)
    except Exception:
        raw = np.asarray(time_da.values)

    if raw.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=int), np.asarray([], dtype=int)

    sample = raw.flat[0]

    def _year_month_day_from_cftime(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        years = np.array([getattr(t, "year", 0) for t in arr], dtype=int)
        months = np.array([getattr(t, "month", 1) for t in arr], dtype=int)
        days = np.array([getattr(t, "day", 15) for t in arr], dtype=int)
        return years, months, days

    # CFTimeIndex or cftime objects (non-standard calendars, extended ranges)
    if hasattr(sample, "calendar") or "cftime" in sample.__class__.__module__:
        years, months, days = _year_month_day_from_cftime(raw)
    else:
        # Attempt numpy datetime64 extraction; handle failure gracefully
        try:
            years = raw.astype("datetime64[Y]").astype(int) + 1970
            months = (raw.astype("datetime64[M]").astype(int) % 12) + 1
            days = (raw.astype("datetime64[D]") - raw.astype("datetime64[M]") + 1).astype(int)
        except Exception:
            years, months, days = _year_month_day_from_cftime(raw)

    # Fractional year with a simple month/day approximation (sufficient for plotting)
    frac = years.astype(float) + (months - 1) / 12.0 + (days - 0.5) / 365.0
    return frac.astype(float), years.astype(int), months.astype(int)


def _extract_lat_lon(arr: xr.DataArray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        if "lat" in arr.coords and "lon" in arr.coords:
            lat = np.asarray(arr["lat"].values)
            lon = np.asarray(arr["lon"].values)
            if lat.shape == lon.shape == (arr.sizes.get("point", 0),):
                return lat, lon
    except Exception:
        pass
    return None, None


def _ensure_lon_0_360_dataset(ds: xr.Dataset) -> xr.Dataset:
    lon_name = next((n for n in ("lon", "longitude", "x") if n in ds.coords), None)
    if lon_name is None:
        return ds
    lon = np.asarray(ds[lon_name].values, dtype=float)
    if lon.size and np.nanmin(lon) < 0.0:
        ds = ds.assign_coords({lon_name: np.mod(ds[lon_name], 360.0)})
        ds = ds.sortby(lon_name)
    return ds


def _align_mask_to_grid(
    mask: np.ndarray,
    mask_lats: np.ndarray,
    mask_lons: np.ndarray,
    data_lats: np.ndarray,
    data_lons: np.ndarray,
) -> np.ndarray:
    mask_lons = np.mod(np.asarray(mask_lons, dtype=float), 360.0)
    data_lons = np.mod(np.asarray(data_lons, dtype=float), 360.0)
    mask_lats = np.asarray(mask_lats, dtype=float)
    data_lats = np.asarray(data_lats, dtype=float)
    lat_index = [int(np.argmin(np.abs(mask_lats - v))) for v in data_lats]
    lon_index = [int(np.argmin(np.abs(mask_lons - v))) for v in data_lons]
    return np.asarray(mask, dtype=bool)[np.ix_(lat_index, lon_index)]


def _load_npz_region_mask_aligned(
    region: str,
    *,
    data_lats: np.ndarray,
    data_lons: np.ndarray,
    regionmask_root: Path = DEFAULT_REGIONMASK_ROOT,
) -> Optional[np.ndarray]:
    region_norm = region.upper().replace(" ", "_").replace("/", "_")
    nlat = int(np.asarray(data_lats).size)
    nlon = int(np.asarray(data_lons).size)
    candidates = [
        regionmask_root / f"{region_norm}_nlat{nlat}_nlon{nlon}_lon360.npz",
        regionmask_root / f"{region_norm}_nlat{nlat}_nlon{nlon}_lon180.npz",
    ]
    for mask_path in candidates:
        if not mask_path.exists():
            continue
        try:
            with np.load(mask_path, allow_pickle=True) as data:
                mask = data["mask"].astype(bool)
                lats = np.asarray(data["lats"], dtype=float)
                lons = np.asarray(data["lons"], dtype=float)
            return _align_mask_to_grid(mask, lats, lons, data_lats, data_lons)
        except Exception:
            continue
    return None


def _canonical_region_mask_aligned(
    region: str,
    *,
    data_lats: np.ndarray,
    data_lons: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if _build_canonical_region_mask is None:
        return None, "canonical-builder-unavailable"
    try:
        mask = _build_canonical_region_mask(
            region,
            np.asarray(data_lats, dtype=float),
            np.asarray(data_lons, dtype=float),
            1,
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return np.asarray(mask, dtype=bool), None


def _region_mask_resolution_summary(
    region: str,
    *,
    data_lats: np.ndarray,
    data_lons: np.ndarray,
    regionmask_root: Path = DEFAULT_REGIONMASK_ROOT,
) -> str:
    region_norm = region.upper().replace(" ", "_").replace("/", "_")
    nlat = int(np.asarray(data_lats).size)
    nlon = int(np.asarray(data_lons).size)
    npz_candidates = [
        regionmask_root / f"{region_norm}_nlat{nlat}_nlon{nlon}_lon360.npz",
        regionmask_root / f"{region_norm}_nlat{nlat}_nlon{nlon}_lon180.npz",
    ]
    npz_present = [path.name for path in npz_candidates if path.exists()]

    if regionmask is None:
        ar6_status = "regionmask-unavailable"
    else:
        try:
            ar6 = regionmask.defined_regions.ar6.all
            region_id = _resolve_ar6_region_id(ar6, region)
            ar6_status = f"matched:{region_id}" if region_id is not None else "no-match"
        except Exception as exc:
            ar6_status = f"error:{type(exc).__name__}:{exc}"

    if npz_present:
        npz_status = "present:" + ",".join(npz_present)
    else:
        npz_status = "missing:" + ",".join(path.name for path in npz_candidates)

    _, canonical_error = _canonical_region_mask_aligned(
        region,
        data_lats=data_lats,
        data_lons=data_lons,
    )
    canonical_status = "matched" if canonical_error is None else f"failed:{canonical_error}"

    return (
        "resolution attempts -> "
        f"ar6={ar6_status}; "
        f"npz={npz_status}; "
        f"canonical={canonical_status}"
    )


def _is_iso3_country_code(region: str) -> bool:
    reg = str(region or "").strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", reg):
        return False
    if pycountry is not None:
        try:
            return pycountry.countries.get(alpha_3=reg) is not None
        except Exception:
            return False
    return True


def _is_ipcc_ar6_region(region: str) -> bool:
    reg = str(region or "").strip()
    if reg.lower() == "global":
        return False
    if regionmask is not None:
        try:
            ar6 = regionmask.defined_regions.ar6.all
            if _resolve_ar6_region_id(ar6, reg) is not None:
                return True
        except Exception:
            pass
    if _is_iso3_country_code(reg):
        return False
    return any(ch in reg for ch in (".", "-", " "))


def _resolve_ar6_region_id(ar6: Any, region: str) -> Optional[int]:
    """Resolve AR6 region id while preserving special-key punctuation."""
    candidates = [str(region or "").strip(), str(region or "").strip().upper()]
    seen: Set[str] = set()
    for key in candidates:
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            return int(ar6.map_keys(key))
        except Exception:
            continue
    return None


def _land_regions() -> Optional[Any]:
    if regionmask is None:
        return None
    try:
        ne = regionmask.defined_regions.natural_earth_v5_0_0
    except Exception:
        return None
    for attr in ("land_110", "land_50", "land_10"):
        reg = getattr(ne, attr, None)
        if reg is not None:
            return reg
    return None


def _land_mask_da(
    *,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    lat_name: str,
    lon_name: str,
) -> Optional[xr.DataArray]:
    land = _land_regions()
    if land is None:
        return None
    try:
        lon_360 = np.mod(np.asarray(lon_vals, dtype=float), 360.0)
        lat_arr = np.asarray(lat_vals, dtype=float)
        land_mask = land.mask(lon_360, lat_arr)
        keep = np.isfinite(np.asarray(land_mask.values))
        return xr.DataArray(
            keep,
            dims=(lat_name, lon_name),
            coords={lat_name: lat_vals, lon_name: lon_vals},
        )
    except Exception:
        return None


def _apply_landmask_to_point_values(
    values: np.ndarray,
    *,
    lat: Optional[np.ndarray],
    lon: Optional[np.ndarray],
    region: str,
    apply_landmask_ipcc_ar6_regions: bool,
) -> np.ndarray:
    if not apply_landmask_ipcc_ar6_regions or not _is_ipcc_ar6_region(region):
        return values
    if lat is None or lon is None:
        return values

    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim < 1:
        return arr

    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.mod(np.asarray(lon, dtype=float), 360.0)
    n_points = lat_arr.size
    if lon_arr.size != n_points:
        return arr

    if arr.ndim == 1:
        if arr.shape[0] != n_points:
            return arr
    else:
        if arr.shape[1] != n_points:
            return arr

    land = _land_regions()
    if land is None:
        return arr
    try:
        da_lon = xr.DataArray(lon_arr, dims=("point",), coords={"point": np.arange(n_points)})
        da_lat = xr.DataArray(lat_arr, dims=("point",), coords={"point": np.arange(n_points)})
        point_mask = land.mask(da_lon, da_lat)
        keep = np.isfinite(np.asarray(point_mask.values))
        if arr.ndim == 1:
            out = arr.copy()
            out[~keep] = np.nan
            return out
        out = arr.copy()
        out[:, ~keep] = np.nan
        return out
    except Exception:
        return arr


def _region_mask_da(
    region: str,
    *,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    lat_name: str,
    lon_name: str,
    apply_landmask_ipcc_ar6_regions: bool = APPLY_LANDMASK_IPCCAR6REGIONS,
) -> Optional[xr.DataArray]:
    if region.lower() == "global":
        mask_arr = np.ones((lat_vals.size, lon_vals.size), dtype=bool)
        return xr.DataArray(mask_arr, dims=(lat_name, lon_name), coords={lat_name: lat_vals, lon_name: lon_vals})

    mask_arr: Optional[np.ndarray] = None

    if regionmask is not None:
        try:
            ar6 = regionmask.defined_regions.ar6.all
            region_id = _resolve_ar6_region_id(ar6, region)
            if region_id is not None:
                mask = ar6.mask(np.mod(lon_vals, 360.0), lat_vals)
                mask_arr = np.asarray(mask.values == region_id, dtype=bool)
        except Exception:
            mask_arr = None

    if mask_arr is None:
        mask_arr = _load_npz_region_mask_aligned(
            region,
            data_lats=lat_vals,
            data_lons=lon_vals,
        )

    if mask_arr is None:
        mask_arr, _ = _canonical_region_mask_aligned(
            region,
            data_lats=lat_vals,
            data_lons=lon_vals,
        )

    if mask_arr is None:
        return None
    out = xr.DataArray(mask_arr, dims=(lat_name, lon_name), coords={lat_name: lat_vals, lon_name: lon_vals})
    if apply_landmask_ipcc_ar6_regions and _is_ipcc_ar6_region(region):
        land_da = _land_mask_da(
            lat_vals=lat_vals,
            lon_vals=lon_vals,
            lat_name=lat_name,
            lon_name=lon_name,
        )
        if land_da is not None:
            out = out & land_da
    return out


def _canonical_pet_method(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    text = str(raw).strip().lower().replace("_", "-")
    if text.startswith("pet-"):
        text = text[4:]
    if "penman" in text:
        return "penman-monteith"
    if "harg" in text:
        return "hargreaves"
    if "thorn" in text:
        return "thornthwaite"
    return None


def _format_pet_method_label(raw: Optional[str]) -> str:
    canon = _canonical_pet_method(raw)
    if canon == "penman-monteith":
        return "Penman-Monteith"
    if canon == "hargreaves":
        return "Hargreaves"
    if canon == "thornthwaite":
        return "Thornthwaite"
    text = (str(raw).strip() if raw is not None else "") or "SPEI"
    return text.replace("_", "-")


def _extract_era5drought_method_fields(ds: xr.Dataset, *, scale: int) -> List[Tuple[str, xr.DataArray]]:
    method_dim_names = {"pet_method", "pet", "method", "petmethod", "pet-method", "pet_methods"}
    fields: List[Tuple[str, xr.DataArray]] = []

    for var_name, da in ds.data_vars.items():
        low = var_name.lower()
        if "spei" not in low:
            continue
        match = re.search(r"spei\s*0*(\d+)", low)
        if match and int(match.group(1)) != int(scale):
            continue

        method_dim = next((d for d in da.dims if d.lower() in method_dim_names), None)
        if method_dim and method_dim in da.coords:
            for method_val in np.asarray(da[method_dim].values).tolist():
                da_sel = da.sel({method_dim: method_val}, drop=True)
                label = _format_pet_method_label(method_val)
                fields.append((label, da_sel))
            continue

        pet_guess = None
        for candidate in (
            da.attrs.get("pet_method"),
            da.attrs.get("method"),
            da.attrs.get("long_name"),
            var_name,
            ds.attrs.get("title"),
            ds.attrs.get("description"),
            ds.attrs.get("source_file"),
        ):
            pet_guess = _canonical_pet_method(candidate)
            if pet_guess:
                break
        label = _format_pet_method_label(pet_guess) if pet_guess else f"SPEI{scale}"
        fields.append((label, da))

    if not fields and len(ds.data_vars) == 1:
        only_name = next(iter(ds.data_vars))
        fields = [(f"{only_name}", ds[only_name])]

    order = {"penman-monteith": 0, "hargreaves": 1, "thornthwaite": 2}

    def _sort_key(item: Tuple[str, xr.DataArray]) -> Tuple[int, str]:
        canon = _canonical_pet_method(item[0])
        return (order.get(canon, 99), item[0].lower())

    fields.sort(key=_sort_key)

    label_counts: Dict[str, int] = {}
    deduped: List[Tuple[str, xr.DataArray]] = []
    for label, da in fields:
        n = label_counts.get(label, 0) + 1
        label_counts[label] = n
        final_label = label if n == 1 else f"{label} #{n}"
        deduped.append((final_label, da))
    return deduped


def _area_weighted_region_mean(
    da: xr.DataArray,
    *,
    region: str,
    apply_landmask_ipcc_ar6_regions: bool = APPLY_LANDMASK_IPCCAR6REGIONS,
) -> Optional[xr.DataArray]:
    lat_name = next((n for n in ("lat", "latitude", "y") if n in da.coords), None)
    lon_name = next((n for n in ("lon", "longitude", "x") if n in da.coords), None)
    time_name = next((n for n in ("time", "Time") if n in da.coords or n in da.dims), None)
    if lat_name is None or lon_name is None or time_name is None:
        return None

    work = da
    lon_vals = np.asarray(work[lon_name].values, dtype=float)
    if lon_vals.size and np.nanmin(lon_vals) < 0.0:
        work = work.assign_coords({lon_name: np.mod(work[lon_name], 360.0)})
        work = work.sortby(lon_name)

    lat_vals = np.asarray(work[lat_name].values, dtype=float)
    lon_vals = np.asarray(work[lon_name].values, dtype=float)

    mask_da = _region_mask_da(
        region,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
        lat_name=lat_name,
        lon_name=lon_name,
        apply_landmask_ipcc_ar6_regions=apply_landmask_ipcc_ar6_regions,
    )
    if mask_da is None:
        return None

    weights = xr.DataArray(
        np.cos(np.deg2rad(lat_vals)).astype(np.float32),
        dims=(lat_name,),
        coords={lat_name: work[lat_name]},
    )
    masked = work.where(mask_da)
    return masked.weighted(weights).mean(dim=(lat_name, lon_name), skipna=True)


def _load_era5drought_keuneetal_series(
    era5drought_file: Path,
    *,
    region: str,
    scale: int,
    apply_landmask_ipcc_ar6_regions: bool = APPLY_LANDMASK_IPCCAR6REGIONS,
) -> List[SPEISeries]:
    _require_xarray()
    if not era5drought_file.exists():
        print(f"⚠️ ERA5Drought Keune et al. file not found: {era5drought_file}")
        return []

    out: List[SPEISeries] = []
    try:
        with _open_dataset_safe(era5drought_file) as ds_raw:
            ds = _ensure_lon_0_360_dataset(ds_raw)
            method_fields = _extract_era5drought_method_fields(ds, scale=scale)
            if not method_fields:
                print(f"⚠️ No SPEI fields found in ERA5Drought file: {era5drought_file}")
                return []

            for method_label, da in method_fields:
                ts = _area_weighted_region_mean(
                    da,
                    region=region,
                    apply_landmask_ipcc_ar6_regions=apply_landmask_ipcc_ar6_regions,
                )
                if ts is None:
                    continue
                time_frac, years, months = _time_arrays(ts["time"])
                values = np.asarray(ts.values, dtype=np.float32).reshape(-1, 1)
                out.append(
                    SPEISeries(
                        label=method_label,
                        source="ERA5Drought Keune et al.",
                        time=time_frac,
                        years=years,
                        months=months,
                        values=values,
                        lat=None,
                        lon=None,
                        pet_method=method_label,
                    )
                )
    except Exception as exc:
        print(
            "⚠️ Could not open/read ERA5Drought Keune et al. file: "
            f"{era5drought_file} ({type(exc).__name__}: {str(exc).splitlines()[0]})"
        )
        return []

    if not out:
        print(
            f"⚠️ Could not derive ERA5Drought regional means for region={region}. "
            "AR6/country mask may be missing."
        )
    return out


def _load_era5drought_keuneetal_map_snapshot(
    era5drought_file: Path,
    *,
    region: str,
    scale: int,
    pet_method: str,
    year: int,
    month: Optional[int] = None,
    apply_landmask_ipcc_ar6_regions: bool = APPLY_LANDMASK_IPCCAR6REGIONS,
) -> Optional[SPEISeries]:
    """
    Build a single-time SPEI map snapshot for ERA5Drought (Keune et al.).
    If month is provided, select that year-month snapshot; otherwise use annual mean.
    The selected PET method follows --pet-method when available; otherwise first method.
    """
    _require_xarray()
    if not era5drought_file.exists():
        return None

    try:
        with _open_dataset_safe(era5drought_file) as ds_raw:
            ds = _ensure_lon_0_360_dataset(ds_raw)
            method_fields = _extract_era5drought_method_fields(ds, scale=scale)
            if not method_fields:
                return None

            target_pet = _canonical_pet_method(pet_method)
            picked_label, picked_da = method_fields[0]
            if target_pet is not None:
                for label, da in method_fields:
                    if _canonical_pet_method(label) == target_pet:
                        picked_label, picked_da = label, da
                        break

            lat_name = next((n for n in ("lat", "latitude", "y") if n in picked_da.coords), None)
            lon_name = next((n for n in ("lon", "longitude", "x") if n in picked_da.coords), None)
            time_name = next((n for n in ("time", "Time") if n in picked_da.coords or n in picked_da.dims), None)
            if lat_name is None or lon_name is None or time_name is None:
                return None

            work = picked_da
            # Drop unsupported extra dims conservatively.
            for d in list(work.dims):
                if d in {time_name, lat_name, lon_name}:
                    continue
                work = work.isel({d: 0}, drop=True)

            _time_frac, years, months = _time_arrays(work[time_name])
            time_mask_np = years == int(year)
            if month is not None:
                time_mask_np = time_mask_np & (months == int(month))
            year_mask = xr.DataArray(
                time_mask_np,
                dims=(time_name,),
                coords={time_name: work[time_name]},
            )
            work_year = work.where(year_mask, drop=True)
            if int(work_year.sizes.get(time_name, 0)) == 0:
                return None

            lat_vals = np.asarray(work_year[lat_name].values, dtype=float)
            lon_vals = np.asarray(work_year[lon_name].values, dtype=float)
            mask_da = _region_mask_da(
                region,
                lat_vals=lat_vals,
                lon_vals=lon_vals,
                lat_name=lat_name,
                lon_name=lon_name,
                apply_landmask_ipcc_ar6_regions=apply_landmask_ipcc_ar6_regions,
            )
            if mask_da is None:
                return None

            mean_field = work_year.where(mask_da).mean(dim=time_name, skipna=True)
            try:
                mean_field = mean_field.transpose(lat_name, lon_name)
            except Exception:
                pass

            vals_2d = np.asarray(mean_field.values, dtype=np.float32)
            if vals_2d.ndim != 2:
                return None
            if vals_2d.shape != (lat_vals.size, lon_vals.size):
                try:
                    vals_2d = vals_2d.reshape(lat_vals.size, lon_vals.size)
                except Exception:
                    return None

            lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
            vals_flat = vals_2d.reshape(-1)
            lat_flat = lat_grid.reshape(-1)
            lon_flat = lon_grid.reshape(-1)
            keep = np.asarray(mask_da.values, dtype=bool).reshape(-1)
            if not np.any(keep):
                return None

            if month is not None:
                time_value = float(year) + (float(month) - 0.5) / 12.0
                out_month = int(month)
                out_label = f"{year}-{int(month):02d}"
            else:
                time_value = float(year) + 0.5
                out_month = 7
                out_label = f"{year} mean"

            return SPEISeries(
                label=f"ERA5Drought {_format_pet_method_label(picked_label)} {out_label}",
                source="ERA5Drought Keune et al.",
                time=np.array([time_value], dtype=float),
                years=np.array([int(year)], dtype=int),
                months=np.array([out_month], dtype=int),
                values=vals_flat[keep].reshape(1, -1).astype(np.float32),
                lat=lat_flat[keep].astype(np.float32),
                lon=lon_flat[keep].astype(np.float32),
                pet_method=_canonical_pet_method(picked_label),
            )
    except Exception as exc:
        print(
            "⚠️ Could not build ERA5Drought map snapshot from file: "
            f"{era5drought_file} ({type(exc).__name__}: {str(exc).splitlines()[0]})"
        )
        return None


def _plot_era5drought_keuneetal_overlays(
    ax: plt.Axes,
    series_list: List[SPEISeries],
) -> List[Any]:
    handles: List[Any] = []
    for s in series_list:
        vals = _region_mean(s.values)
        if vals.size == 0:
            continue
        line, = ax.plot(
            s.time,
            vals,
            color=ERA5DROUGHT_OVERLAY_COLOR,
            linestyle=":",
            linewidth=0.9,
            alpha=0.95,
            zorder=3.2,
            label=s.label,
        )
        handles.append(line)
    return handles


def _pick_era5drought_series_for_pet(
    series_list: Sequence[SPEISeries],
    *,
    pet_method: str,
) -> Optional[SPEISeries]:
    if not series_list:
        return None
    target_pet = _canonical_pet_method(pet_method)
    if target_pet is not None:
        for s in series_list:
            cand = _canonical_pet_method(s.pet_method) or _canonical_pet_method(s.label)
            if cand == target_pet:
                return s
    return series_list[0]


def _available_year_months(series: SPEISeries) -> Set[Tuple[int, int]]:
    years = np.asarray(series.years, dtype=int)
    months = np.asarray(series.months, dtype=int)
    vals = np.asarray(series.values, dtype=float)
    if vals.ndim == 1:
        finite_time = np.isfinite(vals)
    else:
        finite_time = np.any(np.isfinite(vals), axis=1)
    out: Set[Tuple[int, int]] = set()
    n = min(years.size, months.size, finite_time.size)
    for idx in range(n):
        if bool(finite_time[idx]):
            out.add((int(years[idx]), int(months[idx])))
    return out


def _latest_common_year_month(
    series_a: SPEISeries,
    series_b: SPEISeries,
) -> Optional[Tuple[int, int]]:
    common = _available_year_months(series_a) & _available_year_months(series_b)
    if not common:
        return None
    return sorted(common)[-1]


def _build_map_snapshot_from_series(
    series: SPEISeries,
    *,
    year: int,
    month: int,
    label: str,
) -> Optional[SPEISeries]:
    mask = (np.asarray(series.years, dtype=int) == int(year)) & (np.asarray(series.months, dtype=int) == int(month))
    if not np.any(mask):
        return None
    idx = np.where(mask)[0]
    vals = np.asarray(series.values[mask], dtype=np.float32)
    if vals.ndim == 1:
        field = vals.astype(np.float32, copy=False).reshape(-1)
    else:
        field = np.nanmean(vals, axis=0).astype(np.float32, copy=False).reshape(-1)
    if field.size == 0 or not np.any(np.isfinite(field)):
        return None
    time_val = float(np.nanmean(np.asarray(series.time[idx], dtype=float)))
    return SPEISeries(
        label=label,
        source=series.source,
        time=np.array([time_val], dtype=float),
        years=np.array([int(year)], dtype=int),
        months=np.array([int(month)], dtype=int),
        values=field.reshape(1, -1),
        lat=series.lat,
        lon=series.lon,
        pet_method=series.pet_method,
        baseline_source=series.baseline_source,
        baseline_pooling=series.baseline_pooling,
        baseline_strategy=series.baseline_strategy,
        baseline_start_year=series.baseline_start_year,
        baseline_end_year=series.baseline_end_year,
        baseline_fit_file=series.baseline_fit_file,
    )


def _run_matches_token(run_name: str, token: str) -> bool:
    pattern = rf"(?:^|_){re.escape(token)}(?:_|$)"
    return re.search(pattern, run_name) is not None


def _token_mentions_ssp245(token: Optional[str]) -> bool:
    if not token:
        return False
    compact = re.sub(r"[^a-z0-9]+", "", str(token).lower())
    return "ssp245" in compact


def _token_has_nat_suffix(token: Optional[str]) -> bool:
    if not token:
        return False
    t = str(token).strip().lower()
    return t.endswith("-nat")


def _resolve_cmip6_overlay_experiment(
    scenario_tag: Optional[str],
    *,
    fallback: str,
) -> str:
    """
    Map selected scenario tags to CMIP6 overlay experiments.
    If a scenario tag indicates ssp245, use CMIP6 ssp245 overlays.
    Otherwise keep the requested fallback experiment.
    """
    return "ssp245" if _token_mentions_ssp245(scenario_tag) else fallback


def _cmip6_overlay_label(exp: str) -> str:
    exp_norm = str(exp).strip().lower()
    if exp_norm == "hist-nat":
        return "CMIP6 hist-nat"
    if exp_norm == "ssp245":
        return "CMIP6 ssp245"
    return "CMIP6 historical"


def _cmip6_overlay_color(exp: str) -> str:
    exp_norm = str(exp).strip().lower()
    if exp_norm == "hist-nat":
        return ROW_COLORS["cmip6_hist_nat"]
    if exp_norm == "ssp245":
        return ROW_COLORS["cmip6_ssp245"]
    return ROW_COLORS["cmip6_hist"]


def _safe_region_tag(region: str) -> str:
    """
    Filesystem-safe version of an AR6 region key for output paths.
    Example: 'C.North-America' -> 'C_North-America'
    """
    safe = region.strip().replace(" ", "_").replace(".", "_").replace("/", "_")
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe


def _output_region_token(region: str) -> str:
    """Filename-safe region token aligned with output directory/stat naming."""
    return _safe_region_tag(str(region or "")).upper()


def _safe_slug(token: Optional[str], fallback: str = "unknown") -> str:
    """Filesystem-safe slug for scenario labels or other tokens."""
    if not token:
        return fallback
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", token.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or fallback


def _payload_stem(path: Path) -> str:
    """Return the stem of a payload path, handling .json.gz transparently."""
    name = Path(path).name
    if name.endswith(".json.gz"):
        return name[:-8]
    if name.endswith(".json"):
        return name[:-5]
    return Path(path).stem


def _payload_path_variants(path: Path) -> List[Path]:
    """Return plain/gz variants for a payload path, preserving caller order first."""
    p = Path(path)
    if p.name.endswith(".json.gz"):
        plain = p.with_name(p.name[:-3])
        return [p, plain]
    if p.name.endswith(".json"):
        gz = p.with_name(p.name + ".gz")
        return [p, gz]
    return [p]


def _resolve_payload_path(path: Path) -> Path:
    """Resolve a payload path from either .json or .json.gz."""
    for candidate in _payload_path_variants(Path(path)):
        if candidate.exists():
            return candidate
    return Path(path)


def _load_payload(path: Path) -> Dict[str, Any]:
    """Load JSON payloads from plain or gzip-compressed files."""
    resolved = _resolve_payload_path(path)
    opener = gzip.open if resolved.name.endswith(".gz") else open
    with opener(resolved, "rt", encoding="utf-8") as f:
        return json.load(f)


def _compress_payload_json(path: Path, *, remove_original: bool = True, compresslevel: int = 1) -> Optional[Path]:
    """Compress a large payload JSON to .json.gz and optionally remove the plain JSON."""
    plain = Path(path)
    if plain.name.endswith(".json.gz"):
        return plain if plain.exists() else None
    if plain.suffix != ".json" or not plain.exists():
        return None
    gz_path = plain.with_name(plain.name + ".gz")
    tmp_path = gz_path.with_name(gz_path.name + ".tmp")
    with plain.open("rb") as src, gzip.open(tmp_path, "wb", compresslevel=int(compresslevel)) as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    os.replace(tmp_path, gz_path)
    if remove_original and plain.exists():
        plain.unlink()
    return gz_path


def _scenario_pair_tag(scenario1: Optional[str], scenario2: Optional[str]) -> str:
    """Filesystem-safe scenario-pair token used in output folder hierarchy."""
    return f"{_safe_slug(scenario1, 'scenario1')}_{_safe_slug(scenario2, 'scenario2')}"


def _extract_version_tag_from_path(path: Path) -> Optional[str]:
    """Best-effort version tag extraction (e.g., v100, v101, v101gxe) from a path."""
    for candidate in [path, *path.parents]:
        token = candidate.name.strip()
        if re.fullmatch(r"v[0-9]+[A-Za-z0-9_-]*", token):
            return token
    return None


def _resolve_run_version_tag(
    explicit_tag: Optional[str],
    *roots: Path,
) -> str:
    """
    Resolve the model version tag used for output namespacing and JSON metadata.
    Priority: explicit CLI value -> roots -> site default.
    """
    if explicit_tag and str(explicit_tag).strip():
        return str(explicit_tag).strip()
    for root in roots:
        tag = _extract_version_tag_from_path(Path(root))
        if tag:
            return tag
    return _DEFAULT_VERSION_TAG


def _region_subdir(region: str) -> str:
    """Return the region-specific subdirectory name used by 754 (e.g., 'C.North-America' -> 'region-C_North-America')."""
    safe = region.strip().replace(" ", "_").replace(".", "_")
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return f"region-{safe}"


def _normalize_pet_method(pet_method: str) -> str:
    pet = pet_method.lower().strip()
    if pet.startswith("pet-"):
        pet = pet[4:]
    return pet


def _pet_tag(pet_method: str) -> str:
    return f"pet-{_normalize_pet_method(pet_method)}"


def _format_timetag_human(timetag: str) -> str:
    """
    Convert runtime timetag (e.g., 20260131_145512) into '31 Jan 2026 14:55:12'.
    Falls back to the raw string if parsing fails.
    """
    fmt_candidates = ("%Y%m%d_%H%M%S", "%Y-%m-%d_%H-%M-%S", "%Y%m%d", "%Y-%m-%d")
    for fmt in fmt_candidates:
        try:
            dt = datetime.strptime(timetag, fmt)
            return dt.strftime("%d %b %Y %H:%M:%S")
        except Exception:
            continue
    try:
        # ISO-ish fallback
        dt = datetime.fromisoformat(timetag.replace("T", " ").split(".")[0])
        return dt.strftime("%d %b %Y %H:%M:%S")
    except Exception:
        return timetag


def _add_corner_timetag(fig, human_timetag: str) -> None:
    """Place a tiny bottom-right watermark with the timetag."""
    fig.text(
        0.995,
        0.01,
        f"gcmagicc.org {human_timetag}",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#444",
        alpha=0.9,
    )


def _save_png_pdf(fig, png_path: Path, *, dpi: int = 180, tight: bool = True, human_timetag: Optional[str] = None) -> Tuple[Path, Path]:
    """Save figure to PNG/PDF with optional timetag watermark and tight bounds."""
    if human_timetag:
        _add_corner_timetag(fig, human_timetag)
    save_kwargs = {"dpi": dpi}
    if tight:
        save_kwargs["bbox_inches"] = "tight"
        save_kwargs["pad_inches"] = 0.08
    fig.savefig(png_path, **save_kwargs)
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(pdf_path, **save_kwargs)
    return png_path, pdf_path


def _extract_baseline_meta(ds: xr.Dataset, cache: Dict[str, Dict[str, Optional[str]]]) -> Dict[str, Optional[str]]:
    """
    Gather baseline metadata from the dataset attributes. If baseline_start_year / baseline_end_year
    are missing on the segment dataset, fall back to the referenced baseline fit file.
    """
    attrs = ds.attrs or {}
    fit_path = attrs.get("baseline_fit_file")
    meta: Dict[str, Optional[str]] = {
        "pet_method": attrs.get("pet_method"),
        "baseline_source": attrs.get("baseline_source"),
        "baseline_pooling": attrs.get("baseline_pooling"),
        "baseline_strategy": attrs.get("baseline_strategy"),
        "baseline_start_year": attrs.get("baseline_start_year"),
        "baseline_end_year": attrs.get("baseline_end_year"),
        "baseline_fit_file": fit_path,
    }

    # If the time window isn't present, try to read it from the baseline fit file (cached).
    if (meta["baseline_start_year"] is None or meta["baseline_end_year"] is None) and fit_path:
        if fit_path not in cache:
            try:
                with _open_dataset_safe(fit_path) as ds_fit:
                    cache[fit_path] = {
                        "baseline_start_year": ds_fit.attrs.get("baseline_start_year"),
                        "baseline_end_year": ds_fit.attrs.get("baseline_end_year"),
                        "baseline_source": ds_fit.attrs.get("baseline_source"),
                        "baseline_pooling": ds_fit.attrs.get("baseline_pooling"),
                        "baseline_strategy": ds_fit.attrs.get("baseline_strategy"),
                        "pet_method": ds_fit.attrs.get("pet_method"),
                        "baseline_fit_file": fit_path,
                    }
            except Exception:
                cache[fit_path] = {}
        # Merge any cached values
        for k, v in cache.get(fit_path, {}).items():
            if meta.get(k) is None and v is not None:
                meta[k] = v
    return meta


def _extract_stacked_baseline_meta(
    ds: xr.Dataset,
    *,
    run_index: int,
    cache: Dict[str, Dict[str, Optional[str]]],
) -> Dict[str, Optional[str]]:
    attrs = ds.attrs or {}

    def _run_value(name: str) -> Optional[str]:
        if name not in ds:
            return None
        da = ds[name]
        if "run" not in da.dims:
            return None
        try:
            val = np.asarray(da.isel(run=int(run_index)).values).item()
            text = str(val)
            return text if text else None
        except Exception:
            return None

    fit_path = _run_value("baseline_fit_file") or attrs.get("baseline_fit_file")
    meta: Dict[str, Optional[str]] = {
        "pet_method": attrs.get("pet_method"),
        "baseline_source": attrs.get("baseline_source"),
        "baseline_pooling": _run_value("baseline_pooling") or attrs.get("baseline_pooling"),
        "baseline_strategy": _run_value("baseline_strategy") or attrs.get("baseline_strategy"),
        "baseline_start_year": attrs.get("baseline_start_year"),
        "baseline_end_year": attrs.get("baseline_end_year"),
        "baseline_fit_file": fit_path,
    }
    b_source_key = _run_value("baseline_source_key") or attrs.get("baseline_source_key")
    if meta.get("baseline_source") is None and b_source_key:
        meta["baseline_source"] = b_source_key

    if (meta["baseline_start_year"] is None or meta["baseline_end_year"] is None) and fit_path:
        if fit_path not in cache:
            try:
                with _open_dataset_safe(fit_path) as ds_fit:
                    cache[fit_path] = {
                        "baseline_start_year": ds_fit.attrs.get("baseline_start_year"),
                        "baseline_end_year": ds_fit.attrs.get("baseline_end_year"),
                        "baseline_source": ds_fit.attrs.get("baseline_source"),
                        "baseline_pooling": ds_fit.attrs.get("baseline_pooling"),
                        "baseline_strategy": ds_fit.attrs.get("baseline_strategy"),
                        "pet_method": ds_fit.attrs.get("pet_method"),
                        "baseline_fit_file": fit_path,
                    }
            except Exception:
                cache[fit_path] = {}
        for k, v in cache.get(fit_path, {}).items():
            if meta.get(k) is None and v is not None:
                meta[k] = v
    return meta


def _resolve_store_root(
    store_root: Path,
    *,
    tag: Optional[str] = None,
    region: Optional[str] = None,
    pet_method: Optional[str] = None,
) -> Path:
    """
    Locate a SPEIx store. Supports:
      - Legacy layout: <root>/segments.zarr
      - Tagged layout: <root>/<tag>/segments.zarr
      - Region-scoped layout: <root>/<tag>/region-<SAFE_REGION>/segments.zarr
      - PET-scoped layout: <root>/<tag>/region-<SAFE_REGION>/pet-<PET>/segments.zarr

    If no tag is provided, the most recent tagged directory containing a usable
    segments.zarr (directly, region-scoped, or PET-scoped) is used.
    """
    if not store_root.exists():
        raise FileNotFoundError(f"SPEIx root not found: {store_root}")
    direct = store_root / "segments.zarr"
    if direct.exists():
        return store_root

    region_dir = _region_subdir(region) if region else None
    pet_dir = _pet_tag(pet_method) if pet_method else None

    # Untagged region/PET-scoped layouts.
    if region_dir:
        untagged_region = store_root / region_dir
        if pet_dir and (untagged_region / pet_dir / "segments.zarr").exists():
            return untagged_region / pet_dir
        if (untagged_region / "segments.zarr").exists():
            return untagged_region

    if tag:
        tagged = store_root / tag
        if (tagged / "segments.zarr").exists():
            print(f"Using SPEIx tag '{tag}' under {store_root}")
            return tagged
        if region_dir:
            region_path = tagged / region_dir
            if (region_path / "segments.zarr").exists():
                print(f"Using SPEIx tag '{tag}' (region {region}) under {store_root}")
                return region_path
            if pet_dir:
                region_pet_path = region_path / pet_dir
                if (region_pet_path / "segments.zarr").exists():
                    print(
                        f"Using SPEIx tag '{tag}' (region {region}, pet {pet_method}) "
                        f"under {store_root}"
                    )
                    return region_pet_path

    tagged_candidates_region_pet: List[Tuple[str, Path]] = []
    tagged_candidates_region: List[Tuple[str, Path]] = []
    tagged_candidates_generic: List[Tuple[str, Path]] = []
    for p in store_root.iterdir():
        if not p.is_dir():
            continue
        if (p / "segments.zarr").exists():
            tagged_candidates_generic.append((p.name, p))
            continue
        if region_dir:
            region_path = p / region_dir
            if pet_dir:
                region_pet_path = region_path / pet_dir
                if (region_pet_path / "segments.zarr").exists():
                    tagged_candidates_region_pet.append((p.name, region_pet_path))
                    continue
            if (region_path / "segments.zarr").exists():
                tagged_candidates_region.append((p.name, region_path))
                continue
    # Prefer region+PET, then region-specific, then generic candidates.
    if tagged_candidates_region_pet:
        tagged_candidates_region_pet.sort(key=lambda t: t[0])
        chosen_tag, chosen_path = tagged_candidates_region_pet[-1]
        print(
            f"Using latest SPEIx tag '{chosen_tag}' (region {region}, pet {pet_method}) "
            f"under {store_root}"
        )
        return chosen_path
    if tagged_candidates_region:
        tagged_candidates_region.sort(key=lambda t: t[0])
        chosen_tag, chosen_path = tagged_candidates_region[-1]
        print(f"Using latest SPEIx tag '{chosen_tag}' (region {region}) under {store_root}")
        return chosen_path
    if tagged_candidates_generic:
        tagged_candidates_generic.sort(key=lambda t: t[0])
        chosen_tag, chosen_path = tagged_candidates_generic[-1]
        print(f"Using latest SPEIx tag '{chosen_tag}' under {store_root}")
        return chosen_path

    raise FileNotFoundError(f"segments.zarr not found under: {store_root}")


def _discover_runs(store: Path) -> List[Path]:
    zarr_root = store / "segments.zarr" / "runs"
    if zarr_root.exists():
        return sorted([p for p in zarr_root.iterdir() if p.is_dir()])
    return []


def _segment_window_from_name(name: str, *, region: str, scale: int, pet_tag: Optional[str] = None) -> Optional[Tuple[int, int]]:
    parts = name.split("__")
    if len(parts) < 4:
        return None
    var, reg, kind, window = parts[0], parts[1], parts[2], parts[3]
    if var.lower() != f"spei{int(scale)}":
        return None
    if reg.upper() != region.upper():
        return None
    if kind != "grid-points":
        return None
    if parts[-1].lower() != "all":
        return None
    if pet_tag:
        pet_part = None
        for token in parts[4:-1]:
            if token.lower().startswith("pet-"):
                pet_part = token.lower()
                break
        if pet_part and pet_part != pet_tag.lower():
            return None
    try:
        y0, y1 = map(int, window.split("-"))
    except Exception:
        return None
    return int(y0), int(y1)


def _segment_score_from_name(name: str, *, region: str, scale: int, pet_tag: Optional[str] = None) -> Optional[Tuple[int, int]]:
    window = _segment_window_from_name(name, region=region, scale=scale, pet_tag=pet_tag)
    if window is None:
        return None
    y0, y1 = window
    return (int(y1 - y0), int(y1))


def _discover_stacked_groups(store: Path, *, region: str, scale: int, pet_method: str) -> List[Tuple[str, str]]:
    """
    Return best stacked group per forcing as tuples:
      (forcing_token, group_relpath_under_segments.zarr)
    """
    stacked_root = store / "segments.zarr" / "stacked"
    if not stacked_root.exists():
        return []
    pet_tag = _pet_tag(pet_method)
    picks: List[Tuple[str, Tuple[int, int], str]] = []
    for forcing_dir in sorted(stacked_root.iterdir()):
        if not forcing_dir.is_dir():
            continue
        pet_dir = forcing_dir / pet_tag
        if not pet_dir.exists():
            continue
        best: Optional[Tuple[Tuple[int, int], str]] = None
        for seg_dir in pet_dir.iterdir():
            if not seg_dir.is_dir():
                continue
            score = _segment_score_from_name(seg_dir.name, region=region, scale=scale, pet_tag=None)
            if score is None:
                continue
            rel = seg_dir.relative_to(store / "segments.zarr").as_posix()
            if best is None or score > best[0]:
                best = (score, rel)
        if best is not None:
            picks.append((forcing_dir.name, best[0], best[1]))
    picks.sort(key=lambda x: x[0])
    return [(forcing, rel) for forcing, _score, rel in picks]


def _discover_stacked_groups_containing_year(
    store: Path,
    *,
    region: str,
    scale: int,
    pet_method: str,
    required_year: int,
) -> List[Tuple[str, str]]:
    """
    Return stacked groups whose encoded [start, end] window contains required_year.
    Groups are sorted so longer windows (then later end years) are preferred first,
    which keeps best coverage when duplicate run IDs appear across groups.
    """
    stacked_root = store / "segments.zarr" / "stacked"
    if not stacked_root.exists():
        return []
    pet_tag = _pet_tag(pet_method)
    matches: List[Tuple[str, int, int, str]] = []
    for forcing_dir in sorted(stacked_root.iterdir()):
        if not forcing_dir.is_dir():
            continue
        pet_dir = forcing_dir / pet_tag
        if not pet_dir.exists():
            continue
        for seg_dir in pet_dir.iterdir():
            if not seg_dir.is_dir():
                continue
            window = _segment_window_from_name(seg_dir.name, region=region, scale=scale, pet_tag=None)
            if window is None:
                continue
            y0, y1 = window
            if not (int(y0) <= int(required_year) <= int(y1)):
                continue
            rel = seg_dir.relative_to(store / "segments.zarr").as_posix()
            matches.append((forcing_dir.name, int(y1 - y0), int(y1), rel))
    matches.sort(key=lambda t: (t[0], -t[1], -t[2], t[3]))
    return [(forcing, rel) for forcing, _dur, _end, rel in matches]


def _find_best_spei_group(run_dir: Path, *, region: str, scale: int, pet_method: str) -> Optional[str]:
    """
    Pick the most suitable SPEI group (prefer longest window then latest end year).
    Expected group name: spei{scale}__REGION__grid-points__YYYY-YYYY__all
    """
    pet_tag = _pet_tag(pet_method)
    search_dirs = [run_dir / pet_tag, run_dir]  # pet-specific first, then legacy layout
    candidates: List[Tuple[int, int, str]] = []
    for base in search_dirs:
        if not base.exists():
            continue
        for child in base.iterdir():
            if not child.is_dir():
                continue
            parts = child.name.split("__")
            if len(parts) < 4:
                continue
            var, reg, kind, window = parts[0], parts[1], parts[2], parts[3]
            if var.lower() != f"spei{scale}":
                continue
            if reg.upper() != region.upper():
                continue
            if kind != "grid-points":
                continue
            try:
                y0, y1 = map(int, window.split("-"))
            except Exception:
                continue
            # Last token must be "all" (position varies with pet tag)
            if parts[-1].lower() != "all":
                continue
            # If a pet tag is included in the group name, ensure it matches the requested method
            pet_part = None
            for token in parts[4:-1]:
                if token.lower().startswith("pet-"):
                    pet_part = token.lower()
                    break
            if pet_part and pet_part != pet_tag.lower():
                continue
            rel_name = child.relative_to(run_dir).as_posix()
            candidates.append((y1 - y0, y1, rel_name))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][2]


def _available_scales_for_run(run_dir: Path, *, region: str, pet_method: str) -> List[int]:
    """
    Discover available SPEI scales for a run for the requested region/pet layout.
    """
    pet_tag = _pet_tag(pet_method).lower()
    search_dirs = [run_dir / pet_tag, run_dir]
    scales: set[int] = set()
    for base in search_dirs:
        if not base.exists():
            continue
        for child in base.iterdir():
            if not child.is_dir():
                continue
            parts = child.name.split("__")
            if len(parts) < 4:
                continue
            var, reg, kind = parts[0], parts[1], parts[2]
            if reg.upper() != region.upper():
                continue
            if kind != "grid-points":
                continue
            if parts[-1].lower() != "all":
                continue
            pet_part = None
            for token in parts[4:-1]:
                if token.lower().startswith("pet-"):
                    pet_part = token.lower()
                    break
            if pet_part and pet_part != pet_tag:
                continue
            m = re.match(r"^spei(\d+)$", var.strip().lower())
            if m:
                scales.add(int(m.group(1)))
    return sorted(scales)


def _load_spei_store(
    store_root: Path,
    *,
    region: str,
    scale: int,
    limit_ensembles: Optional[int],
    scenario_tag: Optional[str] = None,
    pet_method: str,
    store_tag: Optional[str] = None,
    apply_landmask_ipcc_ar6_regions: bool = APPLY_LANDMASK_IPCCAR6REGIONS,
    stacked_required_year: Optional[int] = None,
) -> Dict[str, SPEISeries]:
    """
    Load SPEI{scale} series for all runs in a 754-produced segment store.
    """
    _require_xarray()
    resolved_root = _resolve_store_root(store_root, tag=store_tag, region=region, pet_method=pet_method)
    store = resolved_root / "segments.zarr"
    if not store.exists():
        raise FileNotFoundError(f"segments.zarr not found under: {resolved_root}")

    baseline_cache: Dict[str, Dict[str, Optional[str]]] = {}

    # 1) Preferred path: run-stacked layout.
    if stacked_required_year is not None:
        stacked_groups = _discover_stacked_groups_containing_year(
            resolved_root,
            region=region,
            scale=scale,
            pet_method=pet_method,
            required_year=int(stacked_required_year),
        )
        if not stacked_groups:
            print(
                f"[WARN] No stacked groups under {resolved_root} matched required year "
                f"{int(stacked_required_year)} (region={region}, pet={pet_method}, scale={scale}). "
                "Falling back to default stacked-group selection."
            )
            stacked_groups = _discover_stacked_groups(resolved_root, region=region, scale=scale, pet_method=pet_method)
    else:
        stacked_groups = _discover_stacked_groups(resolved_root, region=region, scale=scale, pet_method=pet_method)
    if stacked_groups:
        out_stacked: Dict[str, SPEISeries] = {}
        open_errors_stacked: List[Tuple[str, str]] = []
        max_runs = int(limit_ensembles) if limit_ensembles is not None else None

        for forcing_token, group_path in stacked_groups:
            ds: Optional[xr.Dataset] = None
            try:
                ds = xr.open_zarr(store, group=group_path, consolidated=False)
            except Exception as exc:
                open_errors_stacked.append((group_path, f"{type(exc).__name__}: {str(exc).splitlines()[0]}"))
                continue

            try:
                var_candidates = [f"spei{scale}", "spei", "SPEI", f"SPEI{scale}"]
                var_name = next((v for v in var_candidates if v in ds), None)
                if var_name is None:
                    continue
                if "run" not in ds.coords:
                    continue

                da = ds[var_name]
                all_run_names = [str(v) for v in np.asarray(ds["run"].values).ravel().tolist()]
                run_pairs = list(enumerate(all_run_names))
                if scenario_tag:
                    run_pairs = [(idx, rn) for idx, rn in run_pairs if _run_matches_token(rn, scenario_tag)]

                for run_idx, run_name in run_pairs:
                    if max_runs is not None and len(out_stacked) >= max_runs:
                        break
                    if run_name in out_stacked:
                        continue
                    try:
                        da_run = da.isel(run=int(run_idx))
                    except Exception:
                        continue

                    meta = _extract_stacked_baseline_meta(ds, run_index=int(run_idx), cache=baseline_cache)
                    time_frac, time_years, time_months = _time_arrays(da_run["time"])
                    lat, lon = _extract_lat_lon(da_run)
                    values = np.asarray(da_run.values, dtype=np.float32)
                    values = _apply_landmask_to_point_values(
                        values,
                        lat=lat,
                        lon=lon,
                        region=region,
                        apply_landmask_ipcc_ar6_regions=apply_landmask_ipcc_ar6_regions,
                    )
                    out_stacked[run_name] = SPEISeries(
                        label=run_name,
                        source=f"{resolved_root.name}:{forcing_token}",
                        time=time_frac,
                        years=time_years,
                        months=time_months,
                        values=values,
                        lat=lat,
                        lon=lon,
                        pet_method=meta.get("pet_method"),
                        baseline_source=meta.get("baseline_source"),
                        baseline_pooling=meta.get("baseline_pooling"),
                        baseline_strategy=meta.get("baseline_strategy"),
                        baseline_start_year=meta.get("baseline_start_year"),
                        baseline_end_year=meta.get("baseline_end_year"),
                        baseline_fit_file=meta.get("baseline_fit_file"),
                    )
                if max_runs is not None and len(out_stacked) >= max_runs:
                    break
            finally:
                if ds is not None:
                    ds.close()

        if out_stacked:
            return out_stacked
        if open_errors_stacked:
            first_group, first_err = open_errors_stacked[0]
            print(
                f"[WARN] Stacked SPEIx groups found under {resolved_root} but could not be read "
                f"(first error: group={first_group}, error={first_err}). Falling back to legacy runs layout."
            )
        else:
            print(
                f"[WARN] Stacked SPEIx groups found under {resolved_root} but no runs matched "
                f"scenario='{scenario_tag}' for region={region}, pet={pet_method}, scale={scale}. "
                "Falling back to legacy runs layout."
            )

    # 2) Backward-compatible fallback: per-run layout.
    runs_root = store / "runs"
    if not runs_root.exists():
        raise FileNotFoundError(f"segments.zarr not found under: {resolved_root}")

    runs = _discover_runs(resolved_root)
    if scenario_tag:
        runs = [run for run in runs if _run_matches_token(run.name, scenario_tag)]
    if not runs:
        available = [run.name for run in _discover_runs(resolved_root)]
        sample = ", ".join(available[:6]) + (" ..." if len(available) > 6 else "")
        raise RuntimeError(
            f"No runs matched scenario token '{scenario_tag}' in {resolved_root}/segments.zarr. "
            f"Available runs: {sample or 'none'}. "
            "Pass a different --scenario1-tag/--scenario2-tag or --scenario2-suffix to match the stored runs."
        )
    if limit_ensembles is not None and len(runs) > limit_ensembles:
        runs = runs[:limit_ensembles]

    out: Dict[str, SPEISeries] = {}
    available_scales: set[int] = set()
    open_errors: List[Tuple[str, str, str]] = []
    for run_dir in runs:
        run_name = run_dir.name
        best_group = _find_best_spei_group(run_dir, region=region, scale=scale, pet_method=pet_method)
        if best_group is None:
            available_scales.update(_available_scales_for_run(run_dir, region=region, pet_method=pet_method))
            continue
        group_path = f"runs/{run_name}/{best_group}"
        ds: Optional[xr.Dataset] = None
        try:
            ds = xr.open_zarr(store, group=group_path, consolidated=False)
        except Exception as exc:
            open_errors.append((run_name, group_path, f"{type(exc).__name__}: {str(exc).splitlines()[0]}"))
            continue
        try:
            var_candidates = [f"spei{scale}", "spei", "SPEI", f"SPEI{scale}"]
            da = None
            for v in var_candidates:
                if v in ds:
                    da = ds[v]
                    break
            if da is None:
                continue
            meta = _extract_baseline_meta(ds, baseline_cache)
            time_frac, time_years, time_months = _time_arrays(da["time"])
            lat, lon = _extract_lat_lon(da)
            values = np.asarray(da.values, dtype=np.float32)
            values = _apply_landmask_to_point_values(
                values,
                lat=lat,
                lon=lon,
                region=region,
                apply_landmask_ipcc_ar6_regions=apply_landmask_ipcc_ar6_regions,
            )
            out[run_name] = SPEISeries(
                label=run_name,
                source=resolved_root.name,
                time=time_frac,
                years=time_years,
                months=time_months,
                values=values,
                lat=lat,
                lon=lon,
                pet_method=meta.get("pet_method"),
                baseline_source=meta.get("baseline_source"),
                baseline_pooling=meta.get("baseline_pooling"),
                baseline_strategy=meta.get("baseline_strategy"),
                baseline_start_year=meta.get("baseline_start_year"),
                baseline_end_year=meta.get("baseline_end_year"),
                baseline_fit_file=meta.get("baseline_fit_file"),
            )
        finally:
            if ds is not None:
                ds.close()
    if not out:
        available_scales_txt = ", ".join(str(v) for v in sorted(available_scales)) if available_scales else "none"
        hint = ""
        if available_scales and scale not in available_scales:
            hint = f" Possible scale mismatch: requested --scale={scale}, available scales: {available_scales_txt}."
        elif open_errors:
            run_name, group_path, err = open_errors[0]
            hint = f" First read error: run={run_name}, group={group_path}, error={err}"
        raise RuntimeError(
            f"No SPEI groups could be loaded from {resolved_root}/segments.zarr "
            f"for region={region}, pet={pet_method}, scale={scale}.{hint}"
        )
    return out


# -----------------------------------------------------------------------------
# Plotting helpers (from your original 758, trimmed)
# -----------------------------------------------------------------------------
def _plot_maps(
    axs: List[plt.Axes],
    series_list: List[SPEISeries],
    labels: List[str],
    *,
    start_year: int,
    end_year: int,
    region: Optional[str] = None,
    scale: int = DEFAULT_SCALE,
) -> None:
    im = None

    def _wrap_lon(lon: np.ndarray) -> np.ndarray:
        """Wrap longitudes to (-180, 180]."""
        lon = np.asarray(lon, dtype=float)
        return ((lon + 180.0) % 360.0) - 180.0

    def _lon_extent(lon: np.ndarray, pad: float = 2.0) -> Tuple[float, float]:
        """
        Minimal circular span covering the points, with padding.
        Removes the largest gap on the circle, then unwraps to a contiguous interval.
        """
        lon = _wrap_lon(lon)
        if lon.size == 0:
            return -180.0, 180.0
        lon_sorted = np.sort(lon)
        diffs = np.diff(np.concatenate([lon_sorted, lon_sorted[:1] + 360.0]))
        gap_idx = int(np.argmax(diffs))
        start = lon_sorted[(gap_idx + 1) % lon_sorted.size]
        span = 360.0 - diffs[gap_idx]
        lon_min = start - pad
        lon_max = start + span + pad
        while lon_min > 180:
            lon_min -= 360
            lon_max -= 360
        while lon_max <= -180:
            lon_min += 360
            lon_max += 360
        return float(lon_min), float(lon_max)

    def _grid_from_points(lon: np.ndarray, lat: np.ndarray, values: np.ndarray):
        """Rebuild a 2D grid and track which cells were explicitly provided."""
        lon = _wrap_lon(np.asarray(lon, dtype=float))
        lat = np.asarray(lat, dtype=float)
        values = np.asarray(values, dtype=float)
        lon_u = np.unique(lon)
        lat_u = np.unique(lat)
        grid = np.full((lat_u.size, lon_u.size), np.nan, dtype=float)
        present = np.zeros((lat_u.size, lon_u.size), dtype=bool)
        lon_to_idx = {v: i for i, v in enumerate(lon_u)}
        lat_to_idx = {v: i for i, v in enumerate(lat_u)}
        for x, y, v in zip(lon, lat, values):
            yi = lat_to_idx[y]
            xi = lon_to_idx[x]
            grid[yi, xi] = v
            present[yi, xi] = True

        def _edges(arr: np.ndarray) -> np.ndarray:
            arr = np.sort(arr)
            if arr.size == 1:
                step = 0.5
                return np.array([arr[0] - step, arr[0] + step])
            mid = (arr[1:] + arr[:-1]) / 2.0
            first = arr[0] - (arr[1] - arr[0]) / 2.0
            last = arr[-1] + (arr[-1] - arr[-2]) / 2.0
            return np.concatenate([[first], mid, [last]])

        lon_edges = _edges(lon_u)
        lat_edges = _edges(lat_u)
        return lon_edges, lat_edges, grid, present

    for ax, series, label in zip(axs, series_list, labels):
        years = series.years
        mask = (years >= start_year) & (years <= end_year)
        vals = series.values[mask]
        if vals.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            continue
        mean_map = np.nanmean(vals, axis=0)

        if series.lat is not None and series.lon is not None and len(series.lat) == mean_map.size == len(series.lon):
            transform = ccrs.PlateCarree() if ccrs is not None and hasattr(ax, "projection") else None
            grid_res = _grid_from_points(series.lon, series.lat, mean_map)
            if grid_res is not None:
                lon_edges, lat_edges, grid, present = grid_res
                data_layer = np.ma.array(grid, mask=(~present) | (~np.isfinite(grid)))
                mesh_kwargs = {
                    "cmap": SPEI_CMAP,
                    "vmin": -3,
                    "vmax": 3,
                    "alpha": 0.7,
                    "shading": "auto",
                    "edgecolors": "#f5f5f5",
                    "linewidth": 0.2,
                    "antialiased": False,
                }
                if transform is not None:
                    mesh_kwargs["transform"] = transform
                sc = ax.pcolormesh(lon_edges, lat_edges, data_layer, **mesh_kwargs)
                im = sc
                nan_mask = (~present) | np.isfinite(grid)
                if np.any(~nan_mask):
                    nan_layer = np.ma.array(np.ones_like(grid, dtype=float), mask=nan_mask)
                    nan_kwargs = {
                        "cmap": ListedColormap([ERA5DROUGHT_NAN_COLOR]),
                        "vmin": 0.0,
                        "vmax": 1.0,
                        "alpha": 0.95,
                        "shading": "auto",
                        "edgecolors": "none",
                        "linewidth": 0.0,
                        "antialiased": False,
                    }
                    if transform is not None:
                        nan_kwargs["transform"] = transform
                    ax.pcolormesh(lon_edges, lat_edges, nan_layer, **nan_kwargs)
                lon_min, lon_max = _lon_extent(series.lon)
                lat_min, lat_max = float(lat_edges.min()) - 2.0, float(lat_edges.max()) + 2.0
                ax.set_aspect("equal", adjustable="box")
                if transform is not None:
                    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=transform)
                else:
                    ax.set_xlim(lon_min, lon_max)
                    ax.set_ylim(lat_min, lat_max)
            else:
                finite = np.isfinite(mean_map)
                scatter_kwargs = {
                    "c": mean_map[finite],
                    "cmap": SPEI_CMAP,
                    "vmin": -3,
                    "vmax": 3,
                    "s": 80,
                    "marker": "s",
                    "linewidths": 0,
                    "alpha": 0.7,
                }
                if transform is not None:
                    scatter_kwargs["transform"] = transform
                lon_wrapped = _wrap_lon(series.lon)
                if np.any(finite):
                    sc = ax.scatter(lon_wrapped[finite], series.lat[finite], **scatter_kwargs)
                    im = sc
                nan_points = ~finite
                if np.any(nan_points):
                    nan_kwargs = {
                        "s": 80,
                        "marker": "s",
                        "linewidths": 0,
                        "alpha": 0.95,
                        "color": ERA5DROUGHT_NAN_COLOR,
                    }
                    if transform is not None:
                        nan_kwargs["transform"] = transform
                    ax.scatter(lon_wrapped[nan_points], series.lat[nan_points], **nan_kwargs)
                lat_min, lat_max = float(np.nanmin(series.lat)), float(np.nanmax(series.lat))
                lon_min_raw, lon_max_raw = _lon_extent(series.lon)
                lat_pad = 2.0
                if transform is not None:
                    ax.set_extent([lon_min_raw, lon_max_raw, lat_min - lat_pad, lat_max + lat_pad], crs=transform)
                else:
                    ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)
                    ax.set_xlim(lon_min_raw, lon_max_raw)
                    ax.set_aspect("equal", adjustable="box")
            ax.set_title(label, fontsize=10, loc="left")
            ax.grid(alpha=0.2, linewidth=0.5)

            def _deg_fmt(val, _pos):
                return f"{val:.0f}°"

            if FuncFormatter is not None:
                ax.xaxis.set_major_formatter(FuncFormatter(_deg_fmt))
                ax.yaxis.set_major_formatter(FuncFormatter(_deg_fmt))
            ax.tick_params(axis="both", labelsize=8)
            ax.set_xlabel("Lon (°)", fontsize=8)
            ax.set_ylabel("Lat (°)", fontsize=8)

            if region and _is_ipcc_ar6_region(region) and regionmask is not None:
                try:
                    ar6 = regionmask.defined_regions.ar6.all
                    rid = ar6.map_keys(region.upper())
                    poly = ar6.polygons[rid]
                    polys = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
                    for p in polys:
                        if getattr(p, "is_empty", False):
                            continue
                        x, y = p.exterior.xy
                        line_kwargs = {
                            "color": "#8a8a8a",
                            "linewidth": 0.7,
                            "linestyle": "--",
                            "alpha": 0.95,
                            "zorder": 4,
                        }
                        if transform is not None:
                            ax.plot(x, y, transform=transform, **line_kwargs)
                        else:
                            ax.plot(x, y, **line_kwargs)
                except Exception:
                    pass

            if cfeature is not None and ccrs and hasattr(ax, "projection"):
                # Light ocean background without heavy polygon fills: color the axes face, then paint land on top.
                ax.set_facecolor("#e8f7ff")
                ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=0.1)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="#333", zorder=1)
                ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="#333", zorder=1)

            im = sc
        else:
            im = ax.imshow(mean_map.reshape(-1, 1).T, aspect="auto", cmap=SPEI_CMAP, vmin=-3, vmax=3)
            ax.set_title(label, fontsize=10, loc="left")
            ax.set_yticks([])
            ax.set_xticks([])

    if axs:
        from matplotlib.colors import Normalize
        mappable = im if im is not None else plt.cm.ScalarMappable(norm=Normalize(-3, 3), cmap=SPEI_CMAP)
        cbar = plt.colorbar(mappable, ax=axs, fraction=0.035, pad=0.04, orientation="vertical")
        cbar.set_label(f"SPEI{scale}")
        # Wet/dry markers aligned with bar ends and tick column
        cbar.ax.text(1.02, 1.0, "wet", transform=cbar.ax.transAxes, ha="left", va="center", fontsize=8)
        cbar.ax.text(1.02, 0.0, "dry", transform=cbar.ax.transAxes, ha="left", va="center", fontsize=8)
        try:
            cbar.set_ticks([-2.33, -1.65, -1.28, -0.84, 0.0, 0.84, 1.28, 1.65, 2.33])
        except Exception:
            pass
        cbar.ax.tick_params(labelsize=8)


def _plot_timeseries(
    ax: plt.Axes,
    series: SPEISeries,
    color: str,
    *,
    mean_color: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    values = series.values
    if values.ndim != 2:
        return
    n_traces = min(values.shape[1], MAX_GRID_TRACES)
    step = max(1, values.shape[1] // n_traces)
    for col in range(0, values.shape[1], step):
        # Very light, thin traces for individual grid points to reduce clutter
        ax.plot(series.time, values[:, col], color=color, alpha=0.10, linewidth=0.05)
    # Median (faint) and mean (primary) overlays (stronger stroke for median)
    mc = mean_color or color
    ax.plot(series.time, np.nanmedian(values, axis=1), color=mc, alpha=0.3, linewidth=3.0, label=f"{series.label} median")
    ax.plot(series.time, np.nanmean(values, axis=1), color=mc, alpha=0.5, linewidth=1.4, label=f"{series.label} mean")
    ax.axhline(0.0, color="#999", linewidth=0.8, linestyle="--")
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    # Use numeric year axis to avoid pandas datetime limitations
    try:
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(round(x))}"))
    except Exception:
        pass
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.2, linewidth=0.5)


def _overlay_cmip6_individual_lines(
    ax: plt.Axes,
    series_list: List[SPEISeries],
    *,
    color: str,
    label: str,
    linewidth: float = 0.9,
    linestyle: str = "--",
    alpha: float = 0.55,
    zorder: float = 2.5,
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    """Overlay one region-mean line per CMIP6 model run."""
    if not series_list:
        return
    first = True
    for s in series_list:
        vals = _region_mean(s.values)
        if vals.size == 0:
            continue
        ax.plot(
            s.time,
            vals,
            color=color,
            linewidth=float(linewidth),
            linestyle=str(linestyle),
            alpha=float(alpha),
            label=(label if first else None),
            zorder=float(zorder),
        )
        first = False
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])


def _add_scenario_panel_legend(ax: plt.Axes, *, gcmagicc_color: str) -> None:
    if Line2D is None:
        return
    handles = [
        Line2D([0], [0], color=gcmagicc_color, lw=1.2, alpha=0.8, label="GCMAGICC"),
        Line2D([0], [0], color=ROW_MEAN_COLORS["era5"], lw=1.6, alpha=0.9, label="ERA5"),
        Line2D([0], [0], color=CMIP6_PANEL_COLOR, lw=0.8, alpha=0.6, label="CMIP6"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        ncol=3,
        fontsize=7,
        frameon=False,
        handlelength=1.8,
        columnspacing=0.9,
        borderaxespad=0.2,
    )


def _region_mean(values: np.ndarray) -> np.ndarray:
    return values if values.ndim == 1 else np.nanmean(values, axis=1)

def _annual_mean_for_year(series: SPEISeries, year: int) -> Optional[np.ndarray]:
    years = series.years
    mask = years == year
    if not np.any(mask):
        return None
    vals = series.values[mask]
    return np.nanmean(vals, axis=0)

def _max_annual_mean_over_years(series_list: List[SPEISeries], start_year: int, end_year: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (max_field, lat, lon) over annual means across members/years in window."""
    fields = []
    lat = lon = None
    for s in series_list:
        if s.lat is not None and s.lon is not None and lat is None and lon is None:
            lat, lon = s.lat, s.lon
        for y in range(start_year, end_year + 1):
            m = _annual_mean_for_year(s, y)
            if m is not None:
                fields.append(m)
    if not fields:
        return None, lat, lon
    stacked = np.vstack(fields)
    return np.nanmax(stacked, axis=0), lat, lon

def _min_annual_mean_over_years(series_list: List[SPEISeries], start_year: int, end_year: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (min_field, lat, lon) over annual means across members/years in window."""
    fields = []
    lat = lon = None
    for s in series_list:
        if s.lat is not None and s.lon is not None and lat is None and lon is None:
            lat, lon = s.lat, s.lon
        for y in range(start_year, end_year + 1):
            m = _annual_mean_for_year(s, y)
            if m is not None:
                fields.append(m)
    if not fields:
        return None, lat, lon
    stacked = np.vstack(fields)
    return np.nanmin(stacked, axis=0), lat, lon


def _mean_annual_mean_over_years(series_list: List[SPEISeries], start_year: int, end_year: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (mean_field, lat, lon) averaged over annual means across members/years."""
    fields = []
    lat = lon = None
    for s in series_list:
        if s.lat is not None and s.lon is not None and lat is None and lon is None:
            lat, lon = s.lat, s.lon
        for y in range(start_year, end_year + 1):
            m = _annual_mean_for_year(s, y)
            if m is not None:
                fields.append(m)
    if not fields:
        return None, lat, lon
    stacked = np.vstack(fields)
    return np.nanmean(stacked, axis=0), lat, lon

def _wrap_static_series(template: SPEISeries, values: np.ndarray, label: str, time_year: int) -> SPEISeries:
    """Create a single-time SPEISeries carrying metadata/lat/lon from template."""
    time = np.array([float(time_year) + 0.5])
    years = np.array([time_year], dtype=int)
    months = np.array([7], dtype=int)
    vals = values.reshape(1, -1)
    return SPEISeries(
        label=label,
        source=template.source,
        time=time,
        years=years,
        months=months,
        values=vals,
        lat=template.lat,
        lon=template.lon,
        pet_method=template.pet_method,
        baseline_source=template.baseline_source,
        baseline_pooling=template.baseline_pooling,
        baseline_strategy=template.baseline_strategy,
        baseline_start_year=template.baseline_start_year,
        baseline_end_year=template.baseline_end_year,
        baseline_fit_file=template.baseline_fit_file,
    )


def _array_to_json_safe(arr: Optional[np.ndarray]) -> Optional[List]:
    """Convert ndarray to list with NaN/inf replaced by None so JSON stays valid."""
    if arr is None:
        return None
    a = np.asarray(arr, dtype=object)
    if np.issubdtype(a.dtype, np.floating):
        mask = ~np.isfinite(a.astype(float, copy=False))
        a[mask] = None
    return a.tolist()


def _array_from_json_safe(data: Optional[Sequence]) -> Optional[np.ndarray]:
    """Convert JSON list back to numpy array, restoring NaN for None."""
    if data is None:
        return None
    arr = np.array(data, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr


def _array_or_empty(arr: Optional[np.ndarray], *, dtype=None) -> np.ndarray:
    """Return an ndarray, defaulting to an empty array without truth-testing numpy values."""
    if arr is None:
        if dtype is None:
            return np.array([])
        return np.array([], dtype=dtype)
    if dtype is None:
        return np.asarray(arr)
    return np.asarray(arr, dtype=dtype)


def _series_to_dict(series: SPEISeries) -> Dict:
    """Serialize SPEISeries to JSON-friendly dict."""
    def _round(obj):
        if obj is None:
            return None
        arr = np.asarray(obj, dtype=float)
        if arr.ndim == 0:
            return float(f"{arr.item():.5g}") if np.isfinite(arr) else None
        flat = arr.reshape(-1)
        rounded = []
        for x in flat:
            if not np.isfinite(x):
                rounded.append(None)
            else:
                rounded.append(float(f"{x:.5g}"))
        return np.array(rounded, dtype=object).reshape(arr.shape).tolist()

    return {
        "label": series.label,
        "source": series.source,
        "time": _round(series.time),
        "years": _round(series.years),
        "months": _round(series.months),
        "values": _round(series.values),
        "lat": _round(series.lat),
        "lon": _round(series.lon),
        "pet_method": series.pet_method,
        "baseline_source": series.baseline_source,
        "baseline_pooling": series.baseline_pooling,
        "baseline_strategy": series.baseline_strategy,
        "baseline_start_year": int(series.baseline_start_year) if series.baseline_start_year is not None else None,
        "baseline_end_year": int(series.baseline_end_year) if series.baseline_end_year is not None else None,
        "baseline_fit_file": series.baseline_fit_file,
    }


def _series_from_dict(data: Dict) -> SPEISeries:
    """Deserialize SPEISeries from JSON dict."""
    time = _array_from_json_safe(data.get("time"))
    years = _array_from_json_safe(data.get("years"))
    months = _array_from_json_safe(data.get("months"))
    values = _array_from_json_safe(data.get("values"))
    return SPEISeries(
        label=data["label"],
        source=data.get("source", ""),
        time=_array_or_empty(time),
        years=_array_or_empty(years, dtype=int),
        months=_array_or_empty(months, dtype=int),
        values=_array_or_empty(values),
        lat=_array_from_json_safe(data.get("lat")),
        lon=_array_from_json_safe(data.get("lon")),
        pet_method=data.get("pet_method"),
        baseline_source=data.get("baseline_source"),
        baseline_pooling=data.get("baseline_pooling"),
        baseline_strategy=data.get("baseline_strategy"),
        baseline_start_year=data.get("baseline_start_year"),
        baseline_end_year=data.get("baseline_end_year"),
        baseline_fit_file=data.get("baseline_fit_file"),
    )

def _sanitize_json(obj: Any) -> Any:
    """Recursively convert numpy/Python objects to JSON-serializable primitives."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return _sanitize_json(_array_to_json_safe(obj))
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _round_sig(arr: np.ndarray, sig: int = 5) -> np.ndarray:
    """Round to significant digits; preserves NaNs."""
    a = np.asarray(arr, dtype=float)
    flat = a.reshape(-1)
    out = []
    for x in flat:
        if not np.isfinite(x):
            out.append(x)
        else:
            out.append(float(f"{x:.{sig}g}"))
    return np.asarray(out, dtype=float).reshape(a.shape)


def _limit_series_columns(series: SPEISeries, max_cols: int) -> SPEISeries:
    """Subsample columns to at most max_cols (evenly), preserving metadata."""
    vals = series.values
    if vals.ndim != 2 or vals.shape[1] <= max_cols:
        return series
    step = max(1, int(np.ceil(vals.shape[1] / max_cols)))
    cols = list(range(0, vals.shape[1], step))[:max_cols]
    vals_new = vals[:, cols]
    lat_new = series.lat[cols] if series.lat is not None and len(series.lat) == vals.shape[1] else series.lat
    lon_new = series.lon[cols] if series.lon is not None and len(series.lon) == vals.shape[1] else series.lon
    return SPEISeries(
        label=series.label,
        source=series.source,
        time=series.time,
        years=series.years,
        months=series.months,
        values=vals_new,
        lat=lat_new,
        lon=lon_new,
        pet_method=series.pet_method,
        baseline_source=series.baseline_source,
        baseline_pooling=series.baseline_pooling,
        baseline_strategy=series.baseline_strategy,
        baseline_start_year=series.baseline_start_year,
        baseline_end_year=series.baseline_end_year,
        baseline_fit_file=series.baseline_fit_file,
    )


def _median_series(series: SPEISeries) -> SPEISeries:
    """Return a single-trace series with the median across columns (if 2D)."""
    vals = series.values
    if vals.ndim == 2:
        median_vals = np.nanmedian(vals, axis=1, keepdims=True)
    else:
        median_vals = vals
    return SPEISeries(
        label=f"{series.label} median",
        source=series.source,
        time=series.time,
        years=series.years,
        months=series.months,
        values=median_vals,
        lat=None,
        lon=None,
        pet_method=series.pet_method,
        baseline_source=series.baseline_source,
        baseline_pooling=series.baseline_pooling,
        baseline_strategy=series.baseline_strategy,
        baseline_start_year=series.baseline_start_year,
        baseline_end_year=series.baseline_end_year,
        baseline_fit_file=series.baseline_fit_file,
    )


def _annualize_series(series: SPEISeries) -> SPEISeries:
    """Aggregate to annual means for each grid column."""
    years = series.years
    if years.size == 0:
        return series
    uniq = np.unique(years)
    vals = series.values
    annual_vals = []
    for y in uniq:
        mask = years == y
        if mask.any():
            annual_vals.append(np.nanmean(vals[mask], axis=0))
    annual_arr = np.vstack(annual_vals)
    time = uniq.astype(float) + 0.5
    months = np.full_like(uniq, 7, dtype=int)
    return SPEISeries(
        label=series.label,
        source=series.source,
        time=time,
        years=uniq,
        months=months,
        values=annual_arr,
        lat=series.lat,
        lon=series.lon,
        pet_method=series.pet_method,
        baseline_source=series.baseline_source,
        baseline_pooling=series.baseline_pooling,
        baseline_strategy=series.baseline_strategy,
        baseline_start_year=series.baseline_start_year,
        baseline_end_year=series.baseline_end_year,
        baseline_fit_file=series.baseline_fit_file,
    )


def _year_month_arrays(time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # time already stored as fractional years; recover integer year/month using stored arrays instead.
    raise RuntimeError("Use series.years/series.months directly instead of _year_month_arrays")


def _collect_monthly_values(series: SPEISeries, start_year: int, end_year: int) -> Dict[int, List[float]]:
    years, months = series.years, series.months
    vals = _region_mean(series.values)
    mask = (years >= start_year) & (years <= end_year) & np.isfinite(vals)
    month_vals: Dict[int, List[float]] = {m: [] for m in range(1, 13)}
    for v, m in zip(vals[mask], months[mask]):
        month_vals[int(m)].append(float(v))
    return month_vals


def _aggregate_ensemble_months(series_list: List[SPEISeries], start_year: int, end_year: int) -> Dict[int, List[float]]:
    month_vals: Dict[int, List[float]] = {m: [] for m in range(1, 13)}
    for s in series_list:
        mv = _collect_monthly_values(s, start_year, end_year)
        for m, lst in mv.items():
            month_vals[m].extend(lst)
    return month_vals


def _percentile_leq(value: float, samples: List[float]) -> float:
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Empty distribution for percentile computation.")
    return float(np.sum(arr <= value) / arr.size)


def _probabilities_from_ref(
    ref_by_month: Dict[int, float],
    dist_by_month: Dict[int, List[float]],
    *,
    scenario_name: str,
    window: Tuple[int, int],
) -> List[float]:
    probs = []
    for m in range(1, 13):
        if m not in ref_by_month or ref_by_month[m] is None:
            raise ValueError(f"Missing reference value for month {m}")
        samples = dist_by_month.get(m, [])
        if not samples:
            raise RuntimeError(
                f"No samples for month {m} in {scenario_name} during {window[0]}-{window[1]}; "
                "cannot compute percentile. Check that ensemble data cover this window."
            )
        probs.append(_percentile_leq(ref_by_month[m], samples))
    return probs


def _prepare_probabilities(
    era5_series: SPEISeries,
    all_list: List[SPEISeries],
    nat_list: List[SPEISeries],
    *,
    ref_start: int,
    ref_end: int,
    fut_start: int,
    fut_end: int,
    rng: np.random.Generator,
    scenario2_label: str,
) -> Dict[str, Dict[str, List[float]]]:
    def _scenario_by_year_month(series_list: List[SPEISeries], start: int, end: int) -> Dict[Tuple[int, int], List[float]]:
        out: Dict[Tuple[int, int], List[float]] = {}
        for s in series_list:
            years, months = s.years, s.months
            vals = _region_mean(s.values)
            mask = (years >= start) & (years <= end) & np.isfinite(vals)
            for v, y, m in zip(vals[mask], years[mask], months[mask]):
                out.setdefault((int(y), int(m)), []).append(float(v))
        return out

    def _scenario_by_month(series_list: List[SPEISeries], start: int, end: int) -> Dict[int, List[float]]:
        out: Dict[int, List[float]] = {m: [] for m in range(1, 13)}
        for s in series_list:
            years, months = s.years, s.months
            vals = _region_mean(s.values)
            mask = (years >= start) & (years <= end) & np.isfinite(vals)
            for v, m in zip(vals[mask], months[mask]):
                out[int(m)].append(float(v))
        return out

    if not all_list:
        raise RuntimeError("Scenario 1 ensemble list is empty.")
    if not nat_list:
        raise RuntimeError("Scenario 2 ensemble list is empty.")

    # ERA5 month-wise values (per actual month-year)
    years_era, months_era = era5_series.years, era5_series.months
    era_vals_all = _region_mean(era5_series.values)
    era_mask = (years_era >= ref_start) & (years_era <= ref_end) & np.isfinite(era_vals_all)
    era_vals_ref = era_vals_all[era_mask]
    era_years_ref = years_era[era_mask].astype(int)
    era_months_ref = months_era[era_mask].astype(int)
    if era_vals_ref.size == 0:
        raise ValueError("No ERA5 data found in reference window.")

    # Scenario distributions for current (year-month specific) and future (month specific)
    dist_curr_all = _scenario_by_year_month(all_list, ref_start, ref_end)
    dist_curr_nat = _scenario_by_year_month(nat_list, ref_start, ref_end)
    dist_fut_all = _scenario_by_month(all_list, fut_start, fut_end)
    dist_fut_nat = _scenario_by_month(nat_list, fut_start, fut_end)

    def _probs_for_window(dist_all, dist_nat, *, by_month_only: bool) -> Tuple[List[float], List[float]]:
        p_all: List[float] = []
        p_nat: List[float] = []
        for v, y, m in zip(era_vals_ref, era_years_ref, era_months_ref):
            key_all = (m if by_month_only else (y, m))
            key_nat = key_all
            dist_a = dist_all.get(key_all, [])
            dist_n = dist_nat.get(key_nat, [])
            if not dist_a or not dist_n:
                raise RuntimeError(f"No scenario samples for ERA5 month {y}-{m:02d}.")
            p_all.append(_percentile_leq(float(v), dist_a))
            p_nat.append(_percentile_leq(float(v), dist_n))
        return p_all, p_nat

    probs_curr_all, probs_curr_nat = _probs_for_window(dist_curr_all, dist_curr_nat, by_month_only=False)
    probs_fut_all, probs_fut_nat = _probs_for_window(dist_fut_all, dist_fut_nat, by_month_only=True)

    return {
        "current": {"all": probs_curr_all, "nat": probs_curr_nat},
        "future": {"all": probs_fut_all, "nat": probs_fut_nat},
    }


def _compute_probability_ratios(p_all: Sequence[float], p_nat: Sequence[float]) -> np.ndarray:
    """Compute probability ratios with masking to avoid division by zero."""
    a = np.asarray(p_all, dtype=float)
    b = np.asarray(p_nat, dtype=float)
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    mask = np.isfinite(a) & np.isfinite(b) & (b > 0)
    if not mask.any():
        return np.asarray([], dtype=float)
    return np.divide(a[mask], b[mask])


def _compute_probability_products(
    era5_series: SPEISeries,
    all_list: List[SPEISeries],
    nat_list: List[SPEISeries],
    *,
    ref_start: int,
    ref_end: int,
    fut_start: int,
    fut_end: int,
    scenario2_label: str,
) -> Dict[str, Any]:
    """Return probabilities and ratios needed for histogram panels."""
    rng = np.random.default_rng(0)
    probs = _prepare_probabilities(
        era5_series,
        all_list,
        nat_list,
        ref_start=ref_start,
        ref_end=ref_end,
        fut_start=fut_start,
        fut_end=fut_end,
        rng=rng,
        scenario2_label=scenario2_label,
    )
    ratios = {
        "current": _compute_probability_ratios(probs["current"]["all"], probs["current"]["nat"]),
        "future": _compute_probability_ratios(probs["future"]["all"], probs["future"]["nat"]),
    }
    return {"probs": probs, "ratios": ratios}


def _plot_hist_matrix(
    fig,
    row_specs: Sequence,
    era5_series: SPEISeries,
    all_list: List[SPEISeries],
    nat_list: List[SPEISeries],
    *,
    scale: int,
    ref_start: int = 2021,
    ref_end: int = 2024,
    fut_start: int = 2041,
    fut_end: int = 2060,
    scenario1_label: str = "Scenario 1",
    scenario2_label: str = "Scenario 2",
    label_iter: Optional[Iterator[str]] = None,
    prob_products: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if len(row_specs) != 3:
        raise ValueError("row_specs must contain exactly three SubplotSpec entries (one per histogram row).")

    if prob_products is None:
        prob_products = _compute_probability_products(
            era5_series,
            all_list,
            nat_list,
            ref_start=ref_start,
            ref_end=ref_end,
            fut_start=fut_start,
            fut_end=fut_end,
            scenario2_label=scenario2_label,
        )

    def _month_values(series: SPEISeries, start: int, end: int) -> np.ndarray:
        years = series.years
        vals = _region_mean(series.values)
        mask = (years >= start) & (years <= end) & np.isfinite(vals)
        return vals[mask]

    def _ensemble_month_values(series_list: List[SPEISeries], start: int, end: int) -> np.ndarray:
        if not series_list:
            return np.array([])
        return np.concatenate([_month_values(s, start, end) for s in series_list])

    # Gather monthly values for current and future windows (ERA5 current also shown in future panel)
    era5_current_vals = _month_values(era5_series, ref_start, ref_end)
    current_vals = {
        "era5": era5_current_vals,
        "all": _ensemble_month_values(all_list, ref_start, ref_end),
        "nat": _ensemble_month_values(nat_list, ref_start, ref_end),
    }
    future_vals = {
        "era5": era5_current_vals,  # show near-term ERA5 outline in future panel
        "all": _ensemble_month_values(all_list, fut_start, fut_end),
        "nat": _ensemble_month_values(nat_list, fut_start, fut_end),
    }

    era5_region_vals = _region_mean(era5_series.values)
    era5_valid = np.isfinite(era5_region_vals)
    latest_idx: Optional[int] = int(np.where(era5_valid)[0][-1]) if np.any(era5_valid) else None
    latest_era5_value: Optional[float] = float(era5_region_vals[latest_idx]) if latest_idx is not None else None
    latest_era5_year: Optional[int] = int(era5_series.years[latest_idx]) if latest_idx is not None else None
    latest_era5_month: Optional[int] = int(era5_series.months[latest_idx]) if latest_idx is not None else None
    latest_era5_time: Optional[float] = float(era5_series.time[latest_idx]) if latest_idx is not None else None
    latest_era5_tag = (
        f"{latest_era5_year}-{latest_era5_month:02d}"
        if latest_era5_year is not None and latest_era5_month is not None
        else "latest ERA5"
    )

    def _build_bins(arrays: Sequence[np.ndarray]) -> np.ndarray:
        data = [a for a in arrays if a.size > 0]
        if not data:
            # Fallback symmetrical bins
            return np.arange(-3.0, 3.01, 0.2)
        combined = np.concatenate(data)
        span_min, span_max = float(np.min(combined)), float(np.max(combined))
        bin_min = np.floor(span_min / 0.2) * 0.2 - 0.2
        bin_max = np.ceil(span_max / 0.2) * 0.2 + 0.2
        return np.arange(bin_min, bin_max + 0.0001, 0.2)

    bins = _build_bins(
        [
            current_vals["era5"],
            current_vals["all"],
            current_vals["nat"],
            future_vals["era5"],
            future_vals["all"],
            future_vals["nat"],
        ]
    )

    def _plot_value_hist(
        ax: plt.Axes,
        values: Dict[str, np.ndarray],
        title: str,
        *,
        show_secondary_axis: bool,
    ) -> None:
        right_ax = ax.twinx()

        def _draw_left(vals: np.ndarray, color: str, label: str) -> bool:
            if vals.size == 0:
                return False
            ax.hist(
                vals,
                bins=bins,
                color=color,
                alpha=0.55,
                edgecolor="white",
                linewidth=0.6,
                label=label,
            )
            return True

        def _draw_right(vals: np.ndarray, color: str, label: str) -> bool:
            if vals.size == 0:
                return False
            right_ax.hist(
                vals,
                bins=bins,
                histtype="step",
                linestyle="--",
                linewidth=1.1,
                color=color,
                label=label,
            )
            return True

        plotted_any = False
        plotted_any |= _draw_left(values.get("all", np.array([])), ROW_COLORS["all"], scenario1_label)
        plotted_any |= _draw_left(values.get("nat", np.array([])), ROW_COLORS["nat"], scenario2_label)
        plotted_any |= _draw_right(
            values.get("era5", np.array([])),
            ROW_COLORS["era5"],
            f"ERA5 ({ref_start}-{ref_end})",
        )

        ax.set_xlim(bins[0], bins[-1])
        right_ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel(f"SPEI{scale} index")
        ax.set_ylabel("Count (Scenario 1 & 2)")
        if show_secondary_axis:
            right_ax.set_ylabel("Count (ERA5)", fontsize=8, color=ROW_COLORS["era5"])
            right_ax.tick_params(axis="y", labelsize=7, colors=ROW_COLORS["era5"])
        else:
            right_ax.set_ylabel("")
            right_ax.tick_params(axis="y", right=False, labelright=False)
            right_ax.spines["right"].set_visible(False)
        if latest_era5_value is not None and np.isfinite(latest_era5_value):
            ax.axvline(
                latest_era5_value,
                color="teal",
                linestyle="--",
                linewidth=0.2,
                alpha=0.95,
                label=f"Latest ERA5 ({latest_era5_tag})",
            )
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(False)
        right_ax.grid(False)
        handles, labels = ax.get_legend_handles_labels()
        h_r, l_r = right_ax.get_legend_handles_labels()
        handles.extend(h_r)
        labels.extend(l_r)
        if handles:
            ax.legend(handles, labels, fontsize=8, frameon=False)
        else:
            ax.text(0.5, 0.5, "No data for this window", ha="center", va="center")

    # Row 1: histograms of monthly SPEI for current and future windows
    hist_row = row_specs[0].subgridspec(1, 2, wspace=0.18)
    ax_cur_hist = fig.add_subplot(hist_row[0, 0])
    if label_iter:
        _add_panel_label(ax_cur_hist, next(label_iter))
    _plot_value_hist(
        ax_cur_hist,
        current_vals,
        f"{ref_start}-{ref_end}: Month SPEI{scale} (counts)",
        show_secondary_axis=False,
    )

    ax_fut_hist = fig.add_subplot(hist_row[0, 1])
    if label_iter:
        _add_panel_label(ax_fut_hist, next(label_iter))
    _plot_value_hist(
        ax_fut_hist,
        future_vals,
        f"{fut_start}-{fut_end}: Month SPEI{scale} (counts)",
        show_secondary_axis=True,
    )

    def _scenario_distributions_by_year_month(
        series_list: List[SPEISeries],
        *,
        start_year: int,
        end_year: int,
    ) -> Tuple[Dict[Tuple[int, int], List[float]], Dict[Tuple[int, int], float]]:
        out: Dict[Tuple[int, int], List[float]] = {}
        time_lookup: Dict[Tuple[int, int], float] = {}
        for s in series_list:
            years = s.years.astype(int)
            months = s.months.astype(int)
            vals = _region_mean(s.values)
            times = np.asarray(s.time, dtype=float)
            mask = (years >= start_year) & (years <= end_year) & np.isfinite(vals)
            for v, y, m, t in zip(vals[mask], years[mask], months[mask], times[mask]):
                key = (int(y), int(m))
                out.setdefault(key, []).append(float(v))
                if key not in time_lookup:
                    if np.isfinite(t):
                        time_lookup[key] = float(t)
                    else:
                        time_lookup[key] = float(y) + (float(m) - 0.5) / 12.0
        return out, time_lookup

    def _build_risk_line_data(
        distributions: Dict[Tuple[int, int], List[float]],
        time_lookup: Dict[Tuple[int, int], float],
        *,
        return_periods: Sequence[int],
        start_year: int,
        end_year: int,
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        keys = sorted(
            [
                k
                for k, vals in distributions.items()
                if start_year <= k[0] <= end_year and len(vals) > 0
            ]
        )
        x = np.asarray(
            [time_lookup.get(k, float(k[0]) + (float(k[1]) - 0.5) / 12.0) for k in keys],
            dtype=float,
        )
        out: Dict[int, np.ndarray] = {}
        for rp in return_periods:
            q = 100.0 / float(rp)
            vals_line: List[float] = []
            for k in keys:
                arr = np.asarray(distributions.get(k, []), dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    vals_line.append(np.nan)
                else:
                    vals_line.append(float(np.nanpercentile(arr, q)))
            out[int(rp)] = np.asarray(vals_line, dtype=float)
        return x, out

    def _probability_for_threshold(samples: Sequence[float], threshold: Optional[float]) -> Optional[float]:
        if threshold is None or not np.isfinite(threshold):
            return None
        arr = np.asarray(samples, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return float(np.mean(arr <= threshold))

    def _mean_or_none(values: Sequence[float]) -> Optional[float]:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return float(np.nanmean(arr))

    def _format_return_period_cell(return_period: Optional[float]) -> str:
        if return_period is None:
            return "n/a"
        if not np.isfinite(return_period):
            return ">100 y"
        if return_period <= 1.05:
            return "every year"
        if return_period < 10.0:
            return f"{return_period:.1f} y"
        if return_period < 100.0:
            return f"{return_period:.0f} y"
        return ">100 y"

    def _spei_classification(value: Optional[float]) -> str:
        if value is None or not np.isfinite(value):
            return "no data"
        if value <= -2.33:
            return "extremely dry"
        if value <= -1.65:
            return "severely dry"
        if value <= -1.28:
            return "moderately dry"
        if value <= -0.84:
            return "mildly dry"
        if value <= 0.83:
            return "near-normal"
        if value <= 1.27:
            return "mildly wet"
        if value <= 1.64:
            return "moderately wet"
        if value <= 2.32:
            return "severely wet"
        return "extremely wet"

    def _format_spei_cell(value: Optional[float]) -> str:
        if value is None or not np.isfinite(value):
            return "n/a"
        return f"{value:.2f} ({_spei_classification(value)})"

    def _scenario_table_stats(
        distributions: Dict[Tuple[int, int], List[float]],
        *,
        threshold: Optional[float],
        current_key: Optional[Tuple[int, int]],
        years_for_table: Sequence[int],
    ) -> Dict[str, Any]:
        p_current: Optional[float] = None
        q_current: Optional[float] = None
        if current_key is not None:
            current_samples = distributions.get(current_key, [])
            p_current = _probability_for_threshold(current_samples, threshold)
            if p_current is not None:
                arr_cur = np.asarray(current_samples, dtype=float)
                arr_cur = arr_cur[np.isfinite(arr_cur)]
                if arr_cur.size > 0:
                    q_current = float(np.nanpercentile(arr_cur, 100.0 * p_current))

        return_by_year: Dict[int, Optional[float]] = {}
        spei_by_year: Dict[int, Optional[float]] = {}
        for y in years_for_table:
            probs_month: List[float] = []
            quantiles_month: List[float] = []
            for m in range(1, 13):
                samples = distributions.get((int(y), m), [])
                p = _probability_for_threshold(samples, threshold)
                if p is not None:
                    probs_month.append(p)
                if p_current is not None:
                    arr = np.asarray(samples, dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size > 0:
                        quantiles_month.append(float(np.nanpercentile(arr, 100.0 * p_current)))
            p_mean = _mean_or_none(probs_month)
            if p_mean is None:
                return_by_year[int(y)] = None
            elif p_mean <= 0.0:
                return_by_year[int(y)] = float("inf")
            else:
                return_by_year[int(y)] = float(1.0 / p_mean)
            spei_by_year[int(y)] = _mean_or_none(quantiles_month)

        if p_current is None:
            return_current: Optional[float] = None
        elif p_current <= 0.0:
            return_current = float("inf")
        else:
            return_current = float(1.0 / p_current)

        return {
            "prob_current": p_current,
            "return_current": return_current,
            "quantile_current": q_current,
            "return_by_year": return_by_year,
            "spei_by_year": spei_by_year,
        }

    risk_start_year = 1960
    risk_end_year = 2100
    return_periods = (1, 2, 10, 20)

    dist_all, time_lookup_all = _scenario_distributions_by_year_month(
        all_list,
        start_year=risk_start_year,
        end_year=risk_end_year,
    )
    dist_nat, time_lookup_nat = _scenario_distributions_by_year_month(
        nat_list,
        start_year=risk_start_year,
        end_year=risk_end_year,
    )

    def _plot_risk_panel(
        ax: plt.Axes,
        distributions: Dict[Tuple[int, int], List[float]],
        time_lookup: Dict[Tuple[int, int], float],
        *,
        scenario_label: str,
        rp_colors: Dict[int, str],
        show_legend: bool,
    ) -> None:
        x, lines = _build_risk_line_data(
            distributions,
            time_lookup,
            return_periods=return_periods,
            start_year=risk_start_year,
            end_year=risk_end_year,
        )
        if x.size == 0:
            ax.text(0.5, 0.5, "No scenario data in 1960-2100", ha="center", va="center")
            ax.set_axis_off()
            return

        norm = Normalize(-3.0, 3.0)
        cutoff_rp = int(max(return_periods))
        cutoff_line_raw = np.clip(lines.get(cutoff_rp, np.asarray([])), -4.0, 0.0)
        cutoff_line: Optional[np.ndarray]
        if cutoff_line_raw.size == x.size:
            cutoff_line = np.asarray(cutoff_line_raw, dtype=float)
        else:
            cutoff_line = None
        bands = [
            (-4.00, -2.33, "extremely dry"),
            (-2.33, -1.65, "severely dry"),
            (-1.65, -1.28, "moderately dry"),
            (-1.28, -0.84, "mildly dry"),
            (-0.84, 0.00, "near-normal"),
        ]
        for y0, y1, label in bands:
            mid = 0.5 * (y0 + y1)
            band_color = SPEI_CMAP(norm(float(np.clip(mid, -3.0, 3.0))))
            if cutoff_line is None:
                ax.axhspan(y0, y1, color=band_color, alpha=0.33, zorder=0.05)
            else:
                band_floor = np.maximum(float(y0), cutoff_line)
                band_ceiling = np.full_like(band_floor, float(y1))
                band_mask = np.isfinite(band_floor) & np.isfinite(band_ceiling) & (band_floor < band_ceiling)
                if np.any(band_mask):
                    ax.fill_between(
                        x,
                        band_floor,
                        band_ceiling,
                        where=band_mask,
                        color=band_color,
                        alpha=0.33,
                        zorder=0.05,
                        linewidth=0.0,
                    )
            ax.text(
                risk_start_year + 0.8,
                mid,
                label,
                fontsize=6.3,
                color="#2f2f2f",
                va="center",
                ha="left",
                alpha=0.72,
                zorder=0.3,
            )

        line_styles = {
            rp: ("-" if idx % 2 == 0 else (0, (1.8, 1.2)))
            for idx, rp in enumerate(return_periods)
        }
        for rp in return_periods:
            y = np.clip(lines.get(rp, np.asarray([])), -4.0, 0.0)
            if y.size == 0:
                continue
            m = np.isfinite(y)
            if not np.any(m):
                continue
            ax.plot(
                x[m],
                y[m],
                linestyle=line_styles[rp],
                linewidth=0.7,
                color=rp_colors[rp],
                label=f"1 every {rp} year",
                zorder=1.5,
            )

        era_mask = (
            (era5_series.years >= risk_start_year)
            & (era5_series.years <= risk_end_year)
            & np.isfinite(era5_region_vals)
        )
        if np.any(era_mask):
            x_era = np.asarray(era5_series.time[era_mask], dtype=float)
            y_era = np.clip(era5_region_vals[era_mask], -4.0, 0.0)
            ax.plot(x_era, y_era, color="#0f5e5e", linewidth=2.0, label="ERA5 mean", zorder=2.2)

        if (
            latest_era5_time is not None
            and latest_era5_value is not None
            and latest_era5_year is not None
            and risk_start_year <= latest_era5_year <= risk_end_year
        ):
            ax.scatter(
                [latest_era5_time],
                [float(np.clip(latest_era5_value, -4.0, 0.0))],
                color="#0f5e5e",
                s=42,
                linewidths=0.6,
                edgecolors="white",
                zorder=2.5,
                label=f"Latest ERA5 ({latest_era5_tag})",
            )

        ax.set_xlim(float(risk_start_year), float(risk_end_year))
        ax.set_ylim(0.0, -4.0)
        ax.set_xlabel("Year")
        ax.set_ylabel(f"SPEI{scale} drought severity")
        ax.set_title(f"Risk lines: {scenario_label}", fontsize=9, loc="left")
        if MultipleLocator is not None:
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.xaxis.set_minor_locator(MultipleLocator(5))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda xval, _pos: f"{int(round(xval))}"))
        class_boundaries = [-0.84, -1.28, -1.65, -2.33]
        for boundary in class_boundaries:
            ax.axhline(
                float(boundary),
                color="#cfcfcf",
                linewidth=0.45,
                linestyle=(0, (2.0, 2.0)),
                zorder=0.35,
            )
        if show_legend:
            ax.legend(
                fontsize=6.5,
                frameon=False,
                loc="upper right",
                ncol=2,
                handlelength=2.4,
                columnspacing=0.72,
                labelspacing=0.14,
                handletextpad=0.32,
                borderaxespad=0.22,
            )

    # Row 2: scenario-wise risk lines
    risk_row = row_specs[1].subgridspec(1, 2, wspace=0.18)
    ax_risk_all = fig.add_subplot(risk_row[0, 0])
    if label_iter:
        _add_panel_label(ax_risk_all, next(label_iter))
    _plot_risk_panel(
        ax_risk_all,
        dist_all,
        time_lookup_all,
        scenario_label=scenario1_label,
        rp_colors={
            1: "#7A4A1A",
            2: "#A0662A",
            10: "#C58B4A",
            20: "#E2B97A",
        },
        show_legend=False,
    )

    ax_risk_nat = fig.add_subplot(risk_row[0, 1])
    if label_iter:
        _add_panel_label(ax_risk_nat, next(label_iter))
    _plot_risk_panel(
        ax_risk_nat,
        dist_nat,
        time_lookup_nat,
        scenario_label=scenario2_label,
        rp_colors={
            1: "#0B3C6D",
            2: "#1F5C99",
            10: "#4A86C5",
            20: "#9EC3E6",
        },
        show_legend=True,
    )

    table_years = [2030, 2040, 2050, 2060, 2100]
    current_key = (
        (int(latest_era5_year), int(latest_era5_month))
        if latest_era5_year is not None and latest_era5_month is not None
        else None
    )
    stats_all = _scenario_table_stats(
        dist_all,
        threshold=latest_era5_value,
        current_key=current_key,
        years_for_table=table_years,
    )
    stats_nat = _scenario_table_stats(
        dist_nat,
        threshold=latest_era5_value,
        current_key=current_key,
        years_for_table=table_years,
    )

    current_row_label = f"Current ({latest_era5_tag})"
    row_labels = [current_row_label] + [str(y) for y in table_years]

    rp_cells: List[List[str]] = [
        [
            _format_return_period_cell(stats_all.get("return_current")),
            _format_return_period_cell(stats_nat.get("return_current")),
        ]
    ]
    for y in table_years:
        rp_cells.append(
            [
                _format_return_period_cell(stats_all.get("return_by_year", {}).get(y)),
                _format_return_period_cell(stats_nat.get("return_by_year", {}).get(y)),
            ]
        )

    cur_q_all = stats_all.get("quantile_current")
    cur_q_nat = stats_nat.get("quantile_current")
    if cur_q_all is None:
        cur_q_all = latest_era5_value
    if cur_q_nat is None:
        cur_q_nat = latest_era5_value

    spei_cells: List[List[str]] = [
        [
            _format_spei_cell(cur_q_all),
            _format_spei_cell(cur_q_nat),
        ]
    ]
    for y in table_years:
        spei_cells.append(
            [
                _format_spei_cell(stats_all.get("spei_by_year", {}).get(y)),
                _format_spei_cell(stats_nat.get("spei_by_year", {}).get(y)),
            ]
        )

    def _draw_summary_table(ax: plt.Axes, title: str, cells: List[List[str]]) -> None:
        ax.set_axis_off()
        ax.set_title(title, fontsize=8.8, loc="left", pad=4)
        body_rows = [[row_labels[i], cells[i][0], cells[i][1]] for i in range(len(row_labels))]
        table = ax.table(
            cellText=body_rows,
            colLabels=["", scenario1_label, scenario2_label],
            loc="center",
            cellLoc="center",
            rowLoc="center",
            bbox=[0.0, 0.04, 1.0, 0.86],
            colWidths=[0.45, 0.275, 0.275],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.12)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#b8b8b8")
            cell.set_linewidth(0.5)
            if r == 0:
                cell.set_facecolor("#e5eef5")
                cell.set_text_props(weight="bold", color="#1f2b38")
            elif c == 0:
                cell.set_facecolor("#f3f4f6")
                cell.set_text_props(weight="bold", color="#2f2f2f", ha="left")
            else:
                cell.set_facecolor("white")

    # Row 3: summary tables
    table_row = row_specs[2].subgridspec(1, 2, wspace=0.24)
    ax_tab_rp = fig.add_subplot(table_row[0, 0])
    if label_iter:
        _add_panel_label(ax_tab_rp, next(label_iter))
    _draw_summary_table(
        ax_tab_rp,
        f"Return periods for a drought at least as strong as observed in {latest_era5_tag}",
        rp_cells,
    )

    ax_tab_spei = fig.add_subplot(table_row[0, 1])
    if label_iter:
        _add_panel_label(ax_tab_spei, next(label_iter))
    _draw_summary_table(
        ax_tab_spei,
        f"SPEI for the same return period as drought of {latest_era5_tag}",
        spei_cells,
    )

    return prob_products


def _prob_to_return_period(prob: Optional[float]) -> Optional[float]:
    """Convert probability (0-1) to return period in years; 1.0 -> 1 year, 0.2 -> 5 years."""
    if prob is None:
        return None
    if prob <= 0:
        return None
    return float(1.0 / prob)


def _probability_summary(values: Sequence[float]) -> Dict[str, Any]:
    """Median and percentile ranges for probabilities."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "median": None,
            "p10": None,
            "p90": None,
            "p33": None,
            "p67": None,
            "count": 0,
        }
    p10, p33, p50, p67, p90 = np.nanpercentile(arr, [10, 33, 50, 67, 90])
    return {
        "median": float(f"{p50:.6g}"),
        "p10": float(f"{p10:.6g}"),
        "p90": float(f"{p90:.6g}"),
        "p33": float(f"{p33:.6g}"),
        "p67": float(f"{p67:.6g}"),
        "range_10_90": [float(f"{p10:.6g}"), float(f"{p90:.6g}")],
        "range_33_67": [float(f"{p33:.6g}"), float(f"{p67:.6g}")],
        "count": int(arr.size),
    }


def _return_period_summary(prob_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Return period stats derived from a probability summary dict."""
    rp = {k: _prob_to_return_period(prob_summary.get(k)) for k in ("median", "p10", "p90", "p33", "p67")}
    rp["range_10_90"] = [_prob_to_return_period(prob_summary.get("p10")), _prob_to_return_period(prob_summary.get("p90"))]
    rp["range_33_67"] = [_prob_to_return_period(prob_summary.get("p33")), _prob_to_return_period(prob_summary.get("p67"))]
    rp["count"] = prob_summary.get("count", 0)
    return rp


def _ratio_summary(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p10": None,
            "p90": None,
            "p33": None,
            "p67": None,
            "count": 0,
        }
    p10, p33, p50, p67, p90 = np.nanpercentile(arr, [10, 33, 50, 67, 90])
    return {
        "mean": float(f"{np.nanmean(arr):.6g}"),
        "median": float(f"{p50:.6g}"),
        "min": float(f"{np.nanmin(arr):.6g}"),
        "max": float(f"{np.nanmax(arr):.6g}"),
        "p10": float(f"{p10:.6g}"),
        "p90": float(f"{p90:.6g}"),
        "p33": float(f"{p33:.6g}"),
        "p67": float(f"{p67:.6g}"),
        "count": int(arr.size),
    }


def _write_probability_stats(
    prob_seq: Sequence[float],
    *,
    timeperiod_label: str,
    scenario_label: str,
    region: str,
    pet_method: str,
    timetag: str,
    output_dir: Path,
    version_tag: Optional[str] = None,
) -> Path:
    """Write probability and return-period stats to JSON."""
    prob_summary = _probability_summary(prob_seq)
    rp_summary = _return_period_summary(prob_summary)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "timetag": timetag,
        "region": region,
        "scenario": scenario_label,
        "timeperiod": timeperiod_label,
        "pet_method": pet_method,
        "version_tag": version_tag,
        "probability_stats": prob_summary,
        "return_period_years": rp_summary,
        "notes": "Return period computed as 1/probability (1.0 -> 1 year, 0.2 -> 5 years).",
    }
    time_token = _safe_slug(timeperiod_label, "timewindow")
    scen_token = _safe_slug(scenario_label, "scenario")
    region_token = _safe_region_tag(region).upper()
    out_path = output_dir / f"SPEI_STATS_RETURNPERIODS_{time_token}_{scen_token}_{region_token}.json"
    _save_payload(payload, out_path)
    return out_path


def _write_ratio_stats(
    ratio_seq: Sequence[float],
    *,
    timeperiod_label: str,
    scenario1_label: str,
    scenario2_label: str,
    region: str,
    pet_method: str,
    timetag: str,
    output_dir: Path,
    version_tag: Optional[str] = None,
) -> Path:
    """Write probability ratio stats to JSON."""
    ratio_summary = _ratio_summary(ratio_seq)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "timetag": timetag,
        "region": region,
        "timeperiod": timeperiod_label,
        "scenario_ratio": f"{scenario1_label} / {scenario2_label}",
        "scenario1": scenario1_label,
        "scenario2": scenario2_label,
        "pet_method": pet_method,
        "version_tag": version_tag,
        "ratio_stats": ratio_summary,
        "notes": "Ratios correspond to panels o/p (Scenario1 probability divided by Scenario2 probability).",
    }
    time_token = _safe_slug(timeperiod_label, "timewindow")
    scen1_token = _safe_slug(scenario1_label, "scenario1")
    scen2_token = _safe_slug(scenario2_label, "scenario2")
    region_token = _safe_region_tag(region).upper()
    out_path = output_dir / f"SPEI_STATS_PROBRATIO_{time_token}_{scen1_token}_vs_{scen2_token}_{region_token}.json"
    _save_payload(payload, out_path)
    return out_path


def _plot_histogram(ax: plt.Axes, era5: SPEISeries, all_series: List[SPEISeries], nat_series: List[SPEISeries], scale: int) -> None:
    def sample(series: SPEISeries) -> np.ndarray:
        years = series.years
        mask = (years >= HIST_START) & (years <= HIST_END)
        vals = series.values[mask]
        return vals.flatten()

    era_vals = sample(era5)
    all_vals = np.concatenate([sample(s) for s in all_series]) if all_series else np.array([])
    nat_vals = np.concatenate([sample(s) for s in nat_series]) if nat_series else np.array([])

    # Drop NaNs
    era_vals = era_vals[~np.isnan(era_vals)]
    all_vals = all_vals[~np.isnan(all_vals)]
    nat_vals = nat_vals[~np.isnan(nat_vals)]

    if era_vals.size == 0 and all_vals.size == 0 and nat_vals.size == 0:
        ax.text(0.5, 0.5, f"No data for {HIST_START}–{HIST_END}", ha="center", va="center")
        ax.set_axis_off()
        return

    combined = np.concatenate([arr for arr in (era_vals, all_vals, nat_vals) if arr.size > 0])
    span_min, span_max = (float(np.min(combined)), float(np.max(combined))) if combined.size else (-3.0, 3.0)
    # Build 0.2-wide bins that cover the data range
    bin_min = np.floor(span_min / 0.2) * 0.2 - 0.2
    bin_max = np.ceil(span_max / 0.2) * 0.2 + 0.2
    bins = np.arange(bin_min, bin_max + 0.0001, 0.2)

    if era_vals.size:
        ax.hist(era_vals, bins=bins, color=ROW_COLORS["era5"], alpha=0.7, label="ERA5", rwidth=0.9)
    if all_vals.size:
        ax.hist(all_vals, bins=bins, color=ROW_COLORS["all"], alpha=0.5, label="GCMAGICC (SCENARIO1)")
    if nat_vals.size:
        ax.hist(nat_vals, bins=bins, color=ROW_COLORS["nat"], alpha=0.5, label="GCMAGICC (SCENARIO2)")

    ax.set_xlim(bin_min, bin_max)
    ax.set_xlabel(f"SPEI{scale} values ({HIST_START}–{HIST_END})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, linewidth=0.5)


def _format_info_lines(series_list: List[SPEISeries], *, scenario_label: Optional[str] = None) -> List[str]:
    """Create compact info-panel lines for a collection of series."""
    if not series_list:
        return ["No data"]
    pets = sorted({s.pet_method for s in series_list if s.pet_method})
    sources = sorted({s.baseline_source for s in series_list if s.baseline_source})
    strategies = sorted({s.baseline_strategy for s in series_list if s.baseline_strategy})
    poolings = sorted({s.baseline_pooling for s in series_list if s.baseline_pooling})
    periods = sorted(
        {
            f"{int(s.baseline_start_year)}–{int(s.baseline_end_year)}"
            for s in series_list
            if s.baseline_start_year is not None and s.baseline_end_year is not None
        }
    )
    lines = []
    if pets:
        lines.append(f"PET: {', '.join(pets)}")
    if strategies:
        lines.append(f"Baseline: {', '.join(strategies)}")
    elif poolings or sources:
        combo = []
        if sources:
            combo.append(", ".join(sources))
        if poolings:
            combo.append(", ".join(poolings))
        if combo:
            lines.append(f"Baseline: {' | '.join(combo)}")
    if periods:
        lines.append(f"Baseline period: {', '.join(periods)}")
    if scenario_label:
        lines.append(f"Scenario: {scenario_label}")
    return lines or ["No baseline info"]


def _add_info_panel(ax: plt.Axes, lines: List[str]) -> None:
    """Place an information panel to the right of the given axis."""
    y = 0.95
    for line in lines:
        ax.text(1.01, y, line, transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#444")
        y -= 0.08


def _extreme_years(series_list: List[SPEISeries]) -> Tuple[Optional[int], Optional[int]]:
    """Return (year_of_max_median, year_of_min_median) across all series."""
    best_max: Tuple[float, Optional[int]] = (-np.inf, None)
    best_min: Tuple[float, Optional[int]] = (np.inf, None)
    for s in series_list:
        if s.time.size == 0 or s.values.size == 0:
            continue
        meds = np.nanmedian(s.values, axis=1)
        years = s.years
        for val, y in zip(meds, years):
            if np.isnan(val):
                continue
            if val > best_max[0]:
                best_max = (val, int(y))
            if val < best_min[0]:
                best_min = (val, int(y))
    return best_max[1], best_min[1]


def _add_extreme_patches(axs: List[plt.Axes], patches: List[Tuple[int, str]]) -> None:
    """Draw light translucent year patches on all provided axes."""
    for year, color in patches:
        try:
            start = float(year)
            end = float(year + 1)
        except Exception:
            continue
        for ax in axs:
            ax.axvspan(start, end, color=color, alpha=0.0, zorder=0.5)


def _compute_plume(series_list: List[SPEISeries]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute per-year min/median/max envelopes across medians of series_list."""
    if not series_list:
        return None
    years_all: List[int] = []
    for s in series_list:
        years_all.extend(s.years.tolist())
    if not years_all:
        return None
    years_unique = np.unique(np.asarray(years_all, dtype=int))
    ymin = []
    ymax = []
    ymed = []
    for y in years_unique:
        vals_year = []
        for s in series_list:
            years = s.years
            mask = years == y
            if not np.any(mask):
                continue
            med = np.nanmedian(s.values[mask], axis=0)
            if med.size > 1:
                med_val = float(np.nanmean(med))
            else:
                med_val = float(np.nanmean(s.values[mask]))
            if np.isfinite(med_val):
                vals_year.append(med_val)
        if vals_year:
            vals_year = np.asarray(vals_year, dtype=float)
            ymin.append(np.nanmin(vals_year))
            ymax.append(np.nanmax(vals_year))
            ymed.append(np.nanmedian(vals_year))
    if not ymin:
        return None
    return years_unique[: len(ymin)], np.asarray(ymin), np.asarray(ymax)


def _plot_plume(axs: List[plt.Axes], years: np.ndarray, ymin: np.ndarray, ymax: np.ndarray, color: str) -> None:
    """Plot filled plume on all axes."""
    x = np.asarray(years, dtype=float)
    for ax in axs:
        ax.fill_between(x, ymin, ymax, color=color, alpha=0.1, zorder=0.4, linewidth=0)


def _region_long_name(region: str) -> str:
    """Best-effort verbose region name using regionmask long_names then pycountry."""
    if regionmask is not None:
        try:
            ar6 = regionmask.defined_regions.ar6.all
            rid = _resolve_ar6_region_id(ar6, region)
            long_names = getattr(ar6, "long_names", None) or getattr(ar6, "long_name", None)
            if rid is not None and long_names is not None and len(long_names) > rid:
                return str(long_names[rid])
        except Exception:
            pass
    # pycountry fallback for ISO-like codes
    if pycountry is not None:
        try:
            reg_up = region.upper()
            country = None
            if re.fullmatch(r"[A-Z]{3}", reg_up):
                country = pycountry.countries.get(alpha_3=reg_up)
            elif re.fullmatch(r"[A-Z]{2}", reg_up):
                country = pycountry.countries.get(alpha_2=reg_up)
            if country:
                return country.name
        except Exception:
            pass
    # Last resort: prettify delimiters
    pretty = region.replace("_", " ").replace(".", " ").replace("-", " ")
    return pretty
    return region


def _render_figure(
    *,
    era5_series: SPEISeries,
    era5drought_keune_series: Optional[List[SPEISeries]] = None,
    era5drought_map_series: Optional[SPEISeries] = None,
    all_list: List[SPEISeries],
    nat_list: List[SPEISeries],
    map_series: List[SPEISeries],
    map_titles: List[str],
    scenario1_label: str,
    scenario2_label: str,
    scenario1_tag: str,
    scenario2_tag: str,
    region: str,
    region_long_name: str,
    scale: int,
    pet_method: str,
    timetag: str,
    output_dir: Path,
    cmip6_hist_list: Optional[List[SPEISeries]] = None,
    cmip6_hist_nat_list: Optional[List[SPEISeries]] = None,
    cmip6_ssp245_list: Optional[List[SPEISeries]] = None,
    cmip6_hist_label: str = "CMIP6 historical",
    cmip6_hist_nat_label: str = "CMIP6 hist-nat",
    cmip6_hist_color: str = "#b7473a",
    cmip6_hist_nat_color: str = "#3a78b7",
    ref_window: Tuple[int, int] = (HIST_START, HIST_END),
    fut_window: Tuple[int, int] = (2041, 2060),
    prob_products: Optional[Dict[str, Any]] = None,
    plot_era5drought_keuneetal: bool = PLOT_ERA5DROUGHT_KEUNEETAL,
    show: bool = True,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Render the full figure given prepared series and return (png_path, pdf_path, prob_products)."""
    cmip6_hist_list = cmip6_hist_list or []
    cmip6_hist_nat_list = cmip6_hist_nat_list or []
    cmip6_ssp245_list = cmip6_ssp245_list or []
    era5drought_keune_series = era5drought_keune_series or []
    show_ssp245_in_j = _token_mentions_ssp245(scenario1_tag)
    show_histnat_in_k = _token_has_nat_suffix(scenario2_tag)
    fig = plt.figure(figsize=(16, 21))
    label_iter = iter(string.ascii_lowercase)
    gs = GridSpec(8, 5, figure=fig, height_ratios=[1] * 8, hspace=0.38, wspace=0.12)

    # Top map section: one extra small ERA5Drought panel at lower-left, large ERA5 panel
    # in the middle-left, and 2x3 scenario panels on the right.
    map_gs = gs[0:2, :].subgridspec(2, 6, wspace=0.04, hspace=0.08)
    if ccrs is not None:
        proj = ccrs.PlateCarree()
        ax_map_era5drought = fig.add_subplot(map_gs[1, 0], projection=proj)
        ax_map_era5 = fig.add_subplot(map_gs[:, 1:3], projection=proj)
        ax_map_nt_min = fig.add_subplot(map_gs[0, 3], projection=proj)
        ax_map_nt_mean = fig.add_subplot(map_gs[0, 4], projection=proj)
        ax_map_nt_max = fig.add_subplot(map_gs[0, 5], projection=proj)
        ax_map_ft_min = fig.add_subplot(map_gs[1, 3], projection=proj)
        ax_map_ft_mean = fig.add_subplot(map_gs[1, 4], projection=proj)
        ax_map_ft_max = fig.add_subplot(map_gs[1, 5], projection=proj)
    else:
        ax_map_era5drought = fig.add_subplot(map_gs[1, 0])
        ax_map_era5 = fig.add_subplot(map_gs[:, 1:3])
        ax_map_nt_min = fig.add_subplot(map_gs[0, 3])
        ax_map_nt_mean = fig.add_subplot(map_gs[0, 4])
        ax_map_nt_max = fig.add_subplot(map_gs[0, 5])
        ax_map_ft_min = fig.add_subplot(map_gs[1, 3])
        ax_map_ft_mean = fig.add_subplot(map_gs[1, 4])
        ax_map_ft_max = fig.add_subplot(map_gs[1, 5])

    main_axes_map = [
        ax_map_era5,
        ax_map_nt_min,
        ax_map_nt_mean,
        ax_map_nt_max,
        ax_map_ft_min,
        ax_map_ft_mean,
        ax_map_ft_max,
    ]
    main_count = min(len(map_series), len(main_axes_map))
    axes_used = main_axes_map[:main_count]
    if era5drought_map_series is not None:
        _add_panel_label(ax_map_era5drought, next(label_iter))
    for ax in axes_used:
        _add_panel_label(ax, next(label_iter))
    for ax in main_axes_map[main_count:]:
        ax.set_axis_off()

    map_axes_for_plot: List[plt.Axes] = []
    map_series_for_plot: List[SPEISeries] = []
    map_titles_for_plot: List[str] = []

    if era5drought_map_series is not None:
        map_axes_for_plot.append(ax_map_era5drought)
        map_series_for_plot.append(era5drought_map_series)
        map_titles_for_plot.append(ERA5DROUGHT_MAP_TITLE)
    else:
        ax_map_era5drought.set_axis_off()

    map_axes_for_plot.extend(axes_used)
    map_series_for_plot.extend(map_series[:main_count])
    map_titles_for_plot.extend(map_titles[:main_count])

    if map_axes_for_plot and map_series_for_plot:
        _plot_maps(
            map_axes_for_plot,
            map_series_for_plot,
            map_titles_for_plot,
            start_year=MAP_AGG_START,
            end_year=2060,
            region=region,
            scale=scale,
        )

    # Time series rows
    xlim_shared = (1850.0, 2101.0)
    spei_ylabel = f"SPEI{scale}"

    ax_era5 = fig.add_subplot(gs[2, :])
    _add_panel_label(ax_era5, next(label_iter))
    _plot_timeseries(ax_era5, era5_series, ROW_COLORS["era5"], mean_color=ROW_MEAN_COLORS["era5"], xlim=xlim_shared)
    overlay_handles: List[Any] = []
    if plot_era5drought_keuneetal and era5drought_keune_series:
        overlay_handles = _plot_era5drought_keuneetal_overlays(ax_era5, era5drought_keune_series)
        if overlay_handles:
            ax_era5.legend(
                handles=[overlay_handles[0]],
                labels=[ERA5DROUGHT_PANEL_H_LEGEND],
                loc="upper right",
                fontsize=7,
                frameon=False,
            )
    ax_era5.set_ylabel(spei_ylabel, fontsize=9)
    ax_era5.set_title("Drought index derived from ERA5", fontsize=10, loc="left")
    era5_info_lines = _format_info_lines([era5_series], scenario_label="ERA5")
    if plot_era5drought_keuneetal and era5drought_keune_series:
        methods_txt = ", ".join(s.label for s in era5drought_keune_series)
        era5_info_lines.append(f"ERA5Drought overlays: {methods_txt}")
    _add_info_panel(ax_era5, era5_info_lines)
    ax_era5.set_ylim(-5, 5)

    ax_all = fig.add_subplot(gs[3, :])
    _add_panel_label(ax_all, next(label_iter))
    if not all_list and not cmip6_hist_list:
        ax_all.text(0.5, 0.5, f"No data: GCMAGICCxERA5 (SCENARIO1: {scenario1_tag})", ha="center", va="center")
        ax_all.set_axis_off()
    else:
        for s in all_list:
            _plot_timeseries(ax_all, s, ROW_COLORS["all"], mean_color=ROW_MEAN_COLORS["all"], xlim=xlim_shared)
        cmip6_overlay_j = list(cmip6_hist_list)
        if show_ssp245_in_j:
            cmip6_overlay_j.extend(cmip6_ssp245_list)
        _overlay_cmip6_individual_lines(
            ax_all,
            cmip6_overlay_j,
            color=CMIP6_PANEL_COLOR,
            label="CMIP6",
            linewidth=CMIP6_PANEL_LW,
            linestyle="-",
            alpha=CMIP6_PANEL_ALPHA,
            zorder=2.0,
            xlim=xlim_shared,
        )
        ax_all.plot(
            era5_series.time,
            np.nanmedian(era5_series.values, axis=1),
            color=ROW_MEAN_COLORS["era5"],
            alpha=0.6,
            linewidth=1.6,
            label="ERA5 median",
            zorder=10.0,
        )
        ax_all.set_ylabel(spei_ylabel, fontsize=9)
        ax_all.set_title(f"Drought index derived from {scenario1_label}", fontsize=10, loc="left")
        info_all = _format_info_lines(all_list, scenario_label=scenario1_label)
        if cmip6_overlay_j:
            info_all.append(f"CMIP6 overlays: {len(cmip6_overlay_j)} members")
        _add_info_panel(ax_all, info_all)
        _add_scenario_panel_legend(ax_all, gcmagicc_color=ROW_MEAN_COLORS["all"])
        ax_all.set_ylim(-5, 5)

    ax_nat = fig.add_subplot(gs[4, :])
    _add_panel_label(ax_nat, next(label_iter))
    if not nat_list and not cmip6_hist_nat_list:
        ax_nat.text(0.5, 0.5, f"No data: GCMAGICCxERA5 ({scenario2_label})", ha="center", va="center")
        ax_nat.set_axis_off()
    else:
        for s in nat_list:
            _plot_timeseries(ax_nat, s, ROW_COLORS["nat"], mean_color=ROW_MEAN_COLORS["nat"], xlim=xlim_shared)
        cmip6_overlay_k = list(cmip6_hist_nat_list) if show_histnat_in_k else []
        _overlay_cmip6_individual_lines(
            ax_nat,
            cmip6_overlay_k,
            color=CMIP6_PANEL_COLOR,
            label="CMIP6",
            linewidth=CMIP6_PANEL_LW,
            linestyle="-",
            alpha=CMIP6_PANEL_ALPHA,
            zorder=2.0,
            xlim=xlim_shared,
        )
        ax_nat.plot(
            era5_series.time,
            np.nanmedian(era5_series.values, axis=1),
            color=ROW_MEAN_COLORS["era5"],
            alpha=0.6,
            linewidth=1.6,
            label="ERA5 median",
            zorder=10.0,
        )
        ax_nat.set_ylabel(spei_ylabel, fontsize=9)
        ax_nat.set_title(f"Drought index derived from {scenario2_label}", fontsize=10, loc="left")
        info_nat = _format_info_lines(nat_list, scenario_label=scenario2_label)
        if cmip6_overlay_k:
            info_nat.append(f"CMIP6 overlays: {len(cmip6_overlay_k)} members")
        _add_info_panel(ax_nat, info_nat)
        _add_scenario_panel_legend(ax_nat, gcmagicc_color=ROW_MEAN_COLORS["nat"])
        ax_nat.set_ylim(-5, 5)

    if prob_products is None:
        prob_products = _compute_probability_products(
            era5_series,
            all_list,
            nat_list,
            ref_start=ref_window[0],
            ref_end=ref_window[1],
            fut_start=fut_window[0],
            fut_end=fut_window[1],
            scenario2_label=scenario2_label,
        )

    # Histogram + risk + summary-table matrix
    _plot_hist_matrix(
        fig,
        [gs[5, :], gs[6, :], gs[7, :]],
        era5_series,
        all_list,
        nat_list,
        scale=scale,
        ref_start=ref_window[0],
        ref_end=ref_window[1],
        fut_start=fut_window[0],
        fut_end=fut_window[1],
        scenario1_label=scenario1_label,
        scenario2_label=scenario2_label,
        label_iter=label_iter,
        prob_products=prob_products,
    )

    # Extreme-year patches and plumes
    patches: List[Tuple[int, str]] = []
    all_max_year, all_min_year = _extreme_years(all_list)
    nat_max_year, nat_min_year = _extreme_years(nat_list)
    if all_max_year is not None:
        patches.append((all_max_year, ROW_COLORS["all"]))
    if all_min_year is not None:
        patches.append((all_min_year, ROW_COLORS["all"]))
    if nat_max_year is not None:
        patches.append((nat_max_year, ROW_COLORS["nat"]))
    if nat_min_year is not None:
        patches.append((nat_min_year, ROW_COLORS["nat"]))
    _add_extreme_patches([ax_era5, ax_all, ax_nat], patches)

    all_plume = _compute_plume(all_list)
    nat_plume = _compute_plume(nat_list)
    if all_plume is not None:
        y_all_years, y_all_min, y_all_max = all_plume
        _plot_plume([ax_era5, ax_all, ax_nat], y_all_years, y_all_min, y_all_max, ROW_COLORS["all"])
    if nat_plume is not None:
        y_nat_years, y_nat_min, y_nat_max = nat_plume
        _plot_plume([ax_era5, ax_all, ax_nat], y_nat_years, y_nat_min, y_nat_max, ROW_COLORS["nat"])

    plt.subplots_adjust(right=0.86)

    # Titles (left-aligned above map block)
    fig.text(
        0.02,
        0.99,
        f"{region_long_name}: Long-term drought attribution - using SPEI{scale}",
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.955,
        f"Attribution using scenarios {scenario1_label} and {scenario2_label} for region {region_long_name}, using the potential evapotranspiration method {pet_method}",
        ha="left",
        va="top",
        fontsize=14,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pet_tag_str = _normalize_pet_method(pet_method)
    region_token = _output_region_token(region)
    out_path_png = output_dir / f"SPEI_UNIFIED_{pet_tag_str}_ERA5_GCMAGICC_{timetag}_{region_token}.png"
    png_path, pdf_path = _save_png_pdf(fig, out_path_png, dpi=180, tight=True, human_timetag=_format_timetag_human(timetag))

    if show:
        try:
            plt.show()
        except Exception:
            pass
    else:
        plt.close(fig)

    return png_path, pdf_path, prob_products


def _render_maps_only(
    *,
    map_series: List[SPEISeries],
    map_titles: List[str],
    era5drought_map_series: Optional[SPEISeries] = None,
    region: str,
    region_long_name: str,
    scale: int,
    pet_method: str,
    scenario1_label: str,
    scenario2_label: str,
    timetag: str,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Render only the map panels (a–g)."""
    fig = plt.figure(figsize=(16, 8))
    label_iter = iter(string.ascii_lowercase)
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1, 1], wspace=0.12, hspace=0.15)
    map_gs = gs[:, :].subgridspec(2, 6, wspace=0.04, hspace=0.08)

    if ccrs is not None:
        proj = ccrs.PlateCarree()
        ax_map_era5drought = fig.add_subplot(map_gs[1, 0], projection=proj)
        ax_map_era5 = fig.add_subplot(map_gs[:, 1:3], projection=proj)
        ax_map_nt_min = fig.add_subplot(map_gs[0, 3], projection=proj)
        ax_map_nt_mean = fig.add_subplot(map_gs[0, 4], projection=proj)
        ax_map_nt_max = fig.add_subplot(map_gs[0, 5], projection=proj)
        ax_map_ft_min = fig.add_subplot(map_gs[1, 3], projection=proj)
        ax_map_ft_mean = fig.add_subplot(map_gs[1, 4], projection=proj)
        ax_map_ft_max = fig.add_subplot(map_gs[1, 5], projection=proj)
    else:
        ax_map_era5drought = fig.add_subplot(map_gs[1, 0])
        ax_map_era5 = fig.add_subplot(map_gs[:, 1:3])
        ax_map_nt_min = fig.add_subplot(map_gs[0, 3])
        ax_map_nt_mean = fig.add_subplot(map_gs[0, 4])
        ax_map_nt_max = fig.add_subplot(map_gs[0, 5])
        ax_map_ft_min = fig.add_subplot(map_gs[1, 3])
        ax_map_ft_mean = fig.add_subplot(map_gs[1, 4])
        ax_map_ft_max = fig.add_subplot(map_gs[1, 5])

    main_axes_map = [
        ax_map_era5,
        ax_map_nt_min,
        ax_map_nt_mean,
        ax_map_nt_max,
        ax_map_ft_min,
        ax_map_ft_mean,
        ax_map_ft_max,
    ]
    main_count = min(len(map_series), len(main_axes_map))
    axes_used = main_axes_map[:main_count]
    if era5drought_map_series is not None:
        _add_panel_label(ax_map_era5drought, next(label_iter))
    for ax in axes_used:
        _add_panel_label(ax, next(label_iter))
    for ax in main_axes_map[main_count:]:
        ax.set_axis_off()

    map_axes_for_plot: List[plt.Axes] = []
    map_series_for_plot: List[SPEISeries] = []
    map_titles_for_plot: List[str] = []
    if era5drought_map_series is not None:
        map_axes_for_plot.append(ax_map_era5drought)
        map_series_for_plot.append(era5drought_map_series)
        map_titles_for_plot.append(ERA5DROUGHT_MAP_TITLE)
    else:
        ax_map_era5drought.set_axis_off()

    map_axes_for_plot.extend(axes_used)
    map_series_for_plot.extend(map_series[:main_count])
    map_titles_for_plot.extend(map_titles[:main_count])

    if map_axes_for_plot and map_series_for_plot:
        _plot_maps(
            map_axes_for_plot,
            map_series_for_plot,
            map_titles_for_plot,
            start_year=MAP_AGG_START,
            end_year=2060,
            region=region,
            scale=scale,
        )

    fig.text(
        0.02,
        0.98,
        f"{region_long_name}: SPEI{scale} spatial views",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.94,
        f"Scenarios {scenario1_label} and {scenario2_label} | PET method {pet_method}",
        ha="left",
        va="top",
        fontsize=12,
    )

    pet_tag_str = _normalize_pet_method(pet_method)
    region_token = _output_region_token(region)
    out_path_png = output_dir / f"SPEI_MAPS_{pet_tag_str}_ERA5_GCMAGICC_{timetag}_{region_token}.png"
    png_path, pdf_path = _save_png_pdf(fig, out_path_png, dpi=190, tight=True, human_timetag=_format_timetag_human(timetag))
    plt.close(fig)
    return png_path, pdf_path


def _render_timeseries_only(
    *,
    era5_series: SPEISeries,
    era5drought_keune_series: Optional[List[SPEISeries]] = None,
    all_list: List[SPEISeries],
    nat_list: List[SPEISeries],
    scenario1_label: str,
    scenario2_label: str,
    scenario1_tag: str,
    scenario2_tag: str,
    region: str,
    region_long_name: str,
    scale: int,
    pet_method: str,
    timetag: str,
    output_dir: Path,
    cmip6_hist_list: Optional[List[SPEISeries]] = None,
    cmip6_hist_nat_list: Optional[List[SPEISeries]] = None,
    cmip6_ssp245_list: Optional[List[SPEISeries]] = None,
    cmip6_hist_label: str = "CMIP6 historical",
    cmip6_hist_nat_label: str = "CMIP6 hist-nat",
    cmip6_hist_color: str = "#b7473a",
    cmip6_hist_nat_color: str = "#3a78b7",
    plot_era5drought_keuneetal: bool = PLOT_ERA5DROUGHT_KEUNEETAL,
) -> Tuple[Path, Path]:
    """Render only the time-series panels (h–j)."""
    cmip6_hist_list = cmip6_hist_list or []
    cmip6_hist_nat_list = cmip6_hist_nat_list or []
    cmip6_ssp245_list = cmip6_ssp245_list or []
    era5drought_keune_series = era5drought_keune_series or []
    show_ssp245_in_j = _token_mentions_ssp245(scenario1_tag)
    show_histnat_in_k = _token_has_nat_suffix(scenario2_tag)
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1], hspace=0.32)
    label_iter = iter("hij")
    xlim_shared = (1850.0, 2101.0)
    spei_ylabel = f"SPEI{scale}"

    ax_era5 = fig.add_subplot(gs[0, 0])
    _add_panel_label(ax_era5, next(label_iter))
    _plot_timeseries(ax_era5, era5_series, ROW_COLORS["era5"], mean_color=ROW_MEAN_COLORS["era5"], xlim=xlim_shared)
    overlay_handles: List[Any] = []
    if plot_era5drought_keuneetal and era5drought_keune_series:
        overlay_handles = _plot_era5drought_keuneetal_overlays(ax_era5, era5drought_keune_series)
        if overlay_handles:
            ax_era5.legend(
                handles=[overlay_handles[0]],
                labels=[ERA5DROUGHT_PANEL_H_LEGEND],
                loc="upper right",
                fontsize=7,
                frameon=False,
            )
    ax_era5.set_ylabel(spei_ylabel, fontsize=9)
    ax_era5.set_title("Drought index derived from ERA5", fontsize=10, loc="left")
    era5_info_lines = _format_info_lines([era5_series], scenario_label="ERA5")
    if plot_era5drought_keuneetal and era5drought_keune_series:
        methods_txt = ", ".join(s.label for s in era5drought_keune_series)
        era5_info_lines.append(f"ERA5Drought overlays: {methods_txt}")
    _add_info_panel(ax_era5, era5_info_lines)
    ax_era5.set_ylim(-5, 5)

    ax_all = fig.add_subplot(gs[1, 0])
    _add_panel_label(ax_all, next(label_iter))
    if not all_list and not cmip6_hist_list:
        ax_all.text(0.5, 0.5, f"No data: GCMAGICCxERA5 (SCENARIO1: {scenario1_tag})", ha="center", va="center")
        ax_all.set_axis_off()
    else:
        for s in all_list:
            _plot_timeseries(ax_all, s, ROW_COLORS["all"], mean_color=ROW_MEAN_COLORS["all"], xlim=xlim_shared)
        cmip6_overlay_j = list(cmip6_hist_list)
        if show_ssp245_in_j:
            cmip6_overlay_j.extend(cmip6_ssp245_list)
        _overlay_cmip6_individual_lines(
            ax_all,
            cmip6_overlay_j,
            color=CMIP6_PANEL_COLOR,
            label="CMIP6",
            linewidth=CMIP6_PANEL_LW,
            linestyle="-",
            alpha=CMIP6_PANEL_ALPHA,
            zorder=2.0,
            xlim=xlim_shared,
        )
        ax_all.plot(
            era5_series.time,
            np.nanmedian(era5_series.values, axis=1),
            color=ROW_MEAN_COLORS["era5"],
            alpha=0.6,
            linewidth=1.6,
            label="ERA5 median",
            zorder=10.0,
        )
        ax_all.set_ylabel(spei_ylabel, fontsize=9)
        ax_all.set_title(f"Drought index derived from {scenario1_label}", fontsize=10, loc="left")
        info_all = _format_info_lines(all_list, scenario_label=scenario1_label)
        if cmip6_overlay_j:
            info_all.append(f"CMIP6 overlays: {len(cmip6_overlay_j)} members")
        _add_info_panel(ax_all, info_all)
        _add_scenario_panel_legend(ax_all, gcmagicc_color=ROW_MEAN_COLORS["all"])
        ax_all.set_ylim(-5, 5)

    ax_nat = fig.add_subplot(gs[2, 0])
    _add_panel_label(ax_nat, next(label_iter))
    if not nat_list and not cmip6_hist_nat_list:
        ax_nat.text(0.5, 0.5, f"No data: GCMAGICCxERA5 ({scenario2_label})", ha="center", va="center")
        ax_nat.set_axis_off()
    else:
        for s in nat_list:
            _plot_timeseries(ax_nat, s, ROW_COLORS["nat"], mean_color=ROW_MEAN_COLORS["nat"], xlim=xlim_shared)
        cmip6_overlay_k = list(cmip6_hist_nat_list) if show_histnat_in_k else []
        _overlay_cmip6_individual_lines(
            ax_nat,
            cmip6_overlay_k,
            color=CMIP6_PANEL_COLOR,
            label="CMIP6",
            linewidth=CMIP6_PANEL_LW,
            linestyle="-",
            alpha=CMIP6_PANEL_ALPHA,
            zorder=2.0,
            xlim=xlim_shared,
        )
        ax_nat.plot(
            era5_series.time,
            np.nanmedian(era5_series.values, axis=1),
            color=ROW_MEAN_COLORS["era5"],
            alpha=0.6,
            linewidth=1.6,
            label="ERA5 median",
            zorder=10.0,
        )
        ax_nat.set_ylabel(spei_ylabel, fontsize=9)
        ax_nat.set_title(f"Drought index derived from {scenario2_label}", fontsize=10, loc="left")
        info_nat = _format_info_lines(nat_list, scenario_label=scenario2_label)
        if cmip6_overlay_k:
            info_nat.append(f"CMIP6 overlays: {len(cmip6_overlay_k)} members")
        _add_info_panel(ax_nat, info_nat)
        _add_scenario_panel_legend(ax_nat, gcmagicc_color=ROW_MEAN_COLORS["nat"])
        ax_nat.set_ylim(-5, 5)

    patches: List[Tuple[int, str]] = []
    all_max_year, all_min_year = _extreme_years(all_list)
    nat_max_year, nat_min_year = _extreme_years(nat_list)
    if all_max_year is not None:
        patches.append((all_max_year, ROW_COLORS["all"]))
    if all_min_year is not None:
        patches.append((all_min_year, ROW_COLORS["all"]))
    if nat_max_year is not None:
        patches.append((nat_max_year, ROW_COLORS["nat"]))
    if nat_min_year is not None:
        patches.append((nat_min_year, ROW_COLORS["nat"]))
    _add_extreme_patches([ax_era5, ax_all, ax_nat], patches)

    all_plume = _compute_plume(all_list)
    nat_plume = _compute_plume(nat_list)
    if all_plume is not None:
        y_all_years, y_all_min, y_all_max = all_plume
        _plot_plume([ax_era5, ax_all, ax_nat], y_all_years, y_all_min, y_all_max, ROW_COLORS["all"])
    if nat_plume is not None:
        y_nat_years, y_nat_min, y_nat_max = nat_plume
        _plot_plume([ax_era5, ax_all, ax_nat], y_nat_years, y_nat_min, y_nat_max, ROW_COLORS["nat"])

    plt.subplots_adjust(right=0.86)

    fig.text(
        0.02,
        0.985,
        f"{region_long_name}: SPEI{scale} time series",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.955,
        f"Scenarios {scenario1_label} and {scenario2_label} | PET method {pet_method}",
        ha="left",
        va="top",
        fontsize=12,
    )

    pet_tag_str = _normalize_pet_method(pet_method)
    region_token = _output_region_token(region)
    out_path_png = output_dir / f"SPEI_TIMESERIES_{pet_tag_str}_ERA5_GCMAGICC_{timetag}_{region_token}.png"
    png_path, pdf_path = _save_png_pdf(fig, out_path_png, dpi=190, tight=True, human_timetag=_format_timetag_human(timetag))
    plt.close(fig)
    return png_path, pdf_path


def _render_hist_only(
    *,
    era5_series: SPEISeries,
    all_list: List[SPEISeries],
    nat_list: List[SPEISeries],
    scenario1_label: str,
    scenario2_label: str,
    region: str,
    region_long_name: str,
    scale: int,
    pet_method: str,
    timetag: str,
    output_dir: Path,
    ref_window: Tuple[int, int],
    fut_window: Tuple[int, int],
    prob_products: Dict[str, Any],
) -> Tuple[Path, Path]:
    """Render histogram, risk-line, and summary-table panels (k–p)."""
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.2)
    label_iter = iter("klmnop")
    _plot_hist_matrix(
        fig,
        [gs[0, :], gs[1, :], gs[2, :]],
        era5_series,
        all_list,
        nat_list,
        scale=scale,
        ref_start=ref_window[0],
        ref_end=ref_window[1],
        fut_start=fut_window[0],
        fut_end=fut_window[1],
        scenario1_label=scenario1_label,
        scenario2_label=scenario2_label,
        label_iter=label_iter,
        prob_products=prob_products,
    )

    fig.text(
        0.02,
        0.985,
        f"{region_long_name}: SPEI{scale} histograms and risk diagnostics",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.955,
        f"Scenarios {scenario1_label} and {scenario2_label} | PET method {pet_method}",
        ha="left",
        va="top",
        fontsize=12,
    )

    pet_tag_str = _normalize_pet_method(pet_method)
    region_token = _output_region_token(region)
    out_path_png = output_dir / f"SPEI_HISTOGRAMS_{pet_tag_str}_ERA5_GCMAGICC_{timetag}_{region_token}.png"
    png_path, pdf_path = _save_png_pdf(fig, out_path_png, dpi=190, tight=True, human_timetag=_format_timetag_human(timetag))
    plt.close(fig)
    return png_path, pdf_path


def _write_all_stats(
    prob_products: Dict[str, Any],
    *,
    region: str,
    scenario1_label: str,
    scenario2_label: str,
    pet_method: str,
    timetag: str,
    output_dir: Path,
    ref_window: Tuple[int, int],
    fut_window: Tuple[int, int],
    version_tag: Optional[str] = None,
) -> List[Path]:
    """Persist probability and ratio stats to JSON files."""
    paths: List[Path] = []
    probs = prob_products.get("probs", {})
    ratios = prob_products.get("ratios", {})
    time_curr = f"{ref_window[0]}-{ref_window[1]}"
    time_fut = f"{fut_window[0]}-{fut_window[1]}"

    if probs:
        paths.append(
            _write_probability_stats(
                probs.get("current", {}).get("all", []),
                timeperiod_label=time_curr,
                scenario_label=scenario1_label,
                region=region,
                pet_method=pet_method,
                timetag=timetag,
                output_dir=output_dir,
                version_tag=version_tag,
            )
        )
        paths.append(
            _write_probability_stats(
                probs.get("current", {}).get("nat", []),
                timeperiod_label=time_curr,
                scenario_label=scenario2_label,
                region=region,
                pet_method=pet_method,
                timetag=timetag,
                output_dir=output_dir,
                version_tag=version_tag,
            )
        )
        paths.append(
            _write_probability_stats(
                probs.get("future", {}).get("all", []),
                timeperiod_label=time_fut,
                scenario_label=scenario1_label,
                region=region,
                pet_method=pet_method,
                timetag=timetag,
                output_dir=output_dir,
                version_tag=version_tag,
            )
        )
        paths.append(
            _write_probability_stats(
                probs.get("future", {}).get("nat", []),
                timeperiod_label=time_fut,
                scenario_label=scenario2_label,
                region=region,
                pet_method=pet_method,
                timetag=timetag,
                output_dir=output_dir,
                version_tag=version_tag,
            )
        )

    if ratios:
        paths.append(
            _write_ratio_stats(
                ratios.get("current", []),
                timeperiod_label=time_curr,
                scenario1_label=scenario1_label,
                scenario2_label=scenario2_label,
                region=region,
                pet_method=pet_method,
                timetag=timetag,
                output_dir=output_dir,
                version_tag=version_tag,
            )
        )
        paths.append(
            _write_ratio_stats(
                ratios.get("future", []),
                timeperiod_label=time_fut,
                scenario1_label=scenario1_label,
                scenario2_label=scenario2_label,
                region=region,
                pet_method=pet_method,
                timetag=timetag,
                output_dir=output_dir,
                version_tag=version_tag,
            )
        )

    return paths


def _build_payload(
    *,
    era5_series: SPEISeries,
    era5drought_keune_series: Optional[List[SPEISeries]] = None,
    era5drought_map_series: Optional[SPEISeries] = None,
    all_list: List[SPEISeries],
    nat_list: List[SPEISeries],
    map_series: List[SPEISeries],
    map_titles: List[str],
    scenario1_label: str,
    scenario2_label: str,
    scenario1: str,
    scenario2: str,
    scenario_pair_tag: Optional[str],
    region: str,
    region_long_name: str,
    scale: int,
    pet_method: str,
    timetag: str,
    version_tag: Optional[str],
    speix_tag: Optional[str],
    source_roots: Optional[Dict[str, str]],
    ref_window: Tuple[int, int],
    fut_window: Tuple[int, int],
    cmip6_hist_list: Optional[List[SPEISeries]] = None,
    cmip6_hist_nat_list: Optional[List[SPEISeries]] = None,
    cmip6_ssp245_list: Optional[List[SPEISeries]] = None,
    cmip6_hist_label: str = "CMIP6 historical",
    cmip6_hist_nat_label: str = "CMIP6 hist-nat",
    cmip6_hist_color: str = "#b7473a",
    cmip6_hist_nat_color: str = "#3a78b7",
    plot_era5drought_keuneetal: bool = PLOT_ERA5DROUGHT_KEUNEETAL,
) -> Dict:
    """Bundle figure inputs into a JSON-serializable payload."""
    cmip6_hist_list = cmip6_hist_list or []
    cmip6_hist_nat_list = cmip6_hist_nat_list or []
    cmip6_ssp245_list = cmip6_ssp245_list or []
    era5drought_keune_series = era5drought_keune_series or []
    return {
        "version": 1,
        "generated_at": datetime.now().isoformat(),
        "region": region,
        "scale": scale,
        "pet_method": pet_method,
        "version_tag": version_tag,
        "speix_tag": speix_tag,
        "source_roots": source_roots or {},
        "scenario1": scenario1,
        "scenario2": scenario2,
        "scenario_pair_tag": scenario_pair_tag or _scenario_pair_tag(scenario1, scenario2),
        "scenario1_label": scenario1_label,
        "scenario2_label": scenario2_label,
        "region_long_name": region_long_name,
        "map_titles": map_titles,
        "map_window": {"agg_start": MAP_AGG_START, "agg_end": MAP_AGG_END},
        "hist_windows": {"current": list(ref_window), "future": list(fut_window)},
        "cmip6_overlays": {
            "scenario1": {"label": cmip6_hist_label, "color": cmip6_hist_color},
            "scenario2": {"label": cmip6_hist_nat_label, "color": cmip6_hist_nat_color},
        },
        "plot_era5drought_keuneetal": bool(plot_era5drought_keuneetal),
        "timetag": timetag,
        "series": {
            "map": [_series_to_dict(s) for s in map_series],
            "era5": _series_to_dict(era5_series),
            "era5drought_keune": [_series_to_dict(s) for s in era5drought_keune_series],
            "era5drought_map": _series_to_dict(era5drought_map_series) if era5drought_map_series is not None else None,
            "all": [_series_to_dict(s) for s in all_list],
            "nat": [_series_to_dict(s) for s in nat_list],
            "cmip6_hist": [_series_to_dict(s) for s in cmip6_hist_list],
            "cmip6_hist_nat": [_series_to_dict(s) for s in cmip6_hist_nat_list],
            "cmip6_ssp245": [_series_to_dict(s) for s in cmip6_ssp245_list],
        },
    }


def _save_payload(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_json(payload)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sanitized, f, separators=(",", ":"), allow_nan=False)
        f.write("\n")


def _payload_json_variants(path: Path) -> List[Path]:
    resolved = Path(path).expanduser().resolve(strict=False)
    suffixes = resolved.suffixes
    variants: List[Path] = []
    if suffixes[-2:] == [".json", ".gz"]:
        variants.extend([resolved, resolved.with_suffix("")])
    elif suffixes and suffixes[-1] == ".json":
        variants.extend([resolved, resolved.with_name(resolved.name + ".gz")])
    else:
        variants.extend([resolved, resolved.with_name(resolved.name + ".json")])

    seen: Set[Path] = set()
    ordered: List[Path] = []
    for candidate in variants:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def _is_gzip_payload_path(path: Path) -> bool:
    suffixes = Path(path).suffixes
    return suffixes[-2:] == [".json", ".gz"]


def _payload_stem(path: Path) -> str:
    payload_path = Path(path)
    name = payload_path.name
    if name.endswith(".json.gz"):
        return name[: -len(".json.gz")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return payload_path.stem


def _payload_with_suffix(path: Path, suffix: str, *, extension: str = ".json") -> Path:
    if not extension.startswith("."):
        raise ValueError(f"extension must start with '.', got {extension!r}")
    return Path(path).with_name(_payload_stem(path) + suffix + extension)


def _resolve_payload_json_path(path: Path) -> Path:
    for candidate in _payload_json_variants(Path(path)):
        if candidate.exists():
            return candidate
    variants = ", ".join(str(candidate) for candidate in _payload_json_variants(Path(path)))
    raise FileNotFoundError(f"Payload JSON not found. Tried: {variants}")


def _load_payload_json(path: Path) -> Dict[str, Any]:
    payload_path = _resolve_payload_json_path(Path(path))
    if _is_gzip_payload_path(payload_path):
        with gzip.open(payload_path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        with payload_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported payload structure in {payload_path}: expected a JSON object.")
    return payload


def replot(
    json_path: Path,
    *,
    show: bool = True,
    output_timetag: Optional[str] = None,
    output_base: Optional[Path] = None,
    show_cmip6_histnat: bool = True,
) -> Path:
    """Re-render figure purely from a JSON bundle produced earlier."""
    json_path = _resolve_payload_json_path(Path(json_path))
    payload = _load_payload_json(json_path)
    if payload.get("version") != 1:
        raise RuntimeError(f"Unsupported payload version: {payload.get('version')}")

    hist_windows = payload.get("hist_windows", {})
    ref_window = tuple(hist_windows.get("current", (HIST_START, HIST_END)))  # type: ignore
    fut_window = tuple(hist_windows.get("future", (2041, 2060)))  # type: ignore

    map_titles = payload.get("map_titles", [])
    map_series = [_series_from_dict(s) for s in payload["series"]["map"]]
    era5_series = _series_from_dict(payload["series"]["era5"])
    era5drought_keune_series = [_series_from_dict(s) for s in payload["series"].get("era5drought_keune", [])]
    era5drought_map_raw = payload["series"].get("era5drought_map")
    era5drought_map_series = _series_from_dict(era5drought_map_raw) if era5drought_map_raw else None
    all_list = [_series_from_dict(s) for s in payload["series"].get("all", [])]
    nat_list = [_series_from_dict(s) for s in payload["series"].get("nat", [])]
    cmip6_hist_list = [_series_from_dict(s) for s in payload["series"].get("cmip6_hist", [])]
    cmip6_hist_nat_list = [_series_from_dict(s) for s in payload["series"].get("cmip6_hist_nat", [])]
    cmip6_ssp245_list = [_series_from_dict(s) for s in payload["series"].get("cmip6_ssp245", [])]
    cmip6_meta = payload.get("cmip6_overlays", {})
    cmip6_hist_meta = cmip6_meta.get("scenario1", {}) if isinstance(cmip6_meta, dict) else {}
    cmip6_hist_nat_meta = cmip6_meta.get("scenario2", {}) if isinstance(cmip6_meta, dict) else {}
    cmip6_hist_label = str(cmip6_hist_meta.get("label", "CMIP6 historical"))
    cmip6_hist_nat_label = str(cmip6_hist_nat_meta.get("label", "CMIP6 hist-nat"))
    cmip6_hist_color = str(cmip6_hist_meta.get("color", ROW_COLORS["cmip6_hist"]))
    cmip6_hist_nat_color = str(cmip6_hist_nat_meta.get("color", ROW_COLORS["cmip6_hist_nat"]))
    if (not show_cmip6_histnat) and ("hist-nat" in cmip6_hist_nat_label.lower()):
        cmip6_hist_nat_list = []
    plot_era5drought_keuneetal = bool(payload.get("plot_era5drought_keuneetal", PLOT_ERA5DROUGHT_KEUNEETAL))

    region = payload["region"]
    region_long_name = payload.get("region_long_name", region)
    scale = int(payload["scale"])
    pet_method = payload["pet_method"]
    scenario1_label = payload["scenario1_label"]
    scenario2_label = payload["scenario2_label"]
    scenario1_tag = payload["scenario1"]
    scenario2_tag = payload.get("scenario2", "scenario2")
    scenario_pair_tag = str(payload.get("scenario_pair_tag") or _scenario_pair_tag(scenario1_tag, scenario2_tag))
    version_tag = str(payload.get("version_tag") or _DEFAULT_VERSION_TAG)
    timetag = output_timetag or payload.get("timetag") or datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_base is None:
        output_base = Path(json_path).resolve().parent / "replots"
    out_dir = (
        output_base
        / version_tag
        / scenario_pair_tag
        / timetag
        / _safe_region_tag(region)
        / _normalize_pet_method(pet_method)
    )

    prob_products = _compute_probability_products(
        era5_series,
        all_list,
        nat_list,
        ref_start=ref_window[0],
        ref_end=ref_window[1],
        fut_start=fut_window[0],
        fut_end=fut_window[1],
        scenario2_label=scenario2_label,
    )

    unified_png, unified_pdf, prob_products = _render_figure(
        era5_series=era5_series,
        era5drought_keune_series=era5drought_keune_series,
        era5drought_map_series=era5drought_map_series,
        all_list=all_list,
        nat_list=nat_list,
        cmip6_hist_list=cmip6_hist_list,
        cmip6_hist_nat_list=cmip6_hist_nat_list,
        cmip6_ssp245_list=cmip6_ssp245_list,
        cmip6_hist_label=cmip6_hist_label,
        cmip6_hist_nat_label=cmip6_hist_nat_label,
        cmip6_hist_color=cmip6_hist_color,
        cmip6_hist_nat_color=cmip6_hist_nat_color,
        map_series=map_series,
        map_titles=map_titles,
        scenario1_label=scenario1_label,
        scenario2_label=scenario2_label,
        scenario1_tag=scenario1_tag,
        scenario2_tag=scenario2_tag,
        region=region,
        region_long_name=region_long_name,
        scale=scale,
        pet_method=pet_method,
        timetag=timetag,
        output_dir=out_dir,
        ref_window=ref_window,  # type: ignore[arg-type]
        fut_window=fut_window,  # type: ignore[arg-type]
        prob_products=prob_products,
        plot_era5drought_keuneetal=plot_era5drought_keuneetal,
        show=show,
    )
    _render_maps_only(
        map_series=map_series,
        map_titles=map_titles,
        era5drought_map_series=era5drought_map_series,
        region=region,
        region_long_name=region_long_name,
        scale=scale,
        pet_method=pet_method,
        scenario1_label=scenario1_label,
        scenario2_label=scenario2_label,
        timetag=timetag,
        output_dir=out_dir,
    )
    _render_timeseries_only(
        era5_series=era5_series,
        era5drought_keune_series=era5drought_keune_series,
        all_list=all_list,
        nat_list=nat_list,
        cmip6_hist_list=cmip6_hist_list,
        cmip6_hist_nat_list=cmip6_hist_nat_list,
        cmip6_ssp245_list=cmip6_ssp245_list,
        cmip6_hist_label=cmip6_hist_label,
        cmip6_hist_nat_label=cmip6_hist_nat_label,
        cmip6_hist_color=cmip6_hist_color,
        cmip6_hist_nat_color=cmip6_hist_nat_color,
        scenario1_label=scenario1_label,
        scenario2_label=scenario2_label,
        scenario1_tag=scenario1_tag,
        scenario2_tag=scenario2_tag,
        region=region,
        region_long_name=region_long_name,
        scale=scale,
        pet_method=pet_method,
        timetag=timetag,
        output_dir=out_dir,
        plot_era5drought_keuneetal=plot_era5drought_keuneetal,
    )
    _render_hist_only(
        era5_series=era5_series,
        all_list=all_list,
        nat_list=nat_list,
        scenario1_label=scenario1_label,
        scenario2_label=scenario2_label,
        region=region,
        region_long_name=region_long_name,
        scale=scale,
        pet_method=pet_method,
        timetag=timetag,
        output_dir=out_dir,
        ref_window=ref_window,  # type: ignore[arg-type]
        fut_window=fut_window,  # type: ignore[arg-type]
        prob_products=prob_products,
    )
    _write_all_stats(
        prob_products,
        region=region,
        scenario1_label=scenario1_label,
        scenario2_label=scenario2_label,
        pet_method=pet_method,
        timetag=timetag,
        output_dir=out_dir,
        ref_window=ref_window,  # type: ignore[arg-type]
        fut_window=fut_window,  # type: ignore[arg-type]
        version_tag=version_tag,
    )
    return unified_png


# -----------------------------------------------------------------------------
# Derivative path resolution
# -----------------------------------------------------------------------------
def _resolve_speix_store_root(
    base_root: Path,
    *,
    derivatives_layout: str,
    derivatives_run_suffix: str,
    label: str,
) -> Path:
    layout_token = str(derivatives_layout or DEFAULT_DERIVATIVES_LAYOUT).strip().lower()
    preferred = (
        resolve_derivatives_root(
            base_root,
            layout=layout_token,
            suffix=derivatives_run_suffix,
            kind="data_derivatives",
        )
        / "SPEIx"
    )

    def _warn(msg: str) -> None:
        print(f"[WARN] {label}: {msg}")

    resolved = resolve_speix_root(
        base_root,
        layout=layout_token,
        suffix=derivatives_run_suffix,
        kind="data_derivatives",
        fallback_to_inplace=True,
        warn=_warn,
    )
    if resolved.exists():
        if resolved != preferred:
            print(f"[WARN] {label}: using legacy SPEIx path {resolved}")
        return resolved

    alternate_layout = (
        DERIVATIVES_LAYOUT_INPLACE
        if layout_token == DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE
        else DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE
    )
    alternate = (
        resolve_derivatives_root(
            base_root,
            layout=alternate_layout,
            suffix=derivatives_run_suffix,
            kind="data_derivatives",
        )
        / "SPEIx"
    )
    if alternate.exists():
        print(
            f"[WARN] {label}: requested SPEIx layout '{layout_token}' missing at {resolved}; "
            f"falling back to '{alternate_layout}' path {alternate}"
        )
        return alternate

    return resolved


def _prefer_stable_overlay_store(
    overlay_root: Optional[Path],
    *,
    overlay_tag: Optional[str],
    region: str,
    pet_method: str,
    fallback_root: Path,
    label: str,
) -> Tuple[Path, Optional[str]]:
    if overlay_root is None:
        return fallback_root, None
    candidate = Path(overlay_root).expanduser().resolve(strict=False)
    if _overlay_cache.stable_region_store_exists(
        candidate,
        tag=overlay_tag,
        region=region,
        pet_method=pet_method,
    ):
        print(f"Using stable overlay store for {label}: root={candidate}, tag={overlay_tag}")
        return candidate, overlay_tag
    if candidate.exists():
        print(
            f"[WARN] {label}: stable overlay root exists but no matching region/pet/tag was found "
            f"(root={candidate}, tag={overlay_tag}, region={region}, pet={pet_method}); "
            f"falling back to {fallback_root}"
        )
    return fallback_root, None


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot SPEI segments produced by 754_add_SPEI_to_ensemble_outputs.py.")
    p.add_argument(
        "--gcmagicc-scenario1-root",
        "--gcmagicc-all-root",
        dest="gcmagicc_scenario1_root",
        type=Path,
        default=DEFAULT_GCMAGICC_SCENARIO1_ROOT,
        help="SCENARIO1 GCMagicc ensemble root (legacy alias: --gcmagicc-all-root).",
    )
    p.add_argument(
        "--gcmagicc-scenario2-root",
        "--gcmagicc-nat-root",
        dest="gcmagicc_scenario2_root",
        type=Path,
        default=DEFAULT_GCMAGICC_SCENARIO2_ROOT,
        help="SCENARIO2 GCMagicc ensemble root (legacy alias: --gcmagicc-nat-root).",
    )
    p.add_argument(
        "--use-100ssp245plusnat-20260204",
        action="store_true",
        help=(
            "Use preset roots for the 100-member SSP245+NAT dataderivatives case "
            "(debiasloop_100ssp245plusnat_20260204-0448_dataderivatives)."
        ),
    )
    p.add_argument("--era5-file", type=Path, default=DEFAULT_ERA5_FILE, help="ERA5 file path (used only to locate its derivative store).")
    p.add_argument(
        "--era5drought-keuneetal-file",
        type=Path,
        default=DEFAULT_ERA5DROUGHT_KEUNEETAL_FILE,
        help="ERA5Drought SPEI file (Keune et al.) used for panel-h PET-method overlays.",
    )
    p.add_argument("--scenario1", default=DEFAULT_SCENARIO1, help="Base scenario tag for Scenario 1 (e.g., ssp245).")
    p.add_argument("--scenario2", default=DEFAULT_SCENARIO2, help="Scenario tag for Scenario 2 (e.g., ssp245-nat).")
    p.add_argument(
        "--scenario2-tag",
        "--nat-scenario",
        dest="scenario2_tag",
        default=None,
        help="Scenario token for SCENARIO2 runs (legacy alias: --nat-scenario).",
    )
    p.add_argument("--scenario2-suffix", default=DEFAULT_SCENARIO2_SUFFIX, help="Suffix for second scenario (e.g., -nat, -aer, or empty string for full forcing).")
    p.add_argument("--scenario1-label", default=DEFAULT_SCENARIO1_LABEL, help="Label to use for Scenario 1 plots.")
    p.add_argument("--scenario2-label", default=DEFAULT_SCENARIO2_LABEL, help="Label to use for the second scenario in plots.")
    p.add_argument(
        "--include-cmip6",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_INCLUDE_CMIP6,
        help="Overlay CMIP6 one-member-per-source runs from 754 outputs (default: enabled). Use --no-include-cmip6 to disable.",
    )
    p.add_argument(
        "--show-cmip6-histnat",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHOW_CMIP6_HISTNAT,
        help="Show CMIP6 hist-nat overlays (default: enabled). Use --no-show-cmip6-histnat to hide.",
    )
    p.add_argument(
        "--plot-era5drought-keuneetal",
        action=argparse.BooleanOptionalAction,
        default=PLOT_ERA5DROUGHT_KEUNEETAL,
        help="Overlay area-weighted ERA5Drought (Keune et al.) regional SPEI lines in panel h (default: enabled).",
    )
    p.add_argument(
        "--era5-overlay-root",
        type=Path,
        default=DEFAULT_ERA5_OVERLAY_ROOT,
        help="Stable regional SPEIx root preferred for ERA5 overlays (default: sibling SPEIx/overlay_canonical beside the vetted ERA5 root).",
    )
    p.add_argument("--cmip6-root", type=Path, default=DEFAULT_CMIP6_ROOT, help="CMIP6 NetCDF root used by 754 (SPEIx root is resolved via --derivatives-layout).")
    p.add_argument(
        "--cmip6-overlay-root",
        type=Path,
        default=DEFAULT_CMIP6_OVERLAY_ROOT,
        help="Stable regional SPEIx root preferred for CMIP6 overlays (default: sibling SPEIx/overlay_canonical beside the vetted CMIP6 root).",
    )
    p.add_argument(
        "--cmip6-historical-scenario",
        default=DEFAULT_CMIP6_HISTORICAL_SCENARIO,
        help="Scenario token for CMIP6 historical overlays (default historical).",
    )
    p.add_argument(
        "--cmip6-histnat-scenario",
        default=DEFAULT_CMIP6_HISTNAT_SCENARIO,
        help="Scenario token for CMIP6 hist-nat overlays (default hist-nat).",
    )
    p.add_argument(
        "--cmip6-ssp245-scenario",
        default=DEFAULT_CMIP6_SSP245_SCENARIO,
        help="Scenario token for CMIP6 ssp245 overlays (default ssp245).",
    )
    p.add_argument(
        "--cmip6-limit-ensembles",
        type=int,
        default=None,
        help="Optional separate overlay limit for CMIP6 runs (default: no limit).",
    )
    p.add_argument(
        "--scenario1-tag",
        "--all-scenario",
        dest="scenario1_tag",
        default=None,
        help="Scenario token for SCENARIO1 runs (legacy alias: --all-scenario).",
    )
    p.add_argument("--pet-method", default=DEFAULT_PET_METHOD, choices=PET_METHOD_CHOICES, help="PET method subfolder to read from segments.zarr (default penman-monteith).")
    p.add_argument("--speix-tag", default=DEFAULT_SPEIX_TAG, help="Optional SPEIx tag subdirectory under resolved SPEIx root (default: pick latest available).")
    p.add_argument(
        "--overlay-speix-tag",
        default=DEFAULT_OVERLAY_SPEIX_TAG,
        help=f"Stable overlay SPEIx tag under --era5-overlay-root/--cmip6-overlay-root (default: {DEFAULT_OVERLAY_SPEIX_TAG}).",
    )
    p.add_argument(
        "--scenario1-speix-tag",
        default=None,
        help="Optional SPEIx tag override for SCENARIO1 store only (default: --speix-tag value).",
    )
    p.add_argument(
        "--scenario2-speix-tag",
        default=None,
        help="Optional SPEIx tag override for SCENARIO2 store only (default: --speix-tag value).",
    )
    p.add_argument(
        "--derivatives-layout",
        default=DEFAULT_DERIVATIVES_LAYOUT,
        choices=list(DERIVATIVES_LAYOUT_CHOICES),
        help=f"Derivative root layout policy (default: {DEFAULT_DERIVATIVES_LAYOUT}).",
    )
    p.add_argument(
        "--derivatives-run-suffix",
        default=_DEFAULT_DERIVATIVES_RUN_SUFFIX,
        help=f"Suffix used for sibling derivative trees (default: {_DEFAULT_DERIVATIVES_RUN_SUFFIX}).",
    )
    p.add_argument(
        "--storage-access",
        default=get_storage_access_default(),
        choices=list(STORAGE_ACCESS_CHOICES),
        help=(
            "NetCDF read mode: 'mount' uses filesystem paths; 's3_direct' converts eligible "
            "input paths to s3:// and opens directly with xarray/fsspec."
        ),
    )
    p.add_argument("--region", default=DEFAULT_REGION, help="AR6 region key (default IRN).")
    p.add_argument(
        "--apply-landmask-ipcc-ar6-regions",
        action=argparse.BooleanOptionalAction,
        default=APPLY_LANDMASK_IPCCAR6REGIONS,
        help=(
            "For IPCC AR6 regions (not ISO3 countries), intersect region masks with "
            "a land mask so ocean-only grid cells are excluded (default: enabled)."
        ),
    )
    p.add_argument("--scale", type=int, default=DEFAULT_SCALE, help=f"SPEI scale in months (default {DEFAULT_SCALE}).")
    p.add_argument("--limit-ensembles", type=int, default=DEFAULT_LIMIT_ENSEMBLES, help="Limit number of ensemble members plotted per forcing (default: no limit).")
    p.add_argument(
        "--version-tag",
        default=None,
        help="Optional version tag for output namespacing/metadata (default: infer from input roots, then site default).",
    )
    p.add_argument("--output-timetag", default=None, help="Override output timetag directory (default: generated at runtime).")
    p.add_argument("--replot-json", type=Path, default=None, help="Path to a JSON bundle produced by this script to re-render the figure without reloading SPEIx data.")
    return p.parse_args(argv)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    global _ACTIVE_STORAGE_ACCESS
    args = _parse_args(argv)
    _ACTIVE_STORAGE_ACCESS = normalize_storage_access(args.storage_access)
    if bool(args.use_100ssp245plusnat_20260204):
        preset_root = PRESET_100SSP245PLUSNAT_20260204_ROOT.expanduser().resolve(strict=False)
        args.gcmagicc_scenario1_root = preset_root
        args.gcmagicc_scenario2_root = preset_root
        print(
            "Using preset: 100ssp245plusnat_20260204-0448_dataderivatives "
            f"({preset_root})"
        )
    if args.replot_json:
        if plt is None:
            raise RuntimeError("matplotlib is required for plotting.") from MPL_IMPORT_ERROR
        out_png = replot(
            args.replot_json,
            show=True,
            output_timetag=args.output_timetag,
            output_base=None,
            show_cmip6_histnat=bool(args.show_cmip6_histnat),
        )
        print(f"\n✓ Replotted figure saved to {out_png} (source {args.replot_json})")
        return

    _require_xarray()
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.") from MPL_IMPORT_ERROR

    store_all = _resolve_speix_store_root(
        args.gcmagicc_scenario1_root,
        derivatives_layout=args.derivatives_layout,
        derivatives_run_suffix=args.derivatives_run_suffix,
        label="SCENARIO1",
    )
    store_nat = _resolve_speix_store_root(
        args.gcmagicc_scenario2_root,
        derivatives_layout=args.derivatives_layout,
        derivatives_run_suffix=args.derivatives_run_suffix,
        label="SCENARIO2",
    )
    store_era5 = _resolve_speix_store_root(
        args.era5_file.parent,
        derivatives_layout=args.derivatives_layout,
        derivatives_run_suffix=args.derivatives_run_suffix,
        label="ERA5",
    )
    store_cmip6 = _resolve_speix_store_root(
        args.cmip6_root,
        derivatives_layout=args.derivatives_layout,
        derivatives_run_suffix=args.derivatives_run_suffix,
        label="CMIP6",
    )
    preferred_era5_store, preferred_era5_overlay_tag = _prefer_stable_overlay_store(
        args.era5_overlay_root,
        overlay_tag=args.overlay_speix_tag,
        region=args.region,
        pet_method=args.pet_method,
        fallback_root=store_era5,
        label="ERA5",
    )
    preferred_cmip6_store, preferred_cmip6_overlay_tag = _prefer_stable_overlay_store(
        args.cmip6_overlay_root,
        overlay_tag=args.overlay_speix_tag,
        region=args.region,
        pet_method=args.pet_method,
        fallback_root=store_cmip6,
        label="CMIP6",
    )
    print(f"Storage access mode: {_ACTIVE_STORAGE_ACCESS}")

    scenario1_tag = args.scenario1_tag or args.scenario1
    scenario2_tag = args.scenario2_tag or args.scenario2 or f"{args.scenario1}{args.scenario2_suffix}"
    scenario_pair_id = _scenario_pair_id(scenario1_tag, scenario2_tag)
    scenario1_speix_tag = args.scenario1_speix_tag if args.scenario1_speix_tag is not None else args.speix_tag
    scenario2_speix_tag = args.scenario2_speix_tag if args.scenario2_speix_tag is not None else args.speix_tag

    # Human-friendly labels (CLI overrides defaults)
    scenario1_label = args.scenario1_label or DEFAULT_SCENARIO1_LABEL
    scenario2_label = args.scenario2_label or DEFAULT_SCENARIO2_LABEL
    apply_landmask_ar6 = bool(args.apply_landmask_ipcc_ar6_regions)

    if apply_landmask_ar6 and _is_ipcc_ar6_region(args.region):
        print(f"Applying land-only mask to AR6 region '{args.region}' (APPLY_LANDMASK_IPCCAR6REGIONS=True).")

    all_runs = _load_spei_store(
        store_all,
        region=args.region,
        scale=args.scale,
        limit_ensembles=args.limit_ensembles,
        scenario_tag=scenario1_tag,
        pet_method=args.pet_method,
        store_tag=scenario1_speix_tag,
        apply_landmask_ipcc_ar6_regions=apply_landmask_ar6,
    )
    nat_runs = _load_spei_store(
        store_nat,
        region=args.region,
        scale=args.scale,
        limit_ensembles=args.limit_ensembles,
        scenario_tag=scenario2_tag,
        pet_method=args.pet_method,
        store_tag=scenario2_speix_tag,
        apply_landmask_ipcc_ar6_regions=apply_landmask_ar6,
    )
    era5_store_tag = preferred_era5_overlay_tag if preferred_era5_store != store_era5 else args.speix_tag
    try:
        era5_runs = _load_spei_store(
            preferred_era5_store,
            region=args.region,
            scale=args.scale,
            limit_ensembles=None,
            pet_method=args.pet_method,
            store_tag=era5_store_tag,
            apply_landmask_ipcc_ar6_regions=apply_landmask_ar6,
        )
        store_era5 = preferred_era5_store
    except FileNotFoundError as exc:
        if preferred_era5_store != store_era5:
            print(
                f"[WARN] ERA5 stable overlay store unavailable ({exc}); "
                f"falling back to canonical ERA5 SPEIx store: {store_era5}"
            )
            era5_store_tag = args.speix_tag
            try:
                era5_runs = _load_spei_store(
                    store_era5,
                    region=args.region,
                    scale=args.scale,
                    limit_ensembles=None,
                    pet_method=args.pet_method,
                    store_tag=era5_store_tag,
                    apply_landmask_ipcc_ar6_regions=apply_landmask_ar6,
                )
            except FileNotFoundError as inner_exc:
                exc = inner_exc
            else:
                exc = None  # type: ignore[assignment]
        if exc is None:
            pass
        else:
            fallback_store = store_all
            fallback_tag = scenario1_speix_tag if era5_store_tag is None else era5_store_tag
            print(
                f"[WARN] ERA5 SPEIx store unavailable ({exc}); "
                f"falling back to SCENARIO1 SPEIx store: {fallback_store}"
            )
            if fallback_store == store_era5:
                raise
            era5_runs = _load_spei_store(
                fallback_store,
                region=args.region,
                scale=args.scale,
                limit_ensembles=None,
                pet_method=args.pet_method,
                store_tag=fallback_tag,
                apply_landmask_ipcc_ar6_regions=apply_landmask_ar6,
            )
            store_era5 = fallback_store

    if "ERA5" not in era5_runs:
        attempt_errors: List[str] = []
        alt_layout = (
            DERIVATIVES_LAYOUT_INPLACE
            if str(args.derivatives_layout).strip().lower() == DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE
            else DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE
        )
        alt_store_era5 = (
            resolve_derivatives_root(
                args.era5_file.parent,
                layout=alt_layout,
                suffix=args.derivatives_run_suffix,
                kind="data_derivatives",
            )
            / "SPEIx"
        )
        retry_plan: List[Tuple[Path, Optional[str], str]] = [
            (store_era5, None, "ERA5 store latest tag"),
            (alt_store_era5, era5_store_tag, f"ERA5 store alternate layout ({alt_layout})"),
            (alt_store_era5, None, f"ERA5 store alternate layout ({alt_layout}) latest tag"),
            (store_all, scenario1_speix_tag, "SCENARIO1 store"),
            (store_all, None, "SCENARIO1 store latest tag"),
        ]
        seen_attempts: Set[Tuple[str, Optional[str]]] = set()
        for retry_root, retry_tag, retry_label in retry_plan:
            attempt_key = (str(retry_root.resolve(strict=False)), retry_tag)
            if attempt_key in seen_attempts:
                continue
            seen_attempts.add(attempt_key)
            try:
                retry_runs = _load_spei_store(
                    retry_root,
                    region=args.region,
                    scale=args.scale,
                    limit_ensembles=None,
                    pet_method=args.pet_method,
                    store_tag=retry_tag,
                    apply_landmask_ipcc_ar6_regions=apply_landmask_ar6,
                )
                if "ERA5" in retry_runs:
                    era5_runs = retry_runs
                    store_era5 = retry_root
                    era5_store_tag = retry_tag
                    print(
                        f"[WARN] ERA5 run missing in initial store; "
                        f"using fallback source: {retry_label} -> {retry_root}"
                    )
                    break
                attempt_errors.append(
                    f"{retry_label}: loaded store but run id 'ERA5' not present "
                    f"(root={retry_root}, tag={retry_tag or 'latest'})"
                )
            except Exception as exc:
                attempt_errors.append(
                    f"{retry_label}: {type(exc).__name__}: {exc}"
                )

    if "ERA5" not in era5_runs:
        details = "\n  - ".join(attempt_errors) if attempt_errors else "no successful fallback attempts"
        raise RuntimeError(
            f"ERA5 run not found in {store_era5}/segments.zarr (expected run id 'ERA5'). "
            "This is usually a tag/region/pet/scale mismatch against existing 754 outputs "
            f"(requested region={args.region}, pet={args.pet_method}, scale={args.scale})."
            f"\nFallback attempts:\n  - {details}"
        )
    era5_series = era5_runs["ERA5"]
    map_snapshot_year = int(MAP_AGG_END)
    map_snapshot_month: Optional[int] = None
    era5drought_keune_series: List[SPEISeries] = []
    era5drought_map_series: Optional[SPEISeries] = None
    if args.plot_era5drought_keuneetal:
        era5drought_keune_series = _load_era5drought_keuneetal_series(
            args.era5drought_keuneetal_file,
            region=args.region,
            scale=args.scale,
            apply_landmask_ipcc_ar6_regions=apply_landmask_ar6,
        )
        pet_series = _pick_era5drought_series_for_pet(
            era5drought_keune_series,
            pet_method=args.pet_method,
        )
        if pet_series is not None:
            latest_common = _latest_common_year_month(era5_series, pet_series)
            if latest_common is not None:
                map_snapshot_year, map_snapshot_month = latest_common
                print(
                    "Using shared ERA5 map timestamp for panels a/b: "
                    f"{map_snapshot_year}-{map_snapshot_month:02d}"
                )
        era5drought_map_series = _load_era5drought_keuneetal_map_snapshot(
            args.era5drought_keuneetal_file,
            region=args.region,
            scale=args.scale,
            pet_method=args.pet_method,
            year=map_snapshot_year,
            month=map_snapshot_month,
            apply_landmask_ipcc_ar6_regions=apply_landmask_ar6,
        )
        if era5drought_map_series is None and map_snapshot_month is not None:
            print(
                "⚠️ Shared ERA5 map timestamp unavailable in ERA5Drought map fields; "
                f"falling back to annual snapshot {MAP_AGG_END}."
            )
            map_snapshot_year = int(MAP_AGG_END)
            map_snapshot_month = None
            era5drought_map_series = _load_era5drought_keuneetal_map_snapshot(
                args.era5drought_keuneetal_file,
                region=args.region,
                scale=args.scale,
                pet_method=args.pet_method,
                year=map_snapshot_year,
                month=None,
                apply_landmask_ipcc_ar6_regions=apply_landmask_ar6,
            )
        if era5drought_keune_series:
            print(
                "ERA5Drought Keune et al. overlays: "
                + ", ".join(s.label for s in era5drought_keune_series)
            )
        else:
            print("⚠️ ERA5Drought Keune et al. overlay requested but no PET-method series were loaded.")
        if era5drought_map_series is None:
            print("⚠️ ERA5Drought map snapshot unavailable; top-left small panel will be omitted.")
    else:
        print("ERA5Drought Keune et al. overlays disabled (--no-plot-era5drought-keuneetal).")

    all_list = list(all_runs.values())
    nat_list = list(nat_runs.values())
    cmip6_hist_list: List[SPEISeries] = []
    cmip6_hist_nat_list: List[SPEISeries] = []
    cmip6_ssp245_list: List[SPEISeries] = []
    scenario1_is_ssp245 = _token_mentions_ssp245(scenario1_tag)
    scenario2_has_nat_suffix = _token_has_nat_suffix(scenario2_tag)
    cmip6_hist_label = "CMIP6 historical"
    cmip6_hist_nat_label = "CMIP6 hist-nat"
    cmip6_hist_color = CMIP6_PANEL_COLOR
    cmip6_hist_nat_color = CMIP6_PANEL_COLOR

    if args.include_cmip6:
        if not store_cmip6.exists():
            raise FileNotFoundError(
                f"CMIP6 SPEIx store not found at {store_cmip6}. Run 754 first to generate CMIP6 SPEIx outputs."
            )
        cmip6_limit = args.cmip6_limit_ensembles
        cmip6_exp_to_token = {
            "historical": str(args.cmip6_historical_scenario),
            "hist-nat": str(args.cmip6_histnat_scenario),
            "ssp245": str(args.cmip6_ssp245_scenario),
        }
        cmip6_exp_cache: Dict[str, List[SPEISeries]] = {}

        def _load_cmip6_overlay(exp_key: str) -> List[SPEISeries]:
            exp_norm = str(exp_key).strip().lower()
            if exp_norm in cmip6_exp_cache:
                return cmip6_exp_cache[exp_norm]
            scenario_token = cmip6_exp_to_token.get(exp_norm)
            if not scenario_token:
                cmip6_exp_cache[exp_norm] = []
                return cmip6_exp_cache[exp_norm]
            required_year = 2050 if exp_norm == "ssp245" else 2000
            try:
                candidate_root = preferred_cmip6_store
                candidate_tag = preferred_cmip6_overlay_tag if preferred_cmip6_store != store_cmip6 else args.speix_tag
                try:
                    runs = _load_spei_store(
                        candidate_root,
                        region=args.region,
                        scale=args.scale,
                        limit_ensembles=cmip6_limit,
                        scenario_tag=scenario_token,
                        pet_method=args.pet_method,
                        store_tag=candidate_tag,
                        apply_landmask_ipcc_ar6_regions=apply_landmask_ar6,
                        stacked_required_year=required_year,
                    )
                except Exception as overlay_exc:
                    if candidate_root != store_cmip6:
                        print(
                            f"⚠️ {_cmip6_overlay_label(exp_norm)} stable overlay unavailable "
                            f"(token='{scenario_token}'): {overlay_exc}. "
                            "Falling back to the existing CMIP6 SPEIx store."
                        )
                        runs = _load_spei_store(
                            store_cmip6,
                            region=args.region,
                            scale=args.scale,
                            limit_ensembles=cmip6_limit,
                            scenario_tag=scenario_token,
                            pet_method=args.pet_method,
                            store_tag=args.speix_tag,
                            apply_landmask_ipcc_ar6_regions=apply_landmask_ar6,
                            stacked_required_year=required_year,
                        )
                    else:
                        raise
                cmip6_exp_cache[exp_norm] = list(runs.values())
            except Exception as exc:
                print(
                    f"⚠️ {_cmip6_overlay_label(exp_norm)} overlays unavailable "
                    f"(token='{scenario_token}'): {exc}"
                )
                cmip6_exp_cache[exp_norm] = []
            return cmip6_exp_cache[exp_norm]

        cmip6_hist_list = _load_cmip6_overlay("historical")
        if scenario1_is_ssp245:
            cmip6_ssp245_list = _load_cmip6_overlay("ssp245")

        if scenario2_has_nat_suffix and (not args.show_cmip6_histnat):
            cmip6_hist_nat_list = []
            print("CMIP6 hist-nat overlays hidden (--no-show-cmip6-histnat).")
        elif scenario2_has_nat_suffix:
            cmip6_hist_nat_list = _load_cmip6_overlay("hist-nat")
        else:
            cmip6_hist_nat_list = []

        print(
            "CMIP6 overlays: "
            f"historical={len(cmip6_hist_list)}, "
            f"ssp245={len(cmip6_ssp245_list)} (panel-j={scenario1_is_ssp245}), "
            f"hist-nat={len(cmip6_hist_nat_list)} (panel-k-nat={scenario2_has_nat_suffix})"
        )

    # Build map series per requirements
    if map_snapshot_month is not None:
        era5_map_series = _build_map_snapshot_from_series(
            era5_series,
            year=map_snapshot_year,
            month=map_snapshot_month,
            label=f"ERA5 {map_snapshot_year}-{map_snapshot_month:02d}",
        )
        if era5_map_series is None:
            print(
                "⚠️ Shared ERA5 map timestamp unavailable in ERA5 map fields; "
                f"falling back to annual snapshot {MAP_AGG_END}."
            )
            map_snapshot_year = int(MAP_AGG_END)
            map_snapshot_month = None
    if map_snapshot_month is None:
        era5_mean = _annual_mean_for_year(era5_series, map_snapshot_year)
        era5_map_series = (
            _wrap_static_series(era5_series, era5_mean, f"ERA5 {map_snapshot_year} mean", map_snapshot_year)
            if era5_mean is not None
            else era5_series
        )

    all_max_curr, _, _ = _max_annual_mean_over_years(all_list, MAP_AGG_START, MAP_AGG_END)
    all_max_fut, _, _ = _max_annual_mean_over_years(all_list, 2041, 2060)
    all_min_curr, _, _ = _min_annual_mean_over_years(all_list, MAP_AGG_START, MAP_AGG_END)
    all_min_fut, _, _ = _min_annual_mean_over_years(all_list, 2041, 2060)
    all_mean_curr, _, _ = _mean_annual_mean_over_years(all_list, MAP_AGG_START, MAP_AGG_END)
    all_mean_fut, _, _ = _mean_annual_mean_over_years(all_list, 2041, 2060)

    # Wrap into SPEISeries objects (fall back gracefully if missing)
    series_map: List[SPEISeries] = []
    titles_map: List[str] = []

    template_all = all_list[0] if all_list else era5_series

    # Order must match axes: ERA5 (spans 2x2) then near-term min/mean/max, far-term min/mean/max
    series_map.append(era5_map_series)
    if map_snapshot_month is not None:
        titles_map.append(f"ERA5 {map_snapshot_year}-{map_snapshot_month:02d}")
    else:
        titles_map.append(f"ERA5 {map_snapshot_year} mean")

    if all_min_curr is not None:
        series_map.append(_wrap_static_series(template_all, all_min_curr, f"{scenario1_label} min {MAP_AGG_START}-{MAP_AGG_END}", MAP_AGG_END))
        titles_map.append(f"{scenario1_label} min {MAP_AGG_START}-{MAP_AGG_END}")
    if all_mean_curr is not None:
        series_map.append(_wrap_static_series(template_all, all_mean_curr, f"{scenario1_label} med {MAP_AGG_START}-{MAP_AGG_END}", MAP_AGG_END))
        titles_map.append(f"{scenario1_label} medium {MAP_AGG_START}-{MAP_AGG_END}")
    if all_max_curr is not None:
        series_map.append(_wrap_static_series(template_all, all_max_curr, f"{scenario1_label} max {MAP_AGG_START}-{MAP_AGG_END}", MAP_AGG_END))
        titles_map.append(f"{scenario1_label} max {MAP_AGG_START}-{MAP_AGG_END}")

    if all_min_fut is not None:
        series_map.append(_wrap_static_series(template_all, all_min_fut, f"{scenario1_label} min 2041-2060", 2050))
        titles_map.append(f"{scenario1_label} min 2041-2060")
    if all_mean_fut is not None:
        series_map.append(_wrap_static_series(template_all, all_mean_fut, f"{scenario1_label} med 2041-2060", 2050))
        titles_map.append(f"{scenario1_label} medium 2041-2060")
    if all_max_fut is not None:
        series_map.append(_wrap_static_series(template_all, all_max_fut, f"{scenario1_label} max 2041-2060", 2050))
        titles_map.append(f"{scenario1_label} max 2041-2060")

    ref_window = (HIST_START, HIST_END)
    fut_window = (2041, 2060)

    region_long_name = _region_long_name(args.region)

    run_version_tag = _resolve_run_version_tag(
        getattr(args, "version_tag", None),
        Path(args.gcmagicc_scenario1_root),
        Path(args.gcmagicc_scenario2_root),
    )
    scenario_pair_tag = _scenario_pair_tag(scenario1_tag, scenario2_tag)
    timetag = args.output_timetag or datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_root = Path(__file__).resolve().parent.parent
    pet_tag_str = _normalize_pet_method(args.pet_method)
    safe_region = _safe_region_tag(args.region)
    out_dir = (
        repo_root
        / "data"
        / "drought_attribution_758"
        / run_version_tag
        / scenario_pair_tag
        / timetag
        / safe_region
        / pet_tag_str
    )

    prob_products = _compute_probability_products(
        era5_series,
        all_list,
        nat_list,
        ref_start=ref_window[0],
        ref_end=ref_window[1],
        fut_start=fut_window[0],
        fut_end=fut_window[1],
        scenario2_label=scenario2_label,
    )

    unified_png, unified_pdf, prob_products = _render_figure(
        era5_series=era5_series,
        era5drought_keune_series=era5drought_keune_series,
        era5drought_map_series=era5drought_map_series,
        all_list=all_list,
        nat_list=nat_list,
        cmip6_hist_list=cmip6_hist_list,
        cmip6_hist_nat_list=cmip6_hist_nat_list,
        cmip6_ssp245_list=cmip6_ssp245_list,
        cmip6_hist_label=cmip6_hist_label,
        cmip6_hist_nat_label=cmip6_hist_nat_label,
        cmip6_hist_color=cmip6_hist_color,
        cmip6_hist_nat_color=cmip6_hist_nat_color,
        map_series=series_map,
        map_titles=titles_map,
        scenario1_label=scenario1_label,
        scenario2_label=scenario2_label,
        scenario1_tag=scenario1_tag,
        scenario2_tag=scenario2_tag,
        region=args.region,
        region_long_name=region_long_name,
        scale=args.scale,
        pet_method=args.pet_method,
        timetag=timetag,
        output_dir=out_dir,
        ref_window=ref_window,
        fut_window=fut_window,
        prob_products=prob_products,
        plot_era5drought_keuneetal=bool(args.plot_era5drought_keuneetal),
        show=True,
    )

    maps_png, maps_pdf = _render_maps_only(
        map_series=series_map,
        map_titles=titles_map,
        era5drought_map_series=era5drought_map_series,
        region=args.region,
        region_long_name=region_long_name,
        scale=args.scale,
        pet_method=args.pet_method,
        scenario1_label=scenario1_label,
        scenario2_label=scenario2_label,
        timetag=timetag,
        output_dir=out_dir,
    )

    times_png, times_pdf = _render_timeseries_only(
        era5_series=era5_series,
        era5drought_keune_series=era5drought_keune_series,
        all_list=all_list,
        nat_list=nat_list,
        cmip6_hist_list=cmip6_hist_list,
        cmip6_hist_nat_list=cmip6_hist_nat_list,
        cmip6_ssp245_list=cmip6_ssp245_list,
        cmip6_hist_label=cmip6_hist_label,
        cmip6_hist_nat_label=cmip6_hist_nat_label,
        cmip6_hist_color=cmip6_hist_color,
        cmip6_hist_nat_color=cmip6_hist_nat_color,
        scenario1_label=scenario1_label,
        scenario2_label=scenario2_label,
        scenario1_tag=scenario1_tag,
        scenario2_tag=scenario2_tag,
        region=args.region,
        region_long_name=region_long_name,
        scale=args.scale,
        pet_method=args.pet_method,
        timetag=timetag,
        output_dir=out_dir,
        plot_era5drought_keuneetal=bool(args.plot_era5drought_keuneetal),
    )

    hist_png, hist_pdf = _render_hist_only(
        era5_series=era5_series,
        all_list=all_list,
        nat_list=nat_list,
        scenario1_label=scenario1_label,
        scenario2_label=scenario2_label,
        region=args.region,
        region_long_name=region_long_name,
        scale=args.scale,
        pet_method=args.pet_method,
        timetag=timetag,
        output_dir=out_dir,
        ref_window=ref_window,
        fut_window=fut_window,
        prob_products=prob_products,
    )

    stats_paths = _write_all_stats(
        prob_products,
        region=args.region,
        scenario1_label=scenario1_label,
        scenario2_label=scenario2_label,
        pet_method=args.pet_method,
        timetag=timetag,
        output_dir=out_dir,
        ref_window=ref_window,
        fut_window=fut_window,
        version_tag=run_version_tag,
    )

    payload = _build_payload(
        era5_series=era5_series,
        era5drought_keune_series=era5drought_keune_series,
        era5drought_map_series=era5drought_map_series,
        all_list=all_list,
        nat_list=nat_list,
        cmip6_hist_list=cmip6_hist_list,
        cmip6_hist_nat_list=cmip6_hist_nat_list,
        cmip6_ssp245_list=cmip6_ssp245_list,
        cmip6_hist_label=cmip6_hist_label,
        cmip6_hist_nat_label=cmip6_hist_nat_label,
        cmip6_hist_color=cmip6_hist_color,
        cmip6_hist_nat_color=cmip6_hist_nat_color,
        map_series=series_map,
        map_titles=titles_map,
        scenario1_label=scenario1_label,
        scenario2_label=scenario2_label,
        scenario1=scenario1_tag,
        scenario2=scenario2_tag,
        scenario_pair_tag=scenario_pair_tag,
        region=args.region,
        region_long_name=region_long_name,
        scale=args.scale,
        pet_method=args.pet_method,
        timetag=timetag,
        version_tag=run_version_tag,
        speix_tag=args.speix_tag,
        source_roots={
            "scenario1": str(Path(args.gcmagicc_scenario1_root).expanduser().resolve(strict=False)),
            "scenario2": str(Path(args.gcmagicc_scenario2_root).expanduser().resolve(strict=False)),
            "era5": str(store_era5),
            "cmip6": str(preferred_cmip6_store if preferred_cmip6_store.exists() else Path(args.cmip6_root).expanduser().resolve(strict=False)) if args.include_cmip6 else "",
        },
        ref_window=ref_window,
        fut_window=fut_window,
        plot_era5drought_keuneetal=bool(args.plot_era5drought_keuneetal),
    )
    json_path = unified_png.with_suffix(".json")
    _save_payload(payload, json_path)

    # Lighter-weight companion JSONs for web interactivity
    panel_a_payload = _build_payload(
        era5_series=era5_map_series,
        era5drought_keune_series=[],
        era5drought_map_series=None,
        all_list=[],
        nat_list=[],
        cmip6_hist_list=[],
        cmip6_hist_nat_list=[],
        cmip6_ssp245_list=[],
        cmip6_hist_label=cmip6_hist_label,
        cmip6_hist_nat_label=cmip6_hist_nat_label,
        cmip6_hist_color=cmip6_hist_color,
        cmip6_hist_nat_color=cmip6_hist_nat_color,
        map_series=[era5_map_series],
        map_titles=[titles_map[0]],
        scenario1_label=scenario1_label,
        scenario2_label=scenario2_label,
        scenario1=scenario1_tag,
        scenario2=scenario2_tag,
        scenario_pair_tag=scenario_pair_tag,
        region=args.region,
        region_long_name=region_long_name,
        scale=args.scale,
        pet_method=args.pet_method,
        timetag=timetag,
        version_tag=run_version_tag,
        speix_tag=args.speix_tag,
        source_roots={
            "scenario1": str(Path(args.gcmagicc_scenario1_root).expanduser().resolve(strict=False)),
            "scenario2": str(Path(args.gcmagicc_scenario2_root).expanduser().resolve(strict=False)),
            "era5": str(store_era5),
            "cmip6": str(preferred_cmip6_store if preferred_cmip6_store.exists() else Path(args.cmip6_root).expanduser().resolve(strict=False)) if args.include_cmip6 else "",
        },
        ref_window=ref_window,
        fut_window=fut_window,
        plot_era5drought_keuneetal=False,
    )
    _save_payload(panel_a_payload, unified_png.with_name(unified_png.stem + "_panelA_map.json"))

    panel_i_payload = {
        "version": 1,
        "generated_at": datetime.now().isoformat(),
        "region": args.region,
        "region_long_name": region_long_name,
        "scale": args.scale,
        "pet_method": args.pet_method,
        "timetag": timetag,
        "version_tag": run_version_tag,
        "scenario_pair_tag": scenario_pair_tag,
        "scenario1": scenario1_tag,
        "scenario2": scenario2_tag,
        "scenario1_label": scenario1_label,
        "scenario2_label": scenario2_label,
        "speix_tag": args.speix_tag,
        "scenario1": scenario1_tag,
        "scenario2": scenario2_tag,
        "scenario_pair_id": scenario_pair_id,
        "scenario1_label": scenario1_label,
        "scenario2_label": scenario2_label,
        "source_roots": {
            "scenario1": str(Path(args.gcmagicc_scenario1_root).expanduser().resolve(strict=False)),
            "scenario2": str(Path(args.gcmagicc_scenario2_root).expanduser().resolve(strict=False)),
            "era5": str(store_era5),
            "cmip6": str(preferred_cmip6_store if preferred_cmip6_store.exists() else Path(args.cmip6_root).expanduser().resolve(strict=False)) if args.include_cmip6 else "",
        },
        "cmip6_overlays": {
            "scenario1": {"label": cmip6_hist_label, "color": cmip6_hist_color},
            "scenario2": {"label": cmip6_hist_nat_label, "color": cmip6_hist_nat_color},
        },
        "series": {
            "era5": _series_to_dict(_median_series(_limit_series_columns(_annualize_series(era5_series), MAX_GRID_TRACES))),
            "era5drought_keune": [
                _series_to_dict(_median_series(_annualize_series(s))) for s in era5drought_keune_series
            ],
            "all": [
                _series_to_dict(_limit_series_columns(_annualize_series(s), MAX_GRID_TRACES)) for s in all_list
            ] + [
                _series_to_dict(_median_series(_annualize_series(s))) for s in all_list
            ],
            "nat": [
                _series_to_dict(_limit_series_columns(_annualize_series(s), MAX_GRID_TRACES)) for s in nat_list
            ] + [
                _series_to_dict(_median_series(_annualize_series(s))) for s in nat_list
            ],
            "cmip6_hist": [
                _series_to_dict(_median_series(_annualize_series(s))) for s in cmip6_hist_list
            ],
            "cmip6_hist_nat": [
                _series_to_dict(_median_series(_annualize_series(s))) for s in cmip6_hist_nat_list
            ],
            "cmip6_ssp245": [
                _series_to_dict(_median_series(_annualize_series(s))) for s in cmip6_ssp245_list
            ],
        },
        "xlim": [1850.0, 2101.0],
        "panel_label": "i",
    }
    _save_payload(panel_i_payload, unified_png.with_name(unified_png.stem + "_panelI_timeseries.json"))

    print(f"\n✓ Unified figure saved to {unified_png} and {unified_pdf}")
    print(f"✓ Map-only figure saved to {maps_png} and {maps_pdf}")
    print(f"✓ Time-series figure saved to {times_png} and {times_pdf}")
    print(f"✓ Histogram figure saved to {hist_png} and {hist_pdf}")
    print(f"✓ Figure data saved to {json_path}")
    if stats_paths:
        print("✓ Stats JSON files:")
        for p in stats_paths:
            print(f"  - {p}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
