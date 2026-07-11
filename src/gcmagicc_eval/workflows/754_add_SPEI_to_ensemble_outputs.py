#!/usr/bin/env python3
"""
754_add_SPEI_to_ensemble_outputs (rev6 - save P/PET components + summary PDF panels)

Purpose
-------
Compute SPEI{scale} grid-point time series for:
  (1) SCENARIO2 GCMagicc ensemble NetCDF outputs
  (2) SCENARIO1 GCMagicc ensemble NetCDF outputs
  (3) ERA5 historical NetCDF file

Additionally (requested)
------------------------
- Save the rolled accumulation components of WB = P - PET, i.e. accumulated precipitation (P)
  and accumulated potential evapotranspiration (PET) over the last {scale} months for each point.
- Emit a per-run summary PDF with 12 monthly panels (P vs PET with USDM SPEI drought-class background).

Baseline strategy options (new)
------------------------------
This script supports three baseline strategies via --baseline-strategy:

  - pooled     : pool all SCENARIO2 members together for the ERA5 period (month-specific),
                 fit one common baseline distribution, apply to SCENARIO2/SCENARIO1/ERA5.

  - memberwise : fit ERA5 baseline from ERA5; fit baseline per SCENARIO2 member (ERA5 period);
                 standardize SCENARIO1 members using the matching SCENARIO2 member baseline.

  - era5       : fit one baseline distribution from ERA5 (month-specific), apply to SCENARIO2/SCENARIO1/ERA5.

The default baseline window for pooled/memberwise is the ERA5 data coverage
(unless you explicitly set --baseline-start-year/--baseline-end-year).

Baseline fitting details
------------------------
- The fitted object is the per-month distribution of the {scale}-month rolling
  accumulation of the climatic water balance: WB = P - PET (monthly).
- Fits are done per calendar month (Jan..Dec) and per grid point.
- For pooled/memberwise, by default the fit window is restricted to ERA5 coverage
  (prevents future scenario years contaminating the baseline).

Supported standardization (fit) types
-------------------------------------
- zscore       : month-wise μ/σ on rolled WB (fast, robust; parametric normal)
- loglogistic  : month-wise 3-parameter generalized logistic ("log-Logistic" in R SPEI)
                 fitted via unbiased PWM → L-moments, standardized via pglo CDF + qnorm
                 (consistent with SPEI package defaults).

Outputs
-------
For each input root, a Zarr segment store is written to:

  <ROOT>/data_derivatives/SPEIx/segments.zarr

with groups:

  runs/<RUN_ID>/spei{scale}__{REGION}__grid-points__{START}-{END}__all

Additionally, the baseline parameter dataset is written to:

  <BASELINE_ROOT>/data_derivatives/SPEIx/fits/BASEFIT__spei{scale}__{REGION}__{FIT}__{PET}__{STRATEGY}__{ID}.nc

Summary plots (new)
-------------------
For each processed run, a PDF is written to:
  <ROOT>/data_derivatives/SPEIx/summary_pdfs/
containing 12 monthly panels (Jan..Dec) of (P_accum, PET_accum) trajectories with drought-class background.

Run example
-----------
pixi run python notebooks/754_add_SPEI_to_ensemble_outputs.py --force \
  --gcmagicc-scenario1-root data/site_eth/GCMAGICCoutput/ERA5splicedS3/v101/ssp245/AR6/all/n_20/original/run_<latest_verified> \
  --gcmagicc-scenario2-root data/site_eth/GCMAGICCoutput/ERA5splicedS3/v101/ssp245/AR6/nat/n_20/original/run_<latest_verified> \
  --era5-file data/site_eth/out_ERA5_4July2025_1degree_vetted/DAT_ERA5_historical-ERA5_r1i1p1f1_clt-day-hurs-huss-month-pr-psl-rlut-rsds-rsdt-rsnt-rtmt-sfcWind-tas-tasmax-tasmin-ts-year.nc \
  --region IRN --scale 48 --pet-method auto --fit loglogistic \
  --baseline-strategy memberwise \
  --out-start-year 1975 --out-end-year 2024

Notes
-----
- Region selection is limited to the requested AR6 region by using
  a template point set derived from the baseline source file and then sampling
  all datasets (SCENARIO2/SCENARIO1/ERA5) onto that same set of lat/lon points (nearest).
- This design ensures point-by-point parameter reuse and avoids alignment issues.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import subprocess
import tempfile
import traceback
import warnings
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import xarray as xr

# Plotting (optional; required for requested PDF summaries)
try:  # pragma: no cover - optional dependency
    import matplotlib  # type: ignore
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
    from matplotlib.collections import LineCollection  # type: ignore
    from matplotlib.colors import BoundaryNorm, ListedColormap  # type: ignore
    from matplotlib.patches import Polygon  # type: ignore
    try:  # optional inset helper
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # type: ignore
    except Exception:  # pragma: no cover - optional
        inset_axes = None  # type: ignore
except Exception:  # pragma: no cover - optional
    matplotlib = None
    plt = None
    PdfPages = None
    LineCollection = None
    BoundaryNorm = None
    ListedColormap = None
    Polygon = None  # type: ignore
    inset_axes = None  # type: ignore

# Optional dependency; fallback to precomputed masks if missing
try:  # pragma: no cover - optional
    import regionmask  # type: ignore
except Exception:  # pragma: no cover - optional
    regionmask = None

# Optional: zarr warning class used for suppressing unstable dtype chatter when writing
try:  # pragma: no cover - optional
    from zarr.errors import UnstableSpecificationWarning as ZarrUnstableSpecificationWarning  # type: ignore
except Exception:  # pragma: no cover - optional
    ZarrUnstableSpecificationWarning = None

try:  # pragma: no cover - optional
    import zarr  # type: ignore
except Exception:  # pragma: no cover - optional
    zarr = None  # type: ignore

try:  # pragma: no cover - optional
    from zarr.codecs import ZstdCodec as _ZarrZstdCodec  # type: ignore
except Exception:  # pragma: no cover - optional
    _ZarrZstdCodec = None  # type: ignore

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

get_cmip6_vetted_path = _helper_path_utils.get_cmip6_vetted_path
get_created_nc_files_root = _helper_path_utils.get_created_nc_files_root
get_data_root = _helper_path_utils.get_data_root
get_site = _helper_path_utils.get_site
get_era5spliced_root = _helper_path_utils.get_era5spliced_root
build_era5spliced_dataset_path = _helper_path_utils.build_era5spliced_dataset_path
copy_remote_file_atomic_via_rclone = _helper_path_utils.copy_remote_file_atomic_via_rclone
path_uses_rclone_mount = _helper_path_utils.path_uses_rclone_mount
resolve_canonical_dataset_root = _helper_path_utils.resolve_canonical_dataset_root
parse_era5spliced_dataset_path = _helper_path_utils.parse_era5spliced_dataset_path
resolve_derivatives_root = _helper_path_utils.resolve_derivatives_root
resolve_rclone_source_ref = _helper_path_utils.resolve_rclone_source_ref
get_era5_main_file = _helper_path_utils.get_era5_main_file
get_version_default = _helper_path_utils.get_version_default
get_storage_access_default = _helper_path_utils.get_storage_access_default
normalize_storage_access = _helper_path_utils.normalize_storage_access
convert_local_path_to_s3_uri = _helper_path_utils.convert_local_path_to_s3_uri
convert_local_path_to_s3_uri_candidates = _helper_path_utils.convert_local_path_to_s3_uri_candidates
get_site_scratch_data_root = _helper_path_utils.get_site_scratch_data_root
DERIVATIVES_LAYOUT_CHOICES = _helper_path_utils.DERIVATIVES_LAYOUT_CHOICES
DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE = _helper_path_utils.DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE
DEFAULT_DERIVATIVES_RUN_SUFFIX = _helper_path_utils.DEFAULT_DERIVATIVES_RUN_SUFFIX
STORAGE_ACCESS_CHOICES = _helper_path_utils.STORAGE_ACCESS_CHOICES
STORAGE_ACCESS_MOUNT = _helper_path_utils.STORAGE_ACCESS_MOUNT
STORAGE_ACCESS_RCLONE_CACHE = _helper_path_utils.STORAGE_ACCESS_RCLONE_CACHE
STORAGE_ACCESS_S3_DIRECT = _helper_path_utils.STORAGE_ACCESS_S3_DIRECT

DEFAULT_DERIVATIVES_LAYOUT = (
    os.environ.get("GCMAGICC_DERIVATIVES_LAYOUT", DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE).strip().lower()
    or DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE
)
_ACTIVE_DERIVATIVES_LAYOUT = DEFAULT_DERIVATIVES_LAYOUT
_ACTIVE_DERIVATIVES_RUN_SUFFIX = (
    os.environ.get("GCMAGICC_DERIVATIVES_RUN_SUFFIX", DEFAULT_DERIVATIVES_RUN_SUFFIX).strip()
    or DEFAULT_DERIVATIVES_RUN_SUFFIX
)
_ACTIVE_STORAGE_ACCESS = get_storage_access_default()
DEFAULT_RCLONE_CACHE_ROOT = (
    Path(os.environ.get("GCMAGICC_RCLONE_CACHE_ROOT", "")).expanduser().resolve(strict=False)
    if os.environ.get("GCMAGICC_RCLONE_CACHE_ROOT", "").strip()
    else get_site_scratch_data_root() / "GCMAGICC_stage" / "754_cache"
)
_ACTIVE_RCLONE_CACHE_ROOT = DEFAULT_RCLONE_CACHE_ROOT

_CREATED_NC_ROOT = get_created_nc_files_root()
_DEFAULT_VERSION_TAG = get_version_default()
_ERA5SPLICED_ROOT = get_era5spliced_root()
_ACTIVE_SITE = get_site()
_SSP245_100_LOOP_DIR = (
    "debiasloop_100ssp245plusnatv100_20260223-0301"
    if _ACTIVE_SITE == "gus"
    else "debiasloop_100ssp245plusnat_20260204-0448"
)
DEFAULT_AUTO_CONSOLIDATE = os.environ.get("GCMAGICC_AUTO_CONSOLIDATE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DEFAULT_AUTO_CONSOLIDATE_CLEANUP_LOCAL = (
    os.environ.get("GCMAGICC_AUTO_CONSOLIDATE_CLEANUP_LOCAL", "1").strip().lower()
    not in {"0", "false", "no", "off"}
)
_DEFAULT_S3_STAGE_DIR_BY_SITE = (
    "data/site_gus/tmp/gcmagicc_s3_stage_754"
    if _ACTIVE_SITE == "gus"
    else "data/tmp/gcmagicc_s3_stage_754"
)
DEFAULT_S3_ENV_FILE = Path(
    os.environ.get("GCMAGICC_754_S3_ENV_FILE", "/u/maltemh/.config/gcmagicc/ovh_s3.env")
).expanduser().resolve(strict=False)
DEFAULT_S3_STAGE_DIR = Path(
    os.environ.get("GCMAGICC_754_S3_STAGE_DIR", _DEFAULT_S3_STAGE_DIR_BY_SITE)
).expanduser().resolve(strict=False)
DEFAULT_S3_PREFLIGHT = os.environ.get("GCMAGICC_754_S3_PREFLIGHT", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_ACTIVE_S3_STAGE_DIR = DEFAULT_S3_STAGE_DIR
_ACTIVE_S3_PREFLIGHT = DEFAULT_S3_PREFLIGHT
_S3_ENV_ALLOWED_PREFIXES = ("AWS_", "GCMAGICC_S3_")
_S3_ENV_ALLOWED_EXACT = {"GCMAGICC_OBJECT_BUCKET", "OVH_S3_BUCKET"}
_S3_FS: object | None = None
_ACTIVE_S3_LOCAL_FALLBACK_PATHS: set[str] = set()
_ACTIVE_S3_SELECTED_URI_BY_LOCAL: Dict[str, str] = {}
_ACTIVE_S3_PREFLIGHT_COUNTS: Dict[str, int] = {}
_DERIV_WRITE_REDIRECT_CACHE: dict[str, Path] = {}


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


def _resolve_latest_original_root(
    *,
    experiment_id: str,
    arx: str = "AR6",
    runmodus: str = "all",
    n_ensemble: str = "n_20",
) -> Path:
    return resolve_canonical_dataset_root(
        version=_DEFAULT_VERSION_TAG,
        experiment_id=experiment_id,
        arx=arx,
        runmodus=runmodus,
        n_ensemble=n_ensemble,
        kind="original",
        root=_ERA5SPLICED_ROOT,
    )


def _consolidator_script_path() -> Path:
    return (_REPO_ROOT / "scripts" / "2018_consolidate_era5spliced_s3.py").resolve(strict=False)


def _default_autoconsolidate_config() -> Path:
    return (_REPO_ROOT / "scripts" / "2018_consolidate_era5spliced_s3.example.json").resolve(strict=False)


def _run_autoconsolidate(
    *,
    source_paths: Sequence[Path],
    config_path: Optional[Path],
    cleanup_local: bool,
) -> None:
    consolidator = _consolidator_script_path()
    if not consolidator.exists():
        raise FileNotFoundError(f"Consolidator script not found: {consolidator}")
    cfg = Path(config_path).expanduser().resolve(strict=False) if config_path else _default_autoconsolidate_config()
    cmd: List[str] = [
        sys.executable,
        str(consolidator),
        "autoconsolidate",
        "--config",
        str(cfg),
    ]
    for source_path in source_paths:
        cmd.extend(["--source-path", str(Path(source_path).expanduser().resolve(strict=False))])
    cmd.extend(["--apply", "--verify-size-only"])
    if cleanup_local:
        cmd.append("--cleanup-local")
    _log("🔁 Running auto-consolidate:")
    _log("   " + " ".join(cmd))
    cp = subprocess.run(cmd, check=False)
    if cp.returncode != 0:
        raise RuntimeError(
            f"Auto-consolidate failed with exit code {cp.returncode}. "
            "Local staged derivative data was kept."
        )


# -----------------------------------------------------------------------------
# User-facing defaults (requested)
# -----------------------------------------------------------------------------
# DEFAULT_GCMAGICC_SCENARIO1_ROOT = Path(
#     "data/site_eth/GCMAGICCoutput/ERA5splicedS3/"
#     "v101/ssp245/AR6/all/n_20/original/run_<latest_verified>"
# )

# You can switch preset input roots with this token or with env var
# GCMAGICC_754_SET2CRUNCH.
# Supported presets: "ssp245", "NDCs", "CurPol_lowNDC".
set2crunch = os.environ.get("GCMAGICC_754_SET2CRUNCH", "100ssp245") # "100curpol_lowndc")
set2crunch_token = str(set2crunch).strip().lower().replace("-", "_")

if set2crunch_token == "ndcs":
    set2crunch = "NDCs"
    default_scen1_root = _resolve_latest_original_root(
        experiment_id="NDC-Trump-low",
        arx="AR6",
        runmodus="all",
        n_ensemble="n_100",
    )
    default_scen2_root = _resolve_latest_original_root(
        experiment_id="NDC-submitted-low",
        arx="AR6",
        runmodus="all",
        n_ensemble="n_100",
    )
    DEFAULT_GCMAGICC_SCENARIO1_ROOT = Path(
        os.environ.get(
            "GCMAGICC_754_SCENARIO1_ROOT",
            str(default_scen1_root),
        )
    ).expanduser().resolve(strict=False)
    DEFAULT_GCMAGICC_SCENARIO2_ROOT = Path(
        os.environ.get(
            "GCMAGICC_754_SCENARIO2_ROOT",
            str(default_scen2_root),
        )
    ).expanduser().resolve(strict=False)

    # Use None/"" to disable suffix-based filtering for Scenario2 and keep all files
    # under --gcmagicc-scenario2-root (useful when Scenario2 is a distinct scenario root).
    DEFAULT_GCMAGICC_SCENARIO2_SUFFIX = None

    BASELINE_CONFIG: Dict[str, Dict[str, str]] = {
        # Target = ERA5
        "era5": {"source": "era5", "pooling": "pooled"},      # pooling effectively irrelevant for ERA5
        # Target = SCENARIO1 forcing
        "scenario1": {"source": "scenario1", "pooling": "pooled"},
        # Target = SCENARIO2 forcing
        "scenario2": {"source": "scenario2", "pooling": "pooled"},
    }
elif set2crunch_token == "100ssp245":
    set2crunch = "ssp245"
    default_scen1_root = _prefer_existing_path(
        _resolve_latest_original_root(
            experiment_id="ssp245",
            arx="AR6",
            runmodus="all",
            n_ensemble="n_100",
        ),
        _CREATED_NC_ROOT / f"{_SSP245_100_LOOP_DIR}/debias/{_DEFAULT_VERSION_TAG}/ssp245/AR6",
    )
    default_scen2_root = _prefer_existing_path(
        _resolve_latest_original_root(
            experiment_id="ssp245",
            arx="AR6",
            runmodus="nat",
            n_ensemble="n_100",
        ),
        _CREATED_NC_ROOT / f"{_SSP245_100_LOOP_DIR}/debias/{_DEFAULT_VERSION_TAG}/ssp245/AR6",
    )
    DEFAULT_GCMAGICC_SCENARIO1_ROOT = Path(
        os.environ.get(
            "GCMAGICC_754_SCENARIO1_ROOT",
            str(default_scen1_root),
        )
    ).expanduser().resolve(strict=False)
    DEFAULT_GCMAGICC_SCENARIO2_ROOT = Path(
        os.environ.get(
            "GCMAGICC_754_SCENARIO2_ROOT",
            str(default_scen2_root),
        )
    ).expanduser().resolve(strict=False)

    # Keep only files ending with -nat for Scenario2 when using shared ssp245 root.
    DEFAULT_GCMAGICC_SCENARIO2_SUFFIX = "-nat"
    BASELINE_CONFIG: Dict[str, Dict[str, str]] = {
        # Target = ERA5
        "era5": {"source": "era5", "pooling": "pooled"},      # pooling effectively irrelevant for ERA5
        # Target = SCENARIO1 forcing
        "scenario1": {"source": "scenario1", "pooling": "pooled"},
        # Target = SCENARIO2 forcing
        "scenario2": {"source": "scenario1", "pooling": "pooled"},
        }

elif set2crunch_token == "ssp245":
    set2crunch = "ssp245"
    default_scen1_root = _resolve_latest_original_root(
        experiment_id="ssp245",
        arx="AR6",
        runmodus="all",
        n_ensemble="n_20",
    )
    default_scen2_root = _resolve_latest_original_root(
        experiment_id="ssp245",
        arx="AR6",
        runmodus="nat",
        n_ensemble="n_20",
    )
    DEFAULT_GCMAGICC_SCENARIO1_ROOT = Path(
        os.environ.get(
            "GCMAGICC_754_SCENARIO1_ROOT",
            str(default_scen1_root),
        )
    ).expanduser().resolve(strict=False)
    DEFAULT_GCMAGICC_SCENARIO2_ROOT = Path(
        os.environ.get(
            "GCMAGICC_754_SCENARIO2_ROOT",
            str(default_scen2_root),
        )
    ).expanduser().resolve(strict=False)

    # Keep only files ending with -nat for Scenario2 when using shared ssp245 root.
    DEFAULT_GCMAGICC_SCENARIO2_SUFFIX = "-nat"

    BASELINE_CONFIG: Dict[str, Dict[str, str]] = {
        # Target = ERA5
        "era5": {"source": "era5", "pooling": "pooled"},      # pooling effectively irrelevant for ERA5
        # Target = SCENARIO1 forcing
        "scenario1": {"source": "scenario1", "pooling": "pooled"},
        # Target = SCENARIO2 forcing
        "scenario2": {"source": "scenario1", "pooling": "pooled"},
    }
elif set2crunch_token == "curpol_lowndc":
    set2crunch = "CurPol_lowNDC"
    # Scenario2: low-NDC setup (separate root, no "-nat" suffix filtering)
    default_scen2_root = _resolve_latest_original_root(
        experiment_id="NDC-submitted-low",
        arx="AR6",
        runmodus="all",
        n_ensemble="n_20",
    )
    DEFAULT_GCMAGICC_SCENARIO2_ROOT = Path(
        os.environ.get(
            "GCMAGICC_754_SCENARIO2_ROOT",
            str(default_scen2_root),
        )
    ).expanduser().resolve(strict=False)
    # Scenario1: current-policies setup (separate root)
    default_scen1_root = _resolve_latest_original_root(
        experiment_id="Current-Policies-GCAM",
        arx="AR6",
        runmodus="all",
        n_ensemble="n_20",
    )
    DEFAULT_GCMAGICC_SCENARIO1_ROOT = Path(
        os.environ.get(
            "GCMAGICC_754_SCENARIO1_ROOT",
            str(default_scen1_root),
        )
    ).expanduser().resolve(strict=False)
    # Distinct SCENARIO2 root: no suffix-based filtering.
    DEFAULT_GCMAGICC_SCENARIO2_SUFFIX = None

    BASELINE_CONFIG: Dict[str, Dict[str, str]] = {
        # Target = ERA5
        "era5": {"source": "era5", "pooling": "pooled"},      # pooling effectively irrelevant for ERA5
        # Target = SCENARIO1 forcing
        "scenario1": {"source": "scenario1", "pooling": "pooled"},
        # Target = SCENARIO2 forcing
        "scenario2": {"source": "scenario2", "pooling": "pooled"},
    }
elif set2crunch_token == "100curpol_lowndc":
    set2crunch = "100CurPol_lowNDC"
    # Scenario2: low-NDC setup (separate root, no "-nat" suffix filtering)
    default_scen2_root = _prefer_existing_path(
        _resolve_latest_original_root(
            experiment_id="NDC-submitted-low",
            arx="AR6",
            runmodus="all",
            n_ensemble="n_100",
        ),
        # Pragmatic fallback for single-scenario runs when 100-member low-NDC
        # files are unavailable in this environment.
        _resolve_latest_original_root(
            experiment_id="Current-Policies-GCAM",
            arx="AR6",
            runmodus="all",
            n_ensemble="n_100",
        ),
    )
    DEFAULT_GCMAGICC_SCENARIO2_ROOT = Path(
        os.environ.get(
            "GCMAGICC_754_SCENARIO2_ROOT",
            str(default_scen2_root),
        )
    ).expanduser().resolve(strict=False)
    # Scenario1: current-policies setup (separate root)
    default_scen1_root = _resolve_latest_original_root(
        experiment_id="Current-Policies-GCAM",
        arx="AR6",
        runmodus="all",
        n_ensemble="n_100",
    )
    DEFAULT_GCMAGICC_SCENARIO1_ROOT = Path(
        os.environ.get(
            "GCMAGICC_754_SCENARIO1_ROOT",
            str(default_scen1_root),
        )
    ).expanduser().resolve(strict=False)
    # Distinct SCENARIO2 root: no suffix-based filtering.
    DEFAULT_GCMAGICC_SCENARIO2_SUFFIX = None

    BASELINE_CONFIG: Dict[str, Dict[str, str]] = {
        # Target = ERA5
        "era5": {"source": "era5", "pooling": "pooled"},      # pooling effectively irrelevant for ERA5
        # Target = SCENARIO1 forcing
        "scenario1": {"source": "scenario1", "pooling": "pooled"},
        # Target = SCENARIO2 forcing
        "scenario2": {"source": "scenario2", "pooling": "pooled"},
    }

else:
    raise ValueError(
        f"Unsupported set2crunch value '{set2crunch}'. "
        "Use one of: 'ssp245', 'NDCs', 'CurPol_lowNDC' "
        "(or set GCMAGICC_754_SET2CRUNCH)."
    )


# DEFAULT_GCMAGICC_SCENARIO2_ROOT = Path(
#     "data/site_eth/GCMAGICCoutput/ERA5splicedS3/"
#     "v101/ssp245/AR6/nat/n_20/original/run_<latest_verified>"
# )
DEFAULT_ERA5_FILE = Path(
    os.environ.get("GCMAGICC_ERA5_FILE", str(get_era5_main_file()))
).expanduser().resolve(strict=False)

DEFAULT_CMIP6_ROOT = Path(
    os.environ.get("GCMAGICC_CMIP6_ROOT", str(get_cmip6_vetted_path()))
).expanduser().resolve(strict=False)
DEFAULT_CMIP6_EXPERIMENTS = "historical,hist-nat,ssp245"
CMIP6_EXPERIMENT_CHOICES = ("historical", "hist-nat", "ssp245")
DEFAULT_INCLUDE_CMIP6 = False
DEFAULT_CMIP6_ONLY = False
# Optional limiter for non-CMIP6 forcing processing:
#   None/"none" -> run both scenario forcings (and ERA5 forcing output)
#   "scenario1" -> run SCENARIO1 forcing only
#   "scenario2" -> run SCENARIO2 forcing only
LIMIT_TO_SCENARIO = os.environ.get("GCMAGICC_754_LIMIT_TO_SCENARIO", "none")



DEFAULT_REGION = "C.North-America"
DEFAULT_SCALE_MONTHS = 48
DEFAULT_PET_METHOD = "thornthwaite"          # auto|thornthwaite|hargreaves|penman-monteith|all
DEFAULT_SPEI_FIT = "loglogistic"     # zscore|loglogistic
# Requested default: keep enabled unless explicitly turned off via CLI.
APPLY_RSDS_TO_ERA5_BIASADJUSTMENT = True
# NAT-scenario handling modes for RSDS ERA5 bias adjustment:
# - 'excempt': do not apply RSDS bias adjustment to *-nat files
# - 'full'   : apply full time-varying RSDS bias adjustment to *-nat files
# - 'early'  : apply month/point-specific first-ERA5-year offset to all years in *-nat files
RSDSBIASADJUST_NAT_SCENS = "early"
RSDSBIASADJUST_NAT_SCENS_CHOICES = ("excempt", "full", "early")
RSDSBIASADJUST_NAT_SCENARIO_SUFFIX = "-nat"
DEFAULT_RSDS_BIAS_SMOOTHING_YEARS = 21
DEFAULT_RSDS_BIAS_EDGE_TREND_YEARS = 10
DEFAULT_RSDS_BIAS_EDGE_EXTENSION_YEARS = 11
# Output window defaults: "all years" unless explicitly constrained.
DEFAULT_OUT_START_YEAR: Optional[int] = None
DEFAULT_OUT_END_YEAR: Optional[int] = None

# Requested default SPEI baseline climatology period
DEFAULT_BASELINE_START_YEAR = 1991
DEFAULT_BASELINE_END_YEAR = 2010

# choose 1950 to 2010 for the SPEI Drought Monitor consistency, see here: https://spei.csic.es/map/maps.html
# Choose 1991-2010 for the ERA5-Drought indicator consistency, see here: https://cds.climate.copernicus.eu/datasets/derived-drought-historical-monthly?tab=overview


# -----------------------------------------------------------------------------
# Baseline configuration (edit here for runs without CLI args)
#
# Requested defaults:
#   - ERA5 uses a self-baseline over the baseline period
#   - SCENARIO2 and SCENARIO1 both standardize using the same baseline: SCENARIO1 + pooled
# -----------------------------------------------------------------------------


BASELINE_SOURCE_CHOICES = ("era5", "scenario1", "scenario2")
BASELINE_POOLING_CHOICES = ("pooled", "per_member")

# CMIP6 baseline defaults (requested):
# - CMIP6 historical uses CMIP6 historical baseline per source/member
# - CMIP6 hist-nat uses CMIP6 historical baseline per source/member
# - CMIP6 ssp245 uses CMIP6 historical baseline per source/member
CMIP6_BASELINE_CONFIG: Dict[str, Dict[str, str]] = {
    "cmip6_hist": {"source": "historical", "pooling": "per_member"},
    "cmip6_hist_nat": {"source": "historical", "pooling": "per_member"},
    "cmip6_ssp245": {"source": "historical", "pooling": "per_member"},
}
CMIP6_BASELINE_SOURCE_CHOICES = ("historical", "hist-nat", "ssp245", "self")
CMIP6_BASELINE_POOLING_CHOICES = ("per_member", "pooled")

FORCING_SCENARIO1_LABEL = "SCENARIO1"
FORCING_SCENARIO2_LABEL = "SCENARIO2"

DEFAULT_ON_EXISTING = "archive"       # prompt|overwrite|archive|skip|throwerror
DEFAULT_PIVOT_YEAR = 2025  # grayscale->purple transition year in PDFs
DEFAULT_GROUP_PIXELS = 1  # 1=per-point; 9=3x3 mean; 25=5x5 mean (centered)
_ALLOWED_GROUP_PIXELS = (1, 9, 25)
SEGMENTS_LAYOUT_PER_RUN = "per_run"
SEGMENTS_LAYOUT_RUN_STACKED = "run_stacked"
SEGMENTS_LAYOUT_DUAL = "dual"
SEGMENTS_LAYOUT_CHOICES = (
    SEGMENTS_LAYOUT_PER_RUN,
    SEGMENTS_LAYOUT_RUN_STACKED,
    SEGMENTS_LAYOUT_DUAL,
)
DEFAULT_SEGMENTS_LAYOUT = SEGMENTS_LAYOUT_RUN_STACKED
DEFAULT_STACKED_CHUNK_RUN = 32
DEFAULT_STACKED_CHUNK_TIME = 240
DEFAULT_STACKED_CHUNK_POINT = "auto"
DEFAULT_STACKED_COMPRESSION_LEVEL = 3
DEFAULT_STACKED_CONSOLIDATE_METADATA = True
OUTPUT_FORMAT_ZARR = "zarr"
OUTPUT_FORMAT_NETCDF = "netcdf"
OUTPUT_FORMAT_CHOICES = (
    OUTPUT_FORMAT_ZARR,
    OUTPUT_FORMAT_NETCDF,
)
DEFAULT_OUTPUT_FORMAT = OUTPUT_FORMAT_ZARR

# Inset-map styling (requested: "80% transparent white fill" => alpha=0.20)
REGION_INSET_FILL_ALPHA = 0.20

# Minimum samples for log-logistic fitting per month/point
LOGLOG_MIN_SAMPLES = 4


def _region_subdir(region: str) -> str:
    """
    Return the region-specific subdirectory name used for SPEIx outputs.
    Example: 'C.North-America' -> 'region-C_North-America'
    """
    safe = region.strip().replace(" ", "_").replace(".", "_")
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return f"region-{safe}"


def _region_mask_token(region: str) -> str:
    """
    Return filesystem-safe region token used by precomputed mask filenames.
    Keep existing conventions and additionally normalize '/' to '_'.
    """
    return region.upper().replace(" ", "_").replace("/", "_")


REGION_ALIASES = {
    "Greenland.Iceland": "Greenland/Iceland",
    "Greenland_Iceland": "Greenland/Iceland",
}


def _normalize_region_name(region: str) -> str:
    raw = str(region).strip()
    if not raw:
        return raw
    return REGION_ALIASES.get(raw, raw)


# US Drought Monitor drought classes for SPI/SPEI
# None:   -0.49 or above
# D0:     -0.5  to -0.79
# D1:     -0.8  to -1.29
# D2:     -1.3  to -1.59
# D3:     -1.6  to -1.99
# D4:     -2.0  or less
USDM_SPEI_LEVELS = (-2.0, -1.6, -1.3, -0.8, -0.5)

# US Drought Monitor drought class labels (from driest to wettest)
USDM_SPEI_LABELS = (
    "D4 - exceptional",
    "D3 - extreme",
    "D2 - severe",
    "D1 - moderate",
    "D0 - abnormally dry",
    "Normal",
)


# -----------------------------------------------------------------------------
# Repository path + recipe imports
# -----------------------------------------------------------------------------
def get_project_root() -> Path:
    cur = Path(__file__).resolve().parent
    if cur.name == "notebooks":
        return cur.parent
    if (cur / "data").exists() and (cur / "notebooks").exists():
        return cur
    if "notebooks" in cur.parts:
        idx = cur.parts.index("notebooks")
        return Path(*cur.parts[:idx])
    if (cur / "data").exists():
        return cur
    return cur


PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from gcmagicc_eval.recipes.SPEIx import (  # type: ignore
        VAR_PR,
        VAR_RSDS,
        VAR_SFCWIND,
        VAR_HURS,
        VAR_PS,
        VAR_PSL,
        VAR_TAS,
        VAR_TASMAX,
        VAR_TASMIN,
        _resolve_pet_method as recipe_resolve_pet_method,
        _wb_monthly as recipe_wb_monthly,
        _rolling_sum as recipe_rolling_sum,
    )
except Exception as exc:  # pragma: no cover - fail fast if recipe missing
    raise ImportError(
        "Failed to import gcmagicc_eval.recipes.SPEIx. Install this standalone release "
        "or place its src directory on PYTHONPATH."
    ) from exc


# -----------------------------------------------------------------------------
# Small containers
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TemplatePoints:
    """Defines the target (lat, lon) point set used for fitting and standardizing."""
    lat: np.ndarray  # (points,)
    lon: np.ndarray  # (points,) in 0..360 convention
    region: str


@dataclass(frozen=True)
class RsdsEra5BiasAdjustment:
    """Monthly/yearly rsds offsets that align GCMAGICC to ERA5 at template points."""
    years: np.ndarray  # (n_years,)
    era5_raw: np.ndarray  # (12, n_years, point)
    gcmagicc_raw: np.ndarray  # (12, n_years, point)
    era5_smooth: np.ndarray  # (12, n_years, point)
    gcmagicc_smooth: np.ndarray  # (12, n_years, point)
    offsets: np.ndarray  # (12, n_years, point), filled for all years in years[]
    smoothing_window_years: int
    n_gcm_files_used: int


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(msg, flush=True)


# -----------------------------------------------------------------------------
# Safe xarray open helper (avoids netCDF4 segfaults by preferring h5netcdf)
# -----------------------------------------------------------------------------
DEFAULT_XR_ENGINE = "h5netcdf"  # fallback to netcdf4 if unavailable


def _parse_shell_env_assignment(line: str) -> tuple[str, str] | None:
    """
    Parse simple shell assignment rows like:
      KEY=value
      export KEY=value
    """
    text = str(line or "").strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[len("export "):].strip()
    if "=" not in text:
        return None
    key, value = text.split("=", 1)
    key = key.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return None
    value = value.strip()
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        value = value[1:-1]
    return key, value


def _load_s3_env_file(path: Path | str | None) -> None:
    """
    Load only allowed S3-related env vars from an env file.
    Existing process env values take precedence.
    """
    if path is None:
        return
    env_path = Path(path).expanduser().resolve(strict=False)
    if not env_path.exists():
        _log(f"⚠️ S3 env file not found (skipping): {env_path}")
        return
    if not env_path.is_file():
        _log(f"⚠️ S3 env path is not a file (skipping): {env_path}")
        return

    loaded = 0
    skipped = 0
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = _parse_shell_env_assignment(raw)
        if parsed is None:
            continue
        key, value = parsed
        allowed = key in _S3_ENV_ALLOWED_EXACT or any(
            key.startswith(prefix) for prefix in _S3_ENV_ALLOWED_PREFIXES
        )
        if not allowed:
            skipped += 1
            continue
        current = os.environ.get(key)
        if current is not None and str(current).strip():
            # Process env wins over file-provided defaults.
            skipped += 1
            continue
        os.environ[key] = value
        loaded += 1
    _log(f"S3 env load: loaded={loaded}, skipped={skipped} from {env_path}")


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


def _cache_key_for_input(local_path: Path, source_ref: str) -> Path:
    digest = hashlib.sha1(source_ref.encode("utf-8")).hexdigest()
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", local_path.name).strip("-._") or "dataset.nc"
    return _ACTIVE_RCLONE_CACHE_ROOT / digest[:2] / digest / safe_name


def _ensure_rclone_cached_input(path: Path | str) -> Path:
    local_path = Path(path).expanduser().resolve(strict=False)
    if _looks_like_derivative_path(local_path):
        return local_path
    if not path_uses_rclone_mount(local_path, resolve_path=False):
        return local_path

    source_ref = resolve_rclone_source_ref(local_path, resolve_path=False)
    if not source_ref:
        return local_path

    cache_path = _cache_key_for_input(local_path, source_ref)
    if cache_path.exists():
        try:
            if cache_path.stat().st_size > 0:
                return cache_path
        except OSError:
            pass
        try:
            cache_path.unlink()
        except OSError:
            pass

    _log(f"    [rclone_cache] caching {local_path.name} -> {cache_path}")
    copy_remote_file_atomic_via_rclone(source_ref, cache_path)
    return cache_path


def _is_original_source_nc_path(path: Path) -> bool:
    """
    Restrict s3_direct reads to large original-source .nc files.
    """
    if _looks_like_derivative_path(path):
        return False
    if path.suffix.lower() != ".nc":
        return False
    return "original" in {part.lower() for part in path.parts}


def _get_s3_filesystem():
    global _S3_FS
    if _S3_FS is not None:
        return _S3_FS
    try:
        import s3fs  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "storage-access=s3_direct requires the 's3fs' package in the environment."
        ) from exc
    opts = _s3_storage_options()
    _S3_FS = s3fs.S3FileSystem(
        anon=bool(opts.get("anon", False)),
        client_kwargs=opts.get("client_kwargs", {}),
        config_kwargs=opts.get("config_kwargs", {}),
    )
    return _S3_FS


def _resolve_dataset_open_target(path: Path | str) -> tuple[Path | str, Dict[str, object] | None]:
    local_path = Path(path).expanduser().resolve(strict=False)
    if _ACTIVE_STORAGE_ACCESS == STORAGE_ACCESS_RCLONE_CACHE:
        return _ensure_rclone_cached_input(local_path), None
    if _ACTIVE_STORAGE_ACCESS != STORAGE_ACCESS_S3_DIRECT:
        return local_path, None
    if not _is_original_source_nc_path(local_path):
        return local_path, None
    local_key = str(local_path)
    if local_key in _ACTIVE_S3_LOCAL_FALLBACK_PATHS:
        return local_path, None
    selected_uri = _ACTIVE_S3_SELECTED_URI_BY_LOCAL.get(local_key)
    if selected_uri:
        return selected_uri, _s3_storage_options()
    candidates = convert_local_path_to_s3_uri_candidates(local_path)
    if not candidates:
        return local_path, None
    return candidates[0], _s3_storage_options()


def _open_dataset_local_with_engines(
    path: Path | str,
    *,
    decode_times: bool,
    engine_attempts: Sequence[str],
) -> xr.Dataset:
    errors: List[Exception] = []
    for engine_name in engine_attempts:
        try:
            return xr.open_dataset(path, decode_times=decode_times, engine=engine_name)
        except Exception as exc:
            errors.append(exc)
    raise errors[-1]


def _stage_s3_object_to_local_cache(s3_uri: str, *, stage_dir: Path) -> Path:
    if not str(s3_uri).startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {s3_uri}")
    fs = _get_s3_filesystem()
    stage_dir = Path(stage_dir).expanduser().resolve(strict=False)
    stage_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(s3_uri).encode("utf-8")).hexdigest()
    ext = Path(s3_uri).suffix if Path(s3_uri).suffix.lower() == ".nc" else ".nc"
    local_path = (stage_dir / digest[:2] / f"{digest}{ext}").resolve(strict=False)
    lock_path = local_path.with_suffix(local_path.suffix + ".lock")
    local_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            if local_path.exists() and local_path.stat().st_size > 0:
                return local_path
            part_path = local_path.with_name(local_path.name + ".part")
            if part_path.exists():
                try:
                    part_path.unlink()
                except Exception:
                    pass
            remote_path = str(s3_uri)[len("s3://") :]
            with fs.open(remote_path, "rb") as src, part_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
            os.replace(part_path, local_path)
            return local_path
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _preflight_s3_objects(paths: Sequence[Path], *, label: str) -> set[str]:
    global _ACTIVE_S3_SELECTED_URI_BY_LOCAL
    global _ACTIVE_S3_PREFLIGHT_COUNTS
    if _ACTIVE_STORAGE_ACCESS != STORAGE_ACCESS_S3_DIRECT:
        return set()
    if not _ACTIVE_S3_PREFLIGHT:
        return set()
    selected: List[Path] = []
    seen_local: set[str] = set()
    for raw in paths:
        p = Path(raw).expanduser().resolve(strict=False)
        if not _is_original_source_nc_path(p):
            continue
        key = str(p)
        if key in seen_local:
            continue
        seen_local.add(key)
        selected.append(p)
    if not selected:
        return set()

    fs = _get_s3_filesystem()
    missing_map: List[str] = []
    unmapped: List[str] = []
    hard_missing: List[str] = []
    local_fallback_paths: set[str] = set()
    selected_uri_by_local: Dict[str, str] = {}
    exists_cache: Dict[str, bool] = {}
    prefix_counts: Dict[str, int] = {}
    total = len(selected)
    counts: Dict[str, int] = {
        "total": total,
        "s3_direct": 0,
        "local_fallback": 0,
        "hard_missing": 0,
        "unmapped": 0,
    }

    def _prefix_bucket(uri: str) -> str:
        if "/nc/consolidated/era5spliced/" in uri:
            return "consolidated-era5spliced"
        if "/created_nc_files/" in uri:
            return "created-nc"
        if "/era5spliced/" in uri:
            return "site-era5spliced"
        return "other"

    for idx, local_path in enumerate(selected, start=1):
        local_key = str(local_path)
        candidates = convert_local_path_to_s3_uri_candidates(local_path)
        if not candidates:
            unmapped.append(str(local_path))
            counts["unmapped"] += 1
            if local_path.exists():
                local_fallback_paths.add(local_key)
                counts["local_fallback"] += 1
            else:
                hard_missing.append(str(local_path))
                counts["hard_missing"] += 1
            continue
        chosen_uri: Optional[str] = None
        for uri in candidates:
            remote_path = str(uri)[len("s3://") :]
            if remote_path in exists_cache:
                exists = exists_cache[remote_path]
            else:
                try:
                    exists = bool(fs.exists(remote_path))
                except Exception as exc:
                    raise RuntimeError(f"S3 preflight failed for {uri}: {type(exc).__name__}: {exc}") from exc
                exists_cache[remote_path] = exists
            if exists:
                chosen_uri = uri
                break

        if chosen_uri is None:
            if local_path.exists():
                missing_map.append(f"{candidates[0]} <- {local_path}")
                local_fallback_paths.add(local_key)
                counts["local_fallback"] += 1
            else:
                preview = ", ".join(candidates[:3])
                hard_missing.append(f"{local_path} (candidates: {preview})")
                counts["hard_missing"] += 1
        else:
            selected_uri_by_local[local_key] = chosen_uri
            counts["s3_direct"] += 1
            bucket = _prefix_bucket(chosen_uri)
            prefix_counts[bucket] = int(prefix_counts.get(bucket, 0) + 1)

        if idx % 200 == 0:
            _log(f"S3 preflight progress ({label}): {idx}/{total}")

    _ACTIVE_S3_SELECTED_URI_BY_LOCAL = selected_uri_by_local
    _ACTIVE_S3_PREFLIGHT_COUNTS = counts

    if unmapped:
        preview = "\n".join(f"  - {row}" for row in sorted(unmapped)[:10])
        extra = f"\n  ... (+{len(unmapped) - 10} more)" if len(unmapped) > 10 else ""
        _log(
            "⚠️ S3 preflight: unable to map some local source files to s3:// URIs; "
            "falling back to local reads for these files.\n"
            f"Label: {label}\n"
            f"Unmapped files: {len(unmapped)}\n{preview}{extra}"
        )
    if missing_map:
        preview = "\n".join(f"  - {row}" for row in missing_map[:15])
        extra = f"\n  ... (+{len(missing_map) - 15} more)" if len(missing_map) > 15 else ""
        _log(
            "⚠️ S3 preflight: required source objects are missing on S3; "
            "falling back to local reads for those files.\n"
            f"Label: {label}\n"
            f"Missing objects: {len(missing_map)} / {total}\n{preview}{extra}"
        )
    if hard_missing:
        preview = "\n".join(f"  - {row}" for row in hard_missing[:15])
        extra = f"\n  ... (+{len(hard_missing) - 15} more)" if len(hard_missing) > 15 else ""
        raise RuntimeError(
            "S3 preflight failed: source file unavailable both on S3 and locally.\n"
            f"Label: {label}\n"
            f"Hard-missing: {len(hard_missing)} / {total}\n{preview}{extra}"
        )

    if prefix_counts:
        ordered = ", ".join(f"{k}={v}" for k, v in sorted(prefix_counts.items()))
        _log(f"S3 URI source selection ({label}): {ordered}")

    _log(
        f"S3 preflight completed ({label}): s3_direct={counts['s3_direct']}, "
        f"local_fallback={counts['local_fallback']}, unmapped={counts['unmapped']}, total={total}"
    )
    return local_fallback_paths


def _open_dataset_safe(path: Path | str, *, decode_times: bool = True, engine: Optional[str] = None) -> xr.Dataset:
    local_path = Path(path).expanduser().resolve(strict=False)
    target, _storage_options = _resolve_dataset_open_target(path)
    eng = engine or DEFAULT_XR_ENGINE
    is_s3_target = isinstance(target, str) and target.startswith("s3://")

    engine_attempts = [eng]
    if eng != "netcdf4":
        engine_attempts.append("netcdf4")

    if is_s3_target:
        errors: List[Exception] = []
        try:
            staged_path = _stage_s3_object_to_local_cache(str(target), stage_dir=_ACTIVE_S3_STAGE_DIR)
            _log(f"S3 staged local read: {staged_path.name}")
            return _open_dataset_local_with_engines(
                staged_path,
                decode_times=decode_times,
                engine_attempts=engine_attempts,
            )
        except Exception as stage_exc:
            errors.append(stage_exc)
            if local_path.exists():
                _log(
                    "⚠️ S3 read failed and staging fallback failed; "
                    f"retrying from local source path: {local_path}"
                )
                return _open_dataset_local_with_engines(
                    local_path,
                    decode_times=decode_times,
                    engine_attempts=engine_attempts,
                )
            raise RuntimeError(
                "Failed to open NetCDF via s3_direct and local staging fallback.\n"
                f"Local source: {local_path}\n"
                f"S3 source: {target}\n"
                f"Last error: {type(errors[-1]).__name__}: {errors[-1]}"
            ) from errors[-1]

    return _open_dataset_local_with_engines(
        local_path,
        decode_times=decode_times,
        engine_attempts=engine_attempts,
    )


def _to_datetime64(arr: np.ndarray) -> Optional[np.ndarray]:
    """Best-effort conversion to numpy datetime64[ns]."""
    try:
        return np.asarray(arr).astype("datetime64[ns]")
    except Exception:
        try:
            return np.asarray(arr)
        except Exception:
            return None


def _complete_years_from_time_arrays(years: np.ndarray, months: np.ndarray) -> np.ndarray:
    """
    Return years that contain all 12 calendar months exactly once.
    Used to cut off incomplete edge years in ERA5 monthly series.
    """
    y = np.asarray(years, dtype=int).ravel()
    m = np.asarray(months, dtype=int).ravel()
    if y.size == 0 or m.size == 0 or y.size != m.size:
        return np.array([], dtype=int)
    complete: List[int] = []
    for yy in np.unique(y):
        mm = m[y == yy]
        if mm.size == 12 and np.array_equal(np.sort(mm), np.arange(1, 13, dtype=int)):
            complete.append(int(yy))
    return np.asarray(complete, dtype=int)


def _rolling_mean_centered_nan(
    arr: np.ndarray,
    window: int,
    *,
    edge_trend_years: int = DEFAULT_RSDS_BIAS_EDGE_TREND_YEARS,
    edge_extension_years: int = DEFAULT_RSDS_BIAS_EDGE_EXTENSION_YEARS,
) -> np.ndarray:
    """
    Centered rolling mean along axis=0 with NaN-safe averaging and trend-preserving edges.

    Edge treatment (requested):
      1) Fit a linear trend from the first/last `edge_trend_years` years.
      2) Extrapolate each edge by `edge_extension_years` years.
      3) Apply a full-width centered window over the padded series and return only
         the original-year segment.

    Input shape: (n_years, n_points).
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("Expected 2D array (year, point).")
    n_years, n_points = a.shape
    if n_years == 0:
        return np.empty_like(a)

    win = int(max(1, window))
    if win % 2 == 0:
        win += 1
    half = win // 2

    fit_years = int(max(2, edge_trend_years))
    pad = int(max(0, edge_extension_years))
    if pad < half:
        pad = half

    ext = np.full((n_years + 2 * pad, n_points), np.nan, dtype=np.float64)
    ext[pad : pad + n_years, :] = a

    def _fit_line(xv: np.ndarray, yv: np.ndarray) -> Tuple[float, float]:
        if xv.size >= 2:
            try:
                m, b = np.polyfit(xv.astype(np.float64), yv.astype(np.float64), 1)
                if np.isfinite(m) and np.isfinite(b):
                    return float(m), float(b)
            except Exception:
                pass
        if yv.size >= 1 and np.isfinite(yv[0]):
            return 0.0, float(yv[0])
        return 0.0, np.nan

    x_full = np.arange(n_years, dtype=np.float64)
    x_left_pad = np.arange(-pad, 0, dtype=np.float64)
    x_right_pad = np.arange(n_years, n_years + pad, dtype=np.float64)

    for p in range(n_points):
        y = a[:, p]
        fin = np.isfinite(y)
        if not np.any(fin):
            continue

        left_ix_all = np.arange(0, min(n_years, fit_years), dtype=int)
        left_ix = left_ix_all[np.isfinite(y[left_ix_all])]
        if left_ix.size < 2:
            all_ix = np.where(fin)[0]
            left_ix = all_ix[: min(fit_years, all_ix.size)]
        m_l, b_l = _fit_line(x_full[left_ix], y[left_ix])

        right_ix_all = np.arange(max(0, n_years - fit_years), n_years, dtype=int)
        right_ix = right_ix_all[np.isfinite(y[right_ix_all])]
        if right_ix.size < 2:
            all_ix = np.where(fin)[0]
            right_ix = all_ix[max(0, all_ix.size - fit_years) :]
        m_r, b_r = _fit_line(x_full[right_ix], y[right_ix])

        if np.isfinite(m_l) and np.isfinite(b_l):
            ext[:pad, p] = m_l * x_left_pad + b_l
        else:
            ext[:pad, p] = y[np.where(fin)[0][0]]
        if np.isfinite(m_r) and np.isfinite(b_r):
            ext[pad + n_years :, p] = m_r * x_right_pad + b_r
        else:
            ext[pad + n_years :, p] = y[np.where(fin)[0][-1]]

    finite = np.isfinite(ext)
    vals = np.where(finite, ext, 0.0)
    csum = np.cumsum(vals, axis=0, dtype=np.float64)
    ccnt = np.cumsum(finite.astype(np.int32), axis=0, dtype=np.int32)

    out = np.full((n_years, n_points), np.nan, dtype=np.float64)
    for i in range(n_years):
        center = pad + i
        lo = center - half
        hi = center + half
        s = csum[hi] - (csum[lo - 1] if lo > 0 else 0.0)
        c = ccnt[hi] - (ccnt[lo - 1] if lo > 0 else 0)
        valid = c > 0
        out[i, valid] = s[valid] / c[valid]
    return out


def _fill_year_series_linear_with_edge_hold(arr: np.ndarray) -> np.ndarray:
    """
    Fill NaNs in (year, point) arrays by linear interpolation across years
    and constant extension at the start/end.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("Expected 2D array (year, point).")
    n_years, n_points = a.shape
    x = np.arange(n_years, dtype=float)
    out = np.array(a, copy=True, dtype=np.float64)
    for p in range(n_points):
        y = out[:, p]
        ok = np.isfinite(y)
        if not np.any(ok):
            out[:, p] = 0.0
            continue
        xp = x[ok]
        yp = y[ok]
        out[:, p] = np.interp(x, xp, yp)
    return out


def _build_rsds_era5_bias_adjustment(
    *,
    era5_file: Path,
    gcm_files: Sequence[Path],
    template: TemplatePoints,
    smoothing_window_years: int,
) -> Optional[RsdsEra5BiasAdjustment]:
    """
    Build monthly/yearly rsds offsets:
      offset(month, year, point) = smooth21(ERA5) - smooth21(mean_all_GCMAGICC)
    where ERA5 years are restricted to complete years only.
    """
    if not gcm_files:
        _log("RSDS bias adjustment requested but no GCMAGICC files are available; skipping.")
        return None

    # ERA5 reference (template points only), cut to complete years.
    ds_era5 = _open_dataset_safe(era5_file, decode_times=True)
    try:
        ds_era5 = _ensure_lon_0_360(ds_era5)
        if VAR_RSDS not in ds_era5:
            _log(f"RSDS bias adjustment requested but ERA5 file has no '{VAR_RSDS}' variable; skipping.")
            return None
        era5_sub = _select_template_points(ds_era5, template=template, variables=[VAR_RSDS]).load()
    finally:
        try:
            ds_era5.close()
        except Exception:
            pass

    years_all = np.asarray(era5_sub["time"].dt.year.values, dtype=int)
    months_all = np.asarray(era5_sub["time"].dt.month.values, dtype=int)
    complete_years = _complete_years_from_time_arrays(years_all, months_all)
    if complete_years.size == 0:
        _log("RSDS bias adjustment requested but ERA5 has no complete years; skipping.")
        try:
            era5_sub.close()
        except Exception:
            pass
        return None

    y0 = int(np.min(complete_years))
    y1 = int(np.max(complete_years))
    years_grid = np.arange(y0, y1 + 1, dtype=int)
    n_years = int(years_grid.size)
    n_points = int(template.lat.size)

    def _accumulate_month_year_point(
        arr: np.ndarray,
        years: np.ndarray,
        months: np.ndarray,
        *,
        out_sum: np.ndarray,
        out_cnt: np.ndarray,
    ) -> None:
        if arr.ndim != 2:
            raise ValueError("Expected rsds array shape (time, point).")
        mask = (years >= y0) & (years <= y1) & (months >= 1) & (months <= 12)
        if not np.any(mask):
            return

        arr_sel = np.asarray(arr[mask, :], dtype=np.float64)
        years_sel = np.asarray(years[mask], dtype=int) - y0
        months_sel = np.asarray(months[mask], dtype=int) - 1
        finite_sel = np.isfinite(arr_sel)

        for mi in range(12):
            month_mask = months_sel == mi
            if not np.any(month_mask):
                continue
            arr_month = arr_sel[month_mask, :]
            finite_month = finite_sel[month_mask, :]
            years_month = years_sel[month_mask]
            for yi in np.unique(years_month):
                year_mask = years_month == yi
                finite_year = finite_month[year_mask, :]
                if not np.any(finite_year):
                    continue
                vals_year = arr_month[year_mask, :]
                out_sum[mi, int(yi), :] += np.where(finite_year, vals_year, 0.0).sum(axis=0, dtype=np.float64)
                out_cnt[mi, int(yi), :] += finite_year.sum(axis=0, dtype=np.int32)

    era5_sum = np.zeros((12, n_years, n_points), dtype=np.float64)
    era5_cnt = np.zeros((12, n_years, n_points), dtype=np.int32)
    mask_complete = np.isin(years_all, complete_years)
    _accumulate_month_year_point(
        np.asarray(era5_sub[VAR_RSDS].values, dtype=np.float64)[mask_complete, :],
        years_all[mask_complete],
        months_all[mask_complete],
        out_sum=era5_sum,
        out_cnt=era5_cnt,
    )
    try:
        era5_sub.close()
    except Exception:
        pass

    era5_raw = np.full((12, n_years, n_points), np.nan, dtype=np.float64)
    ok_era = era5_cnt > 0
    era5_raw[ok_era] = era5_sum[ok_era] / era5_cnt[ok_era]

    gcm_sum = np.zeros((12, n_years, n_points), dtype=np.float64)
    gcm_cnt = np.zeros((12, n_years, n_points), dtype=np.int32)
    n_gcm_used = 0

    for idx, path in enumerate(gcm_files, 1):
        try:
            ds = _open_dataset_safe(path, decode_times=True)
            ds = _ensure_lon_0_360(ds)
            if VAR_RSDS not in ds:
                _log(f"  [RSDS BC {idx}/{len(gcm_files)}] skip {path.name} (missing {VAR_RSDS})")
                try:
                    ds.close()
                except Exception:
                    pass
                continue
            sub = _select_template_points(ds, template=template, variables=[VAR_RSDS]).load()
            try:
                ds.close()
            except Exception:
                pass
        except Exception as exc:
            _log(f"  [RSDS BC {idx}/{len(gcm_files)}] skip {path.name} (open/select failed: {exc})")
            continue

        try:
            arr = np.asarray(sub[VAR_RSDS].values, dtype=np.float64)
            years = np.asarray(sub["time"].dt.year.values, dtype=int)
            months = np.asarray(sub["time"].dt.month.values, dtype=int)
            _accumulate_month_year_point(arr, years, months, out_sum=gcm_sum, out_cnt=gcm_cnt)
            n_gcm_used += 1
        finally:
            try:
                sub.close()
            except Exception:
                pass

    if n_gcm_used == 0:
        _log("RSDS bias adjustment requested but no usable GCMAGICC files contained rsds; skipping.")
        return None

    gcm_raw = np.full((12, n_years, n_points), np.nan, dtype=np.float64)
    ok_gcm = gcm_cnt > 0
    gcm_raw[ok_gcm] = gcm_sum[ok_gcm] / gcm_cnt[ok_gcm]

    window = int(max(1, smoothing_window_years))
    if window % 2 == 0:
        window += 1
    era5_smooth = np.full_like(era5_raw, np.nan, dtype=np.float64)
    gcm_smooth = np.full_like(gcm_raw, np.nan, dtype=np.float64)
    offsets = np.full_like(era5_raw, np.nan, dtype=np.float64)
    for mi in range(12):
        era5_smooth[mi, :, :] = _rolling_mean_centered_nan(
            era5_raw[mi, :, :],
            window=window,
            edge_trend_years=DEFAULT_RSDS_BIAS_EDGE_TREND_YEARS,
            edge_extension_years=DEFAULT_RSDS_BIAS_EDGE_EXTENSION_YEARS,
        )
        gcm_smooth[mi, :, :] = _rolling_mean_centered_nan(
            gcm_raw[mi, :, :],
            window=window,
            edge_trend_years=DEFAULT_RSDS_BIAS_EDGE_TREND_YEARS,
            edge_extension_years=DEFAULT_RSDS_BIAS_EDGE_EXTENSION_YEARS,
        )
        offsets[mi, :, :] = _fill_year_series_linear_with_edge_hold(
            era5_smooth[mi, :, :] - gcm_smooth[mi, :, :]
        )

    _log(
        "RSDS ERA5 bias-adjustment prepared: "
        f"years={y0}-{y1} (complete ERA5 years), "
        f"smoothing={window}y, edge-trend={DEFAULT_RSDS_BIAS_EDGE_TREND_YEARS}y, "
        f"edge-extension={DEFAULT_RSDS_BIAS_EDGE_EXTENSION_YEARS}y, "
        f"gcm_files_used={n_gcm_used}, points={n_points}"
    )
    return RsdsEra5BiasAdjustment(
        years=years_grid,
        era5_raw=era5_raw.astype(np.float32),
        gcmagicc_raw=gcm_raw.astype(np.float32),
        era5_smooth=era5_smooth.astype(np.float32),
        gcmagicc_smooth=gcm_smooth.astype(np.float32),
        offsets=offsets.astype(np.float32),
        smoothing_window_years=window,
        n_gcm_files_used=n_gcm_used,
    )


def _apply_rsds_bias_adjustment_to_subdataset(
    sub: xr.Dataset,
    *,
    adjustment: Optional[RsdsEra5BiasAdjustment],
    hold_first_reference_year_offsets: bool = False,
) -> xr.Dataset:
    """Apply precomputed ERA5-aligned rsds offsets to a template-point dataset."""
    if adjustment is None or VAR_RSDS not in sub:
        return sub
    if "time" not in sub.dims or "point" not in sub.dims:
        return sub

    years = np.asarray(sub["time"].dt.year.values, dtype=int)
    months = np.asarray(sub["time"].dt.month.values, dtype=int)
    if years.size == 0:
        return sub

    y0 = int(adjustment.years[0])
    y1 = int(adjustment.years[-1])
    if bool(hold_first_reference_year_offsets):
        y_idx = np.zeros_like(years, dtype=int)
    else:
        y_idx = np.clip(years, y0, y1) - y0
    m_idx = np.clip(months, 1, 12) - 1

    offs = np.asarray(adjustment.offsets[m_idx, y_idx, :], dtype=np.float64)
    rsds = np.asarray(sub[VAR_RSDS].values, dtype=np.float64)
    if rsds.shape != offs.shape:
        raise ValueError(
            f"RSDS bias adjustment shape mismatch: rsds={rsds.shape}, offsets={offs.shape}"
        )

    rsds_adj = (rsds + offs).astype(np.float32)
    da = xr.DataArray(
        rsds_adj,
        coords=sub[VAR_RSDS].coords,
        dims=sub[VAR_RSDS].dims,
        attrs=dict(sub[VAR_RSDS].attrs),
        name=VAR_RSDS,
    )
    da.attrs["rsds_bias_adjusted_to_era5"] = "true"
    da.attrs["rsds_bias_adjustment_window_years"] = int(adjustment.smoothing_window_years)
    da.attrs["rsds_bias_adjustment_reference_year_start"] = int(adjustment.years[0])
    da.attrs["rsds_bias_adjustment_reference_year_end"] = int(adjustment.years[-1])
    da.attrs["rsds_bias_adjustment_nat_mode"] = (
        "early" if bool(hold_first_reference_year_offsets) else "full"
    )

    out = sub.copy(deep=False)
    out[VAR_RSDS] = da
    return out


def _write_rsds_bias_adjustment_artifacts(
    *,
    root_for_outputs: Path,
    output_tag: Optional[str],
    region: str,
    template: TemplatePoints,
    adjustment: RsdsEra5BiasAdjustment,
) -> Tuple[Path, Optional[Path]]:
    """Persist rsds bias-adjustment fields and a compact diagnostics PDF."""
    month = np.arange(1, 13, dtype=int)
    year = np.asarray(adjustment.years, dtype=int)
    point = np.arange(template.lat.size, dtype=int)

    ds = xr.Dataset(
        data_vars={
            "offset_rsds": (("month", "year", "point"), np.asarray(adjustment.offsets, dtype=np.float32)),
            "era5_rsds_raw": (("month", "year", "point"), np.asarray(adjustment.era5_raw, dtype=np.float32)),
            "gcmagicc_rsds_raw": (("month", "year", "point"), np.asarray(adjustment.gcmagicc_raw, dtype=np.float32)),
            "era5_rsds_smooth": (("month", "year", "point"), np.asarray(adjustment.era5_smooth, dtype=np.float32)),
            "gcmagicc_rsds_smooth": (("month", "year", "point"), np.asarray(adjustment.gcmagicc_smooth, dtype=np.float32)),
        },
        coords={
            "month": ("month", month),
            "year": ("year", year),
            "point": ("point", point),
            "lat": ("point", np.asarray(template.lat, dtype=float)),
            "lon": ("point", np.asarray(template.lon, dtype=float)),
        },
        attrs={
            "description": "Monthly/yearly rsds ERA5 alignment offsets for GCMAGICC runs.",
            "region": str(region),
            "smoothing_window_years": int(adjustment.smoothing_window_years),
            "edge_trend_years": int(DEFAULT_RSDS_BIAS_EDGE_TREND_YEARS),
            "edge_extension_years": int(DEFAULT_RSDS_BIAS_EDGE_EXTENSION_YEARS),
            "n_gcm_files_used": int(adjustment.n_gcm_files_used),
            "offset_definition": "offset = smooth(ERA5_rsds) - smooth(mean_all_GCMAGICC_rsds)",
            "outside_era5_rule": "Apply first/last ERA5-year offset for years before/after ERA5 complete range.",
            "reference_year_start": int(year[0]) if year.size else "none",
            "reference_year_end": int(year[-1]) if year.size else "none",
        },
    )

    out_nc = (
        _fits_dir(root_for_outputs, output_tag=output_tag, region=region)
        / f"RSDS_ERA5_BIAS_ADJUSTMENT__{region.upper()}__window{int(adjustment.smoothing_window_years)}.nc"
    )
    ds.to_netcdf(out_nc)
    _log(f"Wrote RSDS bias-adjustment factors: {out_nc}")

    out_pdf: Optional[Path] = None
    if plt is not None and PdfPages is not None:
        out_pdf = (
            _summary_pdf_dir(root_for_outputs, output_tag=output_tag)
            / f"RSDS_ERA5_BIAS_ADJUSTMENT__{region.upper()}__window{int(adjustment.smoothing_window_years)}.pdf"
        )
        off = np.asarray(adjustment.offsets, dtype=np.float64)  # (month, year, point)
        era_s = np.asarray(adjustment.era5_smooth, dtype=np.float64)
        gcm_s = np.asarray(adjustment.gcmagicc_smooth, dtype=np.float64)

        with PdfPages(out_pdf) as pp:
            fig, axes = plt.subplots(4, 3, figsize=(15.5, 10.5), sharex=True)
            axes_flat = axes.ravel()
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            for mi in range(12):
                ax = axes_flat[mi]
                y_mean = np.nanmean(off[mi, :, :], axis=1)
                y_p10 = np.nanpercentile(off[mi, :, :], 10, axis=1)
                y_p90 = np.nanpercentile(off[mi, :, :], 90, axis=1)
                ax.fill_between(year, y_p10, y_p90, color="#99c1de", alpha=0.5, linewidth=0.0)
                ax.plot(year, y_mean, color="#0d3b66", linewidth=1.5)
                ax.axhline(0.0, color="#333333", linewidth=0.7, alpha=0.8)
                ax.set_title(month_names[mi], fontsize=10)
                ax.grid(alpha=0.2, linewidth=0.5)
            axes[1, 0].set_ylabel("Offset (ERA5 - GCMAGICC), region mean")
            axes[3, 1].set_xlabel("Year")
            fig.suptitle(
                f"RSDS ERA5 bias-adjustment factors — {region.upper()} "
                f"(smooth={int(adjustment.smoothing_window_years)}y, files={int(adjustment.n_gcm_files_used)})",
                fontsize=13,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pp.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig2, ax2 = plt.subplots(figsize=(13.5, 5.6))
            annual_era = np.nanmean(era_s, axis=(0, 2))
            annual_gcm = np.nanmean(gcm_s, axis=(0, 2))
            annual_off = np.nanmean(off, axis=(0, 2))
            ax2.plot(year, annual_era, color="#111111", linewidth=2.0, label="ERA5 smooth annual mean")
            ax2.plot(year, annual_gcm, color="#a23b72", linewidth=1.8, label="GCMAGICC smooth annual mean")
            ax2b = ax2.twinx()
            ax2b.plot(year, annual_off, color="#2e7d32", linewidth=1.6, alpha=0.9, label="Offset annual mean")
            ax2.set_xlabel("Year")
            ax2.set_ylabel("rsds (ERA5/GCMAGICC smooth)")
            ax2b.set_ylabel("Offset (ERA5 - GCMAGICC)")
            ax2.grid(alpha=0.2, linewidth=0.5)
            h1, l1 = ax2.get_legend_handles_labels()
            h2, l2 = ax2b.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
            ax2.set_title(
                "Annual-mean diagnostics (offset is extended as constant outside ERA5 complete years)",
                fontsize=11,
            )
            fig2.tight_layout()
            pp.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

        _log(f"Wrote RSDS bias-adjustment diagnostics plot: {out_pdf}")

    return out_nc, out_pdf


# -----------------------------------------------------------------------------
# Region helpers (adapted from your 754/758 code)
# -----------------------------------------------------------------------------
def _align_mask_to_grid(
    mask: np.ndarray,
    mask_lats: np.ndarray,
    mask_lons: np.ndarray,
    data_lats: np.ndarray,
    data_lons: np.ndarray,
) -> np.ndarray:
    mask_lons = np.mod(mask_lons, 360.0)
    data_lons = np.mod(data_lons, 360.0)
    lat_index = [int(np.argmin(np.abs(mask_lats - v))) for v in data_lats]
    lon_index = [int(np.argmin(np.abs(mask_lons - v))) for v in data_lons]
    return mask[np.ix_(lat_index, lon_index)]


def _load_npz_region_mask(
    region: str,
    *,
    lon_convention: str = "360",
    nlat: int = 180,
    nlon: int = 360,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    region_norm = _region_mask_token(region)
    mask_path = (
        PROJECT_ROOT.parent
        / "gcmagicc_ensemble_runner"
        / "data"
        / "regionmasks"
        / f"{region_norm}_nlat{nlat}_nlon{nlon}_lon{lon_convention}.npz"
    )
    if not mask_path.exists():
        return None
    with np.load(mask_path, allow_pickle=True) as data:
        return data["mask"].astype(bool), data["lats"], data["lons"]


def _maybe_generate_region_mask(
    region: str,
    *,
    lon_convention: str,
    nlat: int,
    nlon: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    If a requested region mask is missing, attempt to create it on-the-fly by
    invoking 750_create_regionmasks_forGPU_segment_runner.py with the current grid.
    """
    # Try existing masks (requested lon convention first, then the alternate).
    mask_tuple = _load_npz_region_mask(region, lon_convention=lon_convention, nlat=nlat, nlon=nlon)
    if mask_tuple is None:
        alt_lon = "180" if lon_convention == "360" else "360"
        mask_tuple = _load_npz_region_mask(region, lon_convention=alt_lon, nlat=nlat, nlon=nlon)
    if mask_tuple is not None:
        return mask_tuple

    gen_script = PROJECT_ROOT / "notebooks" / "750_create_regionmasks_forGPU_segment_runner.py"
    output_dir = (
        PROJECT_ROOT.parent / "gcmagicc_ensemble_runner" / "data" / "regionmasks"
    )
    if not gen_script.exists():
        _log(f"⚠️ Region mask missing for '{region}' and generator script not found at {gen_script}")
        return None

    cmd = [
        sys.executable,
        str(gen_script),
        "--regions",
        region,
        "--nlat",
        str(nlat),
        "--nlon",
        str(nlon),
        "--lon-convention",
        lon_convention,
        "--output-dir",
        str(output_dir),
    ]
    _log(f"Region mask for '{region}' not found; generating via 750... (grid {nlat}x{nlon}, lon {lon_convention})")
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            _log(f"⚠️ Mask generation failed for '{region}' (exit {result.returncode}). stderr:\n{result.stderr}")
            return None
    except Exception as exc:
        _log(f"⚠️ Mask generation raised for '{region}': {exc}")
        return None

    # Reload after generation (prefer requested lon convention).
    mask_tuple = _load_npz_region_mask(region, lon_convention=lon_convention, nlat=nlat, nlon=nlon)
    if mask_tuple is None:
        alt_lon = "180" if lon_convention == "360" else "360"
        mask_tuple = _load_npz_region_mask(region, lon_convention=alt_lon, nlat=nlat, nlon=nlon)
    if mask_tuple is None:
        _log(f"⚠️ Mask generation completed but mask still missing for '{region}'")
    return mask_tuple


def _ensure_lon_0_360(ds: xr.Dataset) -> xr.Dataset:
    """Normalize lon coordinate to 0..360 and sort by lon for stable selection."""
    if "lon" not in ds.coords:
        return ds
    lon = ds["lon"].values
    if np.nanmin(lon) < 0.0:
        ds = ds.assign_coords(lon=(np.mod(ds["lon"], 360.0)))
        ds = ds.sortby("lon")
    return ds


def _subset_region_stack_points(ds: xr.Dataset, *, region: str) -> xr.Dataset:
    """
    Subset dataset to AR6 region and stack (lat, lon) -> point.
    Assumes coordinates are named 'lat' and 'lon'.
    """
    ds = _ensure_lon_0_360(ds)
    nlat = int(ds["lat"].size) if "lat" in ds.dims else 180
    nlon = int(ds["lon"].size) if "lon" in ds.dims else 360
    lon_convention = "360"

    if region.lower() == "global":
        ds_region = ds
    else:
        ds_region = None
        if regionmask is not None:
            ar6 = regionmask.defined_regions.ar6.all
            try:
                region_id = ar6.map_keys(region.upper())
                mask = ar6.mask(ds, lat_name="lat", lon_name="lon")
                ds_region = ds.where(mask == region_id, drop=True)
            except Exception:
                ds_region = None

        if ds_region is None:
            mask_tuple = _maybe_generate_region_mask(
                region,
                lon_convention=lon_convention,
                nlat=nlat,
                nlon=nlon,
            )
            if mask_tuple is None:
                raise ValueError(f"Unknown region or missing mask: {region}")
            mask_arr, mask_lats, mask_lons = mask_tuple
            aligned_mask = _align_mask_to_grid(
                mask_arr,
                mask_lats,
                mask_lons,
                ds["lat"].values,
                ds["lon"].values,
            )
            ds_region = ds.where(xr.DataArray(aligned_mask, dims=("lat", "lon")), drop=True)

    ds_region = ds_region.stack(point=("lat", "lon"))
    ds_region = ds_region.dropna(dim="point", how="all")

    # Add lat/lon as explicit coords for easier downstream alignment
    point_index = ds_region.indexes.get("point")
    if point_index is not None and hasattr(point_index, "get_level_values"):
        lat_vals = np.asarray(point_index.get_level_values("lat"), dtype=float)
        lon_vals = np.asarray(np.mod(point_index.get_level_values("lon"), 360.0), dtype=float)
        # Drop stacked lat/lon coords before reassigning explicit point-wise coords
        ds_region = ds_region.drop_vars(["lat", "lon"], errors="ignore")
        ds_region = ds_region.assign_coords(lat=("point", lat_vals), lon=("point", lon_vals))
    return ds_region


_ENSEMBLE_ID_RE = re.compile(r"(r\d+i\d+p\d+f\d+)", re.IGNORECASE)


def _ensemble_id_from_stem(stem: str) -> Optional[str]:
    """
    Extract ensemble identifier token (e.g., r16i1p1f1) from a filename stem.
    Used to pair SCENARIO2 baselines to corresponding SCENARIO1 members under memberwise strategy.
    """
    m = _ENSEMBLE_ID_RE.search(stem or "")
    return m.group(1) if m else None


def _infer_time_year_range(path: Path) -> Tuple[int, int]:
    """
    Infer (start_year, end_year) from a NetCDF file's time coordinate.
    """
    ds = _open_dataset_safe(path, decode_times=True)
    try:
        years = ds["time"].dt.year
        y0 = int(years.min().item())
        y1 = int(years.max().item())
        return y0, y1
    finally:
        try:
            ds.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# R-SPEI-compatible log-Logistic (generalized logistic, "glo") helpers
# -----------------------------------------------------------------------------
def _pwm_ub_0_1_2(x: np.ndarray) -> Tuple[float, float, float]:
    """
    Unbiased probability weighted moments b0,b1,b2 for a 1D sample.
    Mirrors PWM(x, order=0:2) used by R SPEI via TLMoments for ub-pwm.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 3:
        return np.nan, np.nan, np.nan
    xs = np.sort(x)
    b0 = float(np.mean(xs))
    i = np.arange(1, n + 1, dtype=np.float64)  # 1..n
    w1 = (i - 1.0) / (n - 1.0)
    b1 = float(np.sum(w1 * xs) / n)
    w2 = (i - 1.0) * (i - 2.0) / ((n - 1.0) * (n - 2.0))
    b2 = float(np.sum(w2 * xs) / n)
    return b0, b1, b2


def _lmom_1_2_3_from_pwm(b0: float, b1: float, b2: float) -> Tuple[float, float, float]:
    """Convert PWMs to L-moments λ1,λ2,λ3."""
    lam1 = b0
    lam2 = 2.0 * b1 - b0
    lam3 = 6.0 * b2 - 6.0 * b1 + b0
    return lam1, lam2, lam3


def _glo_params_from_lmom(lam1: float, lam2: float, tau3: float) -> Tuple[float, float, float]:
    """
    Generalized logistic ("log-Logistic" in SPEI) parameters from L-moments:
      kappa = -tau3
      alpha from λ2 = alpha*kappa*pi/sin(kappa*pi)
      xi from λ1 = xi + alpha*(1/kappa - pi/sin(kappa*pi))
    """
    if not np.isfinite(lam1) or not np.isfinite(lam2) or lam2 == 0 or not np.isfinite(tau3):
        return np.nan, np.nan, np.nan
    kappa = -float(tau3)
    if not np.isfinite(kappa) or abs(kappa) >= 1.0:
        return np.nan, np.nan, np.nan
    if abs(kappa) < 1e-10:
        # Logistic limit
        return float(lam1), float(lam2), 0.0
    s = float(np.sin(kappa * np.pi))
    if not np.isfinite(s) or abs(s) < 1e-14:
        return np.nan, np.nan, np.nan
    alpha = float(lam2 * (s / (kappa * np.pi)))
    if not np.isfinite(alpha) or alpha <= 0:
        return np.nan, np.nan, np.nan
    xi = float(lam1 - alpha * (1.0 / kappa - (np.pi / s)))
    return xi, alpha, kappa


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """
    Inverse standard normal CDF (qnorm) using Acklam's rational approximation.
    Pure NumPy implementation to avoid requiring SciPy at runtime.
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-15, 1.0 - 1e-15)

    a = np.array(
        [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00],
        dtype=np.float64,
    )
    b = np.array(
        [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01],
        dtype=np.float64,
    )
    c = np.array(
        [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00],
        dtype=np.float64,
    )
    d = np.array(
        [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00],
        dtype=np.float64,
    )
    plow = 0.02425
    phigh = 1.0 - plow

    x = np.empty_like(p, dtype=np.float64)

    mlow = p < plow
    mhigh = p > phigh
    mmid = (~mlow) & (~mhigh)

    if np.any(mlow):
        q = np.sqrt(-2.0 * np.log(p[mlow]))
        x[mlow] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                  ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    if np.any(mmid):
        q = p[mmid] - 0.5
        r = q * q
        x[mmid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                  (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)

    if np.any(mhigh):
        q = np.sqrt(-2.0 * np.log(1.0 - p[mhigh]))
        x[mhigh] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                   ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    return x


def _select_template_points(ds: xr.Dataset, *, template: TemplatePoints, variables: Sequence[str]) -> xr.Dataset:
    """
    Select the template (lat, lon) point set from a dataset using nearest neighbour.
    Result dims: time, point; with coords lat(point), lon(point).
    """
    ds = _ensure_lon_0_360(ds)
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("Dataset must have 'lat' and 'lon' coordinates to select template points.")

    da_lat = xr.DataArray(template.lat, dims=("point",), name="lat")
    da_lon = xr.DataArray(template.lon, dims=("point",), name="lon")

    sub = ds[list(variables)].sel(lat=da_lat, lon=da_lon, method="nearest")
    sub = sub.assign_coords(
        point=("point", np.arange(sub.sizes["point"], dtype=int)),
        lat=("point", template.lat.astype(float)),
        lon=("point", template.lon.astype(float)),
    )
    return sub


# -----------------------------------------------------------------------------
# Point grouping (centered mean over neighboring pixels)
# -----------------------------------------------------------------------------
def _build_point_groups(template: TemplatePoints, group_pixels: int) -> Optional[List[np.ndarray]]:
    """
    Build neighbor index lists for each template point to support centered averaging.
    group_pixels must be an odd square count (1, 9, 25).
    """
    if group_pixels <= 1:
        return None
    side = int(round(math.sqrt(group_pixels)))
    if side * side != group_pixels or side % 2 == 0 or group_pixels not in _ALLOWED_GROUP_PIXELS:
        raise ValueError("group_pixels must be one of {1, 9, 25} (odd square window).")
    half = side // 2

    lat_vals = np.round(template.lat, 6)
    lon_vals = np.round(template.lon, 6)
    lat_unique = np.unique(lat_vals)
    lon_unique = np.unique(lon_vals)

    lat_to_idx = {v: i for i, v in enumerate(lat_unique)}
    lon_to_idx = {v: i for i, v in enumerate(lon_unique)}
    grid_lookup = {(lat_to_idx[la], lon_to_idx[lo]): idx for idx, (la, lo) in enumerate(zip(lat_vals, lon_vals))}

    groups: List[np.ndarray] = []
    for la, lo in zip(lat_vals, lon_vals):
        li = lat_to_idx.get(la)
        lj = lon_to_idx.get(lo)
        lat_inds = range(max(0, li - half), min(len(lat_unique) - 1, li + half) + 1)
        lon_inds = range(max(0, lj - half), min(len(lon_unique) - 1, lj + half) + 1)
        idxs: List[int] = []
        for ii in lat_inds:
            for jj in lon_inds:
                pt_idx = grid_lookup.get((ii, jj))
                if pt_idx is not None:
                    idxs.append(pt_idx)
        if not idxs:
            center_idx = grid_lookup.get((li, lj))
            if center_idx is not None:
                idxs = [center_idx]
        groups.append(np.array(sorted(set(idxs)), dtype=int))
    return groups


def _apply_point_grouping(ds: xr.Dataset, *, point_groups: Optional[List[np.ndarray]], group_pixels: int) -> xr.Dataset:
    """Average each point over its neighbor list (center included) before analysis."""
    if group_pixels <= 1 or point_groups is None or "point" not in ds.dims:
        return ds
    ds_out = ds.copy(deep=False)
    for name, da in ds.data_vars.items():
        if "point" not in da.dims:
            continue
        axis = da.get_axis_num("point")
        arr = np.asarray(da.values)
        arr_swap = np.moveaxis(arr, axis, -1)
        grouped = np.empty_like(arr_swap, dtype=np.float64)
        for j, idxs in enumerate(point_groups):
            if idxs.size == 1:
                grouped[..., j] = arr_swap[..., idxs[0]]
            else:
                grouped[..., j] = np.nanmean(arr_swap[..., idxs], axis=-1)
        grouped = np.moveaxis(grouped, -1, axis).astype(np.float32)
        ds_out[name] = xr.DataArray(grouped, dims=da.dims, coords=da.coords, attrs=da.attrs, name=name)
    ds_out.attrs.update(ds.attrs)
    ds_out.attrs["group_pixels"] = int(group_pixels)
    return ds_out


# -----------------------------------------------------------------------------
# File discovery and paths
# -----------------------------------------------------------------------------
def _discover_nc_files(root: Path) -> List[Path]:
    """
    Discover candidate NetCDF files under a root.
    Excludes derivative/output folders to avoid recursion.
    """
    root = root.resolve()
    files: List[Path] = []
    for p in root.rglob("*.nc"):
        if "data_derivatives" in p.parts:
            continue
        if "SPEI" in p.parts:
            continue
        files.append(p)
    return sorted(files)


def _reorder_file_list(files: Sequence[Path], *, mode: str, offset: int) -> List[Path]:
    """
    Reorder input files to reduce synchronized reads across concurrent jobs.

    mode:
      - sorted : keep canonical sorted order (default legacy behavior)
      - rotate : cyclic shift by `offset` so each worker starts at a different file
    """
    out = list(files)
    if not out:
        return out

    mode_n = str(mode or "sorted").strip().lower()
    if mode_n == "sorted":
        return out
    if mode_n == "rotate":
        n = len(out)
        k = int(offset) % n
        return out[k:] + out[:k]

    raise ValueError(f"Unknown file-order mode: {mode}")


def _extract_scenario_from_filename(path: Path) -> Optional[str]:
    """
    Parse scenario name from a GCMagicc NetCDF filename.
    Expected stem pattern: ..._<SCENARIO>_r*, e.g., _ssp245-nat_r16i1p1f1.
    """
    stem = path.stem
    match = re.search(r"_([^_]+)_r\d", stem)
    if match:
        return match.group(1)
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[-2]
    return None


def _scenario_has_suffix(path: Path, *, suffix: str) -> bool:
    """
    True when a parsed scenario token from filename ends with the requested suffix.
    """
    suf = str(suffix or "").strip()
    if not suf or suf.lower() in {"none", "null"}:
        return False
    scenario = _extract_scenario_from_filename(path)
    if scenario is None:
        return False
    return scenario.lower().endswith(suf.lower())


def _normalize_rsdsbiasadjust_nat_scens_mode(mode: str) -> str:
    token = str(mode or "").strip().lower()
    if token == "exempt":
        token = "excempt"
    if token not in RSDSBIASADJUST_NAT_SCENS_CHOICES:
        raise ValueError(
            f"Unknown RSDSBIASADJUST_NAT_SCENS mode: {mode}. "
            f"Supported: {', '.join(RSDSBIASADJUST_NAT_SCENS_CHOICES)}"
        )
    return token


def _rsdsbiasadjust_should_exclude_nat_from_bias_fit(mode: str) -> bool:
    mode_n = _normalize_rsdsbiasadjust_nat_scens_mode(mode)
    # 'early' needs non-nat-derived offsets; 'excempt' also excludes nat by design.
    return mode_n in {"excempt", "early"}


def _resolve_rsds_bias_adjustment_for_path(
    path: Path,
    *,
    adjustment: Optional[RsdsEra5BiasAdjustment],
    rsdsbiasadjust_nat_scens_mode: str,
    nat_suffix: str = RSDSBIASADJUST_NAT_SCENARIO_SUFFIX,
) -> Tuple[Optional[RsdsEra5BiasAdjustment], bool]:
    """
    Resolve per-file RSDS adjustment policy.
    Returns (adjustment_or_none, use_first_era5_year_only_for_offsets).
    """
    if adjustment is None:
        return None, False

    mode_n = _normalize_rsdsbiasadjust_nat_scens_mode(rsdsbiasadjust_nat_scens_mode)
    is_nat = _scenario_has_suffix(path, suffix=nat_suffix)
    if (not is_nat) or mode_n == "full":
        return adjustment, False
    if mode_n == "excempt":
        return None, False
    # mode_n == "early": keep month-specific first ERA5-year offset for nat files.
    return adjustment, True


def _rsds_bias_adjustment_tag(
    adjustment: Optional[RsdsEra5BiasAdjustment],
    *,
    rsdsbiasadjust_nat_scens_mode: str,
) -> str:
    if adjustment is None:
        return "off"
    mode_n = _normalize_rsdsbiasadjust_nat_scens_mode(rsdsbiasadjust_nat_scens_mode)
    return f"era5w{int(adjustment.smoothing_window_years)}_nat-{mode_n}"


def _filter_nc_files_by_forcing(
    files: Sequence[Path],
    *,
    forcing: str,
    suffix_override: Optional[str] = None,
) -> Tuple[List[Path], List[Path]]:
    """
    Split NetCDF files into matching vs skipped based on scenario suffix.
    SCENARIO2 files are those whose scenario endswith '-nat'
    (or the provided suffix_override); SCENARIO1 files are the complement.
    Returns (kept, skipped) where skipped includes unparseable filenames.
    """
    forcing_upper = str(forcing).strip().upper()
    if forcing_upper not in {FORCING_SCENARIO1_LABEL, FORCING_SCENARIO2_LABEL}:
        raise ValueError(
            f"Unknown forcing '{forcing}'. "
            f"Expected {FORCING_SCENARIO1_LABEL} or {FORCING_SCENARIO2_LABEL}."
        )
    want_scenario2 = forcing_upper == FORCING_SCENARIO2_LABEL
    # Treat empty/whitespace suffix as "no suffix filtering"
    nat_suffix = None if suffix_override is None else str(suffix_override).strip()
    if nat_suffix is not None and nat_suffix.lower() in {"none", "null"}:
        nat_suffix = ""
    kept: List[Path] = []
    skipped: List[Path] = []

    if not nat_suffix:
        # No suffix filtering requested: keep everything (parsed or not)
        return list(files), []

    for path in files:
        scenario = _extract_scenario_from_filename(path)
        if scenario is None:
            skipped.append(path)
            continue
        is_nat = scenario.lower().endswith(nat_suffix.lower())
        if (want_scenario2 and is_nat) or (not want_scenario2 and not is_nat):
            kept.append(path)
        else:
            skipped.append(path)

    return kept, skipped


_CMIP6_FILE_RE = re.compile(
    r"^DAT_(?P<source_id>.+?)_(?P<experiment>historical|hist-nat|ssp245)_(?P<member_id>r\d+i\d+p\d+f\d+)_",
    re.IGNORECASE,
)


def _parse_cmip6_file_metadata(path: Path) -> Optional[Tuple[str, str, str]]:
    """
    Parse CMIP6 filename metadata from ETHFOG file names:
      DAT_<source_id>_<experiment>_<member_id>_....
    Returns (source_id, experiment, member_id) or None if unmatched.
    """
    m = _CMIP6_FILE_RE.match(path.name)
    if not m:
        return None
    source_id = str(m.group("source_id"))
    experiment = str(m.group("experiment")).lower()
    member_id = str(m.group("member_id")).lower()
    return source_id, experiment, member_id


def _parse_cmip6_experiments(raw: str) -> List[str]:
    vals = [v.strip().lower() for v in str(raw or "").split(",") if v.strip()]
    if not vals:
        return list(CMIP6_EXPERIMENT_CHOICES)
    allowed = set(CMIP6_EXPERIMENT_CHOICES)
    bad = [v for v in vals if v not in allowed]
    if bad:
        raise ValueError(
            f"Unknown CMIP6 experiment(s): {bad}. Supported: {', '.join(CMIP6_EXPERIMENT_CHOICES)}"
        )
    # preserve order while deduplicating
    out: List[str] = []
    seen = set()
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _select_cmip6_one_member_per_source(
    cmip6_root: Path,
    *,
    experiments: Sequence[str],
    limit_models: Optional[int],
) -> Tuple[List[Path], List[Path], List[Path], Dict]:
    """
    Select one ensemble member per source_id across CMIP6 historical/hist-nat/ssp245 files.

    Selection rule per source_id:
      1) if historical members exist, choose the historical member that appears in the
         largest number of requested experiments (historical/hist-nat/ssp245), then lexicographically;
      2) if historical is unavailable, choose the non-historical member that appears in
         the largest number of requested experiments, then lexicographically.

    Returns:
      (historical_files_selected, hist_nat_files_selected, ssp245_files_selected, manifest_json_dict)
    """
    exp_order = [e for e in ("historical", "hist-nat", "ssp245") if e in set(experiments)]
    if not exp_order:
        return [], [], [], {"selected_sources": [], "selected_run_count": 0}

    # source -> experiment -> member -> [paths]
    grouped: Dict[str, Dict[str, Dict[str, List[Path]]]] = {}
    totals_by_experiment = {e: 0 for e in exp_order}
    unmatched = 0

    for p in sorted(cmip6_root.glob("*.nc")):
        parsed = _parse_cmip6_file_metadata(p)
        if parsed is None:
            unmatched += 1
            continue
        source_id, experiment, member_id = parsed
        if experiment not in exp_order:
            continue
        totals_by_experiment[experiment] += 1
        grouped.setdefault(source_id, {}).setdefault(experiment, {}).setdefault(member_id, []).append(p)

    source_ids = sorted(grouped.keys())
    if limit_models is not None:
        source_ids = source_ids[: int(limit_models)]

    hist_selected: List[Path] = []
    hist_nat_selected: List[Path] = []
    ssp245_selected: List[Path] = []
    manifest_rows: List[Dict[str, object]] = []

    for source_id in source_ids:
        by_exp = grouped[source_id]
        hist_members = set(by_exp.get("historical", {}).keys())
        nat_members = set(by_exp.get("hist-nat", {}).keys())
        ssp245_members = set(by_exp.get("ssp245", {}).keys())

        chosen_member: Optional[str] = None
        if hist_members:
            # Prefer a historical member that appears in as many requested CMIP6 experiments as possible.
            scoring_sets = [hist_members, nat_members, ssp245_members]
            chosen_member = sorted(
                hist_members,
                key=lambda m: (-sum(1 for s in scoring_sets if m in s), m),
            )[0]
        elif nat_members or ssp245_members:
            # No historical run available for this source_id: still keep one member deterministically.
            scoring_sets = [nat_members, ssp245_members]
            union_members = nat_members.union(ssp245_members)
            chosen_member = sorted(
                union_members,
                key=lambda m: (-sum(1 for s in scoring_sets if m in s), m),
            )[0]

        if chosen_member is None:
            continue

        hist_path: Optional[Path] = None
        hist_nat_path: Optional[Path] = None
        ssp245_path: Optional[Path] = None
        if "historical" in by_exp and chosen_member in by_exp["historical"]:
            hist_path = sorted(by_exp["historical"][chosen_member])[0]
            hist_selected.append(hist_path)
        if "hist-nat" in by_exp and chosen_member in by_exp["hist-nat"]:
            hist_nat_path = sorted(by_exp["hist-nat"][chosen_member])[0]
            hist_nat_selected.append(hist_nat_path)
        if "ssp245" in by_exp and chosen_member in by_exp["ssp245"]:
            ssp245_path = sorted(by_exp["ssp245"][chosen_member])[0]
            ssp245_selected.append(ssp245_path)

        manifest_rows.append(
            {
                "source_id": source_id,
                "member_id": chosen_member,
                "historical_file": str(hist_path) if hist_path is not None else None,
                "hist_nat_file": str(hist_nat_path) if hist_nat_path is not None else None,
                "ssp245_file": str(ssp245_path) if ssp245_path is not None else None,
                "historical_member_count": len(hist_members),
                "hist_nat_member_count": len(nat_members),
                "ssp245_member_count": len(ssp245_members),
            }
        )

    manifest = {
        "cmip6_root": str(cmip6_root),
        "experiments_requested": list(exp_order),
        "totals_available_by_experiment": totals_by_experiment,
        "unmatched_nc_filenames": int(unmatched),
        "selected_source_count": len(manifest_rows),
        "selected_historical_count": len(hist_selected),
        "selected_hist_nat_count": len(hist_nat_selected),
        "selected_ssp245_count": len(ssp245_selected),
        "selected_run_count": len(hist_selected) + len(hist_nat_selected) + len(ssp245_selected),
        "selected_sources": manifest_rows,
    }
    return hist_selected, hist_nat_selected, ssp245_selected, manifest


_CMIP6_MANIFEST_SELECTED_COUNT_KEYS = {
    "historical": "selected_historical_count",
    "hist-nat": "selected_hist_nat_count",
    "ssp245": "selected_ssp245_count",
}


class MissingExactCmip6BaselineError(RuntimeError):
    def __init__(self, record: Dict[str, object]):
        self.record = dict(record)
        super().__init__(str(self.record.get("reason", "missing exact CMIP6 baseline member")))


def _ensure_cmip6_manifest_drop_tracking(manifest: Optional[Dict]) -> None:
    if manifest is None:
        return
    manifest.setdefault(
        "post_baseline_selected_count_by_experiment",
        {
            exp: int(manifest.get(count_key, 0))
            for exp, count_key in _CMIP6_MANIFEST_SELECTED_COUNT_KEYS.items()
        },
    )
    manifest.setdefault(
        "dropped_missing_exact_baseline_count_by_experiment",
        {exp: 0 for exp in _CMIP6_MANIFEST_SELECTED_COUNT_KEYS},
    )
    manifest.setdefault(
        "post_baseline_selected_run_count",
        int(sum(int(v) for v in manifest["post_baseline_selected_count_by_experiment"].values())),
    )
    manifest.setdefault("dropped_missing_exact_baseline_count", 0)
    manifest.setdefault("dropped_missing_exact_baseline_runs", [])


def _record_cmip6_exact_baseline_drop(
    *,
    manifest: Optional[Dict],
    record: Dict[str, object],
    cmip6_root: Optional[Path],
    output_tag: Optional[str],
    regions: Sequence[str],
    target_key: Optional[str] = None,
) -> bool:
    if manifest is None:
        return False
    _ensure_cmip6_manifest_drop_tracking(manifest)
    payload = dict(record)
    if target_key is not None:
        payload["target_key"] = str(target_key)

    records = manifest["dropped_missing_exact_baseline_runs"]
    drop_key = (
        str(payload.get("target_key", "")),
        str(payload.get("run_file", "")),
        str(payload.get("baseline_source_scenario", "")),
    )
    for existing in records:
        existing_key = (
            str(existing.get("target_key", "")),
            str(existing.get("run_file", "")),
            str(existing.get("baseline_source_scenario", "")),
        )
        if existing_key == drop_key:
            return False

    records.append(payload)
    scenario = str(payload.get("scenario", "")).strip().lower()
    dropped_by_exp = manifest["dropped_missing_exact_baseline_count_by_experiment"]
    post_by_exp = manifest["post_baseline_selected_count_by_experiment"]
    if scenario in dropped_by_exp:
        dropped_by_exp[scenario] = int(dropped_by_exp.get(scenario, 0)) + 1
    if scenario in post_by_exp:
        post_by_exp[scenario] = max(0, int(post_by_exp.get(scenario, 0)) - 1)
    manifest["dropped_missing_exact_baseline_count"] = int(len(records))
    manifest["post_baseline_selected_run_count"] = int(sum(int(v) for v in post_by_exp.values()))

    if cmip6_root is not None:
        for region in regions:
            _write_cmip6_selection_manifest(
                cmip6_root=cmip6_root,
                manifest=manifest,
                output_tag=output_tag,
                region=region,
            )
    return True


def _write_cmip6_selection_manifest(
    *,
    cmip6_root: Path,
    manifest: Dict,
    output_tag: Optional[str],
    region: str,
) -> Optional[Path]:
    try:
        _ensure_cmip6_manifest_drop_tracking(manifest)
        out_dir = _deriv_root(cmip6_root, output_tag=output_tag, region=region)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "CMIP6_SELECTION__one-member-per-source.json"
        _save_json(manifest, out_path)
        return out_path
    except Exception:
        return None


def _normalize_cmip6_baseline_source_token(token: str) -> str:
    """
    Normalize CMIP6 baseline source aliases to canonical experiment tokens.
    """
    t = str(token or "").strip().lower()
    if t in {"historical", "hist", "cmip6-historical", "cmip6_hist", "cmip6-hist"}:
        return "historical"
    if t in {"hist-nat", "histnat", "cmip6-hist-nat", "cmip6_hist_nat"}:
        return "hist-nat"
    if t in {"ssp245", "cmip6-ssp245", "cmip6_ssp245", "ssp2-4.5", "ssp2_4_5"}:
        return "ssp245"
    if t in {"self", "same"}:
        return "self"
    return t


def _build_cmip6_file_index(
    cmip6_hist_files: Sequence[Path],
    cmip6_hist_nat_files: Sequence[Path],
    cmip6_ssp245_files: Sequence[Path],
) -> Tuple[
    Dict[str, List[Path]],
    Dict[str, Dict[str, Path]],
    Dict[str, Dict[Tuple[str, str], Path]],
]:
    """
    Build CMIP6 lookup tables from selected one-member-per-source files.
    Returns:
      - files by experiment
      - files by experiment/source_id
      - files by experiment/(source_id, member_id)
    """
    by_exp_files: Dict[str, List[Path]] = {"historical": [], "hist-nat": [], "ssp245": []}
    by_exp_source: Dict[str, Dict[str, Path]] = {"historical": {}, "hist-nat": {}, "ssp245": {}}
    by_exp_source_member: Dict[str, Dict[Tuple[str, str], Path]] = {"historical": {}, "hist-nat": {}, "ssp245": {}}

    def _ingest(exp_hint: str, paths: Sequence[Path]) -> None:
        for path in paths:
            parsed = _parse_cmip6_file_metadata(path)
            if parsed is None:
                continue
            source_id, experiment, member_id = parsed
            exp_key = experiment if experiment in by_exp_files else exp_hint
            if exp_key not in by_exp_files:
                continue
            by_exp_files[exp_key].append(path)
            src_key = source_id.lower().strip()
            mem_key = member_id.lower().strip()
            by_exp_source_member[exp_key][(src_key, mem_key)] = path
            if src_key not in by_exp_source[exp_key]:
                by_exp_source[exp_key][src_key] = path

    _ingest("historical", cmip6_hist_files)
    _ingest("hist-nat", cmip6_hist_nat_files)
    _ingest("ssp245", cmip6_ssp245_files)

    for exp in ("historical", "hist-nat", "ssp245"):
        uniq: List[Path] = []
        seen = set()
        for p in by_exp_files[exp]:
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)
        by_exp_files[exp] = uniq

    return by_exp_files, by_exp_source, by_exp_source_member


def _resolve_cmip6_baseline_selection(
    *,
    run_path: Path,
    target_key: str,
    cmip6_baseline_cfg: Dict[str, Dict[str, str]],
    by_exp_files: Dict[str, List[Path]],
    by_exp_source: Dict[str, Dict[str, Path]],
    by_exp_source_member: Dict[str, Dict[Tuple[str, str], Path]],
) -> Tuple[List[Path], str, str, str, str, str, Optional[str]]:
    """
    Resolve CMIP6 baseline selection for one target run.

    Returns:
      (baseline_files, baseline_source_key, baseline_pooling, baseline_id,
       baseline_label, cache_token, optional_warning)
    """
    run_id = run_path.stem

    def _self_result(note: Optional[str]) -> Tuple[List[Path], str, str, str, str, str, Optional[str]]:
        return (
            [run_path],
            "cmip6-self",
            "per_member",
            f"cmip6-self-{run_id}",
            f"CMIP6 self {run_id}",
            f"self:{run_id.lower()}",
            note,
        )

    cfg_default = CMIP6_BASELINE_CONFIG.get(target_key, {"source": "historical", "pooling": "per_member"})
    cfg = cmip6_baseline_cfg.get(target_key, cfg_default)
    source_exp = _normalize_cmip6_baseline_source_token(cfg.get("source", cfg_default["source"]))
    pool_req = str(cfg.get("pooling", cfg_default["pooling"])).strip().lower()
    if pool_req not in set(CMIP6_BASELINE_POOLING_CHOICES):
        pool_req = "per_member"

    if source_exp == "self":
        return _self_result(None)

    if source_exp not in {"historical", "hist-nat", "ssp245"}:
        return _self_result(
            f"unknown CMIP6 baseline source '{cfg.get('source')}' for target={target_key}; "
            f"falling back to self baseline for {run_id}"
        )

    source_key = f"cmip6-{source_exp}"
    if pool_req == "pooled":
        source_files = list(by_exp_files.get(source_exp, []))
        if not source_files:
            return _self_result(
                f"no CMIP6 {source_exp} files available for pooled baseline; "
                f"falling back to self baseline for {run_id}"
            )
        return (
            source_files,
            source_key,
            "pooled",
            f"cmip6-{source_exp}-pooled",
            f"CMIP6 {source_exp} pooled",
            f"{source_exp}:pooled",
            None,
        )

    # per_member
    parsed = _parse_cmip6_file_metadata(run_path)
    if parsed is None:
        scenario_token = {
            "cmip6_hist": "historical",
            "cmip6_hist_nat": "hist-nat",
            "cmip6_ssp245": "ssp245",
        }.get(target_key, source_exp)
        raise MissingExactCmip6BaselineError(
            {
                "run_id": str(run_id),
                "run_file": str(run_path.expanduser().resolve(strict=False)),
                "source_id": "",
                "member_id": "",
                "scenario": str(scenario_token),
                "baseline_source_scenario": str(source_exp),
                "reason": f"could not parse CMIP6 metadata from {run_path.name}",
            }
        )
    source_id, _experiment, member_id = parsed
    src_key = source_id.lower().strip()
    mem_key = member_id.lower().strip()

    baseline_path = by_exp_source_member.get(source_exp, {}).get((src_key, mem_key))
    if baseline_path is None:
        raise MissingExactCmip6BaselineError(
            {
                "run_id": str(run_id),
                "run_file": str(run_path.expanduser().resolve(strict=False)),
                "source_id": str(source_id),
                "member_id": str(member_id),
                "scenario": str(_experiment),
                "baseline_source_scenario": str(source_exp),
                "reason": (
                    f"missing exact {source_exp} baseline member for "
                    f"source_id={source_id}, member_id={member_id}"
                ),
            }
        )

    parsed_base = _parse_cmip6_file_metadata(baseline_path)
    if parsed_base is not None:
        b_source_id, b_experiment, b_member_id = parsed_base
        baseline_id = f"cmip6-{b_experiment}-{b_source_id}-{b_member_id}"
        baseline_label = f"CMIP6 {b_experiment} {b_source_id} {b_member_id}"
        cache_token = f"{b_experiment}:{b_source_id.lower()}:{b_member_id.lower()}"
    else:
        baseline_id = f"cmip6-{source_exp}-{baseline_path.stem}"
        baseline_label = f"CMIP6 {source_exp} {baseline_path.stem}"
        cache_token = f"{source_exp}:{baseline_path.stem.lower()}"

    return [baseline_path], source_key, "per_member", baseline_id, baseline_label, cache_token, None


def _filter_cmip6_runs_with_exact_baselines(
    *,
    files: Sequence[Path],
    target_key: str,
    cmip6_baseline_cfg: Dict[str, Dict[str, str]],
    by_exp_files: Dict[str, List[Path]],
    by_exp_source: Dict[str, Dict[str, Path]],
    by_exp_source_member: Dict[str, Dict[Tuple[str, str], Path]],
) -> Tuple[List[Path], List[Dict[str, object]]]:
    retained: List[Path] = []
    dropped: List[Dict[str, object]] = []
    for path in files:
        try:
            _resolve_cmip6_baseline_selection(
                run_path=path,
                target_key=target_key,
                cmip6_baseline_cfg=cmip6_baseline_cfg,
                by_exp_files=by_exp_files,
                by_exp_source=by_exp_source,
                by_exp_source_member=by_exp_source_member,
            )
        except MissingExactCmip6BaselineError as exc:
            dropped.append(dict(exc.record))
            continue
        retained.append(path)
    return retained, dropped


def _set_derivatives_runtime_config(layout: str, run_suffix: str) -> None:
    global _ACTIVE_DERIVATIVES_LAYOUT
    global _ACTIVE_DERIVATIVES_RUN_SUFFIX
    norm_layout = str(layout or DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE).strip().lower()
    if norm_layout not in set(DERIVATIVES_LAYOUT_CHOICES):
        raise ValueError(
            f"Unsupported derivatives layout '{layout}'. "
            f"Choose one of: {', '.join(DERIVATIVES_LAYOUT_CHOICES)}."
        )
    norm_suffix = str(run_suffix or DEFAULT_DERIVATIVES_RUN_SUFFIX).strip() or DEFAULT_DERIVATIVES_RUN_SUFFIX
    _ACTIVE_DERIVATIVES_LAYOUT = norm_layout
    _ACTIVE_DERIVATIVES_RUN_SUFFIX = norm_suffix


def _is_permission_or_readonly_error(exc: OSError) -> bool:
    return int(getattr(exc, "errno", -1)) in {errno.EROFS, errno.EACCES, errno.EPERM}


def _try_resolve_local_derivatives_root_for_readonly(data_root: Path) -> Optional[Path]:
    """
    For canonical ERA5spliced roots, derive writable local derivatives root under
    created_nc_files so writes can proceed when mounted ERA5splicedS3 is read-only.
    """
    resolved = Path(data_root).expanduser().resolve(strict=False)
    try:
        if str(resolved).startswith(str(_CREATED_NC_ROOT.resolve(strict=False))):
            return None
    except Exception:
        pass

    canonical = parse_era5spliced_dataset_path(resolved)
    if canonical is None:
        return None
    try:
        return resolve_canonical_dataset_root(
            version=str(canonical["version"]),
            experiment_id=str(canonical["experiment_id"]),
            arx=str(canonical["arx"]),
            runmodus=str(canonical["runmodus"]),
            n_ensemble=str(canonical["n_ensemble"]),
            kind="dataderivatives",
            run_instance=str(canonical.get("run_instance") or ""),
            root=_CREATED_NC_ROOT,
            require_verified=False,
        )
    except Exception:
        return None


def _build_speix_deriv_root(
    data_root: Path,
    *,
    output_tag: Optional[str] = None,
    region: Optional[str] = None,
) -> Path:
    out = (
        resolve_derivatives_root(
            data_root.resolve(),
            layout=_ACTIVE_DERIVATIVES_LAYOUT,
            suffix=_ACTIVE_DERIVATIVES_RUN_SUFFIX,
            kind="data_derivatives",
        )
        / "SPEIx"
    )
    if output_tag:
        out = out / output_tag

    region_dir = _region_subdir(region) if region else None
    if region_dir:
        tag_parts = set(Path(str(output_tag)).parts) if output_tag else set()
        if region_dir not in tag_parts:
            out = out / region_dir
    return out


def _deriv_root(data_root: Path, *, output_tag: Optional[str] = None, region: Optional[str] = None) -> Path:
    """
    Base directory for SPEIx derivatives. If a region is supplied, nest outputs
    under region-<SAFE_REGION> to match the latest layout.
    """
    resolved_root = Path(data_root).expanduser().resolve(strict=False)
    cache_key = str(resolved_root)
    write_root = _DERIV_WRITE_REDIRECT_CACHE.get(cache_key, resolved_root)

    out = _build_speix_deriv_root(write_root, output_tag=output_tag, region=region)
    try:
        out.mkdir(parents=True, exist_ok=True)
        return out
    except OSError as exc:
        if not _is_permission_or_readonly_error(exc):
            raise

    fallback_root = _try_resolve_local_derivatives_root_for_readonly(resolved_root)
    if fallback_root is None:
        # No safe redirect available; keep original exception context by re-trying.
        out = _build_speix_deriv_root(resolved_root, output_tag=output_tag, region=region)
        out.mkdir(parents=True, exist_ok=True)
        return out

    _DERIV_WRITE_REDIRECT_CACHE[cache_key] = fallback_root
    redirected = _build_speix_deriv_root(fallback_root, output_tag=output_tag, region=region)
    redirected.mkdir(parents=True, exist_ok=True)
    _log(
        "⚠️ Derivative target is read-only; redirecting writes to local created_nc_files mirror:\n"
        f"  source-root: {resolved_root}\n"
        f"  local-write-root: {fallback_root}"
    )
    return redirected


def _segments_store_dir(data_root: Path, *, output_tag: Optional[str] = None, region: Optional[str] = None) -> Path:
    return _deriv_root(data_root, output_tag=output_tag, region=region) / "segments.zarr"


def _fits_dir(root: Path, *, output_tag: Optional[str] = None, region: Optional[str] = None) -> Path:
    d = _deriv_root(root, output_tag=output_tag, region=region) / "fits"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _world_regional_pet_root(
    output_root: Path,
    *,
    output_tag: Optional[str],
    region: str,
    pet_method: str,
) -> Path:
    base = Path(output_root).expanduser().resolve(strict=False)
    if output_tag and str(output_tag).strip():
        base = base / str(output_tag).strip()
    return base / _region_subdir(region) / f"pet-{_overlay_cache.normalize_pet_method(pet_method)}"


def _resolve_world_regional_output_root(
    world_input_root: Path,
    explicit_output_root: Optional[Path],
) -> Path:
    if explicit_output_root is not None:
        return Path(explicit_output_root).expanduser().resolve(strict=False)
    world_root = _overlay_cache.resolve_world_speix_root(world_input_root, label="world-input")
    return (world_root.parent / "SPEIx").expanduser().resolve(strict=False)


def _select_world_spei_variable(ds: xr.Dataset, scale: int) -> str:
    for candidate in (f"spei{int(scale)}", "spei", "SPEI", f"SPEI{int(scale)}"):
        if candidate in ds:
            return candidate
    raise KeyError(f"No SPEI variable found in world dataset (wanted scale={scale}).")


def _world_segment_group_name(*, region: str, scale: int, start_year: int, end_year: int) -> str:
    return f"spei{int(scale)}__{str(region).upper()}__grid-points__{int(start_year)}-{int(end_year)}__all"


def _world_run_meta_vars_present(ds: xr.Dataset) -> List[str]:
    present: List[str] = []
    for name in ("baseline_source_key", "baseline_pooling", "baseline_strategy", "baseline_fit_file"):
        if name in ds.data_vars or name in ds.variables:
            present.append(name)
    return present


def _copy_world_fit_artifacts(
    *,
    pet_root: Path,
    dest_pet_root: Path,
    force: bool,
) -> int:
    src_dir = pet_root / "fits"
    if not src_dir.exists():
        return 0
    dest_dir = dest_pet_root / "fits"
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(src_dir.glob("*.nc")):
        dest = dest_dir / src.name
        if dest.exists() and not force:
            continue
        shutil.copy2(src, dest)
        copied += 1
    return copied


def _run_world_input_regionalizer(args: argparse.Namespace) -> None:
    if zarr is None:
        raise RuntimeError("world-input regionalizer requires the 'zarr' package.")

    scenario_token = str(args.world_input_scenario or "").strip()
    if not scenario_token:
        raise ValueError("--world-input-scenario is required when --world-input-root is used.")

    output_root = _resolve_world_regional_output_root(args.world_input_root, args.world_output_root)
    world_tag = args.world_input_tag if args.world_input_tag is not None else args.output_tag
    regions = _resolve_execution_regions(args)
    if not regions:
        raise RuntimeError("No regions resolved for world-input regionalizer.")
    pet_methods = _resolve_execution_pet_methods(args)
    if not pet_methods:
        raise RuntimeError("No PET methods resolved for world-input regionalizer.")

    forcing_label = _overlay_cache.default_forcing_label(scenario_token)
    required_year = None if scenario_token.upper() == "ERA5" else _overlay_cache.required_year_for_scenario(scenario_token)
    output_label = _overlay_cache.stacked_output_label(forcing_label, scenario_tag=scenario_token)

    _log("World-input regionalizer mode")
    _log(f"  world-input-root: {Path(args.world_input_root).expanduser().resolve(strict=False)}")
    _log(f"  world-input-tag:  {world_tag or '<latest>'}")
    _log(f"  scenario:         {scenario_token}")
    _log(f"  output-root:      {output_root}")
    _log(f"  output-tag:       {args.output_tag or '<untagged>'}")
    _log(f"  regions={len(regions)} pets={len(pet_methods)} scale={int(args.scale)}")

    total_written = 0
    total_fit_copies = 0
    copied_fit_roots: Set[Path] = set()
    for pet_method in pet_methods:
        pet_root = _overlay_cache.resolve_world_pet_root(
            args.world_input_root,
            tag=world_tag,
            pet_method=pet_method,
            label=f"WORLD {scenario_token}",
        )
        world_files = _overlay_cache.discover_world_files(
            pet_root,
            forcing_label=forcing_label,
            scenario_tag=None if scenario_token.upper() == "ERA5" else scenario_token,
            scale=int(args.scale),
            required_year=required_year,
        )
        if not world_files:
            raise FileNotFoundError(
                f"No world file found for scenario '{scenario_token}' under {pet_root} "
                f"(scale={args.scale}, pet={pet_method})."
            )
        _log(f"  Using {len(world_files)} world file(s) for pet={pet_method}:")
        for world_file in world_files:
            _log(f"    - {world_file}")
            ds_world: Optional[xr.Dataset] = None
            try:
                ds_world = xr.open_dataset(world_file, decode_times=True, engine="netcdf4")
                var_name = _select_world_spei_variable(ds_world, int(args.scale))
                meta_vars = _world_run_meta_vars_present(ds_world)
                years = np.asarray(ds_world["time"].dt.year.values, dtype=int)
                if years.size == 0:
                    raise RuntimeError(f"World file has no time values: {world_file}")
                start_year = int(years.min())
                end_year = int(years.max())
                group_name = _world_segment_group_name(
                    region="REGION_PLACEHOLDER",
                    scale=int(args.scale),
                    start_year=start_year,
                    end_year=end_year,
                )
                run_names = (
                    [str(v) for v in np.asarray(ds_world["run"].values).ravel().tolist()]
                    if "run" in ds_world.coords
                    else [scenario_token.upper() if scenario_token.upper() == "ERA5" else scenario_token]
                )
                run_coord = np.asarray(run_names, dtype=object)

                for region in regions:
                    ds_region = _subset_region_stack_points(ds_world[[var_name]], region=region)
                    if int(ds_region.sizes.get("point", 0)) <= 0:
                        _log(f"    ⚠️ no selected grid points for region={region} pet={pet_method}; skipping")
                        continue

                    dest_pet_root = _world_regional_pet_root(
                        output_root,
                        output_tag=args.output_tag,
                        region=region,
                        pet_method=pet_method,
                    )
                    dest_pet_root.mkdir(parents=True, exist_ok=True)
                    store_dir = dest_pet_root / "segments.zarr"
                    group_rel = (
                        f"stacked/{output_label}/pet-{_overlay_cache.normalize_pet_method(pet_method)}/"
                        f"{group_name.replace('REGION_PLACEHOLDER', str(region).upper())}"
                    )
                    group_dir = store_dir / Path(group_rel)
                    if args.force and group_dir.exists():
                        shutil.rmtree(group_dir, ignore_errors=True)

                    da_region = ds_region[var_name].astype(np.float32)
                    if "run" not in da_region.dims:
                        da_region = da_region.expand_dims(run=run_coord)
                    else:
                        da_region = da_region.assign_coords(run=("run", run_coord))
                    ds_vars: Dict[str, xr.DataArray] = {f"spei{int(args.scale)}": da_region}
                    for meta_name in meta_vars:
                        if meta_name not in ds_world:
                            continue
                        meta_da = ds_world[meta_name]
                        if "run" in meta_da.dims:
                            ds_vars[meta_name] = meta_da.assign_coords(run=("run", run_coord))
                    ds_out = xr.Dataset(ds_vars)
                    attrs = dict(ds_world.attrs)
                    attrs["pet_method"] = _overlay_cache.normalize_pet_method(pet_method)
                    attrs["overlay_source_scenario"] = scenario_token
                    attrs["overlay_world_file"] = str(world_file)
                    ds_out.attrs.update(attrs)
                    ds_out.to_zarr(store_dir, group=group_rel, mode="a")
                    total_written += 1
                    dest_key = dest_pet_root.resolve(strict=False)
                    if dest_key not in copied_fit_roots:
                        total_fit_copies += _copy_world_fit_artifacts(
                            pet_root=pet_root,
                            dest_pet_root=dest_pet_root,
                            force=bool(args.force),
                        )
                        copied_fit_roots.add(dest_key)
                    _log(
                        f"    ✓ wrote region={region} pet={pet_method} "
                        f"-> {dest_pet_root / 'segments.zarr'}::{group_rel}"
                    )
            finally:
                if ds_world is not None:
                    ds_world.close()

    _log(
        "World-input regionalizer done: "
        f"groups_written={total_written}, fit_copies={total_fit_copies}, output_root={output_root}"
    )


def _mirror_fit_to_target(fit_path: Path, target_root: Path, *, region: str, output_tag: Optional[str]) -> Path:
    """
    Ensure a copy of the baseline fit exists under the target scenario's fits directory.
    Returns the path to the copy (or the original if already under the target root).
    """
    try:
        fit_path_resolved = fit_path.resolve()
    except Exception:
        fit_path_resolved = fit_path
    try:
        target_root_resolved = target_root.resolve()
    except Exception:
        target_root_resolved = target_root

    if str(fit_path_resolved).startswith(str(target_root_resolved)):
        return fit_path

    dest_dir = _fits_dir(target_root, output_tag=output_tag, region=region)
    dest = dest_dir / fit_path.name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fit_path, dest)
    return dest


def _write_zarr_json(path: Path, attrs: Dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(attrs, indent=2))
    except Exception:
        pass


def _archive_speix_tree(data_root: Path, *, tag: str, output_tag: Optional[str] = None) -> Optional[Path]:
    """
    Move the entire SPEIx derivative folder to a timestamped archive tree:
      data_derivatives/SPEIx -> data_derivatives_archive/SPEIx/{tag}
    """
    src = (
        resolve_derivatives_root(
            data_root.resolve(),
            layout=_ACTIVE_DERIVATIVES_LAYOUT,
            suffix=_ACTIVE_DERIVATIVES_RUN_SUFFIX,
            kind="data_derivatives",
        )
        / "SPEIx"
    )
    if output_tag:
        src = src / output_tag
    if not src.exists():
        return None
    dest = (
        resolve_derivatives_root(
            data_root.resolve(),
            layout=_ACTIVE_DERIVATIVES_LAYOUT,
            suffix=_ACTIVE_DERIVATIVES_RUN_SUFFIX,
            kind="data_derivatives_archive",
        )
        / "SPEIx"
    )
    if output_tag:
        dest = dest / output_tag
    dest = dest / tag
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src), str(dest))
        return dest
    except Exception:
        return None


def _timestamp_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _archive_existing_path(src: Path, *, archive_root: Path, base_dir: Path) -> Optional[Path]:
    try:
        rel = src.relative_to(base_dir)
    except Exception:
        return None
    dest = archive_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return dest


# -----------------------------------------------------------------------------
# Baseline fitting (pooled over baseline source)
# -----------------------------------------------------------------------------
def _required_vars_for_pet(ds: xr.Dataset, pet_method: str) -> Tuple[str, List[str]]:
    """
    Resolve PET method for this dataset and return (resolved_method, required_vars).
    Enforces that the resolved method is feasible with the dataset variables.
    """
    resolved = recipe_resolve_pet_method(ds, pet_method)
    if pet_method not in {"", "auto"} and resolved != pet_method:
        # For explicit requests we prefer failing fast rather than silently downgrading
        # (important for side-by-side PET method comparisons).
        raise ValueError(f"Requested PET method '{pet_method}' is not feasible for this dataset (resolved='{resolved}').")
    required = [VAR_PR, VAR_TAS]
    if resolved == "penman-monteith":
        # FAO-56 Penman-Monteith in notebooks.recipes.SPEIx
        required.extend([VAR_TASMIN, VAR_TASMAX, VAR_RSDS, VAR_SFCWIND])
        # These are optional in the recipe implementation, but add if present in the template
        if VAR_HURS in ds:
            required.append(VAR_HURS)
        if VAR_PS in ds:
            required.append(VAR_PS)
        elif VAR_PSL in ds:
            required.append(VAR_PSL)
    if resolved == "hargreaves":
        required.extend([VAR_TASMIN, VAR_TASMAX])
        # rsds is optional for Hargreaves in the recipe; include if available
        if VAR_RSDS in ds:
            required.append(VAR_RSDS)
    return resolved, required


def _build_template_from_file(
    src_file: Path,
    *,
    region: str,
    pet_method: str,
    label: str,
) -> Tuple[TemplatePoints, str, List[str]]:
    """
    Use a baseline-source file to:
      - determine template region point set (lat/lon),
      - resolve PET method (if auto),
      - determine required variables.
    """
    _log(f"Building template points from {label} file: {src_file.name}")

    ds = _open_dataset_safe(src_file, decode_times=True)
    ds = _ensure_lon_0_360(ds)

    resolved_pet, required_vars = _required_vars_for_pet(ds, pet_method)
    missing = [v for v in required_vars if v not in ds]
    if missing:
        ds.close()
        raise KeyError(f"Template file missing required variables for PET='{resolved_pet}': {missing}")

    ds_reg = _subset_region_stack_points(ds[required_vars], region=region).load()
    ds.close()

    lat = np.asarray(ds_reg["lat"].values, dtype=float)
    lon = np.asarray(np.mod(ds_reg["lon"].values, 360.0), dtype=float)

    # Ensure deterministic ordering: sort by (lat, lon)
    order = np.lexsort((lon, lat))
    lat = lat[order]
    lon = lon[order]

    ds_reg.close()

    return TemplatePoints(lat=lat, lon=lon, region=region), resolved_pet, required_vars


def _filter_baseline_years(
    wb_roll: xr.DataArray,
    *,
    start_year: Optional[int],
    end_year: Optional[int],
) -> xr.DataArray:
    if start_year is None and end_year is None:
        return wb_roll
    years = wb_roll["time"].dt.year
    mask = xr.ones_like(years, dtype=bool)
    if start_year is not None:
        mask = mask & (years >= int(start_year))
    if end_year is not None:
        mask = mask & (years <= int(end_year))
    return wb_roll.sel(time=mask)


def _fit_baseline_params(
    baseline_files: Sequence[Path],
    *,
    baseline_source_key: str,
    baseline_pooling: str,
    template: TemplatePoints,
    pet_method_resolved: str,
    required_vars: Sequence[str],
    scale: int,
    fit: str,
    limit_runs: Optional[int],
    baseline_label: str,
    baseline_start_year: Optional[int],
    baseline_end_year: Optional[int],
    group_pixels: int,
    point_groups: Optional[List[np.ndarray]],
    rsds_bias_adjustment: Optional[RsdsEra5BiasAdjustment],
    rsdsbiasadjust_nat_scens_mode: str = RSDSBIASADJUST_NAT_SCENS,
    rsdsbiasadjust_nat_suffix: str = RSDSBIASADJUST_NAT_SCENARIO_SUFFIX,
) -> xr.Dataset:
    """
    Pool rolled WB samples across baseline-source files, estimate month-wise parameters
    per point, and return them as a Dataset.

    Returned dataset variables:
      - mu(month, point), sigma(month, point)
      - if fit == loglogistic: xi(month, point), alpha(month, point), kappa(month, point)
    """
    fit_norm = fit.lower()
    if fit_norm not in {"zscore", "loglogistic"}:
        raise ValueError("fit must be one of: zscore, loglogistic")

    # Accumulate samples per month (list of arrays: (samples, points))
    month_samples: Dict[int, List[np.ndarray]] = {m: [] for m in range(1, 13)}
    # Also accumulate region-mean WB samples for plotting background (1D)
    month_samples_mean: Dict[int, List[np.ndarray]] = {m: [] for m in range(1, 13)}

    files = list(baseline_files)
    if limit_runs is not None and len(files) > limit_runs:
        files = files[:limit_runs]
        _log(f"Baseline fit limited to first {len(files)} files due to --limit-runs.")

    n_points = int(template.lat.size)
    if n_points == 0:
        raise RuntimeError(f"Region '{template.region}' produced zero points; cannot fit baseline.")

    year_span = ""
    if baseline_start_year is not None or baseline_end_year is not None:
        year_span = f", years={baseline_start_year or '...'}-{baseline_end_year or '...'}"
    _log(
        f"Fitting baseline on {baseline_label}: {len(files)} files, region={template.region}, "
        f"points={n_points}, scale={scale}, fit={fit_norm}{year_span}, group_pixels={group_pixels}"
    )

    for idx, path in enumerate(files, 1):
        _log(f"  [{idx}/{len(files)}] Baseline ingest: {path.name}")
        try:
            ds = _open_dataset_safe(path, decode_times=True)
            ds = _ensure_lon_0_360(ds)
            sub = _select_template_points(ds, template=template, variables=required_vars).load()
            sub = _apply_point_grouping(sub, point_groups=point_groups, group_pixels=group_pixels)
            rsds_bias_for_file, rsds_bias_hold_early_for_file = _resolve_rsds_bias_adjustment_for_path(
                path,
                adjustment=rsds_bias_adjustment,
                rsdsbiasadjust_nat_scens_mode=rsdsbiasadjust_nat_scens_mode,
                nat_suffix=rsdsbiasadjust_nat_suffix,
            )
            sub = _apply_rsds_bias_adjustment_to_subdataset(
                sub,
                adjustment=rsds_bias_for_file,
                hold_first_reference_year_offsets=rsds_bias_hold_early_for_file,
            )
            ds.close()
        except Exception as exc:
            _log(f"    ⚠️  skip (load/select failed): {exc}")
            continue

        # Compute WB roll on this run
        try:
            wb = recipe_wb_monthly(sub, pet_method_resolved)
            wb_roll = recipe_rolling_sum(wb, scale=scale)  # drops first k-1 months
            wb_roll = _filter_baseline_years(wb_roll, start_year=baseline_start_year, end_year=baseline_end_year)
        except Exception as exc:
            _log(f"    ⚠️  skip (WB/roll failed): {exc}")
            try:
                sub.close()
            except Exception:
                pass
            continue

        if wb_roll.sizes.get("time", 0) == 0:
            _log("    ⚠️  skip (no WB_roll samples)")
            try:
                sub.close()
            except Exception:
                pass
            continue

        months = np.asarray(wb_roll["time"].dt.month.values, dtype=int)
        arr = np.asarray(wb_roll.values, dtype=np.float32)  # (time, point)
        # region-mean WB_roll (time,)
        try:
            wb_roll_mean = wb_roll.mean(dim="point", skipna=True)
            arr_mean = np.asarray(wb_roll_mean.values, dtype=np.float32)
        except Exception:
            arr_mean = None
        for m in range(1, 13):
            sel = (months == m)
            if not np.any(sel):
                continue
            month_samples[m].append(arr[sel, :])
            if arr_mean is not None:
                month_samples_mean[m].append(arr_mean[sel])

        try:
            sub.close()
        except Exception:
            pass

    # Compute mu/sigma
    mu = np.full((12, n_points), np.nan, dtype=np.float32)
    sigma = np.full((12, n_points), np.nan, dtype=np.float32)
    n_samp = np.zeros((12,), dtype=int)
    mu_mean = np.full((12,), np.nan, dtype=np.float32)
    sigma_mean = np.full((12,), np.nan, dtype=np.float32)
    n_samp_mean = np.zeros((12,), dtype=int)

    for m in range(1, 13):
        chunks = month_samples[m]
        if not chunks:
            continue
        cat = np.concatenate(chunks, axis=0)  # (samples, points)
        n_samp[m - 1] = int(cat.shape[0])
        mu[m - 1, :] = np.nanmean(cat, axis=0).astype(np.float32)
        sigma[m - 1, :] = (np.nanstd(cat, axis=0) + 1e-6).astype(np.float32)

        chunks_m = month_samples_mean[m]
        if chunks_m:
            cat_m = np.concatenate(chunks_m, axis=0).astype(np.float32)
            n_samp_mean[m - 1] = int(cat_m.size)
            mu_mean[m - 1] = float(np.nanmean(cat_m))
            sigma_mean[m - 1] = float(np.nanstd(cat_m) + 1e-6)

    data_vars: Dict[str, Tuple[Tuple[str, str], np.ndarray]] = {
        "mu": (("month", "point"), mu),
        "sigma": (("month", "point"), sigma),
    }
    # Store region-mean params for plotting backgrounds
    data_vars.update(
        {
            "mu_mean": (("month",), mu_mean),
            "sigma_mean": (("month",), sigma_mean),
        }
    )

    # Optional log-logistic / generalized logistic params (R SPEI-compatible)
    if fit_norm == "loglogistic":
        xi = np.full((12, n_points), np.nan, dtype=np.float32)
        alpha = np.full((12, n_points), np.nan, dtype=np.float32)
        kappa = np.full((12, n_points), np.nan, dtype=np.float32)
        xi_mean = np.full((12,), np.nan, dtype=np.float32)
        alpha_mean = np.full((12,), np.nan, dtype=np.float32)
        kappa_mean = np.full((12,), np.nan, dtype=np.float32)

        for m in range(1, 13):
            chunks = month_samples[m]
            if not chunks:
                continue
            cat = np.concatenate(chunks, axis=0).astype(np.float64)

            for j in range(n_points):
                x = cat[:, j]
                x = x[np.isfinite(x)]
                if x.size < LOGLOG_MIN_SAMPLES:
                    continue
                sd = float(np.std(x, ddof=1)) if x.size > 1 else np.nan
                if not np.isfinite(sd) or sd == 0:
                    continue

                b0, b1, b2 = _pwm_ub_0_1_2(x)
                if not np.isfinite(b0) or not np.isfinite(b1) or not np.isfinite(b2):
                    continue
                lam1, lam2, lam3 = _lmom_1_2_3_from_pwm(b0, b1, b2)
                if not np.isfinite(lam1) or not np.isfinite(lam2) or lam2 == 0 or not np.isfinite(lam3):
                    continue
                tau3 = lam3 / lam2
                if not np.isfinite(tau3):
                    continue

                xi_j, alpha_j, kappa_j = _glo_params_from_lmom(lam1, lam2, tau3)
                if not np.isfinite(xi_j) or not np.isfinite(alpha_j) or alpha_j <= 0 or not np.isfinite(kappa_j):
                    continue

                xi[m - 1, j] = float(xi_j)
                alpha[m - 1, j] = float(alpha_j)
                kappa[m - 1, j] = float(kappa_j)

            # Region-mean log-logistic params (1D) for plotting
            chunks_m = month_samples_mean[m]
            if chunks_m:
                x_m = np.concatenate(chunks_m, axis=0).astype(np.float64)
                x_m = x_m[np.isfinite(x_m)]
                if x_m.size >= LOGLOG_MIN_SAMPLES:
                    b0, b1, b2 = _pwm_ub_0_1_2(x_m)
                    if np.isfinite(b0) and np.isfinite(b1) and np.isfinite(b2):
                        lam1, lam2, lam3 = _lmom_1_2_3_from_pwm(b0, b1, b2)
                        if np.isfinite(lam1) and np.isfinite(lam2) and lam2 != 0 and np.isfinite(lam3):
                            tau3 = lam3 / lam2
                            if np.isfinite(tau3):
                                xi_m, a_m, k_m = _glo_params_from_lmom(lam1, lam2, float(tau3))
                                if np.isfinite(xi_m) and np.isfinite(a_m) and a_m > 0 and np.isfinite(k_m):
                                    xi_mean[m - 1] = float(xi_m)
                                    alpha_mean[m - 1] = float(a_m)
                                    kappa_mean[m - 1] = float(k_m)

        data_vars.update(
            {
                "xi": (("month", "point"), xi),
                "alpha": (("month", "point"), alpha),
                "kappa": (("month", "point"), kappa),
                "xi_mean": (("month",), xi_mean),
                "alpha_mean": (("month",), alpha_mean),
                "kappa_mean": (("month",), kappa_mean),
            }
        )

    ds_fit = xr.Dataset(
        data_vars=data_vars,
        coords={
            "month": ("month", np.arange(1, 13, dtype=int)),
            "point": ("point", np.arange(n_points, dtype=int)),
            "lat": ("point", template.lat.astype(float)),
            "lon": ("point", template.lon.astype(float)),
        },
        attrs={
            "description": (
                f"Baseline parameters for SPEI standardization "
                f"(source={baseline_label}; pooling={baseline_pooling})."
            ),
            "region": template.region,
            "spei_scale_months": int(scale),
            "pet_method": str(pet_method_resolved),
            "spei_fit": str(fit),
            "baseline_source": str(baseline_label),
            "baseline_source_key": str(baseline_source_key),
            "baseline_pooling": str(baseline_pooling),
            "baseline_start_year": int(baseline_start_year) if baseline_start_year is not None else "none",
            "baseline_end_year": int(baseline_end_year) if baseline_end_year is not None else "none",
            "n_baseline_files_used": int(len(files)),
            "samples_per_month": json.dumps({str(m): int(n_samp[m - 1]) for m in range(1, 13)}),
            "samples_per_month_region_mean": json.dumps({str(m): int(n_samp_mean[m - 1]) for m in range(1, 13)}),
            "group_pixels": int(group_pixels),
        },
    )
    return ds_fit


def _baseline_fit_path(
    baseline_root: Path,
    *,
    region: str,
    scale: int,
    fit: str,
    pet: str,
    baseline_source: str,
    baseline_pooling: str,
    baseline_id: str,
    output_tag: Optional[str],
    group_pixels: int,
    rsds_bias_adjustment_tag: str,
) -> Path:
    src = (baseline_source or "unknown").strip().lower()
    pool = (baseline_pooling or "unknown").strip().lower()
    bid = re.sub(r"[^A-Za-z0-9_-]+", "_", (baseline_id or "unknown").strip())
    group_tag = f"gp{int(group_pixels)}" if group_pixels and group_pixels != 1 else "gp1"
    rsds_tag = re.sub(r"[^A-Za-z0-9_-]+", "_", (rsds_bias_adjustment_tag or "off").strip().lower())
    base = (
        f"BASEFIT__spei{scale}__{region.upper()}__{fit.lower()}__{pet.lower()}"
        f"__src-{src}__pool-{pool}__{group_tag}__rsdsbc-{rsds_tag}__{bid}"
    )
    return _fits_dir(baseline_root, output_tag=output_tag, region=region) / f"{base}.nc"


def _load_or_build_baseline(
    baseline_root: Path,
    baseline_files: Sequence[Path],
    *,
    baseline_source_key: str,
    baseline_pooling: str,
    baseline_id: str,
    baseline_label: str,
    region: str,
    scale: int,
    template: TemplatePoints,
    pet_method_resolved: str,
    required_vars: Sequence[str],
    fit: str,
    limit_runs: Optional[int],
    force: bool,
    baseline_start_year: Optional[int],
    baseline_end_year: Optional[int],
    archive_root: Optional[Path],
    output_tag: Optional[str],
    group_pixels: int,
    point_groups: Optional[List[np.ndarray]],
    rsds_bias_adjustment: Optional[RsdsEra5BiasAdjustment],
    rsds_bias_adjustment_tag: str,
    rsdsbiasadjust_nat_scens_mode: str = RSDSBIASADJUST_NAT_SCENS,
    rsdsbiasadjust_nat_suffix: str = RSDSBIASADJUST_NAT_SCENARIO_SUFFIX,
) -> Tuple[xr.Dataset, Path]:
    """
    Return (baseline_fit_dataset, baseline_fit_file).
    Loads cached baseline file if present unless --force.
    """
    fit_path = _baseline_fit_path(
        baseline_root,
        region=region,
        scale=scale,
        fit=fit,
        pet=pet_method_resolved,
        baseline_source=baseline_source_key,
        baseline_pooling=baseline_pooling,
        baseline_id=baseline_id,
        output_tag=output_tag,
        group_pixels=group_pixels,
        rsds_bias_adjustment_tag=rsds_bias_adjustment_tag,
    )
    if fit_path.exists() and not force:
        _log(f"Loading cached baseline fit: {fit_path}")
        ds_fit = xr.open_dataset(fit_path).load()
        return ds_fit, fit_path

    if not baseline_files:
        raise RuntimeError(
            "No baseline files found and no cached baseline fit is available; cannot build baseline."
        )

    if fit_path.exists() and force and archive_root is not None:
        archive_root.mkdir(parents=True, exist_ok=True)
        archived = _archive_existing_path(
            fit_path, archive_root=archive_root, base_dir=_deriv_root(baseline_root, output_tag=output_tag)
        )
        if archived is not None:
            _log(f"Archived baseline fit -> {archived}")

    ds_fit = _fit_baseline_params(
        baseline_files,
        baseline_source_key=baseline_source_key,
        baseline_pooling=baseline_pooling,
        template=template,
        pet_method_resolved=pet_method_resolved,
        required_vars=required_vars,
        scale=scale,
        fit=fit,
        limit_runs=limit_runs,
        baseline_label=baseline_label,
        baseline_start_year=baseline_start_year,
        baseline_end_year=baseline_end_year,
        group_pixels=group_pixels,
        point_groups=point_groups,
        rsds_bias_adjustment=rsds_bias_adjustment,
        rsdsbiasadjust_nat_scens_mode=rsdsbiasadjust_nat_scens_mode,
        rsdsbiasadjust_nat_suffix=rsdsbiasadjust_nat_suffix,
    )
    ds_fit.attrs["baseline_pooling"] = str(baseline_pooling)
    ds_fit.attrs["baseline_source_key"] = str(baseline_source_key)
    ds_fit.attrs["baseline_strategy"] = f"{baseline_source_key}:{baseline_pooling}"
    ds_fit.attrs["baseline_id"] = str(baseline_id)
    ds_fit.attrs["group_pixels"] = int(group_pixels)
    ds_fit.attrs["rsds_bias_adjustment_tag"] = str(rsds_bias_adjustment_tag)

    fit_path.parent.mkdir(parents=True, exist_ok=True)
    ds_fit.to_netcdf(fit_path)
    _log(f"✓ Wrote baseline fit file: {fit_path}")
    return ds_fit, fit_path


# -----------------------------------------------------------------------------
# Applying baseline params to compute SPEI
# -----------------------------------------------------------------------------
def _apply_baseline_to_wb_roll(
    wb_roll: xr.DataArray,
    *,
    baseline: xr.Dataset,
    fit: str,
) -> xr.DataArray:
    """
    Standardize WB_roll(time, point) -> SPEI(time, point) using baseline params.

    - zscore: (x - mu_m) / sigma_m
    - loglogistic: pglo(x; xi, alpha, kappa) -> qnorm, with zscore fallback
    """
    fit_norm = fit.lower()
    if fit_norm not in {"zscore", "loglogistic"}:
        raise ValueError("fit must be one of: zscore, loglogistic")

    x = np.asarray(wb_roll.values, dtype=np.float64 if fit_norm == "loglogistic" else np.float32)
    months = np.asarray(wb_roll["time"].dt.month.values, dtype=int)

    mu = np.asarray(baseline["mu"].values, dtype=np.float64)
    sigma = np.asarray(baseline["sigma"].values, dtype=np.float64)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)

    out = np.full(x.shape, np.nan, dtype=np.float32)

    if fit_norm == "zscore" or ("xi" not in baseline) or ("alpha" not in baseline) or ("kappa" not in baseline):
        for m in range(1, 13):
            sel = (months == m)
            if not np.any(sel):
                continue
            out[sel, :] = ((x[sel, :] - mu[m - 1, :]) / sigma[m - 1, :]).astype(np.float32)
    else:
        xi = np.asarray(baseline["xi"].values, dtype=np.float64)
        alpha = np.asarray(baseline["alpha"].values, dtype=np.float64)
        kappa = np.asarray(baseline["kappa"].values, dtype=np.float64)

        SMALL = 1e-15
        EPS = 1e-15

        for m in range(1, 13):
            sel = (months == m)
            if not np.any(sel):
                continue
            xm = x[sel, :]  # (n, points)

            xi_m = xi[m - 1, :]
            a_m = alpha[m - 1, :]
            k_m = kappa[m - 1, :]

            bad = (~np.isfinite(xi_m)) | (~np.isfinite(a_m)) | (a_m <= 0) | (~np.isfinite(k_m))

            # Compute pglo CDF with broadcasting
            Y = (xm - xi_m[None, :]) / a_m[None, :]
            K = np.broadcast_to(k_m[None, :], xm.shape)
            F = np.full_like(Y, np.nan, dtype=np.float64)

            m0 = (~bad)[None, :] & (np.abs(K) < 1e-12)
            if np.any(m0):
                F[m0] = 1.0 / (1.0 + np.exp(-Y[m0]))

            # non-zero kappa
            ARG = 1.0 - K * Y
            m1 = (~bad)[None, :] & (np.abs(K) >= 1e-12) & (ARG > SMALL)
            if np.any(m1):
                Y2 = np.empty_like(ARG)
                Y2[m1] = -np.log(ARG[m1]) / K[m1]
                F[m1] = 1.0 / (1.0 + np.exp(-Y2[m1]))

            m2 = (~bad)[None, :] & (np.abs(K) >= 1e-12) & ~(ARG > SMALL)
            if np.any(m2):
                F[m2] = np.where(K[m2] < 0.0, 0.0, 1.0)

            F = np.clip(F, EPS, 1.0 - EPS)
            zm = _norm_ppf(F).astype(np.float32)

            if np.any(bad):
                zm[:, bad] = ((xm[:, bad] - mu[m - 1, bad]) / sigma[m - 1, bad]).astype(np.float32)

            out[sel, :] = zm

    da = xr.DataArray(
        out.astype(np.float32),
        coords={"time": wb_roll["time"], "point": wb_roll["point"]},
        dims=("time", "point"),
        name=f"SPEI{int(baseline.attrs.get('spei_scale_months', 0) or 0)}",
    )
    return da


def _seconds_in_month(time: xr.DataArray) -> xr.DataArray:
    """Return seconds per month for a time coordinate."""
    # Works for both datetime64 and cftime calendars via xarray's dt accessor.
    days = time.dt.days_in_month.astype("float64")
    return days * 86400.0


def _precip_monthly_mm(pr: xr.DataArray) -> xr.DataArray:
    """
    Convert precipitation to monthly totals in mm.
    Supports common CMIP-style units:
      - kg m-2 s-1  (flux)  -> multiply by seconds/month
      - mm/day      (rate)  -> multiply by days/month
      - m           (depth) -> *1000
      - mm or kg m-2 (already accumulated) -> passthrough
    """
    units = str(pr.attrs.get("units", "")).lower().strip()
    units_compact = units.replace(" ", "")

    def _mark(out: xr.DataArray, desc: str) -> xr.DataArray:
        out.attrs = dict(pr.attrs)
        out.attrs["units"] = "mm"
        out.attrs["description"] = desc
        return out

    # Flux in kg m-2 s-1 (1 kg m-2 == 1 mm)
    if ("kg" in units_compact and "m-2" in units_compact and "s-1" in units_compact) or ("kgm-2s-1" in units_compact):
        sec = _seconds_in_month(pr["time"])
        return _mark(pr * sec, "Monthly precipitation total derived from flux (kg m-2 s-1)")

    # Rate in mm/day
    if "mm/day" in units or "mmd-1" in units_compact or "mmd^-1" in units_compact or "mmd-1" in units_compact:
        days = pr["time"].dt.days_in_month.astype("float64")
        return _mark(pr * days, "Monthly precipitation total derived from mm/day rate")

    # Depth in meters
    if units_compact in {"m", "meter", "metre"} or units.endswith(" m"):
        return _mark(pr * 1000.0, "Monthly precipitation depth converted from meters to mm")

    # Heuristic for missing/unknown units: treat very small values as flux (kg m-2 s-1)
    try:
        sample = float(np.nanmedian(np.asarray(pr.values).ravel()))
    except Exception:
        sample = np.nan
    if (units == "" or units is None or units_compact == "") and np.isfinite(sample) and abs(sample) < 1e-2:
        sec = _seconds_in_month(pr["time"])
        return _mark(pr * sec, "Monthly precipitation total derived from flux (heuristic, units missing)")

    # Already monthly total in mm or kg m-2
    out = pr.astype("float32")
    out.attrs = dict(pr.attrs)
    if out.attrs.get("units", "").strip() == "":
        out.attrs["units"] = "mm"
    out.attrs.setdefault("description", "Monthly precipitation total (assumed)")
    return out


def _compute_speix_bundle_for_dataset(
    ds: xr.Dataset,
    *,
    template: TemplatePoints,
    required_vars: Sequence[str],
    pet_method_resolved: str,
    baseline: xr.Dataset,
    scale: int,
    fit: str,
    point_groups: Optional[List[np.ndarray]],
    group_pixels: int,
    rsds_bias_adjustment: Optional[RsdsEra5BiasAdjustment],
    rsds_bias_hold_first_reference_year_offsets: bool = False,
) -> Tuple[xr.Dataset, Dict[str, Tuple[np.ndarray, np.ndarray, Optional[str]]]]:
    ds_n = _ensure_lon_0_360(ds)
    missing = [v for v in required_vars if v not in ds_n]
    if missing:
        raise KeyError(f"Missing required variables: {missing}")

    sub = _select_template_points(ds_n, template=template, variables=required_vars).load()
    sub = _apply_point_grouping(sub, point_groups=point_groups, group_pixels=group_pixels)
    sub = _apply_rsds_bias_adjustment_to_subdataset(
        sub,
        adjustment=rsds_bias_adjustment,
        hold_first_reference_year_offsets=rsds_bias_hold_first_reference_year_offsets,
    )

    # Capture region-mean input series (pr + PET driver variables) for comparison plotting
    vars_for_inputs = [VAR_PR] + [v for v in required_vars if v != VAR_PR]
    input_means: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[str]]] = {}
    try:
        t_vals = np.asarray(sub["time"].values)
        for v in vars_for_inputs:
            if v not in sub:
                continue
            arr = np.asarray(sub[v].values, dtype=float)
            if arr.ndim > 1:
                arr = np.nanmean(arr, axis=tuple(range(1, arr.ndim)))
            input_means[v] = (t_vals, arr, sub[v].attrs.get("units"))
    except Exception:
        input_means = {}

    # Monthly WB (P - PET) from recipe, consistent with SPEI calculation
    wb = recipe_wb_monthly(sub, pet_method_resolved)

    # Monthly precipitation totals (mm)
    pr_mm = _precip_monthly_mm(sub[VAR_PR])

    # Derive PET from WB identity: WB = P - PET => PET = P - WB
    pet_mm = (pr_mm - wb).astype("float32")
    pet_mm.name = "pet_monthly"
    pet_mm.attrs["units"] = "mm"
    pet_mm.attrs["description"] = "Monthly PET derived as P - WB (consistent with recipe WB)"

    # {scale}-month rolling sums (drops first k-1 months)
    wb_roll = recipe_rolling_sum(wb, scale=scale).astype("float32")
    p_roll = recipe_rolling_sum(pr_mm, scale=scale).astype("float32")
    pet_roll = recipe_rolling_sum(pet_mm, scale=scale).astype("float32")

    wb_roll.attrs["scale"] = int(scale)
    p_roll.attrs["scale"] = int(scale)
    pet_roll.attrs["scale"] = int(scale)

    spei = _apply_baseline_to_wb_roll(wb_roll, baseline=baseline, fit=fit).astype("float32").rename(f"spei{scale}")

    try:
        sub.close()
    except Exception:
        pass

    ds_out = xr.Dataset(
        {
            f"spei{scale}": spei,
            f"p{scale}": p_roll.rename(f"p{scale}"),
            f"pet{scale}": pet_roll.rename(f"pet{scale}"),
            f"wb{scale}": wb_roll.rename(f"wb{scale}"),
        },
        coords={
            "time": spei["time"],
            "point": spei["point"],
            "lat": ("point", template.lat.astype(float)),
            "lon": ("point", template.lon.astype(float)),
        },
        attrs={
            "spei_scale_months": int(scale),
            "pet_method": str(pet_method_resolved),
            "spei_fit": str(fit),
            "description": "SPEI bundle: SPEI + accumulated P/PET/WB over the SPEI scale window",
            "group_pixels": int(group_pixels),
            "rsds_bias_adjustment_applied": bool(rsds_bias_adjustment is not None),
            "rsds_bias_adjustment_window_years": (
                int(rsds_bias_adjustment.smoothing_window_years) if rsds_bias_adjustment is not None else 0
            ),
            "rsds_bias_adjustment_nat_mode": (
                "early"
                if (rsds_bias_adjustment is not None and bool(rsds_bias_hold_first_reference_year_offsets))
                else ("full" if rsds_bias_adjustment is not None else "off")
            ),
        },
    )
    return ds_out, input_means


def _compute_speix_bundle_for_file(
    nc_path: Path,
    *,
    template: TemplatePoints,
    required_vars: Sequence[str],
    pet_method_resolved: str,
    baseline: xr.Dataset,
    scale: int,
    fit: str,
    point_groups: Optional[List[np.ndarray]],
    group_pixels: int,
    rsds_bias_adjustment: Optional[RsdsEra5BiasAdjustment],
    rsds_bias_hold_first_reference_year_offsets: bool = False,
) -> Tuple[xr.Dataset, Dict[str, Tuple[np.ndarray, np.ndarray, Optional[str]]]]:
    ds: Optional[xr.Dataset] = None
    try:
        ds = _open_dataset_safe(nc_path, decode_times=True)
        bundle, input_means = _compute_speix_bundle_for_dataset(
            ds,
            template=template,
            required_vars=required_vars,
            pet_method_resolved=pet_method_resolved,
            baseline=baseline,
            scale=scale,
            fit=fit,
            point_groups=point_groups,
            group_pixels=group_pixels,
            rsds_bias_adjustment=rsds_bias_adjustment,
            rsds_bias_hold_first_reference_year_offsets=rsds_bias_hold_first_reference_year_offsets,
        )
        return bundle, input_means
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass


def _filter_spei_output(
    spei: xr.Dataset | xr.DataArray,
    *,
    out_start_year: Optional[int],
    out_end_year: Optional[int],
    months: Optional[Tuple[int, ...]],
) -> xr.Dataset | xr.DataArray:
    if spei.sizes.get("time", 0) == 0:
        return spei

    mask = xr.ones_like(spei["time"], dtype=bool)
    years = spei["time"].dt.year
    if out_start_year is not None:
        mask = mask & (years >= int(out_start_year))
    if out_end_year is not None:
        mask = mask & (years <= int(out_end_year))
    if months is not None and len(months) > 0:
        mask = mask & spei["time"].dt.month.isin(list(months))

    spei_f = spei.sel(time=mask)
    # Keep non-standard calendars (e.g., 360_day) in cftime to avoid subtle shifts.
    try:
        time_index = spei_f.indexes["time"]
        cal = getattr(time_index, "calendar", None)
        if cal is None or cal in {"standard", "gregorian", "proleptic_gregorian", "julian"}:
            dt = time_index.to_datetimeindex()
            spei_f = spei_f.assign_coords(time=("time", np.asarray(dt)))
    except Exception:
        pass
    return spei_f


# -----------------------------------------------------------------------------
# Writing segments
# -----------------------------------------------------------------------------
def _sanitize_run_id(run_id: str) -> str:
    # Keep filesystem-friendly run IDs
    bad = [" ", ":", ";", "/", "\\", "|"]
    out = run_id
    for b in bad:
        out = out.replace(b, "_")
    return out


def _sanitize_stack_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "unknown"


def _sanitize_region_file_token(region: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(region).strip().upper())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "REGION"


def _stacked_netcdf_pet_dir(store_dir: Path, *, pet_method: str) -> Path:
    deriv_root = Path(store_dir).expanduser().resolve(strict=False).parent
    pet_tag = _sanitize_run_id(f"pet-{pet_method}")
    if deriv_root.name.lower() == pet_tag.lower():
        return deriv_root
    return deriv_root / pet_tag


def _resolve_point_chunk_size(*, point_size: int, chunk_point: Optional[int]) -> int:
    if point_size <= 0:
        return 1
    if chunk_point is None:
        return int(point_size)
    return max(1, min(int(point_size), int(chunk_point)))


def _build_zarr_encoding(
    ds: xr.Dataset,
    *,
    chunk_run: Optional[int],
    chunk_time: Optional[int],
    chunk_point: Optional[int],
    compression_level: int,
) -> Dict[str, Dict[str, object]]:
    enc: Dict[str, Dict[str, object]] = {}
    dim_chunks: Dict[str, int] = {}
    if chunk_run is not None and "run" in ds.dims:
        dim_chunks["run"] = max(1, min(int(chunk_run), int(ds.sizes["run"])))
    if chunk_time is not None and "time" in ds.dims:
        dim_chunks["time"] = max(1, min(int(chunk_time), int(ds.sizes["time"])))
    if chunk_point is not None and "point" in ds.dims:
        dim_chunks["point"] = _resolve_point_chunk_size(point_size=int(ds.sizes["point"]), chunk_point=chunk_point)

    codec = None
    try:
        if _ZarrZstdCodec is not None:
            codec = _ZarrZstdCodec(level=max(0, int(compression_level)))
    except Exception:
        codec = None

    for name, da in ds.variables.items():
        cfg: Dict[str, object] = {}
        if da.dims:
            chunks: List[int] = []
            for dim in da.dims:
                if dim in dim_chunks:
                    chunks.append(int(dim_chunks[dim]))
                else:
                    chunks = []
                    break
            if chunks:
                cfg["chunks"] = tuple(chunks)
        if name in ds.data_vars and codec is not None:
            try:
                if np.issubdtype(da.dtype, np.number):
                    cfg["compressors"] = [codec]
            except Exception:
                pass
        if cfg:
            enc[name] = cfg
    return enc


def _build_netcdf_encoding(
    ds: xr.Dataset,
    *,
    chunk_run: Optional[int],
    chunk_time: Optional[int],
    chunk_point: Optional[int],
    compression_level: int,
) -> Dict[str, Dict[str, object]]:
    enc: Dict[str, Dict[str, object]] = {}
    dim_chunks: Dict[str, int] = {}
    if chunk_run is not None and "run" in ds.dims:
        dim_chunks["run"] = max(1, min(int(chunk_run), int(ds.sizes["run"])))
    if chunk_time is not None and "time" in ds.dims:
        dim_chunks["time"] = max(1, min(int(chunk_time), int(ds.sizes["time"])))
    if chunk_point is not None and "point" in ds.dims:
        dim_chunks["point"] = _resolve_point_chunk_size(point_size=int(ds.sizes["point"]), chunk_point=chunk_point)

    for name, da in ds.variables.items():
        cfg: Dict[str, object] = {}
        if da.dims:
            chunks: List[int] = []
            for dim in da.dims:
                if dim in dim_chunks:
                    chunks.append(int(dim_chunks[dim]))
                else:
                    chunks = []
                    break
            if chunks:
                cfg["chunksizes"] = tuple(chunks)
        if name in ds.data_vars:
            try:
                if np.issubdtype(da.dtype, np.number):
                    cfg["zlib"] = True
                    cfg["complevel"] = max(0, int(compression_level))
            except Exception:
                pass
        if cfg:
            enc[name] = cfg
    return enc


def _write_spei_segment(
    *,
    store_dir: Path,
    run_id: str,
    region: str,
    scale: int,
    payload: xr.Dataset,
    forcing_label: str,
    baseline_fit_file: Path,
    baseline_source_key: str,
    baseline_pooling: str,
    baseline_label: str,
    pet_method: str,
    fit: str,
    on_existing: str,
    archive_root: Optional[Path],
    pet_in_path: bool,
    group_pixels: int,
    chunk_time: int = 120,
    chunk_point: Optional[int] = None,
    compression_level: int = 0,
) -> Optional[Path]:
    """
    Write one SPEI/P/PET bundle segment group to a Zarr segment store.
    Returns the segment group directory path if written.
    """
    if payload.sizes.get("time", 0) == 0:
        return None

    years = payload["time"].dt.year.values
    start_year = int(np.nanmin(years))
    end_year = int(np.nanmax(years))

    seg_base = f"spei{scale}__{region.upper()}__grid-points__{start_year}-{end_year}"
    pet_tag = _sanitize_run_id(f"pet-{pet_method}")
    if pet_in_path:
        seg_name = f"{seg_base}__{pet_tag}__all"
    else:
        seg_name = f"{seg_base}__all"
    run_id_s = _sanitize_run_id(run_id)

    if pet_in_path:
        group_rel = Path("runs") / run_id_s / pet_tag / seg_name
    else:
        group_rel = Path("runs") / run_id_s / seg_name
    group_abs = store_dir / group_rel

    if group_abs.exists():
        action = on_existing.lower()
        if action == "skip":
            _log(f"    ↷ exists, skip: {group_rel}")
            return None
        if action == "archive":
            if archive_root is None:
                _log(f"    ↷ exists, skip (archive dir missing): {group_rel}")
                return None
            archived = _archive_existing_path(group_abs, archive_root=archive_root, base_dir=store_dir)
            if archived is None:
                _log(f"    ↷ exists, skip (archive failed): {group_rel}")
                return None
            _log(f"    ↷ archived existing -> {archived}")
        elif action == "throwerror":
            raise RuntimeError(
                f"SPEI segment already exists: {group_abs} (run_id={run_id}, pet={pet_method}). "
                "Use --on-existing overwrite|archive|skip to override."
            )
        elif action == "overwrite":
            shutil.rmtree(group_abs)
        else:
            raise ValueError(f"Unknown on_existing policy: {on_existing}")

    ds_out = payload.copy()
    ds_out.attrs.update(
        {
            "description": (
                f"SPEI{scale} standardized using baseline parameters from {baseline_label} "
                f"(bundle includes P/PET/WB accumulations)"
            ),
            "region": region.upper(),
            "operation": "grid-points",
            "forcing": forcing_label,
            "spei_scale_months": int(scale),
            "spei_fit": fit,
            "pet_method": pet_method,
            "baseline_fit_file": str(baseline_fit_file),
            "baseline_source": str(baseline_label),
            "baseline_source_key": str(baseline_source_key),
            "baseline_pooling": str(baseline_pooling),
            "baseline_strategy": f"{baseline_source_key}:{baseline_pooling}",
            "group_pixels": int(group_pixels),
        }
    )

    point_chunk = (
        _resolve_point_chunk_size(point_size=int(ds_out.sizes.get("point", 1)), chunk_point=chunk_point)
        if "point" in ds_out.dims
        else None
    )
    chunk_kwargs: Dict[str, int] = {}
    if "time" in ds_out.dims:
        chunk_kwargs["time"] = max(1, min(int(chunk_time), int(ds_out.sizes["time"])))
    if "point" in ds_out.dims and point_chunk is not None:
        chunk_kwargs["point"] = point_chunk

    # Prefer stable chunking for time series workloads
    try:
        if chunk_kwargs:
            ds_out = ds_out.chunk(chunk_kwargs)
    except Exception:
        pass

    encoding = _build_zarr_encoding(
        ds_out,
        chunk_run=None,
        chunk_time=chunk_kwargs.get("time"),
        chunk_point=chunk_kwargs.get("point"),
        compression_level=int(compression_level),
    )

    store_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        if ZarrUnstableSpecificationWarning is not None:
            warnings.filterwarnings("ignore", category=ZarrUnstableSpecificationWarning)
        ds_out.to_zarr(
            store_dir,
            group=str(group_rel).replace("\\", "/"),
            consolidated=False,
            encoding=encoding if encoding else None,
        )

    return group_abs


@dataclass
class _StackedGroupState:
    group_rel: Path
    group_abs: Path
    prepared: bool = False
    existing_runs: Set[str] = field(default_factory=set)
    pending: List[xr.Dataset] = field(default_factory=list)
    pending_run_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


class RunStackedWriter:
    """
    Incremental run-stacked writer:
      stacked/<FORCING>/<PET>/spei{scale}__{REGION}__grid-points__{START}-{END}__all
    """

    def __init__(
        self,
        *,
        store_dir: Path,
        forcing_label: str,
        region: str,
        scale: int,
        pet_method: str,
        fit: str,
        on_existing: str,
        archive_root: Optional[Path],
        group_pixels: int,
        chunk_run: int,
        chunk_time: int,
        chunk_point: Optional[int],
        compression_level: int,
    ) -> None:
        self.store_dir = store_dir
        self.forcing_label = str(forcing_label)
        self.region = str(region)
        self.scale = int(scale)
        self.pet_method = str(pet_method)
        self.fit = str(fit)
        self.on_existing = str(on_existing).strip().lower()
        self.archive_root = archive_root
        self.group_pixels = int(group_pixels)
        self.chunk_run = max(1, int(chunk_run))
        self.chunk_time = max(1, int(chunk_time))
        self.chunk_point = None if chunk_point is None else max(1, int(chunk_point))
        self.compression_level = int(compression_level)
        self._groups: Dict[str, _StackedGroupState] = {}
        self._store_touched = False

    def _group_rel(self, *, start_year: int, end_year: int) -> Path:
        seg_name = (
            f"spei{self.scale}__{self.region.upper()}__grid-points__{int(start_year)}-{int(end_year)}__all"
        )
        forcing_tag = _sanitize_stack_token(self.forcing_label)
        pet_tag = _sanitize_run_id(f"pet-{self.pet_method}")
        return Path("stacked") / forcing_tag / pet_tag / seg_name

    def _read_existing_run_ids(self, state: _StackedGroupState) -> Set[str]:
        if not state.group_abs.exists():
            return set()
        ds_old: Optional[xr.Dataset] = None
        try:
            ds_old = xr.open_zarr(
                self.store_dir,
                group=str(state.group_rel).replace("\\", "/"),
                consolidated=False,
            )
            if "run" in ds_old.coords:
                vals = np.asarray(ds_old["run"].values)
                return {str(v) for v in vals.ravel().tolist()}
        except Exception:
            return set()
        finally:
            if ds_old is not None:
                try:
                    ds_old.close()
                except Exception:
                    pass
        return set()

    def _prepare_group(self, state: _StackedGroupState) -> bool:
        if state.prepared:
            return True

        group_exists = state.group_abs.exists()
        if group_exists:
            if self.on_existing == "throwerror":
                raise RuntimeError(
                    f"Run-stacked SPEI group already exists: {state.group_abs}. "
                    "Use --on-existing overwrite|archive|skip to override."
                )
            if self.on_existing == "overwrite":
                shutil.rmtree(state.group_abs)
            elif self.on_existing == "archive":
                if self.archive_root is None:
                    _log(f"    ↷ run-stacked exists, skip (archive dir missing): {state.group_rel}")
                    return False
                archived = _archive_existing_path(state.group_abs, archive_root=self.archive_root, base_dir=self.store_dir)
                if archived is None:
                    _log(f"    ↷ run-stacked exists, skip (archive failed): {state.group_rel}")
                    return False
                _log(f"    ↷ archived existing run-stacked group -> {archived}")
            elif self.on_existing == "skip":
                state.existing_runs = self._read_existing_run_ids(state)
            else:
                raise ValueError(f"Unknown on_existing policy: {self.on_existing}")

        state.prepared = True
        return True

    @staticmethod
    def _meta_var(values: List[str]) -> xr.DataArray:
        width = max(1, max((len(v) for v in values), default=1))
        return xr.DataArray(np.asarray(values, dtype=f"<U{width}"), dims=("run",))

    def write_run(
        self,
        *,
        run_id: str,
        payload: xr.Dataset,
        baseline_fit_file: Path,
        baseline_source_key: str,
        baseline_pooling: str,
        baseline_strategy: str,
    ) -> Optional[Path]:
        if payload.sizes.get("time", 0) == 0:
            return None

        years = payload["time"].dt.year.values
        start_year = int(np.nanmin(years))
        end_year = int(np.nanmax(years))
        group_rel = self._group_rel(start_year=start_year, end_year=end_year)
        key = str(group_rel).replace("\\", "/")
        if key not in self._groups:
            self._groups[key] = _StackedGroupState(group_rel=group_rel, group_abs=self.store_dir / group_rel)
        state = self._groups[key]

        if not self._prepare_group(state):
            return None

        run_id_s = str(run_id)
        if run_id_s in state.existing_runs:
            if self.on_existing == "skip":
                _log(f"    ↷ run-stacked exists, skip run={run_id_s}: {group_rel}")
                return None
            raise RuntimeError(f"Duplicate run '{run_id_s}' found in existing run-stacked group: {group_rel}")
        if run_id_s in state.pending_run_ids:
            raise RuntimeError(f"Duplicate run '{run_id_s}' queued twice in run-stacked writer: {group_rel}")

        ds_run = payload.copy()
        ds_run = ds_run.expand_dims(run=[run_id_s])
        ds_run["baseline_source_key"] = self._meta_var([str(baseline_source_key)])
        ds_run["baseline_pooling"] = self._meta_var([str(baseline_pooling)])
        ds_run["baseline_fit_file"] = self._meta_var([str(baseline_fit_file)])
        ds_run["baseline_strategy"] = self._meta_var([str(baseline_strategy)])
        ds_run["forcing"] = self._meta_var([str(self.forcing_label)])
        ds_run.attrs.update(
            {
                "description": (
                    f"Run-stacked SPEI{self.scale} standardized output (bundle includes P/PET/WB accumulations)"
                ),
                "region": self.region.upper(),
                "operation": "grid-points",
                "forcing": self.forcing_label,
                "spei_scale_months": int(self.scale),
                "spei_fit": self.fit,
                "pet_method": self.pet_method,
                "segments_layout": SEGMENTS_LAYOUT_RUN_STACKED,
                "speix_schema": 2,
                "chunk_run": int(self.chunk_run),
                "chunk_time": int(self.chunk_time),
                "chunk_point": int(
                    _resolve_point_chunk_size(
                        point_size=int(ds_run.sizes.get("point", 1)),
                        chunk_point=self.chunk_point,
                    )
                ),
                "compression_level": int(self.compression_level),
                "group_pixels": int(self.group_pixels),
            }
        )

        state.pending.append(ds_run)
        state.pending_run_ids.append(run_id_s)
        if len(state.pending) >= self.chunk_run:
            self._flush_group(state)
        return state.group_abs

    def _flush_group(self, state: _StackedGroupState) -> None:
        if not state.pending:
            return

        ds_batch = xr.concat(
            state.pending,
            dim="run",
            coords="minimal",
            compat="override",
            join="outer",
        )
        point_chunk = (
            _resolve_point_chunk_size(
                point_size=int(ds_batch.sizes.get("point", 1)),
                chunk_point=self.chunk_point,
            )
            if "point" in ds_batch.dims
            else None
        )
        chunk_kwargs: Dict[str, int] = {}
        if "run" in ds_batch.dims:
            chunk_kwargs["run"] = max(1, min(int(self.chunk_run), int(ds_batch.sizes["run"])))
        if "time" in ds_batch.dims:
            chunk_kwargs["time"] = max(1, min(int(self.chunk_time), int(ds_batch.sizes["time"])))
        if "point" in ds_batch.dims and point_chunk is not None:
            chunk_kwargs["point"] = point_chunk
        try:
            if chunk_kwargs:
                ds_batch = ds_batch.chunk(chunk_kwargs)
        except Exception:
            pass

        encoding = _build_zarr_encoding(
            ds_batch,
            chunk_run=chunk_kwargs.get("run"),
            chunk_time=chunk_kwargs.get("time"),
            chunk_point=chunk_kwargs.get("point"),
            compression_level=self.compression_level,
        )
        append_mode = state.group_abs.exists()
        write_kwargs: Dict[str, object] = {
            "store": self.store_dir,
            "group": str(state.group_rel).replace("\\", "/"),
            "consolidated": False,
        }
        if append_mode:
            write_kwargs["mode"] = "a"
            write_kwargs["append_dim"] = "run"
        else:
            write_kwargs["mode"] = "w"
            if encoding:
                write_kwargs["encoding"] = encoding

        self.store_dir.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            if ZarrUnstableSpecificationWarning is not None:
                warnings.filterwarnings("ignore", category=ZarrUnstableSpecificationWarning)
            ds_batch.to_zarr(**write_kwargs)

        state.existing_runs.update(state.pending_run_ids)
        state.pending.clear()
        state.pending_run_ids.clear()
        self._store_touched = True
        manifest_path = self.store_dir.parent / "stacked_manifests" / state.group_rel / "manifest.json"
        _write_zarr_json(
            manifest_path,
            {
                "segments_layout": SEGMENTS_LAYOUT_RUN_STACKED,
                "speix_schema": 2,
                "forcing": self.forcing_label,
                "pet_method": self.pet_method,
                "region": self.region.upper(),
                "spei_scale_months": int(self.scale),
                "chunk_run": int(self.chunk_run),
                "chunk_time": int(self.chunk_time),
                "chunk_point": int(chunk_kwargs.get("point", 1)),
                "compression_level": int(self.compression_level),
                "run_count": int(len(state.existing_runs)),
                "updated_utc": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
        )

    def finalize(self) -> Optional[Path]:
        for state in self._groups.values():
            self._flush_group(state)
        if self._store_touched:
            return self.store_dir
        return None


class NetCDFRunStackedWriter:
    """
    Incremental run-stacked writer using one NetCDF per forcing x region x PET x year-span:
      <pet-dir>/stacked/<FORCING>/<FORCING>__spei{scale}__{REGION}__grid-points__{START}-{END}__all.nc
    """

    def __init__(
        self,
        *,
        store_dir: Path,
        forcing_label: str,
        region: str,
        scale: int,
        pet_method: str,
        fit: str,
        on_existing: str,
        archive_root: Optional[Path],
        group_pixels: int,
        chunk_run: int,
        chunk_time: int,
        chunk_point: Optional[int],
        compression_level: int,
    ) -> None:
        self.store_dir = store_dir
        self.forcing_label = str(forcing_label)
        self.region = str(region)
        self.scale = int(scale)
        self.pet_method = str(pet_method)
        self.fit = str(fit)
        self.on_existing = str(on_existing).strip().lower()
        self.archive_root = archive_root
        self.group_pixels = int(group_pixels)
        self.chunk_run = max(1, int(chunk_run))
        self.chunk_time = max(1, int(chunk_time))
        self.chunk_point = None if chunk_point is None else max(1, int(chunk_point))
        self.compression_level = int(compression_level)
        self.pet_dir = _stacked_netcdf_pet_dir(store_dir, pet_method=self.pet_method)
        self._groups: Dict[str, _StackedGroupState] = {}
        self._touched_paths: Set[Path] = set()

    def _group_rel(self, *, start_year: int, end_year: int) -> Path:
        forcing_tag = _sanitize_stack_token(self.forcing_label)
        region_tag = _sanitize_region_file_token(self.region)
        file_name = (
            f"{forcing_tag}__spei{self.scale}__{region_tag}__grid-points__"
            f"{int(start_year)}-{int(end_year)}__all.nc"
        )
        return Path("stacked") / forcing_tag / file_name

    @staticmethod
    def _extract_run_ids(ds_old: xr.Dataset) -> Set[str]:
        if "run" in ds_old.coords:
            vals = np.asarray(ds_old["run"].values)
            return {str(v) for v in vals.ravel().tolist()}
        raw = ds_old.attrs.get("run_ids")
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return {str(v) for v in parsed}
            except Exception:
                return {part for part in raw.split(",") if part}
        return set()

    def _read_existing_run_ids(self, state: _StackedGroupState) -> Set[str]:
        if not state.group_abs.exists():
            return set()
        ds_old: Optional[xr.Dataset] = None
        try:
            ds_old = xr.open_dataset(state.group_abs, decode_times=False, engine="netcdf4")
            return self._extract_run_ids(ds_old)
        except Exception:
            return set()
        finally:
            if ds_old is not None:
                try:
                    ds_old.close()
                except Exception:
                    pass

    def _prepare_group(self, state: _StackedGroupState) -> bool:
        if state.prepared:
            return True

        group_exists = state.group_abs.exists()
        if group_exists:
            if self.on_existing == "throwerror":
                raise RuntimeError(
                    f"Run-stacked SPEI NetCDF already exists: {state.group_abs}. "
                    "Use --on-existing overwrite|archive|skip to override."
                )
            if self.on_existing == "overwrite":
                state.group_abs.unlink()
            elif self.on_existing == "archive":
                if self.archive_root is None:
                    _log(f"    ↷ run-stacked NetCDF exists, skip (archive dir missing): {state.group_rel}")
                    return False
                archived = _archive_existing_path(state.group_abs, archive_root=self.archive_root, base_dir=self.pet_dir)
                if archived is None:
                    _log(f"    ↷ run-stacked NetCDF exists, skip (archive failed): {state.group_rel}")
                    return False
                _log(f"    ↷ archived existing run-stacked NetCDF -> {archived}")
            elif self.on_existing == "skip":
                state.existing_runs = self._read_existing_run_ids(state)
            else:
                raise ValueError(f"Unknown on_existing policy: {self.on_existing}")

        state.prepared = True
        return True

    def write_run(
        self,
        *,
        run_id: str,
        payload: xr.Dataset,
        baseline_fit_file: Path,
        baseline_source_key: str,
        baseline_pooling: str,
        baseline_strategy: str,
    ) -> Optional[Path]:
        if payload.sizes.get("time", 0) == 0:
            return None

        years = payload["time"].dt.year.values
        start_year = int(np.nanmin(years))
        end_year = int(np.nanmax(years))
        group_rel = self._group_rel(start_year=start_year, end_year=end_year)
        key = str(group_rel).replace("\\", "/")
        if key not in self._groups:
            self._groups[key] = _StackedGroupState(group_rel=group_rel, group_abs=self.pet_dir / group_rel)
        state = self._groups[key]

        if not self._prepare_group(state):
            return None

        run_id_s = str(run_id)
        if run_id_s in state.existing_runs:
            if self.on_existing == "skip":
                _log(f"    ↷ run-stacked NetCDF exists, skip run={run_id_s}: {group_rel}")
                return None
            raise RuntimeError(f"Duplicate run '{run_id_s}' found in existing run-stacked NetCDF: {group_rel}")
        if run_id_s in state.pending_run_ids:
            raise RuntimeError(f"Duplicate run '{run_id_s}' queued twice in NetCDF run-stacked writer: {group_rel}")

        ds_run = payload.copy().expand_dims(run=[run_id_s])
        state.pending.append(ds_run)
        state.pending_run_ids.append(run_id_s)

        meta = {
            "baseline_source_key": str(baseline_source_key),
            "baseline_pooling": str(baseline_pooling),
            "baseline_strategy": str(baseline_strategy),
            "baseline_fit_file": str(baseline_fit_file),
        }
        if not state.metadata:
            state.metadata.update(meta)
        else:
            for key_name, value in meta.items():
                if key_name in state.metadata and state.metadata[key_name] != value:
                    state.metadata[key_name] = "mixed"
                else:
                    state.metadata.setdefault(key_name, value)

        if len(state.pending) >= self.chunk_run:
            self._flush_group(state)
        return state.group_abs

    def _flush_group(self, state: _StackedGroupState) -> None:
        if not state.pending:
            return

        datasets: List[xr.Dataset] = []
        ds_old: Optional[xr.Dataset] = None
        if state.group_abs.exists():
            try:
                ds_old = xr.open_dataset(state.group_abs, decode_times=True, engine="netcdf4")
                datasets.append(ds_old.load())
            finally:
                if ds_old is not None:
                    try:
                        ds_old.close()
                    except Exception:
                        pass
        datasets.extend(state.pending)

        if len(datasets) == 1:
            ds_batch = datasets[0].copy()
        else:
            ds_batch = xr.concat(
                datasets,
                dim="run",
                coords="minimal",
                compat="override",
                join="outer",
            )

        point_chunk = (
            _resolve_point_chunk_size(
                point_size=int(ds_batch.sizes.get("point", 1)),
                chunk_point=self.chunk_point,
            )
            if "point" in ds_batch.dims
            else None
        )
        chunk_kwargs: Dict[str, int] = {}
        if "run" in ds_batch.dims:
            chunk_kwargs["run"] = max(1, min(int(self.chunk_run), int(ds_batch.sizes["run"])))
        if "time" in ds_batch.dims:
            chunk_kwargs["time"] = max(1, min(int(self.chunk_time), int(ds_batch.sizes["time"])))
        if "point" in ds_batch.dims and point_chunk is not None:
            chunk_kwargs["point"] = point_chunk
        try:
            if chunk_kwargs:
                ds_batch = ds_batch.chunk(chunk_kwargs)
        except Exception:
            pass

        run_ids = [str(v) for v in np.asarray(ds_batch["run"].values).ravel().tolist()] if "run" in ds_batch.coords else []
        ds_batch.attrs.update(
            {
                "description": (
                    f"Run-stacked SPEI{self.scale} standardized output (bundle includes P/PET/WB accumulations)"
                ),
                "region": self.region.upper(),
                "operation": "grid-points",
                "forcing": self.forcing_label,
                "forcing_label": self.forcing_label,
                "spei_scale_months": int(self.scale),
                "scale": int(self.scale),
                "spei_fit": self.fit,
                "pet_method": self.pet_method,
                "segments_layout": SEGMENTS_LAYOUT_RUN_STACKED,
                "output_format": OUTPUT_FORMAT_NETCDF,
                "speix_schema": 2,
                "chunk_run": int(self.chunk_run),
                "chunk_time": int(self.chunk_time),
                "chunk_point": int(chunk_kwargs.get("point", 1)),
                "compression_level": int(self.compression_level),
                "group_pixels": int(self.group_pixels),
                "run_count": int(len(run_ids)),
                "run_ids": json.dumps(run_ids),
                "updated_utc": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        )
        for key_name, value in state.metadata.items():
            ds_batch.attrs[str(key_name)] = str(value)

        encoding = _build_netcdf_encoding(
            ds_batch,
            chunk_run=chunk_kwargs.get("run"),
            chunk_time=chunk_kwargs.get("time"),
            chunk_point=chunk_kwargs.get("point"),
            compression_level=self.compression_level,
        )

        state.group_abs.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f"{state.group_abs.stem}.",
                suffix=".tmp.nc",
                dir=state.group_abs.parent,
                delete=False,
            ) as handle:
                tmp_path = Path(handle.name)
            ds_batch.to_netcdf(
                tmp_path,
                engine="netcdf4",
                format="NETCDF4",
                encoding=encoding if encoding else None,
            )
            os.replace(tmp_path, state.group_abs)
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

        state.existing_runs = set(run_ids)
        state.pending.clear()
        state.pending_run_ids.clear()
        self._touched_paths.add(state.group_abs)

    def finalize(self) -> Optional[Path]:
        for state in self._groups.values():
            self._flush_group(state)
        if self._touched_paths:
            return self.pet_dir
        return None


def _consolidate_zarr_store_metadata(store_dir: Path) -> None:
    if zarr is None:
        _log(f"⚠️ zarr package unavailable; skip metadata consolidation for {store_dir}")
        return
    try:
        zarr.consolidate_metadata(store_dir)
    except Exception as exc:
        _log(f"⚠️ metadata consolidation failed for {store_dir}: {exc}")


def _find_existing_segments(
    store_dir: Path,
    *,
    run_ids: Sequence[str],
    region: str,
    scale: int,
) -> List[Path]:
    existing: List[Path] = []
    prefix = f"spei{scale}__{region.upper()}__grid-points__"
    for run_id in run_ids:
        run_dir = store_dir / "runs" / _sanitize_run_id(run_id)
        if not run_dir.exists():
            continue
        for child in run_dir.iterdir():
            if not child.is_dir():
                continue
            # Old layout: runs/<run>/<segment>
            if child.name.startswith(prefix) and child.name.endswith("__all"):
                existing.append(child)
                continue
            # New layout for PET multi-run: runs/<run>/pet-<method>/<segment>
            if child.name.lower().startswith("pet-"):
                for g in child.iterdir():
                    if g.is_dir() and g.name.startswith(prefix) and g.name.endswith("__all"):
                        existing.append(g)
    return existing


def _find_existing_stacked_segments(
    store_dir: Path,
    *,
    run_ids: Sequence[str],
    region: str,
    scale: int,
) -> List[Path]:
    existing: List[Path] = []
    stacked_root = store_dir / "stacked"
    if not stacked_root.exists():
        return existing
    wanted_runs = {str(r) for r in run_ids}
    prefix = f"spei{int(scale)}__{region.upper()}__grid-points__"

    for forcing_dir in stacked_root.iterdir():
        if not forcing_dir.is_dir():
            continue
        for pet_dir in forcing_dir.iterdir():
            if not pet_dir.is_dir():
                continue
            for seg_dir in pet_dir.iterdir():
                if not seg_dir.is_dir():
                    continue
                name = seg_dir.name
                if not (name.startswith(prefix) and name.endswith("__all")):
                    continue
                if not wanted_runs:
                    existing.append(seg_dir)
                    continue
                rel = seg_dir.relative_to(store_dir).as_posix()
                ds_old: Optional[xr.Dataset] = None
                try:
                    ds_old = xr.open_zarr(store_dir, group=rel, consolidated=False)
                    if "run" not in ds_old.coords:
                        continue
                    runs_present = {str(v) for v in np.asarray(ds_old["run"].values).ravel().tolist()}
                    if runs_present.intersection(wanted_runs):
                        existing.append(seg_dir)
                except Exception:
                    continue
                finally:
                    if ds_old is not None:
                        try:
                            ds_old.close()
                        except Exception:
                            pass
    return existing


def _find_existing_stacked_netcdf_segments(
    store_dir: Path,
    *,
    run_ids: Sequence[str],
    region: str,
    scale: int,
) -> List[Path]:
    existing: List[Path] = []
    wanted_runs = {str(r) for r in run_ids}
    region_root = Path(store_dir).expanduser().resolve(strict=False).parent
    search_roots: List[Path] = []
    if region_root.name.lower().startswith("pet-"):
        search_roots.append(region_root)
    else:
        search_roots.append(region_root)
        search_roots.extend(sorted(p for p in region_root.glob("pet-*") if p.is_dir()))

    seen: Set[str] = set()
    needle = f"__spei{int(scale)}__"
    for pet_root in search_roots:
        stacked_root = pet_root / "stacked"
        if not stacked_root.exists():
            continue
        for nc_path in sorted(stacked_root.rglob("*.nc")):
            key = str(nc_path.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            if needle not in nc_path.name or not nc_path.name.endswith("__all.nc"):
                continue
            if not wanted_runs:
                existing.append(nc_path)
                continue
            ds_old: Optional[xr.Dataset] = None
            try:
                ds_old = xr.open_dataset(nc_path, decode_times=False, engine="netcdf4")
                runs_present = NetCDFRunStackedWriter._extract_run_ids(ds_old)
                if runs_present.intersection(wanted_runs):
                    existing.append(nc_path)
            except Exception:
                continue
            finally:
                if ds_old is not None:
                    try:
                        ds_old.close()
                    except Exception:
                        pass
    return existing


def _find_existing_stacked_outputs(
    store_dir: Path,
    *,
    run_ids: Sequence[str],
    region: str,
    scale: int,
    output_format: str,
) -> List[Path]:
    fmt = _normalize_output_format_token(output_format)
    if fmt == OUTPUT_FORMAT_NETCDF:
        return _find_existing_stacked_netcdf_segments(
            store_dir,
            run_ids=run_ids,
            region=region,
            scale=scale,
        )
    return _find_existing_stacked_segments(
        store_dir,
        run_ids=run_ids,
        region=region,
        scale=scale,
    )


def _make_run_stacked_writer(
    *,
    output_format: str,
    store_dir: Path,
    forcing_label: str,
    region: str,
    scale: int,
    pet_method: str,
    fit: str,
    on_existing: str,
    archive_root: Optional[Path],
    group_pixels: int,
    chunk_run: int,
    chunk_time: int,
    chunk_point: Optional[int],
    compression_level: int,
) -> RunStackedWriter | NetCDFRunStackedWriter:
    fmt = _normalize_output_format_token(output_format)
    common_kwargs = {
        "store_dir": store_dir,
        "forcing_label": forcing_label,
        "region": region,
        "scale": scale,
        "pet_method": pet_method,
        "fit": fit,
        "on_existing": on_existing,
        "archive_root": archive_root,
        "group_pixels": group_pixels,
        "chunk_run": chunk_run,
        "chunk_time": chunk_time,
        "chunk_point": chunk_point,
        "compression_level": compression_level,
    }
    if fmt == OUTPUT_FORMAT_NETCDF:
        return NetCDFRunStackedWriter(**common_kwargs)
    return RunStackedWriter(**common_kwargs)


def _prompt_on_existing(labels: Sequence[str]) -> str:
    if not sys.stdin.isatty():
        _log("Non-interactive session; defaulting to 'skip' for existing SPEI segments.")
        return "skip"
    label_txt = ", ".join(labels)
    while True:
        resp = input(
            f"Existing SPEI segments found in: {label_txt}. "
            "Choose [o]verwrite, [a]rchive, [s]kip, [q]uit: "
        ).strip().lower()
        if resp in {"o", "overwrite"}:
            return "overwrite"
        if resp in {"a", "archive"}:
            return "archive"
        if resp in {"s", "skip"}:
            return "skip"
        if resp in {"q", "quit"}:
            return "quit"

def _summary_pdf_dir(root: Path, *, output_tag: Optional[str] = None) -> Path:
    d = _deriv_root(root, output_tag=output_tag) / "summary_pdfs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.datetime64, datetime)):
        try:
            return np.datetime_as_string(np.datetime64(obj), unit="s")
        except Exception:
            return str(obj)
    return str(obj)


def _save_json(data: dict, path: Path) -> Path:
    """
    Save a dictionary as pretty JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)
    return path


def _save_pdf_and_png(fig, pdf_path: Path, *, dpi: int = 300, bbox_inches: str = "tight") -> Tuple[Path, Path]:
    """
    Save a Matplotlib figure to both PDF and PNG with a shared stem.
    """
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches=bbox_inches)
    png_path = pdf_path.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)
    return pdf_path, png_path


def _month_name(m: int) -> str:
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if 1 <= m <= 12:
        return names[m - 1]
    return str(m)


def _baseline_mean_params_for_month(baseline: xr.Dataset, m: int, *, fit: str) -> Dict[str, float]:
    """Fetch month-specific baseline params for region-mean plotting; fallback to point-mean if needed."""
    mi = int(m) - 1
    out: Dict[str, float] = {}
    mu_mean = None
    sig_mean = None
    if "mu_mean" in baseline and "sigma_mean" in baseline:
        try:
            mu_mean = float(baseline["mu_mean"].isel(month=mi).values)
            sig_mean = float(baseline["sigma_mean"].isel(month=mi).values)
        except Exception:
            mu_mean = None
            sig_mean = None
    if mu_mean is None or not np.isfinite(mu_mean):
        try:
            mu_mean = float(np.nanmean(baseline["mu"].isel(month=mi).values))
        except Exception:
            mu_mean = float("nan")
    if sig_mean is None or not np.isfinite(sig_mean):
        try:
            sig_mean = float(np.nanmean(baseline["sigma"].isel(month=mi).values))
        except Exception:
            sig_mean = float("nan")
    out["mu"] = float(mu_mean)
    out["sigma"] = float(sig_mean if (np.isfinite(sig_mean) and sig_mean > 0) else 1.0)

    fit_norm = (fit or "").lower().strip()
    if fit_norm == "loglogistic":
        xi_m = a_m = k_m = None
        if "xi_mean" in baseline and "alpha_mean" in baseline and "kappa_mean" in baseline:
            try:
                xi_m = float(baseline["xi_mean"].isel(month=mi).values)
                a_m = float(baseline["alpha_mean"].isel(month=mi).values)
                k_m = float(baseline["kappa_mean"].isel(month=mi).values)
            except Exception:
                xi_m = a_m = k_m = None
        if xi_m is None or not np.isfinite(xi_m):
            try:
                xi_m = float(np.nanmean(baseline["xi"].isel(month=mi).values))
            except Exception:
                xi_m = float("nan")
        if a_m is None or not np.isfinite(a_m) or a_m <= 0:
            try:
                a_m = float(np.nanmean(baseline["alpha"].isel(month=mi).values))
            except Exception:
                a_m = float("nan")
        if k_m is None or not np.isfinite(k_m):
            try:
                k_m = float(np.nanmean(baseline["kappa"].isel(month=mi).values))
            except Exception:
                k_m = float("nan")
        out["xi"] = float(xi_m)
        out["alpha"] = float(a_m)
        out["kappa"] = float(k_m)
    return out


def _baseline_point_params_for_month(baseline: xr.Dataset, m: int, point: int, *, fit: str) -> Dict[str, float]:
    """Fetch month-specific baseline params for a single point."""
    mi = int(m) - 1
    pt = int(point)
    out: Dict[str, float] = {}
    try:
        mu = float(baseline["mu"].isel(month=mi, point=pt).values)
    except Exception:
        mu = float("nan")
    try:
        sig = float(baseline["sigma"].isel(month=mi, point=pt).values)
    except Exception:
        sig = float("nan")
    out["mu"] = float(mu)
    out["sigma"] = float(sig if (np.isfinite(sig) and sig > 0) else 1.0)

    fit_norm = (fit or "").lower().strip()
    if fit_norm == "loglogistic":
        try:
            xi = float(baseline["xi"].isel(month=mi, point=pt).values)
        except Exception:
            xi = float("nan")
        try:
            a = float(baseline["alpha"].isel(month=mi, point=pt).values)
        except Exception:
            a = float("nan")
        try:
            k = float(baseline["kappa"].isel(month=mi, point=pt).values)
        except Exception:
            k = float("nan")
        out["xi"] = float(xi)
        out["alpha"] = float(a)
        out["kappa"] = float(k)
    return out


def _write_summary_pdf_multi(
    *,
    root_for_outputs: Path,
    forcing_label: str,
    region: str,
    scale: int,
    pet_method: str,
    fit: str,
    baseline_strategy: str,
    baseline_main: xr.Dataset,
    baseline_label: str,
    bundles: List[Tuple[str, xr.Dataset, xr.Dataset]],
    pivot_year: int,
    era5_bundle: Optional[Tuple[str, xr.Dataset, xr.Dataset]] = None,
    output_tag: Optional[str] = None,
    group_pixels: int,
) -> Optional[Path]:
    """
    Write a summary PDF overlaying all bundles for a forcing (SCENARIO2 or SCENARIO1).
    bundles: list of (run_id, ds_bundle, baseline_used_for_run)
    Also emits a standalone January region-mean excerpt (PDF+PNG).
    """
    if plt is None or PdfPages is None or ListedColormap is None or BoundaryNorm is None or LineCollection is None:
        _log("    ↷ matplotlib unavailable; skipping summary PDF.")
        return None
    if not bundles:
        _log(f"    ↷ no bundles for {forcing_label}; skipping summary PDF.")
        return None

    spei_var = f"spei{scale}"
    p_var = f"p{scale}"
    pet_var = f"pet{scale}"
    # Ensure required vars
    for _, ds, _ in bundles:
        if spei_var not in ds or p_var not in ds or pet_var not in ds:
            _log(f"    ↷ skipping summary for {forcing_label}: missing vars in bundle.")
            return None

    out_dir = _summary_pdf_dir(root_for_outputs, output_tag=output_tag)
    safe_pet = _sanitize_run_id(pet_method)
    safe_fit = (fit or "fit").lower().strip()
    safe_strat = (baseline_strategy or "strategy").lower().strip()
    group_tag = f"grp{int(group_pixels)}"
    out_path = out_dir / f"SUMMARY__{forcing_label}__ALL-RUNS__spei{scale}__{region.upper()}__{safe_pet}__{safe_fit}__{safe_strat}__{group_tag}.pdf"
    forcing_upper = str(forcing_label).upper()
    if forcing_upper == FORCING_SCENARIO1_LABEL:
        jan_excerpt_name = f"EXCERPT__JAN__ALL-RUNS__spei{scale}__{region.upper()}__{safe_pet}__{safe_fit}__{safe_strat}__{group_tag}"
    else:
        jan_excerpt_name = f"EXCERPT__JAN__{forcing_upper}__ALL-RUNS__spei{scale}__{region.upper()}__{safe_pet}__{safe_fit}__{safe_strat}__{group_tag}"
    jan_excerpt_pdf = out_dir / f"{jan_excerpt_name}.pdf"
    _log(f"    ↷ writing summary PDF for {forcing_label} -> {out_path}")

    # Shared axis limits for first page based on region-mean trajectories across all bundles
    p_means_all: List[np.ndarray] = []
    pet_means_all: List[np.ndarray] = []
    years_all: List[np.ndarray] = []
    for _, ds, _ in bundles:
        p_all = np.asarray(ds[p_var].values, dtype=float)
        pet_all = np.asarray(ds[pet_var].values, dtype=float)
        if p_all.size == 0 or pet_all.size == 0:
            continue
        p_means_all.append(np.nanmean(p_all, axis=1))
        pet_means_all.append(np.nanmean(pet_all, axis=1))
        years_all.append(np.asarray(ds["time"].dt.year.values, dtype=int))
    if era5_bundle is not None:
        _, ds_e, _ = era5_bundle
        try:
            p_all = np.asarray(ds_e[p_var].values, dtype=float)
            pet_all = np.asarray(ds_e[pet_var].values, dtype=float)
            if p_all.size > 0 and pet_all.size > 0:
                p_means_all.append(np.nanmean(p_all, axis=1))
                pet_means_all.append(np.nanmean(pet_all, axis=1))
                years_all.append(np.asarray(ds_e["time"].dt.year.values, dtype=int))
        except Exception:
            pass
    if not p_means_all or not pet_means_all:
        _log("    ↷ no finite P/PET values; skipping summary PDF.")
        return None
    p_mean_concat = np.concatenate(p_means_all)
    pet_mean_concat = np.concatenate(pet_means_all)
    p_min = float(np.nanmin(p_mean_concat))
    p_max = float(np.nanmax(p_mean_concat))
    pet_min = float(np.nanmin(pet_mean_concat))
    pet_max = float(np.nanmax(pet_mean_concat))
    px = max(1e-6, (p_max - p_min))
    py = max(1e-6, (pet_max - pet_min))
    p_min -= 0.05 * px
    p_max += 0.05 * px
    pet_min -= 0.05 * py
    pet_max += 0.05 * py

    # Background drought classes (USDM thresholds)
    levels = [-10.0, USDM_SPEI_LEVELS[0], USDM_SPEI_LEVELS[1], USDM_SPEI_LEVELS[2], USDM_SPEI_LEVELS[3], USDM_SPEI_LEVELS[4], 10.0]
    cmap = ListedColormap(["#7f0000", "#d73027", "#f46d43", "#fdae61", "#fff7a8", "#ffffff"])
    norm = BoundaryNorm(levels, ncolors=cmap.N)
    usdm_tick_centers = [
        (-10.0 + USDM_SPEI_LEVELS[0]) / 2.0,
        (USDM_SPEI_LEVELS[0] + USDM_SPEI_LEVELS[1]) / 2.0,
        (USDM_SPEI_LEVELS[1] + USDM_SPEI_LEVELS[2]) / 2.0,
        (USDM_SPEI_LEVELS[2] + USDM_SPEI_LEVELS[3]) / 2.0,
        (USDM_SPEI_LEVELS[3] + USDM_SPEI_LEVELS[4]) / 2.0,
        (USDM_SPEI_LEVELS[4] + 10.0) / 2.0,
    ]
    usdm_tick_labels = ["D4", "D3", "D2", "D1", "D0", "None"]

    # Years for coloring (use first bundle for year bounds)
    years_cat = np.concatenate(years_all)
    if years_cat.size == 0:
        _log("    ↷ empty time axis; skipping summary PDF.")
        return None
    pivot = int(pivot_year)
    hist_years = years_cat[years_cat <= pivot]
    fut_years = years_cat[years_cat > pivot]
    hist_min = int(np.min(hist_years)) if hist_years.size else int(np.min(years_cat))
    hist_max = int(np.max(hist_years)) if hist_years.size else pivot
    fut_min = int(np.min(fut_years)) if fut_years.size else (pivot + 1)
    fut_max = int(np.max(fut_years)) if fut_years.size else (pivot + 1)

    # Grid for background contours
    nx, ny = 220, 220
    xx = np.linspace(p_min, p_max, nx, dtype=float)
    yy = np.linspace(pet_min, pet_max, ny, dtype=float)
    X, Y = np.meshgrid(xx, yy)
    WB = X - Y

    # Region inset data (use first bundle for lat/lon)
    lat_pts = np.asarray(bundles[0][1]["lat"].values, dtype=float) if "lat" in bundles[0][1] else np.array([])
    lon_pts = np.asarray(bundles[0][1]["lon"].values, dtype=float) if "lon" in bundles[0][1] else np.array([])

    # Background selection
    def _background_params(m: int, point: Optional[int] = None, *, baseline_sel: xr.Dataset) -> Dict[str, float]:
        if point is None:
            return _baseline_mean_params_for_month(baseline_sel, m, fit=fit)
        return _baseline_point_params_for_month(baseline_sel, m, point, fit=fit)

    baseline_fill = baseline_main

    def _plot_region_mean_panel(ax, month: int):
        params = _background_params(month, baseline_sel=baseline_fill)
        Z = _spei_from_wb_grid(WB, fit=fit, params=params)
        cf_local = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm, extend="both")
        ax.contour(X, Y, Z, levels=list(USDM_SPEI_LEVELS), colors="k", linewidths=0.4, alpha=0.35)

        for _, ds_b, _ in bundles:
            ds_m = ds_b.sel(time=ds_b["time"].dt.month == month)
            if ds_m.sizes.get("time", 0) == 0:
                continue
            Pm = np.asarray(ds_m[p_var].values, dtype=float)
            PETm = np.asarray(ds_m[pet_var].values, dtype=float)
            t_years = np.asarray(ds_m["time"].dt.year.values, dtype=int)
            cols_t = _year_color_rgba(
                t_years,
                pivot_year=pivot,
                hist_min_year=hist_min,
                hist_max_year=hist_max,
                fut_min_year=fut_min,
                fut_max_year=fut_max,
            )
            Pmean = np.nanmean(Pm, axis=1)
            PETmean = np.nanmean(PETm, axis=1)
            if Pmean.size >= 2:
                seg_mean = np.stack(
                    [np.stack([Pmean[:-1], PETmean[:-1]], axis=1), np.stack([Pmean[1:], PETmean[1:]], axis=1)],
                    axis=1,
                )
                lc_mean = LineCollection(seg_mean, colors=cols_t[:-1, :], linewidths=1.0, alpha=0.85)
                ax.add_collection(lc_mean)
            ax.scatter(Pmean, PETmean, s=18.0, c=cols_t, linewidths=0.0, alpha=0.9)

        if era5_bundle is not None:
            _, ds_era, _ = era5_bundle
            ds_m_e = ds_era.sel(time=ds_era["time"].dt.month == month)
            if ds_m_e.sizes.get("time", 0) > 0:
                Pm_e = np.asarray(ds_m_e[p_var].values, dtype=float)
                PETm_e = np.asarray(ds_m_e[pet_var].values, dtype=float)
                yrs_e = np.asarray(ds_m_e["time"].dt.year.values, dtype=int)
                cols_e = _year_color_green(yrs_e)
                Pmean_e = np.nanmean(Pm_e, axis=1)
                PETmean_e = np.nanmean(PETm_e, axis=1)
                finite_e = np.isfinite(Pmean_e) & np.isfinite(PETmean_e)
                if np.any(finite_e):
                    Pmean_e = Pmean_e[finite_e]
                    PETmean_e = PETmean_e[finite_e]
                    cols_e = cols_e[finite_e]
                    if Pmean_e.size >= 2:
                        seg_e = np.stack(
                            [np.stack([Pmean_e[:-1], PETmean_e[:-1]], axis=1), np.stack([Pmean_e[1:], PETmean_e[1:]], axis=1)],
                            axis=1,
                        )
                        lc_e = LineCollection(seg_e, colors=cols_e[:-1, :], linewidths=1.4, alpha=0.95)
                        ax.add_collection(lc_e)
                    ax.scatter(Pmean_e, PETmean_e, s=20.0, c=cols_e, linewidths=0.0, alpha=0.95, label="ERA5")

        ax.set_title(_month_name(month))
        ax.set_xlim(p_min, p_max)
        ax.set_ylim(pet_min, pet_max)
        return cf_local

    with PdfPages(out_path) as pdf:
        # First page: region mean per month, shared axes
        fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharex=True, sharey=True, constrained_layout=True)
        axes_flat = axes.ravel().tolist()
        mappable_for_cb = None
        for mi, m in enumerate(range(1, 13)):
            ax = axes_flat[mi]
            cf = _plot_region_mean_panel(ax, m)
            if mappable_for_cb is None:
                mappable_for_cb = cf

        for r in range(3):
            axes[r, 0].set_ylabel(f"Accumulated PET over {scale} months (mm)")
        for c in range(4):
            axes[2, c].set_xlabel(f"Accumulated precipitation over {scale} months (mm)")

        if mappable_for_cb is not None:
            cb = fig.colorbar(mappable_for_cb, ax=axes_flat, orientation="horizontal", fraction=0.045, pad=0.08)
            cb.set_ticks(usdm_tick_centers)
            cb.set_ticklabels(usdm_tick_labels)
            cb.set_label("US Drought Monitor drought classification (SPEI thresholds)")

        # Timeline legend showing GCMAGICC (grey→purple) and ERA5 (green) gradients
        try:
            years_gcm = years_cat
            years_era = np.asarray(era5_bundle[1]["time"].dt.year.values, dtype=int) if era5_bundle else np.array([], dtype=int)
            _add_timeline_legend(
                fig,
                years_gcm=years_gcm,
                years_era=years_era,
                hist_min=hist_min,
                hist_max=hist_max,
                fut_min=fut_min,
                fut_max=fut_max,
                pivot=pivot,
                n_ensembles=len(bundles),
            )
        except Exception:
            pass

        title = (
            f"{region.upper()} | {forcing_label} | all runs ({len(bundles)})\n"
            f"SPEI{scale} background using baseline={baseline_label} ({baseline_strategy}), fit={fit}, PET={pet_method}, group={group_pixels}"
        )
        fig.suptitle(title, fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

        # Standalone January region-mean excerpt (PDF + PNG)
        try:
            fig_jan, ax_jan = plt.subplots(figsize=(8.6, 7.0))
            fig_jan.subplots_adjust(left=0.1, right=0.88, bottom=0.18, top=0.9)
            cf_jan = _plot_region_mean_panel(ax_jan, 1)
            cb_jan = fig_jan.colorbar(cf_jan, ax=ax_jan, orientation="vertical", fraction=0.06, pad=0.02)
            cb_jan.set_ticks(usdm_tick_centers)
            cb_jan.set_ticklabels(usdm_tick_labels)
            cb_jan.set_label("US Drought Monitor drought classification (SPEI thresholds)")
            ax_jan.set_xlabel(f"Accumulated precipitation over {scale} months (mm)")
            ax_jan.set_ylabel(f"Accumulated PET over {scale} months (mm)")
            try:
                years_era_jan = np.asarray(era5_bundle[1]["time"].dt.year.values, dtype=int) if era5_bundle else np.array([], dtype=int)
            except Exception:
                years_era_jan = np.array([], dtype=int)
            _add_timeline_legend(
                fig_jan,
                years_gcm=years_cat,
                years_era=years_era_jan,
                hist_min=hist_min,
                hist_max=hist_max,
                fut_min=fut_min,
                fut_max=fut_max,
                pivot=pivot,
                n_ensembles=len(bundles),
            )
            title_jan = (
                f"{region.upper()} | {forcing_label} | all runs ({len(bundles)}) — January region mean\n"
                f"SPEI{scale} background using baseline={baseline_label} ({baseline_strategy}), fit={fit}, PET={pet_method}, group={group_pixels}"
            )
            fig_jan.suptitle(title_jan, fontsize=12)
            _save_pdf_and_png(fig_jan, jan_excerpt_pdf, dpi=320)
            # Persist minimal data needed to regenerate the plot
            jan_series = []
            for run_id_b, ds_b, _ in bundles:
                ds_m = ds_b.sel(time=ds_b["time"].dt.month == 1)
                if ds_m.sizes.get("time", 0) == 0:
                    continue
                Pm = np.asarray(ds_m[p_var].values, dtype=float)
                PETm = np.asarray(ds_m[pet_var].values, dtype=float)
                t_years = np.asarray(ds_m["time"].dt.year.values, dtype=int)
                jan_series.append(
                    {
                        "run_id": run_id_b,
                        "years": t_years.tolist(),
                        "p_mean": np.nanmean(Pm, axis=1).tolist(),
                        "pet_mean": np.nanmean(PETm, axis=1).tolist(),
                    }
                )
            era5_jan = None
            if era5_bundle is not None:
                _, ds_era_j, _ = era5_bundle
                ds_m_e = ds_era_j.sel(time=ds_era_j["time"].dt.month == 1)
                if ds_m_e.sizes.get("time", 0) > 0:
                    Pm_e = np.asarray(ds_m_e[p_var].values, dtype=float)
                    PETm_e = np.asarray(ds_m_e[pet_var].values, dtype=float)
                    yrs_e = np.asarray(ds_m_e["time"].dt.year.values, dtype=int)
                    era5_jan = {
                        "years": yrs_e.tolist(),
                        "p_mean": np.nanmean(Pm_e, axis=1).tolist(),
                        "pet_mean": np.nanmean(PETm_e, axis=1).tolist(),
                    }
            jan_json = {
                "title": title_jan,
                "region": region,
                "forcing_label": forcing_label,
                "scale_months": scale,
                "pet_method": pet_method,
                "fit": fit,
                "baseline_strategy": baseline_strategy,
                "baseline_label": baseline_label,
                "group_pixels": group_pixels,
                "axis": {"p_min": p_min, "p_max": p_max, "pet_min": pet_min, "pet_max": pet_max},
                "usdm_levels": USDM_SPEI_LEVELS,
                "usdm_tick_centers": usdm_tick_centers,
                "usdm_tick_labels": usdm_tick_labels,
                "pivot_year": pivot,
                "hist_min": hist_min,
                "hist_max": hist_max,
                "fut_min": fut_min,
                "fut_max": fut_max,
                "years_all": years_cat.tolist(),
                "baseline_params_jan": _background_params(1, baseline_sel=baseline_fill),
                "runs": jan_series,
                "era5": era5_jan,
                "mesh_size": {"nx": 220, "ny": 220},
            }
            _save_json(jan_json, jan_excerpt_pdf.with_suffix(".json"))
            plt.close(fig_jan)
            _log(f"    ✓ wrote January excerpt -> {jan_excerpt_pdf} (+ PNG)")
        except Exception as exc:
            _log(f"    ⚠️ January excerpt failed: {exc}")

        # ------------------------------------------------------------------
        # Page 2+: January panels per grid point (no shared axes)
        # ------------------------------------------------------------------
        # Assume all bundles share point coordinates from first
        _, ds_ref, baseline_ref = bundles[0]
        jan_ref = ds_ref.sel(time=ds_ref["time"].dt.month == 1)
        if jan_ref.sizes.get("time", 0) > 0 and jan_ref.sizes.get("point", 0) > 0:
            lat_vals = np.asarray(jan_ref["lat"].values, dtype=float) if "lat" in jan_ref else np.full(jan_ref.sizes["point"], np.nan)
            lon_vals = np.asarray(jan_ref["lon"].values, dtype=float) if "lon" in jan_ref else np.full(jan_ref.sizes["point"], np.nan)
            n_pts = jan_ref.sizes["point"]
            per_page = 9
            rows, cols_grid = 3, 3
            for start_idx in range(0, n_pts, per_page):
                fig_j, axes_j = plt.subplots(rows, cols_grid, figsize=(16, 11), sharex=False, sharey=False, constrained_layout=True)
                axes_j_flat = axes_j.ravel().tolist()
                cf = None
                for j_idx in range(per_page):
                    pt = start_idx + j_idx
                    ax = axes_j_flat[j_idx]
                    if pt >= n_pts:
                        ax.set_axis_off()
                        continue
                    # Axis limits per point across all bundles (tight)
                    p_vals_pt = []
                    pet_vals_pt = []
                    for _, ds_b, _ in bundles:
                        jan_b = ds_b.sel(time=ds_b["time"].dt.month == 1)
                        if jan_b.sizes.get("time", 0) == 0:
                            continue
                        p_vals_pt.append(np.asarray(jan_b[p_var].isel(point=pt).values, dtype=float))
                        pet_vals_pt.append(np.asarray(jan_b[pet_var].isel(point=pt).values, dtype=float))
                    if era5_bundle is not None:
                        _, ds_era5_axes, _ = era5_bundle
                        jan_e_axes = ds_era5_axes.sel(time=ds_era5_axes["time"].dt.month == 1)
                        if jan_e_axes.sizes.get("time", 0) > 0 and pt < jan_e_axes.sizes.get("point", 0):
                            p_vals_pt.append(np.asarray(jan_e_axes[p_var].isel(point=pt).values, dtype=float))
                            pet_vals_pt.append(np.asarray(jan_e_axes[pet_var].isel(point=pt).values, dtype=float))
                    if not p_vals_pt or not pet_vals_pt:
                        ax.set_axis_off()
                        continue
                    p_concat = np.concatenate(p_vals_pt)
                    pet_concat = np.concatenate(pet_vals_pt)
                    if (not np.isfinite(p_concat).any()) or (not np.isfinite(pet_concat).any()):
                        ax.set_axis_off()
                        continue
                    p_min_pt = float(np.nanmin(p_concat))
                    p_max_pt = float(np.nanmax(p_concat))
                    pet_min_pt = float(np.nanmin(pet_concat))
                    pet_max_pt = float(np.nanmax(pet_concat))
                    px_pt = max(1e-6, (p_max_pt - p_min_pt))
                    py_pt = max(1e-6, (pet_max_pt - pet_min_pt))
                    p_min_pt -= 0.05 * px_pt
                    p_max_pt += 0.05 * px_pt
                    pet_min_pt -= 0.05 * py_pt
                    pet_max_pt += 0.05 * py_pt

                    xx_pt = np.linspace(p_min_pt, p_max_pt, 220, dtype=float)
                    yy_pt = np.linspace(pet_min_pt, pet_max_pt, 220, dtype=float)
                    X_pt, Y_pt = np.meshgrid(xx_pt, yy_pt)
                    WB_pt = X_pt - Y_pt

                    # Background: per-member contour overlay if needed
                    if baseline_strategy == "per_member":
                        params_fill = _background_params(1, point=pt, baseline_sel=baseline_fill)
                        Z_fill = _spei_from_wb_grid(WB_pt, fit=fit, params=params_fill)
                        cf_local = ax.contourf(X_pt, Y_pt, Z_fill, levels=levels, cmap=cmap, norm=norm, extend="both")
                        cf = cf or cf_local
                        for _, _, bl_extra in bundles[1:]:
                            params_ct = _background_params(1, point=pt, baseline_sel=bl_extra)
                            Z_ct = _spei_from_wb_grid(WB_pt, fit=fit, params=params_ct)
                            ax.contour(X_pt, Y_pt, Z_ct, levels=list(USDM_SPEI_LEVELS), colors="k", linewidths=0.5, alpha=0.5)
                    else:
                        params_fill = _background_params(1, point=pt, baseline_sel=baseline_fill)
                        Z_fill = _spei_from_wb_grid(WB_pt, fit=fit, params=params_fill)
                        cf_local = ax.contourf(X_pt, Y_pt, Z_fill, levels=levels, cmap=cmap, norm=norm, extend="both")
                        ax.contour(X_pt, Y_pt, Z_fill, levels=list(USDM_SPEI_LEVELS), colors="k", linewidths=0.4, alpha=0.35)
                        cf = cf or cf_local

                    # Overlay trajectories per bundle
                    for run_id_b, ds_b, _ in bundles:
                        jan_b = ds_b.sel(time=ds_b["time"].dt.month == 1)
                        if jan_b.sizes.get("time", 0) == 0:
                            continue
                        x_pt = np.asarray(jan_b[p_var].isel(point=pt).values, dtype=float)
                        y_pt = np.asarray(jan_b[pet_var].isel(point=pt).values, dtype=float)
                        yrs_pt = np.asarray(jan_b["time"].dt.year.values, dtype=int)
                        finite = np.isfinite(x_pt) & np.isfinite(y_pt)
                        if not np.any(finite):
                            continue
                        x_pt = x_pt[finite]
                        y_pt = y_pt[finite]
                        yrs_pt = yrs_pt[finite]
                        cols_pt = _year_color_rgba(yrs_pt, pivot_year=pivot, hist_min_year=hist_min, hist_max_year=hist_max, fut_min_year=fut_min, fut_max_year=fut_max)
                        if x_pt.size >= 2:
                            seg_pt = np.stack(
                                [np.stack([x_pt[:-1], y_pt[:-1]], axis=1), np.stack([x_pt[1:], y_pt[1:]], axis=1)],
                                axis=1,
                            )
                            lc_pt = LineCollection(seg_pt, colors=cols_pt[:-1, :], linewidths=0.9, alpha=0.9)
                            ax.add_collection(lc_pt)
                        ax.scatter(x_pt, y_pt, s=12.0, c=cols_pt, linewidths=0.0, alpha=0.95, label=run_id_b)

                    # ERA5 overlay for grid points
                    if era5_bundle is not None:
                        _, ds_era5_pt, _ = era5_bundle
                        jan_e = ds_era5_pt.sel(time=ds_era5_pt["time"].dt.month == 1)
                        if jan_e.sizes.get("time", 0) > 0 and pt < jan_e.sizes.get("point", 0):
                            x_e = np.asarray(jan_e[p_var].isel(point=pt).values, dtype=float)
                            y_e = np.asarray(jan_e[pet_var].isel(point=pt).values, dtype=float)
                            yrs_e = np.asarray(jan_e["time"].dt.year.values, dtype=int)
                            finite_e = np.isfinite(x_e) & np.isfinite(y_e)
                            x_e = x_e[finite_e]
                            y_e = y_e[finite_e]
                            yrs_e = yrs_e[finite_e]
                            cols_e = _year_color_green(yrs_e)
                            if x_e.size >= 2:
                                seg_e = np.stack(
                                    [np.stack([x_e[:-1], y_e[:-1]], axis=1), np.stack([x_e[1:], y_e[1:]], axis=1)],
                                    axis=1,
                                )
                                lc_e = LineCollection(seg_e, colors=cols_e[:-1, :], linewidths=1.1, alpha=0.9)
                                ax.add_collection(lc_e)
                            ax.scatter(x_e, y_e, s=10.0, c=cols_e, linewidths=0.0, alpha=0.95)

                    ax.set_xlim(p_min_pt, p_max_pt)
                    ax.set_ylim(pet_min_pt, pet_max_pt)
                    lat_pt = lat_vals[pt] if pt < lat_vals.size else float("nan")
                    lon_pt = lon_vals[pt] if pt < lon_vals.size else float("nan")
                    ax.set_title(f"Point {pt} (lat={lat_pt:.2f}, lon={lon_pt:.2f})", fontsize=9)
                    ax.tick_params(labelsize=8)
                    if np.isfinite(lat_pt) and np.isfinite(lon_pt):
                        _add_region_inset(
                            ax,
                            region=region,
                            # IMPORTANT: pass the full region point cloud so the inset can
                            # derive a non-degenerate extent when a region polygon is not
                            # available (e.g., country codes like IRN).
                            lat_points=lat_vals,
                            lon_points=lon_vals,
                            fill_region=True,
                            highlight_point=(float(lat_pt), float(lon_pt)),
                            show_frame=False,
                        )

                fig_j.suptitle(f"{region.upper()} | {forcing_label} | January by grid point", fontsize=12)
                pdf.savefig(fig_j)
                plt.close(fig_j)

    _log(f"    ✓ wrote summary PDF: {out_path}")
    return out_path


def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid to avoid overflow in exp for large +/- inputs.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    if np.any(pos):
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    if np.any(~pos):
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
    return out


def _pglo_cdf(x: np.ndarray, *, xi: float, alpha: float, kappa: float) -> np.ndarray:
    """Generalized logistic CDF used by R SPEI ('glo')."""
    x = np.asarray(x, dtype=np.float64)
    if not (np.isfinite(xi) and np.isfinite(alpha) and alpha > 0 and np.isfinite(kappa)):
        return np.full_like(x, np.nan, dtype=np.float64)
    y = (x - xi) / alpha
    k = kappa
    if abs(k) < 1e-12:
        # Logistic
        return _stable_sigmoid(y)
    arg = 1.0 - k * y
    out = np.full_like(y, np.nan, dtype=np.float64)
    m1 = arg > 1e-15
    if np.any(m1):
        y2 = -np.log(arg[m1]) / k
        out[m1] = _stable_sigmoid(y2)
    if np.any(~m1):
        out[~m1] = 0.0 if k < 0 else 1.0
    return out


def _spei_from_wb_grid(wb: np.ndarray, *, fit: str, params: Dict[str, float]) -> np.ndarray:
    fit_norm = (fit or "").lower().strip()
    if fit_norm == "loglogistic" and all(k in params for k in ("xi", "alpha", "kappa")):
        F = _pglo_cdf(wb, xi=params["xi"], alpha=params["alpha"], kappa=params["kappa"])
        F = np.clip(F, 1e-15, 1.0 - 1e-15)
        return _norm_ppf(F)
    # zscore fallback
    mu = float(params.get("mu", 0.0))
    sig = float(params.get("sigma", 1.0))
    if not np.isfinite(sig) or sig <= 0:
        sig = 1.0
    return (wb - mu) / sig


def _region_outline(region: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (lon, lat) outline for an AR6 region if available."""
    if regionmask is None:
        return None
    try:
        ar6 = regionmask.defined_regions.ar6.all
        rid = ar6.map_keys(region.upper())
        poly = ar6.polygons[rid][0]
        lon, lat = poly.exterior.xy
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        lon = np.where(lon > 180.0, lon - 360.0, lon)
        return lon, lat
    except Exception:
        return None


def _region_centroid(lats: np.ndarray, lons: np.ndarray) -> Tuple[float, float]:
    lons_wrap = np.where(lons > 180.0, lons - 360.0, lons)
    return float(np.nanmean(lats)), float(np.nanmean(lons_wrap))


def _add_region_inset(
    ax,
    *,
    region: str,
    lat_points: np.ndarray,
    lon_points: np.ndarray,
    fill_region: bool,
    highlight_point: Optional[Tuple[float, float]] = None,
    show_frame: bool = True,
) -> None:
    if plt is None:
        return
    if lat_points.size == 0 or lon_points.size == 0:
        return
    if inset_axes is None:
        try:
            inset_ax = ax.inset_axes([0.65, 0.58, 0.32, 0.32])
        except Exception:
            return
    else:
        inset_ax = inset_axes(ax, width="32%", height="32%", loc="upper right", borderpad=0.4)
    # Ensure inset is drawn above the main panel plot elements (contours/lines/scatters).
    inset_ax.set_zorder(50)

    outline = _region_outline(region)
    # If regionmask polygons are unavailable (common for country codes like "IRN"),
    # fall back to the same NPZ masks used for subsetting.
    mask_tuple = None
    if outline is None:
        mask_tuple = _load_npz_region_mask(region, lon_convention="360") or _load_npz_region_mask(region, lon_convention="180")

    lon_points_0_360 = np.mod(np.asarray(lon_points, dtype=float), 360.0)
    lons_wrap = np.where(lon_points_0_360 > 180.0, lon_points_0_360 - 360.0, lon_points_0_360)

    lat_center, lon_center_wrap180 = _region_centroid(np.asarray(lat_points, dtype=float), lon_points_0_360)
    lon_center_raw = float(np.mod(lon_center_wrap180, 360.0))
    if highlight_point is not None and all(np.isfinite(highlight_point)):
        lat_center = float(highlight_point[0])
        lon_center_raw = float(np.mod(float(highlight_point[1]), 360.0))
    lon_center_wrap180 = lon_center_raw - 360.0 if lon_center_raw > 180.0 else lon_center_raw

    def _safe_limits(vmin: float, vmax: float, *, pad: float = 0.1) -> Tuple[float, float]:
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return (-1.0, 1.0)
        if abs(vmax - vmin) < 1e-6:
            return (vmin - pad, vmax + pad)
        return (vmin, vmax)

    drawn = False
    lon_dot = lon_center_wrap180  # default for wrap180-based drawings

    if outline is not None and Polygon is not None:
        lon_o, lat_o = outline
        face = (1.0, 1.0, 1.0, REGION_INSET_FILL_ALPHA) if fill_region else "none"
        inset_ax.add_patch(
            Polygon(
                np.column_stack((lon_o, lat_o)),
                closed=True,
                facecolor=face,
                edgecolor="#000000",
                linewidth=0.7,
                alpha=1.0,
                zorder=2,
            )
        )
        lx0, lx1 = _safe_limits(float(np.nanmin(lon_o)), float(np.nanmax(lon_o)))
        ly0, ly1 = _safe_limits(float(np.nanmin(lat_o)), float(np.nanmax(lat_o)))
        inset_ax.set_xlim(lx0, lx1)
        inset_ax.set_ylim(ly0, ly1)
        lon_dot = lon_center_wrap180
        drawn = True

    elif mask_tuple is not None:
        # Draw region fill + outline from the boolean mask grid (NPZ).
        mask_arr, mask_lats, mask_lons = mask_tuple
        mask_arr = np.asarray(mask_arr, dtype=bool)
        mask_lats = np.asarray(mask_lats, dtype=float)
        mask_lons = np.asarray(mask_lons, dtype=float)

        ii, jj = np.where(mask_arr)
        if ii.size > 0 and jj.size > 0:
            # Choose lon convention that keeps the region contiguous.
            lons0 = np.mod(mask_lons, 360.0)
            lons180 = np.where(lons0 > 180.0, lons0 - 360.0, lons0)
            span0 = float(np.nanmax(lons0[jj]) - np.nanmin(lons0[jj]))
            span180 = float(np.nanmax(lons180[jj]) - np.nanmin(lons180[jj]))
            use_wrap180 = span180 <= span0

            lons_plot = lons180 if use_wrap180 else lons0
            lon_dot = (lon_center_wrap180 if use_wrap180 else lon_center_raw)

            Lon2, Lat2 = np.meshgrid(lons_plot, mask_lats)
            Zm = mask_arr.astype(float)

            if fill_region:
                inset_ax.contourf(
                    Lon2,
                    Lat2,
                    Zm,
                    levels=[0.5, 1.5],
                    colors=[(1.0, 1.0, 1.0, REGION_INSET_FILL_ALPHA)],
                    antialiased=True,
                    zorder=2,
                )
            inset_ax.contour(
                Lon2,
                Lat2,
                Zm,
                levels=[0.5],
                colors="k",
                linewidths=0.7,
                alpha=0.9,
                zorder=3,
            )

            # Tight extent around the mask with a small pad.
            lon_sel = lons_plot[jj]
            lat_sel = mask_lats[ii]
            lx0, lx1 = _safe_limits(float(np.nanmin(lon_sel)), float(np.nanmax(lon_sel)), pad=1.0)
            ly0, ly1 = _safe_limits(float(np.nanmin(lat_sel)), float(np.nanmax(lat_sel)), pad=1.0)
            inset_ax.set_xlim(lx0, lx1)
            inset_ax.set_ylim(ly0, ly1)
            drawn = True

    if (not drawn):
        # Final fallback: bounding box of *region point set* (works once caller passes full region points).
        if lons_wrap.size == 0 or np.asarray(lat_points).size == 0 or Polygon is None:
            inset_ax.set_axis_off()
            return
        inset_ax.add_patch(
            Polygon(
                np.array(
                    [
                        [np.nanmin(lons_wrap), np.nanmin(lat_points)],
                        [np.nanmax(lons_wrap), np.nanmin(lat_points)],
                        [np.nanmax(lons_wrap), np.nanmax(lat_points)],
                        [np.nanmin(lons_wrap), np.nanmax(lat_points)],
                    ]
                ),
                closed=True,
                facecolor=(1.0, 1.0, 1.0, REGION_INSET_FILL_ALPHA) if fill_region else "none",
                edgecolor="#000000",
                linewidth=0.7,
                alpha=1.0,
                zorder=2,
            )
        )
        lx0, lx1 = _safe_limits(float(np.nanmin(lons_wrap)), float(np.nanmax(lons_wrap)), pad=1.0)
        ly0, ly1 = _safe_limits(float(np.nanmin(lat_points)), float(np.nanmax(lat_points)), pad=1.0)
        inset_ax.set_xlim(lx0, lx1)
        inset_ax.set_ylim(ly0, ly1)
        lon_dot = lon_center_wrap180

    inset_ax.set_facecolor("none")
    inset_ax.scatter([lon_dot], [lat_center], s=14.0, color="red", zorder=6, edgecolors="none")
    inset_ax.set_aspect("equal", adjustable="box")
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_title("" if not show_frame else region.upper(), fontsize=7, pad=1.0)
    for spine in inset_ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_visible(show_frame)


def _year_color_rgba(
    years: np.ndarray,
    *,
    pivot_year: int,
    hist_min_year: int,
    hist_max_year: int,
    fut_min_year: int,
    fut_max_year: int,
) -> np.ndarray:
    """
    Map year -> RGBA:
      - historical (<=pivot): light grey (oldest) -> black (pivot or hist_max)
      - future (>pivot): dark purple (near-term) -> lighter purple (far future)
    """
    if plt is None:
        return np.zeros((len(years), 4), dtype=float)
    y = np.asarray(years, dtype=int)
    cols = np.zeros((y.size, 4), dtype=float)

    mh = y <= int(pivot_year)
    mf = ~mh
    if np.any(mh):
        y0 = int(hist_min_year)
        y1 = int(hist_max_year)
        den = max(1, (y1 - y0))
        t = (y[mh] - y0) / den
        # Keep away from pure white
        v = 0.20 + 0.80 * np.clip(t, 0.0, 1.0)
        cols[mh, :] = plt.cm.Greys(v)
    if np.any(mf):
        y0 = int(fut_min_year)
        y1 = int(fut_max_year)
        den = max(1, (y1 - y0))
        t = (y[mf] - y0) / den
        # dark -> light with time (near-term darker)
        v = 0.90 - 0.60 * np.clip(t, 0.0, 1.0)
        cols[mf, :] = plt.cm.Purples(v)
    return cols


def _year_color_green(years: np.ndarray) -> np.ndarray:
    """
    Map year -> RGBA on a light-to-dark green gradient:
      earliest -> bright (light) green, most recent -> dark green.
    """
    if plt is None:
        return np.zeros((len(years), 4), dtype=float)
    y = np.asarray(years, dtype=int)
    if y.size == 0:
        return np.zeros((0, 4), dtype=float)
    y0 = int(np.min(y))
    y1 = int(np.max(y))
    den = max(1, (y1 - y0))
    t = (y - y0) / den  # 0=oldest, 1=newest
    # Keep away from pure white; reserve darkest greens for newest years
    v = 0.20 + 0.75 * np.clip(t, 0.0, 1.0)
    return plt.cm.Greens(v)


def _add_timeline_legend(
    fig,
    *,
    years_gcm: np.ndarray,
    years_era: np.ndarray,
    hist_min: int,
    hist_max: int,
    fut_min: int,
    fut_max: int,
    pivot: int,
    n_ensembles: int,
) -> None:
    if plt is None:
        return
    yrs_g = np.asarray(years_gcm, dtype=int)
    if yrs_g.size == 0:
        return
    # Anchor legend on the lower-left; leave room on the right for the colorbar.
    x0, y0, w, h = 0.05, 0.02, 0.4, 0.08
    try:
        ax = fig.add_axes([x0, y0, w, h])
    except Exception:
        return
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)

    g_min = int(np.min(yrs_g))
    g_max = int(np.max(yrs_g))
    g_years = np.linspace(g_min, g_max, num=min(60, max(10, yrs_g.size * 2)), dtype=int)
    cols_g = _year_color_rgba(
        g_years,
        pivot_year=pivot,
        hist_min_year=hist_min,
        hist_max_year=hist_max,
        fut_min_year=fut_min,
        fut_max_year=fut_max,
    )
    ax.scatter(g_years, np.full_like(g_years, 0.75, dtype=float), s=10.0, c=cols_g, linewidths=0.0)
    ax.plot([g_min, g_max], [0.75, 0.75], color="#555", linewidth=0.6, alpha=0.7)
    ax.text(
        g_min,
        0.95,
        f"GCMAGICCxERA5 ensembles (n={n_ensembles})",
        fontsize=8,
        va="center",
        ha="left",
        color="#222",
    )

    if years_era.size > 0:
        yrs_e = np.asarray(years_era, dtype=int)
        e_min = int(np.min(yrs_e))
        e_max = int(np.max(yrs_e))
        e_years = np.linspace(e_min, e_max, num=min(40, max(8, yrs_e.size)), dtype=int)
        cols_e = _year_color_green(e_years)
        ax.scatter(e_years, np.full_like(e_years, 0.35, dtype=float), s=10.0, c=cols_e, linewidths=0.0)
        ax.plot([e_min, e_max], [0.35, 0.35], color="#2e8b57", linewidth=0.6, alpha=0.9)
        ax.text(
            e_min,
            0.55,
            "ERA5",
            fontsize=8,
            va="center",
            ha="left",
            color="#1f5c35",
        )

    xmin = min(g_min, int(np.min(years_era)) if years_era.size else g_min)
    xmax = max(g_max, int(np.max(years_era)) if years_era.size else g_max)
    ax.set_xlim(xmin, xmax)
    # Show ticks every 10 years for readability
    tick_start = int(np.floor(xmin / 10.0) * 10)
    tick_end = int(np.ceil(xmax / 10.0) * 10)
    ticks = np.arange(tick_start, tick_end + 1, 10, dtype=int)
    if ticks.size == 0:
        ticks = np.array([xmin, xmax], dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=8)
    ax.set_ylim(0.1, 1.05)


def _write_inputs_comparison_pdf(
    *,
    inputs_accum: Dict[str, Dict[str, List[Tuple[str, np.ndarray, np.ndarray, Optional[str]]]]],
    pet_method: str,
    region: str,
    output_tag: Optional[str],
    out_root: Path,
) -> Optional[Path]:
    """Write a multi-page PDF comparing SPEI input variables across ERA5/SCENARIO1/SCENARIO2/CMIP6 and save the annual-mean page separately."""
    if plt is None or PdfPages is None:
        _log("    ↷ matplotlib unavailable; skipping COMPARISON_SPEI_INPUTS.")
        return None

    vars_all = set()
    for forc in inputs_accum.values():
        vars_all.update(forc.keys())
    var_names: List[str] = []
    if VAR_PR in vars_all:
        var_names.append(VAR_PR)
        vars_all.remove(VAR_PR)
    var_names.extend(sorted(vars_all))
    if not var_names:
        _log("    ↷ no input series gathered; skipping COMPARISON_SPEI_INPUTS.")
        return None

    # Determine global time bounds
    t_bounds: List[np.ndarray] = []
    for forc in inputs_accum.values():
        for entries in forc.values():
            for _, t_vals, _, _ in entries:
                t_arr = _to_datetime64(t_vals)
                if t_arr is not None and t_arr.size:
                    t_bounds.append(t_arr)
    xmin = min([arr.min() for arr in t_bounds]) if t_bounds else None
    xmax = max([arr.max() for arr in t_bounds]) if t_bounds else None

    out_dir = _summary_pdf_dir(out_root, output_tag=output_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_pet_tag = _sanitize_run_id(pet_method)
    out_path = out_dir / f"COMPARISON_SPEI_INPUTS__{region.upper()}__pet-{safe_pet_tag}.pdf"
    excerpt_path = out_dir / f"EXCERPT_ANNUALMEAN__SPEI_INPUTS__{region.upper()}__pet-{safe_pet_tag}.pdf"

    # Plot styles for comparison panels.
    # Requested emphasis:
    #   - ERA5 as bold black line
    #   - CMIP6 overlays in the background (thin + transparent)
    line_style_map: Dict[str, Dict[str, object]] = {
        "ERA5": {"color": "#000000", "linewidth": 2.4, "alpha": 0.95, "linestyle": "-", "zorder": 4.0},
        FORCING_SCENARIO1_LABEL: {"color": "#dba25c", "linewidth": 0.9, "alpha": 0.45, "linestyle": "-", "zorder": 3.0},
        FORCING_SCENARIO2_LABEL: {"color": "#5c9edb", "linewidth": 0.9, "alpha": 0.45, "linestyle": "-", "zorder": 3.0},
        "CMIP6-HIST": {"color": "#c70000", "linewidth": 0.4, "alpha": 0.32, "linestyle": "-", "zorder": 1.0},
        "CMIP6-HIST-NAT": {"color": "#7b3fa6", "linewidth": 0.4, "alpha": 0.32, "linestyle": "-", "zorder": 1.0},
        "CMIP6-SSP245": {"color": "#2f8f46", "linewidth": 0.4, "alpha": 0.32, "linestyle": "-", "zorder": 1.0},
    }

    def _line_style(forcing_label: str) -> Dict[str, object]:
        return line_style_map.get(
            forcing_label,
            {"color": "#444444", "linewidth": 0.8, "alpha": 0.5, "linestyle": "-", "zorder": 2.0},
        )
    forcing_order_preferred = [
        # Draw CMIP6 first so all other forcings sit on top.
        "CMIP6-HIST",
        "CMIP6-HIST-NAT",
        "CMIP6-SSP245",
        "ERA5",
        FORCING_SCENARIO1_LABEL,
        FORCING_SCENARIO2_LABEL,
    ]
    forcing_order: List[str] = []
    for forcing_label in forcing_order_preferred:
        forc = inputs_accum.get(forcing_label, {})
        has_entries = any(bool(entries) for entries in forc.values())
        if has_entries and forcing_label not in forcing_order:
            forcing_order.append(forcing_label)
    for forcing_label, forc in sorted(inputs_accum.items()):
        has_entries = any(bool(entries) for entries in forc.values())
        if has_entries and forcing_label not in forcing_order:
            forcing_order.append(forcing_label)
    if not forcing_order:
        forcing_order = [
            "ERA5",
            FORCING_SCENARIO1_LABEL,
            FORCING_SCENARIO2_LABEL,
            "CMIP6-HIST",
            "CMIP6-HIST-NAT",
            "CMIP6-SSP245",
        ]

    def _page(entries_by_var: Dict[str, Dict[str, List[Tuple[str, np.ndarray, np.ndarray, Optional[str]]]]],
              title_suffix: str,
              selector: Optional[int]) -> plt.Figure:
        """
        Build one page.
        selector=None -> raw monthly values
        selector=-1   -> annual means over all months
        selector=1..12 -> monthly slices for Jan..Dec
        """
        fig, axes = plt.subplots(len(var_names), 1, figsize=(16, 2.0 + 2.3 * len(var_names)), sharex=True)
        if len(var_names) == 1:
            axes = [axes]  # type: ignore
        legend_handles = {}

        for ax, var in zip(axes, var_names):
            units: Optional[str] = None
            for forcing_label in forcing_order:
                entries = entries_by_var.get(forcing_label, {}).get(var, [])
                style = _line_style(forcing_label)
                color = str(style.get("color", "#444444"))
                linewidth = float(style.get("linewidth", 0.8))
                alpha = float(style.get("alpha", 0.5))
                linestyle = str(style.get("linestyle", "-"))
                zorder = float(style.get("zorder", 2.0))
                for run_id, t_vals, vals, u in entries:
                    t_arr = _to_datetime64(t_vals)
                    if t_arr is None or t_arr.size == 0:
                        continue
                    v_arr = np.asarray(vals, dtype=float)
                    if selector is None:
                        pass  # raw
                    elif selector == -1:
                        # annual mean
                        years = t_arr.astype("datetime64[Y]").astype(int) + 1970
                        uniq = np.unique(years)
                        v_coll = []
                        t_coll = []
                        for y in uniq:
                            mask = years == y
                            if np.any(mask):
                                v_coll.append(np.nanmean(v_arr[mask]))
                                t_coll.append(np.datetime64(f"{y}-07-01"))
                        if not v_coll:
                            continue
                        v_arr = np.asarray(v_coll)
                        t_arr = np.asarray(t_coll, dtype="datetime64[ns]")
                    else:
                        # monthly slice (selector 1..12)
                        months = t_arr.astype("datetime64[M]").astype(int) % 12 + 1
                        mask = months == selector
                        if not np.any(mask):
                            continue
                        v_arr = v_arr[mask]
                        t_arr = t_arr[mask]
                    ax.plot(
                        t_arr,
                        v_arr,
                        color=color,
                        alpha=alpha,
                        linewidth=linewidth,
                        linestyle=linestyle,
                        zorder=zorder,
                    )
                    if units is None and u:
                        units = str(u)
                if entries and forcing_label not in legend_handles:
                    legend_handles[forcing_label] = ax.plot(
                        [],
                        [],
                        color=color,
                        alpha=alpha,
                        linewidth=max(1.2, linewidth),
                        linestyle=linestyle,
                        zorder=max(zorder, 4.5),
                        label=forcing_label,
                    )[0]

            label = var if units is None else f"{var} ({units})"
            ax.set_ylabel(label)
            ax.grid(alpha=0.2, linewidth=0.5)
            if xmin is not None and xmax is not None:
                ax.set_xlim(xmin, xmax)

        if legend_handles:
            axes[0].legend(
                handles=list(legend_handles.values()),
                loc="upper left",
                ncol=min(3, max(1, len(legend_handles))),
                fontsize=9,
            )
        axes[-1].set_xlabel("Time")
        fig.suptitle(f"COMPARISON_SPEI_INPUTS — {region.upper()} — PET={pet_method} — {title_suffix}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    with PdfPages(out_path) as pp:
        # Page 1: raw monthly series
        fig_raw = _page(inputs_accum, "raw", None)
        pp.savefig(fig_raw, bbox_inches="tight")
        plt.close(fig_raw)

        # Page 2: annual means (also saved separately)
        # Precompute annual-mean data for JSON + plotting
        annual_json_vars: List[dict] = []
        for var in var_names:
            var_entry = {"name": var, "units": None, "series": []}
            for forcing_label in forcing_order:
                entries = inputs_accum.get(forcing_label, {}).get(var, [])
                for run_id, t_vals, vals, u in entries:
                    t_arr = _to_datetime64(t_vals)
                    if t_arr is None or t_arr.size == 0:
                        continue
                    v_arr = np.asarray(vals, dtype=float)
                    years = t_arr.astype("datetime64[Y]").astype(int) + 1970
                    uniq = np.unique(years)
                    v_coll = []
                    t_coll = []
                    for y in uniq:
                        mask = years == y
                        if np.any(mask):
                            v_coll.append(np.nanmean(v_arr[mask]))
                            t_coll.append(np.datetime64(f"{y}-07-01"))
                    if not v_coll:
                        continue
                    var_entry["series"].append(
                        {
                            "forcing": forcing_label,
                            "run_id": run_id,
                            "times": [np.datetime_as_string(t, unit="D") for t in np.asarray(t_coll, dtype="datetime64[ns]")],
                            "values": np.asarray(v_coll, dtype=float).tolist(),
                        }
                    )
                    if var_entry["units"] is None and u:
                        var_entry["units"] = str(u)
            annual_json_vars.append(var_entry)

        fig_ann = _page(inputs_accum, "annual mean", -1)
        pp.savefig(fig_ann, bbox_inches="tight")
        try:
            _save_pdf_and_png(fig_ann, excerpt_path, dpi=320)
            excerpt_style_map = {
                forcing_label: {
                    "color": str(_line_style(forcing_label).get("color", "#444444")),
                    "linewidth": float(_line_style(forcing_label).get("linewidth", 0.8)),
                    "alpha": float(_line_style(forcing_label).get("alpha", 0.5)),
                    "linestyle": str(_line_style(forcing_label).get("linestyle", "-")),
                }
                for forcing_label in forcing_order
            }
            excerpt_json = {
                "title": f"COMPARISON_SPEI_INPUTS — {region.upper()} — PET={pet_method} — annual mean",
                "region": region,
                "pet_method": pet_method,
                "x_range": [
                    np.datetime_as_string(xmin, unit="s") if xmin is not None else None,
                    np.datetime_as_string(xmax, unit="s") if xmax is not None else None,
                ],
                "variables": annual_json_vars,
                "color_map": {k: v["color"] for k, v in excerpt_style_map.items()},
                "style_map": excerpt_style_map,
            }
            _save_json(excerpt_json, excerpt_path.with_suffix(".json"))
            _log(f"    ✓ wrote annual-mean excerpt -> {excerpt_path} (+ PNG)")
        except Exception as exc:
            _log(f"    ⚠️ annual-mean excerpt failed: {exc}")
        finally:
            plt.close(fig_ann)

        # Pages 3-14: monthly slices Jan..Dec
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for m, name in enumerate(month_names, start=1):
            fig_m = _page(inputs_accum, f"{name} only", m)
            pp.savefig(fig_m, bbox_inches="tight")
            plt.close(fig_m)

    _log(f"    ✓ wrote COMPARISON_SPEI_INPUTS -> {out_path}")
    return out_path


def replot_jan_excerpt(
    json_path: str | Path,
    *,
    output_pdf: Optional[str | Path] = None,
    output_png: Optional[str | Path] = None,
) -> Tuple[Path, Path]:
    """
    Recreate the January region-mean excerpt plot from its JSON payload.
    """
    if plt is None:
        raise RuntimeError("matplotlib is unavailable; cannot replot excerpt.")
    path = Path(json_path)
    data = json.loads(path.read_text())

    p_min = float(data["axis"]["p_min"])
    p_max = float(data["axis"]["p_max"])
    pet_min = float(data["axis"]["pet_min"])
    pet_max = float(data["axis"]["pet_max"])
    levels = [-10.0] + list(data["usdm_levels"]) + [10.0]
    tick_centers = data.get("usdm_tick_centers", [])
    tick_labels = data.get("usdm_tick_labels", [])
    mesh_nx = int(data.get("mesh_size", {}).get("nx", 220))
    mesh_ny = int(data.get("mesh_size", {}).get("ny", 220))
    scale = int(data["scale_months"])
    fit = data.get("fit", "loglogistic")
    baseline_params = data.get("baseline_params_jan", {})
    region = data.get("region", "")
    forcing_label = data.get("forcing_label", "")
    pet_method = data.get("pet_method", "")
    baseline_label = data.get("baseline_label", "")
    baseline_strategy = data.get("baseline_strategy", "")
    group_pixels = data.get("group_pixels", "")
    pivot = int(data.get("pivot_year", 0))
    hist_min = int(data.get("hist_min", pivot))
    hist_max = int(data.get("hist_max", pivot))
    fut_min = int(data.get("fut_min", pivot + 1))
    fut_max = int(data.get("fut_max", pivot + 1))

    xx = np.linspace(p_min, p_max, mesh_nx, dtype=float)
    yy = np.linspace(pet_min, pet_max, mesh_ny, dtype=float)
    X, Y = np.meshgrid(xx, yy)
    WB = X - Y

    cmap = ListedColormap(["#7f0000", "#d73027", "#f46d43", "#fdae61", "#fff7a8", "#ffffff"])
    norm = BoundaryNorm(levels, ncolors=cmap.N)

    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    fig.subplots_adjust(left=0.1, right=0.88, bottom=0.18, top=0.9)

    Z = _spei_from_wb_grid(WB, fit=fit, params=baseline_params)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm, extend="both")
    ax.contour(X, Y, Z, levels=list(USDM_SPEI_LEVELS), colors="k", linewidths=0.4, alpha=0.35)
    cb = fig.colorbar(cf, ax=ax, orientation="vertical", fraction=0.06, pad=0.02)
    if tick_centers:
        cb.set_ticks(tick_centers)
    if tick_labels:
        cb.set_ticklabels(tick_labels)
    cb.set_label("US Drought Monitor drought classification (SPEI thresholds)")

    runs = data.get("runs", [])
    all_years = []
    for run in runs:
        years = np.asarray(run.get("years", []), dtype=int)
        p_mean = np.asarray(run.get("p_mean", []), dtype=float)
        pet_mean = np.asarray(run.get("pet_mean", []), dtype=float)
        if years.size == 0 or p_mean.size == 0 or pet_mean.size == 0:
            continue
        cols = _year_color_rgba(years, pivot_year=pivot, hist_min_year=hist_min, hist_max_year=hist_max, fut_min_year=fut_min, fut_max_year=fut_max)
        if p_mean.size >= 2:
            seg = np.stack([np.stack([p_mean[:-1], pet_mean[:-1]], axis=1), np.stack([p_mean[1:], pet_mean[1:]], axis=1)], axis=1)
            lc = LineCollection(seg, colors=cols[:-1, :], linewidths=1.0, alpha=0.85)
            ax.add_collection(lc)
        ax.scatter(p_mean, pet_mean, s=18.0, c=cols, linewidths=0.0, alpha=0.9, label=run.get("run_id", "run"))
        all_years.append(years)

    era5 = data.get("era5")
    if era5:
        years_e = np.asarray(era5.get("years", []), dtype=int)
        p_mean_e = np.asarray(era5.get("p_mean", []), dtype=float)
        pet_mean_e = np.asarray(era5.get("pet_mean", []), dtype=float)
        cols_e = _year_color_green(years_e)
        if p_mean_e.size >= 2:
            seg_e = np.stack(
                [np.stack([p_mean_e[:-1], pet_mean_e[:-1]], axis=1), np.stack([p_mean_e[1:], pet_mean_e[1:]], axis=1)],
                axis=1,
            )
            lc_e = LineCollection(seg_e, colors=cols_e[:-1, :], linewidths=1.4, alpha=0.95)
            ax.add_collection(lc_e)
        ax.scatter(p_mean_e, pet_mean_e, s=20.0, c=cols_e, linewidths=0.0, alpha=0.95, label="ERA5")
        if years_e.size:
            all_years.append(years_e)

    ax.set_xlabel(f"Accumulated precipitation over {scale} months (mm)")
    ax.set_ylabel(f"Accumulated PET over {scale} months (mm)")
    ax.set_xlim(p_min, p_max)
    ax.set_ylim(pet_min, pet_max)

    years_cat = np.concatenate(all_years) if all_years else np.array([], dtype=int)
    years_era = np.asarray(era5.get("years", []), dtype=int) if era5 else np.array([], dtype=int)
    try:
        _add_timeline_legend(
            fig,
            years_gcm=years_cat,
            years_era=years_era,
            hist_min=hist_min,
            hist_max=hist_max,
            fut_min=fut_min,
            fut_max=fut_max,
            pivot=pivot,
            n_ensembles=len(runs),
        )
    except Exception:
        pass

    title = data.get(
        "title",
        f"{region.upper()} | {forcing_label} | January region mean — SPEI{scale} (baseline={baseline_label}/{baseline_strategy}, PET={pet_method}, group={group_pixels})",
    )
    fig.suptitle(title, fontsize=12)

    base_output = path.with_suffix("")
    pdf_out = Path(output_pdf) if output_pdf else base_output.with_suffix(".pdf")
    png_out = Path(output_png) if output_png else base_output.with_suffix(".png")
    _save_pdf_and_png(fig, pdf_out, dpi=320)
    plt.close(fig)
    return pdf_out, png_out


def replot_annualmean_excerpt(
    json_path: str | Path,
    *,
    output_pdf: Optional[str | Path] = None,
    output_png: Optional[str | Path] = None,
) -> Tuple[Path, Path]:
    """
    Recreate the annual-mean COMPARISON_SPEI_INPUTS excerpt from its JSON payload.
    """
    if plt is None:
        raise RuntimeError("matplotlib is unavailable; cannot replot excerpt.")
    path = Path(json_path)
    data = json.loads(path.read_text())

    color_map = data.get(
        "color_map",
        {
            "ERA5": "#000000",
            FORCING_SCENARIO1_LABEL: "#dba25c",
            FORCING_SCENARIO2_LABEL: "#5c9edb",
            "CMIP6-HIST": "#c70000",
            "CMIP6-HIST-NAT": "#7b3fa6",
            "CMIP6-SSP245": "#2f8f46",
        },
    )
    style_map = data.get("style_map", {})
    vars_in = data.get("variables", [])
    x_range = data.get("x_range", [None, None])
    xmin = np.datetime64(x_range[0]) if x_range and x_range[0] else None
    xmax = np.datetime64(x_range[1]) if x_range and x_range[1] else None
    region = data.get("region", "")
    pet_method = data.get("pet_method", "")

    fig, axes = plt.subplots(len(vars_in), 1, figsize=(16, 2.0 + 2.3 * max(1, len(vars_in))), sharex=True)
    if len(vars_in) == 1:
        axes = [axes]  # type: ignore

    seen_forcing: List[str] = []
    for ax, var_entry in zip(axes, vars_in):
        units = var_entry.get("units")
        for series in var_entry.get("series", []):
            forcing_label = series.get("forcing", "UNK")
            run_id = series.get("run_id", "run")
            times = np.asarray([np.datetime64(t) for t in series.get("times", [])], dtype="datetime64[ns]")
            vals = np.asarray(series.get("values", []), dtype=float)
            if times.size == 0 or vals.size == 0:
                continue
            style_here = style_map.get(forcing_label, {})
            color = style_here.get("color", color_map.get(forcing_label, "#444444"))
            alpha = float(style_here.get("alpha", 0.5))
            linewidth = float(style_here.get("linewidth", 0.7))
            linestyle = str(style_here.get("linestyle", "-"))
            ax.plot(
                times,
                vals,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                linestyle=linestyle,
                label=f"{forcing_label}:{run_id}",
            )
            if forcing_label not in seen_forcing:
                seen_forcing.append(str(forcing_label))
        label = var_entry.get("name", "var")
        if units:
            label = f"{label} ({units})"
        ax.set_ylabel(label)
        ax.grid(alpha=0.2, linewidth=0.5)
        if xmin is not None and xmax is not None:
            ax.set_xlim(xmin, xmax)
    if vars_in and axes:
        axes[-1].set_xlabel("Time")
    # Build legend from unique forcing labels
    handles = {}
    for forcing in seen_forcing:
        style_here = style_map.get(forcing, {})
        color = style_here.get("color", color_map.get(forcing, "#444444"))
        alpha = float(style_here.get("alpha", 0.5))
        linewidth = float(style_here.get("linewidth", 1.8))
        linestyle = str(style_here.get("linestyle", "-"))
        handles[forcing] = axes[0].plot(
            [],
            [],
            color=color,
            alpha=alpha,
            linewidth=max(1.2, linewidth),
            linestyle=linestyle,
            label=forcing,
        )[0]
    if handles:
        axes[0].legend(handles=list(handles.values()), loc="upper left", ncol=min(4, len(handles)), fontsize=9)

    title = data.get("title", f"COMPARISON_SPEI_INPUTS — {region.upper()} — PET={pet_method} — annual mean")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    base_output = path.with_suffix("")
    pdf_out = Path(output_pdf) if output_pdf else base_output.with_suffix(".pdf")
    png_out = Path(output_png) if output_png else base_output.with_suffix(".png")
    _save_pdf_and_png(fig, pdf_out, dpi=320)
    plt.close(fig)
    return pdf_out, png_out

def _write_summary_pdf(
    *,
    root_for_outputs: Path,
    run_id: str,
    forcing_label: str,
    region: str,
    scale: int,
    pet_method: str,
    fit: str,
    baseline: xr.Dataset,
    ds: xr.Dataset,
    baseline_label: str,
    baseline_strategy: str,
    pivot_year: int,
    output_tag: Optional[str],
    group_pixels: int,
) -> Optional[Path]:
    """
    Write a 12-panel (Jan..Dec) PDF: x=P_accum, y=PET_accum, background drought classes (USDM),
    with per-gridpoint trajectories and region-mean trajectory.
    """
    if plt is None or PdfPages is None or ListedColormap is None or BoundaryNorm is None or LineCollection is None:
        _log("    ↷ matplotlib unavailable; skipping summary PDF.")
        return None

    spei_var = f"spei{scale}"
    p_var = f"p{scale}"
    pet_var = f"pet{scale}"
    if spei_var not in ds or p_var not in ds or pet_var not in ds:
        _log(f"    ↷ missing variables for summary PDF (need {spei_var}, {p_var}, {pet_var}); skipping.")
        return None

    out_dir = _summary_pdf_dir(root_for_outputs, output_tag=output_tag)
    safe_run = _sanitize_run_id(run_id)
    safe_pet = _sanitize_run_id(pet_method)
    safe_fit = (fit or "fit").lower().strip()
    safe_strat = (baseline_strategy or "strategy").lower().strip()
    group_tag = f"grp{int(group_pixels)}"
    out_path = out_dir / f"SUMMARY__{forcing_label}__{safe_run}__spei{scale}__{region.upper()}__{safe_pet}__{safe_fit}__{safe_strat}__{group_tag}.pdf"

    # Axis limits (shared across panels)
    p_all = np.asarray(ds[p_var].values, dtype=float)
    pet_all = np.asarray(ds[pet_var].values, dtype=float)
    if not (np.isfinite(p_all).any() and np.isfinite(pet_all).any()):
        _log("    ↷ no finite P/PET values; skipping summary PDF.")
        return None
    # Use region-mean series for axis limits (tighter focus on the mean trajectory)
    p_mean_series = np.nanmean(p_all, axis=1)
    pet_mean_series = np.nanmean(pet_all, axis=1)
    p_min = float(np.nanmin(p_mean_series))
    p_max = float(np.nanmax(p_mean_series))
    pet_min = float(np.nanmin(pet_mean_series))
    pet_max = float(np.nanmax(pet_mean_series))
    # modest padding
    px = max(1e-6, (p_max - p_min))
    py = max(1e-6, (pet_max - pet_min))
    p_min -= 0.05 * px
    p_max += 0.05 * px
    pet_min -= 0.05 * py
    pet_max += 0.05 * py

    # Background drought classes (USDM thresholds)
    levels = [-10.0, USDM_SPEI_LEVELS[0], USDM_SPEI_LEVELS[1], USDM_SPEI_LEVELS[2], USDM_SPEI_LEVELS[3], USDM_SPEI_LEVELS[4], 10.0]
    # Colors in order: D4, D3, D2, D1, D0, None/Normal
    cmap = ListedColormap(["#7f0000", "#d73027", "#f46d43", "#fdae61", "#fff7a8", "#ffffff"])
    norm = BoundaryNorm(levels, ncolors=cmap.N)

    # Shared year->color mapping across panels
    years_all = np.asarray(ds["time"].dt.year.values, dtype=int)
    if years_all.size == 0:
        _log("    ↷ empty time axis; skipping summary PDF.")
        return None
    pivot = int(pivot_year)
    hist_years = years_all[years_all <= pivot]
    fut_years = years_all[years_all > pivot]
    hist_min = int(np.min(hist_years)) if hist_years.size else int(np.min(years_all))
    hist_max = int(np.max(hist_years)) if hist_years.size else pivot
    fut_min = int(np.min(fut_years)) if fut_years.size else (pivot + 1)
    fut_max = int(np.max(fut_years)) if fut_years.size else (pivot + 1)

    # Grid for background contours
    nx, ny = 220, 220
    xx = np.linspace(p_min, p_max, nx, dtype=float)
    yy = np.linspace(pet_min, pet_max, ny, dtype=float)
    X, Y = np.meshgrid(xx, yy)
    WB = X - Y

    lat_pts = np.asarray(ds["lat"].values, dtype=float) if "lat" in ds else np.array([])
    lon_pts = np.asarray(ds["lon"].values, dtype=float) if "lon" in ds else np.array([])

    # Figure
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharex=True, sharey=True)
    axes_flat = axes.ravel().tolist()

    # One mappable for shared colorbar
    mappable_for_cb = None

    with PdfPages(out_path) as pdf:
        for mi, m in enumerate(range(1, 13)):
            ax = axes_flat[mi]

            # Background SPEI field for this month (region-mean baseline params)
            params = _baseline_mean_params_for_month(baseline, m, fit=fit)
            Z = _spei_from_wb_grid(WB, fit=fit, params=params)
            cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm, extend="both")
            # Thin boundary contours at thresholds
            ax.contour(X, Y, Z, levels=list(USDM_SPEI_LEVELS), colors="k", linewidths=0.4, alpha=0.35)
            if mappable_for_cb is None:
                mappable_for_cb = cf

            # Monthly trajectories
            ds_m = ds.sel(time=ds["time"].dt.month == m)
            if ds_m.sizes.get("time", 0) == 0:
                ax.set_title(_month_name(m))
                continue
            Pm = np.asarray(ds_m[p_var].values, dtype=float)      # (t, point)
            PETm = np.asarray(ds_m[pet_var].values, dtype=float)  # (t, point)
            t_years = np.asarray(ds_m["time"].dt.year.values, dtype=int)

            cols_t = (
                _year_color_green(t_years)
                if forcing_label.upper() == "ERA5"
                else _year_color_rgba(
                    t_years,
                    pivot_year=pivot,
                    hist_min_year=hist_min,
                    hist_max_year=hist_max,
                    fut_min_year=fut_min,
                    fut_max_year=fut_max,
                )
            )

            # Region mean only (requested)
            Pmean = np.nanmean(Pm, axis=1)
            PETmean = np.nanmean(PETm, axis=1)
            n_t = Pmean.size
            if n_t >= 2:
                seg_mean = np.stack(
                    [np.stack([Pmean[:-1], PETmean[:-1]], axis=1), np.stack([Pmean[1:], PETmean[1:]], axis=1)],
                    axis=1,
                )
                lc_mean = LineCollection(seg_mean, colors=cols_t[:-1, :], linewidths=1.2, alpha=0.95)
                ax.add_collection(lc_mean)
            ax.scatter(Pmean, PETmean, s=22.0, c=cols_t, linewidths=0.0, alpha=0.98)

            ax.set_title(_month_name(m))
            ax.set_xlim(p_min, p_max)
            ax.set_ylim(pet_min, pet_max)

        # Axis labels only on outer edges
        for r in range(3):
            axes[r, 0].set_ylabel(f"Accumulated PET over {scale} months (mm)")
        for c in range(4):
            axes[2, c].set_xlabel(f"Accumulated precipitation over {scale} months (mm)")

        title = (
            f"{region.upper()} | {forcing_label} | {run_id}\n"
            f"SPEI{scale} background (USDM classes) using baseline={baseline_label} ({baseline_strategy}), "
            f"fit={fit}, PET={pet_method}, group={group_pixels} | baseline years {baseline.attrs.get('baseline_start_year','?')}-{baseline.attrs.get('baseline_end_year','?')}"
        )
        fig.suptitle(title, fontsize=12)

        # Shared colorbar with class labels (anchored on the lower-right)
        if mappable_for_cb is not None:
            # Reserve space for manual legends/colorbar before adding custom axes
            fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.14, wspace=0.16, hspace=0.18)

            cb_ax = fig.add_axes([0.72, 0.035, 0.24, 0.025], zorder=5)
            cb = fig.colorbar(
                mappable_for_cb,
                cax=cb_ax,
                orientation="horizontal",
            )
            # Place ticks in the middle of each class band
            mids = [
                (-10.0 + USDM_SPEI_LEVELS[0]) / 2.0,
                (USDM_SPEI_LEVELS[0] + USDM_SPEI_LEVELS[1]) / 2.0,
                (USDM_SPEI_LEVELS[1] + USDM_SPEI_LEVELS[2]) / 2.0,
                (USDM_SPEI_LEVELS[2] + USDM_SPEI_LEVELS[3]) / 2.0,
                (USDM_SPEI_LEVELS[3] + USDM_SPEI_LEVELS[4]) / 2.0,
                (USDM_SPEI_LEVELS[4] + 10.0) / 2.0,
            ]
            cb.set_ticks(mids)
            cb.set_ticklabels(["D4", "D3", "D2", "D1", "D0", "None"])
            cb.ax.tick_params(labelsize=8, pad=2)
            cb.set_label("US Drought Monitor drought classification (SPEI thresholds)")

        try:
            _add_timeline_legend(
                fig,
                years_gcm=np.asarray(ds["time"].dt.year.values, dtype=int),
                years_era=np.array([], dtype=int),
                hist_min=hist_min,
                hist_max=hist_max,
                fut_min=fut_min,
                fut_max=fut_max,
                pivot=pivot,
                n_ensembles=1,
            )
        except Exception:
            pass

        fig.tight_layout(rect=[0.02, 0.1, 0.98, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # ------------------------------------------------------------------
        # Page 2+: January panels per grid point
        # ------------------------------------------------------------------
        jan_ds = ds.sel(time=ds["time"].dt.month == 1)
        if jan_ds.sizes.get("time", 0) > 0 and jan_ds.sizes.get("point", 0) > 0:
            Pj = np.asarray(jan_ds[p_var].values, dtype=float)       # (t, point)
            PETj = np.asarray(jan_ds[pet_var].values, dtype=float)   # (t, point)
            years_j = np.asarray(jan_ds["time"].dt.year.values, dtype=int)
            n_pts = Pj.shape[1]

            # Background for January (per-point limits to fully show ERA5 data)
            per_page = 9
            rows, cols_grid = 3, 3
            for start_idx in range(0, n_pts, per_page):
                fig_j, axes_j = plt.subplots(rows, cols_grid, figsize=(16, 11), sharex=False, sharey=False)
                axes_j_flat = axes_j.ravel().tolist()
                cf = None
                for j_idx in range(per_page):
                    pt = start_idx + j_idx
                    ax = axes_j_flat[j_idx]
                    if pt >= n_pts:
                        ax.set_axis_off()
                        continue
                    x_pt = np.asarray(Pj[:, pt], dtype=float)
                    y_pt = np.asarray(PETj[:, pt], dtype=float)
                    finite = np.isfinite(x_pt) & np.isfinite(y_pt)
                    if not np.any(finite):
                        ax.set_title(f"Point {pt} (no data)")
                        continue

                    x_pt = x_pt[finite]
                    y_pt = y_pt[finite]
                    yrs_pt = years_j[finite]

                    p_min_pt = float(np.nanmin(x_pt))
                    p_max_pt = float(np.nanmax(x_pt))
                    pet_min_pt = float(np.nanmin(y_pt))
                    pet_max_pt = float(np.nanmax(y_pt))
                    px_pt = max(1e-6, (p_max_pt - p_min_pt))
                    py_pt = max(1e-6, (pet_max_pt - pet_min_pt))
                    p_min_pt -= 0.05 * px_pt
                    p_max_pt += 0.05 * px_pt
                    pet_min_pt -= 0.05 * py_pt
                    pet_max_pt += 0.05 * py_pt

                    xx_pt = np.linspace(p_min_pt, p_max_pt, 220, dtype=float)
                    yy_pt = np.linspace(pet_min_pt, pet_max_pt, 220, dtype=float)
                    X_pt, Y_pt = np.meshgrid(xx_pt, yy_pt)
                    WB_pt = X_pt - Y_pt

                    # Point-specific baseline params for background
                    params_jan_pt = _baseline_point_params_for_month(baseline, 1, pt, fit=fit)
                    Z_jan_pt = _spei_from_wb_grid(WB_pt, fit=fit, params=params_jan_pt)
                    cf_local = ax.contourf(X_pt, Y_pt, Z_jan_pt, levels=levels, cmap=cmap, norm=norm, extend="both")
                    ax.contour(X_pt, Y_pt, Z_jan_pt, levels=list(USDM_SPEI_LEVELS), colors="k", linewidths=0.4, alpha=0.35)
                    if cf is None:
                        cf = cf_local

                    cols_j = (
                        _year_color_green(yrs_pt)
                        if forcing_label.upper() == "ERA5"
                        else _year_color_rgba(
                            yrs_pt,
                            pivot_year=pivot,
                            hist_min_year=hist_min,
                            hist_max_year=hist_max,
                            fut_min_year=fut_min,
                            fut_max_year=fut_max,
                        )
                    )

                    # Connecting segments by year
                    if x_pt.size >= 2:
                        seg_pt = np.stack(
                            [np.stack([x_pt[:-1], y_pt[:-1]], axis=1), np.stack([x_pt[1:], y_pt[1:]], axis=1)],
                            axis=1,
                        )
                        lc_pt = LineCollection(seg_pt, colors=cols_j[:-1, :], linewidths=0.9, alpha=0.9)
                        ax.add_collection(lc_pt)

                    ax.scatter(x_pt, y_pt, s=16.0, c=cols_j, linewidths=0.0, alpha=0.95)
                    ax.set_xlim(p_min_pt, p_max_pt)
                    ax.set_ylim(pet_min_pt, pet_max_pt)
                    lat_pt = jan_ds["lat"].values[pt] if "lat" in jan_ds else float("nan")
                    lon_pt = jan_ds["lon"].values[pt] if "lon" in jan_ds else float("nan")
                    ax.set_title(f"Point {pt} (lat={lat_pt:.2f}, lon={lon_pt:.2f})")
                    if np.isfinite(lat_pt) and np.isfinite(lon_pt):
                        _add_region_inset(
                            ax,
                            region=region,
                            lat_points=lat_pts,
                            lon_points=lon_pts,
                            fill_region=True,
                            highlight_point=(float(lat_pt), float(lon_pt)),
                            show_frame=False,
                        )

                if cf is not None:
                    cb_j = fig_j.colorbar(
                        cf,
                        ax=axes_j_flat,
                        orientation="horizontal",
                        fraction=0.045,
                        pad=0.1,
                    )
                    cb_j.set_ticks(mids)
                    cb_j.set_ticklabels(["D4", "D3", "D2", "D1", "D0", "None"])
                    cb_j.set_label("US Drought Monitor drought classification (SPEI thresholds)")

                fig_j.suptitle(f"{region.upper()} | {forcing_label} | {run_id} | January by grid point", fontsize=12)
                fig_j.tight_layout(rect=[0.02, 0.1, 0.98, 0.95])
                pdf.savefig(fig_j)
                plt.close(fig_j)

    _log(f"    ✓ wrote summary PDF: {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# Prototype execution mode: file-major scheduler
# -----------------------------------------------------------------------------
def _run_file_major_prototype(
    *,
    args: argparse.Namespace,
    cmip6_only: bool,
    baseline_cfg: Dict[str, Dict[str, str]],
    cmip6_baseline_cfg: Dict[str, Dict[str, str]],
    nat_root: Path,
    all_root: Path,
    era5_file: Path,
    output_tag: Optional[str],
    nat_files: Sequence[Path],
    all_files: Sequence[Path],
    include_cmip6: bool,
    cmip6_root: Optional[Path],
    cmip6_hist_files: Sequence[Path],
    cmip6_hist_nat_files: Sequence[Path],
    cmip6_ssp245_files: Sequence[Path],
    cmip6_manifest: Optional[Dict],
    months: Optional[Tuple[int, ...]],
    baseline_start_year: Optional[int],
    baseline_end_year: Optional[int],
    on_existing: str,
    archive_root_nat: Optional[Path],
    archive_root_all: Optional[Path],
    archive_root_era5: Optional[Path],
    archive_root_cmip6: Optional[Path],
    baseline_archive_root_nat: Optional[Path],
    baseline_archive_root_all: Optional[Path],
    baseline_archive_root_era5: Optional[Path],
    baseline_archive_root_cmip6: Optional[Path],
) -> None:
    regions = _resolve_execution_regions(args)
    pet_methods = _resolve_execution_pet_methods(args)
    nat_scens_mode = _normalize_rsdsbiasadjust_nat_scens_mode(str(args.rsdsbiasadjust_nat_scens))
    limit_to_scenario = _normalize_limit_to_scenario_token(getattr(args, "limit_to_scenario", None))
    selected_scenarios = set(_selected_scenario_targets(limit_to_scenario))
    run_scenario2 = (not cmip6_only) and ("scenario2" in selected_scenarios)
    run_scenario1 = (not cmip6_only) and ("scenario1" in selected_scenarios)
    run_era5 = (not cmip6_only) and (limit_to_scenario is None)
    if not regions:
        raise RuntimeError("No regions resolved for file-major-prototype mode.")
    if not pet_methods:
        raise RuntimeError("No PET methods resolved for file-major-prototype mode.")

    _log(
        "Execution mode=file-major-prototype "
        f"(regions={len(regions)}, pet_methods={len(pet_methods)}, "
        f"combos={len(regions) * len(pet_methods)})"
    )
    _log(f"CMIP6-only mode={cmip6_only}")
    _log(f"LIMIT_TO_SCENARIO={limit_to_scenario or 'none'}")
    _log(
        "RSDS ERA5 bias adjustment in prototype: "
        f"{bool(args.apply_rsds_to_era5_biasadjustment)} "
        f"(smooth={DEFAULT_RSDS_BIAS_SMOOTHING_YEARS}y, "
        f"edge-trend={DEFAULT_RSDS_BIAS_EDGE_TREND_YEARS}y, "
        f"edge-extend={DEFAULT_RSDS_BIAS_EDGE_EXTENSION_YEARS}y, "
        f"nat-mode={nat_scens_mode})"
    )
    if include_cmip6:
        _log(
            "CMIP6 enabled in prototype: "
            f"historical_files={len(cmip6_hist_files)}, "
            f"hist_nat_files={len(cmip6_hist_nat_files)}, "
            f"ssp245_files={len(cmip6_ssp245_files)}"
        )
        _log(
            "CMIP6 baseline config: "
            f"HIST(src={cmip6_baseline_cfg['cmip6_hist']['source']},pool={cmip6_baseline_cfg['cmip6_hist']['pooling']}); "
            f"HIST-NAT(src={cmip6_baseline_cfg['cmip6_hist_nat']['source']},pool={cmip6_baseline_cfg['cmip6_hist_nat']['pooling']}); "
            f"SSP245(src={cmip6_baseline_cfg['cmip6_ssp245']['source']},pool={cmip6_baseline_cfg['cmip6_ssp245']['pooling']})"
        )
    segments_layout = _normalize_segments_layout_token(str(args.segments_layout))
    output_format = _normalize_output_format_token(str(args.output_format))
    write_per_run = segments_layout in {SEGMENTS_LAYOUT_PER_RUN, SEGMENTS_LAYOUT_DUAL}
    write_run_stacked = segments_layout in {SEGMENTS_LAYOUT_RUN_STACKED, SEGMENTS_LAYOUT_DUAL}
    chunk_point = _parse_chunk_point_arg(str(args.chunk_point))
    _log(
        "Segments layout in prototype: "
        f"{segments_layout} "
        f"(output-format={output_format}, per_run={write_per_run}, run_stacked={write_run_stacked}, "
        f"chunk-run={int(args.chunk_run)}, chunk-time={int(args.chunk_time)}, "
        f"chunk-point={args.chunk_point}, zstd={int(args.compression_level)}, "
        f"consolidate={bool(args.consolidate_metadata)})"
    )
    if output_format == OUTPUT_FORMAT_NETCDF and write_per_run:
        _log("NetCDF output format applies to run-stacked bundles; per-run segments remain Zarr groups.")
    if not bool(args.no_summary_pdf):
        _log("⚠️ file-major-prototype currently skips summary/comparison PDFs to prioritize throughput.")

    if include_cmip6 and cmip6_root is not None and cmip6_manifest is not None:
        for region in regions:
            man_path = _write_cmip6_selection_manifest(
                cmip6_root=cmip6_root,
                manifest=cmip6_manifest,
                output_tag=output_tag,
                region=region,
            )
            if man_path is not None:
                _log(f"CMIP6 selection manifest written: {man_path}")

    templates: Dict[str, TemplatePoints] = {}
    point_groups_by_region: Dict[str, Optional[List[np.ndarray]]] = {}
    for region in regions:
        template, _, _ = _build_template_from_file(
            era5_file,
            region=region,
            pet_method="thornthwaite",
            label=f"ERA5 (template:{region})",
        )
        templates[region] = template
        point_groups_by_region[region] = _build_point_groups(template, int(args.group_pixels))
        _log(f"Template ready for region={region}: points={int(template.lat.size)}")

    rsds_bias_by_region: Dict[str, Optional[RsdsEra5BiasAdjustment]] = {r: None for r in regions}
    if bool(args.apply_rsds_to_era5_biasadjustment):
        gcm_for_bias = list(nat_files) + list(all_files)
        if _rsdsbiasadjust_should_exclude_nat_from_bias_fit(nat_scens_mode):
            pre = len(gcm_for_bias)
            gcm_for_bias = [
                p for p in gcm_for_bias
                if (not _scenario_has_suffix(p, suffix=RSDSBIASADJUST_NAT_SCENARIO_SUFFIX))
            ]
            _log(
                "RSDS bias-adjust source files after NAT filtering in prototype: "
                f"{len(gcm_for_bias)} (removed {pre - len(gcm_for_bias)} files with "
                f"suffix '{RSDSBIASADJUST_NAT_SCENARIO_SUFFIX}', mode={nat_scens_mode})"
            )
        if gcm_for_bias:
            for region in regions:
                adj = _build_rsds_era5_bias_adjustment(
                    era5_file=era5_file,
                    gcm_files=gcm_for_bias,
                    template=templates[region],
                    smoothing_window_years=DEFAULT_RSDS_BIAS_SMOOTHING_YEARS,
                )
                rsds_bias_by_region[region] = adj
                if adj is not None:
                    artifact_root = all_root if len(all_files) else (nat_root if len(nat_files) else era5_file.parent)
                    try:
                        _write_rsds_bias_adjustment_artifacts(
                            root_for_outputs=artifact_root,
                            output_tag=output_tag,
                            region=region,
                            template=templates[region],
                            adjustment=adj,
                        )
                    except Exception as exc:
                        _log(f"⚠️ Failed to write RSDS bias-adjustment artifacts for region={region}: {exc}")
        else:
            _log("RSDS ERA5 bias adjustment requested, but no SCENARIO1/SCENARIO2 files are available; skipping.")
    else:
        _log("RSDS ERA5 bias adjustment disabled by option in prototype mode.")

    ds_era5_meta = _open_dataset_safe(era5_file, decode_times=True)
    ds_era5_meta = _ensure_lon_0_360(ds_era5_meta)

    pet_plan: List[Tuple[str, str, List[str]]] = []
    for pet_req in pet_methods:
        try:
            pet_resolved, required_vars = _required_vars_for_pet(ds_era5_meta, pet_req)
        except Exception as exc:
            _log(f"⚠️ PET method '{pet_req}' not feasible on ERA5 template file: {exc} (skip)")
            continue
        miss = [v for v in required_vars if v not in ds_era5_meta]
        if miss:
            _log(f"⚠️ PET method '{pet_req}' missing required vars in ERA5 template: {miss} (skip)")
            continue
        pet_plan.append((pet_req, pet_resolved, list(required_vars)))

    if not pet_plan:
        try:
            ds_era5_meta.close()
        except Exception:
            pass
        raise RuntimeError("No valid PET methods left after validation.")

    _log(
        "Validated PET plan: "
        + ", ".join(f"{req}->{resolved}" for req, resolved, _ in pet_plan)
    )

    nat_by_eid: Dict[str, Path] = {}
    for f in nat_files:
        eid = _ensemble_id_from_stem(f.stem)
        if eid:
            nat_by_eid[eid.lower()] = f
    all_by_eid: Dict[str, Path] = {}
    for f in all_files:
        eid = _ensemble_id_from_stem(f.stem)
        if eid:
            all_by_eid[eid.lower()] = f

    baseline_roots: Dict[str, Path] = {
        "era5": era5_file.parent,
        "scenario2": nat_root,
        "scenario1": all_root,
    }
    target_roots: Dict[str, Path] = {
        "era5": era5_file.parent,
        "scenario2": nat_root,
        "scenario1": all_root,
    }
    archive_roots_forcing: Dict[str, Optional[Path]] = {
        "era5": archive_root_era5,
        "scenario2": archive_root_nat,
        "scenario1": archive_root_all,
        "cmip6_hist": archive_root_cmip6,
        "cmip6_hist_nat": archive_root_cmip6,
        "cmip6_ssp245": archive_root_cmip6,
    }
    baseline_archive_roots: Dict[str, Optional[Path]] = {
        "era5": baseline_archive_root_era5,
        "scenario2": baseline_archive_root_nat,
        "scenario1": baseline_archive_root_all,
    }
    baseline_files_by_source: Dict[str, Sequence[Path]] = {
        "era5": [era5_file],
        "scenario2": nat_files,
        "scenario1": all_files,
    }
    baseline_by_eid: Dict[str, Dict[str, Path]] = {
        "scenario2": nat_by_eid,
        "scenario1": all_by_eid,
    }
    baseline_cache: Dict[tuple, tuple[xr.Dataset, Path, str, str, str]] = {}

    def _rsds_bias_for_region_source(region: str, source_key: str) -> Optional[RsdsEra5BiasAdjustment]:
        if source_key not in {"scenario2", "scenario1"}:
            return None
        if source_key == "scenario2" and nat_scens_mode == "excempt":
            return None
        return rsds_bias_by_region.get(region)

    def _rsds_bias_tag_for_region_source(region: str, source_key: str) -> str:
        adj = _rsds_bias_for_region_source(region, source_key)
        return _rsds_bias_adjustment_tag(
            adj,
            rsdsbiasadjust_nat_scens_mode=nat_scens_mode,
        )

    def _get_baseline_for_target(
        *,
        target: str,
        run_id: Optional[str],
        region: str,
        pet_resolved: str,
        required_vars: Sequence[str],
    ) -> Tuple[xr.Dataset, Path, str, str, str]:
        cfg = baseline_cfg[target]
        src = cfg["source"].lower()
        pool = cfg["pooling"].lower()

        pool_eff = "pooled" if src == "era5" else pool
        if target == "era5" and pool_eff == "per_member" and src in {"scenario2", "scenario1"}:
            _log(
                f"ERA5 target requested per_member baseline from {src.upper()}; "
                "falling back to pooled (no ERA5 ensemble id)."
            )
            pool_eff = "pooled"

        eid = None
        if pool_eff == "per_member":
            eid = _ensemble_id_from_stem(run_id or "")
            if not eid:
                _log("per_member baseline requested but no ensemble id found; falling back to pooled.")
                pool_eff = "pooled"

        rsds_bias_tag = _rsds_bias_tag_for_region_source(region, src)
        rsds_bias_adj = _rsds_bias_for_region_source(region, src)
        cache_key = (region, pet_resolved, src, pool_eff, (eid or "pooled").lower(), rsds_bias_tag)
        if cache_key in baseline_cache:
            ds_cached, fit_cached, b_lbl, b_src, b_pool = baseline_cache[cache_key]
            target_root = target_roots.get(target)
            fit_for_target = fit_cached
            if target_root is not None:
                try:
                    fit_for_target = _mirror_fit_to_target(
                        fit_cached, target_root, region=region, output_tag=output_tag
                    )
                except Exception as exc:
                    _log(f"    ⚠️ failed to mirror baseline fit into {target_root}: {exc}")
            return ds_cached, fit_for_target, b_lbl, b_src, b_pool

        b_root = baseline_roots[src]
        b_archive = baseline_archive_roots.get(src)
        if pool_eff == "pooled":
            b_files = baseline_files_by_source[src]
            b_id = f"{src}-pooled"
            b_label = f"{src.upper()} pooled"
        else:
            assert eid is not None
            if src not in baseline_by_eid or eid.lower() not in baseline_by_eid[src]:
                _log(
                    f"per_member baseline requested (src={src.upper()}) but member {eid} not found; "
                    "falling back to pooled."
                )
                b_files = baseline_files_by_source[src]
                b_id = f"{src}-pooled"
                b_label = f"{src.upper()} pooled"
                pool_eff = "pooled"
            else:
                b_files = [baseline_by_eid[src][eid.lower()]]
                b_id = f"{src}-{eid.lower()}"
                b_label = f"{src.upper()} member {eid.lower()}"

        ds_fit, fit_path = _load_or_build_baseline(
            baseline_root=b_root,
            baseline_files=b_files,
            baseline_source_key=src,
            baseline_pooling=pool_eff,
            baseline_id=b_id,
            baseline_label=b_label,
            region=region,
            scale=args.scale,
            template=templates[region],
            pet_method_resolved=pet_resolved,
            required_vars=required_vars,
            fit=args.fit,
            limit_runs=args.limit_runs if pool_eff == "pooled" else None,
            force=args.force,
            baseline_start_year=baseline_start_year,
            baseline_end_year=baseline_end_year,
            archive_root=b_archive,
            output_tag=output_tag,
            group_pixels=int(args.group_pixels),
            point_groups=point_groups_by_region[region],
            rsds_bias_adjustment=rsds_bias_adj,
            rsds_bias_adjustment_tag=rsds_bias_tag,
            rsdsbiasadjust_nat_scens_mode=nat_scens_mode,
            rsdsbiasadjust_nat_suffix=RSDSBIASADJUST_NAT_SCENARIO_SUFFIX,
        )

        target_root = target_roots.get(target)
        fit_for_target = fit_path
        if target_root is not None:
            try:
                fit_for_target = _mirror_fit_to_target(
                    fit_path, target_root, region=region, output_tag=output_tag
                )
            except Exception as exc:
                _log(f"    ⚠️ failed to mirror baseline fit into {target_root}: {exc}")

        baseline_cache[cache_key] = (ds_fit, fit_path, b_label, src, pool_eff)
        return ds_fit, fit_for_target, b_label, src, pool_eff

    cmip6_by_exp_files, cmip6_by_exp_source, cmip6_by_exp_source_member = _build_cmip6_file_index(
        cmip6_hist_files,
        cmip6_hist_nat_files,
        cmip6_ssp245_files,
    )
    cmip6_baseline_cache: Dict[Tuple[str, str, str, str, str, str], Tuple[xr.Dataset, Path, str, str, str]] = {}

    def _get_cmip6_baseline_for_target(
        *,
        target_key: str,
        run_path: Path,
        region: str,
        pet_resolved: str,
        required_vars: Sequence[str],
    ) -> Tuple[xr.Dataset, Path, str, str, str]:
        if cmip6_root is None:
            raise RuntimeError("CMIP6 root is not configured.")

        (
            baseline_files,
            baseline_source_key,
            baseline_pooling,
            baseline_id,
            baseline_label,
            cache_token,
            warning_note,
        ) = _resolve_cmip6_baseline_selection(
            run_path=run_path,
            target_key=target_key,
            cmip6_baseline_cfg=cmip6_baseline_cfg,
            by_exp_files=cmip6_by_exp_files,
            by_exp_source=cmip6_by_exp_source,
            by_exp_source_member=cmip6_by_exp_source_member,
        )
        if warning_note:
            _log(f"    ⚠️ {warning_note}")

        cache_key = (
            region,
            pet_resolved,
            target_key,
            baseline_source_key,
            baseline_pooling,
            cache_token.lower(),
        )
        if cache_key in cmip6_baseline_cache:
            ds_cached, fit_cached, b_lbl, b_src, b_pool = cmip6_baseline_cache[cache_key]
            fit_for_target = fit_cached
            try:
                fit_for_target = _mirror_fit_to_target(
                    fit_cached, cmip6_root, region=region, output_tag=output_tag
                )
            except Exception:
                pass
            return ds_cached, fit_for_target, b_lbl, b_src, b_pool

        ds_fit, fit_path = _load_or_build_baseline(
            baseline_root=cmip6_root,
            baseline_files=baseline_files,
            baseline_source_key=baseline_source_key,
            baseline_pooling=baseline_pooling,
            baseline_id=baseline_id,
            baseline_label=baseline_label,
            region=region,
            scale=args.scale,
            template=templates[region],
            pet_method_resolved=pet_resolved,
            required_vars=required_vars,
            fit=args.fit,
            limit_runs=args.limit_runs if baseline_pooling == "pooled" else None,
            force=args.force,
            baseline_start_year=baseline_start_year,
            baseline_end_year=baseline_end_year,
            archive_root=baseline_archive_root_cmip6,
            output_tag=output_tag,
            group_pixels=int(args.group_pixels),
            point_groups=point_groups_by_region[region],
            rsds_bias_adjustment=None,
            rsds_bias_adjustment_tag="off",
            rsdsbiasadjust_nat_scens_mode=nat_scens_mode,
            rsdsbiasadjust_nat_suffix=RSDSBIASADJUST_NAT_SCENARIO_SUFFIX,
        )
        fit_for_target = fit_path
        try:
            fit_for_target = _mirror_fit_to_target(
                fit_path, cmip6_root, region=region, output_tag=output_tag
            )
        except Exception:
            pass

        out = (ds_fit, fit_for_target, baseline_label, baseline_source_key, baseline_pooling)
        cmip6_baseline_cache[cache_key] = out
        return out

    forcing_groups = []
    if run_era5:
        forcing_groups.append(("ERA5", [era5_file], era5_file.parent, "era5"))
    if run_scenario2:
        forcing_groups.append((FORCING_SCENARIO2_LABEL, nat_files, nat_root, "scenario2"))
    if run_scenario1:
        forcing_groups.append((FORCING_SCENARIO1_LABEL, all_files, all_root, "scenario1"))
    if include_cmip6 and cmip6_root is not None:
        forcing_groups.extend(
            [
                ("CMIP6-HIST", list(cmip6_hist_files), cmip6_root, "cmip6_hist"),
                ("CMIP6-HIST-NAT", list(cmip6_hist_nat_files), cmip6_root, "cmip6_hist_nat"),
                ("CMIP6-SSP245", list(cmip6_ssp245_files), cmip6_root, "cmip6_ssp245"),
            ]
        )
    if not forcing_groups:
        raise RuntimeError("No forcing groups selected for file-major-prototype run.")

    try:
        for forcing, files, output_root, target_key in forcing_groups:
            if not files:
                _log(f"No files found for forcing={forcing}")
                continue
            stacked_writers: Dict[Tuple[str, str, str], RunStackedWriter | NetCDFRunStackedWriter] = {}

            files_iter = files[: args.limit_runs] if args.limit_runs is not None else files
            if target_key in {"cmip6_hist", "cmip6_hist_nat", "cmip6_ssp245"}:
                files_iter, dropped_records = _filter_cmip6_runs_with_exact_baselines(
                    files=files_iter,
                    target_key=target_key,
                    cmip6_baseline_cfg=cmip6_baseline_cfg,
                    by_exp_files=cmip6_by_exp_files,
                    by_exp_source=cmip6_by_exp_source,
                    by_exp_source_member=cmip6_by_exp_source_member,
                )
                if dropped_records:
                    _log(
                        f"Dropping {len(dropped_records)} {forcing} run(s) without exact "
                        f"{cmip6_baseline_cfg[target_key]['source']} member baselines."
                    )
                    for record in dropped_records[:10]:
                        if _record_cmip6_exact_baseline_drop(
                            manifest=cmip6_manifest,
                            record=record,
                            cmip6_root=cmip6_root,
                            output_tag=output_tag,
                            regions=regions,
                            target_key=target_key,
                        ):
                            _log(f"  - drop {Path(str(record['run_file'])).name}: {record['reason']}")
                    if len(dropped_records) > 10:
                        _log(f"  ... and {len(dropped_records) - 10} more dropped run(s).")
                if not files_iter:
                    _log(f"No retained CMIP6 files remain for forcing={forcing} after exact-baseline filtering.")
                    continue
            _log(
                f"\nFile-major pass for {forcing}: files={len(files_iter)}, "
                f"regions={len(regions)}, pets={len(pet_plan)}"
            )

            for i, path in enumerate(files_iter, 1):
                run_id = "ERA5" if forcing == "ERA5" else path.stem
                _log(f"  [{i}/{len(files_iter)}] {forcing}: {path.name}")

                ds_file: Optional[xr.Dataset] = None
                try:
                    ds_file = _open_dataset_safe(path, decode_times=True)
                    ds_file = _ensure_lon_0_360(ds_file)
                except Exception as exc:
                    _log(f"    ✗ failed to open file: {exc}")
                    if ds_file is not None:
                        try:
                            ds_file.close()
                        except Exception:
                            pass
                    continue

                for region in regions:
                    store_dir = _segments_store_dir(output_root, output_tag=output_tag, region=region)
                    template = templates[region]
                    point_groups = point_groups_by_region[region]

                    for pet_req, pet_resolved, required_vars in pet_plan:
                        combo_tag = f"region={region} pet={pet_resolved} (req={pet_req})"
                        try:
                            if target_key in {"cmip6_hist", "cmip6_hist_nat", "cmip6_ssp245"}:
                                baseline_ds, baseline_fit_file, b_label, b_src_key, b_pool = _get_cmip6_baseline_for_target(
                                    target_key=target_key,
                                    run_path=path,
                                    region=region,
                                    pet_resolved=pet_resolved,
                                    required_vars=required_vars,
                                )
                            else:
                                baseline_ds, baseline_fit_file, b_label, b_src_key, b_pool = _get_baseline_for_target(
                                    target=target_key,
                                    run_id=None if forcing == "ERA5" else run_id,
                                    region=region,
                                    pet_resolved=pet_resolved,
                                    required_vars=required_vars,
                                )
                        except MissingExactCmip6BaselineError as exc:
                            if _record_cmip6_exact_baseline_drop(
                                manifest=cmip6_manifest,
                                record=exc.record,
                                cmip6_root=cmip6_root,
                                output_tag=output_tag,
                                regions=regions,
                                target_key=target_key,
                            ):
                                _log(f"    ⚠️ dropped {run_id}: {exc}")
                            continue
                        except Exception as exc:
                            _log(f"    ⚠️ baseline build failed for {combo_tag}: {exc}")
                            continue

                        try:
                            rsds_bias_raw = (
                                rsds_bias_by_region.get(region)
                                if forcing in {FORCING_SCENARIO1_LABEL, FORCING_SCENARIO2_LABEL}
                                else None
                            )
                            rsds_bias_for_run, rsds_bias_hold_early_for_run = _resolve_rsds_bias_adjustment_for_path(
                                path,
                                adjustment=rsds_bias_raw,
                                rsdsbiasadjust_nat_scens_mode=nat_scens_mode,
                                nat_suffix=RSDSBIASADJUST_NAT_SCENARIO_SUFFIX,
                            )
                            bundle, _ = _compute_speix_bundle_for_dataset(
                                ds_file,
                                template=template,
                                required_vars=required_vars,
                                pet_method_resolved=pet_resolved,
                                baseline=baseline_ds,
                                scale=args.scale,
                                fit=args.fit,
                                point_groups=point_groups,
                                group_pixels=int(args.group_pixels),
                                rsds_bias_adjustment=rsds_bias_for_run,
                                rsds_bias_hold_first_reference_year_offsets=rsds_bias_hold_early_for_run,
                            )
                            bundle_plot = _filter_spei_output(
                                bundle,
                                out_start_year=args.out_start_year,
                                out_end_year=args.out_end_year,
                                months=None,
                            )
                            bundle_save = _filter_spei_output(
                                bundle_plot,
                                out_start_year=args.out_start_year,
                                out_end_year=args.out_end_year,
                                months=months,
                            )
                        except Exception as exc:
                            _log(f"    ✗ compute failed for {combo_tag}: {exc}")
                            continue

                        if write_per_run:
                            wrote = _write_spei_segment(
                                store_dir=store_dir,
                                run_id=run_id,
                                region=region,
                                scale=args.scale,
                                payload=bundle_save,
                                forcing_label=forcing,
                                baseline_fit_file=baseline_fit_file,
                                baseline_source_key=b_src_key,
                                baseline_pooling=b_pool,
                                baseline_label=b_label,
                                pet_method=pet_resolved,
                                fit=args.fit,
                                on_existing=on_existing,
                                archive_root=archive_roots_forcing.get(target_key),
                                pet_in_path=True,
                                group_pixels=int(args.group_pixels),
                                chunk_time=120,
                                chunk_point=None,
                                compression_level=0,
                            )
                            if wrote:
                                _log(f"    ✓ wrote [per_run|{combo_tag}] {wrote}")

                        if write_run_stacked:
                            writer_key = (str(store_dir), region, pet_resolved)
                            if writer_key not in stacked_writers:
                                stacked_writers[writer_key] = _make_run_stacked_writer(
                                    output_format=output_format,
                                    store_dir=store_dir,
                                    forcing_label=forcing,
                                    region=region,
                                    scale=int(args.scale),
                                    pet_method=pet_resolved,
                                    fit=str(args.fit),
                                    on_existing=on_existing,
                                    archive_root=archive_roots_forcing.get(target_key),
                                    group_pixels=int(args.group_pixels),
                                    chunk_run=int(args.chunk_run),
                                    chunk_time=int(args.chunk_time),
                                    chunk_point=chunk_point,
                                    compression_level=int(args.compression_level),
                                )
                            wrote_stacked = stacked_writers[writer_key].write_run(
                                run_id=run_id,
                                payload=bundle_save,
                                baseline_fit_file=baseline_fit_file,
                                baseline_source_key=b_src_key,
                                baseline_pooling=b_pool,
                                baseline_strategy=f"{b_src_key}:{b_pool}",
                            )
                            if wrote_stacked:
                                _log(f"    ✓ wrote [run_stacked|{combo_tag}] {wrote_stacked}")

                try:
                    ds_file.close()
                except Exception:
                    pass

            if write_run_stacked and stacked_writers:
                touched: Set[Path] = set()
                for writer in stacked_writers.values():
                    out_store = writer.finalize()
                    if out_store is not None:
                        touched.add(out_store)
                if output_format == OUTPUT_FORMAT_ZARR and bool(args.consolidate_metadata):
                    for out_store in sorted(touched):
                        _consolidate_zarr_store_metadata(out_store)

    finally:
        try:
            ds_era5_meta.close()
        except Exception:
            pass

    _log("file-major-prototype run complete (summary/comparison PDFs skipped).")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_months(raw: str) -> Optional[Tuple[int, ...]]:
    s = (raw or "").strip().lower()
    if s in {"", "all", "none"}:
        return None
    # Accept comma-separated list: "6,7,8"
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        v = int(p)
        if v < 1 or v > 12:
            raise ValueError("Months must be in 1..12.")
        out.append(v)
    return tuple(out)


def _normalize_segments_layout_token(raw: str) -> str:
    token = str(raw or DEFAULT_SEGMENTS_LAYOUT).strip().lower()
    if token not in set(SEGMENTS_LAYOUT_CHOICES):
        raise ValueError(
            f"Unsupported segments layout '{raw}'. "
            f"Choose one of: {', '.join(SEGMENTS_LAYOUT_CHOICES)}."
        )
    return token


def _normalize_output_format_token(raw: str) -> str:
    token = str(raw or DEFAULT_OUTPUT_FORMAT).strip().lower()
    if token not in set(OUTPUT_FORMAT_CHOICES):
        raise ValueError(
            f"Unsupported output format '{raw}'. "
            f"Choose one of: {', '.join(OUTPUT_FORMAT_CHOICES)}."
        )
    return token


def _parse_chunk_point_arg(raw: str) -> Optional[int]:
    token = str(raw or "").strip().lower()
    if token in {"", "auto", "none"}:
        return None
    try:
        val = int(token)
    except Exception as exc:
        raise ValueError(f"--chunk-point must be an integer or 'auto', got: {raw}") from exc
    if val <= 0:
        raise ValueError(f"--chunk-point must be >= 1 or 'auto', got: {raw}")
    return int(val)


def _parse_csv_items(raw: Optional[str], *, lower: bool = False) -> List[str]:
    if raw is None:
        return []
    vals = [p.strip() for p in str(raw).split(",") if p.strip()]
    if lower:
        vals = [v.lower() for v in vals]
    return vals


def _normalize_baseline_source_token(token: str) -> str:
    """
    Normalize user-facing baseline source token to internal keys.
    """
    t = str(token).strip().lower()
    if t in {"all", "nat"}:
        raise ValueError(
            f"Unsupported baseline source alias '{token}'. "
            "Use 'scenario1' or 'scenario2'."
        )
    return t


def _normalize_limit_to_scenario_token(raw: Optional[str]) -> Optional[str]:
    token = str(raw or "").strip().lower()
    if token in {"", "none", "null"}:
        return None
    if token in {"scenario1", "scenario2"}:
        return token
    raise ValueError(
        f"Unsupported LIMIT_TO_SCENARIO value '{raw}'. "
        "Use one of: none, scenario1, scenario2."
    )


def _selected_scenario_targets(limit_to_scenario: Optional[str]) -> Tuple[str, ...]:
    if limit_to_scenario is None:
        return ("scenario2", "scenario1")
    return (limit_to_scenario,)


def _validate_limit_to_scenario_dependency(
    limit_to_scenario: Optional[str],
    baseline_cfg: Dict[str, Dict[str, str]],
) -> None:
    if limit_to_scenario is None:
        return
    target = str(limit_to_scenario).strip().lower()
    if target not in {"scenario1", "scenario2"}:
        raise ValueError(f"Unexpected scenario limiter token: {limit_to_scenario}")
    other = "scenario1" if target == "scenario2" else "scenario2"
    source = str(baseline_cfg.get(target, {}).get("source", "")).strip().lower()
    if source == other:
        _log(
            f"⚠️ LIMIT_TO_SCENARIO={target} depends on BASELINE_CONFIG[{target!r}]['source']={other!r}. "
            "Continuing only because a cached baseline fit may already exist. "
            "If the cached fit is missing, runtime will fail when the baseline is requested."
        )


DEFAULT_LIMIT_TO_SCENARIO = _normalize_limit_to_scenario_token(LIMIT_TO_SCENARIO)


def _resolve_execution_regions(args: argparse.Namespace) -> List[str]:
    regions = _parse_csv_items(getattr(args, "execution_regions", None), lower=False)
    return regions if regions else [str(args.region)]


def _resolve_execution_pet_methods(args: argparse.Namespace) -> List[str]:
    raw = _parse_csv_items(getattr(args, "execution_pet_methods", None), lower=True)
    if raw:
        if "all" in raw:
            return ["thornthwaite", "hargreaves", "penman-monteith"]
        allowed = {"auto", "thornthwaite", "hargreaves", "penman-monteith"}
        bad = [v for v in raw if v not in allowed]
        if bad:
            raise ValueError(f"Unknown --execution-pet-methods entries: {bad}")
        return raw
    pet = str(args.pet_method).strip().lower()
    if pet == "all":
        return ["thornthwaite", "hargreaves", "penman-monteith"]
    return [pet]


def _print_file_major_dry_run_plan(
    *,
    args: argparse.Namespace,
    baseline_cfg: Dict[str, Dict[str, str]],
    cmip6_baseline_cfg: Dict[str, Dict[str, str]],
    nat_root: Path,
    all_root: Path,
    era5_file: Path,
    output_tag: Optional[str],
) -> None:
    regions = _resolve_execution_regions(args)
    pet_methods = _resolve_execution_pet_methods(args)
    nat_scens_mode = _normalize_rsdsbiasadjust_nat_scens_mode(str(args.rsdsbiasadjust_nat_scens))
    limit_to_scenario = _normalize_limit_to_scenario_token(getattr(args, "limit_to_scenario", None))
    selected_scenarios = set(_selected_scenario_targets(limit_to_scenario))
    run_scenario2 = (not bool(args.cmip6_only)) and ("scenario2" in selected_scenarios)
    run_scenario1 = (not bool(args.cmip6_only)) and ("scenario1" in selected_scenarios)
    run_era5 = (not bool(args.cmip6_only)) and (limit_to_scenario is None)
    months = _parse_months(args.months)

    _log("=== 754_add_SPEI_to_ensemble_outputs (DRY RUN) ===")
    _log("Mode: file-major-prototype (no file discovery, no ERA5 open, no writes)")
    _log(f"SCENARIO2 root:     {nat_root}")
    _log(f"SCENARIO1 root:     {all_root}")
    _log(f"ERA5 file:          {era5_file}")
    _log(f"Output tag:         {output_tag or '(none)'}")
    _log(f"Deriv layout/suffix:{_ACTIVE_DERIVATIVES_LAYOUT} / {_ACTIVE_DERIVATIVES_RUN_SUFFIX}")
    _log(f"Scale/Fit:          spei{args.scale} / {args.fit}")
    _log(
        "Segments layout:    "
        f"{args.segments_layout} "
        f"(output-format={args.output_format}, "
        f"chunk-run={args.chunk_run}, chunk-time={args.chunk_time}, "
        f"chunk-point={args.chunk_point}, zstd={args.compression_level}, "
        f"consolidate={bool(args.consolidate_metadata)})"
    )
    _log(f"Regions ({len(regions)}): {', '.join(regions)}")
    _log(f"PET methods ({len(pet_methods)}): {', '.join(pet_methods)}")
    _log(f"Months filter:      {months or 'all'}")
    _log(f"CMIP6-only mode:    {bool(args.cmip6_only)}")
    _log(f"LIMIT_TO_SCENARIO:  {limit_to_scenario or 'none'}")
    _log(
        "RSDS ERA5 bias adjust: "
        f"{bool(args.apply_rsds_to_era5_biasadjustment)} "
        f"(smooth={DEFAULT_RSDS_BIAS_SMOOTHING_YEARS}y, "
        f"edge-trend={DEFAULT_RSDS_BIAS_EDGE_TREND_YEARS}y, "
        f"edge-extend={DEFAULT_RSDS_BIAS_EDGE_EXTENSION_YEARS}y, "
        f"nat-mode={nat_scens_mode})"
    )
    _log(
        f"CMIP6 enabled:      {bool(args.include_cmip6)}"
        + (
            f" | root={args.cmip6_root} | experiments={args.cmip6_experiments}"
            if bool(args.include_cmip6)
            else ""
        )
    )
    _log(
        "Baseline config: "
        f"ERA5(src={baseline_cfg['era5']['source']},pool={baseline_cfg['era5']['pooling']}); "
        f"SCENARIO1(src={baseline_cfg['scenario1']['source']},pool={baseline_cfg['scenario1']['pooling']}); "
        f"SCENARIO2(src={baseline_cfg['scenario2']['source']},pool={baseline_cfg['scenario2']['pooling']}); "
        f"CMIP6-HIST(src={cmip6_baseline_cfg['cmip6_hist']['source']},pool={cmip6_baseline_cfg['cmip6_hist']['pooling']}); "
        f"CMIP6-HIST-NAT(src={cmip6_baseline_cfg['cmip6_hist_nat']['source']},pool={cmip6_baseline_cfg['cmip6_hist_nat']['pooling']}); "
        f"CMIP6-SSP245(src={cmip6_baseline_cfg['cmip6_ssp245']['source']},pool={cmip6_baseline_cfg['cmip6_ssp245']['pooling']})"
    )
    forcing_plan: List[str] = []
    if run_era5:
        forcing_plan.append("ERA5")
    if run_scenario2:
        forcing_plan.append(FORCING_SCENARIO2_LABEL)
    if run_scenario1:
        forcing_plan.append(FORCING_SCENARIO1_LABEL)
    if bool(args.include_cmip6):
        forcing_plan.extend(["CMIP6-HIST", "CMIP6-HIST-NAT", "CMIP6-SSP245"])
    _log(
        "Planned forcing order: "
        + (" -> ".join(forcing_plan) if forcing_plan else "(none)")
        + " (each file opened once, then region x PET combos)"
    )
    combos = len(regions) * len(pet_methods)
    _log(f"Planned per-file combo fan-out: {combos}")

    sample_rows: List[str] = []
    forcing_tokens = forcing_plan if forcing_plan else ["none"]
    for region in regions[:2]:
        for pet in pet_methods[:3]:
            sample_rows.append(f"forcing=<{ '|'.join(forcing_tokens) }>, region={region}, pet={pet}")
    if sample_rows:
        _log("Sample task rows:")
        for row in sample_rows:
            _log(f"  - {row}")

    _log("Dry run complete.")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute SPEI using a selectable baseline fit; write SPEI segments for SCENARIO1, SCENARIO2, and ERA5."
    )
    p.add_argument(
        "--gcmagicc-scenario2-root",
        dest="gcmagicc_scenario2_root",
        type=Path,
        default=DEFAULT_GCMAGICC_SCENARIO2_ROOT,
        help="SCENARIO2 GCMagicc ensemble NetCDF root.",
    )
    p.add_argument(
        "--gcmagicc-scenario1-root",
        dest="gcmagicc_scenario1_root",
        type=Path,
        default=DEFAULT_GCMAGICC_SCENARIO1_ROOT,
        help="SCENARIO1 GCMagicc ensemble NetCDF root.",
    )
    p.add_argument(
        "--scenario2-suffix",
        default=DEFAULT_GCMAGICC_SCENARIO2_SUFFIX,
        help="Suffix for SCENARIO2 filtering (e.g., -nat). Use empty/None to disable suffix filtering.",
    )
    p.add_argument(
        "--limit-to-scenario",
        type=_normalize_limit_to_scenario_token,
        default=DEFAULT_LIMIT_TO_SCENARIO,
        help=(
            "Optional non-CMIP6 scenario limiter: none (default), scenario1, or scenario2. "
            "When set, only the selected scenario forcing is processed."
        ),
    )
    p.add_argument("--era5-file", type=Path, default=DEFAULT_ERA5_FILE, help="ERA5 historical NetCDF file path.")
    p.add_argument(
        "--include-cmip6",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_INCLUDE_CMIP6,
        help="Also process one-member-per-source CMIP6 historical/hist-nat/ssp245 runs (default: enabled). Use --no-include-cmip6 to disable.",
    )
    p.add_argument(
        "--cmip6-only",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CMIP6_ONLY,
        help="Process only CMIP6 historical/hist-nat/ssp245 runs (skip ERA5/SCENARIO2/SCENARIO1 run processing).",
    )
    p.add_argument(
        "--cmip6-root",
        type=Path,
        default=DEFAULT_CMIP6_ROOT,
        help="CMIP6 NetCDF root (ETHFOG layout with DAT_<source>_<experiment>_<member>_*.nc).",
    )
    p.add_argument(
        "--cmip6-experiments",
        type=str,
        default=DEFAULT_CMIP6_EXPERIMENTS,
        help="Comma-separated CMIP6 experiments to include (supported: historical,hist-nat,ssp245).",
    )
    p.add_argument(
        "--cmip6-limit-models",
        type=int,
        default=None,
        help="Limit number of CMIP6 source_id models selected (debug).",
    )
    p.add_argument(
        "--world-input-root",
        type=Path,
        default=None,
        help="Optional 755 world SPEIx root used to regionalize stable overlay caches into a regional SPEIx tree.",
    )
    p.add_argument(
        "--world-input-tag",
        default=None,
        help="Optional tag under --world-input-root (default: use --output-tag, then latest if needed).",
    )
    p.add_argument(
        "--world-input-scenario",
        default=None,
        help="Scenario token represented by the world-input cache (for example ERA5, historical, hist-nat, ssp245).",
    )
    p.add_argument(
        "--world-output-root",
        type=Path,
        default=None,
        help="Explicit regional SPEIx root written by --world-input-root mode. Defaults to the sibling SPEIx root beside the 755 world root.",
    )

    p.add_argument("--region", default=DEFAULT_REGION, help=f"AR6 region key (default {DEFAULT_REGION}). Use 'GLOBAL' for no mask.")
    p.add_argument(
        "--scale",
        type=int,
        default=DEFAULT_SCALE_MONTHS,
        help=f"SPEI accumulation scale in months (default {DEFAULT_SCALE_MONTHS}).",
    )
    p.add_argument("--pet-method", default=DEFAULT_PET_METHOD, choices=["auto", "thornthwaite", "hargreaves", "penman-monteith", "all"], help="PET method (or 'all' to run thornthwaite+hargreaves+penman-monteith).")
    p.add_argument("--fit", default=DEFAULT_SPEI_FIT, choices=["zscore", "loglogistic"], help="Standardization fit type.")
    p.add_argument(
        "--execution-mode",
        default="legacy",
        choices=["legacy", "file-major-prototype"],
        help="Execution strategy: legacy looping or a prototype file-major scheduler.",
    )
    p.add_argument(
        "--execution-regions",
        default=None,
        help="Comma-separated regions for file-major-prototype (default: use --region).",
    )
    p.add_argument(
        "--execution-pet-methods",
        default=None,
        help="Comma-separated PET methods for file-major-prototype (default: derive from --pet-method).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print execution plan and exit. For file-major-prototype, avoids file discovery/data scans.",
    )

    # New explicit baseline config (requested)
    p.add_argument("--era5-baseline-source", default=BASELINE_CONFIG["era5"]["source"], choices=BASELINE_SOURCE_CHOICES)
    p.add_argument("--era5-baseline-pooling", default=BASELINE_CONFIG["era5"]["pooling"], choices=BASELINE_POOLING_CHOICES)

    p.add_argument(
        "--scenario1-baseline-source",
        dest="scenario1_baseline_source",
        default=BASELINE_CONFIG["scenario1"]["source"],
        choices=BASELINE_SOURCE_CHOICES,
    )
    p.add_argument(
        "--scenario1-baseline-pooling",
        dest="scenario1_baseline_pooling",
        default=BASELINE_CONFIG["scenario1"]["pooling"],
        choices=BASELINE_POOLING_CHOICES,
    )

    p.add_argument(
        "--scenario2-baseline-source",
        dest="scenario2_baseline_source",
        default=BASELINE_CONFIG["scenario2"]["source"],
        choices=BASELINE_SOURCE_CHOICES,
    )
    p.add_argument(
        "--scenario2-baseline-pooling",
        dest="scenario2_baseline_pooling",
        default=BASELINE_CONFIG["scenario2"]["pooling"],
        choices=BASELINE_POOLING_CHOICES,
    )

    p.add_argument(
        "--cmip6-historical-baseline-source",
        dest="cmip6_historical_baseline_source",
        default=CMIP6_BASELINE_CONFIG["cmip6_hist"]["source"],
        choices=CMIP6_BASELINE_SOURCE_CHOICES,
        help="Baseline source for CMIP6 historical runs (default historical).",
    )
    p.add_argument(
        "--cmip6-historical-baseline-pooling",
        dest="cmip6_historical_baseline_pooling",
        default=CMIP6_BASELINE_CONFIG["cmip6_hist"]["pooling"],
        choices=CMIP6_BASELINE_POOLING_CHOICES,
        help="Baseline pooling for CMIP6 historical runs (default per_member).",
    )
    p.add_argument(
        "--cmip6-histnat-baseline-source",
        dest="cmip6_histnat_baseline_source",
        default=CMIP6_BASELINE_CONFIG["cmip6_hist_nat"]["source"],
        choices=CMIP6_BASELINE_SOURCE_CHOICES,
        help="Baseline source for CMIP6 hist-nat runs (default historical).",
    )
    p.add_argument(
        "--cmip6-histnat-baseline-pooling",
        dest="cmip6_histnat_baseline_pooling",
        default=CMIP6_BASELINE_CONFIG["cmip6_hist_nat"]["pooling"],
        choices=CMIP6_BASELINE_POOLING_CHOICES,
        help="Baseline pooling for CMIP6 hist-nat runs (default per_member).",
    )
    p.add_argument(
        "--cmip6-ssp245-baseline-source",
        dest="cmip6_ssp245_baseline_source",
        default=CMIP6_BASELINE_CONFIG["cmip6_ssp245"]["source"],
        choices=CMIP6_BASELINE_SOURCE_CHOICES,
        help="Baseline source for CMIP6 ssp245 runs (default historical).",
    )
    p.add_argument(
        "--cmip6-ssp245-baseline-pooling",
        dest="cmip6_ssp245_baseline_pooling",
        default=CMIP6_BASELINE_CONFIG["cmip6_ssp245"]["pooling"],
        choices=CMIP6_BASELINE_POOLING_CHOICES,
        help="Baseline pooling for CMIP6 ssp245 runs (default per_member).",
    )

    # Backward-compatible / deprecated knobs (only used if explicitly set)
    p.add_argument(
        "--baseline-strategy",
        default=None,
        choices=["pooled", "memberwise", "era5"],
        help="DEPRECATED. Use --*-baseline-source/--*-baseline-pooling instead.",
    )
    p.add_argument(
        "--baseline-source",
        default=None,
        choices=["scenario2", "scenario1", "era5"],
        help="DEPRECATED. Use --*-baseline-source instead.",
    )
    p.add_argument(
        "--baseline-start-year",
        type=int,
        default=DEFAULT_BASELINE_START_YEAR,
        help="First year for baseline fitting (default: 1975).",
    )
    p.add_argument(
        "--baseline-end-year",
        type=int,
        default=DEFAULT_BASELINE_END_YEAR,
        help="Last year for baseline fitting (default: 2000).",
    )

    p.add_argument("--out-start-year", type=int, default=DEFAULT_OUT_START_YEAR, help="First output year for saved segments (default: all years).")
    p.add_argument("--out-end-year", type=int, default=DEFAULT_OUT_END_YEAR, help="Last output year for saved segments (default: all years).")
    p.add_argument("--months", type=str, default="all", help="Comma-separated months (e.g. '6,7,8') or 'all'.")
    p.add_argument("--group-pixels", type=int, default=DEFAULT_GROUP_PIXELS, choices=list(_ALLOWED_GROUP_PIXELS), help="Center-mean each point with its surrounding pixels before WB/SPEI (1=no grouping, 9=3x3, 25=5x5).")
    p.add_argument(
        "--apply-rsds-to-era5-biasadjustment",
        action=argparse.BooleanOptionalAction,
        default=APPLY_RSDS_TO_ERA5_BIASADJUSTMENT,
        help=(
            "Apply monthly/yearly rsds bias adjustment to GCMAGICC runs so their 21-year smoothed "
            "region-point histories align to ERA5 during complete ERA5 years (default: enabled). "
            "Use --no-apply-rsds-to-era5-biasadjustment to disable."
        ),
    )
    p.add_argument(
        "--rsdsbiasadjust-nat-scens",
        default=RSDSBIASADJUST_NAT_SCENS,
        choices=list(RSDSBIASADJUST_NAT_SCENS_CHOICES),
        help=(
            "How to apply RSDS ERA5 bias adjustment to scenarios ending with "
            f"'{RSDSBIASADJUST_NAT_SCENARIO_SUFFIX}': "
            "'excempt' (skip NAT adjustment), "
            "'full' (apply full time-varying adjustment), "
            "'early' (use first ERA5-year month-specific offsets for all NAT years). "
            f"Default: {RSDSBIASADJUST_NAT_SCENS}."
        ),
    )
    p.add_argument(
        "--excempt-nat-scens-from-rsdsbiasadjust",
        "--exempt-nat-scens-from-rsdsbiasadjust",
        dest="legacy_excempt_nat_scens_from_rsdsbiasadjust",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=argparse.SUPPRESS,
    )

    p.add_argument("--limit-runs", type=int, default=None, help="Limit number of ensemble files to process (dev/debug).")
    p.add_argument(
        "--file-order-mode",
        default="sorted",
        choices=["sorted", "rotate"],
        help="Input NetCDF ordering strategy (default: sorted). Use 'rotate' with --file-order-offset to de-sync parallel workers.",
    )
    p.add_argument(
        "--file-order-offset",
        type=int,
        default=0,
        help="Cyclic file-order offset used when --file-order-mode=rotate (default: 0).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing baseline/segments.")
    p.add_argument(
        "--on-existing",
        default=DEFAULT_ON_EXISTING,
        choices=["prompt", "overwrite", "archive", "skip", "throwerror"],
        help="What to do if SPEI segments already exist (default: prompt).",
    )
    p.add_argument(
        "--segments-layout",
        default=DEFAULT_SEGMENTS_LAYOUT,
        choices=list(SEGMENTS_LAYOUT_CHOICES),
        help=(
            "Segment output layout policy: "
            "'per_run' keeps legacy groups under runs/<RUN>/...; "
            "'run_stacked' writes stacked groups with run dimension; "
            "'dual' writes both."
        ),
    )
    p.add_argument(
        "--output-format",
        default=DEFAULT_OUTPUT_FORMAT,
        choices=list(OUTPUT_FORMAT_CHOICES),
        help=(
            "Format for run-stacked SPEIx outputs. "
            "'zarr' keeps the current stacked Zarr layout; "
            "'netcdf' writes one consolidated NetCDF per forcing x region x PET x year-span."
        ),
    )
    p.add_argument(
        "--chunk-run",
        type=int,
        default=DEFAULT_STACKED_CHUNK_RUN,
        help=f"Run chunk size for run-stacked output (default: {DEFAULT_STACKED_CHUNK_RUN}).",
    )
    p.add_argument(
        "--chunk-time",
        type=int,
        default=DEFAULT_STACKED_CHUNK_TIME,
        help=f"Time chunk size for run-stacked output (default: {DEFAULT_STACKED_CHUNK_TIME}).",
    )
    p.add_argument(
        "--chunk-point",
        type=str,
        default=DEFAULT_STACKED_CHUNK_POINT,
        help=f"Point chunk size for run-stacked output (default: {DEFAULT_STACKED_CHUNK_POINT}).",
    )
    p.add_argument(
        "--compression-level",
        type=int,
        default=DEFAULT_STACKED_COMPRESSION_LEVEL,
        help=f"Zstd compression level for run-stacked numeric arrays (default: {DEFAULT_STACKED_COMPRESSION_LEVEL}).",
    )
    p.add_argument(
        "--consolidate-metadata",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_STACKED_CONSOLIDATE_METADATA,
        help=(
            "Consolidate Zarr metadata for run-stacked outputs after each forcing pass "
            "(default: enabled). Use --no-consolidate-metadata to disable."
        ),
    )
    p.add_argument("--pivot-year", type=int, default=DEFAULT_PIVOT_YEAR, help="Pivot year for greyscale (historical) vs purple (future) coloring in summary PDFs (default 2014).")
    p.add_argument("--no-summary-pdf", action="store_true", help="Disable per-run 12-panel summary PDF generation.")
    p.add_argument("--output-tag", type=str, default=None, help="Optional tag to nest outputs under data_derivatives/SPEIx/<output-tag> for isolation.")
    p.add_argument(
        "--derivatives-layout",
        default=DEFAULT_DERIVATIVES_LAYOUT,
        choices=list(DERIVATIVES_LAYOUT_CHOICES),
        help=(
            "Derivative root layout policy. "
            f"Default: {DEFAULT_DERIVATIVES_LAYOUT}."
        ),
    )
    p.add_argument(
        "--derivatives-run-suffix",
        default=_ACTIVE_DERIVATIVES_RUN_SUFFIX,
        help=f"Suffix used for sibling derivative trees (default: {_ACTIVE_DERIVATIVES_RUN_SUFFIX}).",
    )
    p.add_argument(
        "--storage-access",
        default=get_storage_access_default(),
        choices=list(STORAGE_ACCESS_CHOICES),
        help=(
            "NetCDF read mode: 'mount' uses filesystem paths; 's3_direct' converts eligible "
            "input paths to s3:// and opens directly with xarray/fsspec; 'rclone_cache' "
            "copies mounted input files into a local cache before opening them."
        ),
    )
    p.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_RCLONE_CACHE_ROOT,
        help=f"Local cache root used when --storage-access=rclone_cache (default: {DEFAULT_RCLONE_CACHE_ROOT}).",
    )
    p.add_argument(
        "--s3-env-file",
        type=Path,
        default=DEFAULT_S3_ENV_FILE,
        help=(
            "Optional env file used in s3_direct mode. "
            "Only AWS_/GCMAGICC_S3_ keys (+ optional bucket) are imported."
        ),
    )
    p.add_argument(
        "--s3-stage-dir",
        type=Path,
        default=DEFAULT_S3_STAGE_DIR,
        help="Shared local staging cache used by s3_direct fallback.",
    )
    p.add_argument(
        "--s3-preflight",
        dest="s3_preflight",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_S3_PREFLIGHT,
        help="Validate selected source objects exist on S3 before processing (default: enabled).",
    )
    p.add_argument(
        "--auto-consolidate",
        dest="auto_consolidate",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUTO_CONSOLIDATE,
        help="Run scoped 2018 autoconsolidate for written derivative roots (default: enabled).",
    )
    p.add_argument(
        "--auto-consolidate-config",
        type=Path,
        default=None,
        help="Path to 2018 consolidation config JSON (default: scripts/2018_consolidate_era5spliced_s3.example.json).",
    )
    p.add_argument(
        "--auto-consolidate-cleanup-local",
        dest="auto_consolidate_cleanup_local",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUTO_CONSOLIDATE_CLEANUP_LOCAL,
        help="Allow local cleanup after verified upload in autoconsolidate (default: enabled).",
    )
    return p.parse_args(argv)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    global _S3_FS
    global _ACTIVE_S3_PREFLIGHT
    global _ACTIVE_S3_STAGE_DIR
    global _ACTIVE_STORAGE_ACCESS
    global _ACTIVE_RCLONE_CACHE_ROOT
    global _ACTIVE_S3_LOCAL_FALLBACK_PATHS
    global _ACTIVE_S3_SELECTED_URI_BY_LOCAL
    global _ACTIVE_S3_PREFLIGHT_COUNTS
    args = _parse_args(argv)
    args.region = _normalize_region_name(args.region)
    _set_derivatives_runtime_config(args.derivatives_layout, args.derivatives_run_suffix)
    _ACTIVE_STORAGE_ACCESS = normalize_storage_access(args.storage_access)
    _ACTIVE_RCLONE_CACHE_ROOT = Path(args.cache_root).expanduser().resolve(strict=False)
    if _ACTIVE_STORAGE_ACCESS == STORAGE_ACCESS_RCLONE_CACHE:
        _ACTIVE_RCLONE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    _ACTIVE_S3_PREFLIGHT = bool(args.s3_preflight)
    _ACTIVE_S3_STAGE_DIR = Path(args.s3_stage_dir).expanduser().resolve(strict=False)
    _ACTIVE_S3_LOCAL_FALLBACK_PATHS = set()
    _ACTIVE_S3_SELECTED_URI_BY_LOCAL = {}
    _ACTIVE_S3_PREFLIGHT_COUNTS = {}
    _S3_FS = None
    if args.world_input_root is not None:
        _run_world_input_regionalizer(args)
        return
    if _ACTIVE_STORAGE_ACCESS == STORAGE_ACCESS_S3_DIRECT:
        _load_s3_env_file(args.s3_env_file)
        _ACTIVE_S3_STAGE_DIR.mkdir(parents=True, exist_ok=True)
        _log(f"S3 staging cache:  {_ACTIVE_S3_STAGE_DIR}")
        _log(f"S3 preflight:      {_ACTIVE_S3_PREFLIGHT}")
    if args.legacy_excempt_nat_scens_from_rsdsbiasadjust is not None:
        args.rsdsbiasadjust_nat_scens = (
            "excempt" if bool(args.legacy_excempt_nat_scens_from_rsdsbiasadjust) else "full"
        )
        _log(
            "⚠️  Deprecated RSDS NAT bool flag was provided; "
            f"mapping to --rsdsbiasadjust-nat-scens={args.rsdsbiasadjust_nat_scens}."
        )
    nat_scens_mode = _normalize_rsdsbiasadjust_nat_scens_mode(str(args.rsdsbiasadjust_nat_scens))
    cmip6_only = bool(args.cmip6_only)
    if cmip6_only and not bool(args.include_cmip6):
        raise ValueError("--cmip6-only requires --include-cmip6.")
    limit_to_scenario = _normalize_limit_to_scenario_token(getattr(args, "limit_to_scenario", None))
    args.limit_to_scenario = limit_to_scenario

    nat_root: Path = args.gcmagicc_scenario2_root
    all_root: Path = args.gcmagicc_scenario1_root
    era5_file: Path = args.era5_file
    cmip6_root: Path = args.cmip6_root
    output_tag: Optional[str] = args.output_tag
    segments_layout = _normalize_segments_layout_token(str(args.segments_layout))
    output_format = _normalize_output_format_token(str(args.output_format))
    chunk_run = int(args.chunk_run)
    chunk_time = int(args.chunk_time)
    chunk_point = _parse_chunk_point_arg(str(args.chunk_point))
    compression_level = int(args.compression_level)
    consolidate_metadata = bool(args.consolidate_metadata)
    write_per_run_preview = segments_layout in {SEGMENTS_LAYOUT_PER_RUN, SEGMENTS_LAYOUT_DUAL}

    if chunk_run <= 0:
        raise ValueError(f"--chunk-run must be >= 1, got {chunk_run}")
    if chunk_time <= 0:
        raise ValueError(f"--chunk-time must be >= 1, got {chunk_time}")
    if compression_level < 0:
        raise ValueError(f"--compression-level must be >= 0, got {compression_level}")

    # Build baseline config from CLI (or script-level BASELINE_CONFIG defaults)
    baseline_cfg: Dict[str, Dict[str, str]] = {
        "era5": {
            "source": _normalize_baseline_source_token(str(args.era5_baseline_source)),
            "pooling": str(args.era5_baseline_pooling),
        },
        "scenario1": {
            "source": _normalize_baseline_source_token(str(args.scenario1_baseline_source)),
            "pooling": str(args.scenario1_baseline_pooling),
        },
        "scenario2": {
            "source": _normalize_baseline_source_token(str(args.scenario2_baseline_source)),
            "pooling": str(args.scenario2_baseline_pooling),
        },
    }
    cmip6_baseline_cfg: Dict[str, Dict[str, str]] = {
        "cmip6_hist": {
            "source": _normalize_cmip6_baseline_source_token(str(args.cmip6_historical_baseline_source)),
            "pooling": str(args.cmip6_historical_baseline_pooling).strip().lower(),
        },
        "cmip6_hist_nat": {
            "source": _normalize_cmip6_baseline_source_token(str(args.cmip6_histnat_baseline_source)),
            "pooling": str(args.cmip6_histnat_baseline_pooling).strip().lower(),
        },
        "cmip6_ssp245": {
            "source": _normalize_cmip6_baseline_source_token(str(args.cmip6_ssp245_baseline_source)),
            "pooling": str(args.cmip6_ssp245_baseline_pooling).strip().lower(),
        },
    }

    # If deprecated flags were explicitly used, override baseline_cfg accordingly.
    if args.baseline_strategy is not None or args.baseline_source is not None:
        bs = args.baseline_strategy
        bsrc = args.baseline_source
        _log(
            "⚠️  Deprecated baseline flags (--baseline-strategy/--baseline-source) were provided; "
            "overriding the new baseline config."
        )
        if bs == "era5" or bsrc == "era5":
            baseline_cfg = {
                "era5": {"source": "era5", "pooling": "pooled"},
                "scenario1": {"source": "era5", "pooling": "pooled"},
                "scenario2": {"source": "era5", "pooling": "pooled"},
            }
        elif bs == "memberwise":
            baseline_cfg = {
                "era5": {"source": "era5", "pooling": "pooled"},
                "scenario1": {"source": "scenario1", "pooling": "per_member"},
                "scenario2": {"source": "scenario1", "pooling": "per_member"},
            }
        else:
            # pooled legacy
            legacy_src = _normalize_baseline_source_token((bsrc or "scenario1").lower())
            baseline_cfg = {
                "era5": {"source": legacy_src, "pooling": "pooled"},
                "scenario1": {"source": legacy_src, "pooling": "pooled"},
                "scenario2": {"source": legacy_src, "pooling": "pooled"},
            }

    if args.baseline_start_year is not None and args.baseline_end_year is not None:
        if args.baseline_start_year > args.baseline_end_year:
            raise ValueError("baseline-start-year must be <= baseline-end-year")

    _validate_limit_to_scenario_dependency(limit_to_scenario, baseline_cfg)
    selected_scenarios = set(_selected_scenario_targets(limit_to_scenario))
    run_scenario2 = (not cmip6_only) and ("scenario2" in selected_scenarios)
    run_scenario1 = (not cmip6_only) and ("scenario1" in selected_scenarios)
    run_era5 = (not cmip6_only) and (limit_to_scenario is None)

    if run_scenario2 and (not nat_root.exists()):
        raise FileNotFoundError(f"SCENARIO2 root not found: {nat_root}")
    if run_scenario1 and (not all_root.exists()):
        raise FileNotFoundError(f"SCENARIO1 root not found: {all_root}")
    if not era5_file.exists():
        raise FileNotFoundError(f"ERA5 file not found: {era5_file}")
    if args.include_cmip6 and (not cmip6_root.exists()):
        raise FileNotFoundError(f"CMIP6 root not found: {cmip6_root}")
    if args.cmip6_limit_models is not None and int(args.cmip6_limit_models) <= 0:
        raise ValueError(f"--cmip6-limit-models must be >=1, got {args.cmip6_limit_models}")

    if args.dry_run and args.execution_mode == "file-major-prototype":
        _print_file_major_dry_run_plan(
            args=args,
            baseline_cfg=baseline_cfg,
            cmip6_baseline_cfg=cmip6_baseline_cfg,
            nat_root=nat_root,
            all_root=all_root,
            era5_file=era5_file,
            output_tag=output_tag,
        )
        return

    if args.dry_run:
        _log(
            "Dry run requested in legacy mode. "
            "No execution started. (Use --execution-mode file-major-prototype for detailed task plan.)"
        )
        return

    months = _parse_months(args.months)

    era5_y0, era5_y1 = _infer_time_year_range(era5_file)
    baseline_start_year = args.baseline_start_year
    baseline_end_year = args.baseline_end_year

    _log("=== 754_add_SPEI_to_ensemble_outputs (rev6) ===")
    _log(f"SCENARIO2 root:     {nat_root}")
    _log(f"SCENARIO1 root:     {all_root}")
    _log(f"ERA5 file: {era5_file}")
    _log(f"Deriv layout/suffix:{_ACTIVE_DERIVATIVES_LAYOUT} / {_ACTIVE_DERIVATIVES_RUN_SUFFIX}")
    _log(f"Storage access:     {_ACTIVE_STORAGE_ACCESS}")
    if _ACTIVE_STORAGE_ACCESS == STORAGE_ACCESS_RCLONE_CACHE:
        _log(f"Rclone cache root:  {_ACTIVE_RCLONE_CACHE_ROOT}")
    _log(f"CMIP6-only mode:    {cmip6_only}")
    _log(f"LIMIT_TO_SCENARIO:  {limit_to_scenario or 'none'}")
    _log(
        "CMIP6: "
        + (
            f"enabled root={cmip6_root} experiments={args.cmip6_experiments}"
            if args.include_cmip6
            else "disabled"
        )
    )
    group_side = int(round(math.sqrt(int(args.group_pixels))))
    _log(f"Region: {args.region} | Scale: {args.scale} | PET: {args.pet_method} | Fit: {args.fit}")
    _log(f"Execution mode: {args.execution_mode}")
    _log(
        "Segments layout: "
        f"{segments_layout} "
        f"(output-format={output_format}, "
        f"chunk-run={chunk_run}, chunk-time={chunk_time}, "
        f"chunk-point={args.chunk_point}, zstd={compression_level}, "
        f"consolidate={consolidate_metadata})"
    )
    if output_format == OUTPUT_FORMAT_NETCDF and write_per_run_preview:
        _log("NetCDF output format applies to run-stacked bundles; per-run segments remain Zarr groups.")
    _log(
        "RSDS ERA5 bias adjustment: "
        f"{bool(args.apply_rsds_to_era5_biasadjustment)} "
        f"(default={APPLY_RSDS_TO_ERA5_BIASADJUSTMENT}, smooth={DEFAULT_RSDS_BIAS_SMOOTHING_YEARS}y, "
        f"edge-trend={DEFAULT_RSDS_BIAS_EDGE_TREND_YEARS}y, "
        f"edge-extend={DEFAULT_RSDS_BIAS_EDGE_EXTENSION_YEARS}y, "
        f"nat-mode={nat_scens_mode})"
    )
    _log(f"Point grouping: {args.group_pixels} pixel(s) -> {group_side}x{group_side} centered mean before WB/SPEI/plots")
    _log(f"Output window: {args.out_start_year or '...'}–{args.out_end_year or '...'} | Months (saved): {months or 'all'}")
    _log(
        "Baseline config: "
        f"ERA5(src={baseline_cfg['era5']['source']},pool={baseline_cfg['era5']['pooling']}); "
        f"SCENARIO1(src={baseline_cfg['scenario1']['source']},pool={baseline_cfg['scenario1']['pooling']}); "
        f"SCENARIO2(src={baseline_cfg['scenario2']['source']},pool={baseline_cfg['scenario2']['pooling']}); "
        f"CMIP6-HIST(src={cmip6_baseline_cfg['cmip6_hist']['source']},pool={cmip6_baseline_cfg['cmip6_hist']['pooling']}); "
        f"CMIP6-HIST-NAT(src={cmip6_baseline_cfg['cmip6_hist_nat']['source']},pool={cmip6_baseline_cfg['cmip6_hist_nat']['pooling']}); "
        f"CMIP6-SSP245(src={cmip6_baseline_cfg['cmip6_ssp245']['source']},pool={cmip6_baseline_cfg['cmip6_ssp245']['pooling']}); "
        f"Years: {baseline_start_year or '...'}–{baseline_end_year or '...'}"
    )

    scenario2_suffix = args.scenario2_suffix

    nat_files: List[Path] = []
    all_files: List[Path] = []
    if not cmip6_only:
        if run_scenario2:
            nat_candidates = _discover_nc_files(nat_root)
            nat_files, nat_skipped = _filter_nc_files_by_forcing(
                nat_candidates, forcing=FORCING_SCENARIO2_LABEL, suffix_override=scenario2_suffix
            )
            if nat_skipped:
                _log(
                    f"Filtered out {len(nat_skipped)} files under SCENARIO2 root "
                    f"(expect scenario '*{scenario2_suffix}' in filename)."
                )
            _log(f"Using {len(nat_files)} SCENARIO2 files out of {len(nat_candidates)} discovered.")
        else:
            _log("Skipping SCENARIO2 file discovery (excluded by LIMIT_TO_SCENARIO).")

        if run_scenario1:
            all_candidates = _discover_nc_files(all_root)
            all_nat_suffix = "-nat"
            all_files, all_skipped = _filter_nc_files_by_forcing(
                all_candidates, forcing=FORCING_SCENARIO1_LABEL, suffix_override=all_nat_suffix
            )
            if all_skipped:
                _log(
                    "Filtered out "
                    f"{len(all_skipped)} SCENARIO2-tagged or unparseable files under SCENARIO1 root "
                    f"(expect scenario without '{all_nat_suffix}')."
                )
            _log(f"Using {len(all_files)} SCENARIO1 files out of {len(all_candidates)} discovered.")
        else:
            _log("Skipping SCENARIO1 file discovery (excluded by LIMIT_TO_SCENARIO).")
    else:
        _log("CMIP6-only mode: skipping SCENARIO2/SCENARIO1 file discovery.")

    cmip6_hist_files: List[Path] = []
    cmip6_hist_nat_files: List[Path] = []
    cmip6_ssp245_files: List[Path] = []
    cmip6_manifest: Optional[Dict] = None
    if args.include_cmip6:
        cmip6_experiments = _parse_cmip6_experiments(args.cmip6_experiments)
        (
            cmip6_hist_files,
            cmip6_hist_nat_files,
            cmip6_ssp245_files,
            cmip6_manifest,
        ) = _select_cmip6_one_member_per_source(
            cmip6_root,
            experiments=cmip6_experiments,
            limit_models=args.cmip6_limit_models,
        )
        _log(
            "CMIP6 selected runs (one member per source_id): "
            f"historical={len(cmip6_hist_files)}, "
            f"hist-nat={len(cmip6_hist_nat_files)}, "
            f"ssp245={len(cmip6_ssp245_files)}"
        )
        if cmip6_manifest is not None:
            _log(
                "CMIP6 available totals: "
                + ", ".join(
                    f"{k}={v}" for k, v in cmip6_manifest.get("totals_available_by_experiment", {}).items()
                )
                + f" | selected_sources={cmip6_manifest.get('selected_source_count', 0)}"
            )

    _log(
        "Discovered forcing counts: "
        f"SCENARIO2={len(nat_files)}, "
        f"SCENARIO1={len(all_files)}, "
        f"CMIP6-HIST={len(cmip6_hist_files)}, "
        f"CMIP6-HIST-NAT={len(cmip6_hist_nat_files)}, "
        f"CMIP6-SSP245={len(cmip6_ssp245_files)}"
    )
    _log(
        "Baseline source mapping summary: "
        f"SCENARIO2<-{baseline_cfg['scenario2']['source']}({baseline_cfg['scenario2']['pooling']}), "
        f"SCENARIO1<-{baseline_cfg['scenario1']['source']}({baseline_cfg['scenario1']['pooling']}), "
        f"ERA5<-{baseline_cfg['era5']['source']}({baseline_cfg['era5']['pooling']})"
    )

    if run_scenario2 and (not nat_files):
        raise RuntimeError(
            f"No SCENARIO2 NetCDF files found under: {nat_root} "
            f"(looking for scenario ending with '{scenario2_suffix}' in filename)"
        )
    if run_scenario1 and (not all_files):
        _log(f"⚠️  No SCENARIO1 NetCDF files found under: {all_root} (continuing anyway)")

    # Reorder lists before any baseline fitting/processing to reduce synchronized
    # file-open storms when many wrapper jobs run in parallel.
    nat_files = _reorder_file_list(
        nat_files,
        mode=args.file_order_mode,
        offset=int(args.file_order_offset),
    )
    all_files = _reorder_file_list(
        all_files,
        mode=args.file_order_mode,
        offset=int(args.file_order_offset),
    )
    if args.include_cmip6:
        cmip6_hist_files = _reorder_file_list(
            cmip6_hist_files,
            mode=args.file_order_mode,
            offset=int(args.file_order_offset),
        )
        cmip6_hist_nat_files = _reorder_file_list(
            cmip6_hist_nat_files,
            mode=args.file_order_mode,
            offset=int(args.file_order_offset),
        )
        cmip6_ssp245_files = _reorder_file_list(
            cmip6_ssp245_files,
            mode=args.file_order_mode,
            offset=int(args.file_order_offset),
        )
    _log(
        "Input file order: "
        f"mode={args.file_order_mode}, offset={int(args.file_order_offset)}"
    )
    if nat_files:
        _log(f"First SCENARIO2 file after ordering: {nat_files[0].name}")
    if all_files:
        _log(f"First SCENARIO1 file after ordering: {all_files[0].name}")
    if args.include_cmip6 and cmip6_hist_files:
        _log(f"First CMIP6 historical file after ordering: {cmip6_hist_files[0].name}")
    if args.include_cmip6 and cmip6_hist_nat_files:
        _log(f"First CMIP6 hist-nat file after ordering: {cmip6_hist_nat_files[0].name}")
    if args.include_cmip6 and cmip6_ssp245_files:
        _log(f"First CMIP6 ssp245 file after ordering: {cmip6_ssp245_files[0].name}")

    if _ACTIVE_STORAGE_ACCESS == STORAGE_ACCESS_S3_DIRECT and _ACTIVE_S3_PREFLIGHT:
        preflight_paths: List[Path] = []
        if run_scenario2:
            preflight_paths.extend(nat_files)
        if run_scenario1:
            preflight_paths.extend(all_files)
        if preflight_paths:
            _ACTIVE_S3_LOCAL_FALLBACK_PATHS = _preflight_s3_objects(
                preflight_paths, label="SCENARIO originals"
            )
        else:
            _log("S3 preflight skipped: no SCENARIO source files selected.")

    # Build a single, consistent template point-set from ERA5 (stable across strategies and PET methods).
    # Use a minimal PET method (thornthwaite) for template construction to avoid point-set drifting across methods.
    template, _, _ = _build_template_from_file(
        era5_file,
        region=args.region,
        pet_method="thornthwaite",
        label="ERA5 (template)",
    )
    point_groups = _build_point_groups(template, int(args.group_pixels))

    on_existing = "overwrite" if args.force else args.on_existing
    write_summary_pdf = (not bool(args.no_summary_pdf))
    if not write_summary_pdf:
        _log("Summary PDF generation disabled via --no-summary-pdf.")

    store_nat = _segments_store_dir(nat_root, output_tag=output_tag, region=args.region)
    store_all = _segments_store_dir(all_root, output_tag=output_tag, region=args.region)
    store_era5 = _segments_store_dir(era5_file.parent, output_tag=output_tag, region=args.region)
    store_cmip6 = _segments_store_dir(cmip6_root, output_tag=output_tag, region=args.region) if args.include_cmip6 else None

    if on_existing == "prompt":
        labels: List[str] = []
        if not cmip6_only:
            if run_scenario2 and _find_existing_segments(
                store_nat, run_ids=[p.stem for p in nat_files], region=args.region, scale=args.scale
            ):
                labels.append(FORCING_SCENARIO2_LABEL)
            if run_scenario1 and _find_existing_segments(
                store_all, run_ids=[p.stem for p in all_files], region=args.region, scale=args.scale
            ):
                labels.append(FORCING_SCENARIO1_LABEL)
            if run_era5 and _find_existing_segments(
                store_era5, run_ids=["ERA5"], region=args.region, scale=args.scale
            ):
                labels.append("ERA5")
        if args.include_cmip6 and store_cmip6 is not None:
            cmip6_run_ids = (
                [p.stem for p in cmip6_hist_files]
                + [p.stem for p in cmip6_hist_nat_files]
                + [p.stem for p in cmip6_ssp245_files]
            )
            if _find_existing_segments(store_cmip6, run_ids=cmip6_run_ids, region=args.region, scale=args.scale):
                labels.append("CMIP6")
        if segments_layout in {SEGMENTS_LAYOUT_RUN_STACKED, SEGMENTS_LAYOUT_DUAL}:
            if not cmip6_only:
                if run_scenario2 and _find_existing_stacked_outputs(
                    store_nat,
                    run_ids=[p.stem for p in nat_files],
                    region=args.region,
                    scale=args.scale,
                    output_format=output_format,
                ):
                    labels.append(f"{FORCING_SCENARIO2_LABEL}-STACKED")
                if run_scenario1 and _find_existing_stacked_outputs(
                    store_all,
                    run_ids=[p.stem for p in all_files],
                    region=args.region,
                    scale=args.scale,
                    output_format=output_format,
                ):
                    labels.append(f"{FORCING_SCENARIO1_LABEL}-STACKED")
                if run_era5 and _find_existing_stacked_outputs(
                    store_era5,
                    run_ids=["ERA5"],
                    region=args.region,
                    scale=args.scale,
                    output_format=output_format,
                ):
                    labels.append("ERA5-STACKED")
            if args.include_cmip6 and store_cmip6 is not None:
                cmip6_run_ids = (
                    [p.stem for p in cmip6_hist_files]
                    + [p.stem for p in cmip6_hist_nat_files]
                    + [p.stem for p in cmip6_ssp245_files]
                )
                if _find_existing_stacked_outputs(
                    store_cmip6,
                    run_ids=cmip6_run_ids,
                    region=args.region,
                    scale=args.scale,
                    output_format=output_format,
                ):
                    labels.append("CMIP6-STACKED")
        if labels:
            on_existing = _prompt_on_existing(labels)
            if on_existing == "quit":
                _log("Aborted by user.")
                return
        else:
            on_existing = "skip"

    archive_tag = _timestamp_tag() if on_existing == "archive" else None
    archive_root_nat = (store_nat / "archive" / archive_tag) if archive_tag else None
    archive_root_all = (store_all / "archive" / archive_tag) if archive_tag else None
    archive_root_era5 = (store_era5 / "archive" / archive_tag) if archive_tag else None
    archive_root_cmip6 = (store_cmip6 / "archive" / archive_tag) if (archive_tag and store_cmip6 is not None) else None
    baseline_archive_root_nat = (_deriv_root(nat_root, output_tag=output_tag) / "archive" / archive_tag) if (archive_tag and (not cmip6_only)) else None
    baseline_archive_root_all = (_deriv_root(all_root, output_tag=output_tag) / "archive" / archive_tag) if (archive_tag and (not cmip6_only)) else None
    baseline_archive_root_era5 = (_deriv_root(era5_file.parent, output_tag=output_tag) / "archive" / archive_tag) if (archive_tag and (not cmip6_only)) else None
    baseline_archive_root_cmip6 = (_deriv_root(cmip6_root, output_tag=output_tag) / "archive" / archive_tag) if (archive_tag and args.include_cmip6) else None

    # If archiving, move entire SPEIx derivative trees to data_derivatives_archive/SPEIx/<tag>
    if archive_tag:
        roots_to_archive: List[Tuple[str, Path]] = []
        if not cmip6_only:
            if run_scenario2:
                roots_to_archive.append((FORCING_SCENARIO2_LABEL, nat_root))
            if run_scenario1:
                roots_to_archive.append((FORCING_SCENARIO1_LABEL, all_root))
            if run_era5:
                roots_to_archive.append(("ERA5", era5_file.parent))
        if args.include_cmip6:
            roots_to_archive.append(("CMIP6", cmip6_root))
        for root_label, root_path in roots_to_archive:
            moved = _archive_speix_tree(root_path, tag=archive_tag, output_tag=output_tag)
            if moved:
                _log(f"Archived existing SPEIx tree for {root_label} -> {moved}")

    if args.execution_mode == "file-major-prototype":
        _run_file_major_prototype(
            args=args,
            cmip6_only=cmip6_only,
            baseline_cfg=baseline_cfg,
            cmip6_baseline_cfg=cmip6_baseline_cfg,
            nat_root=nat_root,
            all_root=all_root,
            era5_file=era5_file,
            output_tag=output_tag,
            nat_files=nat_files,
            all_files=all_files,
            include_cmip6=args.include_cmip6,
            cmip6_root=cmip6_root if args.include_cmip6 else None,
            cmip6_hist_files=cmip6_hist_files,
            cmip6_hist_nat_files=cmip6_hist_nat_files,
            cmip6_ssp245_files=cmip6_ssp245_files,
            cmip6_manifest=cmip6_manifest,
            months=months,
            baseline_start_year=baseline_start_year,
            baseline_end_year=baseline_end_year,
            on_existing=on_existing,
            archive_root_nat=archive_root_nat,
            archive_root_all=archive_root_all,
            archive_root_era5=archive_root_era5,
            archive_root_cmip6=archive_root_cmip6,
            baseline_archive_root_nat=baseline_archive_root_nat,
            baseline_archive_root_all=baseline_archive_root_all,
            baseline_archive_root_era5=baseline_archive_root_era5,
            baseline_archive_root_cmip6=baseline_archive_root_cmip6,
        )
        _log("\nDone.")
        return

    rsds_bias_adjustment_legacy: Optional[RsdsEra5BiasAdjustment] = None
    if bool(args.apply_rsds_to_era5_biasadjustment):
        gcm_for_bias = list(nat_files) + list(all_files)
        if _rsdsbiasadjust_should_exclude_nat_from_bias_fit(nat_scens_mode):
            pre = len(gcm_for_bias)
            gcm_for_bias = [
                p for p in gcm_for_bias
                if (not _scenario_has_suffix(p, suffix=RSDSBIASADJUST_NAT_SCENARIO_SUFFIX))
            ]
            _log(
                "RSDS bias-adjust source files after NAT filtering: "
                f"{len(gcm_for_bias)} (removed {pre - len(gcm_for_bias)} files with "
                f"suffix '{RSDSBIASADJUST_NAT_SCENARIO_SUFFIX}', mode={nat_scens_mode})"
            )
        if gcm_for_bias:
            rsds_bias_adjustment_legacy = _build_rsds_era5_bias_adjustment(
                era5_file=era5_file,
                gcm_files=gcm_for_bias,
                template=template,
                smoothing_window_years=DEFAULT_RSDS_BIAS_SMOOTHING_YEARS,
            )
            if rsds_bias_adjustment_legacy is not None:
                artifact_root = all_root if all_files else nat_root
                try:
                    _write_rsds_bias_adjustment_artifacts(
                        root_for_outputs=artifact_root,
                        output_tag=output_tag,
                        region=args.region,
                        template=template,
                        adjustment=rsds_bias_adjustment_legacy,
                    )
                except Exception as exc:
                    _log(f"⚠️ Failed to write RSDS bias-adjustment artifacts: {exc}")
        else:
            _log("RSDS ERA5 bias adjustment requested, but no SCENARIO1/SCENARIO2 files are available; skipping.")
    else:
        _log("RSDS ERA5 bias adjustment disabled by option.")

    if args.include_cmip6 and cmip6_manifest is not None:
        man_path = _write_cmip6_selection_manifest(
            cmip6_root=cmip6_root,
            manifest=cmip6_manifest,
            output_tag=output_tag,
            region=args.region,
        )
        if man_path is not None:
            _log(f"CMIP6 selection manifest written: {man_path}")

    # PET method loop (requested: run all 3 for side-by-side testing)
    pet_methods: List[str]
    # Always include PET tag in paths/names to keep outputs distinguishable
    pet_in_path = True
    if str(args.pet_method).lower().strip() == "all":
        pet_methods = ["thornthwaite", "hargreaves", "penman-monteith"]
    else:
        pet_methods = [str(args.pet_method).lower().strip()]
    _log(f"PET methods to run: {', '.join(pet_methods)}")
    write_per_run = segments_layout in {SEGMENTS_LAYOUT_PER_RUN, SEGMENTS_LAYOUT_DUAL}
    write_run_stacked = segments_layout in {SEGMENTS_LAYOUT_RUN_STACKED, SEGMENTS_LAYOUT_DUAL}

    # Open ERA5 once to resolve required vars for each PET method
    ds_era5_meta = _open_dataset_safe(era5_file, decode_times=True)
    ds_era5_meta = _ensure_lon_0_360(ds_era5_meta)

    # -------------------------------------------------------------------------
    # Baseline fit(s) + processing
    # -------------------------------------------------------------------------
    for pet_req in pet_methods:
        _log(f"\n=== PET method run: {pet_req} ===")
        # Resolve required vars for this PET method using ERA5 metadata
        try:
            pet_resolved, required_vars = _required_vars_for_pet(ds_era5_meta, pet_req)
        except Exception as exc:
            _log(f"⚠️  PET method '{pet_req}' not feasible on ERA5 template file: {exc} (skip)")
            continue
        miss = [v for v in required_vars if v not in ds_era5_meta]
        if miss:
            _log(f"⚠️  PET method '{pet_req}' missing required vars in ERA5 template: {miss} (skip)")
            continue

        summary_nat: List[Tuple[str, xr.Dataset, xr.Dataset]] = []
        summary_all: List[Tuple[str, xr.Dataset, xr.Dataset]] = []
        summary_era5: List[Tuple[str, xr.Dataset, xr.Dataset]] = []
        summary_cmip6_hist: List[Tuple[str, xr.Dataset, xr.Dataset]] = []
        summary_cmip6_hist_nat: List[Tuple[str, xr.Dataset, xr.Dataset]] = []
        summary_cmip6_ssp245: List[Tuple[str, xr.Dataset, xr.Dataset]] = []
        inputs_accum: Dict[str, Dict[str, List[Tuple[str, np.ndarray, np.ndarray, Optional[str]]]]] = {
            "ERA5": {},
            FORCING_SCENARIO2_LABEL: {},
            FORCING_SCENARIO1_LABEL: {},
            "CMIP6-HIST": {},
            "CMIP6-HIST-NAT": {},
            "CMIP6-SSP245": {},
        }

        # Build ensemble-id lookup maps for per_member baselines
        nat_by_eid: Dict[str, Path] = {}
        for f in nat_files:
            eid = _ensemble_id_from_stem(f.stem)
            if eid:
                nat_by_eid[eid.lower()] = f

        all_by_eid: Dict[str, Path] = {}
        for f in all_files:
            eid = _ensemble_id_from_stem(f.stem)
            if eid:
                all_by_eid[eid.lower()] = f

        baseline_roots: Dict[str, Path] = {
            "era5": era5_file.parent,
            "scenario2": nat_root,
            "scenario1": all_root,
        }
        target_roots: Dict[str, Path] = {
            "era5": era5_file.parent,
            "scenario2": nat_root,
            "scenario1": all_root,
        }
        if args.include_cmip6 and cmip6_root is not None:
            target_roots["cmip6_hist"] = cmip6_root
            target_roots["cmip6_hist_nat"] = cmip6_root
            target_roots["cmip6_ssp245"] = cmip6_root
        archive_roots_forcing: Dict[str, Optional[Path]] = {
            "era5": archive_root_era5,
            "scenario2": archive_root_nat,
            "scenario1": archive_root_all,
            "cmip6_hist": archive_root_cmip6,
            "cmip6_hist_nat": archive_root_cmip6,
            "cmip6_ssp245": archive_root_cmip6,
        }
        baseline_archive_roots: Dict[str, Optional[Path]] = {
            "era5": baseline_archive_root_era5,
            "scenario2": baseline_archive_root_nat,
            "scenario1": baseline_archive_root_all,
        }
        baseline_files_by_source: Dict[str, Sequence[Path]] = {
            "era5": [era5_file],
            "scenario2": nat_files,
            "scenario1": all_files,
        }
        baseline_by_eid: Dict[str, Dict[str, Path]] = {
            "scenario2": nat_by_eid,
            "scenario1": all_by_eid,
        }

        def _rsds_bias_tag_for_source(source_key: str) -> str:
            return _rsds_bias_adjustment_tag(
                _rsds_bias_for_source(source_key),
                rsdsbiasadjust_nat_scens_mode=nat_scens_mode,
            )

        def _rsds_bias_for_source(source_key: str) -> Optional[RsdsEra5BiasAdjustment]:
            if rsds_bias_adjustment_legacy is None:
                return None
            if source_key == "scenario2" and nat_scens_mode == "excempt":
                return None
            return rsds_bias_adjustment_legacy if source_key in {"scenario2", "scenario1"} else None

        def _baseline_title_label(target: str) -> str:
            cfg = baseline_cfg[target]
            return f"{cfg['source'].upper()} ({cfg['pooling']})"

        def _cmip6_baseline_title_label(target: str) -> str:
            cfg = cmip6_baseline_cfg[target]
            return f"CMIP6 {cfg['source']} ({cfg['pooling']})"

        baseline_cache: Dict[tuple, tuple[xr.Dataset, Path, str, str, str]] = {}
        # cache value: (ds_fit, fit_path, baseline_label, source_key, pooling)

        def _get_baseline_for_target(target: str, run_id: Optional[str]) -> Tuple[xr.Dataset, Path, str, str, str]:
            """
            target: 'era5'|'scenario2'|'scenario1'  (the forcing being standardized)
            run_id: stem of the run file (for ensemble members) or None for ERA5
            returns: (baseline_ds, baseline_fit_file, baseline_label, source_key, pooling)
            """
            cfg = baseline_cfg[target]
            src = cfg["source"].lower()
            pool = cfg["pooling"].lower()

            # Pooling is effectively irrelevant for ERA5 as a baseline source (single member)
            pool_eff = "pooled" if src == "era5" else pool

            # If target is ERA5 and user requests per_member while sourcing from scenario2/scenario1,
            # we can't map a member id; fall back to pooled.
            if target == "era5" and pool_eff == "per_member" and src in {"scenario2", "scenario1"}:
                _log(
                    f"ERA5 target requested per_member baseline from {src.upper()}; falling back to pooled (no ERA5 ensemble id)."
                )
                pool_eff = "pooled"

            eid = None
            if pool_eff == "per_member":
                eid = _ensemble_id_from_stem(run_id or "")
                if not eid:
                    _log("per_member baseline requested but no ensemble id found; falling back to pooled.")
                    pool_eff = "pooled"

            rsds_bias_tag = _rsds_bias_tag_for_source(src)
            rsds_bias_adj = _rsds_bias_for_source(src)
            cache_key = (src, pool_eff, (eid or "pooled").lower(), rsds_bias_tag)
            if cache_key in baseline_cache:
                ds_cached, fit_cached, b_lbl, b_src, b_pool = baseline_cache[cache_key]
                # Mirror fit into target root even when using cached baseline
                target_root = target_roots.get(target)
                fit_for_target = fit_cached
                if target_root is not None:
                    try:
                        fit_for_target = _mirror_fit_to_target(
                            fit_cached, target_root, region=args.region, output_tag=output_tag
                        )
                    except Exception as exc:
                        _log(f"    ⚠️  failed to mirror baseline fit into {target_root}: {exc}")
                return ds_cached, fit_for_target, b_lbl, b_src, b_pool

            b_root = baseline_roots[src]
            b_archive = baseline_archive_roots.get(src)
            if pool_eff == "pooled":
                b_files = baseline_files_by_source[src]
                b_id = f"{src}-pooled"
                b_label = f"{src.upper()} pooled"
            else:
                assert eid is not None
                if src not in baseline_by_eid or eid.lower() not in baseline_by_eid[src]:
                    _log(
                        f"per_member baseline requested (src={src.upper()}) but member {eid} not found; falling back to pooled."
                    )
                    b_files = baseline_files_by_source[src]
                    b_id = f"{src}-pooled"
                    b_label = f"{src.upper()} pooled"
                    pool_eff = "pooled"
                else:
                    b_files = [baseline_by_eid[src][eid.lower()]]
                    b_id = f"{src}-{eid.lower()}"
                    b_label = f"{src.upper()} member {eid.lower()}"

            ds_fit, fit_path = _load_or_build_baseline(
                baseline_root=b_root,
                baseline_files=b_files,
                baseline_source_key=src,
                baseline_pooling=pool_eff,
                baseline_id=b_id,
                baseline_label=b_label,
                region=args.region,
                scale=args.scale,
                template=template,
                pet_method_resolved=pet_resolved,
                required_vars=required_vars,
                fit=args.fit,
                limit_runs=args.limit_runs if pool_eff == "pooled" else None,
                force=args.force,
                baseline_start_year=baseline_start_year,
                baseline_end_year=baseline_end_year,
                archive_root=b_archive,
                output_tag=output_tag,
                group_pixels=int(args.group_pixels),
                point_groups=point_groups,
                rsds_bias_adjustment=rsds_bias_adj,
                rsds_bias_adjustment_tag=rsds_bias_tag,
                rsdsbiasadjust_nat_scens_mode=nat_scens_mode,
                rsdsbiasadjust_nat_suffix=RSDSBIASADJUST_NAT_SCENARIO_SUFFIX,
            )

            target_root = target_roots.get(target)
            fit_for_target = fit_path
            if target_root is not None:
                try:
                    fit_for_target = _mirror_fit_to_target(
                        fit_path, target_root, region=args.region, output_tag=output_tag
                    )
                except Exception as exc:
                    _log(f"    ⚠️  failed to mirror baseline fit into {target_root}: {exc}")

            baseline_cache[cache_key] = (ds_fit, fit_path, b_label, src, pool_eff)
            return ds_fit, fit_for_target, b_label, src, pool_eff

        cmip6_by_exp_files, cmip6_by_exp_source, cmip6_by_exp_source_member = _build_cmip6_file_index(
            cmip6_hist_files,
            cmip6_hist_nat_files,
            cmip6_ssp245_files,
        )
        cmip6_baseline_cache: Dict[Tuple[str, str, str, str, str], Tuple[xr.Dataset, Path, str, str, str]] = {}

        def _get_cmip6_baseline_for_target(target_key: str, run_path: Path) -> Tuple[xr.Dataset, Path, str, str, str]:
            if cmip6_root is None:
                raise RuntimeError("CMIP6 root is not configured.")

            (
                baseline_files,
                baseline_source_key,
                baseline_pooling,
                baseline_id,
                baseline_label,
                cache_token,
                warning_note,
            ) = _resolve_cmip6_baseline_selection(
                run_path=run_path,
                target_key=target_key,
                cmip6_baseline_cfg=cmip6_baseline_cfg,
                by_exp_files=cmip6_by_exp_files,
                by_exp_source=cmip6_by_exp_source,
                by_exp_source_member=cmip6_by_exp_source_member,
            )
            if warning_note:
                _log(f"    ⚠️ {warning_note}")

            cache_key = (
                pet_resolved,
                target_key,
                baseline_source_key,
                baseline_pooling,
                cache_token.lower(),
            )
            if cache_key in cmip6_baseline_cache:
                ds_cached, fit_cached, b_lbl, b_src, b_pool = cmip6_baseline_cache[cache_key]
                fit_for_target = fit_cached
                try:
                    fit_for_target = _mirror_fit_to_target(
                        fit_cached, cmip6_root, region=args.region, output_tag=output_tag
                    )
                except Exception:
                    pass
                return ds_cached, fit_for_target, b_lbl, b_src, b_pool

            ds_fit, fit_path = _load_or_build_baseline(
                baseline_root=cmip6_root,
                baseline_files=baseline_files,
                baseline_source_key=baseline_source_key,
                baseline_pooling=baseline_pooling,
                baseline_id=baseline_id,
                baseline_label=baseline_label,
                region=args.region,
                scale=args.scale,
                template=template,
                pet_method_resolved=pet_resolved,
                required_vars=required_vars,
                fit=args.fit,
                limit_runs=args.limit_runs if baseline_pooling == "pooled" else None,
                force=args.force,
                baseline_start_year=baseline_start_year,
                baseline_end_year=baseline_end_year,
                archive_root=baseline_archive_root_cmip6,
                output_tag=output_tag,
                group_pixels=int(args.group_pixels),
                point_groups=point_groups,
                rsds_bias_adjustment=None,
                rsds_bias_adjustment_tag="off",
                rsdsbiasadjust_nat_scens_mode=nat_scens_mode,
                rsdsbiasadjust_nat_suffix=RSDSBIASADJUST_NAT_SCENARIO_SUFFIX,
            )
            fit_for_target = fit_path
            try:
                fit_for_target = _mirror_fit_to_target(
                    fit_path, cmip6_root, region=args.region, output_tag=output_tag
                )
            except Exception:
                pass

            out = (ds_fit, fit_for_target, baseline_label, baseline_source_key, baseline_pooling)
            cmip6_baseline_cache[cache_key] = out
            return out

        # Process each forcing group with its configured baseline
        forcing_groups = []
        if run_era5:
            forcing_groups.append(("ERA5", [era5_file], store_era5, "era5"))
        if run_scenario2:
            forcing_groups.append((FORCING_SCENARIO2_LABEL, nat_files, store_nat, "scenario2"))
        if run_scenario1:
            forcing_groups.append((FORCING_SCENARIO1_LABEL, all_files, store_all, "scenario1"))
        if args.include_cmip6 and cmip6_root is not None and store_cmip6 is not None:
            forcing_groups.extend(
                [
                    ("CMIP6-HIST", cmip6_hist_files, store_cmip6, "cmip6_hist"),
                    ("CMIP6-HIST-NAT", cmip6_hist_nat_files, store_cmip6, "cmip6_hist_nat"),
                    ("CMIP6-SSP245", cmip6_ssp245_files, store_cmip6, "cmip6_ssp245"),
                ]
            )
        if not forcing_groups:
            raise RuntimeError("No forcing groups selected for processing.")

        for forcing, files, out_root, target_key in forcing_groups:
            if not files:
                _log(f"No files found for forcing={forcing}")
                continue
            stacked_writer: Optional[RunStackedWriter | NetCDFRunStackedWriter] = None

            files_iter = files[: args.limit_runs] if args.limit_runs is not None else files
            if target_key in {"cmip6_hist", "cmip6_hist_nat", "cmip6_ssp245"}:
                files_iter, dropped_records = _filter_cmip6_runs_with_exact_baselines(
                    files=files_iter,
                    target_key=target_key,
                    cmip6_baseline_cfg=cmip6_baseline_cfg,
                    by_exp_files=cmip6_by_exp_files,
                    by_exp_source=cmip6_by_exp_source,
                    by_exp_source_member=cmip6_by_exp_source_member,
                )
                if dropped_records:
                    _log(
                        f"Dropping {len(dropped_records)} {forcing} run(s) without exact "
                        f"{cmip6_baseline_cfg[target_key]['source']} member baselines."
                    )
                    for record in dropped_records[:10]:
                        if _record_cmip6_exact_baseline_drop(
                            manifest=cmip6_manifest,
                            record=record,
                            cmip6_root=cmip6_root,
                            output_tag=output_tag,
                            regions=[args.region],
                            target_key=target_key,
                        ):
                            _log(f"  - drop {Path(str(record['run_file'])).name}: {record['reason']}")
                    if len(dropped_records) > 10:
                        _log(f"  ... and {len(dropped_records) - 10} more dropped run(s).")
                if not files_iter:
                    _log(f"No retained CMIP6 files remain for forcing={forcing} after exact-baseline filtering.")
                    continue
            _log(f"\nWriting {forcing} SPEI segments -> {out_root}")

            for i, path in enumerate(files_iter, 1):
                run_id = "ERA5" if forcing == "ERA5" else path.stem
                _log(f"  [{i}/{len(files_iter)}] {forcing}: {path.name}")
                try:
                    if target_key in {"cmip6_hist", "cmip6_hist_nat", "cmip6_ssp245"}:
                        baseline_ds, baseline_fit_file, b_label, b_src_key, b_pool = _get_cmip6_baseline_for_target(
                            target_key,
                            path,
                        )
                    else:
                        baseline_ds, baseline_fit_file, b_label, b_src_key, b_pool = _get_baseline_for_target(
                            target_key, None if forcing == "ERA5" else run_id
                        )
                except MissingExactCmip6BaselineError as exc:
                    if _record_cmip6_exact_baseline_drop(
                        manifest=cmip6_manifest,
                        record=exc.record,
                        cmip6_root=cmip6_root,
                        output_tag=output_tag,
                        regions=[args.region],
                        target_key=target_key,
                    ):
                        _log(f"    ⚠️ dropped {forcing} ({run_id}): {exc}")
                    continue
                except Exception as exc:
                    _log(f"    ⚠️  baseline build failed for {forcing} ({run_id}): {exc}")
                    traceback.print_exc()
                    continue

                try:
                    rsds_bias_raw = (
                        rsds_bias_adjustment_legacy
                        if forcing in {FORCING_SCENARIO1_LABEL, FORCING_SCENARIO2_LABEL}
                        else None
                    )
                    rsds_bias_for_run, rsds_bias_hold_early_for_run = _resolve_rsds_bias_adjustment_for_path(
                        path,
                        adjustment=rsds_bias_raw,
                        rsdsbiasadjust_nat_scens_mode=nat_scens_mode,
                        nat_suffix=RSDSBIASADJUST_NAT_SCENARIO_SUFFIX,
                    )
                    bundle, input_means = _compute_speix_bundle_for_file(
                        path,
                        template=template,
                        required_vars=required_vars,
                        pet_method_resolved=pet_resolved,
                        baseline=baseline_ds,
                        scale=args.scale,
                        fit=args.fit,
                        point_groups=point_groups,
                        group_pixels=int(args.group_pixels),
                        rsds_bias_adjustment=rsds_bias_for_run,
                        rsds_bias_hold_first_reference_year_offsets=rsds_bias_hold_early_for_run,
                    )

                    bundle_plot = _filter_spei_output(bundle, out_start_year=args.out_start_year, out_end_year=args.out_end_year, months=None)
                    # Collect input series for comparison plot (region-mean)
                    if input_means:
                        for var_name, (t_vals, vals, units) in input_means.items():
                            forc_key = forcing.upper()
                            slot = inputs_accum.setdefault(forc_key, {})
                            slot.setdefault(var_name, []).append((run_id, t_vals, vals, units))
                    if write_summary_pdf:
                        if forcing == "ERA5":
                            summary_era5.append((run_id, bundle_plot, baseline_ds))
                        elif forcing == FORCING_SCENARIO2_LABEL:
                            summary_nat.append((run_id, bundle_plot, baseline_ds))
                        elif forcing == FORCING_SCENARIO1_LABEL:
                            summary_all.append((run_id, bundle_plot, baseline_ds))
                        elif forcing == "CMIP6-HIST":
                            summary_cmip6_hist.append((run_id, bundle_plot, baseline_ds))
                        elif forcing == "CMIP6-HIST-NAT":
                            summary_cmip6_hist_nat.append((run_id, bundle_plot, baseline_ds))
                        elif forcing == "CMIP6-SSP245":
                            summary_cmip6_ssp245.append((run_id, bundle_plot, baseline_ds))

                    bundle_save = _filter_spei_output(bundle_plot, out_start_year=args.out_start_year, out_end_year=args.out_end_year, months=months)
                except Exception as exc:
                    _log(f"    ✗ failed: {exc}")
                    continue

                if write_per_run:
                    wrote = _write_spei_segment(
                        store_dir=out_root,
                        run_id=run_id,
                        region=args.region,
                        scale=args.scale,
                        payload=bundle_save,
                        forcing_label=forcing,
                        baseline_fit_file=baseline_fit_file,
                        baseline_source_key=b_src_key,
                        baseline_pooling=b_pool,
                        baseline_label=b_label,
                        pet_method=pet_resolved,
                        fit=args.fit,
                        on_existing=on_existing,
                        archive_root=archive_roots_forcing.get(target_key),
                        pet_in_path=pet_in_path,
                        group_pixels=int(args.group_pixels),
                        chunk_time=120,
                        chunk_point=None,
                        compression_level=0,
                    )
                    if wrote:
                        _log(f"    ✓ wrote [per_run]: {wrote}")

                if write_run_stacked:
                    if stacked_writer is None:
                        stacked_writer = _make_run_stacked_writer(
                            output_format=output_format,
                            store_dir=out_root,
                            forcing_label=forcing,
                            region=args.region,
                            scale=int(args.scale),
                            pet_method=pet_resolved,
                            fit=str(args.fit),
                            on_existing=on_existing,
                            archive_root=archive_roots_forcing.get(target_key),
                            group_pixels=int(args.group_pixels),
                            chunk_run=chunk_run,
                            chunk_time=chunk_time,
                            chunk_point=chunk_point,
                            compression_level=compression_level,
                        )
                    wrote_stacked = stacked_writer.write_run(
                        run_id=run_id,
                        payload=bundle_save,
                        baseline_fit_file=baseline_fit_file,
                        baseline_source_key=b_src_key,
                        baseline_pooling=b_pool,
                        baseline_strategy=f"{b_src_key}:{b_pool}",
                    )
                    if wrote_stacked:
                        _log(f"    ✓ wrote [run_stacked]: {wrote_stacked}")

            if write_run_stacked and stacked_writer is not None:
                stacked_store = stacked_writer.finalize()
                if stacked_store is not None and output_format == OUTPUT_FORMAT_ZARR and consolidate_metadata:
                    _consolidate_zarr_store_metadata(stacked_store)

        # Write combined summary PDFs per forcing (overlay all runs)
        if write_summary_pdf:
            try:
                if summary_nat:
                    base_nat = summary_nat[0][2]
                    label_nat = _baseline_title_label("scenario2")
                    strat_nat = baseline_cfg["scenario2"]["pooling"].lower()
                    _write_summary_pdf_multi(
                        root_for_outputs=nat_root,
                        forcing_label=FORCING_SCENARIO2_LABEL,
                        region=args.region,
                        scale=args.scale,
                        pet_method=pet_resolved,
                        fit=args.fit,
                        baseline_strategy=strat_nat,
                        baseline_main=base_nat,
                        baseline_label=label_nat,
                        bundles=summary_nat,
                        pivot_year=int(args.pivot_year),
                        era5_bundle=summary_era5[0] if summary_era5 else None,
                        output_tag=output_tag,
                        group_pixels=int(args.group_pixels),
                    )
                if summary_all:
                    base_all = summary_all[0][2]
                    label_all = _baseline_title_label("scenario1")
                    strat_all = baseline_cfg["scenario1"]["pooling"].lower()
                    _write_summary_pdf_multi(
                        root_for_outputs=all_root,
                        forcing_label=FORCING_SCENARIO1_LABEL,
                        region=args.region,
                        scale=args.scale,
                        pet_method=pet_resolved,
                        fit=args.fit,
                        baseline_strategy=strat_all,
                        baseline_main=base_all,
                        baseline_label=label_all,
                        bundles=summary_all,
                        pivot_year=int(args.pivot_year),
                        era5_bundle=summary_era5[0] if summary_era5 else None,
                        output_tag=output_tag,
                        group_pixels=int(args.group_pixels),
                    )
                if summary_era5:
                    strat_era5 = baseline_cfg["era5"]["pooling"].lower()
                    _write_summary_pdf_multi(
                        root_for_outputs=era5_file.parent,
                        forcing_label="ERA5",
                        region=args.region,
                        scale=args.scale,
                        pet_method=pet_resolved,
                        fit=args.fit,
                        baseline_strategy=strat_era5,
                        baseline_main=summary_era5[0][2],
                        baseline_label=_baseline_title_label("era5"),
                        bundles=summary_era5,
                        pivot_year=int(args.pivot_year),
                        era5_bundle=summary_era5[0],
                        output_tag=output_tag,
                        group_pixels=int(args.group_pixels),
                    )
                if summary_cmip6_hist and cmip6_root is not None:
                    strat_cmip6_hist = cmip6_baseline_cfg["cmip6_hist"]["pooling"].lower()
                    _write_summary_pdf_multi(
                        root_for_outputs=cmip6_root,
                        forcing_label="CMIP6-HIST",
                        region=args.region,
                        scale=args.scale,
                        pet_method=pet_resolved,
                        fit=args.fit,
                        baseline_strategy=strat_cmip6_hist,
                        baseline_main=summary_cmip6_hist[0][2],
                        baseline_label=_cmip6_baseline_title_label("cmip6_hist"),
                        bundles=summary_cmip6_hist,
                        pivot_year=int(args.pivot_year),
                        era5_bundle=summary_era5[0] if summary_era5 else None,
                        output_tag=output_tag,
                        group_pixels=int(args.group_pixels),
                    )
                if summary_cmip6_hist_nat and cmip6_root is not None:
                    strat_cmip6_hist_nat = cmip6_baseline_cfg["cmip6_hist_nat"]["pooling"].lower()
                    _write_summary_pdf_multi(
                        root_for_outputs=cmip6_root,
                        forcing_label="CMIP6-HIST-NAT",
                        region=args.region,
                        scale=args.scale,
                        pet_method=pet_resolved,
                        fit=args.fit,
                        baseline_strategy=strat_cmip6_hist_nat,
                        baseline_main=summary_cmip6_hist_nat[0][2],
                        baseline_label=_cmip6_baseline_title_label("cmip6_hist_nat"),
                        bundles=summary_cmip6_hist_nat,
                        pivot_year=int(args.pivot_year),
                        era5_bundle=summary_era5[0] if summary_era5 else None,
                        output_tag=output_tag,
                        group_pixels=int(args.group_pixels),
                    )
                if summary_cmip6_ssp245 and cmip6_root is not None:
                    strat_cmip6_ssp245 = cmip6_baseline_cfg["cmip6_ssp245"]["pooling"].lower()
                    _write_summary_pdf_multi(
                        root_for_outputs=cmip6_root,
                        forcing_label="CMIP6-SSP245",
                        region=args.region,
                        scale=args.scale,
                        pet_method=pet_resolved,
                        fit=args.fit,
                        baseline_strategy=strat_cmip6_ssp245,
                        baseline_main=summary_cmip6_ssp245[0][2],
                        baseline_label=_cmip6_baseline_title_label("cmip6_ssp245"),
                        bundles=summary_cmip6_ssp245,
                        pivot_year=int(args.pivot_year),
                        era5_bundle=summary_era5[0] if summary_era5 else None,
                        output_tag=output_tag,
                        group_pixels=int(args.group_pixels),
                    )
            except Exception as exc:
                _log(f"    ⚠️ summary PDF failed for PET={pet_req}: {exc}")

        # Combined comparison PDF for SPEI inputs (ERA5 vs SCENARIO1 vs SCENARIO2)
        try:
            out_root_inputs = all_root or nat_root or era5_file.parent
            _write_inputs_comparison_pdf(
                inputs_accum=inputs_accum,
                pet_method=pet_resolved,
                region=args.region,
                output_tag=output_tag,
                out_root=out_root_inputs,
            )
        except Exception as exc:
            _log(f"    ⚠️ COMPARISON_SPEI_INPUTS failed for PET={pet_req}: {exc}")

        _log(f"=== Completed PET method: {pet_req} ===")

    try:
        ds_era5_meta.close()
    except Exception:
        pass

    if bool(getattr(args, "auto_consolidate", False)):
        candidate_roots: List[Path] = []
        for root in (nat_root if run_scenario2 else None, all_root if run_scenario1 else None):
            if root is None:
                continue
            resolved = Path(root).expanduser().resolve(strict=False)
            if not resolved.exists():
                continue
            if any(part.startswith("debiasloop_") for part in resolved.parts):
                candidate_roots.append(resolved)
        deduped_roots: List[Path] = []
        seen_roots: Set[str] = set()
        for root in candidate_roots:
            key = str(root.resolve(strict=False))
            if key in seen_roots:
                continue
            seen_roots.add(key)
            deduped_roots.append(root)
        if deduped_roots:
            _run_autoconsolidate(
                source_paths=deduped_roots,
                config_path=getattr(args, "auto_consolidate_config", None),
                cleanup_local=bool(getattr(args, "auto_consolidate_cleanup_local", True)),
            )
        else:
            _log("ℹ️  Skipping auto-consolidate: no eligible debiasloop roots were processed.")

    _log("\nDone.")


if __name__ == "__main__":
    main(sys.argv[1:])
