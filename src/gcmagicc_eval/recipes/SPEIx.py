# ruff: noqa: E402
# SPEIx.py - general multi-scale SPEI (configurable PET: Thornthwaite | Hargreaves | Penman-Monteith)
# -*- coding: utf-8 -*-
"""
SPEIx (Standardized Precipitation-Evapotranspiration Index, multi-scale) - GOF comparator
=========================================================================================

This module computes SPEI at *one or more* rolling accumulation scales (in months),
from monthly precipitation and PET (Thornthwaite or Hargreaves-Samani). It evaluates
map-based goodness-of-fit (GOF) metrics between datasets A/B over user-defined windows.

Compared to the single-scale SPEI40 module, this recipe:
  * Loops over a list of scales (e.g., [3, 6, 24, 48]), producing per-scale metrics
    and a figure per scale.
  * Uses metricdomain = "SPEIx".
  * Embeds the scale in variable/metric names and figure titles.

For each window & scale, we output (same definitions as before):
  * Map_RMSE_SPEI{scale}MAX_<window>  - RMSE(A_max, B_max)
  * Map_RMSE_SPEI{scale}MU_<window>   - RMSE(A_μ,  B_μ)     (μ of {scale}-mo wb)
  * Map_Dev_SPEI{scale}MU_<window>    - mean(A_μ - B_μ)
  * Map_RMSE_SPEI{scale}SD_<window>   - RMSE(A_σ,  B_σ)     (σ of {scale}-mo wb)
  * Map_Dev_SPEI{scale}SD_<window>    - mean(A_σ - B_σ)

Figure (Mollweide, 3×3 panels; one figure per scale):
  Row 1 (μ):  A μ(wb {scale}-mo, mm) | B μ(...) | A-B μ (mm)
  Row 2 (σ):  A σ(wb {scale}-mo, mm) | B σ(...) | A-B σ (mm)
  Row 3 (max):A SPEI{scale}-max      | B SPEI{scale}-max | A-B SPEI{scale}-max
  Each right-most panel carries RMSE & Dev annotations for its row.

Public API
----------
- gof(file_a, file_b, cfg, comparison="nc")          -> List[Dict]
- expected_records(file_a, file_b, cfg, comparison)  -> List[Dict]
- replot_figure(path_to_json)                        -> matplotlib.figure.Figure

Configuration (cfg)
-------------------
- spei_method : {"auto","thornthwaite","hargreaves","penman-monteith"} (default: "auto")
- spei_scales : List[int] (default: [3, 6, 24, 48])
    (If absent, use single int cfg["rolling_scale_months"] or default above.)
- spei_distribution : {"loglogistic","zscore"} (default: "loglogistic")  # aliases: "fisk","llo"
- spei_fit    : {"ub-pwm","mle"} (default: "ub-pwm"; log-logistic only)
    (Backwards-compatible: spei_fit can still be "zscore" or "loglogistic".)
- calibration_period : [str start, str end] | None  (default: None)
   Optional baseline window (e.g., ["1981-01","2010-12"]) used to fit the
   per‑month distribution parameters or μ/σ; the transform is applied to all times.
- output_folder : str (default: "./data/reports")
- detimetag_versiontag : bool (default: False)
- print_pdf : bool (default: True)
- print_png : bool (default: False)
- save_figure_data : bool (default: False)  # also write JSON payload for replot
- debug : bool (default: False)
- heatmap_percentiles : List[int] or None (default: None -> 1..99)   # for DISTQ
- spectral_period_min : int (default: 2)
- spectral_period_max : int (default: 120)
- compute_region_metrics : bool (default: True)
- plot_region_heatmaps : bool (default: True)  # example regions only
- regions_for_heatmaps : List[str] (default: ["MED","CNA","EAU","EAS","ESAF"])
- show_spectral_maps : bool (default: True)

Notes & caveats (unchanged from SPEI40)
---------------------------------------
* **Default standardization**: per‑calendar‑month **log‑logistic (3‑parameter) fit**
  of the {scale}-month water‑balance sum (P−PET), then CDF→probit to standard normal
  (SPEI). This follows the Expert Developer Guidance on UCAR's Climate Data Guide,
  recommending a log‑logistic model for D=P−ETo time series. By default the
  log‑logistic path matches the R SPEI package (fit='ub-pwm' / distribution='log-Logistic'):
  a Generalized Logistic (GLO) fitted via unbiased PWMs / L‑moments, then qnorm(CDF).
  Z‑score remains an explicit alternative.
* **Baseline (calibration) period**: If provided, parameters are estimated from
  the baseline only (per month), then applied to all times, aligning with guidance
  to use a long base period for comparability.
* pr is assumed in kg m⁻^2 s⁻¹ (converted to mm/month internally).
* PET: When `spei_method="auto"` (default), FAO-56 Penman-Monteith is used if
  prerequisites are available (tasmin,tasmax,rsds,sfcWind,+lat), otherwise
  Hargreaves (tas,tasmin,tasmax,+/−rsds), otherwise Thornthwaite (tas only).
  (UCAR guidance order: Penman‑Monteith > Hargreaves > Thornthwaite).

"""

from __future__ import annotations
import os
import json
import warnings
import gc
from typing import Dict, List, Tuple, Optional

import numpy as np
import xarray as xr

try:
    from scipy.stats import fisk, norm
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

from scr.validation_helpers.helper_bench_metric import check_for_existing_records_batch, parse_filename
from scr.validation_helpers.helper_bench_plot import generate_pdf_filename
from scr.validation_helpers.helper_benchmark import _generate_time_windows
from scr.validation_helpers.helper_recipes import setup_carlito_font, add_bold_title, get_segment_title
from scr.validation_helpers.recipe_metadata_utils import build_metric_metadata
from scr.validation_helpers.helper_path_utils import get_output_folder
from numpy import nanpercentile
from scr.validation_helpers.helper_heatmap_metrics import (
    quantile_rmse_map,
    spectral_rmse_map,
    dominant_period_map,
)
from scr.validation_helpers.helper_regions_ar6 import (
    region_means_for_ar6,
    DEFAULT_AR6_EXAMPLE_REGIONS,
)
from scr.validation_helpers.helper_heatmap_plot import plot_region_scale_time_heatmaps

__all__ = ["gof", "replot_figure", "expected_records"]

# ------------------------------------------------------------------------------
# Constants / schema
# ------------------------------------------------------------------------------

VAR_PR = "pr"
VAR_TAS = "tas"
VAR_TASMIN = "tasmin"
VAR_TASMAX = "tasmax"
VAR_RSDS = "rsds"
VAR_SFCWIND = "sfcWind"
VAR_HURS = "hurs"
VAR_PS = "ps"
VAR_PSL = "psl"

DEFAULT_SCALES: List[int] = [3, 6, 24, 48]  # months
DOMAIN = "SPEIx"
SCHEMA = "SPEIx.v1"
_DEFAULT_DISTRIBUTION = "loglogistic"
_DEFAULT_FIT = "ub-pwm"
_LOGLOG_MIN_SAMPLES = 10
_EPS = 1e-6

# ------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------


def _get_scales(cfg: Dict) -> List[int]:
    """Resolve list of scales to process."""
    if "spei_scales" in cfg and cfg["spei_scales"]:
        vals = cfg["spei_scales"]
        if isinstance(vals, (list, tuple, np.ndarray)):
            return [int(x) for x in vals]
        return [int(vals)]
    if "rolling_scale_months" in cfg and cfg["rolling_scale_months"]:
        return [int(cfg["rolling_scale_months"])]
    return list(DEFAULT_SCALES)


def _as_float32_list(a):
    return np.asarray(a, dtype=np.float32).tolist()


def _write_json(path, obj):
    with open(str(path), "w") as f:
        json.dump(obj, f)


def _debug_probe_pr(ds: xr.Dataset, label: str, tag: str):
    """Quick diagnostics for precipitation units & magnitudes."""
    if VAR_PR not in ds:
        return
    try:
        pr = ds[VAR_PR]
        units = pr.attrs.get("units", "")
        raw = np.asarray(pr.values, dtype="float64")
        p5, p50, p95 = np.nanpercentile(raw, [5, 50, 95])
        print(
            f"[{tag}][DEBUG][{label}] pr.units='{units}'  raw(kg m-2 s-1) p5={p5:.2e}, p50={p50:.2e}, p95={p95:.6f}"
        )
        mm = np.asarray(_monthly_pr_mm(pr).values, dtype="float64")
        p5m, p50m, p95m = np.nanpercentile(mm, [5, 50, 95])
        print(
            f"[{tag}][DEBUG][{label}] pr->mm(30d)         p5={p5m:.2f}, p50={p50m:.1f}, p95={p95m:.0f}"
        )
    except Exception:
        pass


def _build_meta(file_a: str, file_b: str, comparison: str, cfg: Dict) -> Dict[str, str]:
    a_model, a_scen, a_ens = parse_filename(os.path.basename(file_a))
    b_model, b_scen, b_ens = parse_filename(os.path.basename(file_b))
    # GCMagicc naming alignment (as in your other modules)
    if comparison in ["nc", "nn"] and os.path.basename(file_b).startswith("GCMagicc-"):
        gcmagicc_b = os.path.basename(file_b).split("_")[0]
        b_model = gcmagicc_b if comparison == "nn" else f"{gcmagicc_b}_{a_model}"
    if comparison == "nn" and os.path.basename(file_a).startswith("GCMagicc-"):
        gcmagicc_a = os.path.basename(file_a).split("_")[0]
        a_model = f"{gcmagicc_a}_{a_model}"
    if comparison == "on" and os.path.basename(file_a).startswith("GCMagicc-"):
        gcmagicc_a = os.path.basename(file_a).split("_")[0]
        a_model = f"{gcmagicc_a}_{a_model}"
    meta = dict(
        model=a_model, scenario=a_scen, member=a_ens, comp_source_id=b_model, comp_member_id=b_ens
    )
    if comparison in ["nc", "nn", "on"]:
        meta["version_tag"] = cfg.get("version_tag", "unknown")
    return meta


# ------------------------------------------------------------------------------
# PET methods
# ------------------------------------------------------------------------------


def _monthly_pr_mm(pr: xr.DataArray) -> xr.DataArray:
    """
    Convert precipitation to monthly totals in mm.

    Handles common cases:
      - kg m-2 s-1 flux (CF default) -> multiply by seconds in each month
      - mm/day                      -> multiply by days_in_month
      - meters                      -> *1000
      - missing/blank units         -> heuristic: if median < 1e-2, treat as flux
    """
    units = str(pr.attrs.get("units", "")).lower().strip()
    units_c = units.replace(" ", "")

    def _mark(arr: xr.DataArray, desc: str) -> xr.DataArray:
        arr.attrs = dict(pr.attrs)
        arr.attrs["units"] = "mm"
        arr.attrs["description"] = desc
        return arr.astype("float32")

    try:
        dim_days = pr.time.dt.days_in_month.astype("float64")
    except Exception:
        dim_len = pr.sizes.get("time", pr.shape[0])
        dim_days = xr.DataArray(np.full(dim_len, 30.0), dims=("time",)).astype("float64")
    sec = dim_days * 86400.0

    # kg m-2 s-1 flux
    if ("kg" in units_c and "m-2" in units_c and "s-1" in units_c) or ("kgm-2s-1" in units_c):
        return _mark(pr * sec, "Monthly precipitation total derived from flux (kg m-2 s-1)")

    # mm/day
    if "mm/day" in units or "mmd-1" in units_c or "mmd^-1" in units_c or "mmd-1" in units_c:
        return _mark(pr * dim_days, "Monthly precipitation total derived from mm/day rate")

    # meters depth
    if units_c in {"m", "meter", "metre"} or units.endswith(" m"):
        return _mark(pr * 1000.0, "Monthly precipitation depth converted from meters to mm")

    # Heuristic for missing/unknown units
    try:
        sample = float(np.nanmedian(np.asarray(pr.values).ravel()))
    except Exception:
        sample = np.nan
    if (units == "" or units_c == "") and np.isfinite(sample) and abs(sample) < 1e-2:
        return _mark(pr * sec, "Monthly precipitation total derived from flux (heuristic, units missing)")

    # Assume already monthly total in mm
    return _mark(pr, "Monthly precipitation total (assumed)")


def _thornthwaite_pet_mm(tas: xr.DataArray) -> xr.DataArray:
    """
    Simplified Thornthwaite PET (mm/month) using temperature only.
    Month-wise PET_t = 16 * (10*T/I)^a, with T>=0degC; I from monthly clim of T.
    """
    tas_c = (tas - 273.15).astype("float32")
    tmon = tas_c.groupby("time.month").mean("time")
    tmon_pos = xr.where(tmon > 0, tmon, 0.0)
    heat_index = ((tmon_pos / 5.0) ** 1.514).sum("month")
    heat_index = xr.where(heat_index <= 0, xr.ones_like(heat_index) * 1.0, heat_index)
    a = (6.75e-7 * heat_index**3) - (7.71e-5 * heat_index**2) + (1.792e-2 * heat_index) + 0.49239
    T = xr.where(tas_c > 0, tas_c, 0.0)
    pet = 16.0 * ((10.0 * T / heat_index) ** a)
    return pet.clip(min=0.0).astype("float32")


def _rh_to_percent(rh: xr.DataArray) -> xr.DataArray:
    """
    Normalize relative humidity to percent [0..100].
    CMIP-style hurs is typically already in %; some sources may be 0..1.
    """
    units = str(rh.attrs.get("units", "") or "").lower()
    if "percent" in units or units.strip() == "%":
        return rh.astype("float32")
    # Heuristic fallback
    try:
        sample = float(np.asarray(rh.isel(time=0).values).ravel()[0])
    except Exception:
        sample = np.nan
    if np.isfinite(sample) and sample <= 1.2:
        return (rh * 100.0).astype("float32")
    return rh.astype("float32")


def _pressure_to_kpa(p: xr.DataArray) -> xr.DataArray:
    """
    Convert pressure to kPa.
    Supports typical CF units: Pa (most common), hPa/mbar, kPa.
    """
    units = str(p.attrs.get("units", "") or "").lower().strip()
    if "kpa" in units:
        return p.astype("float32")
    if "hpa" in units or "mbar" in units:
        return (p / 10.0).astype("float32")  # hPa -> kPa
    if "pa" in units:
        return (p / 1000.0).astype("float32")  # Pa -> kPa
    # Heuristic fallback based on a small sample
    try:
        sample = float(np.asarray(p.isel(time=0).values).ravel()[0])
    except Exception:
        sample = np.nan
    if np.isfinite(sample):
        if sample > 2000.0:  # likely Pa
            return (p / 1000.0).astype("float32")
        if sample > 200.0:   # likely hPa
            return (p / 10.0).astype("float32")
    return p.astype("float32")  # assume already kPa


def _midmonth_dayofyear(time: xr.DataArray) -> xr.DataArray:
    """
    Mid-month day-of-year (like SPEI R package `penman()`):
      J = yday(first_of_month) + round((days_in_month / 2) - 1)

    We implement this robustly for monthly series whose timestamps may be start,
    mid, or end-of-month by reconstructing yday(first_of_month) via:
      yday(first_of_month) = yday(time) - (day_of_month(time) - 1)
    """
    try:
        doy = time.dt.dayofyear.astype("float64")
        dom = time.dt.day.astype("float64")
        dim = time.dt.days_in_month.astype("float64")
        first_doy = doy - (dom - 1.0)
        half = xr.apply_ufunc(np.round, (dim / 2.0) - 1.0).astype("float64")
        return (first_doy + half).astype("float64")
    except Exception:
        n = int(time.sizes.get("time", len(time)))
        return xr.DataArray(np.linspace(15, 350, n), dims=("time",)).astype("float64")


def _penman_monteith_fao56_pet_mm(
    tasmin: xr.DataArray,
    tasmax: xr.DataArray,
    rsds: xr.DataArray,
    *,
    sfcwind: xr.DataArray | None = None,
    lat: xr.DataArray | None = None,
    rh: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
    crop: str = "short",
) -> xr.DataArray:
    """
    FAO-56 Penman-Monteith reference evapotranspiration (ETo), mm/month.

    This follows the SPEI R-package implementation in `penman()` with method="FAO"
    (see CRAN SPEI, file R/penman.R), adapted to gridded xarray DataArrays:
      - Tmin/Tmax in Kelvin (converted to °C)
      - rsds in W m-2 (converted to MJ m-2 d-1)
      - sfcwind in m s-1 (assumed to be wind at 2 m; no height conversion applied)
      - RH optional (if provided, used as in FAO-56 eq. 19); otherwise uses Tmin
        to approximate dewpoint (ed = es(Tmin)), as in the R code.
      - pressure optional (Pa/hPa/kPa). If missing, uses 101.3 kPa (sea level).

    Notes:
      * Clear-sky radiation uses elevation z if available in R; when z is missing,
        R uses a default z=840 m. We replicate that default (z=840 m).
    """
    Tmin = (tasmin - 273.15).astype("float64")
    Tmax = (tasmax - 273.15).astype("float64")
    Tmean = ((Tmin + Tmax) / 2.0).astype("float64")

    try:
        mlen = tasmin.time.dt.days_in_month.astype("float64")
    except Exception:
        n = int(tasmin.sizes.get("time", 0) or 0)
        mlen = xr.DataArray(np.full(n, 30.0), dims=("time",)).astype("float64")

    J = _midmonth_dayofyear(tasmin["time"]).astype("float64")

    if pressure is None:
        P = xr.ones_like(Tmean, dtype="float64") * 101.3
    else:
        P = _pressure_to_kpa(pressure).astype("float64")
        P, Tmean = xr.align(P, Tmean, join="inner")

    gamma = 0.665e-3 * P

    etmx = 0.611 * np.exp((17.27 * Tmax) / (Tmax + 237.3))
    etmn = 0.611 * np.exp((17.27 * Tmin) / (Tmin + 237.3))
    ea = (etmx + etmn) / 2.0

    et = 0.611 * np.exp((17.27 * Tmean) / (Tmean + 237.3))
    Delta = 4099.0 * et / (Tmean + 237.3) ** 2

    if rh is not None:
        RH = _rh_to_percent(rh).astype("float64")
        RH, ea = xr.align(RH, ea, join="inner")
        ed = ea * (RH / 100.0)
    else:
        ed = etmn

    if lat is None:
        raise ValueError("Penman-Monteith requires latitude (lat) to estimate extraterrestrial radiation (Ra).")
    latr = (lat / 57.2957795).astype("float64")
    latr_b, J_b = xr.broadcast(latr, J)
    delta = 0.409 * np.sin(0.0172 * J_b - 1.39)
    dr = 1.0 + 0.033 * np.cos(0.0172 * J_b)

    sset = -np.tan(latr_b) * np.tan(delta)
    # Clip before arccos so xarray does not evaluate invalid branches and spam warnings.
    sset_clipped = xr.where(sset < -1.0, -1.0, xr.where(sset > 1.0, 1.0, sset))
    omegas = xr.where(np.abs(sset) <= 1.0, np.arccos(sset_clipped), 0.0)
    omegas = xr.where(sset < -1.0, np.pi, omegas)

    Ra = 37.6 * dr * (
        omegas * np.sin(latr_b) * np.sin(delta) +
        np.cos(latr_b) * np.cos(delta) * np.sin(omegas)
    )
    Ra = xr.where(Ra < 0, 0.0, Ra)

    Rs = (rsds.astype("float64") * 86400.0 / 1e6)
    Rs, Ra = xr.align(Rs, Ra, join="inner")

    z_default = 840.0
    Rso = (0.75 + 2e-5 * z_default) * Ra
    Rso = xr.where(Rso <= 0, 1e-6, Rso)

    ac, bc = 1.35, -0.35
    a1, b1 = 0.34, -0.14
    alb = 0.23
    sb = 4.9e-9
    longwave = (ac * Rs / Rso + bc) * (a1 + b1 * np.sqrt(ed)) * sb * (
        ((273.15 + Tmax) ** 4 + (273.15 + Tmin) ** 4) / 2.0
    )
    Rn = (1.0 - alb) * Rs - longwave
    Rn = xr.where(Rs == 0, 0.0, Rn)

    n_times = int(Tmean.sizes.get("time", 0))
    if n_times < 2:
        G = xr.zeros_like(Tmean, dtype="float64")
    else:
        G = 0.07 * (Tmean.shift(time=-1) - Tmean.shift(time=1))
        G_first = 0.14 * (Tmean.isel(time=1) - Tmean.isel(time=0))
        G_last = 0.14 * (Tmean.isel(time=-1) - Tmean.isel(time=-2))
        t_index = xr.DataArray(np.arange(n_times), dims=("time",), coords={"time": Tmean["time"]})
        G = xr.where(t_index == 0, G_first, G)
        G = xr.where(t_index == (n_times - 1), G_last, G)

    if sfcwind is None:
        U2 = xr.ones_like(Tmean, dtype="float64") * 2.0
    else:
        U2 = sfcwind.astype("float64")
        U2, Tmean = xr.align(U2, Tmean, join="inner")

    if crop == "short":
        c1, c2 = 900.0, 0.34
    elif crop == "tall":
        c1, c2 = 1600.0, 0.38
    else:
        raise ValueError("crop must be one of {'short','tall'}")

    ET0 = (0.408 * Delta * (Rn - G) + gamma * (c1 / (Tmean + 273.0)) * U2 * (ea - ed)) / (
        Delta + gamma * (1.0 + c2 * U2)
    )
    ET0 = ET0.clip(min=0.0)

    ET0_mon = (ET0 * mlen).astype("float32")
    return ET0_mon.clip(min=0.0).astype("float32")


def _hargreaves_pet_mm(
    tas: xr.DataArray,
    tasmin: xr.DataArray,
    tasmax: xr.DataArray,
    lat: xr.DataArray | None = None,
    rsds: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Hargreaves-Samani PET (mm/month). If rsds provided, uses Rs; otherwise Ra(latitude).
    """
    tas_c = (tas - 273.15).astype("float32")
    tasmin_c = (tasmin - 273.15).astype("float32")
    tasmax_c = (tasmax - 273.15).astype("float32")
    dT = (tasmax_c - tasmin_c).clip(min=0.0)

    try:
        J = tas.time.dt.dayofyear
        dim_len = tas.sizes["time"]
        dim_days = tas.time.dt.days_in_month
    except Exception:
        dim_len = tas.sizes["time"]
        J = xr.DataArray(np.linspace(15, 350, dim_len), dims=("time",))
        dim_days = xr.DataArray(np.full(dim_len, 30), dims=("time",))

    if rsds is not None:
        Rs_day = (rsds * 86400.0 / 1e6).astype("float32")
        Rs_day, tas_c, dT = xr.align(Rs_day, tas_c, dT, join="inner")
        Rstar = Rs_day
    else:
        if lat is None:
            raise ValueError("Hargreaves with no rsds requires latitude to compute Ra.")
        lat_name = next(
            (
                n
                for n in ("lat", "latitude", "y")
                if (isinstance(lat, xr.DataArray) and (n in lat.dims or n == lat.name))
            ),
            None,
        )
        if lat_name is None:
            lat_vals = xr.DataArray(
                np.deg2rad(lat.values if isinstance(lat, xr.DataArray) else lat).astype("float32"),
                dims=(),
            )
        else:
            lat_vals = np.deg2rad(lat).astype("float32")
        Gsc = 0.0820
        dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * (J / 365.0))
        delta = 0.409 * np.sin(2.0 * np.pi * (J / 365.0) - 1.39)
        latb, _ = xr.broadcast(lat_vals, dr)
        latb = latb.astype("float64")
        delta = delta.astype("float64")
        omegas = np.arccos(np.clip(-np.tan(latb) * np.tan(delta), -1.0, 1.0))
        Ra = (
            (24.0 * 60.0 / np.pi)
            * Gsc
            * dr
            * (
                omegas * np.sin(latb) * np.sin(delta)
                + np.cos(latb) * np.cos(delta) * np.sin(omegas)
            )
        )
        Ra = Ra.astype("float32")
        lon_name = next((n for n in ("lon", "longitude", "x") if n in tas.dims), None)
        if lon_name and lat_name:
            Ra = Ra.expand_dims({lon_name: tas.sizes[lon_name]}).transpose(
                "time", lat_name, lon_name
            )
        Rstar = Ra

    # Rstar is in MJ m-2 day-1. The factor 0.408 converts radiative energy
    # to equivalent evaporated water depth (mm), matching the manuscript's
    # explicitly named modified Hargreaves radiation proxy.
    PET_day = 0.0023 * 0.408 * Rstar * (tas_c + 17.8) * np.sqrt(dT)
    PET_mon = PET_day * dim_days
    return PET_mon.clip(min=0.0).astype("float32")

def _resolve_pet_method(ds: xr.Dataset, requested: Optional[str]) -> str:
    """
    Choose PET method per UCAR guidance order (PM > Hargreaves > Thornthwaite),
    implemented here as: Penman-Monteith (FAO-56) > Hargreaves > Thornthwaite.
    """
    if requested:
        r = requested.lower()
        if r in ("thornthwaite", "hargreaves", "auto", "penman-monteith", "penman_monteith", "penman", "pm"):
            return "penman-monteith" if r in ("penman-monteith", "penman_monteith", "penman", "pm") else r
    # Default 'auto': Penman-Monteith if inputs exist; else Hargreaves; else Thornthwaite
    has_lat = any(n in ds.coords for n in ("lat", "latitude", "y")) or ("lat" in ds) or ("latitude" in ds)
    has_penman = (VAR_TASMIN in ds) and (VAR_TASMAX in ds) and (VAR_RSDS in ds) and (VAR_SFCWIND in ds) and has_lat
    if has_penman:
        return "penman-monteith"
    has_hargreaves = (VAR_TAS in ds) and (VAR_TASMIN in ds) and (VAR_TASMAX in ds)
    return "hargreaves" if has_hargreaves else "thornthwaite"


# ------------------------------------------------------------------------------
# SPEI core (water balance -> rolling -> monthwise standardization)
# ------------------------------------------------------------------------------


def _rolling_sum(a: xr.DataArray, scale: int) -> xr.DataArray:
    """
    Rolling sum strictly along the 'time' dimension (trailing window).
    This is the standard definition for SPEI rolling accumulation.
    """
    if "time" not in a.dims:
        raise ValueError("Expected a time dimension in input array.")
    k = int(scale)
    out = a.rolling(time=k, center=False, min_periods=k).sum()
    # Remove leading timestamps that are all-NaN (the first k-1 months for a trailing k-sum)
    out = out.dropna(dim="time", how="all")
    return out.astype("float32")

def _wb_monthly(ds: xr.Dataset, method: str) -> xr.DataArray:
    """Monthly water balance (pr_mm - pet_mm), no rolling."""
    if VAR_PR not in ds:
        raise ValueError(f"Dataset is missing '{VAR_PR}'.")
    pr_mm = _monthly_pr_mm(ds[VAR_PR])
    method = _resolve_pet_method(ds, method)
    pet_mm = None
    # 1) Penman-Monteith (FAO-56) if requested or auto-selected and feasible
    if method == "penman-monteith":
        ok = (VAR_TASMIN in ds) and (VAR_TASMAX in ds) and (VAR_RSDS in ds)
        if ok:
            lat_name = next((n for n in ("lat", "latitude", "y") if n in ds.coords), None)
            lat_da = ds.coords[lat_name] if lat_name else (ds["lat"] if "lat" in ds else None)
            wind_da = ds[VAR_SFCWIND] if VAR_SFCWIND in ds else None
            rh_da = ds[VAR_HURS] if VAR_HURS in ds else None
            p_da = ds[VAR_PS] if VAR_PS in ds else (ds[VAR_PSL] if VAR_PSL in ds else None)
            try:
                pet_mm = _penman_monteith_fao56_pet_mm(
                    ds[VAR_TASMIN], ds[VAR_TASMAX], ds[VAR_RSDS],
                    sfcwind=wind_da, lat=lat_da, rh=rh_da, pressure=p_da, crop="short"
                )
            except Exception:
                pet_mm = None
    if method == "hargreaves":
        tasmin_ok = VAR_TASMIN in ds
        tasmax_ok = VAR_TASMAX in ds
        rsds_da = ds[VAR_RSDS] if VAR_RSDS in ds else None
        lat_da = (
            ds[next((n for n in ('lat','latitude','y') if n in ds.coords), None)]
            if rsds_da is None else None
        )
        if tasmin_ok and tasmax_ok and VAR_TAS in ds:
            try:
                pet_mm = _hargreaves_pet_mm(ds[VAR_TAS], ds[VAR_TASMIN], ds[VAR_TASMAX],
                                            lat=lat_da, rsds=rsds_da)
            except Exception:
                pet_mm = None
    if pet_mm is None:
        pet_mm = _thornthwaite_pet_mm(ds[VAR_TAS]) if VAR_TAS in ds else xr.zeros_like(pr_mm)
    return (pr_mm - pet_mm).astype("float32").rename("WB")

def _normalize_fit_name(name: Optional[str]) -> str:
    n = (name or _DEFAULT_FIT).strip().lower()
    # Unbiased PWM / L-moments (R SPEI default = 'ub-pwm')
    if n in {"ub-pwm", "ubpwm", "pwm", "pp-pwm", "pppwm", "lmom", "l-moments", "lmoments"}:
        return "ub-pwm"
    # Keep MLE available as an optional path (SciPy fisk)
    if n in {"mle", "maxlik", "max-lik", "ml", "maximum-likelihood"}:
        return "mle"
    # Backwards-compatible: allow "zscore" or legacy distribution names here
    if n in {"zscore", "z-score", "z"}:
        return "zscore"
    if n in {"fisk", "llo", "log-logistic", "loglogistic"}:
        return "ub-pwm"
    return n


def _normalize_dist_name(dist: Optional[str]) -> str:
    d = (dist or _DEFAULT_DISTRIBUTION).strip().lower()
    if d in {"loglogistic", "log-logistic", "fisk", "llo"}:
        return "loglogistic"
    if d in {"zscore", "normal", "gaussian"}:
        return "zscore"
    return d


# ---------------------------------------------------------------------------
# GLO ("log-Logistic" in R SPEI) helpers: unbiased PWMs -> L-moments -> (xi,alpha,kappa)
# ---------------------------------------------------------------------------

def _pwm_ub_0_1_2(x: np.ndarray) -> Tuple[float, float, float]:
    """
    Unbiased probability weighted moments (Greenwood et al. style) for r=0,1,2:
        b_r = (1/n) * sum_{i=r..n-1} [ C(i,r) / C(n-1,r) ] * x_(i)
    where x_(i) are order stats with i 0-indexed.
    """
    xs = np.sort(x.astype(np.float64, copy=False))
    n = xs.size
    if n < 3:
        return (np.nan, np.nan, np.nan)

    b0 = float(np.mean(xs))
    i = np.arange(n, dtype=np.float64)

    # r=1
    denom1 = (n - 1)
    if denom1 <= 0:
        return (np.nan, np.nan, np.nan)
    w1 = i / denom1
    b1 = float(np.sum(w1 * xs) / n)

    # r=2
    denom2 = (n - 1) * (n - 2)
    if denom2 <= 0:
        return (np.nan, np.nan, np.nan)
    w2 = (i * (i - 1.0)) / denom2
    b2 = float(np.sum(w2 * xs) / n)

    return (b0, b1, b2)


def _lmom_1_2_3_from_pwm(b0: float, b1: float, b2: float) -> Tuple[float, float, float]:
    # Standard PWM->L-moment relations:
    #   λ1 = b0
    #   λ2 = 2*b1 - b0
    #   λ3 = 6*b2 - 6*b1 + b0
    lam1 = b0
    lam2 = 2.0 * b1 - b0
    lam3 = 6.0 * b2 - 6.0 * b1 + b0
    return (lam1, lam2, lam3)


def _glo_params_from_lmom(lam1: float, lam2: float, tau3: float) -> Tuple[float, float, float]:
    """
    Closed-form GLO parameter estimates from L-moments.
    Parameterization matches lmom::cdfglo and SPEI:
      xi (location), alpha (scale>0), kappa (shape)
    For GLO, τ3 = -kappa  (for |kappa| < 1).
    """
    if not np.isfinite(lam1) or not np.isfinite(lam2) or not np.isfinite(tau3):
        return (np.nan, np.nan, np.nan)
    if lam2 <= 0:
        return (np.nan, np.nan, np.nan)

    kappa = -float(tau3)
    if abs(kappa) >= 1.0:
        return (np.nan, np.nan, np.nan)

    # kappa ~ 0 -> logistic limit; handle separately
    if abs(kappa) < 1e-6:
        alpha = float(lam2)
        xi = float(lam1)
        return (xi, alpha, 0.0)

    s = np.sin(np.pi * kappa)
    if abs(s) < 1e-12:
        return (np.nan, np.nan, np.nan)

    alpha = float(lam2 * (s / (np.pi * kappa)))
    if not np.isfinite(alpha) or alpha <= 0:
        return (np.nan, np.nan, np.nan)

    xi = float(lam1 - alpha * (1.0 / kappa - np.pi / s))
    return (xi, alpha, kappa)


def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _pglo_cdf(x: np.ndarray, xi: float, alpha: float, kappa: float) -> np.ndarray:
    """
    Generalized Logistic CDF used by lmom::cdfglo:
      y = (x - xi) / alpha
      if kappa == 0:  F = 1 / (1 + exp(-y))
      else:          F = 1 / (1 + exp(-(-ln(1 - kappa*y)/kappa)))
    with support handling when 1 - kappa*y <= 0.
    """
    y = (x - xi) / alpha
    if abs(kappa) < 1e-12:
        return _stable_sigmoid(y)

    arg = 1.0 - kappa * y
    F = np.empty_like(y, dtype=np.float64)

    ok = arg > 0
    y2 = np.empty_like(y, dtype=np.float64)
    y2[ok] = -np.log(arg[ok]) / kappa
    F[ok] = _stable_sigmoid(y2[ok])

    # Outside support: arg<=0 implies CDF is 0 or 1 depending on kappa sign.
    F[~ok] = 0.0 if kappa < 0 else 1.0
    return F


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """
    Acklam inverse-normal approximation (matches 754_* implementation).
    """
    p = np.asarray(p, dtype=np.float64)
    x = np.empty_like(p, dtype=np.float64)

    a = np.array([-3.969683028665376e+01, 2.209460984245205e+02,
                  -2.759285104469687e+02, 1.383577518672690e+02,
                  -3.066479806614716e+01, 2.506628277459239e+00])
    b = np.array([-5.447609879822406e+01, 1.615858368580409e+02,
                  -1.556989798598866e+02, 6.680131188771972e+01,
                  -1.328068155288572e+01])
    c = np.array([-7.784894002430293e-03, -3.223964580411365e-01,
                  -2.400758277161838e+00, -2.549732539343734e+00,
                  4.374664141464968e+00, 2.938163982698783e+00])
    d = np.array([7.784695709041462e-03, 3.224671290700398e-01,
                  2.445134137142996e+00, 3.754408661907416e+00])

    plow = 0.02425
    phigh = 1.0 - plow

    lo = p < plow
    hi = p > phigh
    mid = ~(lo | hi)

    if np.any(lo):
        q = np.sqrt(-2.0 * np.log(p[lo]))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        x[lo] = num / den

    if np.any(mid):
        q = p[mid] - 0.5
        r = q*q
        num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
        den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4]) * r + 1.0)
        x[mid] = num / den

    if np.any(hi):
        q = np.sqrt(-2.0 * np.log(1.0 - p[hi]))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        x[hi] = -(num / den)

    return x


def _zscore(ts: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts, dtype=np.float64)
    out = np.full(ts.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(ts)
    if not np.any(valid):
        return out
    mu = float(np.nanmean(ts[valid]))
    sd = float(np.nanstd(ts[valid]) + _EPS)
    out[valid] = ((ts[valid] - mu) / sd).astype(np.float32)
    return out


def _glo_fit_transform_1d(ts: np.ndarray, calmask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Fit GLO (xi,alpha,kappa) on calibration subset (if provided and sufficient),
    then transform all finite values via qnorm(CDF).
    Falls back to z-score if fitting fails.
    """
    ts = np.asarray(ts, dtype=np.float64)
    out = np.full(ts.shape, np.nan, dtype=np.float32)

    valid = np.isfinite(ts)
    if not np.any(valid):
        return out

    if calmask is None:
        fit_mask = valid
    else:
        cm = np.asarray(calmask, dtype=bool)
        fit_mask = valid & cm

    x_fit = ts[fit_mask]
    if x_fit.size < _LOGLOG_MIN_SAMPLES:
        x_fit = ts[valid]

    if x_fit.size < 3:
        return _zscore(ts)

    b0, b1, b2 = _pwm_ub_0_1_2(x_fit)
    lam1, lam2, lam3 = _lmom_1_2_3_from_pwm(b0, b1, b2)
    if not np.isfinite(lam2) or lam2 <= 0:
        return _zscore(ts)

    tau3 = lam3 / lam2
    xi, alpha, kappa = _glo_params_from_lmom(lam1, lam2, tau3)
    if not (np.isfinite(xi) and np.isfinite(alpha) and np.isfinite(kappa) and alpha > 0):
        return _zscore(ts)

    cdf = _pglo_cdf(ts[valid], xi, alpha, kappa)
    cdf = np.clip(cdf, _EPS, 1.0 - _EPS)
    out[valid] = _norm_ppf(cdf).astype(np.float32)
    return out


def _fisk_fit_transform_1d(ts: np.ndarray, calmask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Fit 3-parameter log-logistic (SciPy fisk) and transform by norm.ppf(CDF).
    """
    ts = np.asarray(ts, dtype=float)
    out = np.full(ts.shape, np.nan, dtype=np.float32)

    valid = np.isfinite(ts)
    if not np.any(valid):
        return out

    if calmask is None:
        fit_mask = valid
    else:
        fit_mask = valid & np.asarray(calmask, dtype=bool)

    x_fit = ts[fit_mask]
    if x_fit.size < _LOGLOG_MIN_SAMPLES:
        x_fit = ts[valid]

    if x_fit.size < _LOGLOG_MIN_SAMPLES:
        # fallback
        return _zscore(ts)

    try:
        c, loc, scale_par = fisk.fit(x_fit)
        cdf = fisk.cdf(ts[valid], c, loc=loc, scale=scale_par)
        cdf = np.clip(cdf, _EPS, 1.0 - _EPS)
        out[valid] = norm.ppf(cdf).astype(np.float32)
    except Exception:
        return _zscore(ts)
    return out

def _spei_from_wb_roll(
    wb_roll: xr.DataArray,
    *,
    distribution: str = _DEFAULT_DISTRIBUTION,
    fit: str = _DEFAULT_FIT,
    calibration_mask: Optional[xr.DataArray] = None,
    debug: bool = False,
) -> xr.DataArray:
    """Standardize a rolled water-balance series per calendar month -> SPEI (CDF→probit)."""
    dist = _normalize_dist_name(distribution if distribution is not None else fit)
    fit = _normalize_fit_name(fit)
    scale_str = f"{wb_roll.attrs.get('scale','')}"
    if dist == "zscore" or fit == "zscore":
        return _standardize_monthwise_zscore(wb_roll, calibration_mask=calibration_mask).rename(f"SPEI{scale_str}")

    if dist != "loglogistic":
        raise ValueError(f"Unknown distribution={dist!r}")

    if fit == "mle" and not _HAVE_SCIPY:
        warnings.warn("[SPEIx] SciPy unavailable for MLE log-logistic; falling back to ub-pwm GLO.")
        fit = "ub-pwm"

    def _group_apply(g: xr.DataArray) -> xr.DataArray:
        cm = calibration_mask.sel(time=g.time) if isinstance(calibration_mask, xr.DataArray) else None
        if cm is None:
            return xr.apply_ufunc(
                _fisk_fit_transform_1d if fit == "mle" else _glo_fit_transform_1d,
                g,
                input_core_dims=[["time"]],
                output_core_dims=[["time"]],
                vectorize=True,
                dask=None,                    # avoid nested parallelization
                output_dtypes=[np.float32],
            )
        return xr.apply_ufunc(
            _fisk_fit_transform_1d if fit == "mle" else _glo_fit_transform_1d,
            g,
            cm,
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask=None,                    # avoid nested parallelization
            output_dtypes=[np.float32],
        )

    spei = wb_roll.groupby("time.month").apply(_group_apply).astype("float32")
    spei.attrs.update({"distribution": dist, "fit": fit})
    return spei.rename(f"SPEI{scale_str}")


def _standardize_monthwise_zscore(arr: xr.DataArray, *, calibration_mask: Optional[xr.DataArray] = None) -> xr.DataArray:
    """Month‑wise z‑score using μ/σ from calibration subset if provided."""
    if calibration_mask is not None:
        arr_cal = arr.where(calibration_mask)
        mu_cal = arr_cal.groupby("time.month").mean("time")
        sd_cal = arr_cal.groupby("time.month").std("time") + 1e-6
        # If some month/cell had no calibration data, fall back to full data μ/σ
        mu_all = arr.groupby("time.month").mean("time")
        sd_all = arr.groupby("time.month").std("time") + 1e-6
        mu = xr.where(np.isfinite(mu_cal), mu_cal, mu_all)
        sd = xr.where(np.isfinite(sd_cal), sd_cal, sd_all)
        
        def _zscore_group(g: xr.DataArray) -> xr.DataArray:
            month = g.time.dt.month[0].values
            return ((g - mu.sel(month=month)) / sd.sel(month=month)).astype("float32")
        
        return arr.groupby("time.month").apply(_zscore_group)
    else:
        mu = arr.groupby("time.month").mean("time")
        sd = arr.groupby("time.month").std("time") + 1e-6
        
        def _zscore_group(g: xr.DataArray) -> xr.DataArray:
            month = g.time.dt.month[0].values
            return ((g - mu.sel(month=month)) / sd.sel(month=month)).astype("float32")
        
        return arr.groupby("time.month").apply(_zscore_group)


def _spei_series(
    ds: xr.Dataset,
    *,
    method: str,
    scale: int,
    distribution: str = _DEFAULT_DISTRIBUTION,
    fit: str = _DEFAULT_FIT,
    calibration_mask: Optional[xr.DataArray] = None,
    debug: bool = False,
) -> xr.DataArray:
    """
    SPEI via water‑balance rolling sum @scale with selectable standardization:
      - distribution='zscore' (fast, robust),
      - distribution='loglogistic' with fit='ub-pwm' (default) or fit='mle' (SciPy fisk).
    """
    wb = _wb_monthly(ds, method)
    wb_roll = _rolling_sum(wb, scale=scale)
    wb_roll.attrs["scale"] = scale
    return _spei_from_wb_roll(
        wb_roll,
        distribution=distribution,
        fit=fit,
        calibration_mask=calibration_mask,
        debug=debug,
    ).rename(f"SPEI{scale}")


def _water_balance_roll(ds: xr.Dataset, *, method: str, scale: int) -> xr.DataArray:
    """Underlying water-balance rolling sum (mm) used for standardization."""
    wb = _wb_monthly(ds, method)
    return _rolling_sum(wb, scale=scale).astype("float32").rename("WB_roll")


# ------------------------------------------------------------------------------
# Stats & figure
# ------------------------------------------------------------------------------


def _rmse_and_dev(a: xr.DataArray, b: xr.DataArray) -> Tuple[float, float]:
    diff = (a - b).astype("float64")
    dev = float(diff.mean(skipna=True).values)
    rmse = float(np.sqrt((diff**2).mean(skipna=True).values))
    return rmse, dev


def _compute_maps_for_window(
    dsa: xr.Dataset, dsb: xr.Dataset, *,
    method: str, scale: int, distribution: str, fit: str,
    cal_mask_a: Optional[xr.DataArray] = None,
    cal_mask_b: Optional[xr.DataArray] = None,
    hm_percentiles=None, pmin: int = 2, pmax: int = 120,
) -> Tuple[
    xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray,
    xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray
]:
    """Return (A_MAX,B_MAX,A_MU,B_MU,A_SD,B_SD,QMAP,SMAP,DOMA,DOMB,SPEIA,SPEIB) maps for the window."""
    # Compute monthly water balance once, then roll once, then reuse
    wbA = _wb_monthly(dsa, method)
    wbB = _wb_monthly(dsb, method)
    wbrA = _rolling_sum(wbA, scale=scale); wbrA.attrs["scale"] = scale
    wbrB = _rolling_sum(wbB, scale=scale); wbrB.attrs["scale"] = scale
    # SPEI from rolled WB (standardization only)
    if isinstance(cal_mask_a, xr.DataArray):
        cal_mask_a = cal_mask_a.sel(time=wbrA.time, drop=False)
    if isinstance(cal_mask_b, xr.DataArray):
        cal_mask_b = cal_mask_b.sel(time=wbrB.time, drop=False)
    speiA = _spei_from_wb_roll(wbrA, distribution=distribution, fit=fit, calibration_mask=cal_mask_a)
    speiB = _spei_from_wb_roll(wbrB, distribution=distribution, fit=fit, calibration_mask=cal_mask_b)
    Amax = speiA.max("time", skipna=True).astype("float32")
    Bmax = speiB.max("time", skipna=True).astype("float32")

    Amu = wbrA.mean("time", skipna=True).astype("float32")
    Bmu = wbrB.mean("time", skipna=True).astype("float32")
    Asd = wbrA.std("time", skipna=True).astype("float32")
    Bsd = wbrB.std("time", skipna=True).astype("float32")

    # Timing‑invariant maps
    qrmse_map = quantile_rmse_map(speiA, speiB, percentiles=hm_percentiles)
    spec_map = spectral_rmse_map(speiA, speiB, period_min=pmin, period_max=pmax)
    # Dominant persistence period (months)
    domA = dominant_period_map(speiA, period_min=pmin, period_max=pmax)
    domB = dominant_period_map(speiB, period_min=pmin, period_max=pmax)

    def _reduce_to_map(arr: xr.DataArray, how: str) -> xr.DataArray:
        d = arr.squeeze(drop=True)
        non_spatial = [
            n for n in d.dims if n not in ("lat", "latitude", "y", "lon", "longitude", "x")
        ]
        if non_spatial:
            if how == "max":
                d = d.max(non_spatial, skipna=True)
            elif how == "mean":
                d = d.mean(non_spatial, skipna=True)
            elif how == "std":
                d = d.std(non_spatial, skipna=True)
        return d.astype("float32")

    Amax = _reduce_to_map(Amax, "max")
    Bmax = _reduce_to_map(Bmax, "max")
    Amu = _reduce_to_map(Amu, "mean")
    Bmu = _reduce_to_map(Bmu, "mean")
    Asd = _reduce_to_map(Asd, "std")
    Bsd = _reduce_to_map(Bsd, "std")

    Amax, Bmax = xr.align(Amax, Bmax, join="inner")
    Amu, Bmu = xr.align(Amu, Bmu, join="inner")
    Asd, Bsd = xr.align(Asd, Bsd, join="inner")
    qrmse_map, spec_map = xr.align(qrmse_map, spec_map, join="inner")
    domA, domB = xr.align(domA, domB, join="inner")
    return Amax, Bmax, Amu, Bmu, Asd, Bsd, qrmse_map, spec_map, domA, domB, speiA, speiB


def _row_limits(values: np.ndarray, *, fallback: tuple[float, float]) -> tuple[float, float]:
    p1, p99 = nanpercentile(values, [1, 99])
    if not np.isfinite(p1) or not np.isfinite(p99) or p1 == p99:
        return fallback
    return float(p1), float(p99)


def _figure(payload: Dict):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.colors import Normalize

    try:
        import cartopy.crs as ccrs
    except Exception:
        ccrs = None

    lon = np.asarray(payload["coords"]["lon"])
    lat = np.asarray(payload["coords"]["lat"])
    LON, LAT = np.meshgrid(lon, lat)
    proj = ccrs.Mollweide(central_longitude=0) if ccrs else None
    data_crs = ccrs.PlateCarree() if ccrs else None

    setup_carlito_font()
    fig = plt.figure(figsize=(14.5, 13.6))
    gs = GridSpec(4, 3, hspace=0.08, wspace=0.05, left=0.035, right=0.985, top=0.87, bottom=0.075)

    scale = int(payload.get("scale_months", 40))

    panels = [
        (f"A μ(wb {scale}-mo, mm)", "A_MU"),
        (f"B μ(wb {scale}-mo, mm)", "B_MU"),
        ("A-B μ (mm)", "D_MU"),
        (f"A σ(wb {scale}-mo, mm)", "A_SD"),
        (f"B σ(wb {scale}-mo, mm)", "B_SD"),
        ("A-B σ (mm)", "D_SD"),
        (f"A SPEI{scale}-max", "A_SPEI_MAX"),
        (f"B SPEI{scale}-max", "B_SPEI_MAX"),
        (f"A-B SPEI{scale}-max", "D_SPEI_MAX"),
        (f"A dominant persistence (months)", "A_DOMPER"),
        (f"B dominant persistence (months)", "B_DOMPER"),
        ("A-B dominant persistence (months)", "D_DOMPER"),
    ]

    # per-row shared ranges
    r1_vals = np.concatenate(
        [payload["maps"]["A_MU"], payload["maps"]["B_MU"], payload["maps"]["D_MU"]],
        dtype=np.float32,
    )
    r2_vals = np.concatenate(
        [payload["maps"]["A_SD"], payload["maps"]["B_SD"], payload["maps"]["D_SD"]],
        dtype=np.float32,
    )
    r3_vals = np.concatenate(
        [
            payload["maps"]["A_SPEI_MAX"],
            payload["maps"]["B_SPEI_MAX"],
            payload["maps"]["D_SPEI_MAX"],
        ],
        dtype=np.float32,
    )
    if "A_DOMPER" in payload["maps"]:
        r4_vals = np.concatenate(
            [payload["maps"]["A_DOMPER"], payload["maps"]["B_DOMPER"], payload["maps"]["D_DOMPER"]],
            dtype=np.float32,
        )
        pr_low, pr_high = _row_limits(r4_vals, fallback=(2.0, 60.0))
        pr_low = max(pr_low, 2.0)
        norm4_pos = Normalize(vmin=pr_low, vmax=pr_high)
        M4 = max(abs(pr_low), abs(pr_high))
        norm4_diff = TwoSlopeNorm(vmin=-M4, vcenter=0.0, vmax=M4)
    else:
        r4_vals = np.array([0, 1], dtype=np.float32)
        norm4_pos = Normalize(vmin=2.0, vmax=60.0)
        norm4_diff = TwoSlopeNorm(vmin=-10.0, vcenter=0.0, vmax=10.0)

    r1_vmin, r1_vmax = _row_limits(r1_vals, fallback=(-1.0, 1.0))
    M = max(abs(r1_vmin), abs(r1_vmax))
    r1_vmin, r1_vmax = (-M, M) if M > 0 else (-1, 1)
    r2_vmin, r2_vmax = _row_limits(r2_vals, fallback=(-1.0, 1.0))
    M = max(abs(r2_vmin), abs(r2_vmax))
    r2_vmin, r2_vmax = (-M, M) if M > 0 else (-1, 1)
    r3_vmin, r3_vmax = _row_limits(r3_vals, fallback=(-2.0, 2.0))
    M = max(abs(r3_vmin), abs(r3_vmax))
    r3_vmin, r3_vmax = (-M, M) if M > 0 else (-2, 2)

    norm1 = TwoSlopeNorm(vmin=r1_vmin, vcenter=0.0, vmax=r1_vmax)
    norm2 = TwoSlopeNorm(vmin=r2_vmin, vcenter=0.0, vmax=r2_vmax)
    norm3 = TwoSlopeNorm(vmin=r3_vmin, vcenter=0.0, vmax=r3_vmax)

    ims = []
    axs = []
    for i, (title, key) in enumerate(panels):
        ax = (
            fig.add_subplot(gs[i // 3, i % 3], projection=proj)
            if proj
            else fig.add_subplot(gs[i // 3, i % 3])
        )
        if proj:
            ax.coastlines(linewidth=0.4)
            ax.set_global()
        arr = np.asarray(payload["maps"][key]).reshape(len(lat), len(lon))
        if i < 3:
            norm = norm1
        elif i < 6:
            norm = norm2
        elif i < 9:
            norm = norm3
        else:
            norm = norm4_pos if key != "D_DOMPER" else norm4_diff
        kw = {"transform": data_crs} if proj else {}
        cmap = "viridis" if key.endswith("DOMPER") and key != "D_DOMPER" else "BrBG"
        im = ax.pcolormesh(LON, LAT, arr, cmap=cmap, norm=norm, rasterized=True, **kw)
        if i in (2, 5, 8):  # annotate rightmost column
            if key == "D_MU":
                rmse = payload["stats"]["RMSE_MU"]
                dev = payload["stats"]["DEV_MU"]
            elif key == "D_SD":
                rmse = payload["stats"]["RMSE_SD"]
                dev = payload["stats"]["DEV_SD"]
            else:
                rmse = payload["stats"]["RMSE_SPEIMAX"]
                dev = payload["stats"]["DEV_SPEIMAX"]
            ax.text(
                0.01,
                0.97,
                f"RMSE={rmse:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=2),
            )
            ax.text(
                0.99,
                0.97,
                f"Dev={dev:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=2),
            )
        ax.set_title(title, fontsize=9)
        ims.append(im)
        axs.append(ax)

    # per-row colorbars on the right of each row's rightmost axis
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def _append_cbar_right(fig, ax_right, mappable, label: str):
        divider = make_axes_locatable(ax_right)
        try:
            cax = divider.append_axes("right", size="3%", pad=0.02)
            cb = fig.colorbar(mappable, cax=cax, orientation="vertical")
        except Exception:
            cb = fig.colorbar(
                mappable, ax=ax_right, orientation="vertical", fraction=0.046, pad=0.04
            )
        cb.set_label(label, fontsize=9)
        cb.ax.tick_params(labelsize=8)
        return cb

    _append_cbar_right(fig, axs[2], ims[2], f"μ(water balance {scale}-mo, mm)")
    _append_cbar_right(fig, axs[5], ims[5], f"σ(water balance {scale}-mo, mm)")
    _append_cbar_right(fig, axs[8], ims[8], f"SPEI{scale} (max & diff, units)")
    if "A_DOMPER" in payload["maps"]:
        _append_cbar_right(fig, axs[11], ims[11], "Dominant persistence (months)")

    fa = payload.get("file_a_name", "A")
    fb = payload.get("file_b_name", "B")
    method = payload.get("spei_method", "thornthwaite")
    add_bold_title(fig, get_segment_title("SPEIx"), y=0.975)

    # Add file information with separate fig.text commands
    fig.text(0.05, 0.945,
             f"SPEI{scale} analysis (μ, σ of {scale}-mo water balance; max SPEI; + dominant persistence)",
             fontsize=11, ha="left", va="top", transform=fig.transFigure)
    fig.text(
        0.05, 0.93, f"method={method}", fontsize=11, ha="left", va="top", transform=fig.transFigure
    )
    fig.text(0.05, 0.91, f"A: {fa}", fontsize=11, ha="left", va="top", transform=fig.transFigure)
    fig.text(0.05, 0.89, f"B: {fb}", fontsize=11, ha="left", va="top", transform=fig.transFigure)
    fig.text(
        0.05,
        0.87,
        f"window={payload.get('window','full')}",
        fontsize=11,
        ha="left",
        va="top",
        transform=fig.transFigure,
    )
    return fig


def _prepare_payload(
    *,
    window: str,
    file_a_name: str,
    file_b_name: str,
    maps: Dict[str, np.ndarray],
    coords: Dict[str, np.ndarray],
    stats: Dict[str, float],
    method: str,
    scale: int,
) -> Dict:
    return {
        "schema": SCHEMA,
        "window": window,
        "file_a_name": file_a_name,
        "file_b_name": file_b_name,
        "maps": maps,
        "coords": coords,
        "stats": stats,
        "spei_method": method,
        "scale_months": scale,
    }


def replot_figure(pathname_to_json_file: str):
    with open(pathname_to_json_file, "r") as f:
        blob = json.load(f)
    if blob.get("schema") != SCHEMA:
        raise ValueError("Unsupported or missing schema in figure JSON.")
    return _figure(blob)


# ------------------------------------------------------------------------------
# Expected records for de-duplication
# ------------------------------------------------------------------------------


def _base_record(meta: Dict, comparison: str, scale: int) -> Dict:
    rec = dict(
        metrickey=f"GOF{comparison.upper()}",
        metricdomain=DOMAIN,
        variable=f"SPEI{scale}",
        source_id=meta["model"],
        member_id=meta["member"],
        experiment_id=meta["scenario"],
        comp_source_id=meta["comp_source_id"],
        comp_member_id=meta["comp_member_id"],
    )
    if comparison in ["nc", "nn", "on"]:
        rec["version_tag"] = meta.get("version_tag", "unknown")
    return rec


def expected_records(file_a: str, file_b: str, cfg: Dict, *, comparison: str = "nc") -> List[Dict]:
    meta = _build_meta(file_a, file_b, comparison, cfg)
    scales = _get_scales(cfg)
    try:
        with xr.open_dataset(file_a, use_cftime=True) as da:
            windows = _generate_time_windows(da, {**cfg, "comparison": comparison})
    except Exception:
        windows = [(None, None, "full")]

    tpls: List[Dict] = []
    for scale in scales:
        for _, _, wname in windows:
            base = _base_record(meta, comparison, scale)
            tpls.append({**base, "metrictype": f"Map_RMSE_SPEI{scale}MAX_{wname}"})
            tpls.append({**base, "metrictype": f"Map_RMSE_SPEI{scale}MU_{wname}"})
            tpls.append({**base, "metrictype": f"Map_Dev_SPEI{scale}MU_{wname}"})
            tpls.append({**base, "metrictype": f"Map_RMSE_SPEI{scale}SD_{wname}"})
            tpls.append({**base, "metrictype": f"Map_Dev_SPEI{scale}SD_{wname}"})
            tpls.append({**base, "metrictype": f"Map_RMSE_SPEI{scale}DISTQ_{wname}"})
            tpls.append({**base, "metrictype": f"Map_RMSE_SPEI{scale}SPEC_{wname}"})
            try:
                from scr.validation_helpers.helper_regions_ar6 import DEFAULT_AR6_EXAMPLE_REGIONS as _DEF
                for reg in cfg.get("regions_for_heatmaps", _DEF):
                    tpls.append({**base, "metrictype": f"Reg_RMSE_SPEI{scale}DISTQ_{reg}_{wname}"})
                    tpls.append({**base, "metrictype": f"Reg_RMSE_SPEI{scale}SPEC_{reg}_{wname}"})
            except Exception:
                pass
    return tpls


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------


def gof(file_a: str, file_b: str, cfg: Dict, *, comparison: str = "nc") -> List[Dict]:
    """
    Compute multi-scale SPEI map comparisons for each analysis window.

    cfg keys of interest:
      - spei_method          : "thornthwaite" | "hargreaves" (default: thornthwaite)
      - spei_scales          : List[int], default [3, 6, 24, 48]
        (or single int in rolling_scale_months for backward compatibility)
      - output_folder        : str
      - detimetag_versiontag : bool
      - print_pdf / print_png: bools
      - save_figure_data     : bool (also write JSON payload)
      - debug                : bool
    """
    debug = bool(cfg.get("debug", False))
    method = str(cfg.get("spei_method", "auto")).lower()
    distribution = _normalize_dist_name(cfg.get("spei_distribution", cfg.get("spei_fit", _DEFAULT_DISTRIBUTION)))
    fit = _normalize_fit_name(cfg.get("spei_fit", _DEFAULT_FIT))
    if distribution == "zscore" or fit == "zscore":
        distribution, fit = "zscore", "zscore"
    hm_percentiles = cfg.get("heatmap_percentiles", None)
    pmin = int(cfg.get("spectral_period_min", 2))
    pmax = int(cfg.get("spectral_period_max", 120))
    compute_region_metrics = bool(cfg.get("compute_region_metrics", True))
    plot_region_heatmaps_flag = bool(cfg.get("plot_region_heatmaps", True))
    example_regions = cfg.get("regions_for_heatmaps", DEFAULT_AR6_EXAMPLE_REGIONS)
    show_spectral_maps = bool(cfg.get("show_spectral_maps", True))
    scales = _get_scales(cfg)
    meta = _build_meta(file_a, file_b, comparison, cfg)
    # Optional calibration period mask (baseline) for parameter estimation
    cal_period = cfg.get("calibration_period", None)
    def _parse_cal_period(p):
        if not p or not isinstance(p, (list, tuple)) or len(p) != 2:
            return None
        try:
            return (np.datetime64(p[0]), np.datetime64(p[1]))
        except Exception:
            return None
    cal_bounds = _parse_cal_period(cal_period)

    # De-dup check across ALL scales & windows
    tpls = expected_records(file_a, file_b, cfg, comparison=comparison)
    try:
        if check_for_existing_records_batch(tpls, comparison):
            if debug:
                print("[SPEIx] All records exist - skip")
            return []
    except Exception:
        pass

    # Eagerly load datasets once
    try:
        da = xr.load_dataset(file_a, use_cftime=True)
        da.load()
        db = xr.load_dataset(file_b, use_cftime=True)
        db.load()
        if debug:
            _debug_probe_pr(da, "A", "SPEIx")
            _debug_probe_pr(db, "B", "SPEIx")
    except Exception as e:
        if debug:
            print(f"[SPEIx] Failed to load datasets into memory: {e}")
        raise

    # Windows based on A
    try:
        windows = _generate_time_windows(da, {**cfg, "comparison": comparison})
    except Exception:
        windows = [(None, None, "full")]

    recs: List[Dict] = []
    try:
        # Build full-period calibration masks (if requested)
        cal_mask_a_full = None
        cal_mask_b_full = None
        if cal_bounds is not None:
            tA = da["time"]; tB = db["time"]
            cal_mask_a_full = xr.DataArray(
                (tA.values >= cal_bounds[0]) & (tA.values <= cal_bounds[1]),
                coords={"time": tA}, dims=("time",)
            )
            cal_mask_b_full = xr.DataArray(
                (tB.values >= cal_bounds[0]) & (tB.values <= cal_bounds[1]),
                coords={"time": tB}, dims=("time",)
            )
        for scale in scales:
            region_series_A: dict = {}
            region_series_B: dict = {}
            for w0, w1, wname in windows:
                dsa = da.sel(time=slice(str(w0), str(w1))) if w0 and w1 else da
                dsb = db.sel(time=slice(str(w0), str(w1))) if w0 and w1 else db
                # Align calibration masks to this window (may be empty -> function will fallback)
                cal_a = None if cal_mask_a_full is None else cal_mask_a_full.sel(time=dsa.time, drop=False)
                cal_b = None if cal_mask_b_full is None else cal_mask_b_full.sel(time=dsb.time, drop=False)

                try:
                    (
                        Amax, Bmax, Amu, Bmu, Asd, Bsd,
                        QMAP, SMAP, DOMA, DOMB, SPEIA, SPEIB
                    ) = _compute_maps_for_window(
                        dsa, dsb,
                        method=method,
                        scale=scale,
                        distribution=distribution,
                        fit=fit,
                        cal_mask_a=cal_a, cal_mask_b=cal_b,
                        hm_percentiles=hm_percentiles, pmin=pmin, pmax=pmax
                    )
                except Exception as e:
                    if debug:
                        print(f"[SPEIx] compute failed (scale={scale}, window={wname}): {e}")
                    continue

                # stats
                rmse_max, dev_max = _rmse_and_dev(Amax, Bmax)
                rmse_mu, dev_mu = _rmse_and_dev(Amu, Bmu)
                rmse_sd, dev_sd = _rmse_and_dev(Asd, Bsd)
                # spatial RMS of timing‑invariant maps
                qrmse_val = float(np.sqrt((QMAP.astype("float64") ** 2).mean(skipna=True).values))
                spec_val = float(np.sqrt((SMAP.astype("float64") ** 2).mean(skipna=True).values))

                # metric rows
                base = _base_record(meta, comparison, scale)
                current_records: List[Dict] = []
                for mtype, val in (
                    ("Map_RMSE_SPEI{scale}MAX", float(rmse_max)),
                    ("Map_RMSE_SPEI{scale}MU", float(rmse_mu)),
                    ("Map_Dev_SPEI{scale}MU", float(dev_mu)),
                    ("Map_RMSE_SPEI{scale}SD", float(rmse_sd)),
                    ("Map_Dev_SPEI{scale}SD", float(dev_sd)),
                    ("Map_RMSE_SPEI{scale}DISTQ", float(qrmse_val)),
                    ("Map_RMSE_SPEI{scale}SPEC", float(spec_val)),
                ):
                    record = {
                        **base,
                        "metrictype": f"{mtype.format(scale=scale)}_{wname}",
                        "value": val,
                    }
                    current_records.append(record)
                    recs.append(record)

                # Region‑level metrics and heatmap collection
                if compute_region_metrics:
                    try:
                        meansA = region_means_for_ar6(SPEIA)
                        meansB = region_means_for_ar6(SPEIB)
                        for reg, tsA in meansA.items():
                            tsB = meansB.get(reg, None)
                            if tsB is None:
                                continue
                            q_reg = quantile_rmse_map(tsA, tsB, percentiles=hm_percentiles).values.item()
                            s_reg = spectral_rmse_map(tsA, tsB, period_min=pmin, period_max=pmax).values.item()
                            for mtype, val in (
                                ("Reg_RMSE_SPEI{scale}DISTQ_" + reg, float(q_reg)),
                                ("Reg_RMSE_SPEI{scale}SPEC_" + reg, float(s_reg)),
                            ):
                                record = {**base, "metrictype": f"{mtype.format(scale=scale)}_{wname}", "value": val}
                                current_records.append(record)
                                recs.append(record)
                        if plot_region_heatmaps_flag:
                            for reg in example_regions:
                                if reg in meansA and reg in meansB:
                                    region_series_A.setdefault(reg, {})[scale] = meansA[reg]
                                    region_series_B.setdefault(reg, {})[scale] = meansB[reg]
                    except Exception as e:
                        if debug:
                            print(f"[SPEIx] region metrics skipped (scale={scale}, window={wname}): {e}")

                # figure & payload
                try:
                    Dmax = (Amax - Bmax).astype("float32")
                    Dmu = (Amu - Bmu).astype("float32")
                    Dsd = (Asd - Bsd).astype("float32")
                    Ddom = (DOMA - DOMB).astype("float32")
                    lon_name = next(
                        (n for n in ("lon", "longitude", "x") if n in Amax.coords), None
                    )
                    lat_name = next(
                        (n for n in ("lat", "latitude", "y") if n in Amax.coords), None
                    )
                    lon = Amax[lon_name].values if lon_name else np.arange(Amax.shape[-1])
                    lat = Amax[lat_name].values if lat_name else np.arange(Amax.shape[-2])
                    maps = dict(
                        A_MU=_as_float32_list(Amu.values),
                        B_MU=_as_float32_list(Bmu.values),
                        D_MU=_as_float32_list(Dmu.values),
                        A_SD=_as_float32_list(Asd.values),
                        B_SD=_as_float32_list(Bsd.values),
                        D_SD=_as_float32_list(Dsd.values),
                        A_SPEI_MAX=_as_float32_list(Amax.values),
                        B_SPEI_MAX=_as_float32_list(Bmax.values),
                        D_SPEI_MAX=_as_float32_list(Dmax.values),
                        Q_RMSE=_as_float32_list(QMAP.values),
                        SPEC_RMSE=_as_float32_list(SMAP.values),
                        A_DOMPER=_as_float32_list(DOMA.values),
                        B_DOMPER=_as_float32_list(DOMB.values),
                        D_DOMPER=_as_float32_list(Ddom.values),
                    )
                    coords = dict(lon=_as_float32_list(lon), lat=_as_float32_list(lat))
                    stats = dict(
                        RMSE_SPEIMAX=float(rmse_max),
                        DEV_SPEIMAX=float(dev_max),
                        RMSE_MU=float(rmse_mu),
                        DEV_MU=float(dev_mu),
                        RMSE_SD=float(rmse_sd),
                        DEV_SD=float(dev_sd),
                        RMSE_DISTQ=float(qrmse_val),
                        RMSE_SPEC=float(spec_val),
                    )
                    payload = _prepare_payload(
                        window=wname,
                        file_a_name=os.path.basename(file_a),
                        file_b_name=os.path.basename(file_b),
                        maps=maps,
                        coords=coords,
                        stats=stats,
                        method=method,
                        scale=scale,
                    )

                    import matplotlib.pyplot as plt

                    fig = _figure(payload) if show_spectral_maps else _figure({**payload, "maps": {k: v for k, v in maps.items() if not k.endswith("DOMPER")}})
                    base_dir = str(cfg.get("output_folder", get_output_folder()))
                    os.makedirs(base_dir, exist_ok=True)
                    fig_stub = dict(
                        metrickey=f"GOF{comparison.upper()}",
                        metricdomain=DOMAIN,
                        metrictype=f"Figure_SPEI{scale}_{wname}",
                        variable=f"SPEI{scale}",
                        source_id=meta["model"],
                        member_id=meta["member"],
                        experiment_id=meta["scenario"],
                        comp_source_id=meta["comp_source_id"],
                        comp_member_id=meta["comp_member_id"],
                        version_tag=meta.get("version_tag", "unknown"),
                    )
                    pdf_path = generate_pdf_filename(
                        fig_stub,
                        base_dir,
                        detimetag_versiontag=cfg.get("detimetag_versiontag", False),
                    )
                    if cfg.get("print_pdf", True):
                        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
                    if cfg.get("print_png", False):
                        fig.savefig(pdf_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
                    if cfg.get("save_figure_data", False):
                        payload["metadata"] = build_metric_metadata(current_records)
                        _write_json(pdf_path.replace(".pdf", ".json"), payload)
                    plt.close(fig)
                except Exception as e:
                    if debug:
                        print(f"[SPEIx] Figure build failed (scale={scale}, window={wname}): {e}")

                # Region heatmaps (once per window after all scales aggregated)
                try:
                    if plot_region_heatmaps_flag and region_series_A and region_series_B:
                        import matplotlib.pyplot as plt
                        for reg in example_regions:
                            if reg not in region_series_A or reg not in region_series_B:
                                continue
                            fig = plot_region_scale_time_heatmaps(
                                region_series_A[reg], region_series_B[reg],
                                region=reg, index_name="SPEI", window=wname
                            )
                            base_dir = str(cfg.get("output_folder", get_output_folder()))
                            os.makedirs(base_dir, exist_ok=True)
                            fig_stub = dict(
                                metrickey=f"GOF{comparison.upper()}",
                                metricdomain=DOMAIN,
                                metrictype=f"Figure_SPEI_HEATMAP_{reg}_{wname}",
                                variable="SPEI-heatmap",
                                source_id=meta["model"],
                                member_id=meta["member"],
                                experiment_id=meta["scenario"],
                                comp_source_id=meta["comp_source_id"],
                                comp_member_id=meta["comp_member_id"],
                                version_tag=meta.get("version_tag", "unknown"),
                            )
                            pdf_path = generate_pdf_filename(
                                fig_stub, base_dir, detimetag_versiontag=cfg.get("detimetag_versiontag", False),
                            )
                            if cfg.get("print_pdf", True):
                                fig.savefig(pdf_path, dpi=250, bbox_inches="tight")
                            if cfg.get("print_png", False):
                                fig.savefig(pdf_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
                            plt.close(fig)
                except Exception as e:
                    if debug:
                        print(f"[SPEIx] Region heatmaps skipped: {e}")
    finally:
        try:
            if hasattr(da, "close"):
                da.close()
            if hasattr(db, "close"):
                db.close()
        except Exception:
            pass
        del da, db
        gc.collect()

    return recs
