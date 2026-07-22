#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the locked Iranian SPEI-48 attribution and three-SMILE check.

The workflow reads only the Iran bounding box from the large source NetCDFs.
It calculates one December SPEI-48 value per member and year, fits the
log-logistic/GLO standardisation on pooled factual baseline samples, and applies
the same fit to factual and natural-only members.  The primary result is the
area-weighted mean of grid-cell-standardised SPEI.  A regional-water-balance
standardisation is emitted as a sensitivity.

The direct CMIP6 comparison uses a common 1995--2014 attribution window because
GISS-E2-1-G hist-nat ends in 2014.  GCMagicc uses 2021--2025 and 2041--2060.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xarray.coders import CFDatetimeCoder

from gcmagicc_eval.recipes.SPEIx import (
    _glo_fit_transform_1d,
    _rolling_sum,
    _wb_monthly,
)


DEFAULT_CMIP6_ROOT = Path(os.environ.get("GCMAGICC_CMIP6_ROOT", "data/external/cmip6"))
DEFAULT_GCMAGICC_ROOT = Path(os.environ.get("GCMAGICC_ENSEMBLE_ROOT", "data/external/gcmagicc"))
DEFAULT_ERA5_FILE = Path(os.environ.get("GCMAGICC_ERA5_FILE", "data/external/era5.nc"))
DEFAULT_OUTPUT = Path("data/derived/iran_drought_attribution")
MODELS = ("CanESM5", "MIROC6", "GISS-E2-1-G")
PET_METHODS = ("thornthwaite", "hargreaves", "penman-monteith")
VARIABLES = ("pr", "tas", "tasmin", "tasmax", "rsds", "sfcWind", "hurs", "psl")
RSDS_REFERENCE_YEARS = np.arange(1940, 2026, dtype=np.int16)
PRIMARY_BASELINE = (1991, 2010)
BASELINE_30Y = (1981, 2010)
GCMAGICC_RECENT = (2021, 2025)
GCMAGICC_FUTURE = (2041, 2060)
SMILE_RECENT = (1995, 2014)
BOOTSTRAP_SEED = 20260711
BOOTSTRAP_REPLICATES = 10_000
BLOCK_YEARS = 5
SCALE_MONTHS = 48


@dataclass(frozen=True)
class MemberSpec:
    source: str
    forcing: str
    member: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class Region:
    lat_slice: tuple[int, int]
    lon_slice: tuple[int, int]
    lat: np.ndarray
    lon: np.ndarray
    mask: np.ndarray
    weights: np.ndarray
    mask_source: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return BOOTSTRAP_SEED + int(hashlib.sha256(raw).hexdigest()[:8], 16)


def _open(path: str | Path) -> xr.Dataset:
    errors: list[Exception] = []
    for engine in ("h5netcdf", "netcdf4"):
        try:
            return xr.open_dataset(path, engine=engine, decode_times=CFDatetimeCoder(use_cftime=True))
        except Exception as exc:  # pragma: no cover - engine depends on file
            errors.append(exc)
    raise RuntimeError(f"Unable to open {path}: {errors[-1]}")


def _member_key(filename: str) -> tuple[str, str, str] | None:
    parts = filename.split("_", 4)
    if len(parts) != 5 or parts[0] != "DAT":
        return None
    return parts[1], parts[2], parts[3]


def _filename_has_required_variables(filename: str) -> bool:
    """Use the vetted archive's variable suffix as a cheap eligibility preflight."""
    try:
        suffix = filename.split("_", 4)[4]
    except IndexError:
        return False
    available = set(suffix.removesuffix(".nc").split("-"))
    return set(VARIABLES).issubset(available)


def inventory_cmip6(root: Path, max_members: int | None = None) -> list[MemberSpec]:
    by_model: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    for entry in os.scandir(root):
        if not entry.is_file() or not entry.name.endswith(".nc"):
            continue
        parsed = _member_key(entry.name)
        if parsed is None or not _filename_has_required_variables(entry.name):
            continue
        model, experiment, member = parsed
        if model in MODELS and experiment in {"historical", "hist-nat", "ssp245"}:
            by_model[model][experiment][member] = entry.path

    specs: list[MemberSpec] = []
    for model in MODELS:
        experiments = by_model[model]
        paired = sorted(set(experiments["historical"]) & set(experiments["ssp245"]), key=_member_sort)
        natural = sorted(experiments["hist-nat"], key=_member_sort)
        if max_members:
            paired = paired[:max_members]
            natural = natural[:max_members]
        specs.extend(
            MemberSpec(model, "factual", member, (experiments["historical"][member], experiments["ssp245"][member]))
            for member in paired
        )
        specs.extend(
            MemberSpec(model, "natural", member, (experiments["hist-nat"][member],))
            for member in natural
        )
    return specs


def inventory_gcmagicc(
    root: Path, max_members: int | None = None, source_label: str = "GCMagicc"
) -> list[MemberSpec]:
    rows: list[MemberSpec] = []
    member_re = re.compile(r"_(r\d+i\d+p\d+f\d+)_")
    for entry in os.scandir(root):
        if not entry.is_file() or not entry.name.endswith(".nc"):
            continue
        match = member_re.search(entry.name)
        if not match:
            continue
        forcing = "natural" if "_ssp245-nat_" in entry.name else "factual"
        rows.append(MemberSpec(source_label, forcing, match.group(1), (entry.path,)))
    rows.sort(key=lambda row: (row.forcing, _member_sort(row.member)))
    if max_members:
        selected: list[MemberSpec] = []
        for forcing in ("factual", "natural"):
            selected.extend([row for row in rows if row.forcing == forcing][:max_members])
        rows = selected
    return rows


def _member_sort(member: str) -> tuple[int, str]:
    match = re.match(r"r(\d+)", member)
    return (int(match.group(1)) if match else 10**9, member)


def build_region(era5_file: Path) -> Region:
    ds = _open(era5_file)
    try:
        lat = np.asarray(ds["lat"].values, dtype=float)
        lon = np.asarray(ds["lon"].values, dtype=float)
    finally:
        ds.close()
    try:
        import regionmask

        countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
        iran_id = countries.map_keys("IRN")
        mask_full = np.asarray(countries.mask(lon, lat, wrap_lon=360).values == iran_id)
        source = "Natural Earth v5.0.0 countries_110; ISO3 IRN"
    except Exception as exc:  # pragma: no cover - regionmask is in the release env
        raise RuntimeError("regionmask with Natural Earth v5.0.0 is required") from exc
    yy, xx = np.where(mask_full)
    if yy.size == 0:
        raise RuntimeError("IRN mask selected no grid cells")
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    mask = mask_full[y0:y1, x0:x1]
    lat_sub = lat[y0:y1]
    lon_sub = lon[x0:x1]
    weight_grid = np.cos(np.deg2rad(lat_sub))[:, None] * mask
    weights = weight_grid[mask].astype(np.float64)
    weights /= weights.sum()
    return Region((y0, y1), (x0, x1), lat_sub, lon_sub, mask, weights, source)


def _subset_member(spec: MemberSpec, region: Region, variables: Sequence[str]) -> xr.Dataset:
    pieces: list[xr.Dataset] = []
    y0, y1 = region.lat_slice
    x0, x1 = region.lon_slice
    for path in spec.paths:
        ds = _open(path)
        missing = [name for name in variables if name not in ds]
        if missing:
            ds.close()
            raise ValueError(f"{Path(path).name} missing {missing}")
        piece = ds[list(variables)].isel(lat=slice(y0, y1), lon=slice(x0, x1)).load()
        ds.close()
        pieces.append(piece)
    if len(pieces) == 1:
        return pieces[0]
    combined = xr.concat(pieces, dim="time").sortby("time")
    keys = np.asarray(combined.time.dt.year.values, dtype=int) * 100 + np.asarray(
        combined.time.dt.month.values, dtype=int
    )
    _, unique = np.unique(keys, return_index=True)
    return combined.isel(time=np.sort(unique))


def _extract_points(arr: xr.DataArray, region: Region) -> np.ndarray:
    values = np.asarray(arr.values, dtype=np.float32)
    return values[:, region.mask]


def _accumulate_rsds(spec: MemberSpec, region: Region) -> tuple[np.ndarray, np.ndarray]:
    ds = _subset_member(spec, region, ("rsds",))
    try:
        years = np.asarray(ds.time.dt.year.values, dtype=int)
        months = np.asarray(ds.time.dt.month.values, dtype=int)
        values = _extract_points(ds["rsds"], region).astype(np.float64)
    finally:
        ds.close()
    n_years = len(RSDS_REFERENCE_YEARS)
    n_points = int(region.mask.sum())
    sums = np.zeros((12, n_years, n_points), dtype=np.float64)
    counts = np.zeros((12, n_years, n_points), dtype=np.int16)
    y0 = int(RSDS_REFERENCE_YEARS[0])
    for month in range(1, 13):
        selected = (months == month) & (years >= y0) & (years <= int(RSDS_REFERENCE_YEARS[-1]))
        for index in np.where(selected)[0]:
            yi = int(years[index] - y0)
            finite = np.isfinite(values[index])
            sums[month - 1, yi, finite] += values[index, finite]
            counts[month - 1, yi, finite] += 1
    return sums, counts


def _rolling_mean_centered_nan(
    arr: np.ndarray, window: int = 21, edge_trend_years: int = 10, edge_extension_years: int = 11
) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    n_years, n_points = a.shape
    half = window // 2
    pad = max(half, edge_extension_years)
    ext = np.full((n_years + 2 * pad, n_points), np.nan)
    ext[pad : pad + n_years] = a
    x = np.arange(n_years, dtype=float)
    for point in range(n_points):
        y = a[:, point]
        finite = np.where(np.isfinite(y))[0]
        if finite.size == 0:
            continue
        left = finite[: min(edge_trend_years, finite.size)]
        right = finite[-min(edge_trend_years, finite.size) :]
        for target, fit, xp in (
            (slice(0, pad), left, np.arange(-pad, 0, dtype=float)),
            (slice(pad + n_years, None), right, np.arange(n_years, n_years + pad, dtype=float)),
        ):
            if fit.size >= 2:
                slope, intercept = np.polyfit(x[fit], y[fit], 1)
                ext[target, point] = slope * xp + intercept
            else:
                ext[target, point] = y[fit[0]]
    finite = np.isfinite(ext)
    values = np.where(finite, ext, 0.0)
    csum = np.cumsum(values, axis=0)
    ccnt = np.cumsum(finite.astype(np.int32), axis=0)
    out = np.full_like(a, np.nan)
    for index in range(n_years):
        center = pad + index
        lo, hi = center - half, center + half
        sums = csum[hi] - (csum[lo - 1] if lo else 0.0)
        counts = ccnt[hi] - (ccnt[lo - 1] if lo else 0)
        good = counts > 0
        out[index, good] = sums[good] / counts[good]
    return out


def _fill_years(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True, dtype=np.float64)
    x = np.arange(out.shape[0], dtype=float)
    for point in range(out.shape[1]):
        good = np.isfinite(out[:, point])
        out[:, point] = np.interp(x, x[good], out[good, point]) if good.any() else 0.0
    return out


def build_rsds_offsets(
    source: str,
    factual_specs: Sequence[MemberSpec],
    region: Region,
    era5_file: Path,
    workers: int,
) -> np.ndarray:
    era_spec = MemberSpec("ERA5", "reference", "r1", (str(era5_file),))
    era_sum, era_count = _accumulate_rsds(era_spec, region)
    model_sum = np.zeros_like(era_sum)
    model_count = np.zeros_like(era_count, dtype=np.int32)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_accumulate_rsds, spec, region): spec for spec in factual_specs}
        for index, future in enumerate(as_completed(futures), 1):
            sums, counts = future.result()
            model_sum += sums
            model_count += counts
            print(f"[{source}] rsds pass {index}/{len(futures)}", flush=True)
    era = np.divide(era_sum, era_count, out=np.full_like(era_sum, np.nan), where=era_count > 0)
    model = np.divide(model_sum, model_count, out=np.full_like(model_sum, np.nan), where=model_count > 0)
    offsets = np.empty_like(era)
    for month in range(12):
        era_smooth = _rolling_mean_centered_nan(era[month])
        model_smooth = _rolling_mean_centered_nan(model[month])
        offsets[month] = _fill_years(era_smooth - model_smooth)
    return offsets.astype(np.float32)


def _apply_rsds(ds: xr.Dataset, offsets: np.ndarray, region: Region, natural: bool) -> xr.Dataset:
    years = np.asarray(ds.time.dt.year.values, dtype=int)
    months = np.asarray(ds.time.dt.month.values, dtype=int)
    if natural:
        year_index = np.zeros_like(years)
    else:
        year_index = np.clip(years, int(RSDS_REFERENCE_YEARS[0]), int(RSDS_REFERENCE_YEARS[-1])) - int(
            RSDS_REFERENCE_YEARS[0]
        )
    point_offsets = offsets[months - 1, year_index]
    full = np.zeros((len(years), len(region.lat), len(region.lon)), dtype=np.float32)
    full[:, region.mask] = point_offsets
    out = ds.copy(deep=False)
    out["rsds"] = ds["rsds"] + xr.DataArray(
        full, dims=("time", "lat", "lon"), coords={"time": ds.time, "lat": ds.lat, "lon": ds.lon}
    )
    return out


def _december_wb(ds: xr.Dataset, region: Region, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wb = _wb_monthly(ds, method)
    rolled = _rolling_sum(wb, SCALE_MONTHS)
    selected = rolled.time.dt.month == 12
    dec = rolled.sel(time=selected)
    years = np.asarray(dec.time.dt.year.values, dtype=np.int16)
    points = _extract_points(dec, region)
    regional_monthly = np.nansum(_extract_points(wb, region) * region.weights[None, :], axis=1)
    regional_da = xr.DataArray(regional_monthly, dims=("time",), coords={"time": wb.time})
    regional_dec = _rolling_sum(regional_da, SCALE_MONTHS).sel(
        time=_rolling_sum(regional_da, SCALE_MONTHS).time.dt.month == 12
    )
    return years, points.astype(np.float32), np.asarray(regional_dec.values, dtype=np.float32)


def _cache_name(spec: MemberSpec) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{spec.source}__{spec.forcing}__{spec.member}.npz")


def process_member(
    spec: MemberSpec,
    region: Region,
    offsets: np.ndarray,
    cache_dir: str,
) -> str:
    cache = Path(cache_dir) / _cache_name(spec)
    if cache.is_file():
        return str(cache)
    ds = _subset_member(spec, region, VARIABLES)
    payload: dict[str, np.ndarray] = {}
    try:
        for treatment in ("adjusted", "unadjusted"):
            active = _apply_rsds(ds, offsets, region, spec.forcing == "natural") if treatment == "adjusted" else ds
            for method in PET_METHODS:
                years, points, regional = _december_wb(active, region, method)
                payload["years"] = years
                payload[f"grid__{treatment}__{method}"] = points
                payload[f"regional__{treatment}__{method}"] = regional
    finally:
        ds.close()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)
    return str(cache)


def _load_group(cache_paths: Sequence[str], treatment: str, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    regional: list[np.ndarray] = []
    year_rows: list[np.ndarray] = []
    for path in cache_paths:
        with np.load(path) as data:
            years = np.asarray(data["years"], dtype=int)
            year_rows.append(years)
            rows.append(np.asarray(data[f"grid__{treatment}__{method}"], dtype=np.float32))
            regional.append(np.asarray(data[f"regional__{treatment}__{method}"], dtype=np.float32))
    if not year_rows:
        raise ValueError("empty member group")
    common = set(map(int, year_rows[0]))
    for years in year_rows[1:]:
        common.intersection_update(map(int, years))
    years_ref = np.asarray(sorted(common), dtype=int)
    if years_ref.size == 0:
        raise ValueError("members have no common December years")
    aligned_rows: list[np.ndarray] = []
    aligned_regional: list[np.ndarray] = []
    for years, grid, reg in zip(year_rows, rows, regional, strict=True):
        lookup = {int(year): index for index, year in enumerate(years)}
        index = np.asarray([lookup[int(year)] for year in years_ref], dtype=int)
        aligned_rows.append(grid[index])
        aligned_regional.append(reg[index])
    return years_ref, np.stack(aligned_rows), np.stack(aligned_regional)


def _transform_shared(
    factual: np.ndarray,
    factual_years: np.ndarray,
    target: np.ndarray,
    baseline: tuple[int, int],
) -> np.ndarray:
    n_member, n_year, n_point = factual.shape
    flat_fit_source = factual.reshape(n_member * n_year, n_point)
    flat_years = np.tile(factual_years, n_member)
    calibration = (flat_years >= baseline[0]) & (flat_years <= baseline[1])
    combined = np.concatenate([flat_fit_source, target.reshape(-1, n_point)], axis=0)
    calmask = np.concatenate([calibration, np.zeros(target.shape[0] * target.shape[1], dtype=bool)])
    out = np.full_like(combined, np.nan, dtype=np.float32)
    for point in range(n_point):
        out[:, point] = _glo_fit_transform_1d(combined[:, point], calmask)
    return out[len(flat_fit_source) :].reshape(target.shape)


def _transform_era5(values: np.ndarray, years: np.ndarray, baseline: tuple[int, int]) -> np.ndarray:
    calmask = (years >= baseline[0]) & (years <= baseline[1])
    out = np.full_like(values, np.nan, dtype=np.float32)
    for point in range(values.shape[1]):
        out[:, point] = _glo_fit_transform_1d(values[:, point], calmask)
    return out


def _aggregate_grid(spei: np.ndarray, weights: np.ndarray) -> np.ndarray:
    finite = np.isfinite(spei)
    weighted = np.where(finite, spei * weights[None, None, :], 0.0)
    denom = np.where(finite, weights[None, None, :], 0.0).sum(axis=2)
    return np.divide(weighted.sum(axis=2), denom, out=np.full(spei.shape[:2], np.nan), where=denom > 0)


def _window(matrix: np.ndarray, years: np.ndarray, period: tuple[int, int]) -> np.ndarray:
    selected = (years >= period[0]) & (years <= period[1])
    out = matrix[:, selected]
    if out.shape[1] != period[1] - period[0] + 1:
        raise ValueError(f"incomplete window {period}: found {out.shape[1]} years")
    return out


def _bootstrap_probabilities(
    values: np.ndarray,
    threshold: float,
    replicates: int,
    seed: int,
    block_years: int = BLOCK_YEARS,
) -> np.ndarray:
    events = np.asarray(values <= threshold, dtype=np.uint8)
    members, years = events.shape
    nblocks = math.ceil(years / block_years)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    batch = 200
    offsets = np.arange(block_years, dtype=int)
    for start in range(0, replicates, batch):
        count = min(batch, replicates - start)
        member_index = rng.integers(0, members, size=(count, members))
        block_start = rng.integers(0, years, size=(count, members, nblocks))
        year_index = (block_start[..., None] + offsets) % years
        year_index = year_index.reshape(count, members, -1)[..., :years]
        selected = events[member_index[..., None], year_index]
        draws[start : start + count] = selected.mean(axis=(1, 2))
    return draws


def _probability_row(
    source: str,
    forcing: str,
    method: str,
    treatment: str,
    baseline: tuple[int, int],
    aggregation: str,
    period: tuple[int, int],
    values: np.ndarray,
    threshold: float,
    replicates: int,
) -> tuple[dict[str, object], np.ndarray]:
    draws = _bootstrap_probabilities(
        values, threshold, replicates, _stable_seed(source, forcing, method, treatment, baseline, aggregation, period)
    )
    probability = float(np.mean(values <= threshold))
    effective_block_trials = int(values.shape[0] * math.ceil(values.shape[1] / BLOCK_YEARS))
    zero_upper = 1.0 - 0.05 ** (1.0 / effective_block_trials)
    row: dict[str, object] = {
        "record_type": "probability",
        "source": source,
        "forcing": forcing,
        "pet_method": method,
        "rsds_treatment": treatment,
        "baseline_start": baseline[0],
        "baseline_end": baseline[1],
        "aggregation": aggregation,
        "window_start": period[0],
        "window_end": period[1],
        "members": values.shape[0],
        "years_per_member": values.shape[1],
        "threshold": threshold,
        "events": int(np.sum(values <= threshold)),
        "opportunities": int(values.size),
        "estimate": probability,
        "lower": float(np.quantile(draws, 0.025)) if probability > 0 else 0.0,
        "upper": float(np.quantile(draws, 0.975)) if probability > 0 else zero_upper,
        "one_sided": probability == 0,
        "effective_block_trials": effective_block_trials,
    }
    return row, draws


def _ratio_row(
    factual_row: Mapping[str, object],
    factual_draws: np.ndarray,
    natural_row: Mapping[str, object],
    natural_draws: np.ndarray,
) -> dict[str, object]:
    p1 = float(factual_row["estimate"])
    p0 = float(natural_row["estimate"])
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = np.divide(factual_draws, natural_draws)
    estimate = math.inf if p0 == 0 and p1 > 0 else (p1 / p0 if p0 > 0 else math.nan)
    finite = draws[np.isfinite(draws)]
    one_sided = p0 == 0 or np.mean(~np.isfinite(draws)) >= 0.025
    if p0 == 0 and p1 > 0:
        lower_factual = float(np.quantile(factual_draws, 0.05))
        natural_upper = float(natural_row["upper"])
        lower = lower_factual / natural_upper if natural_upper > 0 else math.inf
        point_bound = p1 / natural_upper if natural_upper > 0 else math.inf
    else:
        lower = float(np.quantile(finite, 0.05 if one_sided else 0.025)) if finite.size else math.nan
        point_bound = math.nan
    upper = math.inf if one_sided else float(np.quantile(finite, 0.975))
    row = dict(factual_row)
    row.update(
        {
            "record_type": "probability_ratio",
            "forcing": "factual/natural",
            "events": "",
            "opportunities": "",
            "estimate": estimate,
            "lower": lower,
            "upper": upper,
            "one_sided": one_sided,
            "factual_probability": p1,
            "natural_probability": p0,
            "one_sided_point_ratio_bound": point_bound,
        }
    )
    return row


def _era5_raw(era5_file: Path, region: Region, treatment: str, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = MemberSpec("ERA5", "reference", "r1", (str(era5_file),))
    ds = _subset_member(spec, region, VARIABLES)
    try:
        years, grid, regional = _december_wb(ds, region, method)
    finally:
        ds.close()
    return years, grid[None, ...], regional[None, ...]


def summarize(
    specs: Sequence[MemberSpec],
    cache_paths: Mapping[tuple[str, str, str], str],
    region: Region,
    era5_file: Path,
    output: Path,
    replicates: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    series_rows: list[dict[str, object]] = []
    figure_data: dict[str, np.ndarray] = {}
    by_source: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for spec in specs:
        by_source[spec.source][spec.forcing].append(cache_paths[(spec.source, spec.forcing, spec.member)])

    for method in PET_METHODS:
        era_years, era_grid_raw, era_reg_raw = _era5_raw(era5_file, region, "unadjusted", method)
        for baseline in (PRIMARY_BASELINE, BASELINE_30Y):
            era_grid = _transform_era5(era_grid_raw[0], era_years, baseline)[None, ...]
            era_reg = _transform_era5(era_reg_raw[0, :, None], era_years, baseline)[None, :, 0]
            era_grid_mean = _aggregate_grid(era_grid, region.weights)[0]
            event_index = np.where(era_years == 2025)[0]
            if event_index.size != 1:
                raise ValueError("ERA5 December 2025 event is missing")
            thresholds = {
                "gridcell-spei-area-mean": float(era_grid_mean[event_index[0]]),
                "regional-water-balance-spei": float(era_reg[0, event_index[0]]),
            }
            if method == "penman-monteith" and baseline == PRIMARY_BASELINE:
                figure_data["era5_years"] = era_years
                figure_data["era5_series"] = era_grid_mean
                figure_data["era5_event_map"] = era_grid[0, event_index[0]]

            for source, forcing_groups in by_source.items():
                factual_paths = forcing_groups["factual"]
                natural_paths = forcing_groups["natural"]
                is_emulator = source.startswith("GCMagicc")
                period = GCMAGICC_RECENT if is_emulator else SMILE_RECENT
                for treatment in ("adjusted", "unadjusted"):
                    factual_years, factual_grid_raw, factual_reg_raw = _load_group(factual_paths, treatment, method)
                    natural_years, natural_grid_raw, natural_reg_raw = _load_group(natural_paths, treatment, method)
                    factual_grid = _transform_shared(factual_grid_raw, factual_years, factual_grid_raw, baseline)
                    natural_grid = _transform_shared(factual_grid_raw, factual_years, natural_grid_raw, baseline)
                    factual_reg = _transform_shared(
                        factual_reg_raw[:, :, None], factual_years, factual_reg_raw[:, :, None], baseline
                    )[:, :, 0]
                    natural_reg = _transform_shared(
                        factual_reg_raw[:, :, None], factual_years, natural_reg_raw[:, :, None], baseline
                    )[:, :, 0]
                    aggregates = {
                        "gridcell-spei-area-mean": (
                            _aggregate_grid(factual_grid, region.weights),
                            _aggregate_grid(natural_grid, region.weights),
                        ),
                        "regional-water-balance-spei": (factual_reg, natural_reg),
                    }
                    for aggregation, (factual_values, natural_values) in aggregates.items():
                        # Emit the four locked protocol lanes only.
                        if baseline == BASELINE_30Y and (treatment != "adjusted" or aggregation != "gridcell-spei-area-mean"):
                            continue
                        if baseline == PRIMARY_BASELINE and treatment == "unadjusted" and aggregation != "gridcell-spei-area-mean":
                            continue
                        factual_window = _window(factual_values, factual_years, period)
                        natural_window = _window(natural_values, natural_years, period)
                        threshold = thresholds[aggregation]
                        factual_row, factual_draws = _probability_row(
                            source, "factual", method, treatment, baseline, aggregation, period,
                            factual_window, threshold, replicates,
                        )
                        natural_row, natural_draws = _probability_row(
                            source, "natural", method, treatment, baseline, aggregation, period,
                            natural_window, threshold, replicates,
                        )
                        rows.extend((factual_row, natural_row, _ratio_row(factual_row, factual_draws, natural_row, natural_draws)))

                        if (
                            baseline == PRIMARY_BASELINE
                            and treatment == "adjusted"
                            and aggregation == "gridcell-spei-area-mean"
                        ):
                            for forcing, values, years in (
                                ("factual", factual_values, factual_years),
                                ("natural", natural_values, natural_years),
                            ):
                                for member_index in range(values.shape[0]):
                                    for year, value in zip(years, values[member_index], strict=True):
                                        series_rows.append(
                                            {
                                                "source": source,
                                                "forcing": forcing,
                                                "pet_method": method,
                                                "member_index": member_index + 1,
                                                "year": int(year),
                                                "december_spei48": float(value),
                                            }
                                        )

                        if is_emulator:
                            future_factual = _window(factual_values, factual_years, GCMAGICC_FUTURE)
                            future_natural = _window(natural_values, natural_years, GCMAGICC_FUTURE)
                            frow, fdraw = _probability_row(
                                source, "factual", method, treatment, baseline, aggregation, GCMAGICC_FUTURE,
                                future_factual, threshold, replicates,
                            )
                            nrow, ndraw = _probability_row(
                                source, "natural", method, treatment, baseline, aggregation, GCMAGICC_FUTURE,
                                future_natural, threshold, replicates,
                            )
                            rows.extend((frow, nrow, _ratio_row(frow, fdraw, nrow, ndraw)))
    return rows, series_rows, figure_data


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: object) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except Exception:
        return np.nan


def make_figures(
    rows: Sequence[Mapping[str, object]],
    series_rows: Sequence[Mapping[str, object]],
    figure_data: Mapping[str, np.ndarray],
    region: Region,
    output: Path,
) -> None:
    primary = [
        row for row in rows
        if row["record_type"] == "probability_ratio"
        and row["baseline_start"] == PRIMARY_BASELINE[0]
        and row["rsds_treatment"] == "adjusted"
        and row["aggregation"] == "gridcell-spei-area-mean"
        and row["window_start"] in {GCMAGICC_RECENT[0], SMILE_RECENT[0]}
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
    event_map = np.full(region.mask.shape, np.nan)
    event_map[region.mask] = figure_data["era5_event_map"]
    mesh = axes[0, 0].pcolormesh(region.lon, region.lat, event_map, cmap="BrBG", vmin=-3, vmax=3, shading="auto")
    axes[0, 0].set(title="a  ERA5 December 2025 SPEI-48", xlabel="Longitude", ylabel="Latitude")
    fig.colorbar(mesh, ax=axes[0, 0], label="SPEI-48")
    axes[0, 1].plot(figure_data["era5_years"], figure_data["era5_series"], color="black", lw=1)
    axes[0, 1].axhline(float(figure_data["era5_series"][-1]), color="firebrick", ls="--", lw=1)
    axes[0, 1].set(title="b  Area-weighted Iranian ERA5 series", xlabel="Year", ylabel="December SPEI-48")

    gcm_rows = [row for row in primary if row["source"] == "GCMagicc"]
    x = np.arange(len(PET_METHODS))
    factual_probability = [_finite(next(row["factual_probability"] for row in gcm_rows if row["pet_method"] == method)) for method in PET_METHODS]
    natural_probability = [_finite(next(row["natural_probability"] for row in gcm_rows if row["pet_method"] == method)) for method in PET_METHODS]
    natural_upper = [
        _finite(
            next(
                row["upper"]
                for row in rows
                if row["record_type"] == "probability"
                and row["source"] == "GCMagicc"
                and row["forcing"] == "natural"
                and row["pet_method"] == method
                and row["baseline_start"] == PRIMARY_BASELINE[0]
                and row["rsds_treatment"] == "adjusted"
                and row["aggregation"] == "gridcell-spei-area-mean"
                and row["window_start"] == GCMAGICC_RECENT[0]
            )
        )
        for method in PET_METHODS
    ]
    width_gcm = 0.36
    axes[1, 0].bar(x - width_gcm / 2, factual_probability, width=width_gcm, color="#4477AA", label="Factual")
    axes[1, 0].bar(x + width_gcm / 2, natural_probability, width=width_gcm, color="#BBBBBB", label="Natural-only")
    axes[1, 0].scatter(x + width_gcm / 2, natural_upper, marker="v", color="black", s=22, label="95% upper bound (zero count)")
    axes[1, 0].set_xticks(x, ["Thornthwaite", "Modified\nHargreaves", "Penman--\nMonteith"])
    axes[1, 0].set(title="c  GCMagicc event probability, 2021--2025", ylabel="Probability")
    axes[1, 0].legend(fontsize=8)

    colors = {"CanESM5": "#228833", "MIROC6": "#CCBB44", "GISS-E2-1-G": "#EE6677"}
    width = 0.24
    for model_index, model in enumerate(MODELS):
        model_rows = [row for row in primary if row["source"] == model]
        vals = []
        one_sided = []
        for method in PET_METHODS:
            row = next(row for row in model_rows if row["pet_method"] == method)
            estimate = _finite(row["estimate"])
            bound = _finite(row.get("one_sided_point_ratio_bound"))
            vals.append(estimate if np.isfinite(estimate) else bound)
            one_sided.append(not np.isfinite(estimate) and np.isfinite(bound))
        axes[1, 1].bar(x + (model_index - 1) * width, vals, width=width, label=model, color=colors[model])
        for xpos, value, bounded in zip(x + (model_index - 1) * width, vals, one_sided, strict=True):
            if bounded:
                axes[1, 1].scatter([xpos], [value], marker="^", color="black", s=18, zorder=4)
    axes[1, 1].scatter([], [], marker="^", color="black", s=18,
                       label="Finite point bound (zero natural events)")
    axes[1, 1].set_xticks(x, ["Thornthwaite", "Modified\nHargreaves", "Penman--\nMonteith"])
    axes[1, 1].set(title="d  CMIP6 SMILE comparison, 1995--2014", ylabel="Probability ratio")
    axes[1, 1].legend(fontsize=8)
    for suffix in ("pdf", "png"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(output / f"iran_drought_attribution_common_protocol.{suffix}", dpi=300, metadata=metadata)
    plt.close(fig)

    # Thin-line overlay requested for the supplement (Penman--Monteith primary lane).
    selected = [row for row in series_rows if row["pet_method"] == "penman-monteith"]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for source in ("GCMagicc", *MODELS):
        for forcing, linestyle in (("factual", "-"), ("natural", "--")):
            group = [row for row in selected if row["source"] == source and row["forcing"] == forcing]
            members: dict[int, list[tuple[int, float]]] = defaultdict(list)
            for row in group:
                members[int(row["member_index"])].append((int(row["year"]), float(row["december_spei48"])))
            color = "#4477AA" if source == "GCMagicc" else colors[source]
            for points in members.values():
                points.sort()
                ax.plot([p[0] for p in points], [p[1] for p in points], color=color, alpha=0.08, lw=0.45, ls=linestyle)
            if members:
                all_years = sorted(set(year for points in members.values() for year, _ in points))
                medians = []
                for year in all_years:
                    values = [value for points in members.values() for yy, value in points if yy == year]
                    medians.append(np.nanmedian(values))
                ax.plot(all_years, medians, color=color, lw=1.8, ls=linestyle, label=f"{source} {forcing}")
    ax.axhline(float(figure_data["era5_series"][-1]), color="black", lw=1, ls=":", label="ERA5 Dec 2025 threshold")
    ax.set(xlim=(1940, 2100), xlabel="Year", ylabel="Area-weighted December SPEI-48", title="Iran SPEI-48: GCMagicc and three CMIP6 SMILEs")
    ax.legend(ncol=2, fontsize=7)
    for suffix in ("pdf", "png"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(output / f"iran_smile_common_protocol.{suffix}", dpi=300, metadata=metadata)
    plt.close(fig)


def write_manifest(
    path: Path,
    specs: Sequence[MemberSpec],
    region: Region,
    output_files: Sequence[Path],
    args: argparse.Namespace,
) -> None:
    source_files = []
    for spec in specs:
        for source_path in spec.paths:
            stat = Path(source_path).stat()
            source_files.append(
                {
                    "source": spec.source,
                    "forcing": spec.forcing,
                    "member": spec.member,
                    "filename": Path(source_path).name,
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    payload = {
        "schema": "gcmagicc-drought-common-protocol/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "region": "IRN",
            "event": "ERA5 December 2025 area-weighted SPEI-48",
            "pet_methods": list(PET_METHODS),
            "scale_months": SCALE_MONTHS,
            "primary_baseline": list(PRIMARY_BASELINE),
            "baseline_sensitivity": list(BASELINE_30Y),
            "gcmagicc_recent": list(GCMAGICC_RECENT),
            "gcmagicc_future": list(GCMAGICC_FUTURE),
            "smile_common_window": list(SMILE_RECENT),
            "rsds_primary": "ERA5 minus source ensemble-mean; centered 21-year monthly smoothing; natural-only holds 1940 offset",
            "rsds_sensitivity": "unadjusted",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "moving_block_years": BLOCK_YEARS,
            "region_mask": region.mask_source,
            "grid_cells": int(region.mask.sum()),
        },
        "root_environment_contract": {
            "cmip6": "GCMAGICC_CMIP6_ROOT",
            "gcmagicc": "GCMAGICC_ENSEMBLE_ROOT",
            "era5": "GCMAGICC_ERA5_FILE",
        },
        "reproduction_command": [
            "python", "-m", "gcmagicc_repro", "reproduce", "--figure", "drought-common-protocol"
        ],
        "software_environment": {
            "python": platform.python_version(),
            **{
                package: importlib.metadata.version(package)
                for package in ("numpy", "xarray", "scipy", "matplotlib", "h5netcdf", "netCDF4", "regionmask")
            },
        },
        "workflow_arguments": {
            "gcmagicc_label": args.gcmagicc_label,
            "only_gcmagicc": args.only_gcmagicc,
            "max_members": args.max_members,
            "workers": args.workers,
        },
        "source_files": source_files,
        "outputs": [
            {"path": file.name, "bytes": file.stat().st_size, "sha256": _sha256(file)} for file in output_files
        ],
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmip6-root", type=Path, default=DEFAULT_CMIP6_ROOT)
    parser.add_argument("--gcmagicc-root", type=Path, default=DEFAULT_GCMAGICC_ROOT)
    parser.add_argument("--era5-file", type=Path, default=DEFAULT_ERA5_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/gcmagicc_drought_common_protocol_cache"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-members", type=int)
    parser.add_argument("--gcmagicc-label", default="GCMagicc")
    parser.add_argument("--only-gcmagicc", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--inventory-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for required in (args.cmip6_root, args.gcmagicc_root, args.era5_file):
        if not required.exists():
            raise FileNotFoundError(required)
    region = build_region(args.era5_file)
    specs = inventory_gcmagicc(args.gcmagicc_root, args.max_members, args.gcmagicc_label)
    if not args.only_gcmagicc:
        specs += inventory_cmip6(args.cmip6_root, args.max_members)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for spec in specs:
        counts[spec.source][spec.forcing] += 1
    print(json.dumps({source: dict(forcing) for source, forcing in counts.items()}, indent=2), flush=True)
    print(f"IRN grid cells: {int(region.mask.sum())}", flush=True)
    if args.inventory_only:
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths: dict[tuple[str, str, str], str] = {}
    for source in dict.fromkeys(spec.source for spec in specs):
        source_specs = [spec for spec in specs if spec.source == source]
        factual_specs = [spec for spec in source_specs if spec.forcing == "factual"]
        offset_path = args.cache_dir / f"{source}__rsds_offsets.npz"
        if offset_path.is_file():
            with np.load(offset_path) as data:
                offsets = np.asarray(data["offsets"], dtype=np.float32)
        else:
            offsets = build_rsds_offsets(source, factual_specs, region, args.era5_file, args.workers)
            np.savez_compressed(offset_path, offsets=offsets)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_member, spec, region, offsets, str(args.cache_dir / "members")): spec
                for spec in source_specs
            }
            for index, future in enumerate(as_completed(futures), 1):
                spec = futures[future]
                cache_paths[(spec.source, spec.forcing, spec.member)] = future.result()
                print(f"[{source}] SPEI input pass {index}/{len(futures)}", flush=True)

    rows, series_rows, figure_data = summarize(
        specs, cache_paths, region, args.era5_file, args.output, args.bootstrap_replicates
    )
    summary_path = args.output / "drought_common_protocol_summary.csv"
    series_path = args.output / "drought_common_protocol_series.csv"
    _write_csv(summary_path, rows)
    _write_csv(series_path, series_rows)
    if not args.no_figures:
        make_figures(rows, series_rows, figure_data, region, args.output)
    output_files = [summary_path, series_path, *sorted(args.output.glob("*.pdf")), *sorted(args.output.glob("*.png"))]
    manifest_path = args.output / "drought_common_protocol_manifest.json"
    write_manifest(manifest_path, specs, region, output_files, args)
    print(f"Wrote {summary_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
