#!/usr/bin/env python3
"""
1050_figurex_resolution
=======================

Build a 9x2 synthesis matrix from RESOLUTIONPLOTS JSON outputs:
- Left column: global map (Mollweide projection)
- Right column: Europe zoom
- Rows: NSIDE = 1, 2, 4, 8, 16, 32, 64, 128, 256

Defaults:
- Time slot: latest available (common across selected variables if possible)
- Variable: all available variables (cycles through all)

Notes:
- Input files are expected at: RESOLUTIONPLOTS/<var>/json/tXXXXXX_hY_levels.json
- The "h" in filenames is only a random suffix/hash used for naming collisions.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import string
import sys
import zlib
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import healpy as hp
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "healpy is required for this script. "
        "Use the project environment, e.g. 'pixi run python ...'."
    ) from exc

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "cartopy is required for this script. "
        "Use the project environment, e.g. 'pixi run python ...'."
    ) from exc

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


NSIDES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
COOLWARM_VARS = {"tas", "tasmax", "tasmin", "ts"}
DEFAULT_EUROPE_EXTENT = (-8.15, 23.15, 36.15, 55.85)
SEPARATE_COLORBARS = True
JSON_NAME_RE = re.compile(r"^t(?P<t>\d+)_h(?P<h>\d+)_levels\.json$")
GLOBAL_CENTRAL_LONGITUDE = 0.0
VAR_LONG_NAME_UNIT: dict[str, tuple[str, str]] = {
    "psl": ("Sea level pressure", "Pa"),
    "tas": ("Surface air temperature", "K"),
    "pr": ("Precipitation", "kg m-2 s-1"),
    "sfcWind": ("Near-surface wind speed", "m s-1"),
    "ts": ("Surface temperature", "K"),
    "tasmin": ("Daily minimum surface air temperature", "K"),
    "tasmax": ("Daily maximum surface air temperature", "K"),
    "rsds": ("Surface downwelling shortwave radiation", "W m-2"),
    "hurs": ("Near-surface relative humidity", "%"),
    "huss": ("Near-surface specific humidity", "kg kg-1"),
}


def resolve_default_input_root(script_dir: Path) -> Path:
    candidates = [
        script_dir / "RESOLUTIONPLOTS",
        script_dir.parent / "data" / "nicolaiplots" / "plotsT1" / "plots_resolutions" / "RESOLUTIONPLOTS",
        Path.cwd() / "data" / "nicolaiplots" / "plotsT1" / "plots_resolutions" / "RESOLUTIONPLOTS",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def panel_letter(idx: int) -> str:
    letters = string.ascii_lowercase
    if idx < len(letters):
        return letters[idx]
    q, r = divmod(idx, len(letters))
    return f"{letters[q - 1]}{letters[r]}"


def resolve_colorbar_label(var: str, payload: dict) -> str:
    meta = payload.get("meta", {})
    var_attrs = meta.get("variable_attrs", {}) if isinstance(meta.get("variable_attrs"), dict) else {}

    long_name = None
    for key in ("long_name", "variable_long_name", "longname", "standard_name"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            long_name = val.strip()
            break
    if long_name is None:
        for key in ("long_name", "standard_name", "name"):
            val = var_attrs.get(key)
            if isinstance(val, str) and val.strip():
                long_name = val.strip()
                break

    unit = None
    for key in ("units", "unit", "variable_units"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            unit = val.strip()
            break
    if unit is None:
        for key in ("units", "unit"):
            val = var_attrs.get(key)
            if isinstance(val, str) and val.strip():
                unit = val.strip()
                break

    fallback_long_name, fallback_unit = VAR_LONG_NAME_UNIT.get(var, (var, ""))
    long_name = long_name or fallback_long_name
    unit = unit or fallback_unit
    if unit:
        return f"{long_name} ({var}) ({unit})"
    return f"{long_name} ({var})"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build 9x2 world/europe resolution synthesis figure(s)."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=resolve_default_input_root(here),
        help="Root directory containing RESOLUTIONPLOTS/<var>/json/*.json",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Default: <input-root>/resolution_sensitivity",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default=None,
        help="Variable name, comma-separated list, or 'all'. Default: all variables.",
    )
    parser.add_argument(
        "--time-slot",
        type=int,
        default=None,
        help="Global time index tXXXXXX to use. Default: latest available.",
    )
    parser.add_argument(
        "--global-nlat",
        type=int,
        default=None,
        help="Latitude resolution for global nearest-neighbor sampling.",
    )
    parser.add_argument(
        "--zoom-oversample",
        type=float,
        default=None,
        help="Europe sampling oversample factor. step=(58.6/nside)/oversample",
    )
    parser.add_argument(
        "--europe-extent",
        type=float,
        nargs=4,
        default=None,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Override europe extent [lon_min lon_max lat_min lat_max].",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Raster DPI for saved figures.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Output file formats (e.g., png svg). PDF is always added.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List variables and available time slots, then exit.",
    )
    parser.add_argument(
        "--separate-colorbars",
        action=argparse.BooleanOptionalAction,
        default=SEPARATE_COLORBARS,
        help=(
            "Use separate colorbars for global/europe columns (default: True). "
            "Use --no-separate-colorbars for a single shared bar."
        ),
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars when available (default: True).",
    )
    return parser.parse_args()


def cmap_for_var(var: str):
    name = "coolwarm" if var in COOLWARM_VARS else "viridis"
    return matplotlib.colormaps[name].resampled(1024)


def normalize_formats(formats: list[str], *, force_pdf: bool = True) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for fmt in formats:
        fmt_clean = fmt.lower().strip().lstrip(".")
        if not fmt_clean or fmt_clean in seen:
            continue
        normalized.append(fmt_clean)
        seen.add(fmt_clean)
    if not normalized:
        normalized.append("png")
        seen.add("png")
    if force_pdf and "pdf" not in seen:
        normalized.append("pdf")
    return normalized


def empirical_vmin_vmax(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float32)
    mask = np.isfinite(vals)
    if not np.any(mask):
        return -1.0, 1.0
    vmin = float(np.min(vals[mask]))
    vmax = float(np.max(vals[mask]))
    if vmin == vmax:
        eps = 1e-6 if vmin == 0.0 else abs(vmin) * 1e-6
        return vmin - eps, vmax + eps
    return vmin, vmax


def _lon_to_phi_rad(lon_deg: np.ndarray) -> np.ndarray:
    return np.deg2rad(lon_deg) + np.pi


def _lat_to_theta_rad(lat_deg: np.ndarray) -> np.ndarray:
    return np.deg2rad(90.0 - lat_deg)


def healpix_to_global_lonlat_grid_nearest(
    hp_map: np.ndarray,
    *,
    nlat: int,
    nest: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hp_map = np.asarray(hp_map, dtype=np.float32)
    nside = hp.npix2nside(hp_map.shape[0])

    lats = 90.0 - (0.5 + np.arange(nlat)) / nlat * 180.0
    nlon = 2 * nlat
    # Sample exactly one 360-degree wrap (not two wrapped copies).
    lons = (0.5 + np.arange(nlon)) / nlon * 360.0
    lons = (lons + 180.0) % 360.0 - 180.0

    lon2d, lat2d = np.meshgrid(lons, lats)
    theta = _lat_to_theta_rad(lat2d)
    phi = _lon_to_phi_rad(lon2d)
    pix = hp.ang2pix(nside, theta, phi, nest=nest)
    grid = hp_map[pix].astype(np.float32)
    return grid, lon2d.astype(np.float32), lat2d.astype(np.float32)


def healpix_to_extent_grid_nearest(
    hp_map: np.ndarray,
    *,
    extent: tuple[float, float, float, float],
    step_deg: float,
    nest: bool,
) -> np.ndarray:
    hp_map = np.asarray(hp_map, dtype=np.float32)
    nside = hp.npix2nside(hp_map.shape[0])

    lon_min, lon_max, lat_min, lat_max = extent
    lats = np.arange(lat_min, lat_max, step_deg, dtype=np.float64) + 0.5 * step_deg
    lons = np.arange(lon_min, lon_max, step_deg, dtype=np.float64) + 0.5 * step_deg
    lats = lats[lats < lat_max]
    lons = lons[lons < lon_max]

    if lats.size < 2 or lons.size < 2:
        lats = np.linspace(lat_min, lat_max, 8, endpoint=False) + (lat_max - lat_min) / 16.0
        lons = np.linspace(lon_min, lon_max, 10, endpoint=False) + (lon_max - lon_min) / 20.0

    lon2d, lat2d = np.meshgrid(lons.astype(np.float32), lats.astype(np.float32))
    theta = _lat_to_theta_rad(lat2d)
    phi = _lon_to_phi_rad(lon2d)
    pix = hp.ang2pix(nside, theta, phi, nest=nest)
    return hp_map[pix].astype(np.float32)


def europe_step_deg_for_nside(nside: int, *, oversample: float) -> float:
    return max(0.02, (58.6 / max(int(nside), 1)) / max(float(oversample), 1e-6))


def decode_f32_zlib_b64(blob: dict) -> np.ndarray:
    shape = tuple(int(x) for x in blob["shape"])
    compressed = base64.b64decode(blob["data_b64_zlib"])
    raw = zlib.decompress(compressed)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
    return arr.astype(np.float32, copy=False)


def find_variables(input_root: Path) -> list[str]:
    vars_found: list[str] = []
    if not input_root.exists():
        return vars_found
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue
        json_dir = child / "json"
        if json_dir.is_dir() and any(json_dir.glob("t*_h*_levels.json")):
            vars_found.append(child.name)
    return vars_found


def index_json_files(json_dir: Path) -> dict[int, list[Path]]:
    by_time: dict[int, list[Path]] = {}
    for p in json_dir.glob("t*_h*_levels.json"):
        m = JSON_NAME_RE.match(p.name)
        if not m:
            continue
        t = int(m.group("t"))
        by_time.setdefault(t, []).append(p)
    for t in by_time:
        by_time[t].sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return by_time


def choose_file_for_time(by_time: dict[int, list[Path]], t_index: int) -> Path | None:
    files = by_time.get(t_index)
    if not files:
        return None
    return files[0]


def parse_variables_arg(raw: str | None, available: list[str]) -> list[str]:
    if raw is None or raw.strip() == "" or raw.strip().lower() == "all":
        return available
    requested = [tok.strip() for tok in raw.split(",") if tok.strip()]
    unknown = [v for v in requested if v not in available]
    if unknown:
        raise SystemExit(
            f"Unknown variable(s): {unknown}. Available variables: {available}"
        )
    return requested


def select_time_slots(
    *,
    requested_time: int | None,
    selected_vars: list[str],
    index_by_var: dict[str, dict[int, list[Path]]],
) -> dict[str, int]:
    chosen: dict[str, int] = {}

    if requested_time is not None:
        for var in selected_vars:
            if requested_time in index_by_var[var]:
                chosen[var] = requested_time
        return chosen

    if len(selected_vars) == 1:
        var = selected_vars[0]
        times = sorted(index_by_var[var].keys())
        if times:
            chosen[var] = times[-1]
        return chosen

    common_times: set[int] | None = None
    for var in selected_vars:
        times = set(index_by_var[var].keys())
        common_times = times if common_times is None else (common_times & times)
    if common_times:
        t_latest = max(common_times)
        return {var: t_latest for var in selected_vars}

    for var in selected_vars:
        times = sorted(index_by_var[var].keys())
        if times:
            chosen[var] = times[-1]
    return chosen


def load_payload(json_path: Path) -> dict:
    with json_path.open("r") as f:
        return json.load(f)


def extract_maps_for_nsides(payload: dict, nsides: list[int]) -> dict[int, np.ndarray | None]:
    out: dict[int, np.ndarray | None] = {}
    levels = payload.get("levels", {})
    for nside in nsides:
        entry = levels.get(str(nside))
        if not entry:
            out[nside] = None
            continue
        blob = entry.get("map")
        if not blob:
            out[nside] = None
            continue
        out[nside] = decode_f32_zlib_b64(blob)
    return out


def resolve_europe_extent(payload: dict, fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    meta = payload.get("meta", {})
    ext = meta.get("europe_extent")
    if isinstance(ext, list) and len(ext) == 4:
        return tuple(float(x) for x in ext)
    return fallback


def resolve_nest(payload: dict, fallback: bool) -> bool:
    meta = payload.get("meta", {})
    val = meta.get("nest")
    if isinstance(val, bool):
        return val
    return fallback


def build_matrix_figure(
    *,
    var: str,
    time_slot: int,
    payload: dict,
    nsides: list[int],
    global_nlat: int,
    zoom_oversample: float,
    europe_extent: tuple[float, float, float, float],
    default_nest: bool,
    separate_colorbars: bool,
    show_progress: bool,
) -> tuple[plt.Figure, str]:
    maps_by_nside = extract_maps_for_nsides(payload, nsides)
    cmap = cmap_for_var(var)
    nest = resolve_nest(payload, fallback=default_nest)
    extent = resolve_europe_extent(payload, fallback=europe_extent)
    lon_min, lon_max, lat_min, lat_max = extent

    prepared_by_nside: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None] = {}
    global_vals: list[np.ndarray] = []
    europe_vals: list[np.ndarray] = []
    for nside in nsides:
        arr = maps_by_nside[nside]
        if arr is None:
            prepared_by_nside[nside] = None
            continue
        grid_g, lon2d_g, lat2d_g = healpix_to_global_lonlat_grid_nearest(
            arr, nlat=max(global_nlat, 2 * nside), nest=nest
        )
        step = europe_step_deg_for_nside(nside, oversample=zoom_oversample)
        grid_e = healpix_to_extent_grid_nearest(
            arr, extent=extent, step_deg=step, nest=nest
        )
        prepared_by_nside[nside] = (grid_g, lon2d_g, lat2d_g, grid_e)

        finite_g = grid_g[np.isfinite(grid_g)]
        if finite_g.size:
            global_vals.append(finite_g)
        finite_e = grid_e[np.isfinite(grid_e)]
        if finite_e.size:
            europe_vals.append(finite_e)

    if not global_vals and not europe_vals:
        raise ValueError("No valid maps found in payload.")

    global_vmin, global_vmax = (
        empirical_vmin_vmax(np.concatenate(global_vals))
        if global_vals
        else empirical_vmin_vmax(np.concatenate(europe_vals))
    )
    europe_vmin, europe_vmax = (
        empirical_vmin_vmax(np.concatenate(europe_vals))
        if europe_vals
        else empirical_vmin_vmax(np.concatenate(global_vals))
    )
    shared_vmin, shared_vmax = empirical_vmin_vmax(
        np.concatenate(global_vals + europe_vals)
    )
    global_plot_vmin, global_plot_vmax = (
        (global_vmin, global_vmax)
        if separate_colorbars
        else (shared_vmin, shared_vmax)
    )
    europe_plot_vmin, europe_plot_vmax = (
        (europe_vmin, europe_vmax)
        if separate_colorbars
        else (shared_vmin, shared_vmax)
    )

    cbar_y = 0.012
    cbar_h = 0.012
    cbar_gap = 0.003

    fig = plt.figure(figsize=(12.8, 31.0))
    gs = fig.add_gridspec(
        nrows=len(nsides),
        ncols=2,
        hspace=0.04,
        wspace=0.03,
        top=0.985,
        bottom=cbar_y + cbar_h + cbar_gap,
    )
    mappable_global = None
    mappable_europe = None
    global_crs = ccrs.PlateCarree(central_longitude=GLOBAL_CENTRAL_LONGITUDE)
    global_proj = ccrs.Mollweide(central_longitude=GLOBAL_CENTRAL_LONGITUDE)

    row_iter = enumerate(nsides)
    if show_progress and tqdm is not None:
        row_iter = tqdm(
            row_iter,
            total=len(nsides),
            desc=f"{var}: rows",
            leave=False,
            dynamic_ncols=True,
        )

    for row_idx, nside in row_iter:
        prepared = prepared_by_nside[nside]

        ax_global = fig.add_subplot(gs[row_idx, 0], projection=global_proj)
        ax_global.set_global()
        ax_global.coastlines(linewidth=0.35)
        ax_global.add_feature(cfeature.BORDERS, linewidth=0.2)
        ax_global.set_axis_off()
        ax_europe = fig.add_subplot(gs[row_idx, 1], projection=ccrs.PlateCarree())
        ax_europe.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax_europe.coastlines(linewidth=0.35)
        ax_europe.add_feature(cfeature.BORDERS, linewidth=0.2)
        ax_europe.set_xticks([])
        ax_europe.set_yticks([])

        ax_global.text(
            0.02,
            0.98,
            panel_letter(2 * row_idx),
            transform=ax_global.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            zorder=5,
        )
        ax_europe.text(
            0.02,
            0.98,
            panel_letter(2 * row_idx + 1),
            transform=ax_europe.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            zorder=5,
        )
        ax_global.text(
            0.98,
            0.98,
            f"nside={nside}",
            transform=ax_global.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            zorder=5,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.2},
        )

        if prepared is None:
            ax_global.text(
                0.5, 0.5, "missing", transform=ax_global.transAxes, ha="center", va="center", fontsize=8
            )
            ax_europe.text(
                0.5, 0.5, "missing", transform=ax_europe.transAxes, ha="center", va="center", fontsize=8
            )
            continue

        grid_g, lon2d_g, lat2d_g, grid_e = prepared
        pm = ax_global.pcolormesh(
            lon2d_g,
            lat2d_g,
            grid_g,
            transform=global_crs,
            shading="nearest",
            cmap=cmap,
            vmin=global_plot_vmin,
            vmax=global_plot_vmax,
        )
        pm.set_rasterized(True)

        im = ax_europe.imshow(
            grid_e,
            origin="lower",
            extent=[lon_min, lon_max, lat_min, lat_max],
            transform=ccrs.PlateCarree(),
            interpolation="nearest",
            cmap=cmap,
            vmin=europe_plot_vmin,
            vmax=europe_plot_vmax,
            aspect="auto",
        )
        im.set_rasterized(True)

        if mappable_global is None:
            mappable_global = pm
        if mappable_europe is None:
            mappable_europe = im

        if row_idx == 0:
            ax_global.set_title("Global", fontsize=11)
            ax_europe.set_title("Europe", fontsize=11)

    if mappable_global is None or mappable_europe is None:
        raise ValueError("Could not create any map mappable from payload.")

    fig.suptitle(
        f"Resolution Synthesis | variable={var} | time slot t={time_slot:06d}",
        fontsize=13,
        y=0.995,
    )
    cbar_label = resolve_colorbar_label(var, payload)
    if separate_colorbars:
        cax_left = fig.add_axes([0.08, cbar_y, 0.39, cbar_h])
        cb_left = fig.colorbar(mappable_global, cax=cax_left, orientation="horizontal")
        cb_left.set_label(cbar_label, fontsize=9)
        cb_left.ax.tick_params(labelsize=8)

        cax_right = fig.add_axes([0.53, cbar_y, 0.39, cbar_h])
        cb_right = fig.colorbar(mappable_europe, cax=cax_right, orientation="horizontal")
        cb_right.set_label(cbar_label, fontsize=9)
        cb_right.ax.tick_params(labelsize=8)
    else:
        cax = fig.add_axes([0.14, cbar_y, 0.72, cbar_h])
        cb = fig.colorbar(mappable_global, cax=cax, orientation="horizontal")
        cb.set_label(cbar_label, fontsize=9)
        cb.ax.tick_params(labelsize=8)

    stem = f"1050_resolution_matrix_t{time_slot:06d}_{var}"
    return fig, stem


def load_run_meta(input_root: Path) -> dict:
    meta_path = input_root / "run_meta.json"
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def list_data(index_by_var: dict[str, dict[int, list[Path]]]) -> None:
    print("Available variables and time slots:")
    for var, by_time in sorted(index_by_var.items()):
        times = sorted(by_time.keys())
        if not times:
            continue
        print(f"  - {var}: {times[0]} ... {times[-1]} (n={len(times)})")


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    outdir = (args.outdir.expanduser().resolve() if args.outdir else (input_root / "resolution_sensitivity"))

    run_meta = load_run_meta(input_root)
    global_nlat = int(args.global_nlat if args.global_nlat is not None else run_meta.get("GLOBAL_NLAT", 360))
    zoom_oversample = float(
        args.zoom_oversample if args.zoom_oversample is not None else run_meta.get("ZOOM_OVERSAMPLE", 2.5)
    )
    europe_extent = (
        tuple(float(x) for x in args.europe_extent)
        if args.europe_extent is not None
        else tuple(float(x) for x in run_meta.get("EUROPE_EXTENT", DEFAULT_EUROPE_EXTENT))
    )

    available_vars = find_variables(input_root)
    if not available_vars:
        print(f"No variables found under {input_root}", file=sys.stderr)
        return 2

    selected_vars = parse_variables_arg(args.variable, available_vars)
    index_by_var = {
        var: index_json_files(input_root / var / "json")
        for var in selected_vars
    }

    if args.list:
        list_data(index_by_var)
        return 0

    chosen_times = select_time_slots(
        requested_time=args.time_slot,
        selected_vars=selected_vars,
        index_by_var=index_by_var,
    )
    if not chosen_times:
        print("Could not resolve any usable variable/time selection.", file=sys.stderr)
        return 2

    missing_for_requested_time = [
        var for var in selected_vars if args.time_slot is not None and var not in chosen_times
    ]
    if missing_for_requested_time:
        print(
            f"WARNING: requested time slot t={args.time_slot:06d} is missing for: {missing_for_requested_time}",
            file=sys.stderr,
        )

    output_formats = normalize_formats(args.formats, force_pdf=True)
    if "pdf" not in {fmt.lower().strip().lstrip(".") for fmt in args.formats if fmt.strip()}:
        print("NOTE: adding PDF output alongside requested formats.")

    outdir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    if args.progress and tqdm is None:
        print("NOTE: tqdm is not installed; running without progress bars.")

    var_iter = selected_vars
    if args.progress and tqdm is not None:
        var_iter = tqdm(
            selected_vars,
            total=len(selected_vars),
            desc="Variables",
            leave=True,
            dynamic_ncols=True,
        )

    for var in var_iter:
        t_idx = chosen_times.get(var)
        if t_idx is None:
            print(f"Skipping {var}: no available time slot.")
            continue
        json_path = choose_file_for_time(index_by_var[var], t_idx)
        if json_path is None:
            print(f"Skipping {var}: no json file for t={t_idx:06d}.")
            continue

        payload = load_payload(json_path)
        fig, stem = build_matrix_figure(
            var=var,
            time_slot=t_idx,
            payload=payload,
            nsides=NSIDES,
            global_nlat=global_nlat,
            zoom_oversample=zoom_oversample,
            europe_extent=europe_extent,
            default_nest=bool(run_meta.get("nest", False)),
            separate_colorbars=bool(args.separate_colorbars),
            show_progress=bool(args.progress),
        )

        for fmt in output_formats:
            out_path = outdir / f"{stem}.{fmt}"
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            saved_files.append(out_path)
        plt.close(fig)

        hash_note = "random h suffix in source filename"
        print(f"[{var}] t={t_idx:06d} <- {json_path.name} ({hash_note})")

    if not saved_files:
        print("No figures were generated.", file=sys.stderr)
        return 2

    print("\nSaved files:")
    for path in saved_files:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
