#!/usr/bin/env python3
"""
1070_aerosol_pattern
===========================

Create a 3xn matrix of Mollweide aerosol-pattern maps from
`aer_ERF_*_T1new_tasmax.json` files with:
- coastlines on each panel,
- small bold panel letters (a, b, c, ...),
- normal-weight model name labels,
- one shared colorbar across all panels.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import re
import string
import zlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

try:
    import healpy as hp
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "healpy is required for this script. Use the project environment."
    ) from exc

try:
    import cartopy.crs as ccrs
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "cartopy is required for this script (coastline plotting)."
    ) from exc


AER_JSON_RE = re.compile(r"^aer_ERF_(?P<mc>\d+)_T1new_tasmax\.json$")


@dataclass
class AerosolMapEntry:
    mc: int
    model_name: str
    hp_map: np.ndarray


def panel_letter(idx: int) -> str:
    letters = string.ascii_lowercase
    if idx < len(letters):
        return letters[idx]
    q, r = divmod(idx, len(letters))
    return f"{letters[q - 1]}{letters[r]}"


def decode_f32_zlib_b64(obj: dict) -> np.ndarray:
    comp = base64.b64decode(obj["data_b64_zlib"].encode("ascii"))
    raw = zlib.decompress(comp)
    return np.frombuffer(raw, dtype=np.float32).reshape(obj["shape"])


def _get_map(payload: dict, field: str) -> np.ndarray:
    if field in payload:
        arr = decode_f32_zlib_b64(payload[field])
        if arr.ndim != 1:
            raise ValueError(f"Expected '{field}' to be 1D (P,), got {arr.shape}")
        return arr.astype(np.float32, copy=False)

    if field == "yh_mean" and "yh_tasmax" in payload:
        yh = decode_f32_zlib_b64(payload["yh_tasmax"])
        if yh.ndim != 2:
            raise ValueError(f"Legacy 'yh_tasmax' expected 2D (nsim,P), got {yh.shape}")
        return yh.mean(axis=0).astype(np.float32)

    if field == "ycf_mean" and "ycf_tasmax" in payload:
        ycf = decode_f32_zlib_b64(payload["ycf_tasmax"])
        if ycf.ndim != 2:
            raise ValueError(f"Legacy 'ycf_tasmax' expected 2D (nsim,P), got {ycf.shape}")
        return ycf.mean(axis=0).astype(np.float32)

    raise KeyError(
        f"Field '{field}' not found in JSON. Available keys: {sorted(payload.keys())}"
    )


def _healpix_map_to_latlon_grid(
    hp_map: np.ndarray, *, nlat: int = 180, nsub: int = 3, order: str = "RING"
) -> np.ndarray:
    hp_map = np.asarray(hp_map, dtype=np.float32)
    if hp_map.ndim != 1:
        raise ValueError("hp_map must be 1D (NPIX,)")

    npix = hp_map.shape[0]
    nside = hp.npix2nside(npix)
    if 12 * nside * nside != npix:
        raise ValueError(f"NPIX={npix} is not a valid HEALPix size (12*nside^2)")

    lats = 90.0 - (0.5 + np.arange(nlat)) / nlat * 180.0
    lons = -180.0 + (0.5 + np.arange(2 * nlat)) / nlat * 180.0

    offs = (((np.arange(nsub) + 0.5) / nsub - 0.5) / nlat * 180.0)
    lat_sub = lats[:, None, None, None] + offs[None, None, :, None]
    lon_sub = lons[None, :, None, None] + offs[None, None, None, :]

    theta_sub = np.radians(90.0 - lat_sub)
    phi_sub = np.radians((lon_sub + 180.0) % 360.0)
    theta_sub, phi_sub = np.broadcast_arrays(theta_sub, phi_sub)

    nest = order.upper() == "NEST"
    vals = hp.get_interp_val(hp_map, theta_sub.ravel(), phi_sub.ravel(), nest=nest)
    vals = vals.reshape(nlat, 2 * nlat, nsub, nsub).mean(axis=(-1, -2))
    return vals.astype(np.float32)


def _resolve_model_csv_candidates(repo_root: Path) -> list[Path]:
    return [
        repo_root / "data" / "nicolaiplots" / "plots" / "plots_emergent_constraint" / "UniqueModels_10Jun.csv",
        repo_root.parent / "gcm_firefly_data" / "model_NthreeversT1" / "modelsA" / "UniqueModels_10Jun.csv",
    ]


def _load_model_index_map(model_csv: Path | None, repo_root: Path) -> dict[int, str]:
    candidates: list[Path] = []
    if model_csv is not None:
        candidates.append(model_csv)
    candidates.extend(_resolve_model_csv_candidates(repo_root))

    for candidate in candidates:
        if not candidate.exists():
            continue
        mapping: dict[int, str] = {}
        with candidate.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    idx = int(row["model_index"])
                    name = str(row["source_id"]).strip()
                except (KeyError, ValueError, TypeError):
                    continue
                if name:
                    mapping[idx] = name
        if mapping:
            return mapping
    return {}


def _discover_json_files(json_dir: Path) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in sorted(json_dir.glob("aer_ERF_*_T1new_tasmax.json")):
        match = AER_JSON_RE.match(path.name)
        if not match:
            continue
        files.append((int(match.group("mc")), path))
    return sorted(files, key=lambda x: x[0])


def _load_entries(
    *,
    json_dir: Path,
    field: str,
    model_names: dict[int, str],
) -> list[AerosolMapEntry]:
    entries: list[AerosolMapEntry] = []
    for mc_from_name, path in _discover_json_files(json_dir):
        with path.open("r") as f:
            payload = json.load(f)

        meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
        mc = int(meta.get("mc", mc_from_name))
        hp_map = _get_map(payload, field=field)
        model_name = model_names.get(mc, f"model_index {mc}")
        entries.append(AerosolMapEntry(mc=mc, model_name=model_name, hp_map=hp_map))

    entries.sort(key=lambda e: e.mc)
    return entries


def plot_aerosol_pattern_matrix(
    *,
    json_dir: Path,
    out_path: Path,
    model_csv: Path | None = None,
    field: str = "delta_mean",
    nrows: int = 3,
    nlat_for_cartopy: int = 180,
    nsub_for_cartopy: int = 3,
    central_longitude: float = 0.0,
    cmap_name: str = "RdBu_r",
    cbar_label: str = "Change in diurnal temperature range ? in period ?? (K ?)",
    dpi: int = 220,
) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    model_names = _load_model_index_map(model_csv=model_csv, repo_root=repo_root)
    entries = _load_entries(json_dir=json_dir, field=field, model_names=model_names)
    if not entries:
        raise FileNotFoundError(
            f"No matching JSON files found in {json_dir} "
            f"(expected aer_ERF_*_T1new_tasmax.json)."
        )

    nrows = max(1, int(nrows))
    ncols = int(math.ceil(len(entries) / nrows))

    vabs = 0.0
    for entry in entries:
        this_v = float(np.nanmax(np.abs(entry.hp_map)))
        if np.isfinite(this_v):
            vabs = max(vabs, this_v)
    if vabs == 0.0:
        vabs = 1.0

    cmap = matplotlib.colormaps[cmap_name].resampled(1024)
    norm = colors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=+vabs)

    fig_w = max(4.0 * ncols, 10.0)
    fig_h = max(2.8 * nrows + 1.0, 7.0)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        subplot_kw={"projection": ccrs.Mollweide(central_longitude=central_longitude)},
        constrained_layout=False,
    )
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        axes = axes.reshape(nrows, ncols)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.11, wspace=0.02, hspace=0.10)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(entries):
            ax.set_visible(False)
            continue

        entry = entries[idx]
        data_grid = _healpix_map_to_latlon_grid(
            entry.hp_map,
            nlat=nlat_for_cartopy,
            nsub=nsub_for_cartopy,
            order="RING",
        )

        nlat = data_grid.shape[0]
        nlon = data_grid.shape[1]
        lons = -180.0 + (0.5 + np.arange(nlon)) * (360.0 / nlon)
        lats = 90.0 - (0.5 + np.arange(nlat)) / nlat * 180.0
        lon2d, lat2d = np.meshgrid(lons, lats)

        pm = ax.pcolormesh(
            lon2d,
            lat2d,
            data_grid,
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        pm.set_rasterized(True)

        ax.set_global()
        ax.coastlines(linewidth=0.45, color="black")

        label = panel_letter(idx)
        ax.text(
            0.012,
            1.01,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            clip_on=False,
        )
        ax.text(
            0.045 + 0.016 * max(len(label) - 1, 0),
            1.01,
            entry.model_name,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            fontweight="normal",
            clip_on=False,
        )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=[ax for ax in axes.flat if ax.get_visible()],
        orientation="horizontal",
        fraction=0.022,
        pad=0.055,
        aspect=80,
    )
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(cbar_label, fontsize=9)

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["pdf.compression"] = 9

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_json_dir = (
        repo_root
        / "data"
        / "nicolaiplots"
        / "plots"
        / "plots_aerosol"
        / "DAER3_tasmax_mean"
    )

    parser = argparse.ArgumentParser(
        description="Create 3xn aerosol Mollweide matrix with one shared colorbar."
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=default_json_dir,
        help="Directory containing aer_ERF_*_T1new_tasmax.json files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output figure path. Default: <json-dir>/aer_ERF_matrix_3xn_shared_cbar.pdf",
    )
    parser.add_argument(
        "--model-csv",
        type=Path,
        default=None,
        help="Optional path to UniqueModels_10Jun.csv for model-index labels.",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="delta_mean",
        choices=["delta_mean", "yh_mean", "ycf_mean", "delta_std"],
        help="JSON field to plot.",
    )
    parser.add_argument("--nrows", type=int, default=3, help="Number of subplot rows (default: 3).")
    parser.add_argument("--nlat", type=int, default=180, help="Latitude bins for HEALPix regridding.")
    parser.add_argument("--nsub", type=int, default=3, help="Subsampling factor for HEALPix interpolation.")
    parser.add_argument(
        "--central-longitude",
        type=float,
        default=0.0,
        help="Mollweide central longitude in degrees (default: 0, i.e. -180 to +180).",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Output raster DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out
    if out_path is None:
        out_path = args.json_dir / "aer_ERF_matrix_3xn_shared_cbar.pdf"

    written = plot_aerosol_pattern_matrix(
        json_dir=args.json_dir,
        out_path=out_path,
        model_csv=args.model_csv,
        field=args.field,
        nrows=args.nrows,
        nlat_for_cartopy=args.nlat,
        nsub_for_cartopy=args.nsub,
        central_longitude=args.central_longitude,
        dpi=args.dpi,
    )
    print(f"Wrote {written}")


if __name__ == "__main__":
    main()
