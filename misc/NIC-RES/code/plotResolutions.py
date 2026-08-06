#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RESOLUTIONPLOTS: Multilevel snapshot plotting (Nxl / versA5)

Adapted from gcm_Nthree to use the modelsNfour architecture from model_Nxl_versA5.

Fixes for "oddities":
1) Global plot looked too smooth at low NSIDE:
   - We now render maps using *nearest-neighbor* sampling from Healpix via hp.ang2pix
     (no hp.get_interp_val, no sub-sampling, no averaging).
   - In the global cartopy Mollweide plot we also force pcolormesh shading='nearest'.

2) Europe zoom looked same resolution independent of NSIDE:
   - Europe zoom grid resolution now *depends on NSIDE* (finer grid for higher NSIDE),
     again with nearest-neighbor Healpix sampling.

Other settings:
- Color scale: per-map empirical min/max (not symmetric around 0)
- Colormap: coolwarm for tas, tasmax, tasmin, ts; viridis for all others
- Default NSIDE_MAX=256
- Outputs in RESOLUTIONPLOTS/<var>/{pdf,json}/
- Filenames include time index + NSIDE + hash 0..10

Coastlines:
- If cartopy installed, coastlines + borders are drawn for global/europe/inset.
- If cartopy missing, only global healpy mollview is produced; europe/inset skipped.
"""

import os
import math
import json
import base64
import zlib
from pathlib import Path

import numpy as np
import torch
import healpy as hp
import h5py
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Model classes (Nxl: no M1 or BiasSum needed)
from models import LB2, L4de, L4adde, LB2mini, L4addemini


# ----------------------------- JSON helpers ----------------------------- #
def _encode_f32_zlib_b64(arr: np.ndarray) -> dict:
    """Store float32 arrays compactly in JSON (zlib-compressed + base64)."""
    arr = np.asarray(arr, dtype=np.float32, order="C")
    raw = arr.tobytes(order="C")
    comp = zlib.compress(raw, level=9)
    return {
        "dtype": "float32",
        "shape": list(arr.shape),
        "data_b64_zlib": base64.b64encode(comp).decode("ascii"),
    }


def _write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


# ----------------------------- Colormap / range ----------------------------- #
_COOLWARM_VARS = {"tas", "tasmax", "tasmin", "ts"}


def cmap_for_var(var: str):
    name = "coolwarm" if var in _COOLWARM_VARS else "viridis"
    return matplotlib.colormaps[name].resampled(1024)  # mpl>=3.6 (get_cmap removed in 3.9)


def empirical_vmin_vmax(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float32)
    m = np.isfinite(x)
    if not np.any(m):
        return -1.0, 1.0
    vmin = float(np.min(x[m]))
    vmax = float(np.max(x[m]))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return -1.0, 1.0
    if vmin == vmax:
        eps = 1e-6 if vmin == 0.0 else abs(vmin) * 1e-6
        vmin -= eps
        vmax += eps
    return vmin, vmax


# ----------------------------- Healpix -> grids (NEAREST, no smoothing) ----------------------------- #
def _lon_to_phi_rad(lon_deg: np.ndarray) -> np.ndarray:
    """lon in degrees in [-180,180) -> phi in [0, 2pi)."""
    return np.deg2rad(lon_deg) + np.pi


def _lat_to_theta_rad(lat_deg: np.ndarray) -> np.ndarray:
    """lat in degrees -> theta in [0,pi], colatitude."""
    return np.deg2rad(90.0 - lat_deg)


def healpix_to_global_lonlat_grid_nearest(
    hp_map: np.ndarray,
    *,
    nlat: int = 180,
    nest: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Global regular grid (lat x lon) sampled from Healpix via *nearest neighbor* (hp.ang2pix).
    Returns:
      grid (nlat, 2*nlat), lon2d, lat2d
    lon2d is in [-180,180).
    """
    hp_map = np.asarray(hp_map, dtype=np.float32)
    if hp_map.ndim != 1:
        raise ValueError("hp_map must be 1D (NPIX,)")

    npix = hp_map.shape[0]
    nside = hp.npix2nside(npix)

    lats = 90.0 - (0.5 + np.arange(nlat)) / nlat * 180.0
    lons = (0.5 + np.arange(2 * nlat)) / nlat * 360.0  # [0,360)
    lons = (lons + 180.0) % 360.0 - 180.0              # [-180,180)

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
    nest: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extent grid sampled from Healpix via *nearest neighbor* (hp.ang2pix).
    Returns:
      grid (Ny, Nx), lon2d, lat2d
    """
    hp_map = np.asarray(hp_map, dtype=np.float32)
    if hp_map.ndim != 1:
        raise ValueError("hp_map must be 1D (NPIX,)")

    lonmin, lonmax, latmin, latmax = extent
    if not (lonmin < lonmax and latmin < latmax):
        raise ValueError("extent must satisfy lonmin<lonmax and latmin<latmax")

    npix = hp_map.shape[0]
    nside = hp.npix2nside(npix)

    step_deg = float(step_deg)
    if step_deg <= 0:
        raise ValueError("step_deg must be > 0")

    lats = np.arange(latmin, latmax, step_deg, dtype=np.float64) + 0.5 * step_deg
    lons = np.arange(lonmin, lonmax, step_deg, dtype=np.float64) + 0.5 * step_deg
    lats = lats[lats < latmax]
    lons = lons[lons < lonmax]

    if lats.size < 2 or lons.size < 2:
        lats = np.linspace(latmin, latmax, 8, endpoint=False) + (latmax - latmin) / 16.0
        lons = np.linspace(lonmin, lonmax, 10, endpoint=False) + (lonmax - lonmin) / 20.0

    lon2d, lat2d = np.meshgrid(lons.astype(np.float32), lats.astype(np.float32))
    theta = _lat_to_theta_rad(lat2d)
    phi = _lon_to_phi_rad(lon2d)

    pix = hp.ang2pix(nside, theta, phi, nest=nest)
    grid = hp_map[pix].astype(np.float32)
    return grid, lon2d.astype(np.float32), lat2d.astype(np.float32)


def europe_step_deg_for_nside(nside: int, *, oversample: float = 1.5) -> float:
    """
    Approx healpix pixel angular scale ~ 58.6 / nside degrees.
    We use step = (58.6/nside)/oversample to ensure higher nside yields more detail.
    """
    nside = int(nside)
    oversample = float(oversample)
    return max(0.02, (58.6 / max(nside, 1)) / max(oversample, 1e-6))


# ----------------------------- Plotting ----------------------------- #
def _set_pdf_quality_defaults():
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["pdf.compression"] = 9


def plot_global_pdf(
    hp_map: np.ndarray,
    out_pdf: str | Path,
    *,
    var: str,
    rot: float = 180.0,
    box_extent: tuple[float, float, float, float] | None = None,
    add_coastlines: bool = True,
    nlat_global: int = 180,
    nest: bool = False,
) -> bool:
    """
    Global Mollweide plot. Uses nearest-neighbor sampling from Healpix (no smoothing).
    Returns True if plotted with cartopy, False if fell back to healpy.
    """
    _set_pdf_quality_defaults()
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    hp_map = np.asarray(hp_map, dtype=np.float32)
    vmin, vmax = empirical_vmin_vmax(hp_map)
    cmap = cmap_for_var(var)

    try:
        import cartopy.crs as ccrs

        grid, lon2d, lat2d = healpix_to_global_lonlat_grid_nearest(
            hp_map, nlat=nlat_global, nest=nest
        )

        fig = plt.figure(figsize=(10.5, 6.0))
        fig.patch.set_facecolor("white")
        ax = plt.axes(projection=ccrs.Mollweide(central_longitude=rot))
        ax.set_global()
        ax.set_facecolor("white")

        pm = ax.pcolormesh(
            lon2d, lat2d, grid,
            transform=ccrs.PlateCarree(),
            shading="nearest",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
        )
        pm.set_rasterized(True)

        if add_coastlines:
            ax.coastlines(linewidth=0.6)

        if box_extent is not None:
            lonmin, lonmax, latmin, latmax = box_extent
            xs = [lonmin, lonmax, lonmax, lonmin, lonmin]
            ys = [latmin, latmin, latmax, latmax, latmin]
            ax.plot(xs, ys, transform=ccrs.PlateCarree(), linewidth=1.0)

        cb = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.06, fraction=0.06)
        cb.ax.tick_params(labelsize=9)

        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return True

    except ImportError:
        # healpy fallback (no coastlines)
        fig = plt.figure()
        fig.patch.set_facecolor("white")
        hp.mollview(
            hp_map,
            fig=fig.number,
            flip="geo",
            title="",
            xsize=4000,
            rot=rot,
            cmap=cmap,
            min=vmin,
            max=vmax,
            bgcolor="white",
        )
        # Remove blue background outside the Mollweide ellipse
        for ax in fig.axes:
            ax.set_facecolor("white")
            try:
                ax.patch.set_facecolor("white")
                ax.patch.set_alpha(1.0)
            except Exception:
                pass
        try:
            hp.graticule(dpar=30, dmer=60, verbose=False)
        except TypeError:
            hp.graticule(dpar=30, dmer=60)

        plt.savefig(out_pdf, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        return False


def plot_zoom_europe_pdf(
    hp_map: np.ndarray,
    out_pdf: str | Path,
    *,
    var: str,
    extent: tuple[float, float, float, float],
    nside: int,
    add_coastlines: bool = True,
    zoom_oversample: float = 1.5,
    nest: bool = False,
) -> None:
    """
    Europe zoom plot showing actual Healpix structure via nearest-neighbor sampling.
    The zoom grid resolution scales with nside.
    Requires cartopy.
    """
    _set_pdf_quality_defaults()
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    hp_map = np.asarray(hp_map, dtype=np.float32)
    cmap = cmap_for_var(var)

    step = europe_step_deg_for_nside(nside, oversample=zoom_oversample)
    grid, lon2d, lat2d = healpix_to_extent_grid_nearest(
        hp_map, extent=extent, step_deg=step, nest=nest
    )

    vmin, vmax = empirical_vmin_vmax(grid)

    fig = plt.figure(figsize=(8.2, 6.0))
    fig.patch.set_facecolor("white")
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_facecolor("white")

    lonmin, lonmax, latmin, latmax = extent
    im = ax.imshow(
        grid,
        origin="lower",
        extent=[lonmin, lonmax, latmin, latmax],
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    im.set_rasterized(True)

    if add_coastlines:
        ax.coastlines(linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)

    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.06, fraction=0.06)
    cb.ax.tick_params(labelsize=9)

    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_global_with_inset_pdf(
    hp_map: np.ndarray,
    out_pdf: str | Path,
    *,
    var: str,
    rot: float = 180.0,
    extent: tuple[float, float, float, float],
    nside: int,
    add_coastlines: bool = True,
    nlat_global: int = 180,
    zoom_oversample: float = 1.5,
    nest: bool = False,
) -> None:
    """
    Global Mollweide (nearest sampled) + bottom-right inset (nearest sampled at nside-dependent res).
    Requires cartopy.
    """
    _set_pdf_quality_defaults()
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    hp_map = np.asarray(hp_map, dtype=np.float32)
    vmin, vmax = empirical_vmin_vmax(hp_map)
    cmap = cmap_for_var(var)

    grid_g, lon2d_g, lat2d_g = healpix_to_global_lonlat_grid_nearest(
        hp_map, nlat=nlat_global, nest=nest
    )

    step = europe_step_deg_for_nside(nside, oversample=zoom_oversample)
    grid_z, _, _ = healpix_to_extent_grid_nearest(
        hp_map, extent=extent, step_deg=step, nest=nest
    )

    fig = plt.figure(figsize=(10.5, 6.0))
    fig.patch.set_facecolor("white")
    ax = plt.axes(projection=ccrs.Mollweide(central_longitude=rot))
    ax.set_global()
    ax.set_facecolor("white")

    pm = ax.pcolormesh(
        lon2d_g, lat2d_g, grid_g,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        cmap=cmap,
        vmin=vmin, vmax=vmax,
    )
    pm.set_rasterized(True)

    if add_coastlines:
        ax.coastlines(linewidth=0.6)

    lonmin, lonmax, latmin, latmax = extent
    xs = [lonmin, lonmax, lonmax, lonmin, lonmin]
    ys = [latmin, latmin, latmax, latmax, latmin]
    ax.plot(xs, ys, transform=ccrs.PlateCarree(), linewidth=1.0)

    cb = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.06, fraction=0.06)
    cb.ax.tick_params(labelsize=9)

    # inset axes
    inset = fig.add_axes([0.64, 0.07, 0.33, 0.33], projection=ccrs.PlateCarree())
    inset.set_extent(extent, crs=ccrs.PlateCarree())
    inset.set_facecolor("white")

    im = inset.imshow(
        grid_z,
        origin="lower",
        extent=[lonmin, lonmax, latmin, latmax],
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    im.set_rasterized(True)

    if add_coastlines:
        inset.coastlines(linewidth=0.6)
        inset.add_feature(cfeature.BORDERS, linewidth=0.35)

    for spine in inset.spines.values():
        spine.set_linewidth(0.8)

    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------- Nxl sampler with per-level outputs ----------------------------- #
def sample_Nxl_levels(
    x: torch.Tensor,
    *,
    device: str,
    dirname: str,
    DATE: str,
    nside_max: int,
    dependence: bool = False,
    seed: int | None = None,
) -> tuple[dict[int, torch.Tensor], list[str]]:
    """
    Generate outputs for ALL intermediate resolution levels up to nside_max.

    Returns:
      levels_out: dict {level_nside: y_obs_level} with y_obs_level shape (T, C, P_level)
      vars_order: channel order list
    """
    dev = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    dirm = os.path.join(dirname, "modelsNfour_")

    def _prep_model(model, ckpt_dir):
        sd = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location="cpu")
        model.load_state_dict(sd)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    meta_filename = os.path.join(dirname, f"meta_{DATE}.pkl")
    with open(meta_filename, "rb") as f:
        meta_data = pickle.load(f)
    transformation_scalars = meta_data["transformation_scalars"]
    variables = meta_data["variables"]

    ranges = torch.load(os.path.join(dirname, f"ranges_{DATE}.pt"), map_location="cpu")
    y_min = ranges["y_min"]
    y_max = ranges["y_max"]

    def cla(yhat: torch.Tensor) -> torch.Tensor:
        nside_here = int(math.sqrt(yhat.shape[2] // 12))
        mi = y_min[str(nside_here)].to(yhat.device)
        ma = y_max[str(nside_here)].to(yhat.device)
        tmp = yhat.clamp(min=mi, max=ma)
        return (yhat + yhat.mean(dim=2, keepdim=True) - tmp.mean(dim=2, keepdim=True)).clamp(min=mi, max=ma)

    vars_order = ["psl", "tas", "pr", "sfcWind", "ts", "tasmin", "tasmax", "rsds", "hurs", "huss"]
    vars_order = [v for v in vars_order if v in variables]

    pre = torch.tensor([transformation_scalars[var]["pre"] for var in vars_order], dtype=torch.float32, device=dev)
    post = torch.tensor([transformation_scalars[var]["post"] for var in vars_order], dtype=torch.float32, device=dev)
    inv_pre = 1.0 / pre

    def _latent_to_obs(y_lat: torch.Tensor) -> torch.Tensor:
        return (y_lat - post[None, :, None]) * inv_pre[None, :, None]

    if seed is not None:
        import random as _py_random
        _py_random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if dependence:
        x = torch.cat([x[:12].repeat(12, 1), x], dim=0)

    x = x.to(dev, non_blocking=True)
    n_features = len(variables)
    x_features = x.shape[1]
    xbias = x.clone()

    # --- Nxl model config (versA5): starts at nside=2 ---
    levels = [2, 4, 8, 16, 32, 64]
    if nside_max > 64:
        levels.append(128)
    if nside_max > 128:
        levels.append(256)

    def _load_level(nside_hi: int):
        nside_lo = nside_hi // 2

        # --- bias model ---
        if nside_hi == 2:
            modelbias = _prep_model(
                LB2(0, nside_hi, n_features=n_features),
                dirm + DATE + f"_bS{nside_hi}",
            )
        elif nside_hi in (4, 8, 16, 32, 64):
            modelbias = _prep_model(
                LB2(nside_lo, nside_hi, n_features=n_features),
                dirm + DATE + f"_b{nside_hi}",
            )
        else:  # 128, 256
            modelbias = _prep_model(
                LB2mini(nside_lo, nside_hi, n_features=n_features),
                dirm + DATE + f"_b{nside_hi}",
            )

        # --- effect model ---
        if nside_hi == 2:
            maxlag = 72
            variab = 10
            lags = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 27, 30, 33, 36, 42, 48, 54, 60, 66, 72]
            modelcore = _prep_model(
                L4de(nside_hi, n_features=n_features, x_features=x_features,
                     variab=variab, lags=lags, add_latent_dim=50),
                dirm + DATE + f"_bdeSxlsp{nside_hi}_{maxlag}_{variab}",
            )
        elif nside_hi == 4:
            maxlag = 6
            variab = 0
            lags = [i for i in range(1, maxlag)]
            modelcore = _prep_model(
                L4de(nside_hi, n_features=n_features, x_features=x_features,
                     variab=variab, lags=lags),
                dirm + DATE + f"_bde{nside_hi}_{maxlag}_{variab}",
            )
        elif nside_hi == 8:
            maxlag = 2
            variab = 0
            lags = [i for i in range(1, maxlag)]
            modelcore = _prep_model(
                L4de(nside_hi, n_features=n_features, x_features=x_features,
                     variab=variab, lags=lags),
                dirm + DATE + f"_bde{nside_hi}_{maxlag}_{variab}",
            )
        elif nside_hi in (16, 32, 64):
            modelcore = _prep_model(
                L4adde(nside_hi, n_features=n_features, x_features=x_features),
                dirm + DATE + f"_bdadde{nside_hi}",
            )
        else:  # 128, 256
            modelcore = _prep_model(
                L4addemini(nside_hi, n_features=n_features, x_features=x_features),
                dirm + DATE + f"_bdadde{nside_hi}",
            )

        modelbias = modelbias.to(dev, non_blocking=True)
        modelcore = modelcore.to(dev, non_blocking=True)
        return modelbias, modelcore

    def _free_level(*mods):
        for m in mods:
            try:
                m.cpu()
            except Exception:
                pass
            del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    levels_out: dict[int, torch.Tensor] = {}

    with torch.inference_mode():
        y = None
        for lvl in levels:
            mbias, mcore = _load_level(lvl)

            if lvl == 2:
                y = cla(mbias(x=xbias, y_low=None).contiguous())
                y = cla(mcore(y_in=y, x=x, dependence=dependence).contiguous())
            elif lvl in (4, 8):
                y = cla(mbias(y_low=y, x=xbias).contiguous())
                y = cla(mcore(y_in=y, x=x, dependence=dependence).contiguous())
            elif lvl in (16, 32, 64):
                y = cla(mbias(y_low=y, x=xbias).contiguous())
                y = cla(mcore(y_in=y, x=x).contiguous())
            elif lvl in (128, 256):
                y = mbias(y_low=y, x=xbias).contiguous()
                y = mcore(y_in=y, x=x).contiguous()
            else:
                raise RuntimeError(f"Unexpected level {lvl}")

            y_obs = _latent_to_obs(y)
            if dependence:
                y_obs = y_obs[144:]
            levels_out[lvl] = y_obs.detach().cpu()
            _free_level(mbias, mcore)

    return levels_out, vars_order


# ----------------------------- Main driver ----------------------------- #
def main():
    # ===================== defaults (edit here) ===================== #
    DATE = "7Augext"
    # Release: paths are env-configurable so the bundle reproduces without the
    # original checkout. DATA_DIR should hold the normalized data_DAT_*.h5 files
    # (see EXTERNAL_MANIFEST.csv); MODEL_DIR holds the A5 checkpoints.
    _data_dir = os.environ.get(
        "DATA_DIR",
        os.environ.get("SCRATCH_BASE", "/scratch2")
        + "/userdata/nicolai/cmip6/normalized_" + DATE)
    H5_PATH = os.environ.get(
        "H5_PATH",
        _data_dir
        + "/data_DAT_ERA5_historical-ERA5_r1i1p1f1_clt-day-hurs-huss-month-pr-psl-rlut-rsds-rsdt-rsnt-rtmt-sfcWind-tas-tasmax-tasmin-ts-year.nc.h5"
    )

    MODEL_DIR = os.environ.get("MODEL_DIR", "../checkpoints/modelsA/")
    DEVICE = os.environ.get("DEVICE", "cuda:0")

    # Record paths in output metadata relative to $SCRATCH_BASE so regenerated
    # JSONs stay environment-agnostic and reproduce byte-for-byte across machines.
    _SB = os.environ.get("SCRATCH_BASE")
    def _tidy_path(p):
        if _SB and isinstance(p, str) and p.startswith(_SB):
            return "${SCRATCH_BASE}" + p[len(_SB):]
        return p

    DEPENDENCE = False
    NSIDE_MAX = 256

    YEARS_TAIL = 10
    K_TIMEPOINTS = 10
    SEED_TIMEPOINTS = 123
    MC = 0

    OUT_ROOT = Path("RESOLUTIONPLOTS")

    EUROPE_EXTENT = (-8.15, 23.15, 36.15, 55.85)

    ADD_COASTLINES = True

    GLOBAL_NLAT = 360

    ZOOM_OVERSAMPLE = 2.5

    NEST = False
    # ================================================================ #

    with h5py.File(H5_PATH, "r") as h5f:
        x_np = h5f["X"][:]
    x_full = torch.from_numpy(x_np).float()
    T_full = x_full.shape[0]

    tail_len = min(YEARS_TAIL * 12, T_full)
    start_idx = T_full - tail_len
    x_tail = x_full[start_idx:].clone()
    x_tail[:, 0] = float(MC)

    rng_tp = np.random.default_rng(SEED_TIMEPOINTS)
    K = min(K_TIMEPOINTS, tail_len)
    chosen_local = np.sort(rng_tp.choice(tail_len, size=K, replace=False))
    chosen_global = (chosen_local + start_idx).astype(int)

    x_sel = x_tail[chosen_local].clone()

    rng_file = np.random.default_rng()

    levels_out, vars_order = sample_Nxl_levels(
        x_sel,
        device=DEVICE,
        dirname=MODEL_DIR,
        DATE=DATE,
        nside_max=NSIDE_MAX,
        dependence=DEPENDENCE,
        seed=SEED_TIMEPOINTS,
    )

    try:
        import cartopy  # noqa: F401
        CARTOPY_OK = True
    except ImportError:
        CARTOPY_OK = False
        print("WARNING: cartopy not available. Only global PDFs via healpy will be produced. Zoom/inset skipped.")

    for var in vars_order:
        (OUT_ROOT / var / "pdf").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / var / "json").mkdir(parents=True, exist_ok=True)

    for j in range(K):
        t_global = int(chosen_global[j])
        x_row = x_sel[j].detach().cpu().numpy().astype(np.float32)

        for vi, var in enumerate(vars_order):
            var_dir_pdf = OUT_ROOT / var / "pdf"
            var_dir_json = OUT_ROOT / var / "json"

            payload = {
                "meta": {
                    "DATE": DATE,
                    "MODEL": "Nxl",
                    "MODEL_DIR": _tidy_path(MODEL_DIR),
                    "H5_PATH": _tidy_path(H5_PATH),
                    "DEVICE": DEVICE,
                    "DEPENDENCE": bool(DEPENDENCE),
                    "NSIDE_MAX": int(NSIDE_MAX),
                    "MC": int(MC),
                    "t_index_global": t_global,
                    "t_index_local_in_tail": int(chosen_local[j]),
                    "years_tail": int(YEARS_TAIL),
                    "seed_timepoints": int(SEED_TIMEPOINTS),
                    "variable": var,
                    "channel_index_in_vars_order": int(vi),
                    "vars_order": vars_order,
                    "x_row": x_row.tolist(),
                    "europe_extent": list(EUROPE_EXTENT),
                    "colormap": "coolwarm" if var in _COOLWARM_VARS else "viridis",
                    "vmin_vmax": "per-map empirical min/max",
                    "healpix_sampling": "nearest (hp.ang2pix), no interpolation",
                    "global_grid_nlat": int(GLOBAL_NLAT),
                    "zoom_step_rule_deg": f"(58.6/nside)/{ZOOM_OVERSAMPLE}",
                    "nest": bool(NEST),
                },
                "levels": {},
            }

            for lvl, y_lvl in levels_out.items():
                hp_map = y_lvl[j, vi, :].detach().cpu().numpy().astype(np.float32)

                payload["levels"][str(lvl)] = {
                    "nside": int(lvl),
                    "npix": int(hp_map.shape[0]),
                    "map": _encode_f32_zlib_b64(hp_map),
                }

                h = int(rng_file.integers(0, 11))
                base = f"t{t_global:06d}_nside{lvl}_h{h}"

                # (a) global
                pdf_global = var_dir_pdf / f"{base}_global.pdf"
                _ = plot_global_pdf(
                    hp_map,
                    pdf_global,
                    var=var,
                    rot=180.0,
                    box_extent=None,
                    add_coastlines=ADD_COASTLINES and CARTOPY_OK,
                    nlat_global=max(GLOBAL_NLAT, int(2*lvl)),
                    nest=NEST,
                )

                if CARTOPY_OK:
                    # (b) europe zoom
                    pdf_eu = var_dir_pdf / f"{base}_europe.pdf"
                    plot_zoom_europe_pdf(
                        hp_map,
                        pdf_eu,
                        var=var,
                        extent=EUROPE_EXTENT,
                        nside=int(lvl),
                        add_coastlines=ADD_COASTLINES,
                        zoom_oversample=ZOOM_OVERSAMPLE,
                        nest=NEST,
                    )

            hjson = int(rng_file.integers(0, 11))
            json_path = var_dir_json / f"t{t_global:06d}_h{hjson}_levels.json"
            _write_json(json_path, payload)
            print(f"Wrote {json_path}")

    run_meta = {
        "OUT_ROOT": str(OUT_ROOT),
        "DATE": DATE,
        "H5_PATH": _tidy_path(H5_PATH),
        "MODEL_DIR": _tidy_path(MODEL_DIR),
        "DEVICE": DEVICE,
        "DEPENDENCE": bool(DEPENDENCE),
        "NSIDE_MAX": int(NSIDE_MAX),
        "YEARS_TAIL": int(YEARS_TAIL),
        "K_TIMEPOINTS": int(K),
        "SEED_TIMEPOINTS": int(SEED_TIMEPOINTS),
        "MC": int(MC),
        "chosen_time_indices_global": chosen_global.tolist(),
        "chosen_time_indices_local_in_tail": chosen_local.tolist(),
        "vars_order": vars_order,
        "EUROPE_EXTENT": list(EUROPE_EXTENT),
        "cartopy_available": CARTOPY_OK,
        "coolwarm_vars": sorted(list(_COOLWARM_VARS)),
        "other_vars_colormap": "viridis",
        "vmin_vmax_rule": "per-map empirical min/max",
        "healpix_sampling": "nearest (hp.ang2pix)",
        "GLOBAL_NLAT": int(GLOBAL_NLAT),
        "ZOOM_OVERSAMPLE": float(ZOOM_OVERSAMPLE),
        "nest": bool(NEST),
    }
    _write_json(OUT_ROOT / "run_meta.json", run_meta)
    print(f"Wrote {OUT_ROOT / 'run_meta.json'}")


if __name__ == "__main__":
    main()
