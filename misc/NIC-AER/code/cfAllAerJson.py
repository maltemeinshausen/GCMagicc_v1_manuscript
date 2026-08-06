#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run counterfactual sampling and plot ONLY tasmax mean effect (default: tasmax - tasmin),
save result to compact JSON, and write a high-quality PDF.

Adapted for Nxl (modelsNfour) architecture from model_Nxl_versA5.

Defaults requested:
- variable: tasmax only
- subtract_tasmin: True by default (so output is tasmax - tasmin)
- only mean effect (delta_mean) is produced (no std map)
- do NOT store counterfactuals by default (option to store means only)

Coastlines:
- If cartopy is installed and --coastlines is passed, we plot with cartopy Mollweide + coastlines.
- Otherwise we fall back to healpy.mollview (no true coastlines; can add graticule).
"""

import os
import math
import json
import base64
import zlib
import random
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
    """Store float32 arrays compactly inside JSON (zlib-compressed + base64)."""
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


# ----------------------------- plotting helpers ----------------------------- #
def _healpix_map_to_latlon_grid(
    hp_map: np.ndarray, nlat: int = 180, nsub: int = 3, order: str = "RING"
) -> np.ndarray:
    """
    Convert a single healpix map (P,) to a lat-lon grid (H=nlat, W=2*nlat)
    using the same sub-sampling interpolation logic you use elsewhere.
    """
    hp_map = np.asarray(hp_map, dtype=np.float32)
    if hp_map.ndim != 1:
        raise ValueError("hp_map must be 1D (NPIX,)")

    lats = 90 - (0.5 + np.arange(nlat)) / nlat * 180
    lons = (0.5 + np.arange(2 * nlat)) / nlat * 180

    NPIX = hp_map.shape[0]
    nside_local = hp.npix2nside(NPIX)
    if 12 * nside_local * nside_local != NPIX:
        raise ValueError(f"NPIX={NPIX} invalid (must be 12*NSIDE^2)")

    nest = order.upper() == "NEST"
    offs = (((np.arange(nsub) + 0.5) / nsub - 0.5) / nlat * 180)

    lat_sub = lats[:, None, None, None] + offs[None, None, :, None]
    lon_sub = lons[None, :, None, None] + offs[None, None, None, :]

    theta_sub = np.radians(90.0 - lat_sub)
    phi_sub = np.radians(lon_sub) + np.pi
    theta_sub, phi_sub = np.broadcast_arrays(theta_sub, phi_sub)

    theta_flat = theta_sub.ravel()
    phi_flat = phi_sub.ravel()

    vals = hp.get_interp_val(hp_map, theta_flat, phi_flat, nest=nest)
    vals = vals.reshape(nlat, 2 * nlat, nsub, nsub).mean(axis=(-1, -2))
    return vals.astype(np.float32)


def plot_delta_mean_pdf(
    hp_map: np.ndarray,
    out_pdf: str | Path,
    *,
    add_coastlines: bool = True,
    use_graticule_if_no_coastlines: bool = False,
    nlat_for_cartopy: int = 180,
    nsub_for_cartopy: int = 3,
    rot: float = 180.0,
):
    """
    Plot a healpix map as Mollweide PDF.
    - If add_coastlines and cartopy is available: plot grid + coastlines.
    - Else: healpy.mollview.
    """
    hp_map = np.asarray(hp_map, dtype=np.float32)
    v = float(np.nanmax(np.abs(hp_map)))
    if not np.isfinite(v) or v == 0.0:
        v = 1.0

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["pdf.compression"] = 9

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    if add_coastlines:
        try:
            import cartopy.crs as ccrs

            data_grid = _healpix_map_to_latlon_grid(
                hp_map, nlat=nlat_for_cartopy, nsub=nsub_for_cartopy
            )
            nlat = data_grid.shape[0]
            nlon = data_grid.shape[1]

            # lons in [0, 360); convert to [-180, 180)
            lons = (0.5 + np.arange(nlon)) / nlat * 180.0
            lons = (lons + 180.0) % 360.0 - 180.0
            lats = 90.0 - (0.5 + np.arange(nlat)) / nlat * 180.0

            lon2d, lat2d = np.meshgrid(lons, lats)

            fig = plt.figure(figsize=(10.5, 6.0))
            fig.patch.set_facecolor("white")

            ax = plt.axes(projection=ccrs.Mollweide(central_longitude=rot))
            ax.set_global()
            ax.set_facecolor("white")

            pm = ax.pcolormesh(
                lon2d,
                lat2d,
                data_grid,
                transform=ccrs.PlateCarree(),
                shading="auto",
                cmap=matplotlib.colormaps["coolwarm"].resampled(1024),
                vmin=-v,
                vmax=+v,
            )
            pm.set_rasterized(True)

            ax.coastlines(linewidth=0.6)

            cb = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.06, fraction=0.06)
            cb.ax.tick_params(labelsize=9)

            fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            return

        except ImportError:
            pass

    # ---- healpy fallback ----
    fig = plt.figure()
    fig.patch.set_facecolor("white")

    hp.mollview(
        hp_map,
        fig=fig.number,
        flip="geo",
        title="",
        xsize=4000,
        rot=rot,
        cmap=matplotlib.colormaps["coolwarm"].resampled(1024),
        min=-v,
        max=+v,
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

    if use_graticule_if_no_coastlines:
        try:
            hp.graticule(dpar=30, dmer=60, verbose=False)
        except TypeError:
            hp.graticule(dpar=30, dmer=60)

    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


# ----------------------------- sampler (Nxl / versA5) ----------------------------- #
def sample_Nxl(
    x,
    device=os.environ.get("DEVICE", "cuda:0"),
    dirname=os.environ.get("MODEL_DIR", "../checkpoints/modelsA/"),
    DATE="7Augext",
    dependence=False,
    nside=64,
    rectangular=True,
    nlat=180,
    nsub=1,
    asnumpy=True,
    usebias_model=None,
    useeffect_model=None,
    seed=None,
    *,
    nsim: int = 1,
    reduce: str | None = None,
):
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
        return (yhat + yhat.mean(dim=2, keepdim=True) - tmp.mean(dim=2, keepdim=True)).clamp(
            min=mi, max=ma
        )

    def healpix_to_latlon_grid(hp_cube, nlat, *, order="RING", nsub=3):
        lats = 90 - (0.5 + np.arange(nlat)) / nlat * 180
        lons = (0.5 + np.arange(2 * nlat)) / nlat * 180
        if hp_cube.ndim != 3:
            raise ValueError("hp_cube must have shape (N, C, NPIX)")
        N, C, NPIX = hp_cube.shape
        nside_local = hp.npix2nside(NPIX)
        if 12 * nside_local * nside_local != NPIX:
            raise ValueError(f"NPIX={NPIX} invalid (must be 12*NSIDE^2)")
        nest = order.upper() == "NEST"
        offs = (((np.arange(nsub) + 0.5) / nsub - 0.5) / nlat * 180)
        lat_sub = lats[:, None, None, None] + offs[None, None, :, None]
        lon_sub = lons[None, :, None, None] + offs[None, None, None, :]
        theta_sub = np.radians(90.0 - lat_sub)
        phi_sub = np.radians(lon_sub) + np.pi
        theta_sub, phi_sub = np.broadcast_arrays(theta_sub, phi_sub)
        theta_flat = theta_sub.ravel()
        phi_flat = phi_sub.ravel()
        maps = hp_cube.reshape(-1, NPIX)
        vals = hp.get_interp_val(maps, theta_flat, phi_flat, nest=nest)
        vals = vals.reshape(N, C, nlat, 2 * nlat, nsub, nsub)
        vals = vals.mean(axis=(-1, -2))
        return vals

    def _reduce_over_time(y: torch.Tensor, how: str):
        if how is None:
            return y
        if how == "mean":
            y_f = y if y.is_floating_point() else y.float()
            return y_f.mean(dim=0, keepdim=False)
        if how == "max":
            return y.max(dim=0, keepdim=False).values
        if how == "min":
            return y.min(dim=0, keepdim=False).values
        if how == "max-mean":
            y_f = y if y.is_floating_point() else y.float()
            ymax = y.max(dim=0, keepdim=False).values
            ymean = y_f.mean(dim=0, keepdim=False)
            return ymax - ymean
        if how == "maxtas":
            if y.ndim not in (2, 3):
                raise ValueError("y must be (T,C) or (T,C,P) for 'maxtas'")
            if y.size(1) <= 1:
                raise ValueError("need C>=2 for 'maxtas'")
            idx = torch.argmax(y, dim=0)
            if y.ndim == 3:
                ref = idx[1, :]
                dif = idx - ref.unsqueeze(0)
            else:
                ref = idx[1]
                dif = idx - ref
            mask = (dif.abs() <= 12)
            out_dtype = y.dtype if y.is_floating_point() else torch.float32
            return mask.to(out_dtype)
        if how in ("driest", "wettest"):
            if y.ndim not in (2, 3):
                raise ValueError("y must be (T,C) or (T,C,P) for 'driest'/'wettest'")
            if y.size(1) <= 2:
                raise ValueError("need C>=3; uses channel index 2 as 'pr'")
            if y.ndim == 3:
                pr = y[:, 1, :]
                tstar = pr.argmin(dim=0) if how == "driest" else pr.argmax(dim=0)
                idx = tstar.view(1, 1, -1).expand(1, y.size(1), y.size(2))
                out = torch.gather(y, dim=0, index=idx).squeeze(0)
            else:
                pr = y[:, 1]
                tstar = pr.argmin(dim=0) if how == "driest" else pr.argmax(dim=0)
                out = y[tstar, :]
            return out
        raise ValueError(
            "reduce must be None, 'max', 'mean', 'min', 'max-mean', 'maxtas', 'driest', or 'wettest'"
        )

    if dependence:
        x = torch.cat([x[:12].repeat(12, 1), x], dim=0)

    x = x.to(dev, non_blocking=True)
    n_features = len(variables)
    x_features = x.shape[1]

    xbias = x.clone()
    if usebias_model is not None:
        xbias[:, 0] = usebias_model
    if useeffect_model is not None:
        x[:, 0] = useeffect_model

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # --- Nxl model config (versA5): starts at nside=2 ---
    levels = [2, 4, 8, 16, 32, 64]
    if nside > 64:
        levels.append(128)
    if nside > 128:
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

    def _one_sim(sim_seed_offset: int | None):
        if seed is not None and sim_seed_offset is not None:
            s = seed + sim_seed_offset
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

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
                _free_level(mbias, mcore)

            vars_order = ["psl", "tas", "pr", "sfcWind", "ts", "tasmin", "tasmax", "rsds", "hurs", "huss"]
            vars_order = [v for v in vars_order if v in variables]
            pre = torch.tensor(
                [transformation_scalars[var]["pre"] for var in vars_order],
                dtype=torch.float32,
                device=y.device,
            )
            post = torch.tensor(
                [transformation_scalars[var]["post"] for var in vars_order],
                dtype=torch.float32,
                device=y.device,
            )

            yh_obs = (y - post[None, :, None]) * (1.0 / pre[None, :, None])

            if dependence:
                yh_obs = yh_obs[144:]

            if reduce is not None:
                yh_obs = _reduce_over_time(yh_obs, reduce)

            return yh_obs

    sims = []
    for i in range(nsim):
        sims.append(_one_sim(i if nsim > 1 else None))

    def _to_numpy(t):
        return t if isinstance(t, np.ndarray) else t.to("cpu").numpy()

    if reduce is None:
        out = sims[0] if nsim == 1 else torch.stack(sims, dim=0)
    else:
        out = sims[0] if nsim == 1 else torch.stack(sims, dim=0)

    if rectangular:
        arr = _to_numpy(out)
        if reduce is None:
            if arr.ndim == 3:
                arr_rect = healpix_to_latlon_grid(arr, nlat, nsub=nsub)
            else:
                ns, T, C, P = arr.shape
                tmp = healpix_to_latlon_grid(arr.reshape(ns * T, C, P), nlat, nsub=nsub)
                H, W = tmp.shape[-2:]
                arr_rect = tmp.reshape(ns, T, C, H, W)
            return arr_rect if asnumpy else torch.from_numpy(arr_rect)
        else:
            if arr.ndim == 2:
                arr_rect = healpix_to_latlon_grid(arr[None, ...], nlat, nsub=nsub)[0]
            else:
                arr_rect = healpix_to_latlon_grid(arr, nlat, nsub=nsub)
            return arr_rect if asnumpy else torch.from_numpy(arr_rect)

    return _to_numpy(out) if asnumpy else out


# ----------------------------- main driver ----------------------------- #
def main():
    # === USER-REQUESTED DEFAULTS ===
    TARGET_VAR = "tasmax"
    SUBTRACT_TASMIN_DEFAULT = True
    STORE_COUNTERFACTUALS_DEFAULT = False
    ADD_COASTLINES_DEFAULT = False

    DATE = "7Augext"

    # Release: env-configurable input path (see EXTERNAL_MANIFEST.csv).
    _data_dir = os.environ.get(
        "DATA_DIR",
        os.environ.get("SCRATCH_BASE", "/scratch2")
        + "/userdata/nicolai/cmip6/normalized_" + DATE)
    H5_PATH = os.environ.get(
        "H5_PATH",
        _data_dir
        + "/data_DAT_ERA5_historical-ERA5_r1i1p1f1_clt-day-hurs-huss-month-pr-psl-rlut-rsds-rsdt-rsnt-rtmt-sfcWind-tas-tasmax-tasmin-ts-year.nc.h5"
    )

    MODELS = ["Nxl"]

    dependence = False
    reduce = "mean"

    rectangular = False
    device = os.environ.get("DEVICE", "cuda:0")
    nsim = 100
    nside = 64
    years = 10

    COFS = ["13"]
    MC_RANGE = range(0, 20)

    subtract_tasmin = SUBTRACT_TASMIN_DEFAULT
    store_counterfactuals = STORE_COUNTERFACTUALS_DEFAULT
    add_coastlines = ADD_COASTLINES_DEFAULT

    variables_X = [
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

    variables = ["psl", "tas", "pr", "sfcWind", "ts", "tasmin", "tasmax", "rsds", "hurs", "huss"]
    vars_order = [v for v in ["psl", "tas", "pr", "sfcWind", "ts", "tasmin", "tasmax", "rsds", "hurs", "huss"] if v in variables]

    if TARGET_VAR not in vars_order:
        raise ValueError(f"{TARGET_VAR} not found in vars_order={vars_order}")

    # meta lives in ../config/ in the release bundle; fall back to the script dir,
    # the model dir, or a META_PATH override.
    _md = os.environ.get("MODEL_DIR", "../checkpoints/modelsA/")
    _cands = [os.environ.get("META_PATH", ""),
              f"meta_{DATE}.pkl",
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", f"meta_{DATE}.pkl"),
              os.path.join(_md, f"meta_{DATE}.pkl")]
    meta_path = next((c for c in _cands if c and os.path.exists(c)), f"meta_{DATE}.pkl")
    with open(meta_path, "rb") as f:
        meta_data = pickle.load(f)
    transformation_scalars = meta_data["transformation_scalars"]

    with h5py.File(H5_PATH, "r") as h5f:
        x_np = h5f["X"][:]
        y_obs = h5f["y64"][:]

    x_tensor = torch.from_numpy(x_np).float()
    yo = torch.from_numpy(y_obs).float()

    pre = torch.tensor([transformation_scalars[var]["pre"] for var in vars_order], dtype=torch.float32)
    inv_pre = 1.0 / pre
    post_constant = torch.tensor([transformation_scalars[var]["post"] for var in vars_order], dtype=torch.float32)

    y = (yo - post_constant[None, :, None]) * inv_pre[None, :, None]
    _ = y.mean(dim=0)

    tasmax_idx = vars_order.index("tasmax")
    tasmin_idx = vars_order.index("tasmin") if "tasmin" in vars_order else None
    if subtract_tasmin and tasmin_idx is None:
        raise ValueError("subtract_tasmin=True but tasmin not present.")

    for cof in COFS:
        for mc in MC_RANGE:
            for model in MODELS:
                if cof == "nowpast":
                    x_cf = x_tensor.clone()
                    x_actual = x_tensor.clone()
                    x_cf[:, 4:] = x_tensor[0, 4:]
                    x_actual[:, 4:] = x_actual[-1, 4:]
                    x_cf = x_cf[0 : (years * 12), :]
                    x_actual = x_actual[0 : (years * 12), :]
                else:
                    chvar = int(cof)
                    x_cf = x_tensor.clone()
                    x_actual = x_tensor.clone()
                    x_cf[:, 4:] = x_tensor[-1, 4:]
                    x_actual[:, 4:] = x_actual[-1, 4:]
                    x_cf[:, chvar] = 0
                    x_cf = x_cf[-(years * 12) :, :]
                    x_actual = x_actual[-(years * 12) :, :]

                x_cf[:, 0] = mc
                x_actual[:, 0] = mc

                yh = sample_Nxl(
                    x_actual,
                    device=device,
                    nsim=nsim,
                    reduce=reduce,
                    dependence=dependence,
                    nside=nside,
                    rectangular=rectangular,
                    nlat=180,
                    nsub=3,
                    asnumpy=False,
                )
                yh = yh.cpu()
                torch.cuda.empty_cache()

                ycf = sample_Nxl(
                    x_cf,
                    device=device,
                    nsim=nsim,
                    reduce=reduce,
                    dependence=dependence,
                    nside=nside,
                    rectangular=rectangular,
                    nlat=180,
                    nsub=3,
                    asnumpy=False,
                )
                ycf = ycf.cpu()
                torch.cuda.empty_cache()

                if yh.ndim != 3:
                    raise RuntimeError(f"Expected yh to be (nsim,C,P), got {tuple(yh.shape)}")
                if ycf.shape != yh.shape:
                    raise RuntimeError(f"Shape mismatch: yh={tuple(yh.shape)} ycf={tuple(ycf.shape)}")

                if subtract_tasmin:
                    yh = yh - yh[:, tasmin_idx : tasmin_idx + 1, :]
                    ycf = ycf - ycf[:, tasmin_idx : tasmin_idx + 1, :]

                yh_v = yh[:, tasmax_idx, :]
                ycf_v = ycf[:, tasmax_idx, :]

                delta_mean = (yh_v.mean(dim=0) - ycf_v.mean(dim=0)).numpy().astype(np.float32)

                yh_mean = yh_v.mean(dim=0).numpy().astype(np.float32)
                ycf_mean = ycf_v.mean(dim=0).numpy().astype(np.float32)

                outdir = Path(f"DAER3_{TARGET_VAR}_{reduce}")
                if cof == "nowpast":
                    outname = f"{cof}_{mc}_{model}"
                else:
                    outname = f"{variables_X[int(cof)]}_{mc}_{model}"

                outdir.mkdir(parents=True, exist_ok=True)

                payload = {
                    "meta": {
                        "DATE": DATE,
                        "model": model,
                        "cof": cof,
                        "mc": int(mc),
                        "reduce": reduce,
                        "dependence": bool(dependence),
                        "nsim": int(nsim),
                        "nside": int(nside),
                        "variable": TARGET_VAR,
                        "subtract_tasmin": bool(subtract_tasmin),
                        "store_counterfactuals": bool(store_counterfactuals),
                        "v_absmax": float(np.nanmax(np.abs(delta_mean))) if np.size(delta_mean) else None,
                        "note": "Arrays are float32, zlib-compressed and base64-encoded inside JSON.",
                    },
                    "delta_mean": _encode_f32_zlib_b64(delta_mean),
                }

                if store_counterfactuals:
                    payload["yh_mean"] = _encode_f32_zlib_b64(yh_mean)
                    payload["ycf_mean"] = _encode_f32_zlib_b64(ycf_mean)

                json_path = outdir / f"{outname}_{TARGET_VAR}.json"
                _write_json(json_path, payload)
                print(f"Wrote {json_path}")

                pdf_path = outdir / f"{outname}_{TARGET_VAR}_delta_mean.pdf"
                plot_delta_mean_pdf(
                    delta_mean,
                    pdf_path,
                    add_coastlines=add_coastlines,
                    use_graticule_if_no_coastlines=False,
                    nlat_for_cartopy=180,
                    nsub_for_cartopy=3,
                    rot=180.0,
                )
                print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
