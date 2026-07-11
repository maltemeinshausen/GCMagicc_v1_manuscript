import os
import math
import random
import numpy as np
import torch
import pickle
import healpy as hp
from models import LB2, L4de, L4adde, LB2mini, L4addemini

def sample_from_combined_model(
    x: torch.Tensor,
    device=None,                      # can be None, str, or torch.device
    dirname: str = './modelsA/',
    DATE: str = '7Augext',
    dependence: bool = False,
    nside: int = 64,
    rectangular: bool = True,
    nlat: int = 180,
    nsub: int = 1,
    asnumpy: bool = True,
    usebias_model=None,
    useeffect_model=None,
    seed=None,
    dtype: torch.dtype = torch.float32,
    memoryoptimized: bool | None = None,
    enable_gpu_optimizations: bool | None = None,
):
    """
    Unified sampler with:
    - Device selection, GPU optimizations, and memory-optimized loading as in the
      original `sample_from_combined_model`.
    - Model architecture and checkpoint selection exactly as in `sampleA5`
      (using the `modelsNfour_` directory and the same suffixes/hyperparams).

    Arguments and return behaviour are identical to the original
    `sample_from_combined_model`.
    """

    # -------------------- GPU optimisation helper --------------------
    def _setup_gpu_optimizations(dev: torch.device) -> None:
        """
        Enable TF32 and basic allocator / device tweaks for CUDA devices.
        """
        if dev.type != 'cuda' or not torch.cuda.is_available():
            return
        try:
            if hasattr(torch.backends, 'cuda'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.allow_tf32 = True

            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            if dev.index is not None:
                torch.cuda.set_device(dev.index)
        except Exception:
            # Best-effort only.
            pass

    # -------------------- automatic device selection --------------------
    def _auto_select_device() -> torch.device:
        """
        If CUDA is available, choose the visible GPU with the most free memory.
        Otherwise, return CPU.
        """
        if not torch.cuda.is_available():
            return torch.device('cpu')

        try:
            num_devices = torch.cuda.device_count()
        except Exception:
            return torch.device('cuda:0')

        if num_devices == 0:
            return torch.device('cpu')

        best_idx = 0
        best_free = -1
        for idx in range(num_devices):
            try:
                free, total = torch.cuda.mem_get_info(idx)
            except Exception:
                best_idx = 0
                break
            if free > best_free:
                best_free = free
                best_idx = idx

        return torch.device(f'cuda:{best_idx}')

    # -------------------- device handling --------------------
    if device is None:
        dev = _auto_select_device()
    else:
        requested_dev = torch.device(device)
        if requested_dev.type == 'cuda' and not torch.cuda.is_available():
            dev = torch.device('cpu')
        else:
            dev = requested_dev

    if memoryoptimized is None:
        memoryoptimized = (dev.type == 'cuda')

    if enable_gpu_optimizations is None:
        enable_gpu_optimizations = (dev.type == 'cuda')

    if enable_gpu_optimizations:
        _setup_gpu_optimizations(dev)

    # IMPORTANT: use the *sampleA5* directory
    dirm = os.path.join(dirname, 'modelsNfour_')

    # -------------------- helpers ---------------------------
    def _prep_model(model, ckpt_dir: str):
        sd = torch.load(os.path.join(ckpt_dir, 'best_model.pt'), map_location='cpu')
        model.load_state_dict(sd)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model  # moved to device + dtype later

    meta_filename = os.path.join(dirname, f'meta_{DATE}.pkl')
    with open(meta_filename, "rb") as f:
        meta_data = pickle.load(f)
    transformation_scalars = meta_data["transformation_scalars"]
    variables = meta_data["variables"]

    ranges = torch.load(os.path.join(dirname, f'ranges_{DATE}.pt'), map_location='cpu')
    y_min = ranges['y_min']
    y_max = ranges['y_max']

    def cla(yhat: torch.Tensor) -> torch.Tensor:
        """
        Clamp-and-recentre (same numerics as original combined sampler,
        but using y_min/y_max as in sampleA5).
        """
        nside_here = int(math.sqrt(yhat.shape[2] // 12))
        mi = y_min[str(nside_here)].to(yhat.device, dtype=yhat.dtype)
        ma = y_max[str(nside_here)].to(yhat.device, dtype=yhat.dtype)
        tmp = yhat.clamp(min=mi, max=ma)
        return (yhat + yhat.mean(dim=2, keepdim=True) - tmp.mean(dim=2, keepdim=True)).clamp(min=mi, max=ma)

    def healpix_to_latlon_grid(hp_cube, nlat, *, order="RING", nsub=3):
        """
        Convert (N, C, NPIX) HEALPix maps to a regular lat-lon grid (N, C, nlat, 2*nlat),
        with simple sub-pixel averaging.
        """
        lats = 90 - (0.5 + np.arange(nlat)) / nlat * 180
        lons = (0.5 + np.arange(2 * nlat)) / nlat * 180

        if hp_cube.ndim != 3:
            raise ValueError("hp_cube must have shape (N, C, NPIX)")
        N, C, NPIX = hp_cube.shape

        nside_local = hp.npix2nside(NPIX)
        if 12 * nside_local * nside_local != NPIX:
            raise ValueError(f"NPIX={NPIX} invalid (must be 12×NSIDE²)")

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

    # -------------------- dependence & inputs -----------------------------
    if dependence:
        x = torch.cat([x[:12].repeat(12, 1), x], dim=0)

    if not torch.is_floating_point(x):
        x = x.float()
    x = x.to(dev, dtype=dtype, non_blocking=True)

    n_features = len(variables)
    x_features = x.shape[1]

    xbias = x.clone()
    if usebias_model is not None:
        xbias[:, 0] = usebias_model
    if useeffect_model is not None:
        x[:, 0] = useeffect_model

    # -------------------- global seeding ---------------------------------
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # -------------------- NSIDE levels (as in sampleA5) -------------------
    # sampleA5 always runs 2,4,8,16,32,64 and optionally 128,256
    levels = [2, 4, 8, 16, 32, 64]
    if nside > 64:
        levels.append(128)
    if nside > 128:
        levels.append(256)

    def _load_level(nside_hi: int):
        """
        Load and construct models for a given NSIDE level, following
        *exactly* the choices in sampleA5 (LB2/L4de/L4adde/LB2mini/L4addemini
        and checkpoint naming).
        """
        # ----- NSIDE = 2: special LB2 (nside_lo=0) and L4de with custom lags -----
        if nside_hi == 2:
            nside_lo = 0
            # LB2 bias model: modelsNfour_7Augext_bS2
            modelbias = _prep_model(
                LB2(nside_lo, nside_hi, n_features=n_features),
                dirm + DATE + f'_bS{nside_hi}'
            )

            # L4de core: modelsNfour_7Augext_bdeSxlsp2_72_10
            maxlag = 72
            variab = 10
            lags = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20,
                    24, 27, 30, 33, 36, 42, 48, 54, 60, 66, 72]
            sufflow = '_bdeSxlsp'
            modelcore = _prep_model(
                L4de(
                    nside_hi,
                    n_features=n_features,
                    x_features=x_features,
                    variab=variab,
                    lags=lags,
                    add_latent_dim=50,
                ),
                dirm + DATE + f'{sufflow}{nside_hi}_{maxlag}_{variab}'
            )

        # ----- NSIDE = 4: LB2 + L4de (maxlag=6, variab=0, '_bde4_6_0') -----
        elif nside_hi == 4:
            nside_lo = nside_hi // 2
            modelbias = _prep_model(
                LB2(nside_lo, nside_hi, n_features=n_features),
                dirm + DATE + f'_b{nside_hi}'
            )
            maxlag = 6
            variab = 0
            sufflow = '_bde'
            lags = [i for i in range(1, maxlag)]
            modelcore = _prep_model(
                L4de(
                    nside_hi,
                    n_features=n_features,
                    x_features=x_features,
                    variab=variab,
                    lags=lags,
                ),
                dirm + DATE + f'{sufflow}{nside_hi}_{maxlag}_{variab}'
            )

        # ----- NSIDE = 8: LB2 + L4de (maxlag=2, variab=0, '_bde8_2_0') -----
        elif nside_hi == 8:
            nside_lo = nside_hi // 2
            modelbias = _prep_model(
                LB2(nside_lo, nside_hi, n_features=n_features),
                dirm + DATE + f'_b{nside_hi}'
            )
            maxlag = 2
            variab = 0
            sufflow = '_bde'
            lags = [i for i in range(1, maxlag)]
            modelcore = _prep_model(
                L4de(
                    nside_hi,
                    n_features=n_features,
                    x_features=x_features,
                    variab=variab,
                    lags=lags,
                ),
                dirm + DATE + f'{sufflow}{nside_hi}_{maxlag}_{variab}'
            )

        # ----- NSIDE = 16,32,64: LB2 + L4adde ('_b{nside}' and '_bdadde{nside}') -----
        elif nside_hi in (16, 32, 64):
            nside_lo = nside_hi // 2
            modelbias = _prep_model(
                LB2(nside_lo, nside_hi, n_features=n_features),
                dirm + DATE + f'_b{nside_hi}'
            )
            modelcore = _prep_model(
                L4adde(
                    nside_hi,
                    n_features=n_features,
                    x_features=x_features,
                ),
                dirm + DATE + f'_bdadde{nside_hi}'
            )

        # ----- NSIDE = 128,256: LB2mini + L4addemini (as in sampleA5) -----
        elif nside_hi in (128, 256):
            nside_lo = nside_hi // 2
            modelbias = _prep_model(
                LB2mini(nside_lo, nside_hi, n_features=n_features),
                dirm + DATE + f'_b{nside_hi}'
            )
            modelcore = _prep_model(
                L4addemini(
                    nside_hi,
                    n_features=n_features,
                    x_features=x_features,
                ),
                dirm + DATE + f'_bdadde{nside_hi}'
            )
        else:
            raise ValueError(f"Unsupported NSIDE level {nside_hi} for sampleA5-style models")

        # Move to device + dtype here so both memory modes share same path
        modelbias = modelbias.to(dev, dtype=dtype, non_blocking=True)
        modelcore = modelcore.to(dev, dtype=dtype, non_blocking=True)
        return modelbias, modelcore

    def _free_level(*mods):
        """Free models for memory-optimised mode."""
        for m in mods:
            try:
                m.cpu()
            except Exception:
                pass
            del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Pre-load all levels if we are not in memory-optimised mode.
    persistent_models = None
    if not memoryoptimized:
        persistent_models = {}
        for lvl in levels:
            persistent_models[lvl] = _load_level(lvl)

    # -------------------- multi-resolution inference ---------------------
    with torch.inference_mode():
        y = None
        for lvl in levels:
            if memoryoptimized:
                mbias, mcore = _load_level(lvl)
            else:
                mbias, mcore = persistent_models[lvl]

            # EXACT chaining as in sampleA5:
            # y_hat2 -> y_hat4 -> y_hat8 -> y_hat16 -> y_hat32 -> y_hat64 -> [128][256]

            if lvl == 2:
                # NSIDE=2: LB2(y_low=None) + L4de with dependence
                y = cla(mbias(y_low=None, x=xbias).contiguous())
                y = cla(mcore(y_in=y, x=x, dependence=dependence).contiguous())

            elif lvl in (4, 8):
                # NSIDE=4,8: LB2 + L4de with dependence, starting from previous y
                y = cla(mbias(y_low=y, x=xbias).contiguous())
                y = cla(mcore(y_in=y, x=x, dependence=dependence).contiguous())

            elif lvl in (16, 32, 64):
                # NSIDE=16,32,64: LB2 + L4adde, no dependence flag in core
                y = cla(mbias(y_low=y, x=xbias).contiguous())
                y = cla(mcore(y_in=y, x=x).contiguous())

            elif lvl in (128, 256):
                # NSIDE=128,256: LB2mini + L4addemini; no cla between these in sampleA5
                y = mbias(y_low=y, x=xbias).contiguous()
                y = mcore(y_in=y, x=x).contiguous()

            else:
                raise RuntimeError(f"Unexpected NSIDE level {lvl}")

            if memoryoptimized:
                _free_level(mbias, mcore)

        # --------- transform back to observation space --------------------
        vars_order = [
            "psl", "tas", "pr", "sfcWind", "ts",
            "tasmin", "tasmax", "rsds", "hurs", "huss",
        ]
        vars_order = [v for v in vars_order if v in variables]

        pre = torch.tensor(
            [transformation_scalars[var]["pre"] for var in vars_order],
            dtype=dtype,
            device=y.device,
        )
        post = torch.tensor(
            [transformation_scalars[var]["post"] for var in vars_order],
            dtype=dtype,
            device=y.device,
        )

        inv_pre = 1.0 / pre
        yh_obs = (y - post[None, :, None]) * inv_pre[None, :, None]

        if dependence:
            yh_obs = yh_obs[144:]

    # -------------------- rectangular projection & output -----------------
    def _to_numpy(t):
        return t if isinstance(t, np.ndarray) else t.to('cpu').numpy()

    if rectangular:
        arr = _to_numpy(yh_obs)  # (T, C, P)
        arr_rect = healpix_to_latlon_grid(arr, nlat, nsub=nsub)  # (T, C, H, W)
        return arr_rect if asnumpy else torch.from_numpy(arr_rect)

    return _to_numpy(yh_obs) if asnumpy else yh_obs
