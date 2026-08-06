import os
import numpy as np
import torch
import healpy as hp
import torch.nn.functional as F
from help_functions import DownsampleWithNoise
from train_functions import BiasSum
from models import LB2, L4de, L4adde, LB2mini, L4addemini, L4des  # keep as in your file
import pickle
import random
import math
import re
import h5py
import matplotlib.pyplot as plt
import csv


def sample_from_combined_model(
    x: torch.Tensor,
    device=None,                      # can be None, str, or torch.device
    dirname: str = './modelsA/',
    DATE: str = '7Augext',
    dependence: bool = False,
    variables=None,                   # None | list[int] | list[str]
    nside: int = 64,                  # API compatibility; this simplified version runs ONLY NSIDE=1
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
    # --------- optional parameters (applied to NSIDE=1) ----------
    maxlag: int | None = 1,
    variab: int | None = 10,
    add_latent_dim: int | None = 50,
    # --------- NEW ----------
    useT: bool = False,               # if True: LB2 -> M1 -> L4de
):
    """
    Simplified sampler: runs ONLY the NSIDE=1 level (no multi-level chain).

    Robust handling of `variables`:
      - variables=None        -> use meta_data["variables"] in canonical order
      - variables=[0,1,2]     -> interpret as indices into meta_data["variables"]
      - variables=["tas","pr"]-> interpret as names (must be in meta_data["variables"])

    NEW:
      - useT=False (default): identical behaviour as before: LB2 -> L4de
      - useT=True: LB2 -> M1 -> L4de  (i.e., sample y via LB2, refine with M1, then feed into L4de)
    """

    # Optional local import to make useT work even if caller didn't import M1 into globals
    try:
        from models import LB2 as _LB2, L4de as _L4de, M1 as _M1
    except Exception:
        # fallback to globals; note M1 must exist in globals in that case
        _LB2, _L4de, _M1 = LB2, L4de, M1

    # -------------------- GPU optimisation helper --------------------
    def _setup_gpu_optimizations(dev: torch.device) -> None:
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
            pass

    # -------------------- automatic device selection --------------------
    def _auto_select_device() -> torch.device:
        if not torch.cuda.is_available():
            return torch.device('cpu')
        try:
            num_devices = torch.cuda.device_count()
        except Exception:
            return torch.device('cuda:0')
        if num_devices == 0:
            return torch.device('cpu')
        best_idx, best_free = 0, -1
        for idx in range(num_devices):
            try:
                free, _total = torch.cuda.mem_get_info(idx)
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

    # A3 uses modelsNfour_
    dirm = os.path.join(dirname, 'modelsNfour_')

    # -------------------- helpers ---------------------------
    def _prep_model(model, ckpt_dir: str):
        ckpt = os.path.join(ckpt_dir, 'best_model.pt')
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        sd = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(sd)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    # -------------------- metadata --------------------------
    meta_filename = os.path.join("./", f'meta_{DATE}.pkl')
    with open(meta_filename, "rb") as f:
        meta_data = pickle.load(f)

    transformation_scalars = meta_data["transformation_scalars"]
    all_vars = list(meta_data["variables"])  # canonical name order (e.g. length 10)

    def _resolve_vars(meta_vars, variables_arg):
        if variables_arg is None:
            idx = list(range(len(meta_vars)))
            names = list(meta_vars)
            return names, idx

        if isinstance(variables_arg, (list, tuple)) and len(variables_arg) == 0:
            raise ValueError("variables was provided as an empty list; expected None or a non-empty subset.")

        if all(isinstance(v, (int, np.integer)) for v in variables_arg):
            idx = [int(v) for v in variables_arg]
            if any(i < 0 or i >= len(meta_vars) for i in idx):
                raise IndexError(f"variables indices out of range for meta variables (0..{len(meta_vars)-1}).")
            names = [meta_vars[i] for i in idx]
            return names, idx

        # assume list[str]
        names = [str(v) for v in variables_arg]
        missing = [v for v in names if v not in meta_vars]
        if missing:
            raise ValueError(f"Unknown variable names not in meta_data['variables']: {missing}")
        idx = [meta_vars.index(v) for v in names]
        return names, idx

    var_names, var_idx_list = _resolve_vars(all_vars, variables)
    var_idx = torch.tensor(var_idx_list, dtype=torch.long)
    n_features = len(var_names)

    # -------------------- ranges ----------------------------
    ranges = torch.load(os.path.join("./", f'ranges_{DATE}.pt'), map_location='cpu')
    y_min = ranges['y_min']
    y_max = ranges['y_max']

    def _select_channels(t: torch.Tensor, idx: torch.Tensor, n_all: int) -> torch.Tensor:
        idx = idx.to(device=t.device, dtype=torch.long)
        if idx.numel() == 0:
            return t

        chan_dim = None
        for d in range(t.ndim):
            if t.size(d) == n_all:
                chan_dim = d
                break

        if chan_dim is None:
            max_i = int(idx.max().item())
            for d in range(t.ndim):
                if t.size(d) > max_i:
                    chan_dim = d
                    break

        if chan_dim is None:
            raise IndexError(
                f"Cannot find a channel dimension in tensor of shape {tuple(t.shape)} "
                f"compatible with idx.max()={int(idx.max().item())}."
            )

        return t.index_select(chan_dim, idx)

    def cla(yhat: torch.Tensor) -> torch.Tensor:
        nside_here = int(math.sqrt(yhat.shape[2] // 12))
        key = str(nside_here)

        mi_full = y_min[key].to(yhat.device, dtype=yhat.dtype)
        ma_full = y_max[key].to(yhat.device, dtype=yhat.dtype)

        mi = _select_channels(mi_full, var_idx, n_all=len(all_vars))
        ma = _select_channels(ma_full, var_idx, n_all=len(all_vars))

        tmp = yhat.clamp(min=mi, max=ma)
        return (yhat + yhat.mean(dim=2, keepdim=True) - tmp.mean(dim=2, keepdim=True)).clamp(min=mi, max=ma)

    def healpix_to_latlon_grid(hp_cube, nlat, *, order="RING", nsub=3):
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

    # -------------------- load NSIDE=1 models -----------------------------
    nside_hi = 1
    nside_lo = 0

    maxlag_eff = 1 if maxlag is None else int(maxlag)
    variab_eff = 10 if variab is None else int(variab)
    add_latent_eff = 50 if add_latent_dim is None else int(add_latent_dim)

    lags = list(range(1, maxlag_eff + 1))

    def _load_level_1():
        # LB2
        modelb = _prep_model(
            _LB2(nside_lo, nside_hi, n_features=n_features),
            dirm + DATE + f'_b{nside_hi}',
        ).to(dev, dtype=dtype, non_blocking=True)

        # Optional M1
        modelm = None
        if useT:
            modelm = _prep_model(
                _M1(nside_hi, n_features=n_features, x_features=x_features),
                dirm + DATE + f'_m{nside_hi}',
            ).to(dev, dtype=dtype, non_blocking=True)

        # L4de
        modell = _prep_model(
            _L4de(
                nside_hi,
                n_features=n_features,
                x_features=x_features,
                variab=variab_eff,
                lags=lags,
                add_latent_dim=add_latent_eff,
            ),
            dirm + DATE + f'_l{nside_hi}_{maxlag_eff}_{variab_eff}',
        ).to(dev, dtype=dtype, non_blocking=True)

        return modelb, modelm, modell

    def _free_level(*mods):
        for m in mods:
            if m is None:
                continue
            try:
                m.cpu()
            except Exception:
                pass
            del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------- inference (NSIDE=1 only) -------------------------
    with torch.inference_mode():
        modelb, modelm, modell = _load_level_1()

        # LB2 always first
        y = cla(modelb(x=xbias, y_low=None).contiguous())


        # Optional M1 refinement: LB2 -> M1
        if useT:
            # Use xbias here to keep the same "bias model index" semantics as LB2.
            y = cla(modelm(y_in=y, x=xbias).contiguous())

        # Then L4de: (...)-> L4de
        y = cla(modell(y_in=y, x=x, dependence=dependence).contiguous())

        if memoryoptimized:
            _free_level(modelb, modelm, modell)

        # --------- transform back to observation space --------------------
        pre = torch.tensor(
            [transformation_scalars[v]["pre"] for v in var_names],
            dtype=dtype,
            device=y.device,
        )
        post = torch.tensor(
            [transformation_scalars[v]["post"] for v in var_names],
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
        arr = _to_numpy(yh_obs)  # (T, C_subset, P) with P=12 at NSIDE=1
        arr_rect = healpix_to_latlon_grid(arr, nlat, nsub=nsub)
        return arr_rect if asnumpy else torch.from_numpy(arr_rect)

    return _to_numpy(yh_obs) if asnumpy else yh_obs

def linear_trend_1d(y_1d: np.ndarray) -> float:
    """
    Returns slope from least-squares fit y = a*t + b over t = 0..n-1.
    Ignores NaNs/infs. Slope is 'per sample index' (since no time axis given).
    """
    y = np.asarray(y_1d, dtype=np.float64)
    n = y.shape[0]
    if n < 2:
        return float("nan")
    t = np.arange(n, dtype=np.float64)
    m = np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    a, b = np.polyfit(t[m], y[m], 1)
    return float(a)

def last_average_1d(y_1d: np.ndarray, lastobs: int = 240) -> float:
    """
    Returns the average of the last `lastobs` observations in y_1d.
    Ignores NaNs/infs. If fewer than `lastobs` samples exist, uses all samples.
    """
    y = np.asarray(y_1d, dtype=np.float64).ravel()
    n = y.shape[0]
    if n == 0:
        return float("nan")

    k = int(lastobs)
    if k <= 0:
        return float("nan")

    y_tail = y[-k:] if n >= k else y
    m = np.isfinite(y_tail)
    if m.sum() == 0:
        return float("nan")

    return float(y_tail[m].mean())


def last20_minus_baseline(series, end=None, offset=0.85):
    N20 = 20 * 12
    i0, i1 = 1740, 1980   # 1995-01 .. 2014-12 (end-exclusive)
    s = np.asarray(series[:end] if end is not None else series)
    if s.shape[0] < N20:
        raise ValueError(f"Need at least {N20} months for last-20y average, got {s.shape[0]}.")
    if s.shape[0] < i1:
        raise ValueError(f"Series too short to include 1995–2014 window (needs length >= {i1}).")
    last_mean = np.nanmean(s[-N20:])
    base_mean = np.nanmean(s[i0:i1])
    return last_mean - (base_mean - offset)

DATE = '7Augext'
meta_filename = 'meta_' + DATE + '.pkl'
with open(meta_filename, "rb") as f:
    meta_data = pickle.load(f)
transformation_scalars = meta_data["transformation_scalars"]
variables = meta_data["variables"]

variables = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
scen = "ssp585"
#scen = "ssp126"
scen = "ssp585"


for scen in ["ssp119",  "ssp370", "ssp434", "ssp460", "ssp534", "ssp534-over"]:

    fileno = 0
    variab = 1
    useLast = False
    forceEraModel = True

    namS = 'Ssmall'
    # Release: per-model/ERA5 checkpoints ship in the bundle under ../checkpoints/.
    dirS = os.environ.get('MODEL_DIR', '../checkpoints/models' + namS + '/')
    writedir = 'figchangeera' + namS + '_' + str(variab)
    if forceEraModel:
        writedir = 'figchangeforceera' + namS + '_' + str(variab)


    maxlag = 6 #6/3
    dependence = True
    niter = 3
    useT = (namS == 'Sthree')
    print("\n ****** samplesS ***** \n")
    print(namS)
    print(dirS)
    print(maxlag)
    print(dependence)

    os.makedirs(writedir, exist_ok=True)
    trends_path = os.path.join(writedir, "trends_" + scen + "_" + str(dependence) + "_" + str(maxlag) + ".csv")
    need_header = not os.path.exists(trends_path)

    print(trends_path)


    with open("UniqueModels_10Jun.csv", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        model_to_index = {src: int(idx) for src, idx in reader}
    num_models = max(model_to_index.values())
    versions = [src for src, idx in sorted(model_to_index.items(), key=lambda kv: kv[1]) if src != "ERA5"]

    for version in versions:
        try:
            norm_dir = os.environ.get(
                "DATA_DIR",
                os.environ.get("SCRATCH_BASE", "/scratch2")
                + f"/userdata/nicolai/cmip6/normalized_{DATE}")
            dirname = dirS + 'modelsS_' + version + '/'
            if forceEraModel:
                dirname = dirS + 'modelsS_' + 'ERA5' + '/'

            # Build candidates once per version
            candidates = []
            for fn in os.listdir(norm_dir):
                if not fn.endswith(".h5"):
                    continue
                parts = fn.split("_")
                if len(parts) < 4:
                    continue
                third = parts[2]
                fourth = parts[3]
                if not (re.search(version, third) and re.search(scen, fourth)):
                    continue 
                path = os.path.join(norm_dir, fn) 
                required_T = (2100 - 2015 + 1) * 12   
                try:
                    with h5py.File(path, "r") as h5f:
                        if "X" not in h5f:
                            continue
                        if h5f["X"].shape[0] != required_T:   
                            continue
                except OSError:
                    continue  
                candidates.append(path)
            candidates.sort()
            
            candidateshist = []
            for fn in os.listdir(norm_dir):
                if not fn.endswith(".h5"):
                    continue
                parts = fn.split("_")
                if len(parts) < 4:
                    continue
                third = parts[2]
                fourth = parts[3]
                if not (re.search(version, third) and re.search("historical", fourth)):
                    continue 
                path = os.path.join(norm_dir, fn) 
                required_T = (2014 - 1850 + 1) * 12   
                try:
                    with h5py.File(path, "r") as h5f:
                        if "X" not in h5f:
                            continue
                        if h5f["X"].shape[0] != required_T:   
                            continue
                except OSError:
                    continue  
                candidateshist.append(path)
            candidateshist.sort()
            

            if not candidates:
                raise FileNotFoundError(
                    f"No .h5 files in {norm_dir} with 3rd part matching {version!r} "
                    f"and 4th part matching {scen!r}"
                )

            for it in range(1, niter + 1):
                # Select first file: exact fileno if valid, else random candidate
                if 1 <= fileno <= len(candidates):
                    selected = candidates[fileno - 1]
                else:
                    selected = random.choice(candidates)
                selectedhist = random.choice(candidateshist)

                # NEW: second choice must be distinct AND have parts[3] exactly equal to the selected file's parts[3]
                sel_parts = os.path.basename(selected).split("_")
                sel_fourth = sel_parts[3] if len(sel_parts) >= 4 else None

                remaining = []
                if sel_fourth is not None:
                    for c in candidates:
                        if c == selected:
                            continue
                        c_parts = os.path.basename(c).split("_")
                        if len(c_parts) >= 4 and c_parts[3] == sel_fourth:
                            remaining.append(c)
                
                selected_second = random.choice(remaining) if remaining else None
                print("\n ------------------------------ selected" + version + "_" + scen)
                print(selected)
                print(selected_second)
                with h5py.File(selected, "r") as h5f:
                    x_np = h5f['X'][:]
                    y_obs = h5f['y64'][:]
                    
                print(selectedhist)
                with h5py.File(selectedhist, "r") as h5f:
                    x_np_hist = h5f['X'][:]
                    y_obs_hist = h5f['y64'][:]

                # NEW: load second y64 only (if available)
                y_obs_second = None
                if selected_second is not None:
                    with h5py.File(selected_second, "r") as h5f2:
                        y_obs_second = h5f2['y64'][:]
                        
                x_cat = np.concatenate([x_np_hist, x_np], axis=0)
                y_cat = np.concatenate([y_obs_hist, y_obs], axis=0)
                y_catsecond = np.concatenate([y_obs_hist, y_obs_second], axis=0)

                x_tensor = torch.from_numpy(x_cat).float()
                yo = torch.from_numpy(y_cat).float()
                
                x_era = x_tensor.clone()
                x_era[:,0] = 0
                
                rectangular = False
                with torch.no_grad():
                    yh = sample_from_combined_model(
                        x_tensor,
                        dirname=dirname,
                        device='cpu',
                        dependence=dependence,
                        variables=variables,
                        nside=64,
                        rectangular=rectangular,
                        nlat=180,
                        nsub=3,
                        asnumpy=False,
                        maxlag=maxlag,
                        useT=useT
                    )
                    yh_era = sample_from_combined_model(
                        x_era,
                        dirname=dirname,
                        device='cpu',
                        dependence=dependence,
                        variables=variables,
                        nside=64,
                        rectangular=rectangular,
                        nlat=180,
                        nsub=3,
                        asnumpy=False,
                        maxlag=maxlag,
                        useT=useT
                    )

                from help_functions import Downsample
                downs = Downsample()

                def down_k(y, k):
                    for _ in range(k):
                        y = downs(y)
                    return y

                all_vars = ["psl", "tas", "pr", "sfcWind", "ts", "tasmin", "tasmax", "rsds", "hurs", "huss"]
                var_idx = [int(i) for i in variables]
                vars_order = [all_vars[i] for i in var_idx]
                yo = yo[:, var_idx, :]  # (T, len(var_idx), P)

                pre = torch.tensor([transformation_scalars[v]["pre"] for v in vars_order], dtype=torch.float32)
                inv_pre = 1.0 / pre
                post_constant = torch.tensor([transformation_scalars[v]["post"] for v in vars_order], dtype=torch.float32)

                y = (yo - post_constant[None, :, None]) * inv_pre[None, :, None]

                yd = down_k(y, 0)
                yhd = down_k(yh, 0)
                yhe = down_k(yh_era, 0)

                ytrue_series = yd[:, variab, :].mean(dim=1).detach().cpu().numpy()    # (T,)
                yhat_series = yhd[:, variab, :].mean(dim=1).detach().cpu().numpy()   # (T,)
                yhatera_series = yhe[:, variab, :].mean(dim=1).detach().cpu().numpy()   # (T,)

                half = ytrue_series.shape[0]
                if not useLast:                                
                    trend_true   = last20_minus_baseline(ytrue_series)
                    trend_hat    = last20_minus_baseline(yhat_series)
                    trendera_hat = last20_minus_baseline(yhatera_series)
                else:
                    trend_true = last_average_1d(ytrue_series[:half])
                    trend_hat = last_average_1d(yhat_series[:half])
                    trendera_hat = last_average_1d(yhatera_series[:half])

                # NEW: trend_y_truesecond from second distinct candidate (same fourth), else NaN
                if y_obs_second is None:
                    trend_true_second = float("nan")
                else:
                    yo2 = torch.from_numpy(y_catsecond).float()
                    yo2 = yo2[:, var_idx, :]

                    y2 = (yo2 - post_constant[None, :, None]) * inv_pre[None, :, None]
                    yd2 = down_k(y2, 0)

                    ytrue_series2 = yd2[:, variab, :].mean(dim=1).detach().cpu().numpy()
                    half2 = ytrue_series2.shape[0]

                    if not useLast:
                        trend_true_second = last20_minus_baseline(ytrue_series2) 
                    else:
                        trend_true_second = last_average_1d(ytrue_series2[:half2])

                with open(trends_path, "a", newline="") as f:
                    w = csv.writer(f)
                    if need_header:
                        w.writerow(["version", "iter", "trend_y_true", "trend_y_truesecond", "trend_y_hat", "trend_yera_hat"])
                        need_header = False
                    w.writerow([version, it, trend_true, trend_true_second, trend_hat, trendera_hat])

                # ------------------------------------------------------------
                # Plotting (same as before)
                # ------------------------------------------------------------
                plotit = False
                if plotit:
                    n_vars = yd.shape[1]
                    cols = min(4, n_vars)
                    rows = math.ceil(n_vars / cols)

                    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True)
                    axes = np.atleast_1d(axes).ravel()

                    for v in range(n_vars):
                        ax = axes[v]
                        y_true = yd[:, v, :].mean(dim=1).detach()
                        y_hat = yhd[:, v, :].mean(dim=1).detach()

                        ax.plot(y_true.cpu().numpy(), marker="o", linewidth=1, markersize=2, label="yd")
                        ax.plot(y_hat.cpu().numpy(), marker="o", linewidth=1, markersize=2, label="yhd")
                        ax.set_title(f"var {v}")
                        ax.grid(alpha=0.2)
                        if v == 0:
                            ax.legend()

                    for k in range(n_vars, rows * cols):
                        fig.delaxes(axes[k])

                    fig.tight_layout()

                    fname = f"{writedir}/version{version}_scen{scen}_dep{dependence}_ml{maxlag}_iter{it}.png"
                    fig.savefig(fname, dpi=300, bbox_inches="tight")
                    plt.close(fig)
        except: 
            print("no file for " + version)