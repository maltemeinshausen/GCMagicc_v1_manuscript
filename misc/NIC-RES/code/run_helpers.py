import os
import numpy as np
import torch
import healpy as hp
import h5py
import torch.nn.functional as F
from help_functions import DownsampleWithNoise  
import os
import pickle
import random


from models import LB2, L4de, L4adde, LB2mini, L4addemini

def sample_from_combined_model(
    x,
    device='cpu',
    dirname='./modelsA/',
    DATE='7Augext',
    dependence=False,
    nside=64,
    rectangular=True,
    nlat=180,
    nsub=1,
    asnumpy=True,
    usebias_model=None,
    useeffect_model=None,
    seed=None
):
    dirm = dirname + 'modelsNfour_'

    # ---- helper: load on CPU -> eval+freeze -> move to device ----------
    def _prep_model(model, ckpt_dir):
        sd = torch.load(os.path.join(ckpt_dir, 'best_model.pt'), map_location='cpu')
        model.load_state_dict(sd)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model.to(device, non_blocking=True)

    meta_filename = dirname + 'meta_' + DATE + '.pkl'
    with open(meta_filename, "rb") as f:
        meta_data = pickle.load(f)
    transformation_scalars = meta_data["transformation_scalars"]
    variables = meta_data["variables"]

    ranges = torch.load(dirname + 'ranges_' + DATE + '.pt', map_location='cpu')
    y_min = ranges['y_min']
    y_max = ranges['y_max']

    def cla(yhat):
        nside_str = str(np.sqrt(yhat.shape[2] // 12.0).astype(np.int8))
        mi = y_min[nside_str].to(yhat.device)
        ma = y_max[nside_str].to(yhat.device)
        tmp = yhat.clamp(min=mi, max=ma)
        yhat = (yhat + yhat.mean(dim=2, keepdim=True) - tmp.mean(dim=2, keepdim=True)).clamp(min=mi, max=ma)
        return yhat

    def healpix_to_latlon_grid(hp_cube, nlat, *, order="RING", nsub=3):
        lats = 90 - (0.5 + np.arange(nlat)) / nlat * 180
        lons = (0.5 + np.arange(2 * nlat)) / nlat * 180
        if hp_cube.ndim != 3:
            raise ValueError("hp_cube must have shape (N, C, NPIX)")
        N, C, NPIX = hp_cube.shape
        nside_local = hp.npix2nside(NPIX)
        if 12 * nside_local * nside_local != NPIX:
            raise ValueError(f"NPIX={NPIX} is not valid (must equal 12×NSIDE² for some integer NSIDE)")
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

    if dependence:
        x = torch.cat([x[:12].repeat(12, 1), x], dim=0)
    x = x.to(device, non_blocking=True)
    n_features = len(variables)
    x_features = x.shape[1]

    xbias = x.clone()
    if usebias_model is not None:
        xbias[:, 0] = usebias_model
    if useeffect_model is not None:
        x[:, 0] = useeffect_model

    # ------------------- load all models (eval + frozen) -----------------
    # nside=2
    nside_hi = 2
    nside_lo = 0
    modelbias_2 = _prep_model(LB2(nside_lo, nside_hi, n_features=n_features),
                              dirm + DATE + f'_bS{nside_hi}')
    maxlag = 72
    variab = 10
    lags = [1,2,3,4,5,6,8,10,12,14,16,18,20,24,27,30,33,36,42,48,54,60,66,72]
    sufflow = '_bdeSxlsp'
    model_2 = _prep_model(L4de(nside_hi, n_features=n_features, x_features=x_features,
                               variab=variab, lags=lags, add_latent_dim=50),
                          dirm + DATE + f'{sufflow}{nside_hi}_{maxlag}_{variab}')

    # nside=4
    nside_hi = 4
    nside_lo = nside_hi // 2
    modelbias_4 = _prep_model(LB2(nside_lo, nside_hi, n_features=n_features),
                              dirm + DATE + f'_b{nside_hi}')
    maxlag = 6
    variab = 0
    sufflow = '_bde'
    lags = [i for i in range(1, maxlag)]
    model_4 = _prep_model(L4de(nside_hi, n_features=n_features, x_features=x_features,
                               variab=variab, lags=lags),
                          dirm + DATE + f'{sufflow}{nside_hi}_{maxlag}_{variab}')

    # nside=8
    nside_hi = 8
    maxlag = 2
    variab = 0
    sufflow = '_bde'
    nside_lo = nside_hi // 2
    modelbias_8 = _prep_model(LB2(nside_lo, nside_hi, n_features=n_features),
                              dirm + DATE + f'_b{nside_hi}')
    lags = [i for i in range(1, maxlag)]
    model_8 = _prep_model(L4de(nside_hi, n_features=n_features, x_features=x_features,
                               variab=variab, lags=lags),
                          dirm + DATE + f'{sufflow}{nside_hi}_{maxlag}_{variab}')

    # nside=16
    nside_hi = 16
    nside_lo = nside_hi // 2
    modelbias_16 = _prep_model(LB2(nside_lo, nside_hi, n_features=n_features),
                               dirm + DATE + f'_b{nside_hi}')
    model_16 = _prep_model(L4adde(nside_hi, n_features=n_features, x_features=x_features),
                           dirm + DATE + f'_bdadde{nside_hi}')

    # nside=32
    nside_hi = 32
    nside_lo = nside_hi // 2
    modelbias_32 = _prep_model(LB2(nside_lo, nside_hi, n_features=n_features),
                               dirm + DATE + f'_b{nside_hi}')
    model_32 = _prep_model(L4adde(nside_hi, n_features=n_features, x_features=x_features),
                           dirm + DATE + f'_bdadde{nside_hi}')

    # nside=64
    nside_hi = 64
    nside_lo = nside_hi // 2
    modelbias_64 = _prep_model(LB2(nside_lo, nside_hi, n_features=n_features),
                               dirm + DATE + f'_b{nside_hi}')
    model_64 = _prep_model(L4adde(nside_hi, n_features=n_features, x_features=x_features),
                           dirm + DATE + f'_bdadde{nside_hi}')

    # optional higher nsides
    if nside > 64:
        nside_hi = 128
        nside_lo = nside_hi // 2
        modelbias_128 = _prep_model(LB2mini(nside_lo, nside_hi, n_features=n_features),
                                    dirm + DATE + f'_b{nside_hi}')
        model_128 = _prep_model(L4addemini(nside_hi, n_features=n_features, x_features=x_features),
                                dirm + DATE + f'_bdadde{nside_hi}')

    if nside > 128:
        nside_hi = 256
        nside_lo = nside_hi // 2
        modelbias_256 = _prep_model(LB2mini(nside_lo, nside_hi, n_features=n_features),
                                    dirm + DATE + f'_b{nside_hi}')
        model_256 = _prep_model(L4addemini(nside_hi, n_features=n_features, x_features=x_features),
                                dirm + DATE + f'_bdadde{nside_hi}')

    # ------------------------- seeding -----------------------------------
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ------------------------- inference ---------------------------------
    with torch.inference_mode():
        # y_hat1 is commented out in the original — keep behavior the same
        y_hat2 = cla(modelbias_2(y_low=None, x=xbias).contiguous())
        y_hat2 = cla(model_2(y_in=y_hat2, x=x, dependence=dependence).contiguous())
        y_hat4 = cla(modelbias_4(y_low=y_hat2, x=xbias).contiguous())
        y_hat4 = cla(model_4(y_in=y_hat4, x=x, dependence=dependence).contiguous())
        y_hat8 = cla(modelbias_8(y_low=y_hat4, x=xbias).contiguous())
        y_hat8 = cla(model_8(y_in=y_hat8, x=x, dependence=dependence).contiguous())
        y_hat16 = cla(modelbias_16(y_low=y_hat8, x=xbias).contiguous())
        y_hat16 = cla(model_16(y_in=y_hat16, x=x).contiguous())
        y_hat32 = cla(modelbias_32(y_low=y_hat16, x=xbias).contiguous())
        y_hat32 = cla(model_32(y_in=y_hat32, x=x).contiguous())
        y_hat = cla(modelbias_64(y_low=y_hat32, x=xbias).contiguous())
        y_hat = cla(model_64(y_in=y_hat, x=x).contiguous())
        if nside > 64:
            y_hat = modelbias_128(y_low=y_hat, x=xbias).contiguous()
            y_hat = model_128(y_in=y_hat, x=x).contiguous()
        if nside > 128:
            y_hat = modelbias_256(y_low=y_hat, x=xbias).contiguous()
            y_hat = model_256(y_in=y_hat, x=x).contiguous()

        vars_order = ["psl", "tas", "pr", "sfcWind", "ts", "tasmin", "tasmax", "rsds", "hurs", "huss"]
        vars_order = [v for v in vars_order if v in variables]

        pre = torch.tensor([transformation_scalars[var]["pre"] for var in vars_order],
                           dtype=torch.float32, device=y_hat.device)
        inv_pre = 1 / pre
        post_constant = torch.tensor([transformation_scalars[var]["post"] for var in vars_order],
                                     dtype=torch.float32, device=y_hat.device)
        yh_obs = (y_hat - post_constant[None, :, None]) * inv_pre[None, :, None]

        if dependence:
            yh_obs = yh_obs[144:]

        if asnumpy:
            yh_obs = yh_obs.to('cpu').numpy()

    if rectangular:
        # healpix_to_latlon_grid expects numpy arrays
        if not isinstance(yh_obs, np.ndarray):
            yh_obs = yh_obs.to('cpu').numpy()
        yh_obs = healpix_to_latlon_grid(yh_obs, nlat, nsub=nsub)

    return yh_obs
