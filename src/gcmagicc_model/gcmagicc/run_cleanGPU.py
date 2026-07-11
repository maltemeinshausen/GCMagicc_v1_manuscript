# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
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
from models import LB2, L4de, L4adde, LB2mini, L4addemini, L4des

# GPU optimization imports
import gc
import warnings



def sample_from_combined_model(
    x,
    device = 'cpu',
    dirname = './modelsA/',
    DATE = '7Augext',
    dependence = False,
    nside = 64,
    rectangular = True,
    nlat = 180,
    nsub = 1,
    asnumpy = True,
    usebias_model = None,
    useeffect_model = None,
    seed = None
):
    # GPU optimizations setup
    _setup_gpu_optimizations(device)
    
    # Suppress torch.load warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*weights_only.*")

    dirm = dirname + 'modelsNfour_'
    
    meta_filename = dirname + 'meta_' + DATE + '.pkl'
    with open(meta_filename, "rb") as f:
        meta_data = pickle.load(f)
    transformation_scalars = meta_data["transformation_scalars"]
    variables = meta_data["variables"]

    ranges = torch.load(dirname + 'ranges_' + DATE + '.pt', map_location=device, weights_only=True)
    # Ensure all nested tensors in ranges are on correct device
    if isinstance(device, str) and device.startswith('cuda'):
        ranges = _move_dict_to_device(ranges, device)
    y_min = ranges['y_min']
    y_max = ranges['y_max']
    def cla(yhat):
        nside = str(np.sqrt(yhat.shape[2] // 12.0).astype(np.int8))
        mi = y_min[nside].to(yhat.device)
        ma = y_max[nside].to(yhat.device)
        tmp = yhat.clamp(min=mi, max=ma)
        yhat = (yhat + yhat.mean(dim=2, keepdim=True) - tmp.mean(dim=2, keepdim=True)).clamp(min=mi, max=ma)
        return yhat
    
        
    def healpix_to_latlon_grid(hp_cube, nlat, *, order="RING", nsub=3):
        
        lats = 90 - (0.5 + np.arange(nlat)) / nlat*180       # +89.5 … –89.5 °
        lons = (0.5 + np.arange(2*nlat)) / nlat * 180      # 0.5, 1.5, …, 359.5 °

        if hp_cube.ndim != 3:
            raise ValueError("hp_cube must have shape (N, C, NPIX)")
        N, C, NPIX = hp_cube.shape
        nside = hp.npix2nside(NPIX)
        if 12 * nside * nside != NPIX:
            raise ValueError(f"NPIX={NPIX} is not valid (must equal 12×NSIDE² for some integer NSIDE)")
        nest = order.upper() == "NEST"

        # sub-cell offsets in degrees, centered in each subcell (range ~[-0.5, +0.5])
        offs = (((np.arange(nsub) + 0.5) / nsub - 0.5) / nlat * 180)

        # Broadcast to a (nlat, 2*nlat, nsub, nsub) grid of sub-sample centers
        lat_sub = lats[:, None, None, None] + offs[None, None, :, None]    # varies along third axis
        lon_sub = lons[None, :, None, None] + offs[None, None, None, :]    # varies along fourth axis

        # Convert to healpy angles
        theta_sub = np.radians(90.0 - lat_sub)           # colat
        phi_sub   = np.radians(lon_sub) + np.pi          # long 
        theta_sub, phi_sub = np.broadcast_arrays(theta_sub, phi_sub)

        # Flatten sample positions
        theta_flat = theta_sub.ravel()
        phi_flat   = phi_sub.ravel()

        # Vectorise over (N, C)
        maps = hp_cube.reshape(-1, NPIX)                 # (N*C, NPIX)

        # Interpolate at all sub-sample points
        vals = hp.get_interp_val(maps, theta_flat, phi_flat, nest=nest)
        vals = vals.reshape(N, C, nlat, 2*nlat, nsub, nsub)    
        vals = vals.mean(axis=(-1, -2))  # simple arithmetic mean

        # Result: (N, C, nlat, 2*nlat)
        return vals


    if dependence:
        x = torch.cat([x[:12].repeat(12, 1), x], dim=0)
    x = x.to(device)
    n_features=len(variables)
    x_features=x.shape[1]
    
    xbias  = x.clone()
    if usebias_model is not None:
        xbias[:,0] = usebias_model
    if useeffect_model is not None:
        x[:,0] = useeffect_model

    # nside_hi = 1
    # nside_lo=nside_hi//2
    # modelbias_1 = LB2( nside_lo, nside_hi, n_features=n_features).to(device)
    # model_dir_load = dirm + DATE + '_b' + str(nside_hi)
    # modelbias_1.load_state_dict(torch.load(os.path.join(model_dir_load, 'best_model.pt'), map_location=device, weights_only=True))
    
    # maxlag = 48
    # variab = 10
    # add_latent_dim = 50
    # sufflow = '_bdexl'
    # lags = [i for i in range(1,maxlag)]
    # model_1 = L4de( nside_hi, n_features=n_features, x_features=x_features, variab=variab, lags = lags, add_latent_dim=add_latent_dim).to(device)
    # model_dir_load = dirm + DATE + sufflow + str(nside_hi) + "_" + str(maxlag) + "_" + str(variab)
    # model_1.load_state_dict(torch.load(os.path.join(model_dir_load, 'best_model.pt'), map_location=device, weights_only=True))
    
    nside_hi = 2
    nside_lo = 0 
    modelbias_2 = LB2( nside_lo, nside_hi, n_features=n_features).to(device)
    model_dir_load = dirm + DATE + '_bS' + str(nside_hi)
    modelbias_2.load_state_dict(torch.load(os.path.join(model_dir_load, 'best_model.pt'), map_location=device, weights_only=True))
    
    maxlag = 72
    variab = 10
    lags = [1,2,3,4,5,6,8,10,12,14,16,18,20,24,27,30,33,36,42,48,54,60,66,72]
    sufflow = '_bdeSxlsp'    
    model_2 = L4de( nside_hi, n_features=n_features, x_features=x_features, variab=variab, lags=lags, add_latent_dim= 50).to(device)
    #model_2 = L4de( nside_hi, n_features=n_features, x_features=x_features, lags=lags, variab=variab, add_latent_dim=add_latent_dim).to(device)
    model_dir_load = dirm + DATE + sufflow + str(nside_hi) + "_" + str(maxlag) + "_" + str(variab)
    model_2.load_state_dict(torch.load(os.path.join(model_dir_load, 'best_model.pt'), map_location=device, weights_only=True))
    
    nside_hi = 4
    nside_lo=nside_hi//2
    modelbias_4 = LB2( nside_lo, nside_hi, n_features=n_features).to(device)
    model_dir_load = dirm + DATE + '_b' + str(nside_hi)
    modelbias_4.load_state_dict(torch.load(os.path.join(model_dir_load, 'best_model.pt'), map_location=device, weights_only=True))

    maxlag = 6
    variab = 0
    sufflow = '_bde'
    lags = [i for i in range(1,maxlag)]
    model_4 = L4de( nside_hi, n_features=n_features, x_features=x_features, variab=variab, lags=lags).to(device)
    model_dir_load = dirm + DATE + sufflow + str(nside_hi) + "_" + str(maxlag) + "_" + str(variab)
    model_4.load_state_dict(torch.load(os.path.join(model_dir_load, 'best_model.pt'), map_location=device, weights_only=True))
    
    nside_hi = 8
    maxlag = 2
    variab = 0 
    sufflow = '_bde'
    nside_lo=nside_hi//2
    modelbias_8 = LB2( nside_lo, nside_hi, n_features=n_features).to(device)
    model_dir_load = dirm + DATE + '_b' + str(nside_hi)
    modelbias_8.load_state_dict(torch.load(os.path.join(model_dir_load, 'best_model.pt'), map_location=device, weights_only=True))

    lags = [i for i in range(1,maxlag)]
    model_8 = L4de( nside_hi, n_features=n_features, x_features=x_features, variab=variab, lags=lags).to(device)
    model_dir_load = dirm + DATE + sufflow + str(nside_hi) + "_" + str(maxlag) + "_" + str(variab)
    model_8.load_state_dict(torch.load(os.path.join(model_dir_load, 'best_model.pt'), map_location=device, weights_only=True))
            
    nside_hi = 16
    nside_lo=nside_hi//2
    modelbias_16 = LB2( nside_lo, nside_hi, n_features=n_features).to(device)
    model_dir_load = dirm + DATE + '_b' + str(nside_hi)
    modelbias_16.load_state_dict(torch.load(os.path.join(model_dir_load, 'best_model.pt'), map_location=device, weights_only=True))
   
    model_16 = L4adde( nside_hi, n_features=n_features, x_features=x_features).to(device)
    model_dir_load = dirm + DATE + '_bdadde' + str(nside_hi)
    model_16.load_state_dict(torch.load(os.path.join(model_dir_load, 'best_model.pt'), map_location=device, weights_only=True))
            
    
    nside_hi = 32
    nside_lo=nside_hi//2
    modelbias_32 = LB2( nside_lo, nside_hi, n_features=n_features).to(device)
    model_dir_load = dirm + DATE + '_b' + str(nside_hi)
    modelbias_32.load_state_dict(torch.load(os.path.join(model_dir_load, 'best_model.pt'), map_location=device, weights_only=True))
   
    model_32 = L4adde( nside_hi, n_features=n_features, x_features=x_features).to(device)
    model_dir_load = dirm + DATE + '_bdadde' + str(nside_hi)
    model_32.load_state_dict(torch.load(os.path.join(model_dir_load, 'best_model.pt'), map_location=device, weights_only=True))
    
    nside_hi = 64
    nside_lo=nside_hi//2
    modelbias_64 = LB2( nside_lo, nside_hi, n_features=n_features).to(device)
    model_dir_load = dirm + DATE + '_b' + str(nside_hi)
    modelbias_64.load_state_dict(torch.load(os.path.join(model_dir_load, 'best_model.pt'), map_location=device, weights_only=True))
   
    model_64 = L4adde( nside_hi, n_features=n_features, x_features=x_features).to(device)
    model_dir_load = dirm + DATE + '_bdadde' + str(nside_hi)
    model_64.load_state_dict(torch.load(os.path.join(model_dir_load, 'best_model.pt'), map_location=device, weights_only=True))
    
    if nside>64:
        nside_hi = 128
        nside_lo=nside_hi//2
        modelbias_128 = LB2mini( nside_lo, nside_hi, n_features=n_features).to(device)
        model_dir_load = dirm + DATE + '_b' + str(nside_hi)
        modelbias_128.load_state_dict(torch.load(os.path.join(model_dir_load, 'best_model.pt'), map_location=device, weights_only=True))
    
        model_128 = L4addemini( nside_hi, n_features=n_features, x_features=x_features).to(device)
        model_dir_load = dirm + DATE + '_bdadde' + str(nside_hi)
        model_128.load_state_dict(torch.load(os.path.join(model_dir_load, 'best_model.pt'), map_location=device, weights_only=True))
        
    if nside>128:
        nside_hi = 256
        nside_lo=nside_hi//2
        modelbias_256 = LB2mini( nside_lo, nside_hi, n_features=n_features).to(device)
        model_dir_load = dirm + DATE + '_b' + str(nside_hi)
        modelbias_256.load_state_dict(torch.load(os.path.join(model_dir_load, 'best_model.pt'), map_location=device, weights_only=True))
    
        model_256 = L4addemini( nside_hi, n_features=n_features, x_features=x_features).to(device)
        model_dir_load = dirm + DATE + '_bdadde' + str(nside_hi)
        model_256.load_state_dict(torch.load(os.path.join(model_dir_load, 'best_model.pt'), map_location=device, weights_only=True))
        
    
    if seed is not None:
        random.seed(seed)                  
        np.random.seed(seed)               
        torch.manual_seed(seed)            
        torch.cuda.manual_seed_all(seed)   
        
    #downs = DownsampleWithNoise() 
    #y_hat1 = cla(modelbias_1( x=xbias, y_low=None).contiguous())
    #y_hat1 = cla(model_1(y_in=y_hat1, x=x, dependence=dependence).contiguous())
    y_hat2 = cla(modelbias_2( y_low=None, x=xbias).contiguous())
    y_hat2 = cla(model_2(y_in=y_hat2, x=x, dependence=dependence).contiguous())
    try:
        del modelbias_2
        del model_2
    except NameError:
        pass
    if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    y_hat4 = cla(modelbias_4( y_low=y_hat2,x=xbias).contiguous())
    y_hat4 = cla(model_4(y_in=y_hat4, x=x, dependence=dependence).contiguous())
    try: del y_hat2
    except NameError: pass
    try:
        del modelbias_4
        del model_4
    except NameError:
        pass
    if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    y_hat8 = cla(modelbias_8( y_low=y_hat4,x=xbias).contiguous())
    y_hat8 = cla(model_8(y_in=y_hat8, x=x, dependence=dependence).contiguous())
    try: del y_hat4
    except NameError: pass
    try:
        del modelbias_8
        del model_8
    except NameError:
        pass
    if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    y_hat16 = cla(modelbias_16( y_low=y_hat8,x=xbias).contiguous())
    y_hat16 = cla(model_16(y_in=y_hat16, x=x).contiguous())
    try: del y_hat8
    except NameError: pass
    try:
        del modelbias_16
        del model_16
    except NameError:
        pass
    if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    y_hat32 = cla(modelbias_32( y_low=y_hat16,x=xbias).contiguous())
    y_hat32 = cla(model_32(y_in=y_hat32, x=x).contiguous())
    try: del y_hat16
    except NameError: pass
    try:
        del modelbias_32
        del model_32
    except NameError:
        pass
    if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    y_hat = cla(modelbias_64( y_low=y_hat32,x=xbias).contiguous())
    y_hat = cla(model_64(y_in=y_hat, x=x).contiguous())
    try: del y_hat32
    except NameError: pass
    try:
        del modelbias_64
        del model_64
    except NameError:
        pass
    if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if nside>64:
        y_hat = modelbias_128(y_low = y_hat, x=xbias).contiguous()
        y_hat = model_128(y_in = y_hat, x=x).contiguous()
        try:
            del modelbias_128
            del model_128
        except NameError:
            pass
        if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
    if nside>128:
        y_hat = modelbias_256(y_low = y_hat, x=xbias).contiguous()
        y_hat = model_256(y_in = y_hat, x=x).contiguous()
        try:
            del modelbias_256
            del model_256
        except NameError:
            pass
        if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()

    vars_order = ["psl", "tas", "pr", "sfcWind", "ts", "tasmin", "tasmax", "rsds", "hurs", "huss"]
    vars_order = [v for v in vars_order if v in variables]

    pre = torch.tensor([transformation_scalars[var]["pre"] for var in vars_order], dtype=torch.float32, device=y_hat.device)
    inv_pre = 1 / pre  
    post_constant = torch.tensor([transformation_scalars[var]["post"] for var in vars_order], dtype=torch.float32, device=y_hat.device)
    yh_obs = (y_hat - post_constant[None, :, None])  * inv_pre[None, :, None]
    
    if dependence:
        yh_obs = yh_obs[144:]
    
    yh_obs = yh_obs.detach()
    # Ensure CPU numpy array before healpy interpolation when rectangular=True
    if rectangular and isinstance(yh_obs, torch.Tensor):
        yh_obs = yh_obs.cpu().numpy()
    elif asnumpy:
        yh_obs = yh_obs.cpu().numpy() if getattr(yh_obs, 'is_cuda', False) else (yh_obs.numpy() if hasattr(yh_obs, 'numpy') else np.asarray(yh_obs))
    if rectangular:
        if isinstance(yh_obs, torch.Tensor):
            yh_obs = yh_obs.detach().to('cpu').numpy()
        yh_obs = healpix_to_latlon_grid(yh_obs, nlat, nsub=nsub)
    
    # Delete models/tensors to free memory before return (guarded)
    try: del modelbias_2
    except NameError: pass
    try: del model_dir_load
    except NameError: pass
    try: del model_2
    except NameError: pass
    try: del modelbias_4
    except NameError: pass
    try: del model_4
    except NameError: pass
    try: del modelbias_8
    except NameError: pass
    try: del model_8
    except NameError: pass
    try: del modelbias_16
    except NameError: pass
    try: del model_16
    except NameError: pass
    try: del modelbias_32
    except NameError: pass
    try: del model_32
    except NameError: pass
    try: del modelbias_64
    except NameError: pass
    try: del model_64
    except NameError: pass
    try: del modelbias_128
    except NameError: pass
    try: del model_128
    except NameError: pass
    try: del modelbias_256
    except NameError: pass
    try: del model_256
    except NameError: pass
    try: del y_hat2
    except NameError: pass
    try: del y_hat4
    except NameError: pass
    try: del y_hat8
    except NameError: pass
    try: del y_hat16
    except NameError: pass
    try: del y_hat32
    except NameError: pass
    try: del y_hat
    except NameError: pass
    gc.collect()
    # CUDA-specific cleanup if using GPU
    if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    # GPU memory cleanup before return
    _gpu_memory_cleanup(device)
    return yh_obs

def _move_dict_to_device(obj, device):
    """Recursively move all tensors in a nested dict/list structure to the specified device"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: _move_dict_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_move_dict_to_device(item, device) for item in obj)
    else:
        return obj

def _gpu_memory_cleanup(device):
    """Clean up GPU memory to prevent fragmentation and OOM errors"""
    if isinstance(device, str) and device.startswith('cuda'):
        try:
            # Force garbage collection
            gc.collect()
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Synchronize to ensure cleanup is complete
                torch.cuda.synchronize()
        except Exception:
            pass

def _setup_gpu_optimizations(device):
    """Setup GPU optimizations for better performance and memory management"""
    if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
        try:
            # Enable TF32 for faster computation on Ampere+ GPUs
            if hasattr(torch.backends, 'cuda'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.allow_tf32 = True
            
            # Set memory allocation strategy to reduce fragmentation
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            
            # Set device
            if ':' in device:
                device_idx = int(device.split(':')[1])
            else:
                device_idx = 0
            torch.cuda.set_device(device_idx)
            
        except Exception as e:
            # Suppress warnings but continue - GPU optimizations are optional
            pass
