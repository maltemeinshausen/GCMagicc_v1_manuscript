"""
helper_regions_ar6.py
=====================
IPCC AR6 land regions aggregation using `regionmask`.
We compute area‑weighted region means with cosine‑latitude weights.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import xarray as xr

DEFAULT_AR6_EXAMPLE_REGIONS = ["MED", "CNA", "EAU", "EAS", "ESAF"]

def _coord_names(obj) -> Tuple[str, str]:
    lat_name = next((n for n in ("lat", "latitude", "y") if n in obj.coords), None)
    lon_name = next((n for n in ("lon", "longitude", "x") if n in obj.coords), None)
    if lat_name is None or lon_name is None:
        raise ValueError("Could not find latitude/longitude coordinates.")
    return lat_name, lon_name

def _coslat_weights(da: xr.DataArray) -> xr.DataArray:
    lat_name, lon_name = _coord_names(da)
    lat2d, lon2d = xr.broadcast(da[lat_name], da[lon_name])
    w = np.cos(np.deg2rad(lat2d)).astype("float32")
    return w

def get_ar6_land_mask(obj: xr.Dataset | xr.DataArray):
    """
    Return (mask, numbers, abbrevs) for AR6 land regions on obj's grid.
    mask dims: (lat, lon) or (y, x); values are integer region numbers, NaN elsewhere.
    """
    try:
        import regionmask
    except Exception as e:
        raise ImportError("regionmask is required to compute IPCC AR6 region means.") from e
    if isinstance(obj, xr.Dataset):
        # pick a representative 2D field to infer grid
        da = next(iter(obj.data_vars.values()))
    else:
        da = obj
    ar6 = regionmask.defined_regions.ar6.land
    mask = ar6.mask(da)  # broadcast to da grid
    numbers = list(ar6.numbers)
    abbrevs = list(ar6.abbrevs)
    return mask, numbers, abbrevs

def region_means_for_ar6(da: xr.DataArray, *, regions: Optional[List[str]] = None) -> Dict[str, xr.DataArray]:
    """
    Area‑weighted region means for each AR6 land region present in `da`.
    Returns a dict {REGION_ABBR: 1D time series}.
    """
    mask, numbers, abbrevs = get_ar6_land_mask(da)
    lat_name, lon_name = _coord_names(da)
    weights = _coslat_weights(da)
    # dict number->abbr
    num2abbr = {n: a for n, a in zip(numbers, abbrevs)}
    abbrs = [num2abbr[n] for n in numbers if (regions is None or num2abbr[n] in regions)]
    out: Dict[str, xr.DataArray] = {}
    for n in numbers:
        r = num2abbr[n]
        if (regions is not None) and (r not in regions):
            continue
        reg_mask = xr.where(mask == n, 1.0, np.nan)
        w = (weights * reg_mask).astype("float32")
        denom = w.sum(dim=(lat_name, lon_name), skipna=True)
        num = (da * w).sum(dim=(lat_name, lon_name), skipna=True)
        ts = (num / xr.where(denom > 0, denom, np.nan)).astype("float32")
        out[r] = ts
    return out
