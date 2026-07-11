# ruff: noqa: E402
from __future__ import annotations
import numpy as np
import xarray as xr


def to_360_lon(da: xr.DataArray | xr.Dataset, lon_name: str = "lon"):
    """Return a copy with longitude in [0, 360). Keeps original order otherwise."""
    if lon_name not in da.coords:
        return da
    lon = da[lon_name]
    lon360 = (lon % 360).astype(lon.dtype)
    out = da.assign_coords({lon_name: lon360})
    # sort by lon to be monotonic
    out = out.sortby(lon_name)
    return out


def _area_weights(lat: xr.DataArray) -> xr.DataArray:
    """Cosine latitude weights normalized to mean=1 (approx area weighting)."""
    w = np.cos(np.deg2rad(lat))
    return (w / w.mean()).astype("float32")


def coarse_mean_05_to_10(
    da: xr.DataArray, lat_name: str = "lat", lon_name: str = "lon"
) -> xr.DataArray:
    """
    Regrid 0.5deg regular grid to 1.0deg by simple 2x2 block averaging
    (area-weighted by cos(lat)). Assumes exact 0.5deg spacing.
    """
    # Ensure sorted & equally spaced
    da = da.sortby([lat_name, lon_name])
    # Build weights (broadcast to data dims for coarsen)
    wlat = _area_weights(da[lat_name])
    # Expand weights to full array shape
    w = xr.ones_like(da) * wlat
    num = (da * w).coarsen({lat_name: 2, lon_name: 2}, boundary="trim").sum()
    den = w.coarsen({lat_name: 2, lon_name: 2}, boundary="trim").sum()
    out = (num / xr.where(den > 0, den, np.nan)).astype("float32")
    # Recompute target coords as integer degree cell centers
    out = out.assign_coords(
        {
            lat_name: np.round(out[lat_name].values).astype("float32"),
            lon_name: np.round(out[lon_name].values).astype("float32"),
        }
    )
    return out


def regrid_halfdeg_to_onedeg_360(
    da: xr.DataArray, lat_name: str = "lat", lon_name: str = "lon"
) -> xr.DataArray:
    """Convenience: convert lon->[0,360), then 0.5deg->1.0deg block average (area‑weighted)."""
    d2 = to_360_lon(da, lon_name=lon_name)
    return coarse_mean_05_to_10(d2, lat_name=lat_name, lon_name=lon_name)
