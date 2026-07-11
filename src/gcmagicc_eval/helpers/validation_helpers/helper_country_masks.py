# ruff: noqa: E402
# -*- coding: utf-8 -*-
"""
helper_country_masks - build lon360 1×1deg masks from country polygons
--------------------------------------------------------------------
Defaults to Natural Earth Admin-0 via Cartopy's shapereader.
Uses Rasterio to rasterize when available (fast), else a Shapely fallback.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
import xarray as xr

# Optional deps
try:
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_origin as rio_from_origin

    _HAS_RASTERIO = True
except Exception:
    _HAS_RASTERIO = False

try:
    from cartopy.io import shapereader as ne_reader
except Exception:
    ne_reader = None

try:
    from shapely.geometry import shape as shp_shape, Point as shp_Point
    from shapely.ops import unary_union as shp_union
    from shapely.prepared import prep as shp_prep

    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False

__all__ = ["masks_from_naturalearth", "western_europe_default_iso3"]


def western_europe_default_iso3() -> List[str]:
    return [
        "PRT",
        "ESP",
        "FRA",
        "BEL",
        "NLD",
        "LUX",
        "IRL",
        "GBR",
        "CHE",
        "AUT",
        "ITA",
        "DEU",
        "DNK",
        "SWE",
        "NOR",
        "FIN",
        "ISL",
    ]


def _collect_iso3_set(groups: Dict[str, List[str]]) -> set:
    s = set()
    for v in groups.values():
        s.update([x.upper() for x in v])
    return s


def _adm0_iso3(rec_attrs: dict) -> str | None:
    # Be robust to field-name differences across NE versions
    for key in ("ADM0_A3", "ISO_A3", "ISO_A3_EH", "WB_A3", "SOV_A3"):
        val = rec_attrs.get(key)
        if val and str(val).upper() != "-99":
            return str(val).upper()
    return None


def _rasterize_union(
    geom, width: int, height: int, *, lon0=-180.0, lat0=90.0, dx=1.0, dy=1.0
) -> np.ndarray:
    """Rasterize 'geom' onto a grid defined by origin/spacing in lon/lat degrees."""
    if _HAS_RASTERIO:
        tfm = rio_from_origin(lon0, lat0, dx, dy)
        arr = rio_rasterize(
            [(geom, 1)], out_shape=(height, width), transform=tfm, fill=0, dtype="uint8"
        )
        return arr
    # Shapely fallback: point-in-polygon at cell centers (fine for 1deg grid)
    xs = np.arange(lon0 + dx / 2.0, lon0 + width * dx, dx)
    ys = np.arange(lat0 - dy / 2.0, lat0 - height * dy, -dy)  # top to bottom
    X, Y = np.meshgrid(xs, ys)
    pts = np.vstack([X.ravel(), Y.ravel()]).T
    P = shp_prep(geom)
    out = np.zeros(pts.shape[0], dtype="uint8")
    for i, (x, y) in enumerate(pts):
        out[i] = 1 if P.contains(shp_Point(float(x), float(y))) else 0
    return out.reshape((height, width))


def _to_lon360_mask(mask_lon180: np.ndarray) -> np.ndarray:
    """Shift width=360 array from lon[-180..180) to lon[0..360)."""
    # columns 0..179 -> -180..-1; 180..359 -> 0..179
    return np.roll(mask_lon180, shift=180, axis=1)


def masks_from_naturalearth(
    target_grid: xr.Dataset, iso3_groups: Dict[str, List[str]], *, scale: str = "110m"
) -> Dict[str, xr.DataArray]:
    """
    Build boolean masks (lat, lon) on target_grid (expects lat ascending, lon in [0,360)),
    using Natural Earth Admin-0 countries at the requested scale.
    """
    if ne_reader is None or (not _HAS_SHAPELY):
        raise RuntimeError(
            "cartopy (with shapereader) and shapely are required for Natural Earth masks."
        )

    shp_path = ne_reader.natural_earth(
        resolution=scale, category="cultural", name="admin_0_countries"
    )
    reader = ne_reader.Reader(shp_path)
    wanted = _collect_iso3_set(iso3_groups)

    # Collect and union geometries per ISO3
    geoms_by_iso: dict[str, list] = {}
    for rec in reader.records():
        iso = _adm0_iso3(rec.attributes)
        if (iso is None) or (iso not in wanted):
            continue
        # geometry is lon in [-180, 180]
        g = rec.geometry
        if g is None:
            try:
                g = shp_shape(rec.geometry)  # robustness if geometry is mapping
            except Exception:
                continue
        geoms_by_iso.setdefault(iso, []).append(g)

    # Prepare output grid dims
    lat = np.asarray(target_grid["lat"].values)
    lon = np.asarray(target_grid["lon"].values)
    assert np.isclose(lat[1] - lat[0], 1.0) and np.isclose(
        lon[1] - lon[0], 1.0
    ), "Target grid must be 1deg"
    # Grid dimensions: usually 180×360
    # H = lat.size; W = lon.size  # unused variables removed

    masks: Dict[str, xr.DataArray] = {}
    for name, iso_list in iso3_groups.items():
        # union all requested ISO3 geometries in lon[-180,180]
        parts = [p for iso in (x.upper() for x in iso_list) for p in geoms_by_iso.get(iso, [])]
        if not parts:
            masks[name] = xr.zeros_like(
                target_grid["lat"].broadcast_like(target_grid["lon"]).transpose(), dtype=bool
            )
            continue
        U = shp_union(parts)
        # rasterize on lon[-180..180),lat[90..-90]
        mask_180 = _rasterize_union(
            U, width=360, height=180, lon0=-180.0, lat0=90.0, dx=1.0, dy=1.0
        )
        # shift to lon[0..360)
        mask_360 = _to_lon360_mask(mask_180)
        # align to target lat ascending (our raster y is top->bottom; target lat likely ascending)
        if lat[0] < lat[-1]:
            mask_360 = mask_360[::-1, :]
        da = xr.DataArray(
            mask_360.astype(bool),
            dims=("lat", "lon"),
            coords=dict(lat=lat, lon=lon),
            name=f"mask_{name}",
        )
        masks[name] = da
    return masks
