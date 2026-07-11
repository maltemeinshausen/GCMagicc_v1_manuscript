"""
helper_heatmap_metrics.py
=========================
Vectorized, timing‑invariant comparison metrics for SPI/SPEI "heatmaps"
and related spectral diagnostics.

Provided metrics (no internal parallelization):
  * quantile_rmse_map(a, b): L2 distance between empirical quantile
    functions per grid cell (distributional match).
  * spectral_rmse_map(a, b): L2 distance between normalized periodograms
    per grid cell over a band of periods (persistence/temporal‑structure match).
  * dominant_period_map(x): dominant (peak) period [months] of x's normalized
    periodogram over a band of periods (persistence diagnostic).

All inputs are xarray.DataArray with a "time" dimension (and optional spatial dims).
Outputs are DataArrays mapped over the spatial dims (or scalar if only "time").
"""
from __future__ import annotations
import numpy as np
import xarray as xr
from typing import Optional, Tuple

__all__ = ["quantile_rmse_map", "spectral_rmse_map", "dominant_period_map"]


def _flatten_time_space(da: xr.DataArray) -> Tuple[np.ndarray, Tuple[str, ...], Tuple[int, ...], dict]:
    if "time" not in da.dims:
        raise ValueError("Input must have a 'time' dimension.")
    spatial_dims = tuple(d for d in da.dims if d != "time")
    arr = da.transpose("time", *spatial_dims).values  # (T, ...)
    T = arr.shape[0]
    shape_space = arr.shape[1:]
    arr2d = arr.reshape(T, -1)
    coords = {d: da[d] for d in spatial_dims}
    return arr2d, spatial_dims, shape_space, coords


def quantile_rmse_map(a: xr.DataArray, b: xr.DataArray, *, percentiles=None) -> xr.DataArray:
    """
    Per‑cell L2 distance between empirical quantile functions of A and B.
    Timing‑invariant (depends only on marginal distributions).
    """
    a, b = xr.align(a, b, join="inner")
    A, spatial_dims, shape_space, coords = _flatten_time_space(a)
    B, _, _, _ = _flatten_time_space(b)
    if percentiles is None:
        percentiles = np.arange(1, 100, 1, dtype=np.float32)  # 1..99
    P = np.asarray(percentiles, dtype=np.float32)
    if A.size == 0 or B.size == 0:
        out = np.full(shape_space, np.nan, dtype=np.float32)
        return xr.DataArray(out, coords={d: coords[d] for d in spatial_dims}, dims=spatial_dims, name="QRMSE")
    qA = np.nanpercentile(A, P, axis=0)  # (P, N)
    qB = np.nanpercentile(B, P, axis=0)
    rmse = np.sqrt(((qA - qB) ** 2).mean(axis=0)).astype(np.float32)  # (N,)
    out = rmse.reshape(shape_space)
    return xr.DataArray(out, coords={d: coords[d] for d in spatial_dims}, dims=spatial_dims, name="QRMSE")


def _period_band_to_indices(T: int, period_min: int, period_max: Optional[int]) -> Tuple[int, int]:
    """Translate [period_min, period_max] (months) into rFFT index bounds for length T."""
    pmin = max(2, int(period_min))
    pmax = int(period_max) if period_max else T // 2
    pmax = max(2, min(pmax, T // 2))
    # rFFT frequency k corresponds to period T/k (k>=1)
    kmin = max(1, int(np.floor(T / pmax)))
    kmax = max(kmin, int(np.floor(T / pmin)))
    return kmin, kmax


def _prep_center_window(X: np.ndarray) -> np.ndarray:
    """Center columns (ignore NaNs) and apply Hann window (vectorized) -> float32."""
    mask = np.isfinite(X)
    n = mask.sum(axis=0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nansum(X, axis=0) / np.where(n > 0, n, 1.0)
    Xc = np.where(mask, X - mu, 0.0).astype(np.float32)
    w = np.hanning(X.shape[0]).astype(np.float32)[:, None]
    return Xc * w


def _norm_periodogram(Xw: np.ndarray, kmin: int, kmax: int) -> np.ndarray:
    """Normalized one‑sided periodogram over the selected rFFT bins [kmin..kmax]."""
    F = np.fft.rfft(Xw, axis=0)
    P = (F.real ** 2 + F.imag ** 2).astype(np.float64)
    Pband = P[kmin : (kmax + 1), :]
    S = np.maximum(Pband.sum(axis=0, keepdims=True), 1e-12)
    return (Pband / S).astype(np.float32)  # (K, N)


def spectral_rmse_map(
    a: xr.DataArray,
    b: xr.DataArray,
    *,
    period_min: int = 2,
    period_max: Optional[int] = 120,
) -> xr.DataArray:
    """
    Per‑cell L2 distance between normalized periodograms (timing‑invariant spectral RMSE).
    """
    a, b = xr.align(a, b, join="inner")
    A, spatial_dims, shape_space, coords = _flatten_time_space(a)
    B, _, _, _ = _flatten_time_space(b)
    if A.size == 0 or B.size == 0:
        out = np.full(shape_space, np.nan, dtype=np.float32)
        return xr.DataArray(out, coords={d: coords[d] for d in spatial_dims}, dims=spatial_dims, name="SPEC_RMSE")
    T = A.shape[0]
    if T < 8:
        out = np.full(shape_space, np.nan, dtype=np.float32)
        return xr.DataArray(out, coords={d: coords[d] for d in spatial_dims}, dims=spatial_dims, name="SPEC_RMSE")
    kmin, kmax = _period_band_to_indices(T, period_min, period_max)
    Aw = _prep_center_window(A)
    Bw = _prep_center_window(B)
    PA = _norm_periodogram(Aw, kmin, kmax)  # (K,N)
    PB = _norm_periodogram(Bw, kmin, kmax)
    rmse = np.sqrt(((PA - PB) ** 2).mean(axis=0)).astype(np.float32)  # (N,)
    out = rmse.reshape(shape_space)
    return xr.DataArray(out, coords={d: coords[d] for d in spatial_dims}, dims=spatial_dims, name="SPEC_RMSE")


def dominant_period_map(
    x: xr.DataArray,
    *,
    period_min: int = 2,
    period_max: Optional[int] = 120,
) -> xr.DataArray:
    """
    Dominant (peak) period [months] of x's normalized periodogram in the given band.
    Useful for "persistence" / typical wet/dry spell length diagnostics.
    """
    X, spatial_dims, shape_space, coords = _flatten_time_space(x)
    if X.size == 0:
        out = np.full(shape_space, np.nan, dtype=np.float32)
        return xr.DataArray(out, coords={d: coords[d] for d in spatial_dims}, dims=spatial_dims, name="DOMPER")
    T = X.shape[0]
    if T < 8:
        out = np.full(shape_space, np.nan, dtype=np.float32)
        return xr.DataArray(out, coords={d: coords[d] for d in spatial_dims}, dims=spatial_dims, name="DOMPER")
    kmin, kmax = _period_band_to_indices(T, period_min, period_max)
    Xw = _prep_center_window(X)
    P = _norm_periodogram(Xw, kmin, kmax)  # (K,N)
    k_rel = P.argmax(axis=0)  # (N,)
    k_bins = (kmin + k_rel).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        per = (T / np.maximum(k_bins, 1)).astype(np.float32)
    out = per.reshape(shape_space)
    return xr.DataArray(out, coords={d: coords[d] for d in spatial_dims}, dims=spatial_dims, name="DOMPER")
