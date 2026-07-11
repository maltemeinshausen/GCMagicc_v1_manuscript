"""
helper_heatmap_plot.py
======================
Region‑scale SPEI/SPI heatmaps (scale × time).
The visual style follows the layout popularized in drought index literature
(cf. Figure 5 in van Mourik et al., 2025, JOSS doi:10.21105/joss.08454),
implemented here independently without external dependencies.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

__all__ = ["plot_region_scale_time_heatmaps"]

def _matrix_from_series(series_by_scale: Dict[int, xr.DataArray]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    if not series_by_scale:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype="datetime64[ns]"), []
    scales = sorted(series_by_scale.keys())
    # intersect time across scales
    times = None
    for s in scales:
        t = series_by_scale[s]["time"].values
        times = t if times is None else np.intersect1d(times, t)
    if times.size == 0:
        # fall back to union with NaNs
        times = np.unique(np.concatenate([series_by_scale[s]["time"].values for s in scales]))
    M = []
    for s in scales:
        ts = series_by_scale[s].sel(time=times, drop=False)
        M.append(np.asarray(ts.values, dtype=np.float32))
    mat = np.vstack(M)  # (S, T)
    return mat, times, scales

def _draw_heatmap(ax, mat: np.ndarray, times: np.ndarray, scales: List[int], *, title: str, vmin=-3, vmax=3):
    if mat.size == 0:
        ax.set_title(title + " (no data)")
        return
    # x: monthly index (0..T-1) to avoid date parsing; label years sparsely
    T = mat.shape[1]
    im = ax.imshow(mat, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(len(scales)))
    ax.set_yticklabels([str(s) for s in scales])
    # crude yearly ticks
    years = np.unique(np.array([np.datetime64(t, "Y").astype(object).year for t in times]))
    xticks = []
    xlabels = []
    for y in years[:: max(1, len(years) // 10) ]:
        # find first occurrence index
        idx = np.argmax(np.array([np.datetime64(t, "Y").astype(object).year for t in times]) == y)
        xticks.append(idx)
        xlabels.append(str(y))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Time scale (months)")
    ax.set_title(title, fontsize=10)
    return im

def plot_region_scale_time_heatmaps(
    series_A: Dict[int, xr.DataArray],
    series_B: Dict[int, xr.DataArray],
    *,
    region: str,
    index_name: str,
    window: str,
    vmin: float = -3.0,
    vmax: float = 3.0,
):
    """Two‑panel heatmap (A vs B) of index across scales for a region."""
    matA, timesA, scalesA = _matrix_from_series(series_A)
    matB, timesB, scalesB = _matrix_from_series(series_B)
    # unify color scale (vmin/vmax) for both panels
    vmin = np.nanpercentile(np.concatenate([matA.ravel(), matB.ravel()]) if matA.size and matB.size else (matA.ravel() if matA.size else matB.ravel()), 1) if (matA.size or matB.size) else vmin
    vmax = np.nanpercentile(np.concatenate([matA.ravel(), matB.ravel()]) if matA.size and matB.size else (matA.ravel() if matA.size else matB.ravel()), 99) if (matA.size or matB.size) else vmax
    fig = plt.figure(figsize=(12, 4.5))
    ax1 = fig.add_subplot(1, 2, 1)
    im1 = _draw_heatmap(ax1, matA, timesA, scalesA, title=f"A {index_name} (region={region}, window={window})", vmin=vmin, vmax=vmax)
    ax2 = fig.add_subplot(1, 2, 2)
    im2 = _draw_heatmap(ax2, matB, timesB, scalesB, title=f"B {index_name} (region={region}, window={window})", vmin=vmin, vmax=vmax)
    # shared colorbar on the right
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cb = fig.colorbar(im2 if im2 is not None else im1, cax=cax, orientation="vertical")
    cb.set_label(f"{index_name} (z)", fontsize=9)
    fig.tight_layout()
    return fig
