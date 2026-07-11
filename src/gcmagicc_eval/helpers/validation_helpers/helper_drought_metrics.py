"""
Shared helpers for monthly dry-spell map segments
=================================================
* cos-lat weights
* lon 0-360 -> -180-180 reorder
* Mollweide map producer (three stacked panels)
"""

from __future__ import annotations
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


# -- weighting & area helpers ------------------------------------------------
def _weights(lat: xr.DataArray) -> xr.DataArray:
    """cos φ weights broadcastable to lat×lon"""
    return np.cos(np.deg2rad(lat))


def _area_w_mean(arr: xr.DataArray, denom: xr.DataArray | None = None) -> float:
    """area-weighted spatial mean of *arr*"""
    w = _weights(arr.lat)
    if denom is None:
        val = (arr * w).mean(("lat", "lon"))
    else:
        val = ((arr / denom) * w).mean(("lat", "lon"))
    return float(val.values)  # scalar


# -- lon re-ordering (needed for Mollweide) ----------------------------------
def _wrap_180(da: xr.DataArray) -> xr.DataArray:
    """shift lon from 0-360 -> -180-180 if necessary; ensures monotonic"""
    if da.lon.max() > 180:
        da = da.roll(lon=(da.lon.size // 2), roll_coords=True)
        da["lon"] = ((da.lon + 180) % 360) - 180
        da = da.sortby("lon")
    return da


# -- Figure helper (3 × Mollweide) -------------------------------------------
def draw_three_panel_map(
    cmip: xr.DataArray,
    cmp: xr.DataArray,
    diff: xr.DataArray,
    label_cmip: str,
    label_cmp: str,
    title: str,
    cmap_base="viridis",
    cmap_diff="coolwarm",
) -> plt.Figure:
    """Return a 3-row Mollweide figure (CMIP6, comparison, difference)."""
    import cartopy.crs as ccrs

    cmip, cmp, diff = [_wrap_180(d) for d in (cmip, cmp, diff)]

    fig, axes = plt.subplots(nrows=3, figsize=(8, 9), subplot_kw=dict(projection=ccrs.Mollweide()))
    vmax = float(max(abs(cmip.max()), abs(cmp.max())))
    diffmax = float(abs(diff).max())
    for ax, da, lab, cmap, vlim in zip(
        axes,
        (cmip, cmp, diff),
        (label_cmip, label_cmp, "Difference"),
        (cmap_base, cmap_base, cmap_diff),
        (vmax, vmax, diffmax),
    ):
        im = ax.pcolormesh(
            da.lon,
            da.lat,
            da,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=-vlim if lab == "Difference" else 0,
            vmax=vlim,
        )
        ax.coastlines(linewidth=0.3)
        ax.set_title(lab, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.7, orientation="horizontal", pad=0.05)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


# -- SPEI fallback (no-numba) ------------------------------------------------
def spei_fallback(pr_mon: xr.DataArray, tas_mon: xr.DataArray, scale: int) -> xr.DataArray:
    """γ-fit SPEI without numba; returns DataArray aligned to pr_mon."""
    from scipy.stats import gamma, norm

    # Thornthwaite PET (very light)
    tas = tas_mon.clip(min=0)
    heat_index = ((tas.groupby("time.year").mean() / 5) ** 1.514).sum("year")
    a = 0.49239 + 1.792e-2 * heat_index - 7.71e-5 * heat_index**2 + 6.75e-7 * heat_index**3
    pet = 16 * ((10 * tas / heat_index) ** a)  # mm month⁻¹  (daylength omitted)

    wb = (pr_mon - pet).rolling(time=scale).sum().dropna("time")

    def _per_mon(arr):
        try:
            # Filter out invalid values for gamma fitting
            valid_data = arr[np.isfinite(arr) & (arr > 0)]
            if len(valid_data) < 10:  # Need sufficient data for fitting
                return np.full_like(arr, np.nan)

            shp, loc, scl = gamma.fit(valid_data, floc=0)
            # Ensure scale is positive
            if scl <= 0:
                return np.full_like(arr, np.nan)

            # Apply transformation only to valid data
            result = np.full_like(arr, np.nan)
            valid_mask = np.isfinite(arr) & (arr > 0)
            result[valid_mask] = norm.ppf(gamma.cdf(arr[valid_mask], shp, loc=0, scale=scl))
            return result
        except Exception:
            # Return NaN array if fitting fails
            return np.full_like(arr, np.nan)

    spei = wb.groupby("time.month").map(_per_mon)
    # Convert time series to spatial map by taking mean over time
    spei = spei.mean("time").rename(f"SPEI{scale}")
    return spei.load()


def _pmesh(ax, da, cmap, vlim, title):
    im = ax.pcolormesh(
        da.lon, da.lat, da, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vlim[0], vmax=vlim[1]
    )
    ax.coastlines(linewidth=0.3)
    ax.set_title(title, fontsize=9)
    return im


def four_panel_dsi_figure(
    window: str,
    stats_cmip: tuple[xr.DataArray, xr.DataArray, xr.DataArray],
    stats_cmp: tuple[xr.DataArray, xr.DataArray, xr.DataArray],
    ts_cmip: xr.DataArray,  # annual global mean series
    ts_cmp: xr.DataArray,
    label_cmip: str,
    label_cmp: str,
) -> plt.Figure:
    """
    Build the 4×3 figure required by DroughtSeverityIndex.py.

    Parameters
    ----------
    window       : str
        Text label for the window (e.g. "2081-2100").
    stats_cmip, stats_cmp
        Tuples of (mean, var, exceed) lat×lon maps.
    ts_cmip, ts_cmp
        Annual global-mean time-series (1-D).
    label_cmip, label_cmp
        Legend labels for the two models.
    """
    mean1, var1, exc1 = [_wrap_180(d) for d in stats_cmip]
    mean2, var2, exc2 = [_wrap_180(d) for d in stats_cmp]

    diff_m, diff_v, diff_e = mean1 - mean2, var1 - var2, exc1 - exc2

    fig = plt.figure(figsize=(10, 11))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.2, 1, 1, 1])

    # -- row 0 - timeseries --------------------------------------------
    ax_ts = fig.add_subplot(gs[0, :])
    ax_ts.plot(ts_cmip.year, ts_cmip, label=label_cmip)
    ax_ts.plot(ts_cmp.year, ts_cmp, label=label_cmp)
    ax_ts.set_title(f"Annual DSI - {label_cmip} vs {label_cmp}")
    ax_ts.set_ylabel("mm")
    ax_ts.legend()
    ax_ts.grid(alpha=0.3)

    # -- helper for three map rows -------------------------------------
    cmap_main, cmap_diff = "inferno", "coolwarm"
    vmax_m = float(max(abs(mean1).max(), abs(mean2).max()))
    vmax_v = float(max(abs(var1).max(), abs(var2).max()))
    vmax_e = 1.0  # frequency 0-1

    rowspec = [
        (mean1, mean2, diff_m, vmax_m, "Mean DSI"),
        (var1, var2, diff_v, vmax_v, "Variance DSI"),
        (exc1, exc2, diff_e, vmax_e, "Exceed-freq (>5 mm)"),
    ]

    for r, (a, b, d, vmax, row_title) in enumerate(rowspec):
        for c, da, lab in [(0, a, label_cmip), (1, b, label_cmp), (2, d, "Δ")]:
            ax = fig.add_subplot(gs[r + 1, c], projection=ccrs.Mollweide())
            vlim = (-vmax, vmax) if lab == "Δ" else (0, vmax)
            cmap = cmap_diff if lab == "Δ" else cmap_main
            im = _pmesh(ax, da, cmap, vlim, lab)
            if c == 2:  # metrics overlay on diff map
                dev, rmse = _area_w_mean(d), _area_w_mean(d**2) ** 0.5
                ax.text(
                    0.02,
                    0.03,
                    f"Dev = {dev:.2f}\nRMSE = {rmse:.2f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    bbox=dict(fc="w", alpha=0.7),
                )
            fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.7)

    fig.suptitle(f"DSI diagnostics - window {window}", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    return fig


def four_panel_pdsi_figure(
    window: str,
    stats_cmip: tuple[xr.DataArray, xr.DataArray, xr.DataArray],
    stats_cmp: tuple[xr.DataArray, xr.DataArray, xr.DataArray],
    ts_cmip: xr.DataArray,  # monthly global mean series
    ts_cmp: xr.DataArray,
    label_cmip: str,
    label_cmp: str,
) -> plt.Figure:
    """
    Build the 4×3 figure required by PalmerDroughtSeverityIndex.py.

    Parameters
    ----------
    window       : str
        Text label for the window (e.g. "2081-2100").
    stats_cmip, stats_cmp
        Tuples of (mean, p10, p90) lat×lon maps.
    ts_cmip, ts_cmp
        Monthly global-mean time-series (1-D).
    label_cmip, label_cmp
        Legend labels for the two models.
    """
    mean1, p10_1, p90_1 = [_wrap_180(d) for d in stats_cmip]
    mean2, p10_2, p90_2 = [_wrap_180(d) for d in stats_cmp]

    diff_m, diff_p10, diff_p90 = mean1 - mean2, p10_1 - p10_2, p90_1 - p90_2

    fig = plt.figure(figsize=(10, 11))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.2, 1, 1, 1])

    # -- row 0 - timeseries --------------------------------------------
    ax_ts = fig.add_subplot(gs[0, :])

    # Convert cftime to fractional years for matplotlib compatibility
    def cftime_to_fractional_year(time_coord):
        """Convert cftime datetime to fractional years."""
        years = []
        for t in time_coord.values:
            # Extract year and month, convert to fractional year
            year = t.year
            month = t.month
            fractional_year = year + (month - 1) / 12.0
            years.append(fractional_year)
        return np.array(years)

    # Convert time coordinates
    ts_cmip_years = cftime_to_fractional_year(ts_cmip.time)
    ts_cmp_years = cftime_to_fractional_year(ts_cmp.time)

    # Plot monthly global mean values
    ax_ts.plot(ts_cmip_years, ts_cmip, label=label_cmip, linewidth=1.5)
    ax_ts.plot(ts_cmp_years, ts_cmp, label=label_cmp, linewidth=1.5)

    # Calculate and plot percentile ranges for both datasets
    try:
        ts_cmip_p10 = ts_cmip.quantile(0.1)
        ts_cmip_p90 = ts_cmip.quantile(0.9)
        ts_cmp_p10 = ts_cmp.quantile(0.1)
        ts_cmp_p90 = ts_cmp.quantile(0.9)

        # Add filled ranges
        ax_ts.fill_between(
            ts_cmip_years,
            ts_cmip_p10,
            ts_cmip_p90,
            alpha=0.3,
            color="blue",
            label=f"{label_cmip} (10th-90th percentile)",
        )
        ax_ts.fill_between(
            ts_cmp_years,
            ts_cmp_p10,
            ts_cmp_p90,
            alpha=0.3,
            color="red",
            label=f"{label_cmp} (10th-90th percentile)",
        )
    except Exception:
        # Fallback if percentile calculation fails
        pass

    ax_ts.set_title(f"Monthly PDSI - {label_cmip} vs {label_cmp}")
    ax_ts.set_ylabel("PDSI")
    ax_ts.set_xlabel("Year")
    ax_ts.legend()
    ax_ts.grid(alpha=0.3)

    # -- helper for three map rows -------------------------------------
    # Custom colormap: red (dry) to white (zero) to blue (wet)
    from matplotlib.colors import LinearSegmentedColormap

    colors_pdsi = ["darkred", "red", "lightcoral", "white", "lightblue", "blue", "darkblue"]
    cmap_pdsi = LinearSegmentedColormap.from_list("pdsi_red_white_blue", colors_pdsi, N=256)

    # Find the maximum absolute value for consistent scaling
    all_data = [mean1, mean2, p10_1, p10_2, p90_1, p90_2, diff_m, diff_p10, diff_p90]
    vmax = max(abs(d.max()) for d in all_data if d.size > 0)
    vmin = -vmax

    rowspec = [
        (mean1, mean2, diff_m, "Mean PDSI"),
        (p10_1, p10_2, diff_p10, "10th Percentile PDSI"),
        (p90_1, p90_2, diff_p90, "90th Percentile PDSI"),
    ]

    for r, (a, b, d, row_title) in enumerate(rowspec):
        for c, da, lab in [(0, a, label_cmip), (1, b, label_cmp), (2, d, "Δ")]:
            ax = fig.add_subplot(gs[r + 1, c], projection=ccrs.Mollweide())
            vlim = (vmin, vmax)  # Same scale for all plots
            cmap = cmap_pdsi

            im = _pmesh(ax, da, cmap, vlim, lab)
            if c == 2:  # metrics overlay on diff map
                dev, rmse = _area_w_mean(d), _area_w_mean(d**2) ** 0.5
                ax.text(
                    0.02,
                    0.03,
                    f"Dev = {dev:.2f}\nRMSE = {rmse:.2f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    bbox=dict(fc="w", alpha=0.7),
                )
            fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.7)

    fig.suptitle(f"PDSI diagnostics - window {window}", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    return fig
