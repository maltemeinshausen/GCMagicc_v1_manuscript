# helper_bench_plot.py
#
# This module now contains only generic, low-level plotting utilities
# and file parsing functions that can be used by any segment.
# -----------------------------------------------------------------------------

import os
import re
import glob
from datetime import datetime
import numpy as np
import xarray as xr
import cftime
from pathlib import Path


def compute_rmse_score(series1, series2):
    """Compute RMSE between two time series."""
    if hasattr(series1, "values"):
        series1 = series1.values
    if hasattr(series2, "values"):
        series2 = series2.values
    valid_mask = ~(np.isnan(series1) | np.isnan(series2))
    if not np.any(valid_mask):
        return np.nan
    series1_clean, series2_clean = series1[valid_mask], series2[valid_mask]
    if len(series1_clean) == 0:
        return np.nan
    return float(np.sqrt(np.mean((series1_clean - series2_clean) ** 2)))


def compute_rmse_for_ensemble_means(cmip6_series_list, gcmagicc_series_list):
    """Compute RMSE between ensemble means."""
    if not cmip6_series_list or not gcmagicc_series_list:
        return np.nan

    try:
        # Handle single-member datasets (no ensemble dimension)
        if len(cmip6_series_list) == 1:
            cmip6_mean = cmip6_series_list[0]
        else:
            # Try to concatenate with ensemble dimension
            try:
                cmip6_mean = xr.concat(cmip6_series_list, dim="ensemble").mean(dim="ensemble")
            except ValueError:
                # If ensemble dimension doesn't exist, just take the mean of the list
                cmip6_mean = xr.concat(cmip6_series_list, dim="member").mean(dim="member")

        if len(gcmagicc_series_list) == 1:
            gcmagicc_mean = gcmagicc_series_list[0]
        else:
            try:
                gcmagicc_mean = xr.concat(gcmagicc_series_list, dim="ensemble").mean(
                    dim="ensemble"
                )
            except ValueError:
                gcmagicc_mean = xr.concat(gcmagicc_series_list, dim="member").mean(dim="member")

        # Align to common time
        cmip6_aligned, gcmagicc_aligned = xr.align(cmip6_mean, gcmagicc_mean, join="inner")

        return compute_rmse_score(cmip6_aligned, gcmagicc_aligned)
    except Exception as e:
        print(f"Error in compute_rmse_for_ensemble_means: {e}")
        return np.nan


def convert_cftime_to_numeric(time_coord):
    """Convert cftime objects to numeric values for plotting."""
    if hasattr(time_coord, "values"):
        time_values = time_coord.values
    else:
        time_values = time_coord

    # Check if we have cftime objects
    if len(time_values) > 0 and isinstance(time_values[0], cftime._cftime.Datetime360Day):
        # Convert to fractional years
        numeric_times = []
        for t in time_values:
            if hasattr(t, "year"):
                # Convert to fractional year
                year = t.year
                if hasattr(t, "month"):
                    month = t.month
                    # Approximate fractional year
                    frac_year = year + (month - 1) / 12.0
                else:
                    frac_year = year
                numeric_times.append(frac_year)
            else:
                # Fallback: convert to float if possible
                try:
                    numeric_times.append(float(t))
                except (ValueError, TypeError):
                    numeric_times.append(0.0)
        return np.array(numeric_times)
    else:
        # Already numeric or other format
        return time_values


def add_rmse_text_to_ax(
    ax, rmse_value, position="top_right", fontsize=10, color="red", prefix="RMSE: "
):
    """Add RMSE text to a matplotlib axis, with optional prefix."""
    if rmse_value is None or np.isnan(rmse_value):
        return

    # Format with 4 significant digits using scientific notation
    if rmse_value < 1e-3:
        rmse_text = f"{prefix}{rmse_value:.3e}"
    else:
        rmse_text = f"{prefix}{rmse_value:.3f}"

    positions = {
        "top_right": (0.98, 0.98, "right", "top"),
        "top_left": (0.02, 0.98, "left", "top"),
        "bottom_right": (0.98, 0.02, "right", "bottom"),
        "bottom_left": (0.02, 0.02, "left", "bottom"),
        "center_left": (0.02, 0.5, "left", "center"),
        "center_right": (0.98, 0.5, "right", "center"),
    }
    x, y, ha, va = positions.get(position, positions["top_right"])
    ax.text(
        x,
        y,
        rmse_text,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8DC", alpha=0.9, edgecolor="none"),
        ha=ha,
        va=va,
        zorder=1000,
    )


def extract_gcmagicc_code(filename):
    """
    Extract the GCMagiccCode from a GCMagicc filename.

    The GCMagiccCode is the first part of the filename before the first underscore.
    Example: 'GCMagicc-v100d1b0e34m0-20250718-1030' from
             'GCMagicc-v100d1b0e34m0-20250718-1030_GISS-E2-2-G_historical_r1i1p1f1_psl-tas-pr-sfcWind.nc'

    Parameters
    ----------
    filename : str
        The GCMagicc filename

    Returns
    -------
    str or None
        The GCMagiccCode if found, None otherwise
    """
    if not filename.startswith("GCMagicc"):
        return None

    # Find the first underscore
    underscore_pos = filename.find("_")
    if underscore_pos == -1:
        return None

    return filename[:underscore_pos]


def compose_gcmagicc_model_name(file_path: str, base_model: str | None = None) -> str | None:
    """Return a normalized model identifier for GCMagicc files."""

    prefix = extract_gcmagicc_code(os.path.basename(file_path))
    if not prefix:
        return None
    if base_model:
        return f"{prefix}_{base_model}"
    return prefix


# Optional behavior for GCMagicc files:
#  - use_pseudo_member = True  -> member = f"{mem}__{prefix}"
#  - use_pseudo_member = False -> member = mem (plain CMIP6)
#  - use_pseudo_member = None  -> read env PSEUDO_GCMAGICC_MEMBER (fallback to legacy PSEUDO_NICOLAI_MEMBER)


def parse_filename(fname: str, use_pseudo_member: bool | None = None):
    base = os.path.basename(fname)
    if base.endswith(".nc"):
        base = base[:-3]
    parts = base.split("_")
    if len(parts) < 4:
        raise ValueError(f"Filename does not match expected pattern: {fname}")

    # Resolve toggle
    if use_pseudo_member is None:
        env_flag = os.environ.get("PSEUDO_GCMAGICC_MEMBER")
        if env_flag is None:
            # Backward compatibility with historical configuration
            env_flag = os.environ.get("PSEUDO_NICOLAI_MEMBER", "0")
        use_pseudo_member = str(env_flag) == "1"

    # DAT_* (CMIP6) -> DAT_MODEL_SCENARIO_MEMBER_...
    if base.startswith("DAT_"):
        return parts[1], parts[2], parts[3]  # (source_id, experiment_id, member_id)

    # GCMagicc_* -> GCMagicc-<ver>-<dep>-<YYYYMMDD-HHMM>_<MODEL>_<SCENARIO>_<MEMBER>_...
    # GXE variant (no CMIP6 model embedded): GCMagicc-<ver>-<dep>-<YYYYMonDD>_<HHMM>_<SCENARIO>_<MEMBER>
    if base.startswith("GCMagicc"):
        # GXE compact variant: timestamp chunk then scenario + member
        if len(parts) == 4 and re.match(r"^\d{4}$", parts[1]):
            scen = parts[2]
            mem = parts[3]
            prefix = parts[0]
            model = "GXE"
            if use_pseudo_member:
                pseudo = f"{mem}__{prefix}"
                return model, scen, pseudo
            return model, scen, mem
        src = parts[1]  # CMIP6 source_id
        scen = parts[2]  # experiment_id
        mem = parts[3]  # CMIP6 member_id
        prefix = parts[0]  # GCMagicc prefix (contains timestamp)
        if use_pseudo_member:
            pseudo = f"{mem}__{prefix}"
            return src, scen, pseudo
        else:
            return src, scen, mem

    # Generic fallback (MODEL_SCENARIO_MEMBER_...)
    return parts[1], parts[2], parts[3]


_GCMAGICC_TS_RE = re.compile(r"^GCMagicc-[^-]+-[^-]+-(\d{8}-\d{4})_")


def is_gcmagicc_file(fname: str) -> bool:
    return os.path.basename(fname).startswith("GCMagicc")


def get_gcmagicc_prefix_ts(fname: str):
    """
    Return datetime for GCMagicc prefix timestamp (YYYYMMDD-HHMM) or None.
    """
    base = os.path.basename(fname)
    m = _GCMAGICC_TS_RE.match(base)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d-%H%M")
    except Exception:
        return None


def get_model_details_from_pair(pair, config):
    """
    Extract detailed model information from a pair for use in plot labels.

    Parameters
    ----------
    pair : dict
        Pair dictionary with cmip6_files and gcmagicc_files
    config : dict
        Configuration dictionary with label1 and label2

    Returns
    -------
    dict
        Dictionary with detailed model information for plotting
    """
    model_details = {}

    # Extract CMIP6 model details
    if pair["cmip6_files"]:
        cmip6_file = pair["cmip6_files"][0]["file"]
        cmip6_model, cmip6_scenario, cmip6_ensemble = parse_filename(os.path.basename(cmip6_file))
        model_details["cmip6"] = {
            "source_id": cmip6_model,
            "experiment_id": cmip6_scenario,
            "member_id": cmip6_ensemble,
            "label": config.get("label1", "CMIP6"),
            "full_label": f"{cmip6_model} {cmip6_scenario} {cmip6_ensemble}"
            if cmip6_model
            else config.get("label1", "CMIP6"),
        }

    # Extract GCMagicc model details
    if pair["gcmagicc_files"]:
        gcmagicc_file = pair["gcmagicc_files"][0]["file"]
        gcmagicc_model, gcmagicc_scenario, gcmagicc_ensemble = parse_filename(
            os.path.basename(gcmagicc_file)
        )
        gcmagicc_gcm_code = extract_gcmagicc_code(os.path.basename(gcmagicc_file))
        model_details["gcmagicc"] = {
            "source_id": gcmagicc_model,
            "experiment_id": gcmagicc_scenario,
            "member_id": gcmagicc_ensemble,
            "gcmagicc_gcm_code": gcmagicc_gcm_code,
            "label": config.get("label2", gcmagicc_gcm_code),
            "full_label": f"{gcmagicc_model} {gcmagicc_scenario} {gcmagicc_ensemble}"
            if gcmagicc_model
            else config.get("label2", "GCMagicc"),
        }

    return model_details


def get_paired_files(
    GCMagiccoutputfolder,
    cmip6folder,
    cmip6_filter="",
    use_ensemble=True,
    model_whitelist=None,
    scenario_whitelist=None,
    *,
    allow_gcmagicc_only: bool = False,
    gcmagicc_selection: list[str] | None = None,
    treat_gcmagicc_as_historical: bool = False,
):
    """Get paired files between GCMagicc and CMIP6 outputs."""
    cmip6_files = glob.glob(os.path.join(cmip6folder, "*.nc"))
    if gcmagicc_selection:
        gcmagicc_files = [str(p) for p in gcmagicc_selection]
    else:
        gcmagicc_files = glob.glob(os.path.join(GCMagiccoutputfolder, "*.nc"))
    if cmip6_filter:
        print(f"Filtering CMIP6 files with {cmip6_filter}")
        cmip6_files = [f for f in cmip6_files if cmip6_filter in f]
        print(f"Found {len(cmip6_files)} CMIP6 files")

    cmip6_groups, gcmagicc_groups = {}, {}
    for f in cmip6_files:
        # Always use PLAIN CMIP6 member for pairing keys (immune to env flag)
        model, scenario, ensemble = parse_filename(os.path.basename(f), use_pseudo_member=False)
        if model and scenario:
            # Apply whitelist filtering if provided
            if model_whitelist and model not in model_whitelist:
                continue
            if scenario_whitelist and scenario not in scenario_whitelist:
                continue
            key = (model, scenario, ensemble)
            if key not in cmip6_groups:
                cmip6_groups[key] = []
            cmip6_groups[key].append(
                {"file": f, "model": model, "scenario": scenario, "ensemble": ensemble}
            )

    for f in gcmagicc_files:
        # Force plain member here too; NN will use pseudo members at write time
        model, scenario, ensemble = parse_filename(os.path.basename(f), use_pseudo_member=False)
        if model and scenario:
            # Apply whitelist filtering if provided
            if model_whitelist and model not in model_whitelist:
                continue
            if scenario_whitelist and scenario not in scenario_whitelist:
                continue
            key = (model, scenario, ensemble)
            if key not in gcmagicc_groups:
                gcmagicc_groups[key] = []
            gcmagicc_groups[key].append(
                {"file": f, "model": model, "scenario": scenario, "ensemble": ensemble}
            )

    paired_files = []
    for key in gcmagicc_groups:  # Iterate over GCMagicc files to find matches in CMIP6
        if key in cmip6_groups:
            paired_files.append(
                {
                    "key": key,
                    "cmip6_files": cmip6_groups[key],
                    "gcmagicc_files": gcmagicc_groups[key],
                    "gcmagicc_only": False,
                    "treat_as_historical_for_era5": False,
                }
            )
        elif allow_gcmagicc_only:
            paired_files.append(
                {
                    "key": key,
                    "cmip6_files": [],
                    "gcmagicc_files": gcmagicc_groups[key],
                    "gcmagicc_only": True,
                    "treat_as_historical_for_era5": bool(treat_gcmagicc_as_historical),
                }
            )
    return paired_files


_SAFE_SEG_MAX = 64
_UNSAFE = re.compile(r"[^A-Za-z0-9._+%-]+")


def _strip_time_tag(version_tag: str) -> str:
    """
    Strip time tag from version_tag to group runs by model version.

    Examples:
        "Nextvers5hist17Aug_18Aug2025_1321" -> "Nextvers5hist17Aug"
        "Nextvers5hist17Aug_17Aug2025_1142" -> "Nextvers5hist17Aug"
        "Nextvers5hist17Aug" -> "Nextvers5hist17Aug" (no time tag to strip)
    """
    if not version_tag:
        return version_tag

    # Look for pattern: modelname_Date_Time (e.g., Nextvers5hist17Aug_18Aug2025_1321)
    # Split by underscore and check if last two parts look like date and time
    parts = version_tag.split("_")
    if len(parts) >= 3:
        # Check if the last two parts look like date and time
        # Date pattern: DDMMMYYYY (e.g., 18Aug2025)
        # Time pattern: HHMM (e.g., 1321)
        last_part = parts[-1]
        second_last_part = parts[-2]

        # Check if last part is 4 digits (time)
        if len(last_part) == 4 and last_part.isdigit():
            # Check if second last part looks like a date (contains letters and numbers)
            if any(c.isalpha() for c in second_last_part) and any(
                c.isdigit() for c in second_last_part
            ):
                # Strip the last two parts (date and time)
                return "_".join(parts[:-2])

    return version_tag


def _slug_segment(value: str, max_len: int = _SAFE_SEG_MAX) -> str:
    """
    Make a filesystem-safe, reasonably short path segment. Preserves dots/hyphens/underscores.
    """
    s = str(value or "").strip()
    # Guard against accidental path traversal
    s = s.replace(os.sep, "_")
    if os.altsep:
        s = s.replace(os.altsep, "_")
    s = _UNSAFE.sub("_", s)
    if not s:
        s = "NA"
    if len(s) > max_len:
        # Keep head/tail for readability; avoid Unicode ellipsis for maximum portability
        s = f"{s[:max_len-15]}---{s[-12:]}"
    return s


def generate_pdf_filename(
    metric_record: dict,
    output_dir: str,
    metrictype: str = None,
    detimetag_versiontag: bool = False,
) -> str:
    """
    Return a nested **output path** ending in 'figure.pdf', creating parent folders.

    New behavior (replaces long flat filenames):
      {output_dir}/
        {metrickey}/
          {version_tag|versionless}/
            {metricdomain}/{variable}/
              {source_id}/{member_id}/{experiment_id}/{comp_source_id}/{comp_member_id}/figure.pdf

    Notes:
      * 'version_tag' is taken from metric_record if present; otherwise the folder 'versionless' is used.
      * We do **not** include 'metrictype' in the path because one figure summarizes multiple metrics.
      * All segments are sanitized and length-limited for filesystem safety.
      * If detimetag_versiontag=True, time tags are stripped from version_tag to group runs by model version.
    """
    # Extract components (fallbacks keep paths predictable)
    metrickey = _slug_segment(metric_record.get("metrickey", "UNKNOWN"))
    metricdomain = _slug_segment(metric_record.get("metricdomain", "UNKNOWN"))
    variable = _slug_segment(metric_record.get("variable", "UNKNOWN"))
    source_id = _slug_segment(metric_record.get("source_id", "UNKNOWN"))
    member_id = _slug_segment(metric_record.get("member_id", "UNKNOWN"))
    experiment_id = _slug_segment(metric_record.get("experiment_id", "UNKNOWN"))
    comp_source_id = _slug_segment(metric_record.get("comp_source_id", "UNKNOWN"))
    comp_member_id = _slug_segment(metric_record.get("comp_member_id", "UNKNOWN"))

    # Handle version_tag with optional time tag stripping
    raw_version_tag = metric_record.get("version_tag", "versionless")
    if detimetag_versiontag and raw_version_tag != "versionless":
        version_tag = _slug_segment(_strip_time_tag(raw_version_tag))
    else:
        version_tag = _slug_segment(raw_version_tag)

    base = Path(output_dir)
    outdir = (
        base
        / metrickey
        / version_tag
        / metricdomain
        / variable
        / source_id
        / member_id
        / experiment_id
        / comp_source_id
        / comp_member_id
    )
    outdir.mkdir(parents=True, exist_ok=True)
    return str(outdir / "figure.pdf")


def get_standard_colors():
    """Return a dictionary of standard colors for plotting."""
    return {
        "cmip6": "#3693ba",  # Blue (from .helper_benchmark)
        "gcmagicc": "#f5ba67",  # Orange (from .helper_benchmark)
        "cmip6_alpha": "#77c4e6",  # Light blue (from .helper_benchmark)
        "gcmagicc_alpha": "#e6c377",  # Light orange (from .helper_benchmark)
        "cmip6_dark": "#215c75",  # Dark blue (from .helper_benchmark)
        "gcmagicc_dark": "#7a5a2d",  # Dark orange (from .helper_benchmark)
        "reference": "#2ca02c",  # Green
        "observation": "#d62728",  # Red
        "ensemble": "#9467bd",  # Purple
        "trend": "#8c564b",  # Brown
        "anomaly": "#e377c2",  # Pink
        "climatology": "#7f7f7f",  # Gray
        "threshold": "#bcbd22",  # Olive
        "index": "#17becf",  # Cyan
    }


def get_common_variables_from_pair(pair, blacklist=None):
    """Get common variables for a pair of datasets."""
    try:
        # Ensure file lists are not empty
        if not pair["cmip6_files"] or not pair["gcmagicc_files"]:
            return []

        with xr.open_dataset(pair["cmip6_files"][0]["file"], use_cftime=True) as ds1:
            vars1 = set(ds1.data_vars.keys())
        with xr.open_dataset(pair["gcmagicc_files"][0]["file"], use_cftime=True) as ds2:
            vars2 = set(ds2.data_vars.keys())
        common_vars = list(vars1.intersection(vars2))
        if blacklist:
            common_vars = [v for v in common_vars if v not in blacklist]
        return common_vars
    except Exception as e:
        print(f"Warning: Could not get common variables for pair {pair.get('key')}. Error: {e}")
        return []


def process_gcmagicc_gcm_files(pair, config, debug=False):
    """
    Centralized function to process GCMagicc files for a pair.

    This function handles the case where there are multiple GCMagicc files
    and multiple CMIP6 files for the same (model, scenario, ensemble) combination.
    It processes each file combination separately and yields individual results.

    Parameters
    ----------
    pair : dict
        Pair dictionary with cmip6_files and gcmagicc_files
    config : dict
        Configuration dictionary
    debug : bool
        Enable debug output

    Yields
    ------
    tuple : (gcmagicc_gcm_code, ds_cmip6, ds_gcmagicc)
        For each file combination, yields the GCM code and the aligned datasets
    """
    import xarray as xr
    from coordinate_utils import ensure_consistent_coordinates

    # Get file lists
    cmip6_files = [f["file"] for f in pair["cmip6_files"]]
    gcmagicc_files = [f["file"] for f in pair["gcmagicc_files"]]

    # Process each CMIP6 file separately
    for cmip6_file in cmip6_files:
        try:
            # Open individual CMIP6 file
            ds_cmip6 = xr.open_dataset(cmip6_file)

            if debug:
                print(f"     ✓ opened CMIP6 file {os.path.basename(cmip6_file)}")

            # Process each GCMagicc file separately
            for gcmagicc_file in gcmagicc_files:
                try:
                    # Open individual GCMagicc file
                    ds_gcmagicc = xr.open_dataset(gcmagicc_file)
                    gcmagicc_gcm_code = extract_gcmagicc_code(os.path.basename(gcmagicc_file))

                    if debug:
                        print(f"     ✓ opened GCMagicc file {os.path.basename(gcmagicc_file)}")

                    # Handle ensemble dimensions if present
                    ds_cmip6_processed = ds_cmip6.copy()
                    ds_gcmagicc_processed = ds_gcmagicc.copy()

                    for _lab, ds in [
                        ("CMIP6", ds_cmip6_processed),
                        ("GCMagicc", ds_gcmagicc_processed),
                    ]:
                        if "ensemble" in ds.dims:
                            ds = ds.mean("ensemble")
                            if debug:
                                print(f"   * {_lab} - averaged over ensemble members")

                    # Ensure coordinate consistency
                    ds_cmip6_aligned, ds_gcmagicc_aligned = ensure_consistent_coordinates(
                        ds_cmip6_processed, ds_gcmagicc_processed
                    )

                    # The code `yield` in Python is used inside a generator function to produce a series of
                    # values. When the generator function is called, it returns an iterator object but does
                    # not start execution immediately. The `yield` keyword is used to yield a value from the
                    # generator function, and the function's state is saved. The next time the generator
                    # function is called, it resumes execution from where it left off.
                    yield gcmagicc_gcm_code, ds_cmip6_aligned, ds_gcmagicc_aligned

                except Exception as e:
                    if debug:
                        print(
                            f"Could not process GCMagicc file {os.path.basename(gcmagicc_file)}: {e}"
                        )
                    continue
                finally:
                    if "ds_gcmagicc" in locals():
                        ds_gcmagicc.close()

        except Exception as e:
            if debug:
                print(f"Could not load CMIP6 file {os.path.basename(cmip6_file)}: {e}")
            continue
        finally:
            if "ds_cmip6" in locals():
                ds_cmip6.close()


def create_metric_name_with_gcm_code(base_name, gcmagicc_gcm_code):
    """
    Create a metric name that includes the GCMagicc code.

    Parameters
    ----------
    base_name : str
        Base metric name (e.g., "SpatialAbsDev_Map_Mean_tas")
    gcmagicc_gcm_code : str
        GCMagicc code (e.g., "GCMagicc-v100d1b0e34m0-20250718-1030")

    Returns
    -------
    str
        Metric name with GCM code appended
    """
    if gcmagicc_gcm_code:
        return f"{base_name}_{gcmagicc_gcm_code}"
    else:
        return base_name
