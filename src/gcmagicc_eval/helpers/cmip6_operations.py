# %%
# Standard library imports
from __future__ import annotations

import hashlib

import os
import re
import sys
from datetime import datetime
import ipynbname

from pathlib import Path
import fsspec                     # gcsfs, s3fs, local FS under one roof


# Third-party library imports for data manipulation and analysis
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import zarr
from netCDF4 import Dataset


# ----------------------------------------------------------------------
# Force all ESMF / xESMF log files into one predictable directory
# ----------------------------------------------------------------------

# Your desired directory (relative to repo root, works on any machine)
ESMF_LOG_DIR = Path(__file__).resolve().parents[2] / "notebooks" / "process_logs"
ESMF_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Use ESMF_LOGDIR instead of ESMF_LOGFILE for better parallel handling
os.environ["ESMF_LOGDIR"] = str(ESMF_LOG_DIR)

import xesmf as xe

# Time handling imports
import cftime
import nc_time_axis
import scipy.ndimage

# Visualization library imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Data intake and preprocessing imports
import intake
import xmip

# Utilities
from dask.delayed import delayed
# Initialize Dask client at the top of your notebook
from dask.distributed import Client, wait, progress
import dask
import dask.array as da
#client = Client()
from tqdm.auto import tqdm 
from tqdm import tqdm
import difflib

# parallel processing
from joblib import Parallel, delayed, parallel_backend

# from pyts.decomposition import SingularSpectrumAnalysis
from statsmodels.nonparametric.smoothers_lowess import lowess

from functools   import partial
from itertools   import starmap
from typing      import Any, Iterable, Dict, List

import logging
import joblib


import warnings



# %%
# print(f'17 June - 3-15pm')
print(f'10 june 25 - 6am')


# choose the backend.. 
# backend="threading", "loky" or "multiprocessing"
CHOSEN_BACKEND = "loky"


# ------------------------------------------------------------------
# ✨ NEW GENERIC I/O LAYER ------------------------------------------
# ------------------------------------------------------------------

TIME_RE = re.compile(r"_(\d{6})-(\d{6})\\.nc$")

def _time_key(p: Path) -> int:
    """Sort helper – returns first YYYYMM as int or 0."""
    m = TIME_RE.search(p.name)
    return int(m.group(1)) if m else 0

def expand_time_series_nc(first_file: str) -> list[str]:
    """
    Given one NetCDF path on *fog* return **all** consecutive pieces belonging
    to the same run, sorted chronologically.

    Works by: strip the trailing `_YYYYMM-YYYYMM.nc`, glob in the directory,
    filter files that share the same stem, sort by first timestamp.
    """
    p = Path(first_file)
    m = TIME_RE.search(p.name)
    if not m:          # no timestring → single file only
        return [str(p)]
    stem = p.name[: m.start()]   # everything up to '_185001-202012'
    pattern = f"{stem}_*-*.nc"
    parts = sorted(p.parent.glob(pattern), key=_time_key)
    return [str(fp) for fp in parts]

# >>> customise this once, then forget about it
_NET_ATMOS_ROOT = "data/cmip6_source"

def _localise(path: str) -> str:
    """
    If *path* starts with '/net/atmos', prepend the local root so that

        /net/atmos/foo/bar.nc    →
        data/cmip6_source/net/atmos/foo/bar.nc
    """
    if path.startswith("/net/atmos"):
        # strip leading "/" so we don’t lose the ‘net’ directory
        return os.path.join(_NET_ATMOS_ROOT, path.lstrip("/"))
    return path


# not really needed.. but called optionally in open_any_cmip. 
def debug_pipe_concatenation(nc_paths):
    """
    Visual debugging for pipe-separated NetCDF files.
    Checks and visualizes time coverage for each chunk.
    """
    time_starts = []
    time_ends   = []
    time_counts = []
    file_labels = []

    print("\n── Verifying individual files ─────────────────────")
    for p in nc_paths:
        exists = os.path.exists(p)
        size = os.path.getsize(p) if exists else "N/A"
        try:
            ftype = subprocess.check_output(["file", p]).decode("utf-8").strip()
        except Exception as e:
            ftype = f"Error: {e}"
        print(f"{p}  →  exists: {exists}, size: {size}, type: {ftype}")

        

    for p in nc_paths:
        try:
            with xr.open_dataset(p, decode_times=True, engine="h5netcdf") as ds:
                if "time" in ds:
                    t = ds["time"]
                    time_starts.append(t[0].values)
                    time_ends.append(t[-1].values)
                    time_counts.append(len(t))
                    file_labels.append(os.path.basename(p))
                else:
                    warnings.warn(f"File {p} has no 'time' coordinate.")
        except Exception as e:
            warnings.warn(f"Failed to open {p}: {e}")
            continue

    # Timeline plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))

    for label, start, end in zip(file_labels, time_starts, time_ends):
        ax[0].plot([start, end], [1, 1], marker="|", label=label)
    ax[0].set_title("Time coverage per file")
    ax[0].set_yticks([])
    ax[0].grid(True)

    # Histogram of number of time steps
    ax[1].hist(time_counts, bins=range(0, max(time_counts) + 10, 10),
               color="grey", edgecolor="black")
    ax[1].set_title("Number of time steps per file")
    ax[1].set_xlabel("Time steps")
    ax[1].set_ylabel("File count")

    plt.tight_layout()
    plt.show()

    # Check for gaps or overlaps
    sorted_times = sorted(zip(time_starts, time_ends))
    for (end1, start2) in zip([e for _, e in sorted_times[:-1]],
                              [s for s, _ in sorted_times[1:]]):
        if start2 > end1:
            warnings.warn(f"Gap detected between {end1} and {start2}")
        elif start2 < end1:
            warnings.warn(f"Overlap detected between {end1} and {start2}")



def open_any_cmip(path_or_url: str, debug_option: bool = False, **xr_kw):
    """
    Unified opener for CMIP data that supports:
    ───────────────────────────────────────────
    • Zarr stores (.zarr local or on GCS)
    • NetCDF files under /net/atmos (rewritten to local root)
    • Pipe-separated lists of NetCDF files
    • Plain local NetCDF files

    Parameters
    ----------
    path_or_url : str
        File path or pipe-separated list of NetCDFs.
    debug_option : bool
        If True and pipe-separated, plot and check time coverage across chunks.
    """
    consolidated = xr_kw.pop("consolidated", None)

    def _localise(p):
        return p.replace(
            "/net/atmos",
            os.environ.get("_NET_ATMOS_ROOT", "data/cmip6_source/net/atmos")
        )

    # ── 1. Pipe-separated list of .nc files ──────────────────────
    if "|" in path_or_url:
        paths = sorted(_localise(p) for p in path_or_url.split("|"))

        if debug_option:
            print(f'Paths to be opened now: {paths}') 
            debug_pipe_concatenation(paths)

        # note.. the engine here is netcdf4, as h5netcdf might not be able to handle by_coords coordination. 
        return xr.open_mfdataset(
                paths, engine="netcdf4", combine="by_coords", **xr_kw
            )

    # ── 2. Zarr store ─────────────────────────────────────────────
    if path_or_url.endswith(".zarr") or path_or_url.startswith("gs://"):
        if path_or_url.startswith("gs://"):
            import gcsfs
            mapper = gcsfs.GCSFileSystem().get_mapper(path_or_url)
        else:
            mapper = fsspec.get_mapper(_localise(path_or_url))

        return xr.open_zarr(
            mapper,
            consolidated=False if consolidated is None else consolidated,
            **xr_kw,
        )

    # ── 3. Single NetCDF file ─────────────────────────────────────
    if path_or_url.endswith(".nc"):
        return xr.open_dataset(_localise(path_or_url),
                               engine="h5netcdf",
                               **xr_kw)

    # ── 4. Fallback ───────────────────────────────────────────────
    return xr.open_dataset(_localise(path_or_url), **xr_kw)



def open_any_cmip_withoutdebug(path_or_url: str, **xr_kw):
    """
    Unified opener for CMIP data that supports
    ───────────────────────────────────────────
    • Zarr stores (`*.zarr` local or on GCS)  
    • NetCDF files under /net/atmos (now localised)  
    • Pipe-separated lists of NetCDF files  
    • Plain local NetCDF files

    Localisation rule
    -----------------
    Any pathname beginning with **/net/atmos** is internally rewritten
    to point into ``/_NET_ATMOS_ROOT/net/atmos/...`` before opening.
    """
    consolidated = xr_kw.pop("consolidated", None)

    # --------------------------------------------------- 1. Pipe-separated list
    if "|" in path_or_url:
        paths = sorted(_localise(p) for p in path_or_url.split("|"))
        return xr.open_mfdataset(
            paths, engine="h5netcdf", combine="by_coords", **xr_kw
        )

    # --------------------------------------------------- 2. Zarr stores
    if path_or_url.endswith(".zarr") or path_or_url.startswith("gs://"):
        if path_or_url.startswith("gs://"):
            import gcsfs
            mapper = gcsfs.GCSFileSystem().get_mapper(path_or_url)
        else:                                      # local Zarr → maybe localised
            mapper = fsspec.get_mapper(_localise(path_or_url))

        return xr.open_zarr(
            mapper,
            consolidated=False if consolidated is None else consolidated,
            **xr_kw,
        )

    # --------------------------------------------------- 3. Single NetCDF file
    if path_or_url.endswith(".nc"):
        return xr.open_dataset(_localise(path_or_url),
                               engine="h5netcdf",
                               **xr_kw)

    # --------------------------------------------------- 4. Fallback (rare)
    return xr.open_dataset(_localise(path_or_url), **xr_kw)


def open_any_cmip_withSSHFS(path_or_url: str, **xr_kw):
    """
    Unified opener for CMIP data that transparently supports

    • Zarr stores (`*.zarr`, local or remote, incl. Google Cloud Storage)  
    • NetCDF files on fog2.ethz.ch (default), reached via SSH/SFTP  
    • Pipe-separated lists of NetCDF files (to stitch consecutive chunks)  
    • Plain local NetCDF files

    Parameters
    ----------
    path_or_url : str
        * A single file path or URL:
          - `/net/atmos/.../file.nc`
          - `/local/path/file.nc`
          - `/net/atmos/.../model.zarr`
          - `gs://bucket/pfx/model.zarr`
        * Or a pipe-separated list of NetCDF paths:
          ``"/net/atmos/.../1850.nc|/net/atmos/.../1900.nc"``

    **xr_kw
        Additional keyword arguments forwarded to
        :pyfunc:`xarray.open_dataset`, :pyfunc:`xarray.open_mfdataset`,
        or :pyfunc:`xarray.open_zarr` as appropriate.  If the caller gives
        a ``consolidated`` flag it is only applied when opening Zarr.

    Returns
    -------
    xarray.Dataset
        The opened dataset (or multifile dataset) ready for further
        processing.

    Notes
    -----
    * Remote NetCDF access is streamed through ``fsspec`` + SFTP and
      read with the **h5netcdf** backend, which accepts arbitrary
      file-like objects.  ``netcdf4`` is used only for local files.
    * Authentication is by SSH key *id_ed25519_fog* **or** SSH agent.
      If your key is loaded via ``ssh-add`` you may omit
      ``key_filename`` below.
    * The function makes no attempt to cache remote files; every open
      goes through SSH.

    Examples
    --------
    >>> ds = open_any_cmip("/net/atmos/data/cmip6/CMIP6/CMIP/..."
    ...                    "tas_Amon_Model_hist_r1i1p1f1_gn_185001-189912.nc")
    >>> ds = open_any_cmip("file1.nc|file2.nc|file3.nc", chunks={"time": 120})
    >>> ds = open_any_cmip("gs://cmip6-pds/CMIP6/.../model.zarr",
    ...                    consolidated=True)
    """
    SSH_HOST = os.environ.get("CMIP6_REMOTE_HOST", os.environ.get("CMIP6_SSH_HOST", "fog2.ethz.ch"))
    SSH_USER = os.environ.get("CMIP6_SSH_USER", "maltemh")

    # Extract and hold 'consolidated' so we don't pass it to NetCDF openers
    consolidated = xr_kw.pop("consolidated", None)

    # ------------------------------------------------------------------ helpers
    def ssh_fs():
        """Return an fsspec SSHFileSystem configured for selected CMIP6 host."""
        return fsspec.filesystem(
            "ssh",
            host=SSH_HOST,
            username=SSH_USER,
            key_filename=os.path.expanduser(os.environ.get("CMIP6_SSH_KEY", "~/.ssh/id_ed25519_fog")),
            allow_agent=True,
            look_for_keys=True,
        )

    def to_ssh_url(p: str) -> str:
        """Convert absolute fog path to ssh:// URI."""
        return f"ssh://{SSH_HOST}{p}"

    # ---------------------------------------------------------------- 1. pipes
    if "|" in path_or_url:
        paths = sorted(path_or_url.split("|"))
        if all(p.startswith("/net/atmos") for p in paths):
            fs = ssh_fs()
            files = [fs.open(to_ssh_url(p), "rb") for p in paths]
            return xr.open_mfdataset(
                files,
                engine="h5netcdf",        # file-like objects → h5netcdf
                combine="by_coords",
                **xr_kw,
            )
        # local multifile
        return xr.open_mfdataset(paths, combine="by_coords", **xr_kw)

    # ---------------------------------------------------------------- 2. Zarr
    if path_or_url.endswith(".zarr") or path_or_url.startswith("gs://"):
        if path_or_url.startswith("/net/atmos"):
            fs = ssh_fs()
            mapper = fs.get_mapper(to_ssh_url(path_or_url))
        elif path_or_url.startswith("gs://"):
            import gcsfs
            mapper = gcsfs.GCSFileSystem().get_mapper(path_or_url)
        else:  # local .zarr
            mapper = fsspec.get_mapper(path_or_url)

        return xr.open_zarr(
            mapper,
            consolidated=False if consolidated is None else consolidated,
            **xr_kw,
        )

    # ---------------------------------------------------------------- 3. single remote NetCDF
    if path_or_url.endswith(".nc") and path_or_url.startswith("/net/atmos"):
        fs = ssh_fs()
        return xr.open_dataset(
            fs.open(to_ssh_url(path_or_url), "rb"),
            engine="h5netcdf",
            **xr_kw,
        )

    # ---------------------------------------------------------------- 4. local NetCDF fallback
    return xr.open_dataset(path_or_url, **xr_kw)

# %%


def check_availability(
    col,
    variable_table_list,
    limit_to_first_ensemble_member=False,
    whitelist_source_id=None,
    blacklist_source_id=None,
    whitelist_experiment_id=None,
):
    """
    Build availability table for (source_id, experiment_id, member_id) rows and
    requested variables as columns.

    This version is vectorized: it performs one filtered search and one grouped
    aggregation, avoiding repeated full-DataFrame scans per row/variable.
    """
    # 1) Apply source/experiment filters
    source_id_filter = col.df['source_id']
    if whitelist_source_id:
        source_id_filter = source_id_filter[source_id_filter.isin(whitelist_source_id)]
    if blacklist_source_id:
        source_id_filter = source_id_filter[~source_id_filter.isin(blacklist_source_id)]

    source_ids = source_id_filter.unique()
    experiment_ids = (
        col.df['experiment_id'].unique()
        if not whitelist_experiment_id
        else whitelist_experiment_id
    )

    # 2) Single broad search
    search_criteria = {
        'source_id': source_ids,
        'experiment_id': experiment_ids,
        'variable_id': [var for _, var in variable_table_list],
        'table_id': [table_id for table_id, _ in variable_table_list],
    }
    search_results = col.search(**search_criteria)

    # Keep only pairs explicitly requested in variable_table_list.
    pair_df = pd.DataFrame(variable_table_list, columns=['table_id', 'variable_id']).drop_duplicates()
    df = search_results.df.merge(pair_df, on=['table_id', 'variable_id'], how='inner')

    # 3) Precompute available members per (source_id, experiment_id)
    members_by_ms = df.groupby(['source_id', 'experiment_id'], sort=False)['member_id'].agg(lambda s: pd.unique(s))

    triplets = []
    regex_pattern = re.compile(r'r\d+i1p1f1')
    for model in source_ids:
        for scenario in experiment_ids:
            all_members = members_by_ms.get((model, scenario), np.array([], dtype=object))
            if len(all_members) == 0:
                continue

            if limit_to_first_ensemble_member:
                if 'r1i1p1f1' in all_members:
                    unique_members = ['r1i1p1f1']
                else:
                    filtered_members = [m for m in all_members if regex_pattern.match(m)]
                    unique_members = [filtered_members[0]] if filtered_members else []
            else:
                unique_members = list(all_members)

            for member in unique_members:
                triplets.append((model, scenario, member))

    # 4) Aggregate once: all files per exact (table_id, variable_id)
    agg = (
        df.groupby(
            ['source_id', 'experiment_id', 'member_id', 'table_id', 'variable_id'],
            sort=False,
        )['zstore']
        .agg(lambda s: '|'.join(sorted(s.tolist())))
        .reset_index()
    )
    agg['pair_key'] = agg['table_id'] + '::' + agg['variable_id']

    wide = agg.pivot(
        index=['source_id', 'experiment_id', 'member_id'],
        columns='pair_key',
        values='zstore',
    )

    # 5) Build output with same semantics as original implementation
    out = pd.DataFrame(triplets, columns=['source_id', 'experiment_id', 'member_id'])
    if out.empty:
        out = out.set_index(['source_id', 'experiment_id', 'member_id'])
    else:
        out = out.drop_duplicates().set_index(['source_id', 'experiment_id', 'member_id'])
        out = out.join(wide, how='left')

    # Same column semantics: assign in requested order; repeated var names overwrite.
    for table_id, var in variable_table_list:
        key = f'{table_id}::{var}'
        out[var] = out[key] if key in out.columns else None

    ordered_vars = list(dict.fromkeys([var for _, var in variable_table_list]))
    out = out[ordered_vars]
    return out
# %%
def filter_files2crunch_df(files2crunch_df, filter_rows_files2crunch):
    # If crunch_rows_files2crunch is 'all', return the entire DataFrame
    if filter_rows_files2crunch == 'all':
        return files2crunch_df

        # If crunch_rows_files2crunch is an integer, convert to a list with one element
    elif isinstance(filter_rows_files2crunch, int):
        return files2crunch_df.iloc[[filter_rows_files2crunch]]

    # If crunch_rows_files2crunch is a list, filter by rows
    elif isinstance(filter_rows_files2crunch, list):
        # Check if the list contains integers or tuples (for multi-index)
        if all(isinstance(item, int) for item in filter_rows_files2crunch):
            return files2crunch_df.iloc[filter_rows_files2crunch]
        elif all(isinstance(item, tuple) for item in filter_rows_files2crunch):
            return files2crunch_df.loc[filter_rows_files2crunch]
        else:
            raise ValueError("List items in filter_rows_files2crunch must be all integers or all tuples")

 
    # If crunch_rows_files2crunch is a dictionary, filter by multi-index criteria
    elif isinstance(filter_rows_files2crunch, dict):
        mask = pd.Series([True] * len(files2crunch_df), index=files2crunch_df.index)

        for key, value in filter_rows_files2crunch.items():
            if isinstance(value, list):
                mask &= files2crunch_df.index.get_level_values(key).isin(value)
            else:
                mask &= files2crunch_df.index.get_level_values(key) == value

        return files2crunch_df[mask]

    else:
        raise ValueError("Invalid input for filter_files2crunch_df")


# Example usage:
# filtered_df = filter_files2crunch_df(files2crunch_df, [1,3,5])
# filtered_df = filter_files2crunch_df(files2crunch_df, 'all')
# filtered_df = filter_files2crunch_df(files2crunch_df, {'source_id': 'HadGEM3-GC31-MM', 'experiment_id': ['historical', 'ssp585']})


# %% [markdown]
# ## some observational data helper functions. 

# %%
def generate_obs_data_dict(df, obs_folder, obsflag):
    """
    Generates a dictionary with observation data paths based on available variable names in the dataframe
    and the specified reanalysis dataset. If 'rsdt', 'rsut', and 'rlut' are not all found, it will add 'rsnt'.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the CMIP6 data paths with MultiIndex.
    obs_folder (str): The folder where processed observation data is stored.
    obsflag (str): The observation dataset flag ('ERA5' or '20CR').

    Returns:
    dict: Dictionary with variable names as keys and lists of file paths as values.
    """
    # Initialize an empty dictionary to hold data paths
    data = {}

    # Track whether we need to add 'rsnt' if 'rsdt', 'rsut', and 'rlut' are not all found
    add_rsnt = False

    # Loop through each column in the dataframe which corresponds to variable names
    for variable in df.columns:
        # Construct the filename from the variable name
        file_name = f"{obsflag}_best_{variable}.nc"
        file_path = os.path.join(obs_folder, file_name)

        # Check if the file exists
        if os.path.exists(file_path):
            data[variable] = [file_path]
        else:
            print(f"File not found for variable {variable}: {file_path}")
            data[variable] = [None]  # or you can choose to not add it to the dictionary

            # Check if the missing variable is one of 'rsdt', 'rsut', or 'rlut'
            if variable in ['rsdt', 'rsut', 'rlut']:
                add_rsnt = True

    # If 'rsdt', 'rsut', and 'rlut' were searched for but not all were found, add 'rsnt'
    if add_rsnt:
        file_path_rsnt = os.path.join(obs_folder, f"{obsflag}_best_rsnt.nc")
        if os.path.exists(file_path_rsnt):
            data['rsnt'] = [file_path_rsnt]
        else:
            print(f"File not found for 'rsnt': {file_path_rsnt}")
            data['rsnt'] = [None]  # or you can choose to not add it to the dictionary

    return data



# %%
def create_obs_data_frame(data, obs_folder, obsflag):
    """
    Creates a DataFrame from the observational/reanalysis data dictionary with proper MultiIndex.

    Parameters:
    data (dict): Dictionary with variable names and corresponding file paths.
    obs_folder (str): Default folder path, used if data paths are missing.

    Returns:
    pandas.DataFrame: DataFrame containing the OBS data paths.
    """
    # Create a MultiIndex for the DataFrame
    index = pd.MultiIndex.from_tuples([(obsflag, 'historical-'+obsflag, 'r1i1p1f1')], names=['source_id', 'experiment_id', 'member_id'])

    # Create the DataFrame using the data dictionary and the constructed index
    files2crunch_df = pd.DataFrame(data, index=index)
    
    return files2crunch_df


# %%
# get a comparable ERA5 file.. 

def generate_era5_data_dict(df, era5_folder):
    """
    Generates a dictionary with ERA5 data paths based on available variable names in the dataframe.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the CMIP6 data paths with MultiIndex.
    era5_folder (str): The folder where ERA5 processed data is stored.

    Returns:
    dict: Dictionary with variable names as keys and lists of file paths as values.
    """
    # Initialize an empty dictionary to hold data paths
    data = {}

    # Loop through each column in the dataframe which corresponds to variable names
    for variable in df.columns:
        # Construct the filename from the variable name
        file_name = f"ERA5_best_{variable}.nc"
        file_path = os.path.join(era5_folder, file_name)

        # Check if the file exists
        if os.path.exists(file_path):
            data[variable] = [file_path]
        else:
            print(f"File not found for variable {variable}: {file_path}")
            data[variable] = [None]  # or you can choose to not add it to the dictionary

    return data

def create_era5_data_frame(data, era5_folder='../data/processed/processed_ERA5/24Apr2024'):
    """
    Creates a DataFrame from the ERA5 data dictionary with proper MultiIndex.

    Parameters:
    data (dict): Dictionary with variable names and corresponding file paths.
    era5_folder (str): Default folder path, used if data paths are missing.

    Returns:
    pandas.DataFrame: DataFrame containing the ERA5 data paths.
    """
    # Create a MultiIndex for the DataFrame
    index = pd.MultiIndex.from_tuples([('ERA5', 'historical-ERA5', 'r1i1p1f1')], names=['source_id', 'experiment_id', 'member_id'])

    # Create the DataFrame using the data dictionary and the constructed index
    files2crunch_df = pd.DataFrame(data, index=index)
    
    return files2crunch_df


# %% [markdown]
# ## Regridding

# %%

def create_target_grid(resolution, lon_convention: str = "180"):
    """
    Create a target grid dataset based on the specified resolution, typical for CMIP6.

    Parameters
    ----------
    resolution : float
        The resolution degree, one of [0.5, 1, 2, 4, 5, 10].
    lon_convention : {"180", "360"}
        Longitude wrapping: "-180..180" if "180", or "0..360" if "360".

    Returns
    -------
    xarray.Dataset
        The target grid dataset.
    """
    if resolution not in [0.1, 0.25, 0.5, 1, 2, 4, 5, 10]:
        raise ValueError("Resolution must be one of [0.1, 0.25, 0.5, 1, 2, 4, 5, 10].")
    lon_convention = str(lon_convention)
    if lon_convention not in {"180", "360"}:
        raise ValueError("lon_convention must be '180' or '360'.")
        
    # Define the edges of the grid cells for given resolution
    lat = np.arange(-90 + resolution/2, 90, resolution)
    if lon_convention == "360":
        lon = np.arange(0 + resolution/2, 360, resolution)
    else:
        lon = np.arange(-180 + resolution/2, 180, resolution)
    
    # Create the target grid dataset
    ds_out = xr.Dataset({
        'lat': (['lat'], lat),
        'lon': (['lon'], lon)
    })
    
    return ds_out



# ────────────────────────── PRE-PROCESSING & DEBUGGING HELPERS ──────────────────────────

def _drop_exact_pole(ds: xr.Dataset) -> xr.Dataset:
    if "lat" not in ds.coords: return ds
    ds_copy = ds.copy()
    for pole in (-90.0, 90.0):
        if ds_copy["lat"].size > 0 and np.isclose(ds_copy["lat"][-1], pole):
            ds_copy = ds_copy.isel(lat=slice(None, -1))
        if ds_copy["lat"].size > 0 and np.isclose(ds_copy["lat"][0], pole):
            ds_copy = ds_copy.isel(lat=slice(1, None))
    return ds_copy

def _transpose_for_esmf(ds: xr.Dataset) -> xr.Dataset:
    try:
        lat_name = next(d for d in ds.dims if d.lower().startswith("lat"))
        lon_name = next(d for d in ds.dims if d.lower().startswith("lon"))
        other_dims = [d for d in ds.dims if d not in (lat_name, lon_name)]
        return ds.transpose(*other_dims, lat_name, lon_name)
    except StopIteration:
        return ds

def _grid_fingerprint(ds: xr.Dataset) -> str:
    coords = np.concatenate([ds["lat"].values, ds["lon"].values]).tobytes()
    return hashlib.md5(coords).hexdigest()[:16]

def _log_grid_details(ds: xr.Dataset, stage_name: str, zstore_link: str):
    # (This function is unchanged from the previous answer)
    logging.info(f"--- GRID DIAGNOSTICS for {os.path.basename(zstore_link)} [{stage_name}] ---")
    try:
        logging.info(f"    Dataset dims: {ds.dims}, Coords: {list(ds.coords)}")
        if "lat" in ds.coords and "lon" in ds.coords:
            lat, lon = ds["lat"], ds["lon"]
            logging.info(f"    lat | dtype: {lat.dtype}, ndim: {lat.ndim}, size: {lat.size}, min: {lat.min().item():.2f}, max: {lat.max().item():.2f}")
            logging.info(f"    lon | dtype: {lon.dtype}, ndim: {lon.ndim}, size: {lon.size}, min: {lon.min().item():.2f}, max: {lon.max().item():.2f}, span: {lon.max().item() - lon.min().item():.2f}")
            if lat.ndim == 1 and lat.size > 1:
                lat_mono = np.all(np.diff(lat) > 0)
                lat_even = np.allclose(np.diff(lat), np.diff(lat)[0])
                logging.info(f"    lat | monotonic_increasing: {lat_mono}, evenly_spaced: {lat_even}")
            if lon.ndim == 1 and lon.size > 1:
                lon_mono = np.all(np.diff(lon) > 0)
                lon_even = np.allclose(np.diff(lon), np.diff(lon)[0])
                logging.info(f"    lon | monotonic_increasing: {lon_mono}, evenly_spaced: {lon_even}")
        else:
            logging.warning("    lat/lon coordinates not found for detailed diagnostics.")
        logging.info("-" * 70)
    except Exception as e:
        logging.error(f"    Error during grid diagnostics: {e}")


# head-less plotting for debug figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# 1.  Target grid creator (unchanged, but included for completeness)
# -------------------------------------------------------------------------
def create_target_grid(resolution: float = 1.0, lon_convention: str = "180") -> xr.Dataset:
    """
    Regular lon/lat target grid centred on cell mid-points.

    Parameters
    ----------
    resolution : float
        Target grid spacing in degrees.
    lon_convention : {"180", "360"}
        Longitude wrapping: "-180..180" if "180", or "0..360" if "360".
    """
    if resolution not in [0.1, 0.25, 0.5, 1, 2, 4, 5, 10]:
        raise ValueError("Resolution must be one of "
                         "[0.1, 0.25, 0.5, 1, 2, 4, 5, 10].")
    lon_convention = str(lon_convention)
    if lon_convention not in {"180", "360"}:
        raise ValueError("lon_convention must be '180' or '360'.")

    lat = np.arange(-90 + resolution / 2,  90, resolution)
    if lon_convention == "360":
        lon = np.arange(0 + resolution / 2, 360, resolution)
    else:
        lon = np.arange(-180 + resolution / 2, 180, resolution)

    return xr.Dataset({'lat': (['lat'], lat),
                       'lon': (['lon'], lon)})


# -------------------------------------------------------------------------
# 2.  Helpers for longitude handling
# -------------------------------------------------------------------------
def _wrap_lon_values(lon_values, lon_convention: str) -> np.ndarray:
    """Wrap longitudes to either [-180, 180) or [0, 360)."""
    lon_convention = str(lon_convention)
    if lon_convention not in {"180", "360"}:
        raise ValueError("lon_convention must be '180' or '360'.")
    lon_arr = np.asarray(lon_values, dtype="float64")
    if lon_convention == "360":
        return np.mod(np.mod(lon_arr, 360.0) + 360.0, 360.0)
    return (lon_arr + 180.0) % 360.0 - 180.0


def _add_cyclic(ds: xr.Dataset, lon_convention: str = "180") -> xr.Dataset:
    """
    Duplicate the first longitude column at the next seam **only** when

    *  the grid is 1-D & regular,
    *  the longitudes span < 360 °.

    Works for both -180..180 and 0..360 conventions.
    """
    if "lon" not in ds.coords or ds["lon"].ndim != 1:
        return ds  # curvilinear – leave untouched

    lon = ds["lon"].values
    if lon.size < 4:
        return ds  # too few points to decide safely

    dλ = np.diff(lon)
    if not np.allclose(dλ, dλ[0], atol=1e-6):
        logging.debug("Skip _add_cyclic – irregular λ spacing")
        return ds

    cell_span = lon[-1] - lon[0] + dλ[0]
    if np.isclose(cell_span, 360.0, atol=1e-4):
        return ds  # seam already present

    wrap_lon = lon[-1] + dλ[0]
    wrap = ds.isel(lon=0).assign_coords(lon=wrap_lon)
    return xr.concat([ds, wrap], dim="lon")


# -------------------------------------------------------------------------
# 3.  Helper: quick lon/lat diagnostics to the process log
# -------------------------------------------------------------------------
def _dump_lon_report(ds: xr.Dataset, stage: str, zlink: str) -> None:
    lon = ds["lon"].values
    lat = ds["lat"].values
    logging.info(
        "%s [%s]  nλ=%d  λmin=%.3f λmax=%.3f  span=%.3f"
        "  nφ=%d  φmin=%.3f φmax=%.3f",
        os.path.basename(zlink), stage,
        lon.size, lon.min(), lon.max(), lon[-1] - lon[0] + np.diff(lon)[0],
        lat.size, lat.min(), lat.max()
    )


# -------------------------------------------------------------------------
# 4.  Helper: scatter plot of the source grid (saved, not shown)
# -------------------------------------------------------------------------
def _plot_grid_dots(ds: xr.Dataset, zlink: str, stage: str) -> None:
    try:
        lon, lat = np.meshgrid(ds['lon'].values, ds['lat'].values)
        fig = plt.figure(figsize=(12, 6))
        plt.scatter(lon, lat, s=1)
        plt.title(f"{stage} grid for {os.path.basename(zlink)}")
        plt.xlabel("longitude"); plt.ylabel("latitude"); plt.grid(True)

        out_dir = Path("_debug_plots")
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f"{Path(zlink).stem}_{stage}.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logging.info("%s — debug plot saved to %s", zlink, out_file)
    except Exception as e:
        logging.error("%s — failed to create debug plot: %s", zlink, e)


# ───────────────────── MAIN REGRIDDING ROUTINE (FINAL ROBUST VERSION) ───────────────────────────
# ---------------------------------------------------------------------
# MAIN REGRIDDING ROUTINE
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# MAIN REGRIDDING ROUTINE  •  FINAL “NO-RC-545” EDITION
# ---------------------------------------------------------------------
def regrid_dataset_if_needed(
    zstore_link: str,
    *,
    resolution: float = 1.0,
    method: str = "bilinear",
    choose_ensemble_member: str = "smallest",
    debug_zero_check: bool = False,
    weights_cache_dir: str | None = None,
    lon_convention: str = "180",
) -> xr.Dataset | None:
    """
    Open, pre-process, and, if necessary, regrid a CMIP–style file
    to a regular lon/lat mesh.

    • Always *try* periodic xESMF first – but only when the longitudes
      span **exactly 360°** (±1 × 10⁻⁶).  
    • If that or the non-periodic retry blow up with rc = 545 we fall
      back to xarray’s native bilinear interpolation (no NaNs; slower).
    • Any fatal problem is logged, the function returns *None* so that
      embarrassingly parallel batch jobs can keep running.

    Longitude wrapping is controlled via ``lon_convention``:
    set to ``"360"`` to keep 0..360° grids, or leave at ``"180"`` for
    the default -180..180° convention.
    """
    # ─────────────────────────────────────────────────────────── 0  open
    try:
        src = open_any_cmip(zstore_link)
    except Exception as err:
        logging.error("%s — could not be opened: %s", zstore_link, err)
        return None

    try:
        ds = src  # shorthand
        lon_convention = str(lon_convention)
        if lon_convention not in {"180", "360"}:
            raise ValueError("lon_convention must be '180' or '360'.")
        # ────────────────────────────────────────────────── 1  unit fixes
        if "expver" in ds.dims:
            ds = ds.groupby("time").apply(
                lambda x: select_best_expver(x, list(range(1, 8)))
            )
        if choose_ensemble_member == "smallest" and "number" in ds.dims:
            ds = select_smallest_ensemble_member(ds)

        if "time" in ds.coords and ds["time"].size:
            sec_per_month = ds["time"].dt.days_in_month * 24 * 3600
            for var in ("tnsr", "tntr"):
                if var in ds:
                    ds[var] = ds[var] / sec_per_month

        if "pr" in ds and ds["pr"].attrs.get("units") == "m":
            ds["pr"] = ds["pr"] * 1000.0 / (24 * 3600)
            ds["pr"].attrs["units"] = "kg m-2 s-1"

        if "evspsbl" in ds and (ds["evspsbl"].mean().compute() < 0):
            ds["evspsbl"] = -ds["evspsbl"]
        if "hurs" in ds:
            ds["hurs"] = ds["hurs"].clip(max=100)

        # ─────────────────────────────────────────── 2  coord clean-up
        if "latitude" in ds.coords:
            ds = ds.rename({"latitude": "lat"})
        if "longitude" in ds.coords:
            ds = ds.rename({"longitude": "lon"})

        # ---- robust lon wrapping -------------------------------------------------
        if "lon" in ds.coords:
            lon = _wrap_lon_values(ds["lon"].astype("float64").values, lon_convention)
            ds = ds.assign_coords(lon=lon)
            ds = ds.sortby("lon")
            # de-duplicate
            dup = np.concatenate([[False], np.diff(ds["lon"]) == 0])
            if dup.any():
                logging.warning("%s — %d duplicate longitudes dropped",
                                zstore_link, dup.sum())
                ds = ds.isel(lon=~xr.DataArray(dup, dims="lon"))
        if "lat" in ds.dims:
            ds = ds.sortby("lat")

        # drop “bounds” helpers – xESMF dislikes them
        drop = [v for v in ds.variables if v.endswith(("_bnds", "_bounds"))]
        if drop:
            ds = ds.drop_vars(drop)

        # ensure 1-D, contiguous, C-order float64
        try:
            assert ds["lat"].ndim == ds["lon"].ndim == 1
            ds["lat"] = xr.DataArray(
                np.ascontiguousarray(ds["lat"].astype("float64").values),
                coords={"lat": ds["lat"].values},
                dims="lat",
                attrs=ds["lat"].attrs,
            )
            ds["lon"] = xr.DataArray(
                np.ascontiguousarray(ds["lon"].astype("float64").values),
                coords={"lon": ds["lon"].values},
                dims="lon",
                attrs=ds["lon"].attrs,
            )
        except Exception:
            logging.error("%s — curvilinear grid detected — skipped", zstore_link)
            return None

        # ─────────────────────────────────────────── 3  target grid
        target = create_target_grid(resolution, lon_convention=lon_convention)
        # Target grid already uses the requested convention
        target = target.sortby("lat").sortby("lon")

        if (ds.sizes["lat"] == target.sizes["lat"]
            and ds.sizes["lon"] == target.sizes["lon"]
            and np.allclose(ds["lat"], target["lat"])
            and np.allclose(ds["lon"], target["lon"])):
            ds_out = ds       # already on target grid
        else:
            # ───────────────────────────────── 4  xESMF regridding
            ds_pre  = _drop_exact_pole(_transpose_for_esmf(_add_cyclic(ds, lon_convention)))
            tgt_pre = _transpose_for_esmf(target)

            _dump_lon_report(ds_pre , "src-pre", zstore_link)
            _dump_lon_report(tgt_pre, "tgt"    , zstore_link)

            weight_file = None
            if weights_cache_dir:
                src_hash = _grid_fingerprint(ds_pre)
                tgt_hash = _grid_fingerprint(tgt_pre)
                wf_dir   = Path(weights_cache_dir); wf_dir.mkdir(exist_ok=True)
                weight_file = wf_dir / f"{src_hash}-to-{tgt_hash}.nc"

            # ----- 4a  periodic = True  (only if span = 360°) ----------------
            # For -180 to +180 convention, check if data spans the full range
            lon_span = ds_pre["lon"][-1] - ds_pre["lon"][0] + np.diff(ds_pre["lon"])[0]
            periodic_ok = np.isclose(lon_span, 360.0, atol=1e-6)
            first_try_failed = False
            if periodic_ok:
                try:
                    rgr = xe.Regridder(
                        ds_pre, tgt_pre,
                        method=method, periodic=True,
                        extrap_method="nearest_s2d",
                        unmapped_to_nan=True, ignore_degenerate=True,
                        filename=str(weight_file) if weight_file else None,
                        reuse_weights=bool(weight_file and weight_file.exists())
                    )
                    ds_out = rgr(ds_pre, keep_attrs=True)
                except Exception as err:
                    first_try_failed = True
                    logging.error("%s — periodic rc=545 (%s)",
                                  zstore_link, err)
                    _plot_grid_dots(ds_pre, zstore_link, "failed_periodic")

            # ----- 4b  non-periodic retry -------------------------------------
            if (not periodic_ok) or first_try_failed:
                try:
                    rgr = xe.Regridder(
                        ds_pre, tgt_pre,
                        method=method, periodic=False,
                        extrap_method="nearest_s2d",
                        unmapped_to_nan=True, ignore_degenerate=True,
                    )
                    ds_out = rgr(ds_pre, keep_attrs=True)
                except Exception as err2:
                    logging.error("%s — non-periodic rc=545 (%s)",
                                  zstore_link, err2)
                    _plot_grid_dots(ds_pre, zstore_link, "failed_nonperiodic")

                    # ----- 4c  LAST RESORT: xarray interp (bilinear) ----------
                    logging.warning("%s — fall-back to xarray.interp",
                                    zstore_link)
                    try:
                        ds_out = (
                            ds_pre
                            .interp(
                                lon=target["lon"],
                                lat=target["lat"],
                                method="linear",
                                kwargs={"fill_value": "extrapolate"},
                            )
                            .transpose(..., "lat", "lon")
                        )
                    except Exception as err3:
                        logging.error("%s — xarray.interp also failed: %s",
                                      zstore_link, err3)
                        return None

            # ----- 4d  pole fill + attrs --------------------------------------
            ds_out = ds_out.interpolate_na(
                dim="lat", method="nearest", fill_value="extrapolate"
            )
            for v in ds.variables:
                if v in ds_out:
                        ds_out[v].attrs = ds[v].attrs
            ds_out.attrs = ds.attrs
            if "time" in ds_out and "time" in ds:
                ds_out["time"].attrs = ds["time"].attrs

        if "lon" in ds_out.coords:
            lon = _wrap_lon_values(ds_out["lon"].values, lon_convention)
            ds_out = ds_out.assign_coords(lon=lon).sortby("lon")

        # ───────────────────────────── 5  optional zero check
        if debug_zero_check and "time" in ds_out and ds_out["time"].size:
            var0 = next(iter(ds_out.data_vars))
            frac0 = float((ds_out[var0].isel(time=0) == 0).sum()) \
                    / ds_out[var0].isel(time=0).size
            if frac0 > 0.5:
                logging.warning("%s — >50 %% zeros in first slice of %s "
                                "(possible wrapping bug)",
                                zstore_link, var0)

        # realise & detach from source
        return ds_out.load().copy(deep=True)

    finally:
        try:
            src.close()
        except Exception:
            pass





def regrid_dataset_if_needed_old9June(
    zstore_link: str,
    *,
    resolution: float = 1.0,
    method: str = "bilinear",
    choose_ensemble_member: str = "smallest",
    debug_zero_check: bool = False,
    weights_cache_dir: str | None = None,      # kept – but optional now
) -> xr.Dataset | None:
    """
    Open a CMIP/ERA-like file, apply **unit fixes**, coordinate clean-up
    and (if necessary) re-grid it onto a regular lon/lat mesh.

    – Longitude is *always* wrapped to **[0, 360)** once.  
    – We **ask xESMF for a periodic grid** first; on rc = 545 failures
      we   fall back to non-periodic, dump a grid report and save a
      scatter plot to `_debug_plots/`.
    – Any unrecoverable problem is only logged and the function
      returns `None` so large batch runs can simply skip the file.
    """
    # ──────────────────────────────────────────────────────────── 0. open
    try:
        src = open_any_cmip(zstore_link)
    except Exception as err:
        logging.error("%s — could not be opened: %s", zstore_link, err)
        return None

    try:
        ds = src           # shorthand

        # ─────────────────────────────────────────────── 1. unit fixes
        if "expver" in ds.dims:
            ds = ds.groupby("time").apply(
                lambda x: select_best_expver(x, list(range(1, 8)))
            )

        if choose_ensemble_member == "smallest" and "number" in ds.dims:
            ds = select_smallest_ensemble_member(ds)

        if "time" in ds.coords and ds["time"].size:
            sec_per_month = ds["time"].dt.days_in_month * 24 * 3600
            for var in ("tnsr", "tntr"):
                if var in ds:
                    ds[var] = ds[var] / sec_per_month

        if "pr" in ds and ds["pr"].attrs.get("units") == "m":
            ds["pr"] = ds["pr"] * 1000.0 / (24 * 3600)
            ds["pr"].attrs["units"] = "kg m-2 s-1"

        if "evspsbl" in ds and (ds["evspsbl"].mean().compute() < 0):
            ds["evspsbl"] = -ds["evspsbl"]

        if "hurs" in ds:
            ds["hurs"] = ds["hurs"].clip(max=100)

        # ─────────────────────────────────────── 2. coordinate clean-up
        if "latitude"  in ds.coords: ds = ds.rename({"latitude":  "lat"})
        if "longitude" in ds.coords: ds = ds.rename({"longitude": "lon"})

        if "lon" in ds.coords:
            # wrap once, then sort & de-duplicate
            ds = ds.assign_coords(lon=((ds["lon"] + 360) % 360).astype(float))
            ds = ds.sortby("lon")
            dup = np.concatenate([[False], np.diff(ds["lon"]) == 0])
            if dup.any():
                logging.warning("%s — %d duplicate longitudes dropped",
                                zstore_link, dup.sum())
                ds = ds.isel(lon=~xr.DataArray(dup, dims="lon"))

        if "lat" in ds.dims:
            ds = ds.sortby("lat")

        # drop “bounds” helper variables – xESMF dislikes 2-D coords
        to_drop = [v for v in ds.variables if v.endswith(("_bnds", "_bounds"))]
        if to_drop:
            ds = ds.drop_vars(to_drop)

        # must have 1-D lon/lat
        if not ("lat" in ds.coords and "lon" in ds.coords
                and ds["lat"].ndim == ds["lon"].ndim == 1):
            logging.error("%s — curvilinear grid detected — skipped", zstore_link)
            return None

        # ─────────────────────────────────────── 3. target grid (regular)
        target = create_target_grid(resolution)
        target["lon"] = ((target["lon"] + 360) % 360).astype(float)
        target = target.sortby("lat").sortby("lon")

        already_on_target = (
            ds.sizes["lat"] == target.sizes["lat"]
            and ds.sizes["lon"] == target.sizes["lon"]
            and np.allclose(ds["lat"], target["lat"])
            and np.allclose(ds["lon"], target["lon"])
        )
        if already_on_target:
            ds_out = ds

        # ───────────────────────────────────────────── 4. regridding
        else:
            ds_pre  = _drop_exact_pole(_transpose_for_esmf(_add_cyclic(ds)))
            tgt_pre = _transpose_for_esmf(target)

            _dump_lon_report(ds_pre,  "src-pre", zstore_link)
            _dump_lon_report(tgt_pre, "tgt"    , zstore_link)

            # 4a  weight caching (optional, same file names as before)
            weight_file = None
            if weights_cache_dir is not None:
                src_hash = _grid_fingerprint(ds_pre)
                tgt_hash = _grid_fingerprint(tgt_pre)
                cache_dir = Path(weights_cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                weight_file = cache_dir / f"{src_hash}-to-{tgt_hash}.nc"

            # 4b  first try: periodic = True
            try:
                rgr = xe.Regridder(
                    ds_pre, tgt_pre,
                    method          = method,
                    periodic        = True,
                    extrap_method   = "nearest_s2d",
                    unmapped_to_nan = True,
                    ignore_degenerate=True,
                    filename        = str(weight_file) if weight_file else None,
                    reuse_weights   = bool(weight_file and weight_file.exists()),
                )
                ds_out = rgr(ds_pre, keep_attrs=True)

            except Exception as err:
                logging.error("%s — periodic regridding failed: %s",
                              zstore_link, err)
                _plot_grid_dots(ds_pre, zstore_link, "failed_periodic")

                # 4c fallback: periodic = False
                try:
                    rgr = xe.Regridder(
                        ds_pre, tgt_pre,
                        method          = method,
                        periodic        = False,
                        extrap_method   = "nearest_s2d",
                        unmapped_to_nan = True,
                        ignore_degenerate=True,
                        filename        = None,  # don't pollute cache
                    )
                    ds_out = rgr(ds_pre, keep_attrs=True)
                except Exception as err2:
                    logging.error("%s — non-periodic regridding ALSO failed: %s",
                                  zstore_link, err2)
                    _plot_grid_dots(ds_pre, zstore_link, "failed_nonperiodic")
                    return None

            # tidy up: fill poles & copy attrs
            ds_out = ds_out.interpolate_na(dim="lat", method="nearest",
                                           fill_value="extrapolate")
            for v in ds.variables:
                if v in ds_out:
                    ds_out[v].attrs = ds[v].attrs
            ds_out.attrs = ds.attrs
            if "time" in ds_out and "time" in ds:
                ds_out["time"].attrs = ds["time"].attrs

        # ─────────────────────────────── 5. optional ‘too many zeros?’ check
        if debug_zero_check and "time" in ds_out and ds_out["time"].size:
            first = next(iter(ds_out.data_vars))
            da0   = ds_out[first].isel(time=0)
            frac0 = float((da0 == 0).sum()) / da0.size
            if frac0 > 0.5:
                logging.warning("%s — >50 %% zeros in first slice of %s "
                                "(possible wrapping bug)", zstore_link, first)

        # realise into RAM & sever ties to original file
        return ds_out.load().copy(deep=True)

    finally:
        try:
            src.close()
        except Exception:
            pass


if False: 
    # last chatgpt attempt. Mon 9 June 2-41pm. # ------------------------------------------------------------------------
    def _add_cyclic(ds: xr.Dataset) -> xr.Dataset:
        """Duplicate the first lon column at +360° so ESMF can wrap cleanly."""
        lon = ds["lon"]
        if lon.ndim == 1 and not np.isclose(lon[0], lon[-1] - 360):
            wrap = ds.isel(lon=0)
            wrap = wrap.assign_coords(lon=wrap.lon + 360)
            ds = xr.concat([ds, wrap], dim="lon")
        return ds
    
    
    def _drop_exact_pole(ds: xr.Dataset) -> xr.Dataset:
        """Remove a row that lies *exactly* on ±90° to avoid ESMF edge cases."""
        for pole in (-90.0, 90.0):
            if float(ds["lat"][-1]) == pole:
                ds = ds.isel(lat=slice(None, -1))
            if float(ds["lat"][0]) == pole:
                ds = ds.isel(lat=slice(1, None))
        return ds
    
    
    def _transpose_for_esmf(ds: xr.Dataset) -> xr.Dataset:
        """Ensure (..., y, x) order because ESMF expects (lat, lon) last."""
        lat_name = [d for d in ds.dims if d.lower().startswith("lat")][0]
        lon_name = [d for d in ds.dims if d.lower().startswith("lon")][0]
        other = [d for d in ds.dims if d not in (lat_name, lon_name)]
        return ds.transpose(*other, lat_name, lon_name)
    
    
    def _grid_fingerprint(ds: xr.Dataset) -> str:
        """Hash of the 1-D lat/lon coordinates – used for the weight cache."""
        coords = np.concatenate([ds["lat"].values, ds["lon"].values]).tobytes()
        return hashlib.md5(coords).hexdigest()[:16]
    
    
    # ───────────────────── main regridding routine ───────────────────────────
    def regrid_dataset_if_needed(
        zstore_link: str,
        *,
        resolution: float = 1.0,
        method: str = "bilinear",
        choose_ensemble_member: str = "smallest",
        debug_zero_check: bool = False,
        weights_cache_dir: str | None = None,
    ) -> xr.Dataset | None:
        """
        Open a CMIP/ERA/re-analysis file, apply unit fixes, and (if necessary)
        re-grid it onto a regular lon/lat mesh.
    
        If anything goes irrecoverably wrong, the function logs the problem
        and returns ``None`` so massive parallel runs can keep going.
        """
        # 0 ── open dataset ───────────────────────────────────────────────────
        try:
            src = (
                open_any_cmip(zstore_link)
                if os.path.isfile(zstore_link)
                else open_any_cmip(zstore_link, consolidated=True)
            )
        except Exception as err:
            logging.error("%s — could not be opened: %s", zstore_link, err)
            return None
    
        try:
            ds = src  # local shorthand
    
            # 1 ── quick fixes & unit conversions (unchanged) ────────────────
            if "expver" in ds.dims:
                ds = ds.groupby("time").apply(
                    lambda x: select_best_expver(x, [1, 2, 3, 4, 5, 6, 7])
                )
            if choose_ensemble_member == "smallest" and "number" in ds.dims:
                ds = select_smallest_ensemble_member(ds)
    
            sec_per_month = ds["time"].dt.days_in_month * 24 * 3600
            for var in ("tnsr", "tntr"):
                if var in ds:
                    ds[var] = ds[var] / sec_per_month
    
            if "pr" in ds and ds["pr"].attrs.get("units") == "m":
                ds["pr"] = ds["pr"] * 1000.0 / (24 * 3600)
                ds["pr"].attrs["units"] = "kg m-2 s-1"
    
            if "evspsbl" in ds and (ds["evspsbl"].mean().compute() < 0):
                ds["evspsbl"] = -ds["evspsbl"]
            if "hurs" in ds:
                ds["hurs"] = ds["hurs"].clip(max=100)
    
            # 2 ── coordinate normalisation ─────────────────────────────────
            coord_renames = {}
            if "latitude" in ds and "lat" not in ds:
                coord_renames["latitude"] = "lat"
            if "longitude" in ds and "lon" not in ds:
                coord_renames["longitude"] = "lon"
            if coord_renames:
                ds = ds.rename(coord_renames)
    
            # bring lon to [-180, 180)
            if (ds["lon"] > 180).any() or (ds["lon"] < -180).any():
                ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180)
    
            # sort ascending
            if not (ds["lon"].diff("lon") > 0).all():
                ds = ds.sortby("lon")
            if not (ds["lat"].diff("lat") > 0).all():
                ds = ds.sortby("lat")
    
            # 2a ── curvilinear sanity check ────────────────────────────────
            try:
                assert ds["lat"].ndim == 1 and ds["lon"].ndim == 1
            except AssertionError:
                logging.error(
                    "%s — lat/lon are not 1-D (curvilinear grid?) — skipped",
                    zstore_link,
                )
                return None
    
            # 3 ── target grid (always 1-D lat/lon) ─────────────────────────
            target = create_target_grid(resolution)
            target = target.sortby("lat").sortby("lon")
    
            # already on requested grid?
            if (
                ds.sizes.get("lat") == target.sizes["lat"]
                and ds.sizes.get("lon") == target.sizes["lon"]
                and np.allclose(ds["lat"], target["lat"])
                and np.allclose(ds["lon"], target["lon"])
            ):
                ds_out = ds
            else:
                # 4 ── ESMF preparation ─────────────────────────────────────
                ds_pre = _drop_exact_pole(_transpose_for_esmf(_add_cyclic(ds)))
                tgt_pre = _transpose_for_esmf(target)
    
                # 4a ── weight caching
                weight_file = None
                if weights_cache_dir is not None:
                    src_hash = _grid_fingerprint(ds_pre)
                    tgt_hash = _grid_fingerprint(tgt_pre)
                    cache_dir = Path(weights_cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    weight_file = cache_dir / f"{src_hash}-to-{tgt_hash}.nc"
    
                # 4b ── build / reuse weights
                try:
                    rgr = xe.Regridder(
                        ds_pre,
                        tgt_pre,
                        method=method,
                        periodic=True,
                        extrap_method="nearest_s2d",
                        unmapped_to_nan=True,
                        ignore_degenerate=True,
                        reuse_weights=False,
                        filename=str(weight_file) if weight_file else None,
                    )
                    ds_out = rgr(ds_pre)
    
                    # copy attrs
                    for v in ds.variables:
                        if v in ds_out:
                            ds_out[v].attrs = ds[v].attrs
                    ds_out.attrs = ds.attrs
                    ds_out["time"].attrs = ds["time"].attrs
    
                except ValueError as err:
                    # fallback: turn off periodic and try once more
                    logging.warning(
                        "%s — periodic regridding failed (%s); retrying non-periodic",
                        zstore_link,
                        err,
                    )
                    try:
                        rgr = xe.Regridder(
                            ds_pre,
                            tgt_pre,
                            method=method,
                            periodic=False,
                            extrap_method="nearest_s2d",
                            unmapped_to_nan=True,
                            ignore_degenerate=True,
                            reuse_weights=False,
                        )
                        ds_out = rgr(ds_pre)
                    except Exception as err2:
                        logging.error("%s — regridding failed: %s", zstore_link, err2)
                        return None
    
            # 5 ── optional zero-diagnostic ────────────────────────────────
            if debug_zero_check:
                var0 = next(iter(ds_out.data_vars))
                da0 = ds_out[var0].isel(time=0)
                frac_zero = float((da0 == 0).sum()) / da0.size
                if frac_zero > 0.5:
                    import matplotlib.pyplot as plt
    
                    plt.figure(figsize=(6, 4))
                    da0.plot(cmap="viridis", vmin=float(da0.min()), vmax=float(da0.max()))
                    plt.title(
                        f"{var0} @ time=0 has {frac_zero:.0%} zeros – "
                        "possible lon-wrapping issue?"
                    )
                    plt.show()
                    logging.error(
                        "%s — too many zeros after regridding (%.0f%%) — skipped",
                        zstore_link,
                        frac_zero * 100,
                    )
                    return None
    
            # 6 ── detach from the source & return ──────────────────────────
            ds_out = ds_out.load()           # realise the data (no chunk magic)
            ds_out = ds_out.copy(deep=True)  # sever ties to *src*
            return ds_out
    
        finally:
            # make *really* sure the original file handle is closed
            try:
                src.close()
            except Exception:
                pass


# ────────────────────────────── helpers ──────────────────────────────────
if False: 
    def _add_cyclic(ds: xr.Dataset) -> xr.Dataset:
        """Duplicate the first lon column at +360° so ESMF can wrap cleanly."""
        lon = ds["lon"]
        if lon.ndim == 1 and not np.isclose(lon[0], lon[-1] - 360):
            wrap = ds.isel(lon=0)
            wrap = wrap.assign_coords(lon=wrap.lon + 360)
            ds = xr.concat([ds, wrap], dim="lon")
        return ds
    
    
    def _drop_exact_pole(ds: xr.Dataset) -> xr.Dataset:
        """Remove a row that lies *exactly* on ±90° to avoid ESMF edge cases."""
        for pole in (-90.0, 90.0):
            if float(ds["lat"][-1]) == pole:
                ds = ds.isel(lat=slice(None, -1))
            if float(ds["lat"][0]) == pole:
                ds = ds.isel(lat=slice(1, None))
        return ds
    
    
    def _transpose_for_esmf(ds: xr.Dataset) -> xr.Dataset:
        """Ensure (..., y, x) order because ESMF expects (lat, lon) last."""
        lat_name = [d for d in ds.dims if d.lower().startswith("lat")][0]
        lon_name = [d for d in ds.dims if d.lower().startswith("lon")][0]
        other = [d for d in ds.dims if d not in (lat_name, lon_name)]
        return ds.transpose(*other, lat_name, lon_name)
    
    
    def _grid_fingerprint(ds: xr.Dataset) -> str:
        """Hash of the 1-D lat/lon coordinates – used for the weight cache."""
        coords = np.concatenate([ds["lat"].values, ds["lon"].values]).tobytes()
        return hashlib.md5(coords).hexdigest()[:16]
    
    
    # ───────────────────── main regridding routine ───────────────────────────
    def regrid_dataset_if_needed_goodregridbutmemoryissue8June(
        zstore_link: str,
        *,
        resolution: float = 1.0,
        method: str = "bilinear",
        choose_ensemble_member: str = "smallest",
        debug_zero_check: bool = False,
        weights_cache_dir: str | None = None,
    ) -> xr.Dataset | None:
        """
        Open a CMIP/ERA/re-analysis file, apply unit fixes, and (if necessary)
        re-grid it onto a regular lon/lat mesh.
    
        Any unrecoverable problem is *logged* and returns ``None`` so that
        big embarrassingly-parallel runs can just skip the bad apples.
    
        Parameters
        ----------
        zstore_link : str
            Local path, NetCDF, GRIB, or Zarr reference.
        resolution : float
            Target grid spacing in degrees (default 1°).
        method : str
            xESMF interpolation method (``bilinear``, ``nearest_s2d`` …).
        choose_ensemble_member : {"smallest", "none"}
            If the file contains an ensemble ``number`` dimension, pick the
            member with the smallest data volume (handy for massive ERA5).
        debug_zero_check : bool
            Plot the first time slice if more than 50 % of its cells are zero
            after regridding – useful when you’re hunting wrapping bugs.
        weights_cache_dir : str | None
            If given, weight matrices are saved / reused here; file names are
            `<src-hash>-to-<tgt-hash>.nc`.  Saves *a lot* of time when you
            crunch tens of thousands of files that share only a handful of
            native grids.
        """
        # 0 ── open dataset ───────────────────────────────────────────────────
        try:
            src = (
                open_any_cmip(zstore_link)
                if os.path.isfile(zstore_link)
                else open_any_cmip(zstore_link, consolidated=True)
            )
        except Exception as err:
            logging.error("%s — could not be opened: %s", zstore_link, err)
            return None
    
        try:
            ds = src                         # local shorthand
    
            # 1 ── quick fixes & unit conversions ────────────────────────────
            if "expver" in ds.dims:
                ds = ds.groupby("time").apply(
                    lambda x: select_best_expver(x, [1, 2, 3, 4, 5, 6, 7])
                )
            if choose_ensemble_member == "smallest" and "number" in ds.dims:
                ds = select_smallest_ensemble_member(ds)
    
            sec_per_month = ds["time"].dt.days_in_month * 24 * 3600
            for var in ("tnsr", "tntr"):
                if var in ds:
                    ds[var] = ds[var] / sec_per_month
    
            if "pr" in ds and ds["pr"].attrs.get("units") == "m":
                ds["pr"] = ds["pr"] * 1000.0 / (24 * 3600)
                ds["pr"].attrs["units"] = "kg m-2 s-1"
    
            if "evspsbl" in ds and (ds["evspsbl"].mean().compute() < 0):
                ds["evspsbl"] = -ds["evspsbl"]
            if "hurs" in ds:
                ds["hurs"] = ds["hurs"].clip(max=100)
    
            # 2 ── coordinate normalisation ─────────────────────────────────
            coord_renames = {}
            if "latitude" in ds and "lat" not in ds:
                coord_renames["latitude"] = "lat"
            if "longitude" in ds and "lon" not in ds:
                coord_renames["longitude"] = "lon"
            if coord_renames:
                ds = ds.rename(coord_renames)
    
            # bring lon to [-180, 180)
            if (ds["lon"] > 180).any() or (ds["lon"] < -180).any():
                ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180)
    
            # sort ascending
            if not (ds["lon"].diff("lon") > 0).all():
                ds = ds.sortby("lon")
            if not (ds["lat"].diff("lat") > 0).all():
                ds = ds.sortby("lat")
    
            # 3 ── target grid (always 1-D lat/lon) ─────────────────────────
            target = create_target_grid(resolution)
            target = target.sortby("lat").sortby("lon")
    
            # already on requested grid?
            if (
                ds.sizes.get("lat") == target.sizes["lat"]
                and ds.sizes.get("lon") == target.sizes["lon"]
                and np.allclose(ds["lat"], target["lat"])
                and np.allclose(ds["lon"], target["lon"])
            ):
                ds_out = ds
            else:
                # 4 ── ESMF preparation ─────────────────────────────────────
                ds_pre = _drop_exact_pole(_transpose_for_esmf(_add_cyclic(ds)))
                tgt_pre = _transpose_for_esmf(target)
    
                # 4a ── weight caching
                weight_file = None
                if weights_cache_dir is not None:
                    src_hash = _grid_fingerprint(ds_pre)
                    tgt_hash = _grid_fingerprint(tgt_pre)
                    cache_dir = Path(weights_cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    weight_file = cache_dir / f"{src_hash}-to-{tgt_hash}.nc"
    
                # 4b ── build / reuse weights
                try:
                    rgr = xe.Regridder(
                        ds_pre,
                        tgt_pre,
                        method=method,
                        periodic=True,
                        extrap_method="nearest_s2d",
                        unmapped_to_nan=True,
                        ignore_degenerate=True,
                        reuse_weights=False,
                        filename=str(weight_file) if weight_file else None,
                    )
                    ds_out = rgr(ds_pre)
    
                    # copy attrs
                    for v in ds.variables:
                        if v in ds_out:
                            ds_out[v].attrs = ds[v].attrs
                    ds_out.attrs = ds.attrs
                    ds_out["time"].attrs = ds["time"].attrs
    
                except ValueError as err:
                    # fallback: turn off periodic and try once more
                    logging.warning(
                        "%s — periodic regridding failed (%s); retrying non-periodic",
                        zstore_link,
                        err,
                    )
                    try:
                        rgr = xe.Regridder(
                            ds_pre,
                            tgt_pre,
                            method=method,
                            periodic=False,
                            extrap_method="nearest_s2d",
                            unmapped_to_nan=True,
                            ignore_degenerate=True,
                            reuse_weights=False,
                        )
                        ds_out = rgr(ds_pre)
                    except Exception as err2:
                        logging.error("%s — regridding failed: %s", zstore_link, err2)
                        return None
    
            # 5 ── optional zero-diagnostic ─────────────────────────────────
            if debug_zero_check:
                var0 = next(iter(ds_out.data_vars))
                da0 = ds_out[var0].isel(time=0)
                frac_zero = float((da0 == 0).sum()) / da0.size
                if frac_zero > 0.5:
                    import matplotlib.pyplot as plt
    
                    plt.figure(figsize=(6, 4))
                    da0.plot(cmap="viridis", vmin=float(da0.min()), vmax=float(da0.max()))
                    plt.title(
                        f"{var0} @ time=0 has {frac_zero:.0%} zeros – "
                        "possible lon-wrapping issue?"
                    )
                    plt.show()
                    raise RuntimeError(
                        f"Too many zeros ({frac_zero:.0%}) in first timestep; "
                        "check your lon coordinate wrapping."
                    )
    
            # 6 ── detach from the source & return ──────────────────────────
            ds_out = ds_out.load()            # Dask brings it into memory
            ds_out = ds_out.copy(deep=True)   # sever ties to *src*
            return ds_out
    
        finally:
            # make *really* sure the original file handle is closed
            try:
                src.close()
            except Exception:
                pass


def regrid_dataset_if_needed_workingwithNaNs8June(
    zstore_link: str,
    *,
    resolution: float = 1.0,
    method: str = "bilinear",
    choose_ensemble_member: str = "smallest",
    debug_zero_check: bool = False,
) -> xr.Dataset | None:
    """
    Open a CMIP/ERA/re-analysis file, apply unit fixes, and (if necessary)
    re-grid it onto a regular lon/lat mesh.

    The function is resilient – any unrecoverable problem is *logged* and
    returns ``None`` so that large parallel jobs can skip bad files.
    """
    # ─────────────────────────────────────────────────────────────────────
    # 0. open dataset   (keep a handle so we can close it explicitly later)
    # ─────────────────────────────────────────────────────────────────────
    try:
        src = (
            open_any_cmip(zstore_link)
            if os.path.isfile(zstore_link)
            else open_any_cmip(zstore_link, consolidated=True)
        )
    except Exception as err:
        logging.error("%s — could not be opened: %s", zstore_link, err)
        return None

    try:
        ds = src                # alias used below for brevity

        # ───── 1. quick fixes & unit conversions ───────────────────────
        if "expver" in ds.dims:
            ds = ds.groupby("time").apply(
                lambda x: select_best_expver(x, [1, 2, 3, 4, 5, 6, 7])
            )
        if choose_ensemble_member == "smallest" and "number" in ds.dims:
            ds = select_smallest_ensemble_member(ds)

        sec_per_month = ds["time"].dt.days_in_month * 24 * 3600
        if "tnsr" in ds:
            ds["tnsr"] = ds["tnsr"] / sec_per_month
        if "tntr" in ds:
            ds["tntr"] = ds["tntr"] / sec_per_month

        if "pr" in ds and ds["pr"].attrs.get("units") == "m":
            ds["pr"] = ds["pr"] * 1000 / (24 * 3600)
            ds["pr"].attrs["units"] = "kg m-2 s-1"

        if "evspsbl" in ds and ds["evspsbl"].mean().compute() < 0:
            ds["evspsbl"] = -ds["evspsbl"]
        if "hurs" in ds:
            ds["hurs"] = ds["hurs"].clip(max=100)

        # ───── 2. coordinate normalisation ─────────────────────────────
        if "latitude" in ds and "lat" not in ds:
            ds = ds.rename({"latitude": "lat"})
        if "longitude" in ds and "lon" not in ds:
            ds = ds.rename({"longitude": "lon"})

        if (ds["lon"] > 180).any() or (ds["lon"] < -180).any():
            ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180)

        if not (ds["lon"].diff("lon") > 0).all():
            ds = ds.sortby("lon")
        if not (ds["lat"].diff("lat") > 0).all():
            ds = ds.sortby("lat")

        # ───── 3. already on target grid? ──────────────────────────────
        target = create_target_grid(resolution)
        if (
            ds.sizes.get("lat") == target.sizes["lat"]
            and ds.sizes.get("lon") == target.sizes["lon"]
        ):
            ds_out = ds
        else:
            # helper to copy attrs
            def _with_attrs(src, out):
                for v in src.variables:
                    if v in out:
                        out[v].attrs = src[v].attrs
                out.attrs = src.attrs
                out["time"].attrs = src["time"].attrs
                return out

            # ───── 4. xESMF (periodic & non-periodic) ──────────────────
            for periodic in (True, False):
                try:
                    print(f'Now trying xESMF regridding with periodic set to {periodic}') 
                    rgr = xe.Regridder(
                        ds,
                        target,
                        method=method,
                        periodic=periodic,
                        reuse_weights=False,
                        extrap_method="nearest_s2d",   # or "nearest_idavg"
                        # extrap_periodic=periodic,      # keep it periodic if you want
                        ignore_degenerate=True,        # don’t fail on tiny slivers
                    )
                    ds_out = _with_attrs(ds, rgr(ds))
                    break
                except ValueError as err:
                    if "ESMC_GridCreate" not in str(err):
                        logging.error("%s — unexpected xESMF error: %s", zstore_link, err)
                        return None
                    logging.warning(
                        "%s — xESMF periodic=%s failed (%s)", zstore_link, periodic, err
                    )
            else:
                print(f'xESMF regridding failed.. Now trying xarray regridding') 
                # ───── 5. pure-xarray fallback ─────────────────────────
                if ds["lat"].ndim == ds["lon"].ndim == 1:
                    logging.warning("%s — falling back to xarray.interp (slow)", zstore_link)
                    try:
                        ds_out = _with_attrs(
                            ds,
                            ds.interp(
                                lon=target["lon"],
                                lat=target["lat"],
                                kwargs={"bounds_error": False},
                            ),
                        )
                    except Exception as err:
                        logging.error("%s — xarray.interp failed: %s", zstore_link, err)
                        return None
                else:
                    logging.error("%s — regridding failed (curvilinear grid)", zstore_link)
                    return None

        # ───── 6. optional zero-fill diagnostic ────────────────────────
        if debug_zero_check:
            var0 = next(iter(ds_out.data_vars))
            da0 = ds_out[var0].isel(time=0)
            frac_zero = float((da0 == 0).sum()) / da0.size
            if frac_zero > 0.5:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6, 4))
                da0.plot(cmap="viridis", vmin=da0.min(), vmax=da0.max())
                plt.title(
                    f"{var0} @ time=0 has {frac_zero:.0%} zeros – "
                    "possible lon-wrapping issue?"
                )
                plt.show()
                raise RuntimeError(
                    f"Too many zeros ({frac_zero:.0%}) in first timestep; "
                    "check your lon coordinate wrapping."
                )

        # ───── 7. DETACH from the source file & return ─────────────────
        ds_out = ds_out.load()            # read into memory / dask cache
        ds_out = ds_out.copy(deep=True)   # sever links to *src*
        return ds_out

    finally:
        # Make *really* sure the original file handle is closed
        try:
            src.close()
        except Exception:
            pass



# %%
def regrid_dataset_if_needed_old(
    zstore_link: str,
    *,
    resolution: float = 1.0,
    method: str = "bilinear",
    choose_ensemble_member: str = "smallest",
    debug_zero_check: bool = False,
) -> xr.Dataset | None:
    """
    Open a CMIP/ERA/re-analysis file, apply unit fixes, and (if necessary)
    re-grid it onto a regular lon/lat mesh, with an optional zero-fill
    diagnostic to catch 0–360 vs. –180–180 longitude errors.

    The function is **resilient by design** – any unrecoverable problem is
    *logged* and causes a ``return None`` instead of raising, so that massively
    parallel workflows can just continue with the next file.  If
    ``debug_zero_check=True``, it will plot the first time slice and raise an
    exception when > 50 % of grid points are zero.

    Workflow
    --------
    1.  Open file(s) with :pyfunc:`open_any_cmip`
    2.  *Quirk fixes*  
        - drop inferior ``expver`` versions (ERA-5)  
        - keep only the smallest ``number`` ensemble (optional)  
        - convert various units (``pr``, ``tnsr``, ``tntr`` …)  
        - cap relative humidity, flip negative evaporation, etc.
    3.  Normalize coordinates (rename, wrap lon to ±180°, sort lat/lon)
    4.  **Regridding strategy**  
        - xESMF with ``periodic=True``  
        - xESMF with ``periodic=False``  
        - fallback to ``xr.Dataset.interp`` for rectilinear grids  
    5.  **(Optional)** Zero‐fill diagnostic: if enabled, compute the fraction
        of zeros in the first timestep of the first variable; if > 50 %, plot
        and raise an error.

    Parameters
    ----------
    zstore_link : str
        Anything supported by :pyfunc:`open_any_cmip`
        (local path, ssh `/net/atmos` path, `gs://` Zarr, pipe list …).
    resolution : float
        Target grid spacing in degrees (allowed: 0.5, 1, 2, 4, 5, 10).
    method : str
        xESMF interpolation method (`"bilinear"`, `"nearest_s2d"`, …).
    choose_ensemble_member : str
        `"smallest"` → select the member with the lowest `number` dim value.
        Any other string leaves ensemble data untouched.
    debug_zero_check : bool
        If True, after regridding compute the fraction of gridpoints equal to
        zero at time index 0. If that fraction exceeds 0.5, plot the first‐
        timestep map and raise a RuntimeError.

    Returns
    -------
    xarray.Dataset | None
        - re-gridded dataset – or the original if already on target grid.  
        - None when the file cannot be opened, when all regridding attempts
          fail, or when the zero‐fill diagnostic triggers (if enabled).

    Notes
    -----
    - The pure-xarray fallback can be 10–100× slower than xESMF but never
      touches ESMF and therefore avoids the well-known `GridCreate` bug.
    - For curvilinear grids the function will still give up after the two
      xESMF attempts, returning None.
    """
    # ────────────── 0. open dataset ──────────────────────────────────────
    try:
        ds = (
            open_any_cmip(zstore_link)
            if os.path.isfile(zstore_link)
            else open_any_cmip(zstore_link, consolidated=True)
        )
    except Exception as err:
        logging.error("%s — could not be opened: %s", zstore_link, err)
        return None

    # ────────────── 1. quick fixes & unit conversions ────────────────────
    if "expver" in ds.dims:
        ds = ds.groupby("time").apply(
            lambda x: select_best_expver(x, [1, 2, 3, 4, 5, 6, 7])
        )
    if choose_ensemble_member == "smallest" and "number" in ds.dims:
        ds = select_smallest_ensemble_member(ds)

    sec_per_month = ds["time"].dt.days_in_month * 24 * 3600
    if "tnsr" in ds:
        ds["tnsr"] = ds["tnsr"] / sec_per_month
    if "tntr" in ds:
        ds["tntr"] = ds["tntr"] / sec_per_month

    if "pr" in ds and ds["pr"].attrs.get("units") == "m":
        ds["pr"] = ds["pr"] * 1000 / (24 * 3600)
        ds["pr"].attrs["units"] = "kg m-2 s-1"

    if "evspsbl" in ds and ds["evspsbl"].mean().compute() < 0:
        ds["evspsbl"] = -ds["evspsbl"]
    if "hurs" in ds:
        ds["hurs"] = ds["hurs"].clip(max=100)

    # ────────────── 2. coordinate normalisation ──────────────────────────
    if "latitude" in ds and "lat" not in ds:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds and "lon" not in ds:
        ds = ds.rename({"longitude": "lon"})

    # wrap any 0–360 or >360 or <-180 longitudes into [-180,180]
    if (ds["lon"] > 180).any() or (ds["lon"] < -180).any():
        ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180)

    if not (ds["lon"].diff("lon") > 0).all():
        ds = ds.sortby("lon")
    if not (ds["lat"].diff("lat") > 0).all():
        ds = ds.sortby("lat")

    # ────────────── 3. already on target grid? ───────────────────────────
    target = create_target_grid(resolution)
    if (
        ds.sizes.get("lat") == target.sizes["lat"]
        and ds.sizes.get("lon") == target.sizes["lon"]
    ):
        return ds  # nothing to do

    # helper to copy variable & global attributes
    def _with_attrs(src, out):
        for v in src.variables:
            if v in out:
                out[v].attrs = src[v].attrs
        out.attrs = src.attrs
        out["time"].attrs = src["time"].attrs
        return out

    # ────────────── 4. xESMF (periodic & non-periodic) ───────────────────
    for periodic in (True, False):
        try:
            rgr = xe.Regridder(
                ds,
                target,
                method=method,
                periodic=periodic,
                reuse_weights=False,
            )
            ds_out = _with_attrs(ds, rgr(ds))
            break
        except ValueError as err:
            if "ESMC_GridCreate" not in str(err):
                logging.error("%s — unexpected xESMF error: %s", zstore_link, err)
                return None
            logging.warning(
                "%s — xESMF periodic=%s failed (%s)", zstore_link, periodic, err
            )
    else:
        # ────────────── 5. pure-xarray fallback ────────────────────────────
        if ds["lat"].ndim == ds["lon"].ndim == 1:
            logging.warning("%s — falling back to xarray.interp (slow fallback)", zstore_link)
            try:
                ds_out = _with_attrs(
                    ds,
                    ds.interp(
                        lon=target["lon"],
                        lat=target["lat"],
                        kwargs={"bounds_error": False},
                    ),
                )
            except Exception as err:
                logging.error("%s — xarray.interp failed: %s", zstore_link, err)
                return None
        else:
            logging.error("%s — regridding failed for curvilinear grid; file skipped", zstore_link)
            return None

    # ────────────── 6. optional zero‐fill diagnostic ─────────────────────
    if debug_zero_check:
        var0 = next(iter(ds_out.data_vars))
        da0 = ds_out[var0].isel(time=0)
        frac_zero = float((da0 == 0).sum()) / da0.size
        if frac_zero > 0.5:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            da0.plot(cmap="viridis", vmin=da0.min(), vmax=da0.max())
            plt.title(
                f"{var0} @ time=0 has {frac_zero:.0%} zeros – "
                "possible lon‐wrapping issue?"
            )
            plt.show()
            raise RuntimeError(
                f"Too many zeros ({frac_zero:.0%}) in first timestep; "
                "check your lon coordinate wrapping."
            )

    return ds_out




# %%
def select_smallest_ensemble_member(ds):
    """
    Selects the smallest ensemble member based on the 'number' variable and removes the 'number' dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the 'number' dimension.

    Returns
    -------
    xarray.Dataset
        The dataset with the smallest 'number' selected and 'number' dimension removed.
    """
    smallest_number = min(ds.number.values)  # Get the smallest 'number' value
    return ds.sel(number=smallest_number).drop('number')


# %%
def select_best_expver(ds, expver_priority):
    """
    Selects the best 'expver' based on availability and a defined priority order, and removes the 'expver' dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the 'expver' dimension.
    expver_priority : list
        The list of 'expver' values in the order of priority.

    Returns
    -------
    xarray.Dataset
        The dataset with the best 'expver' selected and 'expver' dimension removed.
    """
    for expver in expver_priority:
        if expver in ds.expver.values:
            return ds.sel(expver=expver).drop('expver')
    return ds.drop('expver')  # Fallback if none of the preferred 'expver' values are found



# %%
def reset_time(ds, start_year=1850):
    """
    Reset the time coordinate of a dataset to start from a specific year, 
    maintaining the relative difference in years.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset whose time coordinate is to be reset.
    start_year : int
        The year from which the time coordinate should start.

    Returns
    -------
    xarray.Dataset
        The dataset with the reset time coordinate.
    """
    original_time_values = ds['time'].values
    new_time_values = []

    # Determine the year difference from the first time value
    if isinstance(original_time_values[0], cftime.Datetime360Day):
        year_diff = start_year - original_time_values[0].year
    else:
        raise ValueError("Time values are not in cftime.Datetime360Day format")

    for time_value in original_time_values:
        new_year = time_value.year + year_diff
        new_date = cftime.Datetime360Day(new_year, time_value.month, time_value.day)
        new_time_values.append(new_date)

    ds['time'] = xr.DataArray(new_time_values, dims='time', name='time')
    return ds



# %%
def convert_to_target_cftime(ds, zstorefn, reset_time_coord=False, create_checkplot=True):
    """
    Converts an xarray Dataset's time coordinate to cftime.Datetime360Day calendar type,
    with special handling for monthly and daily data.

    Args:
        ds (xarray.Dataset): Dataset to convert.
        zstorefn (str): Zstore filename from which ds originates, just for logging purposes in case there are errors.
        reset_time_coord (bool): Whether to reset the time coordinate starting from a specific year.
        create_checkplot (bool): Whether to create a plot comparing original and converted time values.

    Returns:
        xarray.Dataset: Dataset with converted time coordinate in cftime.Datetime360Day format.
    """
    time_values = ds['time'].values

    # Directly handle cftime objects for time differences
    time_diffs = np.diff(time_values).astype('timedelta64[s]').astype(float)
    median_diff_days = np.median(time_diffs) / 86400

    print(f"The median_diff is {median_diff_days} days, which is used to decide the data frequency.")
    sys.stdout.flush()

    target_calendar = '360_day'
    converted_dates = []
    originaltime_tobeconverted_dates = []
    unique_dates_set = set()  # Track unique dates

    is_monthly_data = 27.0 <= median_diff_days <= 33.0

    for date in time_values:
        try:
            if isinstance(date, cftime.Datetime360Day):
                # Already in the target calendar, no conversion needed
                converted_date = date
            elif isinstance(date, np.datetime64):
                date_pd = pd.to_datetime(date)  # Convert np.datetime64 to pandas datetime

                # Check if it's monthly data and the day is the 1st
                if is_monthly_data and date_pd.day == 1:
                    date_pd += pd.Timedelta(days=14)  # Shift from 1st to 15th of the month

                day_of_year = (date_pd - pd.Timestamp(date_pd.year, 1, 1)).days + 1
                # Handle days beyond 360
                if day_of_year > 360:
                    if date_pd.month < 12 or (date_pd.month == 12 and date_pd.day < 25):
                        print(f"Warning: Date {date_pd} beyond 360th day is being discarded.")
                        sys.stdout.flush()
                    continue
                month = (day_of_year - 1) // 30 + 1
                day = ((day_of_year - 1) % 30) + 1
                converted_date = cftime.Datetime360Day(date_pd.year, month, day)
            elif isinstance(date, (cftime.DatetimeNoLeap, cftime.DatetimeProlepticGregorian, cftime.DatetimeJulian, cftime.DatetimeGregorian)):
                # Check if it's monthly data and the day is the 1st
                if is_monthly_data and date.day == 1:
                    date = date.replace(day=15)  # Shift from 1st to 15th of the month

                day_of_year = (date - date.replace(month=1, day=1)).days + 1
                # Handle days beyond 360
                if day_of_year > 360:
                    if date.month < 12 or (date.month == 12 and date.day < 25):
                        print(f"Warning: Date {date} beyond 360th day is being discarded.")
                        sys.stdout.flush()
                    continue
                month = (day_of_year - 1) // 30 + 1
                day = ((day_of_year - 1) % 30) + 1
                converted_date = cftime.Datetime360Day(date.year, month, day)
            else:
                # General conversion for other cftime types
                try:
                    date_num = cftime.date2num(date, target_calendar)
                    date_converted = cftime.num2date(date_num, target_calendar)
                    day_of_year = (date_converted - cftime.Datetime360Day(date_converted.year, 1, 1)).days + 1
                except ValueError:
                    logging.error(f"{zstorefn} --- Unsupported calendar type: {type(date)}")
                    return None  # Return None to indicate error

            if is_monthly_data:
                # Monthly data: Keep 15th of the month
                converted_date = cftime.Datetime360Day(converted_date.year, converted_date.month, 15)
            elif 0.95 <= median_diff_days <= 1.05:
                # Daily data already handled above
                pass
            else:
                logging.error(f"{zstorefn} --- Time intervals of {median_diff_days} days do not correspond to assumed monthly or daily data.")
                return None  # Return None to indicate error

        except Exception as e:
            logging.error(f"{zstorefn} --- Error converting time value: {date}. {e}")
            return None  # Return None to indicate error

        # Check for duplicate dates
        if converted_date in unique_dates_set:
            logging.warning(f"{zstorefn} --- Duplicate converted date found: {converted_date}. Discarding.")
            continue

        unique_dates_set.add(converted_date)
        converted_dates.append(converted_date)
        originaltime_tobeconverted_dates.append(date)

    # Debugging: Check lengths before subsetting
    sys.stdout.flush()

    # Ensure no None values
    valid_indices = [i for i, converted_time in enumerate(converted_dates) if converted_time is not None]
    valid_original_dates = [originaltime_tobeconverted_dates[i] for i in valid_indices]
    valid_converted_dates = [converted_dates[i] for i in valid_indices]

    sys.stdout.flush()

    # Check for duplicates in valid_original_dates
    if len(valid_original_dates) != len(set(valid_original_dates)):
        print("Duplicate dates found in valid_original_dates")
        sys.stdout.flush()
        unique_valid_original_dates, indices = np.unique(valid_original_dates, return_index=True)
        valid_converted_dates = [valid_converted_dates[i] for i in indices]
        valid_original_dates = unique_valid_original_dates.tolist()

    # Subset the dataset based on the original time values that were successfully converted
    ds = ds.sel(time=valid_original_dates)
    # Assign the new converted time values to the time dimension
    ds['time'] = xr.DataArray(valid_converted_dates, dims='time', name='time')

    if len(time_values) > len(converted_dates):
        ds.attrs['calendar_conversion_warning'] = "Some days beyond the 360th day have been discarded during conversion."

    if reset_time_coord:
        ds = reset_time(ds, start_year=1850)

    if create_checkplot:
        converted_time_values = ds['time'].values
        plot_time_comparison(time_values, converted_time_values)

    return ds



# %%

def delete_duplicate_timeentries(ds):
    """
    Delete duplicate time entries in an xarray.Dataset.

    Parameters:
    ds (xarray.Dataset): The input dataset with a time coordinate.

    Returns:
    xarray.Dataset: The dataset with duplicate time entries removed.
    """
    # Extract time values
    time_values = ds['time'].values

    # Identify duplicate time points
    _, unique_indices = np.unique(time_values, return_index=True)
    if len(unique_indices) == len(time_values):
        # No duplicates found
        return ds

    # Identify indices of duplicate entries
    duplicate_indices = np.setdiff1d(np.arange(len(time_values)), unique_indices)

    # Filter out the duplicate time entries
    ds_filtered = ds.isel(time=unique_indices)

    return ds_filtered



# %%
def plot_time_comparison(original_time_values, converted_time_values):
    """
    Plots a comparison graph between original and converted time values.

    Args:
        original_time_values (array-like): Original time values (cftime or numpy.datetime64).
        converted_time_values (array-like): Converted time values (cftime).
    """

    # Check input types and extract years and days accordingly
    if isinstance(original_time_values[0], cftime.datetime):
        original_years = [d.year for d in original_time_values]
        original_days = [d.day for d in original_time_values]
    else:
        original_years = [d.year for d in pd.to_datetime(original_time_values)]  # Convert to pandas datetimes
        original_days = [d.day for d in pd.to_datetime(original_time_values)]

    converted_years = [d.year for d in converted_time_values]
    converted_days = [d.day for d in converted_time_values]

    # Creating the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plotting original dates
    ax.scatter(original_years, original_days, alpha=0.6, edgecolor='none', label='Original')

    # Plotting converted dates
    ax.scatter(converted_years, converted_days, alpha=0.6, edgecolor='none', label='Converted')

    # Formatting the plot
    ax.set_xlabel('Year')
    ax.set_ylabel('Day of Month')
    ax.set_title('Comparison of Original and Converted Time Values')
    ax.grid(True)
    ax.legend()

    plt.show()



# %%
def clean_fill_attributes(ds):
    """
    Cleans the _FillValue and missing_value attributes in an xarray.Dataset.
    Ensures that only one of them exists per variable with consistent values.
    Additionally, removes any variables that are unnamed (i.e., have a name of None).
    
    Parameters:
    - ds (xarray.Dataset): The dataset to clean.
    
    Returns:
    - xarray.Dataset: The cleaned dataset.
    """
    variables_to_remove = []
    
    for var_name in ds.variables:
        # Check for variables named None
        if var_name is None:
            print("Found a variable with name 'None'. Removing it.")
            variables_to_remove.append(var_name)
            continue
        
        da = ds[var_name]
        
        # Check for conflicting _FillValue and missing_value attributes
        if '_FillValue' in da.attrs and 'missing_value' in da.attrs:
            if da.attrs['_FillValue'] != da.attrs['missing_value']:
                # Decide to keep only _FillValue and remove missing_value
                print(f"Variable '{var_name}' has conflicting _FillValue and missing_value. Removing 'missing_value'.")
                del da.attrs['missing_value']
            else:
                # If they are the same, remove one to avoid redundancy
                print(f"Variable '{var_name}' has identical _FillValue and missing_value. Removing 'missing_value'.")
                del da.attrs['missing_value']
    
    # Remove unnamed variables
    if variables_to_remove:
        ds = ds.drop_vars(variables_to_remove)
    
    return ds




# %%
def process_regridding_files(files2crunch_df, target_resolution, lon_convention: str = "180"):
    regridded_data_handles = {}
    exclude_vars = ['rsdt', 'rlut', 'rsut', 'rsnt', 'rsdt_parent', 'rlut_parent', 'rsut_parent', \
                    'tnsr','tntr',  'mtdwswrf', 'mtnlwrf', 'mtnswrf', 'siarean'] # ,'tnsr','tntr'
    exclude_vars = ['rsdt_parent', 'rlut_parent', 'rsut_parent', \
                     'siarean'] # ,'tnsr','tntr'
    new_scenarios = ['abrupt-4xCO2', 'abrupt-2xCO2', 'abrupt-0p5xCO2', '1pctCO2']

    # Iterate over the MultiIndex
    for (source_id, experiment_id, member_id), row in files2crunch_df.iterrows():
        print(f"--------------------------------------------------------")
        print(f"Now Processing {source_id}, {experiment_id}, {member_id}....")
 
        reset_time_coord = experiment_id in new_scenarios 
        
        # Iterate through all variables in the row
        for var in row.index:
            if var in exclude_vars:
                continue

            zstore_filename = row[var]
            
            if var.endswith('_parent'): 
                childorparent = 'parent'
                actualvar = var.split('_parent')[0]
            else: 
                childorparent = 'child'
                actualvar = var
                

            if pd.notna(zstore_filename):
                print(f'.....Now trying to regrid the {childorparent} variable {var} from file {zstore_filename}')
                
                try:
                    # Attempt to regrid the dataset
                    gridded_data = regrid_dataset_if_needed(
                        zstore_filename,
                        resolution=target_resolution,
                        method='bilinear',
                        lon_convention=lon_convention,
                    )
                except Exception as e:
                    error_message = (f"Error when regridding {source_id}, {experiment_id}, {member_id}, "
                                     f"variable {var} from file {zstore_filename}: {str(e)}")
                    logging.error(error_message)
                    continue

                if gridded_data is None: 
                    error_message = (f"f'Error when reading {source_id}, {experiment_id}, {member_id} and file {zstore_filename}...Check logging ")
                    logging.error(error_message)
                    continue
                    
                gridded_data = delete_duplicate_timeentries(gridded_data) 
                    
                converted_data = convert_to_target_cftime(gridded_data, zstore_filename, reset_time_coord=reset_time_coord, create_checkplot=False)

                if converted_data is None or converted_data.sizes['time'] != gridded_data.sizes['time']:
                    error_message = (f"{zstore_filename} --- Error after converting time coordinate: dimension mismatch.")
                    logging.error(error_message)
                    continue

                regridded_data_handles[(source_id, experiment_id, member_id, var)] = converted_data
            else: 
                print(f'.')

    return regridded_data_handles



# %%
def process_single_row(
    key: tuple[str, str, str],
    row: "pd.Series",
    target_resolution: float,
    exclude_vars: list[str],
    lon_convention: str = "180",
) -> dict[tuple[str, str, str, str], "xr.Dataset"]:
    """
    Re-grid **all variables of one catalogue row** (model–scenario–member)
    and convert their calendar to a common CF-time representation.

    The function is intentionally *fault-tolerant*: every failure on a single
    variable/file is **logged** and silently skipped, while the rest of the
    variables are still processed.  This is crucial for large parallel jobs
    where the occasional corrupt or exotic grid must not abort the whole run.

    Parameters
    ----------
    key
        The *MultiIndex* key coming from ``DataFrame.iterrows()``:
        ``(source_id, experiment_id, member_id)``.
    row
        ``pd.Series`` whose *index* are variable names and whose *values* are
        file paths (or pipe-separated path lists) pointing to NetCDF/Zarr data.
    target_resolution
        Desired output grid spacing in degrees (0.5, 1, 2, 4 …) – passed
        straight to :pyfunc:`regrid_dataset_if_needed`.
    exclude_vars
        List of variable names that should be ignored completely
        (parent/child bookkeeping, RSUT_parent, …).

    Returns
    -------
    dict
        Mapping **(source_id, experiment_id, member_id, variable)** →
        *regridded* & *calendar-fixed* :pyclass:`xarray.Dataset`.

        Variables that failed to open or re-grid are *not* included.
    """
    source_id, experiment_id, member_id = key
    print(f"Processing row → {source_id}, {experiment_id}, {member_id}")

    # certain idealised experiments start at fixed model year → reset cal.
    reset_time_coord = experiment_id in {
        "abrupt-4xCO2",
        "abrupt-2xCO2",
        "abrupt-0p5xCO2",
        "1pctCO2",
    }

    out: dict[tuple[str, str, str, str], "xr.Dataset"] = {}

    for var in row.index:
        if var in exclude_vars:
            continue

        zstore_filename = row[var]
        if pd.isna(zstore_filename):
            continue  # empty cell in availability table

        try:
            # 1. open + re-grid (may return None on failure)
            ds = regrid_dataset_if_needed(
                zstore_filename,
                resolution=target_resolution,
                method="bilinear",
                lon_convention=lon_convention,
            )
            if ds is None:
                logging.error(
                    "%s – %s/%s/%s returned None → skipped",
                    zstore_filename,
                    source_id,
                    experiment_id,
                    var,
                )
                continue

            # 2. remove duplicate timestamps (if any)
            ds = delete_duplicate_timeentries(ds)

            # 3. convert to *proleptic_gregorian* / target CF calendar
            ds = convert_to_target_cftime(
                ds,
                zstore_filename,
                reset_time_coord=reset_time_coord,
                create_checkplot=False,
            )
            if ds is None:
                logging.error(
                    "%s – calendar conversion failed → skipped", zstore_filename
                )
                continue

            # 4. store in the result dict
            out[(source_id, experiment_id, member_id, var)] = ds

        except Exception as err:
            logging.error(
                "Unhandled error while processing %s (%s/%s/%s): %s",
                zstore_filename,
                source_id,
                experiment_id,
                var,
                err,
                exc_info=True,
            )
            # continue with the next variable

    return out



# %%

# -----------------------------------------------------------------------------#
#  PARALLEL REGRIDDER                                                           #
# -----------------------------------------------------------------------------#
def process_files_parallel(
    files2crunch_df: pd.DataFrame,
    target_resolution: float | int,
    num_jobs: int = -1,
    lon_convention: str = "180",
) -> dict[tuple[str, str, str, str], xr.Dataset]:
    """
    Re-grid – in parallel – all CMIP/obs files listed in *files2crunch_df*.

    The dispatcher wraps :pyfunc:`process_single_row` in a **thread** pool
    (no pickle barrier) and merges the per-row results into a single dict.

    Parameters
    ----------
    files2crunch_df
        Multi-indexed by *(source_id, experiment_id, member_id)* and one
        column per variable holding a path (or pipe-separated paths).
    target_resolution
        Output grid spacing in degrees (e.g. ``1`` → 1 × 1°).
    num_jobs
        Thread count.  ``-1`` = all logical CPUs; ``1`` runs serial.

    Returns
    -------
    dict
        Mapping ``(source_id, experiment_id, member_id, variable)``
        → re-gridded :pyclass:`xarray.Dataset`.
    """
    # variables that do *not* need re-gridding here
    exclude_vars = [
        "rsdt_parent", "rlut_parent", "rsut_parent", "siarean",
        # rsdt/rlut/rsut/rsnt stay in the dict; they are handled elsewhere
    ]

    # ------------------------------------------------------------------ 1. work list
    tasks: list[tuple[tuple[str, str, str], pd.Series]] = [
        (key, row) for key, row in files2crunch_df.iterrows()
    ]
    if not tasks:
        logging.warning("process_files_parallel: nothing to do – empty DataFrame?")
        return {}

    print(f"→ Re-gridding {len(tasks)} model-scenario-member rows "
          f"with {len(files2crunch_df.columns)} variables each")

    # ------------------------------------------------------------------ 2. run
    def _run_single(key, row):
        return process_single_row(key, row, target_resolution, exclude_vars, lon_convention)

    if num_jobs == 1:                                  # ------- serial fallback
        results = [ _run_single(k, r) for k, r in tasks ]
    else:                                              # ------- threaded
        with Parallel(n_jobs=num_jobs,
                      backend="threading",
                      prefer="threads") as PL:
            results = PL(
                delayed(_run_single)(k, r) for k, r in tasks
            )

    # ------------------------------------------------------------------ 3. merge
    merged: dict[tuple[str, str, str, str], xr.Dataset] = {}
    for d in results:
        merged.update(d)

    print(f"✓  Finished re-gridding – collected {len(merged)} variable-level handles")
    return merged

# %%

def process_files_parallel_notworking(
    files2crunch_df: pd.DataFrame,
    target_resolution: float | int,
    num_jobs: int = -1,
    lon_convention: str = "180",
) -> dict[tuple[str, str, str, str], xr.Dataset]:
    """
    Re-grid – in parallel – all CMIP/obs files listed in *files2crunch_df*.

    The dispatcher wraps :pyfunc:`process_single_row` in a **thread** pool
    (no pickle barrier) and automatically shuts down that pool when the
    block exits, so no orphan Joblib threads survive the notebook.

    Parameters
    ----------
    files2crunch_df
        Multi-indexed by *(source_id, experiment_id, member_id)* and one
        column per variable holding a path (or pipe-separated paths).
    target_resolution
        Output grid spacing in degrees (e.g. ``1`` → 1 × 1°).
    num_jobs
        Thread count. ``-1`` = all logical CPUs. ``1`` runs serial.

    Returns
    -------
    dict
        Mapping ``(source_id, experiment_id, member_id, variable)``
        → re-gridded :pyclass:`xarray.Dataset`.
    """
    # variables that do *not* need re-gridding here
    exclude_vars = [
        "rsdt_parent", "rlut_parent", "rsut_parent", "siarean",
        # rsdt/rlut/rsut/rsnt stay in the dict; they are handled elsewhere
    ]

    # ------------------------------------------------------------------ helper
    def _submit():
        return (
            delayed(process_single_row)(key, row, target_resolution,
                                        exclude_vars, lon_convention)
            for key, row in files2crunch_df.iterrows()
        )

    # ------------------------------------------------------------------ run
    merged: dict = {}

    if num_jobs == 1:                                    # serial fall-back
        for d in map(lambda t: t(), _submit()):
            merged.update(d)
    else:
        from joblib import Parallel, delayed
        with Parallel(n_jobs=num_jobs,
                      backend="threading",
                      prefer="threads") as PL:
            for d in PL(_submit()):
                merged.update(d)

    return merged




# %%

def process_files_parallel_old(
    files2crunch_df,
    target_resolution: float | int,
    num_jobs: int = -1,
) -> dict:
    """
    Regrid – in parallel – all CMIP/obs files listed in *files2crunch_df*.

    The function is a thin thread-based dispatcher around
    :pyfunc:`process_single_row`, which handles one
    ``(source_id, experiment_id, member_id)`` row at a time.

    Why threads instead of processes?
    ---------------------------------
    * Each worker opens datasets with **h5netcdf**; the resulting
      ``xarray.Dataset`` still owns an internal ``_thread.lock`` tied to
      the low-level HDF5 file handle.  Such objects are **not picklable**,
      so returning them from a forked process would fail with  
      ``TypeError: cannot pickle '_thread.lock' object``.
    * The workload is mostly **I/O-bound** (SSH/SFTP streaming +
      regridding) – threads give almost the same throughput as
      multiprocessing without the pickle barrier or extra RAM.

    Parameters
    ----------
    files2crunch_df : pandas.DataFrame
        Multi-indexed by (``source_id``, ``experiment_id``, ``member_id``)
        and with one column per variable containing the local/remote
        filename(s) (pipe-separated when split over chunks).
    target_resolution : float or int
        Desired output grid resolution (e.g. 1 → 1 × 1 degree).
    num_jobs : int, default ``-1``
        Number of threads.  ``-1`` = all logical CPUs.  Positive values
        set an explicit upper bound; ``1`` makes the call serial.

    Returns
    -------
    dict
        ``{(source_id, experiment_id, member_id, variable): xarray.Dataset}``

        The dictionary merges the individual results from each worker.

    Notes
    -----
    * Variables in *exclude_vars* are skipped to avoid needless work
      (e.g. parent chunks already handled elsewhere).
    * If you later need **true process-level parallelism** for
      CPU-heavy steps, either:
        1. call ``ds.load(); ds = ds.copy(deep=True); ds.close()`` inside
           *process_single_row* before returning the dataset, **or**
        2. switch to dask-backed arrays and return only lightweight
           dask graphs (which are picklable).

    Examples
    --------
    >>> handles = process_files_parallel(files2crunch_df, target_resolution=1)
    >>> list(handles.keys())[:3]
    [('CNRM-CM6-1', 'historical', 'r1i1p1f2', 'tas'),
     ('CNRM-CM6-1', 'historical', 'r1i1p1f2', 'pr'),
     ('CNRM-CM6-1', 'historical', 'r1i1p1f2', 'rsdt')]
    """
    # variables that do NOT need regridding here
    exclude_vars = [
        "rsdt_parent", "rlut_parent", "rsut_parent", "siarean",
        # previously excluded: rsdt, rlut, rsut, rsnt, ... etc.
    ]

    results = Parallel(
        n_jobs=num_jobs,
        backend="threading",   # ← avoids pickling Dataset objects
        prefer="threads",
    )(
        delayed(process_single_row)(
            key, row, target_resolution, exclude_vars
        )
        for key, row in files2crunch_df.iterrows()
    )

    # merge per-row dictionaries into one
    merged: dict = {}
    for d in results:
        merged.update(d)
    return merged


# %% [markdown]
## calculate the global mean



# -----------------------------------------------------------------------------#
#  GLOBAL-MEAN WORKER                                                          #
# -----------------------------------------------------------------------------#


# -----------------------------------------------------------------------------#
#  GLOBAL-MEAN WORKER                                                          #
# -----------------------------------------------------------------------------#
def _global_mean_worker(
    key: tuple[str, str, str, str],
    ds: xr.Dataset,
    var_trunk: str,
    period_years: int,
) -> tuple[
    tuple[str, str, str, str],    # ← full key
    xr.DataArray | None,          # ← raw global-mean   (°C, W m-2, …)
    xr.DataArray | None,          # ← smoothed (low-pass) series
]:
    """Internal helper executed in *threads* (never pickled).

    It receives only the **dataset it needs**, so there are no `_thread.lock`
    objects to pickle – one of the reasons the original ``loky`` run failed.
    After the global mean is calculated the function closes *ds* to release
    the underlying NetCDF/Zarr file handle, while the in-memory copy of the
    data remains available for step 5.
    """
    try:
        gm = calculate_global_mean(ds, var_trunk)
        sm = lowpass_filter_timeseries(gm, period_years=period_years,
                                       plotoutput=False)
        return key, gm, sm
    except Exception as err:
        logging.error("Global-mean failed for %s: %s", key, err, exc_info=True)
        return key, None, None
    finally:
        # Detach from any open HDF5/Zarr reader – keeps memory, frees FD/cache
        try:
            ds.close()
        except Exception:
            pass



def _global_mean_worker_old(
    key: tuple[str, str, str, str],
    ds: xr.Dataset,
    var_trunk: str,
    period_years: int,
) -> tuple[
    tuple[str, str, str, str],    # ← full key
    xr.DataArray | None,          # ← raw global-mean   (°C, W m-2, …)
    xr.DataArray | None,          # ← smoothed (low-pass) series
]:
    """Internal helper executed in *threads* (never pickled).

    It receives only the **dataset it needs**, so there are no `_thread.lock`
    objects to pickle – one of the reasons the original ``loky`` run failed.
    """
    try:
        gm = calculate_global_mean(ds, var_trunk)
        sm = lowpass_filter_timeseries(gm, period_years=period_years, plotoutput=False)
        return key, gm, sm
    except Exception as err:
        logging.error("Global-mean failed for %s: %s", key, err, exc_info=True)
        return key, None, None


# -----------------------------------------------------------------------------#
#  PUBLIC API                                                                  #
# -----------------------------------------------------------------------------#
def process_global_means_parallel(
    regridded_data_handles: dict[tuple[str, str, str, str], xr.Dataset],
    vars4globalmean:        Iterable[str],
    *,
    period_years: int        = 21,
    use_parallel: bool       = False,
    n_jobs: int              = 20,
) -> tuple[
    dict[tuple[str, str, str, str], xr.DataArray],
    dict[tuple[str, str, str, str], xr.DataArray],
]:
    """
    Compute **area-weighted global means** and 21-year *loess*-smoothed
    versions for a whole collection of already-regridded CMIP/obs datasets.

    The function is *thread-based* to avoid the cloudpickle ``_thread.lock``
    problem that appears when trying to ship :pyclass:`xarray.Dataset`
    instances to separate *processes*.

    Parameters
    ----------
    regridded_data_handles
        Mapping ``(source_id, experiment_id, member_id, variable) → Dataset``.
        Typically the output of :pyfunc:`process_files_parallel`.
    vars4globalmean
        Only these variable names are processed.  Suffixes like ``_parent``
        or ``_parentandchild`` are stripped *before* the membership test.
    period_years
        Window size for the low-pass (loess) filter.
    use_parallel
        ``True`` → multi-threaded with :pymod:`joblib`  
        ``False`` → simple sequential loop (useful for debugging).
    n_jobs
        Number of worker threads when ``use_parallel`` is *True*.

    Returns
    -------
    (global_means, smoothed_means)
        Two dictionaries with the **same keys** as *regridded_data_handles*
        (but only for successfully processed variables).

    Notes
    -----
    * A variable that raises *any* exception during global-mean or smoothing
      is logged and silently skipped – the rest of the workflow continues.
    * Threading backend keeps memory-sharing cheap and avoids any pickle
      overhead.  This means you must **not** switch to the ``loky`` backend
      here unless you first prove that all objects can indeed be pickled.
    """
    # ─────────────────────── 0.  build work list ──────────────────────────
    work: list[tuple[
        tuple[str, str, str, str],  # ← original key
        xr.Dataset,                 # ← dataset
        str                         # ← plain variable name (no suffix)
    ]] = []

    for key, ds in regridded_data_handles.items():
        var = key[3]
        var_trunk: str = (
            var.replace("_parentandchild", "").replace("_parent", "")
        )
        if var_trunk in vars4globalmean and not var.endswith("_parent"):
            work.append((key, ds, var_trunk))

    if not work:
        logging.warning("process_global_means_parallel: no variables matched.")
        return {}, {}

    # ─────────────────────── 1.  run (threaded or serial) ─────────────────
    global_means:   dict[Any, xr.DataArray] = {}
    smoothed_means: dict[Any, xr.DataArray] = {}

    if use_parallel:
        func = partial(_global_mean_worker, period_years=period_years)
        with joblib.Parallel(
            n_jobs=n_jobs, backend="threading", prefer="threads"
        ) as PL:
            for k, gm, sm in tqdm(
                PL(joblib.delayed(func)(k, ds, vt) for k, ds, vt in work),
                total=len(work),
            ):
                if gm is not None:
                    global_means[k]   = gm
                    smoothed_means[k] = sm
    else:
        for k, ds, vt in tqdm(work):
            k, gm, sm = _global_mean_worker(
                k, ds, vt, period_years=period_years
            )
            if gm is not None:
                global_means[k]   = gm
                smoothed_means[k] = sm

    return global_means, smoothed_means




# %%
def process_global_means_for_key(key, regridded_data_handles, vars4globalmean, period_years):
    var = key[3]
    vartrunk = var.replace('_parentandchild', '').replace('_parent', '')

    if var.endswith('_parent') or vartrunk not in vars4globalmean:
        return None, None

    chunk_size = determine_ideal_chunk_size(regridded_data_handles, key)
    global_mean = calculate_global_mean_dask(regridded_data_handles[key], vartrunk, chunk_size)

    smoothed_global_mean = lowpass_filter_timeseries(global_mean, period_years, plotoutput=False)
    print(f'type of smoothed global mean: {type(smoothed_global_mean)}')

    return (key, global_mean), (key, smoothed_global_mean)



# %%
def calculate_global_mean(data_input, variable):
    """
    Calculate the global-mean of a variable from a zstore link or an xarray Dataset.
    
    Parameters
    ----------
    data_input : str or xarray.Dataset
        The zarr store link to the dataset in the cloud or an xarray Dataset.
    variable : str
        The variable name to calculate the global-mean for.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray with the time dimension and the global-mean of the data field.
    """
    # Check the type of data_input and open the dataset accordingly
    if isinstance(data_input, str):
        ds = open_any_cmip(data_input, consolidated=True)
    elif isinstance(data_input, xr.Dataset):
        ds = data_input
    else:
        raise ValueError("data_input must be a zstore filename (str) or an xarray.Dataset")

    # Ensure the specified variable exists in the dataset
    if variable not in ds:
        raise ValueError(f"Variable '{variable}' not found in the dataset")

    # Get the variable of interest
    da = ds[variable]

    # Detect latitude and longitude dimension names
    lat_name = 'lat' if 'lat' in da.dims else 'latitude'
    lon_name = 'lon' if 'lon' in da.dims else 'longitude'

    # Check for the presence of latitude and longitude dimensions
    if lat_name not in da.dims or lon_name not in da.dims:
        raise ValueError(f"Latitude or longitude dimensions not found in the dataset for variable '{variable}'")

    # Calculate the weights as the cosine of the latitude
    weights = np.cos(np.deg2rad(da[lat_name]))

    print(f'Check: Mean of lats: {np.mean(da[lat_name])}')
    print(f'Min of lat: {np.nanmin(da[lat_name])}')
    print(f'Max of lat: {np.nanmax(da[lat_name])}')
    print(f'Check: 5-percentile of lat: {np.nanpercentile(da[lat_name], 5)}')
    print(f'Check: 95-percentile of lat: {np.nanpercentile(da[lat_name], 95)}')

    # Handle weights close to the poles - set weights to zero for latitudes very close to the poles
    weights = np.where((weights < 0) & (((da[lat_name] <= -90.0) & (da[lat_name] >= -90.1)) | ((da[lat_name] >= 90.0) & (da[lat_name] <= 90.1))), 0, weights)

    # Broadcast weights to match the dimensions of the data array
    weights = xr.DataArray(weights, dims=[lat_name], coords={lat_name: da[lat_name]})
    weights = weights.broadcast_like(da)

    # Sense checks for weights
    if np.any(weights < 0):
        raise ValueError("Weights contain negative values, check latitude data")
    if np.isnan(weights).any():
        raise ValueError("Weights contain NaN values, check latitude data")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Weights contain infinite values, check latitude data")

    print(f'Check: Mean of weights: {weights.mean().values}')
    print(f'Min of weight: {weights.min().values}')
    print(f'Max of weight: {weights.max().values}')
    print(f'Check: 5-percentile of weights: {np.nanpercentile(weights, 5)}')
    print(f'Check: 95-percentile of weights: {np.nanpercentile(weights, 95)}')

    # Calculate the weighted mean
    weighted_sum = (da * weights).sum(dim=[lat_name, lon_name], skipna=True)
    total_weight = weights.sum(dim=[lat_name, lon_name], skipna=True)

    if np.isnan(weighted_sum).all() or np.isnan(total_weight).all():
        raise ValueError("Weighted sum or total weight calculation resulted in NaN values, check data and weights")

    print(f'The weighted_sum is: {weighted_sum} and the total weight is: {total_weight}')
    global_mean = weighted_sum / total_weight

    # Adjust the name of the DataArray
    global_mean.name = f'{variable}_global_mean'

    # Copy the global attributes from the source dataset, preserving calendar type
    global_mean.attrs = ds.attrs
    global_mean["time"].attrs = ds["time"].attrs

    # Create a fake data field filled with ones for verification
    fake_da = xr.DataArray(np.ones_like(da.values), dims=da.dims, coords=da.coords)
    fake_weighted_sum = (fake_da * weights).sum(dim=[lat_name, lon_name], skipna=True)
    fake_global_mean = fake_weighted_sum / total_weight

    print(f"Check: Global mean of fake data field should be 1. Calculated value: {fake_global_mean.values}")

    return global_mean






# %%
def determine_ideal_chunk_size(regridded_data_handles, key):
    dataset = regridded_data_handles[key]
    if isinstance(dataset, xr.Dataset):
        chunk_sizes = {dim: min(size, 100) for dim, size in dataset.sizes.items()}
        return chunk_sizes
    return None



# %%
def calculate_global_mean_dask(data_input, variable, chunk_size=(100, 100, 100)):
    """
    Calculate the global-mean of a variable from a zstore link or an xarray Dataset.
    
    Parameters:
    - data_input: The zarr store link to the dataset in the cloud or an xarray Dataset.
    - variable: The variable name to calculate the global-mean for.
    - chunk_size: Tuple representing the chunk sizes for each dimension.

    Returns:
    An xarray.DataArray with the time dimension and the global-mean of the data field.
    """
    
    # Check if data_input is a future and retrieve the result if it is
    if isinstance(data_input, dask.distributed.Future):
        data_input = data_input.result()

    print(f'Debug: data_input type is {type(data_input)}')
    
    # Chunk the dataset 
    if isinstance(data_input, str):
        ds = open_any_cmip(data_input, consolidated=True).chunk(chunks=chunk_size)
    elif isinstance(data_input, xr.Dataset):
        ds = data_input.chunk(chunks=chunk_size)
    else:
        raise ValueError("data_input must be a zstore filename (str) or an xarray.Dataset")

    if variable not in ds:
        raise ValueError(f"Variable '{variable}' not found in the dataset")

    da = ds[variable]
    lat_name = 'lat' if 'lat' in da.dims else 'latitude'
    lon_name = 'lon' if 'lon' in da.dims else 'longitude'

    if lat_name not in da.dims or lon_name not in da.dims:
        raise ValueError(f"Latitude or longitude dimensions not found in the dataset for variable '{variable}'")

    weights = np.cos(np.deg2rad(da[lat_name]))
    weights = np.where((weights < 0) & (((da[lat_name] <= -90.0) & (da[lat_name] >= -90.1)) | ((da[lat_name] >= 90.0) & (da[lat_name] <= 90.1))), 0, weights)
    weights = xr.DataArray(weights, dims=[lat_name], coords={lat_name: da[lat_name]})
    weights = weights.broadcast_like(da)

    weighted_sum = (da * weights).sum(dim=[lat_name, lon_name], skipna=True)
    total_weight = weights.sum(dim=[lat_name, lon_name], skipna=True)
    global_mean = weighted_sum / total_weight

    global_mean.name = f'{variable}_global_mean'
    global_mean.attrs = ds.attrs
    global_mean["time"].attrs = ds["time"].attrs

    return global_mean


# %%
def plot_global_mean_timeseries(global_mean_da, variable_name, smoothed_da=None):
    """
    Plot a time series of global mean values from an xarray DataArray with appropriate labeling
    derived from the data's attributes, and optionally plot a smoothed series on top.

    Parameters
    ----------
    global_mean_da : xarray.DataArray
        The DataArray containing the global mean time series.
    variable_name : str
        The name of the variable to be plotted.
    smoothed_da : xarray.DataArray, optional
        An optional DataArray containing a smoothed time series to be plotted on top.
    """
    # Ensure that 'time' is a datetime index compatible with matplotlib
    if not isinstance(global_mean_da['time'].values[0], np.datetime64):
        time = xr.coding.cftimeindex.CFTimeIndex(global_mean_da['time'].values).to_datetimeindex()
    else:
        time = global_mean_da['time'].values

    # Use attributes to determine units and long_name, if available
    units = global_mean_da.attrs.get('units', '')
    long_name = global_mean_da.attrs.get('long_name', variable_name)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot the original data series
    ax.plot(time, global_mean_da.values, label=f'Global Mean {long_name}', color='blue')

    # Plot the smoothed data series if provided
    if smoothed_da is not None:
        # Ensure that 'time' for smoothed_da is a datetime index compatible with matplotlib
        if not isinstance(smoothed_da['time'].values[0], np.datetime64):
            smoothed_time = xr.coding.cftimeindex.CFTimeIndex(smoothed_da['time'].values).to_datetimeindex()
        else:
            smoothed_time = smoothed_da['time'].values

        ax.plot(smoothed_time, smoothed_da.values, label=f'Smoothed {long_name}', color='red')

    # Formatting the plot
    ax.set_title(f'Global Mean {long_name} Time Series')
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{long_name} ({units})')
    ax.legend()
    ax.grid(True)

    plt.show()

# Example usage:
# plot_global_mean_timeseries(global_mean_result, 'tas', global_mean_smoothed_result)



# %% [markdown]
# ## lowpass filter global mean

# %%

def lowpass_filter_timeseries(da, period_years, plotoutput=False):
    """
    Apply multiple lowpass filter methods to a time series DataArray and return the smoothed time series.
    
    Methods included:
    - LOESS (Locally Estimated Scatterplot Smoothing)
    
    Parameters:
    - da: xarray DataArray, the time series to be smoothed.
    - period_years: int, the smoothing period in years.
    - plotoutput: bool, if True, plots the original and smoothed time series.
    
    Returns:
    - smoothed_results: dict, containing smoothed time series from each method.
    """
    
    # Determine window size based on the 'time' coordinate frequency
    days_in_year = 365 if da.time.dt.calendar == 'noleap' else 365.25
    calendar_type = da.time.attrs.get('calendar_type', None)
    interval_days = da.time.diff('time').mean().dt.days.values

    if calendar_type == '360_day':
        days_in_year = 360
    elif calendar_type == 'noleap' or calendar_type == '365_day':
        days_in_year = 365
    elif calendar_type == 'gregorian' or calendar_type == 'standard':
        days_in_year = 365.25
    elif calendar_type == 'julian':
        days_in_year = 365.25
    else:
        days_in_year = 360
        print(f'Assuming {days_in_year} given that calendar type is: {calendar_type}')

    if interval_days > 28:  # assume it is monthly data
        days_in_year_inferred = interval_days * 12
        if days_in_year != days_in_year_inferred:
            print(f'Careful.. there seems to be some inconsistency in the calendar information and the number of days in a year.. ')
            print(f'The days_in_year_inferred is now used {days_in_year_inferred} rather than the calendar info of {days_in_year}')
            days_in_year = days_in_year_inferred

    window_size = int(period_years * days_in_year / interval_days)
    window_size = min(len(da["time"]), window_size)

    y = da.values
    smoothed_results = {}

    # Method 2: LOESS (Locally Estimated Scatterplot Smoothing)
    frac = window_size / len(y)
    loess_smoothed = lowess(y, np.arange(len(y)), frac=frac, return_sorted=False)

    # Calculate how much padding was added by LOESS
    padding = (len(loess_smoothed) - len(da.time)) // 2

    # Truncate both sides equally if padding exists
    if padding > 0:
        loess_smoothed = loess_smoothed[padding:-padding]

    # Assign the truncated result to the smoothed_results dictionary
    smoothed_results = xr.DataArray(loess_smoothed, dims=da.dims, coords=da.coords, name='loess_smoothed')
    

    if plotoutput:
        plt.figure(figsize=(12, 6))
        plt.plot(da.time, y, label='Original', color='gray', alpha=0.7)
        plt.plot(da.time, smoothed_results, label='LOESS', linestyle='-.')
        plt.title(f'Smoothing Results for {da.name}')
        plt.xlabel('Time')
        plt.ylabel(da.name)
        plt.legend()
        plt.show()

    return smoothed_results

# %%
if False: 
    def generate_synthetic_temperature_data():
        """
        Generate synthetic monthly temperature data with an underlying trend and seasonality.

        Returns:
        - da: xarray.DataArray containing the synthetic temperature data with a 'time' dimension.
        """
        # Set a seed for reproducibility
        np.random.seed(42)

        lengthyears = 50

        # Generate a time range for 10 years of monthly data
        time = pd.date_range(start='2000-01-01', periods=lengthyears*12, freq='MS')

        # Generate an underlying temperature trend with three segments
        trend = np.concatenate([
            1 * np.arange(int(lengthyears*12/3)),        # First third: mild upward trend
            0.06 * np.arange(int(lengthyears*12/3))+40,        # Second third: steeper upward trend
            -2 * np.arange(lengthyears*12-2*int(lengthyears*12/3)) +(40+0.06*40)   # Final third: strong downward trend
        ])    


            # Generate a seasonal component (e.g., annual temperature cycle)
        seasonality = lengthyears * np.sin(2 * np.pi * (time.month - 1) / 12)

        # Combine trend and seasonality and add some noise
        temperature = trend + seasonality + np.random.normal(0, 100, lengthyears*12)

        # Create an xarray DataArray
        da = xr.DataArray(temperature, dims='time', coords={'time': time}, name='temperature')

        return da

    def test_lowpass_filter_timeseries():
        """
        Test the lowpass_filter_timeseries function using synthetic temperature data.
        """
        # Generate synthetic data
        da = generate_synthetic_temperature_data()

        # Apply the lowpass filter with a 2-year smoothing period
        smoothed_da = lowpass_filter_timeseries(da, period_years=21, plotoutput=True)

    # Call the test function
    test_lowpass_filter_timeseries()



# %% [markdown]
# ## Create new variables alas combine some variables.. (add/subtract)


# %%
def process_combine_files(files2crunch_df, target_resolution, regridded_data_handles, use_parallel=False, crunch_observations_instead=False, lon_convention: str = "180"):
    
    # Initialize combine_vars to None
    combine_vars = None

    # Check for the availability and validity (non-NaN, non-zero) of variables in files2crunch_df
    def check_variable_availability(variable_name, row):
        return variable_name in row and not pd.isna(row[variable_name]) and row[variable_name] != 0

    # Iterate through the rows to set the appropriate combine_vars
    for _, row in files2crunch_df.iterrows():
        rsdt_available = check_variable_availability('rsdt', row)
        rlut_available = check_variable_availability('rlut', row)
        rsut_available = check_variable_availability('rsut', row)
        rsnt_available = check_variable_availability('rsnt', row)

        if rsdt_available and rlut_available and rsut_available:
            if crunch_observations_instead:
                combine_vars = {'rtmt': (['rsdt'], ['rlut', 'rsut'])}
            else:
                combine_vars = {'rtmt': (['rsdt'], ['rlut', 'rsut']),
                                'rtmt_parent': (['rsdt_parent'], ['rlut_parent', 'rsut_parent'])}
            break
        elif rlut_available and rsnt_available:
            if crunch_observations_instead:
                combine_vars = {'rtmt': (['rsnt', 'rlut'], [])} 
                print(f'ERA5 adds rsnt and rlut together in order to get rtmt')
                # for ERA5, both the tsr (total solar radiation = rsnt) and the ttr (total thermal radiation = rlut) 
                # are already in the same convention, i.e. positive = downwards.. hence, they just need to be added up. 
            else:
                combine_vars = {'rtmt': (['rsnt'], ['rlut']),
                                'rtmt_parent': (['rsnt_parent'], ['rlut_parent'])}
            break
        else:
            print(f"Required variables are missing or invalid (NaN/zero) in the files.")
            return regridded_data_handles  # Exit function early if no valid combination is found

    # Check if necessary filenames are available
    for var, (sum_vars, subtract_vars) in combine_vars.items():
        for v in sum_vars + subtract_vars:
            if not all(v in row for _, row in files2crunch_df.iterrows()):
                print(f"Missing required files for {var}: {v} not available in all rows.")
                return regridded_data_handles  # Exit function early

    if use_parallel:
        # Using Parallel with delayed, iterating over rows using iterrows()
        results = Parallel(n_jobs=2)(
            delayed(process_specific_combination)(
                files2crunch_df,
                regridded_data_handles,
                combine_vars,
                target_resolution,
                source_id,
                experiment_id,
                member_id,
                lon_convention=lon_convention,
            ) for (source_id, experiment_id, member_id), _ in files2crunch_df.iterrows()
        )

        # Combine results from all parallel tasks
        for result in results:
            regridded_data_handles.update(result)
    else:
        for (source_id, experiment_id, member_id), row in files2crunch_df.iterrows():
            print(f"--------------------------------------------------------")
            print(f"Now Processing {source_id}, {experiment_id}, {member_id}....")
            regridded_data_handles = process_specific_combination(
                files2crunch_df,
                regridded_data_handles,
                combine_vars,
                target_resolution,
                source_id,
                experiment_id,
                member_id,
                lon_convention=lon_convention,
            )
    
    return regridded_data_handles



# %%

def combine_datasets(
    summing_list,
    subtracting_list,
    resolution,
    new_var_name='rtmt',
    reset_time_coord=False,
    lon_convention: str = "180",
):
    """
    Combine datasets from zstore links by summing and subtracting after regridding to a common resolution.
    
    Parameters
    ----------
    summing_list : list of str
        List of zstore links to datasets to be added.
    subtracting_list : list of str
        List of zstore links to datasets to be subtracted.
    resolution : float
        The target grid resolution to which all datasets will be regridded.
    new_var_name : str
        New variable name for the combined dataset.
    reset_time_coord : bool, optional
        Flag to reset the time coordinate, default is False.
    
    Returns
    -------
    xarray.Dataset or None
        The resulting dataset after regridding, summing, and subtracting, or None if an error occurs.
    """

    def regrid_and_rename(zstore, resolution, new_name=None, reset_time_coord=False):
        ds = regrid_dataset_if_needed(zstore, resolution=resolution, lon_convention=lon_convention)
        if ds is None:
            logging.error(f"Error in regridding dataset from {zstore}")
            return None

        ds = convert_to_target_cftime(ds, zstore, reset_time_coord=reset_time_coord, create_checkplot=False)
        if ds is None:
            logging.error(f"Error in converting time for dataset from {zstore}")
            return None

        if new_name:
            ds = ds.rename({var: new_name for var in ds.data_vars})
        return ds

    # Regrid all datasets
    regridded_summing_datasets = [regrid_and_rename(zstore, resolution, new_var_name, reset_time_coord) for zstore in summing_list]
    regridded_subtracting_datasets = [regrid_and_rename(zstore, resolution, new_var_name, reset_time_coord) for zstore in subtracting_list]

    # Check if any regridded dataset is None
    if any(ds is None for ds in regridded_summing_datasets + regridded_subtracting_datasets):
        logging.error("Error in processing datasets. Aborting combine_datasets.")
        return None

    # Verify that all datasets have the same units
    units = set(ds.attrs.get('units') for ds in regridded_summing_datasets + regridded_subtracting_datasets)
    if len(units) > 1:
        logging.error("Not all datasets have the same units.")
        return None

    # Initialize the combined dataset
    combined_dataset = regridded_summing_datasets[0]

    # Sum the datasets
    for ds in regridded_summing_datasets[1:]:
        combined_dataset += ds

    # Subtract the datasets
    for ds in regridded_subtracting_datasets:
        combined_dataset -= ds

    # Update global attributes
    combined_dataset.attrs['history'] = 'Combined datasets: summed [{}] and subtracted [{}]'.format(
        ', '.join(summing_list),
        ', '.join(subtracting_list))
    combined_dataset.attrs['new_variable_name'] = new_var_name

    return combined_dataset



# %%
def process_specific_combination(
    files2crunch_df,
    regridded_data_handles,
    combine_vars,
    target_resolution,
    source_id,
    experiment_id,
    member_id,
    new_scenarios=['abrupt-4xCO2', 'abrupt-2xCO2', 'abrupt-0p5xCO2', '1pctCO2'],
    lon_convention: str = "180",
):
    """
    Process a specific combination in files2crunch_df to combine datasets as specified in combine_vars.

    Parameters
    ----------
    files2crunch_df : pandas.DataFrame
        DataFrame containing file links for each variable and combination.
    regridded_data_handles : dict
        Dictionary containing handles to regridded datasets.
    combine_vars : dict
        Dictionary specifying how to combine variables to create new variables.
    target_resolution : int
        The target resolution for the combined datasets.
    source_id : str
        Source ID to process.
    experiment_id : str
        Experiment ID to process.
    member_id : str
        Member ID to process.

    Returns
    -------
    None
        Updates regridded_data_handles in-place with combined datasets.
    """
    # Check if the experiment_id is in the new scenarios list
    reset_time_coord = experiment_id in new_scenarios

    # Filter the DataFrame for the specified combination
    if (source_id, experiment_id, member_id) in files2crunch_df.index:
        row = files2crunch_df.loc[(source_id, experiment_id, member_id)]

        for newvar, (sum_vars, subtract_vars) in combine_vars.items():
            newvartrunk = newvar.replace('_parentandchild', '').replace('_parent', '')
            summing_list = []
            subtracting_list = []

            # Validate and collect Zarr store links
            for var in sum_vars + subtract_vars:
                zstore_link = row[var]
                if pd.notna(zstore_link):
                    list_to_add_to = summing_list if var in sum_vars else subtracting_list
                    list_to_add_to.append(zstore_link)
                else:
                    print(f".")
                    break

            # Only proceed if all necessary links are valid
            if len(summing_list) == len(sum_vars) and len(subtracting_list) == len(subtract_vars):
                xr_out = combine_datasets(
                    summing_list,
                    subtracting_list,
                    target_resolution,
                    new_var_name=newvartrunk,
                    reset_time_coord=reset_time_coord,
                    lon_convention=lon_convention,
                )
                if xr_out is None: 
                    continue
                
                regridded_data_handles[(source_id, experiment_id, member_id, newvar)] = xr_out
                print(f"Successfully created {newvartrunk} / {newvar} for {source_id}, {experiment_id}, {member_id}")
            else:
                print(f".")

    return regridded_data_handles

# Example usage
# regridded_data_handles = process_specific_combination(files2crunch_df, regridded_data_handles, combine_vars, target_resolution, 'HadGEM3-GC31-MM', 'ssp585', 'r3i1p1f3')



# %% [markdown]
# ## Stitching together of child and parent scenario

# %%
def stitch_function(parent_ds, child_ds):
    """
    Stitches two xarray datasets along the time dimension.

    Parameters
    ----------
    parent_ds : xarray.Dataset
        The parent dataset.
    child_ds : xarray.Dataset
        The child dataset.

    Returns
    -------
    xarray.Dataset
        A new dataset with parent and child datasets concatenated along the time dimension.
    """

    # Check for None datasets
    if parent_ds is None or child_ds is None:
        logging.error("One of the datasets is None.")
        return None

    try:
        # Check if the time coordinates are contiguous
        last_parent_time = parent_ds.time.max().values
        first_child_time = child_ds.time.min().values

        if last_parent_time >= first_child_time:
            logging.error(f"The last_parent_time {last_parent_time} is not earlier than the first_child_time {first_child_time}.")
            return None

        # Concatenate along the time dimension
        stitched_ds = xr.concat([parent_ds, child_ds], dim='time')
        return stitched_ds

    except Exception as e:
        logging.error(f"Error in stitching: {str(e)}")
        return None

# Example usage:
# stitched_ds = stitch_function(parent_ds, child_ds)



# %%
def stitch_parent_child_datasets(regridded_data_handles):
    """
    Stitches together parent and child datasets based on matching variables in the regridded_data_handles.

    Parameters:
    - regridded_data_handles (dict): A dictionary where keys are tuples (source_id, experiment_id, member_id, variable) 
      and values are xarray DataArrays or Datasets. The dictionary should contain both 'parent' and regular versions 
      of variables for stitching.

    The function iterates through each unique combination of source_id, experiment_id, and member_id found in the 
    keys of regridded_data_handles. For each combination, it identifies variables that have corresponding 'parent' 
    variables and performs stitching using a predefined stitch_function.

    The stitched datasets are then added back into regridded_data_handles with the variable name suffixed by 
    '_parentandchild'.

    Outputs:
    - The function returns the updated regridded_data_handles dictionary containing the original and the newly 
      stitched datasets.

    Notes:
    - The function prints progress messages throughout its execution, including the start of the stitching process, 
      the combinations being processed, and the completion of the stitching process.
    - It assumes the existence of a stitch_function that takes two datasets (parent and child) and returns a 
      stitched dataset.
    """
    
    print("Starting stitching process...")

    # Identify unique combinations of source_id, experiment_id, member_id
    unique_combinations = set((source_id, experiment_id, member_id) for source_id, experiment_id, member_id, _ in regridded_data_handles.keys())
    #print(f"Unique combinations found: {unique_combinations}")

    # Iterate through each unique combination
    for source_id, experiment_id, member_id in unique_combinations:
        #print(f"Processing combination: {source_id}, {experiment_id}, {member_id}")
        print(f"--------------------------------------------------------")
        print(f"Now Processing {source_id}, {experiment_id}, {member_id}....")

        # Extract all variables for this combination
        variables = set()
        for key in regridded_data_handles.keys():
            if key[:3] == (source_id, experiment_id, member_id):
                variables.add(key[3])
        print(f"Variables for this combination: {variables}")

        # Find variables that have a corresponding parent
        for var in variables:
            parent_var = f'{var}_parent'
            if parent_var in variables:
                child_ds = regridded_data_handles.get((source_id, experiment_id, member_id, var))
                parent_ds = regridded_data_handles.get((source_id, experiment_id, member_id, parent_var))

                try:
                    stitched_ds = stitch_function(parent_ds, child_ds)

                    if stitched_ds is None: 
                        logging.error(f"Variable {var} for {source_id}, {experiment_id}, {member_id} cannot be stitched.")
                        continue

                    stitched_var_name = f'{var}_parentandchild'
                    regridded_data_handles[(source_id, experiment_id, member_id, stitched_var_name)] = stitched_ds
                    print(f"Stitched dataset added for {stitched_var_name}")

                except Exception as e:
                    logging.error(f"Error in stitching {var} for {source_id}, {experiment_id}, {member_id}: {str(e)}")
                    continue
                
            else: 
                print(f".")

    print("Stitching process completed.")

    return regridded_data_handles


# %%

# %% [markdown]
# ## Radiative forcing operations: Add CO2 and retrieve scenario specific forcings. 

# %%
def add_co2_scenarios(df, pre_industrial_co2=278, RF2xCO2=3.71):
    new_scenarios = {
        'abrupt-4xCO2': 4,
        'abrupt-2xCO2': 2,
        'abrupt-0p5xCO2': 0.5,
        '1pctCO2': None  # Special case
    }

    a = RF2xCO2 / np.log(2)  # Calculate the alpha parameter a
    
    if 'Scenario' not in df.index.names:
        raise ValueError("The MultiIndex of the DataFrame does not contain 'Scenario' as a level.")

    for scenario, multiplier in new_scenarios.items():
        print(f'Now creating scenario {scenario}')
        try:
            historical_scenario = df.xs('historical', level='Scenario', drop_level=False)
        except KeyError:
            raise KeyError("'historical' scenario not found in the DataFrame.")

        new_scenario_df = historical_scenario.copy()

        # Create new index for the new scenario
        new_index = []
        for idx in new_scenario_df.index:
            new_idx_list = list(idx)
            scenario_idx = df.index.names.index('Scenario')
            new_idx_list[scenario_idx] = scenario
            new_index.append(tuple(new_idx_list))

        new_scenario_df.index = pd.MultiIndex.from_tuples(new_index, names=df.index.names)

        # Set radiative forcing to pre-industrial value for all rows except CO2
        pre_industrial_rf = historical_scenario.loc[:, '1850']
        for year in df.columns:
            new_scenario_df.loc[:, year] = pre_industrial_rf

        co2_rf_index = new_scenario_df.index.get_level_values('Variable').isin(['Radiative Forcing|Anthropogenic|CO2', 'Effective Radiative Forcing|Anthropogenic|CO2'])
 
        # print(co2_rf_index)
        # Update CO2 radiative forcing
        for year in df.columns:
            if year > '1850':
                if multiplier:  # Abrupt scenarios
                    ct = pre_industrial_co2 * multiplier
                else:  # 1pctCO2 scenario
                    i = int(year) - 1850
                    ct = pre_industrial_co2 * (1.01 ** i)
                rt = a * np.log(ct / pre_industrial_co2)       
                new_scenario_df.loc[co2_rf_index, year] = rt


        # Add CO2 radiative forcing to GHG and Total for both Radiative Forcing and Effective Radiative Forcing
        for var_base in ['Radiative Forcing|Anthropogenic', 'Effective Radiative Forcing|Anthropogenic']:
            co2_var = f'{var_base}|CO2'
            ghg_var = f'{var_base}|GHG'
            total_var = var_base

            # print(new_scenario_df.index.get_level_values('Variable'))
            if co2_var in new_scenario_df.index.get_level_values('Variable'):
                for year in new_scenario_df.columns:
                    co2_rf = new_scenario_df.xs(co2_var, level='Variable', drop_level=False).loc[:, year]
                    
                    if ghg_var in new_scenario_df.index.get_level_values('Variable'):
                        new_scenario_df.loc[(slice(None), scenario, slice(None), ghg_var), year] += co2_rf
                        
                    if total_var in new_scenario_df.index.get_level_values('Variable'):
                        new_scenario_df.loc[(slice(None), scenario, slice(None), total_var), year] += co2_rf
                    
                    
        # Append new scenario to original DataFrame
        df = pd.concat([df, new_scenario_df])


    return df


# %%
# # figure out how to treat these scenarios: 

# # new subfuntion to ammend ExternalForcingFile_fullpath
#  'abrupt-4xCO2', --> either create synthetically or insert into the RCMIP radforcing data structure.. 
#  'abrupt-2xCO2', --> either create synthetically or insert into the RCMIP radforcing data structure.. 
#  'abrupt-0p5xCO2', --> either create synthetically or insert into the RCMIP radforcing data structure.. 
#  '1pctCO2', --> either create synthetically or insert into the RCMIP radforcing data structure.. 

# # alterations of 'get_radforcing_for_scen'
#  'hist-piAer', --> make special case of keeping aer etc. constant at preind values. 
#  'hist-piNTCF', --> make special case of keeping CH4?, aer etc. constant at preind values. 
#  'ssp370-lowNTCF' --> change scenario name to External forcing file 'aerchemmip'


# %%
import difflib
from pathlib import Path
from typing import Dict, List

import pandas as pd
import xarray as xr



import difflib
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr
import cftime


# -----------------------------------------------------------------------------#
#  helpers                                                                     #
# -----------------------------------------------------------------------------#
def _months_between(a: cftime.datetime, b: cftime.datetime) -> int:
    """
    Signed distance **a → b** in whole 360-day-calendar months
    (positive = *after*).
    """
    return (b.year - a.year) * 12 + (b.month - a.month)


MAX_EXTRAP_MONTHS = 24      # ↑ allowed nearest-neighbour padding (±2 years)


# -----------------------------------------------------------------------------#
#  main routine                                                                #
# -----------------------------------------------------------------------------#
def get_radforcing_for_scen(
    scen: str,
    time_coord: xr.DataArray,
    df: pd.DataFrame,
    components: Dict[str, List[str] | str],
    pi_components: Dict[str, List[str]],
    ExternalForcingFile_fullpath: str | Path,
    scenarioaliases: Dict[str, List[str]] | None = None,
) -> xr.Dataset:
    """
    Return radiative-forcing time series for *scen* on **monthly**
    (360-day) *time_coord*.

    Annual values in *df* are assumed to represent the year **mid-point
    (1 July)** and are carried forward/backward with nearest-neighbour
    interpolation, but never beyond **±12 months** – a longer gap raises
    ``ValueError``.
    """
    scenarioaliases = scenarioaliases or {}
    pre_industrial_year = "1850"

    # ------------------------------------------------------------------ 0) suffix logic
    special_component = None
    special_pi_component = None
    if "-" in scen:
        pre, post = scen.split("-", 1)
        if post in components:
            special_component = post
            scen2readfrom = pre
        elif post in pi_components:
            special_pi_component = post
            scen2readfrom = pre
        else:
            scen2readfrom = scen
    else:
        scen2readfrom = scen

    # ------------------------------------------------------------------ 1) alias resolution
    scenario_to_use = scen2readfrom
    scen_index = df.index.get_level_values("Scenario")
    if scen2readfrom not in scen_index:
        for alias in scenarioaliases.get(scen2readfrom, []):
            if alias in scen_index:
                scenario_to_use = alias
                print(
                    f"Using alias '{scenario_to_use}' instead of "
                    f"'{scen2readfrom}' for radiative-forcing data."
                )
                break
        else:
            raise ValueError(
                f"Scenario '{scen2readfrom}' (from '{scen}') not found in "
                "DataFrame index and no valid alias supplied."
            )

    # ------------------------------------------------------------------ 2) (optional) hist→ssp245 stitch
    max_year_needed = int(time_coord.dt.year.max())
    base_lower = scenario_to_use.lower()
    if base_lower in {"hist", "historical"} and max_year_needed >= 2016:
        future_scen = "ssp245"
        if future_scen not in scen_index:
            raise ValueError(
                "Rule requires scenario 'ssp245' but it is absent "
                "from the input DataFrame."
            )

        df_hist = df.xs(scenario_to_use, level="Scenario")
        df_fut = df.xs(future_scen, level="Scenario")

        hist_cols = [c for c in df_hist.columns if int(c) <= 2014]
        fut_cols = [c for c in df_fut.columns if int(c) >= 2015]

        df_scen = pd.concat(
            [df_hist.reindex(columns=hist_cols),
             df_fut.reindex(columns=fut_cols)],
            axis=1,
        ).reindex(sorted(df_hist.columns.union(df_fut.columns),
                         key=int), axis=1)
    else:
        df_scen = df.xs(scenario_to_use, level="Scenario")

    # ------------------------------------------------------------------ 3) bookkeeping containers
    dataarrays: Dict[str, xr.DataArray] = {}
    component_details = {}
    warnings_subcomponents = {}
    pre_industrial_fixing = {}
    missing_components = {}

    # survivors when “…-GHG” or “…-nat” suffix is used
    _allowed: Dict[str, set[str]] = {
        "GHG": {"GHG", "CO2"},
        "nat": {"nat", "sol", "volc"},
    }

    # ------------------------------------------------------------------ 4) iterate components
    force_years = [int(y) for y in df_scen.columns]
    force_time = [cftime.Datetime360Day(y, 7, 1) for y in force_years]

    for comp_name, variables in components.items():
        comp_data = pd.Series(index=df_scen.columns, dtype="float64")
        missing_subcomponents: List[str] = []

        for var in (variables if isinstance(variables, list) else [variables]):
            if var in df_scen.index.get_level_values("Variable"):
                if (special_pi_component is not None and
                        var in pi_components[special_pi_component]):
                    # fix at pre-industrial level
                    pi_val = (
                        df_scen.xs(var, level="Variable")[pre_industrial_year]
                        .values[0]
                    )
                    comp_data += pi_val
                    pre_industrial_fixing[var] = (
                        f"Fixed {var} to PI {pre_industrial_year} value {pi_val}"
                    )
                else:
                    comp_data = comp_data.add(
                        df_scen.xs(var, level="Variable").sum(axis=0),
                        fill_value=0,
                    )
            else:
                missing_subcomponents.append(var)
                suggestions = difflib.get_close_matches(
                    var,
                    df_scen.index.get_level_values("Variable").unique(),
                    n=3,
                )
                print(
                    f"Warning: variable '{var}' not found for scenario '{scen}'."
                    f"\n  Suggestions: {suggestions}"
                )

        # suffix logic: freeze unwanted components
        if special_component is not None:
            survivors = _allowed.get(special_component, {special_component})
            if comp_name not in survivors:
                pi_val = comp_data[pre_industrial_year]
                print(f"------- fixing '{comp_name}' to PI {pre_industrial_year} "
                      f"value {pi_val}")
                comp_data[:] = pi_val
                pre_industrial_fixing[comp_name] = (
                    f"Fixed {comp_name} to PI {pre_industrial_year} value {pi_val}"
                )

        # convert to DataArray (annual mid-year stamps)
        dataarrays[comp_name] = xr.DataArray(
            comp_data.values,
            dims=("time",),
            coords={"time": force_time},
            name=comp_name,
        )
        component_details[comp_name] = variables
        missing_components[comp_name] = missing_subcomponents
        if missing_subcomponents:
            warnings_subcomponents[comp_name] = {
                "Missing data for": ", ".join(missing_subcomponents)
            }

    # ------------------------------------------------------------------ 5) interpolate / extrapolate safely
    target_times = time_coord.values
    t_first_req = target_times[0]
    t_last_req  = target_times[-1]

    for k, da in dataarrays.items():
        too_early = _months_between(t_first_req, da.time.values[0]) > MAX_EXTRAP_MONTHS
        too_late  = _months_between(da.time.values[-1], t_last_req)  >  MAX_EXTRAP_MONTHS
        if too_early or too_late:
            side = "start" if too_early else "end"
            
            raise ValueError(
                f"{scen}: monthly request at the {side} requires more than "
                f"±{MAX_EXTRAP_MONTHS} months extrapolation.... t_first_req {t_first_req} | t_last_re {t_last_req} | da.time.values[0] {da.time.values[0]} | da.time.values[-1] {da.time.values[-1]} | _months_between(t_first_req, da.time.values[0]) {_months_between(t_first_req, da.time.values[0])} |  _months_between(da.time.values[-1], t_last_req) {_months_between(da.time.values[-1], t_last_req)} "
            )

        # nearest-neighbour (step-wise) fill
        dataarrays[k] = da.sel(time=target_times, method="nearest")

    # ------------------------------------------------------------------ 6) assemble & annotate
    radforce_xr = xr.Dataset(dataarrays)
    radforce_xr.attrs.update(
        {
            "source_data": str(ExternalForcingFile_fullpath),
            "component_status": component_details,
            "missing_components": missing_components,
            "pre_industrial_fixing": pre_industrial_fixing,
            "warnings": warnings_subcomponents,
            "note":
                "Annual RF values assigned to 1 July; monthly series filled "
                f"with nearest neighbour (≤ {MAX_EXTRAP_MONTHS} months).",
        }
    )
    return radforce_xr



def get_radforcing_for_scen_oldJune2025(
    scen: str,
    time_coord: xr.DataArray,
    df: pd.DataFrame,
    components: Dict[str, List[str] | str],
    pi_components: Dict[str, List[str]],
    ExternalForcingFile_fullpath: str | Path,
    scenarioaliases: Dict[str, List[str]] | None = None,
) -> xr.Dataset:
    """
    Build an xarray.Dataset with radiative-forcing time series for *scen*.

    Parameters
    ----------
    scen
        Scenario name, e.g. ``"ssp119"``, or special forms like
        ``"hist-GHG"`` / ``"hist-nat"``.  
        • “<base>-GHG”  → only *GHG & CO2* vary, others fixed at PI.  
        • “<base>-nat”  → only *nat, sol, volc* vary, others fixed at PI.  
        • Any other “…-suffix”  → only *suffix* varies.
    time_coord
        Time coordinate whose (year) values are used for interpolation.
    df
        Radiative-forcing table indexed by ``['Scenario', 'Variable']`` with
        yearly columns (strings or ints).
    components
        Mapping *component → variable(s)* to sum.
    pi_components
        Mapping *suffix → list of variables* that may be PI-fixed.
    ExternalForcingFile_fullpath
        Path recorded as provenance in the output attributes.
    scenarioaliases
        Optional aliases for scenarios (dict *base → [alias1, …]*).

    “Historical → SSP245” stitching
    --------------------------------
    If *base* is **“hist”** or **“historical”** **and** any requested year
    ≥ 2016, the function returns a composite series that uses *historical*
    data up to 2015 **and** *ssp245* from 2016 onward.  All subsequent
    suffix / PI-fix logic operates on that stitched dataset.

    Returns
    -------
    xr.Dataset  –  one variable per forcing component.
    """
    scenarioaliases = scenarioaliases or {}
    pre_industrial_year = "1850"

    # ------------------------------------------------------------------
    # 0) split possible suffix  (e.g.  hist-GHG  →  pre='hist', post='GHG')
    # ------------------------------------------------------------------
    special_component = None
    special_pi_component = None

    if "-" in scen:
        pre, post = scen.split("-", 1)
        if post in components:
            special_component = post
            scen2readfrom = pre
        elif post in pi_components:
            special_pi_component = post
            scen2readfrom = pre
        else:
            scen2readfrom = scen
    else:
        scen2readfrom = scen

    # ------------------------------------------------------------------
    # 1) resolve aliases  (ssp245 for instance often aliased as 'SSP2-4.5')
    # ------------------------------------------------------------------
    scenario_to_use = scen2readfrom
    if scen2readfrom not in df.index.get_level_values("Scenario"):
        for alias in scenarioaliases.get(scen2readfrom, []):
            if alias in df.index.get_level_values("Scenario"):
                scenario_to_use = alias
                print(
                    f"Using alias '{scenario_to_use}' instead of "
                    f"'{scen2readfrom}' for radiative-forcing data."
                )
                break
        else:
            raise ValueError(
                f"Scenario '{scen2readfrom}' (from '{scen}') not found in the "
                "DataFrame index and no valid alias supplied."
            )

    # ------------------------------------------------------------------
    # 2) optional *historical → ssp245* stitching
    # ------------------------------------------------------------------
    max_year_needed = int(time_coord.dt.year.max())
    base_lower = scenario_to_use.lower()

    if base_lower in {"hist", "historical"} and max_year_needed >= 2016:
        future_scen = "ssp245"

        if future_scen not in df.index.get_level_values("Scenario"):
            raise ValueError(
                "Rule requires scenario 'ssp245' but it is absent "
                "from the input DataFrame."
            )

        df_hist = df.xs(scenario_to_use, level="Scenario")
        df_fut = df.xs(future_scen, level="Scenario")

        # split columns by year
        hist_cols = [c for c in df_hist.columns if int(c) <= 2014]
        fut_cols = [c for c in df_fut.columns if int(c) >= 2015]

        df_scen = pd.concat(
            [
                df_hist.reindex(columns=hist_cols),
                df_fut.reindex(columns=fut_cols),
            ],
            axis=1,
        )

        # ensure chronological order
        df_scen = df_scen.reindex(sorted(df_scen.columns, key=int), axis=1)

    else:
        df_scen = df.xs(scenario_to_use, level="Scenario")

    # ------------------------------------------------------------------
    # 3) bookkeeping containers
    # ------------------------------------------------------------------
    dataarrays = {}
    component_details = {}
    warnings_subcomponents = {}
    pre_industrial_fixing = {}
    missing_components = {}

    # components that *survive* when a suffix is used
    _allowed: Dict[str, set[str]] = {
        "GHG": {"GHG", "CO2"},
        "nat": {"nat", "sol", "volc"},
    }

    # ------------------------------------------------------------------
    # 4) iterate through each forcing component
    # ------------------------------------------------------------------
    for comp_name, variables in components.items():
        comp_data = pd.Series(index=df_scen.columns, dtype="float64")
        missing_subcomponents: List[str] = []

        # ---- sum (or PI-fix) sub-variables
        for var in (variables if isinstance(variables, list) else [variables]):
            if var in df_scen.index.get_level_values("Variable"):

                if (
                    special_pi_component is not None
                    and var in pi_components[special_pi_component]
                ):
                    # keep variable fixed at its PI value
                    pi_val = (
                        df_scen.xs(var, level="Variable")[pre_industrial_year]
                        .values[0]
                    )
                    comp_data += pi_val
                    pre_industrial_fixing[var] = (
                        f"Fixed {var} to PI {pre_industrial_year} value {pi_val}"
                    )
                else:
                    comp_data = comp_data.add(
                        df_scen.xs(var, level="Variable").sum(axis=0),
                        fill_value=0,
                    )
            else:
                missing_subcomponents.append(var)
                suggestions = difflib.get_close_matches(
                    var,
                    df_scen.index.get_level_values("Variable").unique(),
                    n=3,
                )
                print(
                    f"Warning: variable '{var}' not found for scenario '{scen}'."
                    f"\n  Suggestions: {suggestions}"
                )

        # ---- freeze whole component if suffix demands it
        if special_component is not None:
            survivors = _allowed.get(special_component, {special_component})
            if comp_name not in survivors:
                pi_val = comp_data[pre_industrial_year]
                print(
                    f"------- fixing '{comp_name}' to PI {pre_industrial_year} "
                    f"value {pi_val}"
                )
                comp_data[:] = pi_val
                pre_industrial_fixing[comp_name] = (
                    f"Fixed {comp_name} to PI {pre_industrial_year} value {pi_val}"
                )

        # ---- convert to xarray & book-keep
        component_details[comp_name] = variables
        missing_components[comp_name] = missing_subcomponents

        dataarrays[comp_name] = xr.DataArray(
            comp_data.values,
            dims=["time"],
            coords={"time": pd.to_numeric(df_scen.columns)},
        )

        if missing_subcomponents:
            warnings_subcomponents[comp_name] = {
                "Missing data for": ", ".join(missing_subcomponents)
            }

    # ------------------------------------------------------------------
    # 5) interpolate, assemble & annotate
    # ------------------------------------------------------------------
    target_years = pd.to_numeric(time_coord.dt.year)
    for k, da in dataarrays.items():
        dataarrays[k] = da.interp(time=target_years)

    radforce_xr = xr.Dataset(dataarrays)
    radforce_xr.attrs.update(
        {
            "source_data": str(ExternalForcingFile_fullpath),
            "component_status": component_details,
            "missing_components": missing_components,
            "pre_industrial_fixing": pre_industrial_fixing,
            "warnings": warnings_subcomponents,
        }
    )

    return radforce_xr



# %%
def get_radforcing_for_scen_old(scen, time_coord, df, components, pi_components, ExternalForcingFile_fullpath,scenarioaliases={}):
    """
    Collates and aggregates radiative forcing for a given climate scenario by aggregating component data.

    Parameters:
    - scen (str): The scenario name to process, which may include a special suffix for component-specific treatment.
    - time_coord (xarray DataArray): Time coordinates for interpolating the data.
    - df (pandas DataFrame): DataFrame containing radiative forcing data, indexed by 'Scenario' and 'Variable'.
    - components (dict): Dictionary mapping component names to their respective variables in 'df'.
    - pi_components (dict): Dictionary specifying components that should be held constant at pre-industrial levels.
    - ExternalForcingFile_fullpath (str): Path to the external forcing data file.

    Returns:
    - radforce_xr (xarray.Dataset): An xarray Dataset containing interpolated radiative forcing data for each component.

    The function processes the specified scenario, handling special cases where the scenario name includes a 
    component-specific suffix. It then aggregates data for each component, taking into account special cases such 
    as components to be held at pre-industrial levels. Missing data and discrepancies are handled and logged. 
    Finally, the data for each component is interpolated to the provided time coordinates and combined into a single 
    xarray Dataset.

    Notes:
    - The function prints messages about special cases and missing data.
    - Raises a ValueError if the provided scenario is not found in the DataFrame's index.
    """
    
    
    dataarrays = {}
    component_details = {}
    warnings_subcomponents = {}
    pre_industrial_fixing = {} 
    missing_components = {} 
    
    # Check if the scenario name matches the special case
    special_component = None
    special_pi_component = None
    
    if '-' in scen:
        pre_hyphen, post_hyphen = scen.split('-', 1)
         
        if post_hyphen in components:
            special_component = post_hyphen
            scen2readfrom = pre_hyphen
            print(f'The provided scenario {scen} is considered to be a {scen2readfrom}, \
            but limited to the forcing component {post_hyphen}')
        elif post_hyphen in pi_components:
            special_pi_component = post_hyphen
            scen2readfrom = pre_hyphen
            print(f'The provided scenario {scen} is considered to be a {scen2readfrom}, \
            but forcing component {post_hyphen} means that {pi_components[post_hyphen]} are held constant')

        else: 
            scen2readfrom = scen
    else: 
        scen2readfrom = scen

    pre_industrial_year = '1850' 
    
    
    print(f'The scenarioaliases are: {scenarioaliases}')
    print(f'The scen is: {scen}')
    print(f'The scen2readfrom is: {scen2readfrom}')
    # Check if the scenario exists or use aliases if provided
    scenario_to_use = scen2readfrom
    if scen2readfrom not in df.index.get_level_values('Scenario'):
        if scen2readfrom in scenarioaliases:
            for alias in scenarioaliases[scen2readfrom]:
                if alias in df.index.get_level_values('Scenario'):
                    scenario_to_use = alias
                    print(f'Now using {scenario_to_use} instead of {scen2readfrom} ({scen}) to retrieve radiative forcing data')
                    break
        else:
            raise ValueError(f"Scenario '{scen2readfrom}' (from '{scen}') not found in DataFrame index and no valid alias provided.")
    else:
        scenario_to_use = scen2readfrom
                             
    # Now use scenario_to_use for the rest of the function
    df_scen = df.xs(scenario_to_use, level='Scenario')

    # Iterate through the components and collect data
    for comp_name, variables in components.items():
        comp_data = pd.Series(index=df.columns, dtype='float64')
        missing_subcomponents = []
        
        
        
        for variable in (variables if isinstance(variables, list) else [variables]):
            if variable in df_scen.index.get_level_values('Variable'):
                if special_pi_component is not None and variable in pi_components[special_pi_component]:
                    # Extract pre-industrial value as a scalar
                    # print(f'column names... {df_scen.columns}')
                    pre_industrial_value = df_scen.xs(variable, level='Variable')[pre_industrial_year].values[0]
                    print(f'------- NOW fixing {variable} to preindustrial {pre_industrial_year} \
                    value of {pre_industrial_value}')
                    # Add scalar value to comp_data
                    comp_data += pre_industrial_value
                    pre_industrial_fixing[variable] = f'Fixed {variable} to preindustrial \
                    {pre_industrial_year} value of {pre_industrial_value}'
                    
                    pre_industrial_fixing[variable] = f'Fixed {variable} to preindustrial {pre_industrial_year} value of {pre_industrial_value}'

                else: 
                    comp_data = comp_data.add(df_scen.xs(variable, level='Variable').sum(axis=0), fill_value=0)
            else:
                missing_subcomponents.append(variable)
                suggestions = difflib.get_close_matches(variable, \
                                                        df_scen.index.get_level_values('Variable').unique(), n=3)
                print(f"Warning: Variable '{variable}' not found for scenario '{scen}'. \n Suggestions: \n \
                {suggestions}")
    
        if special_component is not None and comp_name != special_component:
            # Set other components to pre-industrial level
            pre_industrial_value = comp_data[pre_industrial_year]
            print(f'------- NOW fixing {comp_name} to preindustrial {pre_industrial_year} value of {pre_industrial_value}')
            comp_data[:]=pre_industrial_value
            pre_industrial_fixing[comp_name] = f'Fixed {comp_name} to preindustrial {pre_industrial_year} value of {pre_industrial_value}'

   
        component_details[comp_name] = components[comp_name]
    
        missing_components[comp_name] = missing_subcomponents
        
        #print(f'The {comp_name} comp_data is {comp_data}')
        #print(f'The {comp_name} preindustrial data is {comp_data[pre_industrial_year]}')
       
                
        # Convert to xarray DataArray
        dataarrays[comp_name] = xr.DataArray(
            comp_data.values, 
            dims=['time'], 
            coords={'time': pd.to_numeric(df.columns)}
        )
        
        if missing_subcomponents:
            warnings_subcomponents[comp_name] = {
                'Missing data for': ', '.join(missing_subcomponents)
            }


    # Interpolate the DataArrays to the input time coordinates
    for key, da in dataarrays.items():
        target_time_values = pd.to_numeric(time_coord.dt.year)
        dataarrays[key] = da.interp(time=target_time_values)

    # Combine all DataArrays into one Dataset
    radforce_xr = xr.Dataset(dataarrays)

    # Add global attributes
    radforce_xr.attrs['source_data'] = ExternalForcingFile_fullpath
    radforce_xr.attrs['component_status'] = component_details
    radforce_xr.attrs['missing_components'] = missing_components
    radforce_xr.attrs['pre_industrial_fixing'] = pre_industrial_fixing
    radforce_xr.attrs['warnings'] = warnings_subcomponents

    return radforce_xr

# Usage example:
# radforce_xr = get_radforcing_for_scen('ssp119', time_coord, RadiativeForcing_RCMIP, components, fullpath2load, scenarioaliases)

# %%
def define_radiative_forcing_components():

    components = {
        'GHG': [
            'Effective Radiative Forcing|Anthropogenic|N2O',
            'Effective Radiative Forcing|Anthropogenic|CH4',
            'Effective Radiative Forcing|Anthropogenic|CO2',
            'Effective Radiative Forcing|Anthropogenic|Other|Other WMGHGs'
        ],
        'stratO3': 'Effective Radiative Forcing|Anthropogenic|Stratospheric Ozone',
        'totalO3': [
            'Effective Radiative Forcing|Anthropogenic|Stratospheric Ozone', 
            'Effective Radiative Forcing|Anthropogenic|Tropospheric Ozone'
        ],
        'nat': 'Effective Radiative Forcing|Natural',
        'aer': 'Effective Radiative Forcing|Anthropogenic|Aerosols',
        'sol': 'Effective Radiative Forcing|Natural|Solar',
        'volc': 'Effective Radiative Forcing|Natural|Volcanic',
        'CO2': 'Effective Radiative Forcing|Anthropogenic|CO2',
        'other': ['Effective Radiative Forcing|Anthropogenic|Other|BC on Snow',
                'Effective Radiative Forcing|Anthropogenic|Other|CH4 Oxidation Stratospheric H2O',
                'Effective Radiative Forcing|Anthropogenic|Other|Contrails and Contrail-induced Cirrus',
                'Effective Radiative Forcing|Anthropogenic|Albedo Change']
    }

    # define the components that shall be kept constant if the scenario is called, e.g. piAer
    pi_components = {
        'piNTCF': [
            'Effective Radiative Forcing|Anthropogenic|Stratospheric Ozone', 
            'Effective Radiative Forcing|Anthropogenic|Tropospheric Ozone',
            'Effective Radiative Forcing|Anthropogenic|CH4',
            'Effective Radiative Forcing|Anthropogenic|Aerosols',
        ],
        'piAer': 'Effective Radiative Forcing|Anthropogenic|Aerosols',
    }

    return components, pi_components


# %%
def create_radiativeforcing_handles(df, unique_experiment_ids, components, pi_components, ExternalForcingFile_fullpath,scenarioaliases={}):
    """
    Creates handles for radiative forcing datasets for different scenarios using the "no leap" calendar.

    Parameters:
    - df (pandas DataFrame): DataFrame containing radiative forcing data.
    - unique_experiment_ids (Index or list): List of unique experiment identifiers.
    - components (dict): Dictionary defining the components for radiative forcing calculation.
    - pi_components (dict): Dictionary defining pre-industrial components for special scenarios.
    - ExternalForcingFile_fullpath (str): File path to the external forcing data.

    Returns:
    - radiativeforcing_handles (dict): A dictionary where keys are scenario names and values are xarray Datasets 
      of radiative forcing data for each scenario.
    """
    radiativeforcing_handles = {}

    # Create a time coordinate using cftime from 1850 to 2500
    time_coord = [cftime.Datetime360Day(year, 7, 1) for year in range(1850, 2501)]
    time_coord_xr = xr.DataArray(time_coord, dims=['time'], name='time')

    for scenario in unique_experiment_ids:
        radforce_xr = get_radforcing_for_scen(scenario, time_coord_xr, df, components, pi_components, ExternalForcingFile_fullpath,scenarioaliases=scenarioaliases)

        # Correctly assign the cftime time coordinate
        radforce_xr = radforce_xr.assign_coords({"time": time_coord_xr})
        radiativeforcing_handles[scenario] = radforce_xr
        print(f'-------')
        print(radforce_xr)

    return radiativeforcing_handles



# %% [markdown]
# ## combination of the data

# %%

# %%
def add_time_components(data):
    """Add year, month, and day components as variables to the dataset."""
    if 'time' in data.coords:
        time_coord = data['time']

        data['year'] = xr.DataArray(time_coord.dt.year, coords={'time': time_coord}, name='year')
        data['month'] = xr.DataArray(time_coord.dt.month, coords={'time': time_coord}, name='month')
        data['day'] = xr.DataArray(time_coord.dt.day, coords={'time': time_coord}, name='day')

    return data



def consolidate_datasets_new(regridded_handles, globalmean_handles, smoothed_handles, radiativeforcing_handles, filter_for_modelscenmember=None):
    """
    Consolidates datasets from different handles into two xarray datasets: one for 'raw' variables
    and one for '_parentandchild' variables, along with radiative forcing data for the specific scenario.

    Parameters as described in your request...
    """
    # Initialize dictionaries to store data variables
    data_vars_raw = {}
    data_vars_parent_child = {}

    # Define suffixes for global mean and smoothed data
    suffix_global_mean = '_globalmean'
    suffix_smoothed = '_smoothed'
    suffix_forcing = '_ERF'

    # Process each type of handle
    for handle_type, suffix in [(regridded_handles, ''), (globalmean_handles, suffix_global_mean), (smoothed_handles, suffix_smoothed), (radiativeforcing_handles, suffix_forcing)]:
        print(f'Loop 1 - handle type suffix: {suffix}')
        for key, data in handle_type.items():
            
            if not isinstance(key, tuple): 
                key = (key,)
            
            # Check if the key should be filtered out
            print(f'Loop 2b: - key: {key} (while asked to look at {filter_for_modelscenmember})')
            if filter_for_modelscenmember:  
                # Filter based on the first three elements of key and var_name not ending with '_parent'
                #print(f' the length of the key is {len(key)} and the type is {type(key)}')
                if len(key) >= 3:
                    if (key[:3] != filter_for_modelscenmember or key[3].endswith('_parent')):
                        print(f'Skipping {key} to next set as we look for {filter_for_modelscenmember}...')
                        continue
                elif len(key) == 1: # RADIATIVE FORCING
                    if (key[0] != filter_for_modelscenmember[1]) :
                        print(f'Skipping {key[0]} to next ERF set as we look for {filter_for_modelscenmember}...')
                        continue
            print(f'---> CONSOLIDATION with .... suffix is {suffix}')
            print(f' .... suffix is {suffix}')
            is_radiative_forcing = handle_type == radiativeforcing_handles
            print(f' .... and is_radiative_forcing is: {is_radiative_forcing}')
            print(f' .... and the key is: {key}')
            processed_vars = process_variable(data, key, suffix, is_radiative_forcing)
            for varname, data_processed in processed_vars:
                add_variable_to_dict(varname, data_processed, data_vars_raw, data_vars_parent_child, key,is_radiative_forcing)

    # Interpolate all variables to match the time dimension of 'tas' from regridded_handles

    data_vars_raw = interpolate_time_dimension(data_vars_raw, regridded_handles, 'tas',filter_for_modelscenmember)
    dataset_raw = xr.Dataset(data_vars_raw)
    
    if 'tas' in data_vars_parent_child:
        data_vars_parent_child = interpolate_time_dimension(data_vars_parent_child, regridded_handles, 'tas_parentandchild',filter_for_modelscenmember)
        dataset_parent_child = xr.Dataset(data_vars_parent_child)
    else: 
        dataset_parent_child = xr.Dataset({})  # Empty Dataset
    
    # Apply the new function to add year, month, and day
    dataset_raw = add_time_components(dataset_raw)
    # Apply the new function to add year, month, and day
    dataset_parent_child = add_time_components(dataset_parent_child)

     
    return dataset_raw, dataset_parent_child




def process_variable(data, key, suffix, is_radiative_forcing):
    """
    Process a variable from the data handles and apply the appropriate suffix.

    Parameters:
    - data: The data (xarray.DataArray or xarray.Dataset) to process.
    - key: Tuple key representing (source_id, experiment_id, member_id, var_name) or 
           just the experiment_id for radiative forcing.
    - suffix: Suffix to append to the variable name.
    - is_radiative_forcing: Boolean indicating if the variable is a radiative forcing variable.

    Returns:
    - List of tuples with processed variable name and data.
    """
    # print(f' Within process_variable: processing key is {key}')
    # print(f'suffix is {suffix}'
    
     
    # In cmip6cruncher.operations.py, inside process_variable:
    def check_and_remove_coords(data, var_name):
        coords_to_examine = [
            coord for coord in data.coords
            if coord not in ['time', 'lat', 'lon', 'latitude', 'longitude']
        ]
        coords_actually_dropped_attributes = {}
    
        for coord in coords_to_examine:
            if coord == 'expver':
                # If 'expver' is found, assume it's okay to drop it for this processing stage,
                # regardless of whether it's scalar or varies with time.
                print(f"    INFO: Dropping 'expver' coordinate from variable '{var_name}' (if it exists).")
                if coord in data.coords: # Check if it actually exists before trying to drop
                    data = data.drop_vars(coord)
                    coords_actually_dropped_attributes[f'dropped_{coord}'] = "Removed 'expver' coordinate"
                continue # Done with expver
    
            # Existing logic for other non-primary coordinates:
            # Only attempt to treat as scalar if not 'expver'
            if coord in data.coords: # Check if it still exists (might have been dropped if it was also expver, though unlikely)
                coord_values = data[coord].values
                is_scalar_like = False
                unique_value_str = "Multiple values"
    
                if data[coord].ndim == 0 or data[coord].size == 1:
                    is_scalar_like = True
                    unique_value_str = str(coord_values.item() if hasattr(coord_values, 'item') else coord_values)
                elif data[coord].ndim > 0: # Check if multi-dimensional but effectively scalar
                    # Use nan-aware unique check if applicable
                    unique_vals = pd.unique(coord_values.ravel()) # pd.unique handles NaNs correctly as one unique value
                    if len(unique_vals) == 1:
                        is_scalar_like = True
                        unique_value_str = str(unique_vals[0])
                
                if is_scalar_like:
                    coords_actually_dropped_attributes[f'dropped_scalar_{coord}'] = unique_value_str
                    data = data.drop_vars(coord)
                    print(f"    INFO: Removed scalar-like coordinate '{coord}' (value: {unique_value_str}) from variable '{var_name}'.")
                else:
                    # This is where the original error was raised for expver.
                    # For other coordinates, you might still want to error or warn.
                    print(f"    WARNING: Coordinate '{coord}' of variable '{var_name}' has multiple unique values ({pd.Series(coord_values.ravel()).nunique()} unique) and was not removed by check_and_remove_coords.")
                    # If this is an error condition for other variables:
                    # raise ValueError(f"Coordinate '{coord}' of {var_name} has multiple values and cannot be removed.")
        
        if isinstance(data, xr.DataArray):
            data.attrs.update(coords_actually_dropped_attributes)
        # If data could be xr.Dataset, handle attrs update appropriately (though unlikely given data[varname])
                
        return data    

     
    if is_radiative_forcing:
        processed_data = []
        for var_name in data.data_vars:
            data_processed = check_and_remove_coords(data[var_name], var_name)
            processed_data.append((var_name + suffix, data_processed))
            
        return processed_data
    else:
        _, _, _, var_name = key
        varname = var_name.replace('_parentandchild', '') + suffix
        data_processed = check_and_remove_coords(data[varname], var_name) if isinstance(data, xr.Dataset) else data
                
        return [(varname, data_processed)]


def add_variable_to_dict(varname, data, data_vars_raw, data_vars_parent_child, key, is_radiative_forcing):
    """
    Add the processed variable to the appropriate dataset dictionary.

    Parameters:
    - varname: The name of the variable.
    - data: The data (xarray.DataArray or xarray.Dataset) associated with the variable.
    - data_vars_raw: Dictionary for the 'raw' dataset variables.
    - data_vars_parent_child: Dictionary for the 'parent and child' dataset variables.
    - key: Tuple key representing (source_id, experiment_id, member_id, var_name).
    """

    
    # List of experiment_ids to check for zero aer_ERF values
    idealised_experiments = ['abrupt-4xCO2', 'abrupt-2xCO2', 'abrupt-0p5xCO2', '1pctCO2']
    
    if is_radiative_forcing:
        # For radiative forcing data, key is just the experiment_id
        # and radiative forcing should populate both the data_vars_parent_child as well as the data_vars_raw. 
        # Add the variable to both dictionaries
        print(f'Now adding radiative forcing data {varname} to both the parentchild and raw frames as {data}')
        data_vars_parent_child[varname] = data
        data_vars_raw[varname] = data
        
        # Check for non-zero aer_ERF values in idealised scenarios
        if 'aer_ERF' in varname and key[0] in idealised_experiments:
            # Check if any value in the aer_ERF timeseries is non-zero
            if (data != 0).any():
                print(f"Error: Non-zero aer_ERF values found in experiment {experiment_id} for {varname}")
                breakpoint()  # Set a breakpoint for debugging
                # raise ValueError(f"Non-zero aer_ERF values found in experiment {experiment_id} for {varname}")
        else: 
            print(f'Varname {varname} and {key[0]} is not part of idealized experiments... all good')

    else:
        _, _, _, original_var_name = key

        # Check if the variable is a parent/child variable
        is_parent_child_var = original_var_name.endswith('_parentandchild')

        # Add the variable to the appropriate dictionary
        if is_parent_child_var:
            data_vars_parent_child[varname] = data
        else:
            data_vars_raw[varname] = data


# %%
def interpolate_time_dimension(data_vars_dict, regridded_handles, tas_var_name, filter_for_modelscenmember=None):


    # Check if filter_for_modelscenmember is specified and unpack it
    if filter_for_modelscenmember is not None:
        source_id_filter, experiment_id_filter, member_id_filter = filter_for_modelscenmember

        # Find the 'tas' variable's time dimension based on the filter
        tas_key = next((k for k in regridded_handles
                        if k[3] == tas_var_name and k[0] == source_id_filter and
                        k[1] == experiment_id_filter and k[2] == member_id_filter), None)
    else:
        # Find the 'tas' variable's time dimension without filter
        tas_key = next((k for k in regridded_handles if k[3] == tas_var_name), None)

    if tas_key is None:
        raise ValueError(f"'tas' (actually {tas_var_name} in {filter_for_modelscenmember}) variable not found in regridded_handles for interpolation.")
    else:
        print(f'Found tas key is {tas_key}')

    tas_time = regridded_handles[tas_key][tas_var_name.replace('_parentandchild','')]['time']

    # Ensure tas_time is in a uniform cftime format
    tas_time_values = ensure_cftime(tas_time.values)

    updated_data_vars_dict = data_vars_dict.copy()

    for var_name, data in updated_data_vars_dict.items():
        if 'time' in data.dims:
            print(f'Now updating time dimension for {var_name} for {filter_for_modelscenmember}')
            # Convert data['time'] to the same cftime format as tas_time_values
            data_time_values = ensure_cftime(data['time'].values)
            data = data.assign_coords(time=data_time_values)

            # Extend data to cover the full range of tas_time_values
            earliest_data_time = data_time_values[0]
            latest_data_time = data_time_values[-1]

            # Identify times in tas_time_values that are outside data_time_values
            early_times = tas_time_values[tas_time_values < earliest_data_time]
            late_times = tas_time_values[tas_time_values > latest_data_time]

            data_list = [data]

            if len(early_times) > 0:
                # Get the first data value along the time dimension
                first_data = data.isel(time=0)
                # Expand first_data along the time dimension to match the length of early_times
                early_data = first_data.expand_dims(time=early_times)
                # Assign the time coordinate to early_times
                early_data = early_data.assign_coords(time=early_times)
                # Add early_data to data_list at the beginning
                data_list.insert(0, early_data)

            if len(late_times) > 0:
                # Get the last data value along the time dimension
                last_data = data.isel(time=-1)
                # Expand last_data along the time dimension to match the length of late_times
                late_data = last_data.expand_dims(time=late_times)
                # Assign the time coordinate to late_times
                late_data = late_data.assign_coords(time=late_times)
                # Add late_data to data_list at the end
                data_list.append(late_data)

            # Concatenate data_list along the time dimension
            data_extended = xr.concat(data_list, dim='time')

            # Ensure the time dimension is sorted
            data_extended = data_extended.sortby('time')

            # Perform interpolation
            interpolated_data = data_extended.interp(time=tas_time_values, method='nearest')

            updated_data_vars_dict[var_name] = interpolated_data

    return updated_data_vars_dict


# %%
def ensure_cftime(time_array):
    """
    Ensure that the time array is in cftime.Datetime360Day format.

    Parameters:
    - time_array: Array of time values to be converted.

    Returns:
    - np.array: An array of cftime.Datetime360Day objects.
    """
    converted_array = []
    for dt in time_array:
        # Convert all cftime.datetime objects to cftime.Datetime360Day
        if isinstance(dt, cftime.datetime):
            dt = cftime.Datetime360Day(dt.year, dt.month, dt.day)
        elif isinstance(dt, (np.datetime64, pd.Timestamp)):
            # Convert numpy datetime64 or pandas Timestamp to cftime.Datetime360Day
            dt = pd.Timestamp(dt).to_pydatetime()
            dt = cftime.Datetime360Day(dt.year, dt.month, dt.day)
        else:
            raise TypeError(f"Unsupported datetime object type: {type(dt)}")
        converted_array.append(dt)

    return np.array(converted_array)


# %% [markdown]
# ## write Netcdf files.. 
def write_compressed_netcdf(
    output_folder: str,
    ds: xr.Dataset,
    prefix: str,
    *,
    time_chunk: int = 12,
    complevel: int = 4,
) -> str:
    """
    Stream-write *ds* to NetCDF with time-first chunking.
    ─────────────────────────────────────────────────────────────────────
    NEW: every variable that is truly 3-D (time-lat-lon) is explicitly
    stored as **float32** (both in-memory and on disk).
    """
    # ------------------------------------------------ 0. file name
    ignored = ("_globalmean", "_smoothed", "_ERF")
    var_tag = "-".join(sorted(v for v in ds.data_vars
                              if not v.endswith(ignored)))
    outfile = os.path.join(output_folder, f"{prefix}_{var_tag}.nc")

    os.makedirs(output_folder, exist_ok=True)
    if os.path.exists(outfile):
        os.remove(outfile)

    # ------------------------------------------------ 1. rechunk
    ds = ds.chunk({"time": time_chunk})

    # ------------------------------------------------ 2. make sure 3-D vars are float32
    def _force_float32(da: xr.DataArray) -> xr.DataArray:
        is_3d = {"time", "lat", "lon"}.issubset(da.dims)
        if is_3d and da.dtype != np.float32:
            return da.astype(np.float32)
        return da

    ds = ds.map(_force_float32, keep_attrs=True)

    # ------------------------------------------------ 3. per-variable encoding
    encoding: dict[str, dict] = {}
    for v in ds.data_vars:
        var = ds[v]
        chunksizes = tuple(
            time_chunk if d == "time" else s
            for d, s in zip(var.dims, var.shape)
        )
        enc = {
            "zlib": True,
            "complevel": complevel,
            "chunksizes": chunksizes,
        }
        if {"time", "lat", "lon"}.issubset(var.dims):
            enc["dtype"] = "float32"     # <- on-disk data type
        encoding[v] = enc

    # ------------------------------------------------ 4. write
    ds.to_netcdf(outfile, engine="netcdf4", mode="w",
                 encoding=encoding, compute=True)

    print("✅  Dataset written to:", outfile)
    return outfile



# %% 
def write_compressed_netcdf_oldJune25(
    output_folder: str,
    ds: xr.Dataset,
    prefix: str,
    *,
    time_chunk: int = 12,
    complevel: int = 4,
) -> str:
    """
    Stream-write *ds* to a NetCDF file with **time-first chunking** so that
    the writer never materialises the entire Dataset in RAM.

    Parameters
    ----------
    output_folder : str
        Directory in which the file will be created (is auto-made if absent).
    ds : xarray.Dataset
        Dataset to write.  The function does **not** modify *ds* in place.
    prefix : str
        Prefix of the file name; the variable list is appended for clarity.
    time_chunk : int, default ``12``
        Number of time steps per chunk (≈ one year for monthly data).
    complevel : int, default ``4``
        zlib compression level (0–9). 4–5 gives a good speed/size trade-off.

    Returns
    -------
    str
        Absolute path of the written ``.nc`` file.

    Notes
    -----
    * Variables whose names end with ``_globalmean``, ``_smoothed`` or
      ``_ERF`` are **kept** but *not* part of the name tag.
    * Uses the *netcdf4* engine – no Zarr staging, so the writer flushes each
      chunk as soon as it is encoded.
    """
    # ---------- build a descriptive file name --------------------------------
    ignored = ("_globalmean", "_smoothed", "_ERF")
    var_tag = "-".join(sorted(v for v in ds.data_vars
                              if not v.endswith(ignored)))
    outfile = os.path.join(output_folder, f"{prefix}_{var_tag}.nc")

    # ---------- make sure the directory exists --------------------------------
    os.makedirs(output_folder, exist_ok=True)
    if os.path.exists(outfile):
        os.remove(outfile)

    # ---------- rechunk and build per-variable encoding -----------------------
    ds = ds.chunk({"time": time_chunk})

    encoding: dict[str, dict] = {}
    for v in ds.data_vars:
        var = ds[v]
        # Compose chunksizes tuple that matches the new chunking
        chunksizes = tuple(
            time_chunk if d == "time" else s
            for d, s in zip(var.dims, var.shape)
        )
        encoding[v] = {
            "zlib": True,
            "complevel": complevel,
            "chunksizes": chunksizes,
        }

    # ---------- write (streaming) --------------------------------------------
    ds.to_netcdf(outfile, engine="netcdf4", mode="w",
                 encoding=encoding, compute=True)

    print("✅  Dataset written to:", outfile)
    return outfile




# %%
def write_compressed_netcdf_old(output_folder, ds, prefix, compression="zstd"):
    """
    Writes a compressed NetCDF file using Zarr backend for each data variable.
    Includes rechunking to ensure uniform chunk sizes for Zarr compatibility.

    Args:
    ds: xarray.Dataset containing the data.
    prefix: Prefix for the output NetCDF file.
    compression: Compression method for Zarr groups.
    """
    print(f'Variables in dataset: {list(ds.data_vars)}')    
    
    ignored_suffixes = ['_globalmean', '_smoothed', '_ERF']
    sorted_variable_names = sorted([var_name for var_name in ds.data_vars 
                                    if not any(var_name.endswith(suffix) for suffix in ignored_suffixes)])
    dataset_variable_names = "-".join(sorted_variable_names)

   
    output_file = os.path.join(output_folder, f'{prefix}_{dataset_variable_names}.nc')
    print(f'Now writing to file {output_file} in folder {output_folder}')

    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Handle existing file
    if os.path.exists(output_file):
        os.remove(output_file)
    
    use_zarrtosave = False

    if use_zarrtosave: 
        # Define and apply uniform chunk sizes
        uniform_chunk_size = 100  # Define your chunk size here
        ds_uniform_chunks = ds.chunk({dim: uniform_chunk_size for dim in ds.dims})

        # Define compression settings
        compressor = zarr.Blosc(cname=compression) if compression else None
        encoding = {var_name: {"compressor": compressor} for var_name in ds_uniform_chunks.data_vars}

        # Use Zarr to create groups and write data
        zarr_store = zarr.DirectoryStore(output_file.replace('.nc', '')) # Storing in a directory
        ds_uniform_chunks.to_zarr(store=zarr_store, mode='w', encoding=encoding)

        # Convert Zarr store to NetCDF
        ds_zarr = open_any_cmip(zarr_store)
        ds_zarr.to_netcdf(os.path.join(output_folder, output_file))
        
    else: 
        # Define compression settings
        comp = dict(zlib=True, complevel=5)  # You can adjust the compression level
        encoding = {var: comp for var in ds.data_vars}

        # Write to NetCDF with compression
        ds.to_netcdf(output_file, mode='w', encoding=encoding)

    
    print(f"Dataset written to: {output_file} in folder {output_folder}")
    return output_file



# %%
#
# %%
# Get the full path of the current notebook
#notebook_path = ipynbname.path()

# Print the full path
#print(f"Full path of the current notebook: {notebook_path}")


# %%
