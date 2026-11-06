# %% [markdown]
"""
# ERA5 API Data Download Notebook

Downloads ERA5 monthly reanalysis data in per‐variable, per‐decade ZIPs and ERA5 daily statistics (tasmin, tasmax) per‐year.
It resumes incomplete downloads, checks the latest data chunk for staleness (>72 h) and re‐downloads if needed.

Directory layout:
```
../data/raw/ERA5/
    monthly/<variable>/ERA5_monthly_<variable>_<start>-<end>.zip
    daily/<base_daily_var>/<statistic>/ERA5_daily_<statistic>_<year>.zip
```
"""

# %%
import cdsapi
from pathlib import Path
import time
from datetime import datetime, timezone
from tqdm.cli import tqdm
from itertools import product

# %% [markdown]
"""
## Configuration
"""

# %%
# CDS API datasets
monthly_dataset = "reanalysis-era5-single-levels-monthly-means"
daily_dataset   = "derived-era5-single-levels-daily-statistics"

# Variables for monthly data
monthly_vars = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "mean_sea_level_pressure",
    "total_precipitation",
    "skin_temperature",
    "10m_wind_speed",
    "surface_solar_radiation_downwards",
    "toa_incident_solar_radiation",
    "top_net_solar_radiation",
    "top_net_thermal_radiation",
    "total_cloud_cover",
]

# Daily statistics configuration: (output name, daily_statistic code)
daily_stats = [
    ("tasmin", "daily_minimum"),
    ("tasmax", "daily_maximum"),
]
# Base variable for daily statistics
base_daily_var = "2m_temperature"

tz = timezone.utc
start_year = 1940
end_year = datetime.now(tz).year

# Common time blocks
months = [f"{m:02d}" for m in range(1, 13)]
times  = ["00:00"]
days   = [f"{d:02d}" for d in range(1, 32)]

# Chunk sizes
monthly_chunk_years = 10
# build decade chunks
monthly_chunks = []
for y0 in range(start_year, end_year+1, monthly_chunk_years):
    y1 = min(y0 + monthly_chunk_years - 1, end_year)
    label = f"{y0}-{y1}"
    years = [str(y) for y in range(y0, y1+1)]
    monthly_chunks.append((label, years))
# daily uses one-year chunks
daily_years = [str(y) for y in range(start_year, end_year+1)]

# Download folders
root = Path("../data/raw/ERA5")
monthly_folder = root / "monthly"
daily_folder   = root / "daily"
monthly_folder.mkdir(parents=True, exist_ok=True)
daily_folder.mkdir(parents=True, exist_ok=True)

# Download control
min_file_size = 1e6      # bytes
max_retries   = 3
retry_delay   = 60       # seconds
stale_hours   = 72       # re-download if file older than this

# Data format keys
data_format     = "netcdf"
download_format = "unarchived"

# Initialize CDS API client
client = cdsapi.Client()

# %% [markdown]
"""
## Helpers
- `is_stale(path)`: True if file exists and is older than `stale_hours`.
"""
# %%
def is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    age = time.time() - path.stat().st_mtime
    return age > stale_hours * 3600

# %% [markdown]
"""
## Monthly download
Downloads each variable‐chunk ZIP, skipping complete chunks, re‐downloading stale latest chunk.
"""
# %%
def download_monthly(var: str, label: str, years: list[str]):
    var_dir = monthly_folder / var
    var_dir.mkdir(parents=True, exist_ok=True)
    fname = f"ERA5_monthly_{var}_{label}.nc"
    target = var_dir / fname

    # Determine if we need to download:
    if str(end_year) in years:
        # latest chunk: download if missing or stale
        if not target.exists() or is_stale(target):
            if target.exists(): target.unlink()
        else:
            return
    else:
        # older chunk: skip if exists and meets size
        if target.exists() and target.stat().st_size >= min_file_size:
            return
        if target.exists(): target.unlink()

    request = {
        "product_type":     ["monthly_averaged_reanalysis"],
        "variable":         [var],
        "year":             years,
        "month":            months,
        "time":             times,
        "data_format":      data_format,
        "download_format":  download_format,
    }
    for attempt in range(1, max_retries+1):
        try:
            client.retrieve(monthly_dataset, request).download(str(target))
            print(f"Monthly: {var} {label} downloaded.")
            return
        except Exception as e:
            print(f"Error monthly {var} {label}, att {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise

# %% [markdown]
"""
## Daily download
Downloads each daily statistic per year, skipping complete years, re‐downloading stale latest year.
"""
# %%
def download_daily(stat_name: str, stat_code: str, year: str):
    # Use base_daily_var subfolder, then stat_name
    var_dir = daily_folder / base_daily_var / stat_name
    var_dir.mkdir(parents=True, exist_ok=True)
    fname = f"ERA5_daily_{stat_name}_{year}.zip"
    target = var_dir / fname

    # Latest year stale check
    if year == str(end_year):
        if not target.exists() or is_stale(target):
            if target.exists(): target.unlink()
        else:
            return
    else:
        if target.exists() and target.stat().st_size >= min_file_size:
            return
        if target.exists(): target.unlink()

    request = {
        "product_type":    "reanalysis",
        "variable":        [base_daily_var],
        "year":            [year],
        "month":           months,
        "day":             days,
        "daily_statistic": stat_code,
        "time_zone":       "utc+00:00",
        "frequency":       "1_hourly",
        "data_format":     data_format,
        "download_format": download_format,
    }
    for attempt in range(1, max_retries+1):
        try:
            client.retrieve(daily_dataset, request).download(str(target))
            print(f"Daily: {stat_name} {year} downloaded.")
            return
        except Exception as e:
            print(f"Error daily {stat_name} {year}, att {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise

# %% [markdown]
"""
## Main loops
1. Monthly: per-variable, per-decade
2. Daily: per-statistic, per-year
"""
# %%
# Monthly tasks
tasks_monthly = list(product(monthly_vars, monthly_chunks))
for var, (label, years) in tqdm(tasks_monthly, desc="Monthly downloads", total=len(tasks_monthly)):
    download_monthly(var, label, years)

# Daily tasks
tasks_daily = list(product(daily_stats, daily_years))
for (stat_name, stat_code), year in tqdm(tasks_daily, desc="Daily downloads", total=len(tasks_daily)):
    download_daily(stat_name, stat_code, year)

# %% [markdown]
"""
## Next steps
- Unzip and combine with xarray notebooks
- Extend to other derived variables as needed
"""
