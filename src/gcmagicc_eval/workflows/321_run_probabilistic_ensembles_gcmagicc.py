# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %%
# ruff: noqa: E402
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# Standard imports
import gc
import inspect
import os
import sys
import json
import shlex
import time
import pickle
import subprocess
import random
import re
import math
import fnmatch
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Optional, Sequence

# Add parent directory to Python path to find scr module
# Handle both script and notebook contexts
try:
    # Running as a script - __file__ is defined
    parent_dir = Path(__file__).parent.parent
except NameError:
    # Running in a notebook - __file__ is not defined
    # Try to detect workspace root: go up from notebooks/ if we're in it, otherwise use cwd
    cwd = Path(os.getcwd())
    if cwd.name == 'notebooks' and cwd.parent.exists():
        parent_dir = cwd.parent
    else:
        # Fallback: assume we're already at workspace root or adjust as needed
        parent_dir = cwd
sys.path.insert(0, str(parent_dir))

from scr.validation_helpers.helper_path_utils import (
    CANONICAL_KIND_CHOICES,
    CANONICAL_KIND_ORIGINAL,
    build_era5spliced_dataset_path,
    get_era5spliced_root,
    get_archived_databases_path,
    get_data_path,
    get_gcmagicc_path,
    get_metric_databases_path,
    get_newscenario_inputs_root,
    get_output_folder,
    get_repo_path,
    get_sqlite_path,
    normalize_n_ensemble_label,
    normalize_runmodus_canonical,
    split_experiment_and_runmodus,
)

DEFAULT_AUTO_CONSOLIDATE = os.environ.get("GCMAGICC_AUTO_CONSOLIDATE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DEFAULT_AUTO_CONSOLIDATE_CLEANUP_LOCAL = (
    os.environ.get("GCMAGICC_AUTO_CONSOLIDATE_CLEANUP_LOCAL", "1").strip().lower()
    not in {"0", "false", "no", "off"}
)


def _consolidator_script_path() -> Path:
    return (get_repo_path("gcmmagicc") / "scripts" / "2018_consolidate_era5spliced_s3.py").resolve(strict=False)


def _default_autoconsolidate_config() -> Path:
    return (
        get_repo_path("gcmmagicc") / "scripts" / "2018_consolidate_era5spliced_s3.example.json"
    ).resolve(strict=False)


def _is_debiasloop_root(path: Path) -> bool:
    for part in path.resolve(strict=False).parts:
        if part.startswith("debiasloop_"):
            return True
    return False


def _run_autoconsolidate(
    *,
    source_paths: Sequence[Path],
    config_path: Optional[Path],
    cleanup_local: bool,
) -> None:
    consolidator = _consolidator_script_path()
    if not consolidator.exists():
        raise FileNotFoundError(f"Consolidator script not found: {consolidator}")
    cfg = Path(config_path).expanduser().resolve(strict=False) if config_path else _default_autoconsolidate_config()
    cmd: list[str] = [
        sys.executable,
        str(consolidator),
        "autoconsolidate",
        "--config",
        str(cfg),
    ]
    for source_path in source_paths:
        cmd.extend(["--source-path", str(Path(source_path).expanduser().resolve(strict=False))])
    cmd.extend(["--apply", "--verify-size-only"])
    if cleanup_local:
        cmd.append("--cleanup-local")
    print("🔁 Running auto-consolidate:")
    print("   " + " ".join(shlex.quote(token) for token in cmd))
    cp = subprocess.run(cmd, check=False)
    if cp.returncode != 0:
        raise RuntimeError(
            f"Auto-consolidate failed with exit code {cp.returncode}. "
            "Staged local data was kept for inspection."
        )

# %% [markdown]
# # 321 - Run GCMagicc new scenarios from MAGICCxERA5 predictors
#
# This notebook/script executes **GCMagicc v6** using ERA5-spliced predictor files produced by
# **616_* scripts**. It is designed to be **robust** against memory buildup by spawning a fresh
# Python worker **process per ensemble member**.
#
# ## What it does
#
# - Lets you **choose scenarios** and **subset of ensembles** to run (via config *and* CLI)
# - Lets you pick **naming conventions and output folders**
# - Mirrors the 300_/310_ workflow: parallelized orchestration with taskset/ionice/nice,
#   aggressively avoids memory leakage by process isolation, plus explicit GC
# - Supports two output modes:
#   - **simple**: one NetCDF per run containing all predicted variables
#   - **cmip6**: CMIP6-like directory layout with a NetCDF per variable
#
# ## Inputs expected (from 616_*):
# - ERA5-spliced predictors from `GCMAGICC_ERA5_SPLICED_PREDICTOR_DIR`
# - Structure: `{base}/magicc_based_predictors_*/n_*/{AR6|AR7}/runmode_{all|natural|aerosol|anthropogenic}/predictors/{scenario}/`
#
# ## Outputs
# - In `<OUTPUT_ROOT>/<scenario>/...` according to your chosen mode/template.
#
# ---
# **Tip:** You can run this file as a script with CLI overrides *or* open as a notebook and edit
# the **config block** below.

# %%
# --- per-run thread caps (tunable, default=8). Must come before numpy/torch/xarray imports. ---
import os as _os_threadcap

_THREADS = int(_os_threadcap.environ.get("THREADS_PER_RUN", "8"))
_os_threadcap.environ.setdefault("OMP_NUM_THREADS", str(_THREADS))
_os_threadcap.environ.setdefault("MKL_NUM_THREADS", str(_THREADS))
_os_threadcap.environ.setdefault("OPENBLAS_NUM_THREADS", str(_THREADS))
_os_threadcap.environ.setdefault("BLIS_NUM_THREADS", str(_THREADS))
_os_threadcap.environ.setdefault("VECLIB_MAXIMUM", str(_THREADS))
_os_threadcap.environ.setdefault("NUMEXPR_MAX_THREADS", str(_THREADS))
_os_threadcap.environ.setdefault("TORCH_NUM_THREADS", str(_THREADS))
_os_threadcap.environ.setdefault("KMP_BLOCKTIME", "0")
_os_threadcap.environ.setdefault("OMP_DYNAMIC", "FALSE")
_os_threadcap.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_os_threadcap.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Set low priority for this computational process (optional)
try:
    from priority_manager import set_low_priority

    set_low_priority(enable=True, cpu_nice=6, io_class="best-effort")
    print("✓ Set low priority for orchestration process")
except Exception:
    pass

# %%
# Additional imports
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import cftime
import torch

# Clean up temp import var
del _os_threadcap

# Device helpers (GPU first like 320_*, with run_general preflight)
def normalize_device_string(device: Optional[str]) -> Optional[str]:
    if device is None:
        return None
    dev = device.strip()
    if not dev:
        return None
    low = dev.lower()
    if low in {"auto", "default"}:
        return None
    if low in {"gpu", "cuda"}:
        return "cuda"
    if low.startswith("cuda:"):
        suffix = low.split(":", 1)[1].strip()
        return f"cuda:{suffix}" if suffix else "cuda"
    if low.startswith("cuda") and low[4:].isdigit():
        return f"cuda:{low[4:]}"
    if low.startswith("gpu") and low[3:].isdigit():
        return f"cuda:{low[3:]}"
    if low == "cpu":
        return "cpu"
    return dev


def _device_is_cuda(dev: Optional[str]) -> bool:
    return isinstance(dev, str) and dev.lower().startswith("cuda")


def _cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _explain_cuda_status() -> None:
    try:
        print(f"   • torch version: {torch.__version__}")
        print(f"   • torch.version.cuda: {getattr(torch.version, 'cuda', None)}")
        print(f"   • torch.version.hip : {getattr(torch.version, 'hip', None)}")
    except Exception:
        return
    for k in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "LOCAL_RANK", "SLURM_LOCALID"):
        v = os.environ.get(k)
        if v is not None:
            print(f"   • {k}: {v}")


def _gpu_preflight(device_str: str, warn: bool = True) -> bool:
    if not isinstance(device_str, str) or not device_str.startswith("cuda"):
        return False
    if not torch.cuda.is_available():
        return False
    try:
        idx = int(device_str.split(":", 1)[1]) if ":" in device_str else 0
    except Exception:
        idx = 0
    try:
        torch.cuda.set_device(idx)
        t = torch.empty(1, device=f"cuda:{idx}")
        t += 1
        torch.cuda.synchronize()
        return True
    except Exception as exc:
        if warn:
            print(f"⚠️  GPU preflight failed on {device_str}: {exc}")
        return False


def _maybe_enable_tf32(enable: bool = True) -> None:
    try:
        if enable and hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if enable and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high" if enable else "medium")
        except Exception:
            pass
    except Exception:
        return


def _read_meminfo_kb() -> tuple[Optional[int], Optional[int]]:
    """
    Lightweight /proc/meminfo parser; returns (total_kb, available_kb).
    """
    try:
        total = avail = None
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        total = int(parts[1])
                elif line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        avail = int(parts[1])
                if total is not None and avail is not None:
                    break
        return total, avail
    except Exception:
        return None, None


def _memory_gb() -> tuple[float, float]:
    """
    Return (total_GB, available_GB) with psutil fallback to /proc/meminfo.
    """
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return vm.total / (1024**3), vm.available / (1024**3)
    except Exception:
        total_kb, avail_kb = _read_meminfo_kb()
        if total_kb is None or avail_kb is None:
            return 0.0, 0.0
        return total_kb / (1024**2), avail_kb / (1024**2)


def _estimate_cpu_worker_count(
    *,
    per_job_cpus: int = 7,
    per_job_ram_gb: float = 40.0,
    safety_fraction: float = 0.20,
) -> int:
    """
    Estimate a safe CPU worker count given per-job CPU/RAM requirements.

    - Removes a safety buffer (default 20%) from total CPU and RAM.
    - Limits by whichever resource (CPU or RAM) is more constrained.
    - Falls back to 1 if resource detection fails.
    """
    total_cpus = cpu_count() or os.cpu_count() or 1
    cpu_budget = max(1, int(math.floor(total_cpus * (1.0 - safety_fraction))))
    total_mem_gb, avail_mem_gb = _memory_gb()
    mem_cap_gb = total_mem_gb * (1.0 - safety_fraction) if total_mem_gb > 0 else 0.0
    mem_budget_gb = min(avail_mem_gb, mem_cap_gb) if mem_cap_gb > 0 else avail_mem_gb

    cpu_limited = max(1, cpu_budget // max(1, int(per_job_cpus)))

    if per_job_ram_gb > 0 and mem_budget_gb > 0:
        mem_limited = max(1, int(math.floor(mem_budget_gb / per_job_ram_gb)))
    else:
        mem_limited = cpu_limited

    workers = max(1, min(cpu_limited, mem_limited))
    return workers


def _prepare_writable_dir(path: Path) -> tuple[bool, Optional[str]]:
    """
    Ensure directory exists and is writable by creating/removing a probe file.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".gcmagicc_write_probe_{os.getpid()}_{int(time.time() * 1_000_000)}"
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok\n")
        probe.unlink(missing_ok=True)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _candidate_output_roots() -> List[Path]:
    candidates: List[Path] = []
    env_fallback = os.environ.get("GCMAGICC_OUTPUT_FALLBACK_ROOT", "").strip()
    if env_fallback:
        candidates.append(Path(env_fallback))
    try:
        candidates.append(get_era5spliced_root())
    except Exception:
        pass
    try:
        candidates.append(get_repo_path("gcmagicc_ensemble_runner") / "created_nc_files")
    except Exception:
        pass
    candidates.append(parent_dir / "created_nc_files")
    candidates.append(Path.cwd() / "created_nc_files")
    candidates.append(Path.home() / "gcmagicc_outputs")
    return candidates


def resolve_writable_output_root(
    preferred_root: Union[str, Path],
    *,
    context: str = "output",
    allow_fallback: bool = True,
) -> Path:
    """
    Prefer the requested output root, but optionally fall back to known writable roots.
    """
    preferred = Path(preferred_root).expanduser().resolve(strict=False)
    ok, err = _prepare_writable_dir(preferred)
    if ok:
        return preferred

    if not allow_fallback:
        raise OSError(f"{context} root is not writable: {preferred} ({err})")

    seen: set[str] = {str(preferred)}
    candidates: List[Path] = []
    for raw in _candidate_output_roots():
        cand = raw.expanduser().resolve(strict=False)
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(cand)

    for cand in candidates:
        ok_cand, err_cand = _prepare_writable_dir(cand)
        if ok_cand:
            print(
                f"⚠️  Preferred {context} root is not writable: {preferred} ({err}). "
                f"Falling back to {cand}"
            )
            return cand
        err = err_cand or err

    tried = ", ".join(str(p) for p in candidates) if candidates else "<none>"
    raise OSError(
        f"Could not find a writable {context} root. Preferred: {preferred} ({err}). "
        f"Tried fallbacks: {tried}"
    )


def _amp_dtype_cuda():
    try:
        return torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
    except Exception:
        return torch.float16


_DEVICE_RESOLUTION_LOG: list[str] = []


def _record_device_note(message: str) -> None:
    _DEVICE_RESOLUTION_LOG.append(message)


def detect_default_device(force_gpu: bool = False) -> str:
    env_device = normalize_device_string(os.environ.get("GCMAGICC_DEVICE"))
    if env_device:
        if env_device.startswith("cuda") and not _cuda_available() and not force_gpu:
            _record_device_note(
                f"GCMAGICC_DEVICE='{env_device}' requested GPU but CUDA unavailable; falling back to CPU."
            )
            print(f"⚠️  Requested GPU device '{env_device}' but torch.cuda.is_available() is False; falling back to CPU.")
            return "cpu"
        _record_device_note(f"Using device override from GCMAGICC_DEVICE='{env_device}'.")
        return env_device

    env_gpu_name = os.environ.get("GCMAGICC_GPU_NAME")
    if env_gpu_name and _cuda_available():
        try:
            count = torch.cuda.device_count()
        except Exception:
            count = 0
        for idx in range(count):
            try:
                dev_name = torch.cuda.get_device_name(idx)
            except Exception:
                continue
            if env_gpu_name.strip().lower() in dev_name.lower():
                _record_device_note(f"Matched GPU name '{env_gpu_name}' to CUDA device index {idx}.")
                return f"cuda:{idx}"
        print(f"⚠️  Could not match GCMAGICC_GPU_NAME='{env_gpu_name}' to an available GPU; continuing with automatic selection.")
        _record_device_note(f"GCMAGICC_GPU_NAME='{env_gpu_name}' did not match any visible GPU.")

    if _cuda_available() or force_gpu:
        if force_gpu and not _cuda_available():
            _record_device_note("Force GPU enabled; attempting CUDA despite availability check.")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            first = visible.split(",")[0].strip()
            if first:
                _record_device_note(f"Selected first CUDA_VISIBLE_DEVICES entry '{first}'.")
                return f"cuda:{first}" if first.isdigit() else "cuda"
        _record_device_note("Defaulting to CUDA (no specific index).")
        return "cuda"

    _record_device_note("CUDA unavailable: torch.cuda.is_available() returned False.")
    return "cpu"


def report_device_status(device: str) -> None:
    print(f"🖥️  GCMagicc device: {device}")
    for note in _DEVICE_RESOLUTION_LOG:
        print(f"   • {note}")
    cuda_available = _cuda_available()
    print(f"   • torch.cuda.is_available(): {cuda_available}")
    _explain_cuda_status()
    if cuda_available and device.startswith("cuda"):
        try:
            idx = int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
        except Exception:
            idx = 0
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception:
            name = "unknown"
        print(f"   • Using CUDA device {idx}: {name}")

# Device defaults / env toggles
DEFAULT_TF32 = os.environ.get("GCMAGICC_TF32", "1").lower() not in ("0", "false", "no")
DEFAULT_AMP = os.environ.get("GCMAGICC_USE_AMP", "1").lower() not in ("0", "false", "no")
MIN_FREE_MEMORY_GB = float(os.environ.get("GCMAGICC_MIN_FREE_MEMORY_GB", "0.0"))
_FORCE_GPU = bool(os.environ.get("GCMAGICC_FORCE_GPU", ""))
DEFAULT_DEVICE = detect_default_device(force_gpu=_FORCE_GPU)

# %% [markdown]
# ## 0) User config (edit me)

# %%
# --- Paths ---
WORKSPACE_ROOT = parent_dir  # resolved above to repo root whether running as script or notebook
# Canonical base output directory:
#   <ERA5spliced>/<version>/<experiment>/<ARX>/<runmodus>/<n_ensemble>/<original|dataderivatives>
DEFAULT_OUTPUT_ROOT = get_era5spliced_root()
OUTPUT_ROOT = Path(
    os.environ.get(
        "GCMAGICC_ENSEMBLE_OUTPUT_ROOT",
        str(DEFAULT_OUTPUT_ROOT),
    )
).expanduser().resolve(strict=False)
DEFAULT_CANONICAL_LAYOUT = os.environ.get("GCMAGICC_CANONICAL_LAYOUT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
DEFAULT_CANONICAL_KIND = os.environ.get("GCMAGICC_CANONICAL_KIND", CANONICAL_KIND_ORIGINAL).strip().lower()

# Choose GCMagicc model variant (v6 is Nextvers5)
SERVER_DEVICE = os.environ.get("GCMAGICC_SERVER_DEVICE", "gus_server").strip() or "gus_server"
MODEL_VERSION = os.environ.get("GCMAGICC_MODEL_VERSION", "NxlversA5").strip() or "NxlversA5"
# GCMagicc path: can be set via GCMAGICC_MODEL_ROOT env var, or defaults to absolute path
# Expected structure: {GCMagiccpath}/run_general.py and {GCMagiccpath}/{MODELS_SUBDIR}/
GCMagiccpath = os.environ.get(
    "GCMAGICC_MODEL_ROOT",
    str(get_repo_path("gcm_firefly_data") / f"model_{MODEL_VERSION}")
)
MODELS_SUBDIR = "modelsA"

# Device & threading for the *worker* (each run)
DEVICE = normalize_device_string(os.environ.get("GCMAGICC_DEVICE")) or DEFAULT_DEVICE
THREADS_PER_RUN = int(os.environ.get("THREADS_PER_RUN", 8))  # worker will inherit/override env

# Output mode: "simple" (one NetCDF containing all vars) or "cmip6"
OUTPUT_MODE = "simple"

# Naming template for simple mode (used to create filename, without extension)
# Available fields: scenario, ens (int), version (MODEL_VERSION), date, dep ("d1"/"d0"), bias, effect, magicc_ens, runmodus
# Format: GCMagicc-v{version}-{dep}b{bias}e{effect}m{magicc_ens}-{date}_{scenario}[-{runmodus}]_{ens}
# where [-{runmodus}] is omitted if runmodus is 'all', otherwise '-nat', '-aer', or '-anthrop'
NAMING_TEMPLATE = (
    "GCMagicc-v{version}-{dep}b{bias}e{effect}m{magicc_ens}-{date}_{scenario}{runmodus}_{ens}"
)

# Dependence flag passed to the model (run_general.sample_from_combined_model)
DEPENDENCE = True

# Grid & calendar for output
NLAT = 180
CAL = "360_day"
LON_CONVENTION = "360"

# Parallel orchestration
MAX_WORKERS = 8  # concurrent worker processes
CPU_PIN = True  # pin each process to a CPU slice via taskset
NICE_IO = True  # wrap with nice/ionice
RESUME = True  # skip outputs that already exist
DRY_RUN = False  # show tasks without running

# Scenario and ensemble selection (also overridable via CLI)
# SCENARIOS: Optional[List[str]] = ["ssp245"]  # None -> auto-discover
# data/site_eth/projects/gcmmagicc/data/newscenario_inputs/magicc_based_predictors_20260206_032503/n_20/AR6/runmode_all/predictors/Current-Policies-GCAM
SCENARIOS: Optional[List[str]] = ["SSP2-com",] # ["NDC-submitted-low", ] # ["Current-Policies-GCAM",] # ["NDC-submitted-low", "NDC-Trump-low"]   # None -> auto-discover
# SCENARIOS: Optional[List[str]] = ["NDC-submitted-low", "NDC-Trump-low"]   # None -> auto-discover
# SCENARIOS: Optional[List[str]] = [ "ssp434", "ssp3h", "ssp534-over", "ssp585", "NDC-submitted-high", "NDC-submitted-low", "NDC-Trump-high", "NDC-Trump-low"] # ["ssp245"]  # None -> auto-discover from predictors layout
# SCENARIOS: Optional[List[str]] = ["ssp434", "ssp460", "ssp534-over", "ssp585"] # [ "ssp434", "ssp534-over", "ssp585"] # ["ssp245"]  # None -> auto-discover from predictors layout
# SCENARIO_WHITELIST: regex patterns to filter scenarios (e.g., ['ssp.*', 'ssp245'] or ['^ssp'])
# If None or empty, all discovered scenarios are used. Patterns are matched using re.search().
_SCENARIO_WHITELIST_DEFAULT: Optional[List[str]] = None # ["^ssp245$"]
#  _SCENARIO_WHITELIST_DEFAULT = ["L", "LN", "M", "VL",]
SCENARIO_WHITELIST: Optional[List[str]] = (
    os.environ.get("GCMAGICC_SCENARIO_WHITELIST", "").split(",")
    if os.environ.get("GCMAGICC_SCENARIO_WHITELIST")
    else _SCENARIO_WHITELIST_DEFAULT
)
if SCENARIO_WHITELIST and len(SCENARIO_WHITELIST) == 1 and not SCENARIO_WHITELIST[0]:
    SCENARIO_WHITELIST = None
TEST_ONE = False  # if True, run a single ensemble per scenario

# Runmodus selection: 'all' | 'natural' | 'aerosol' | 'anthropogenic'
# Can be a comma-separated whitelist (e.g., "all,natural")
# USE_RUNMODUSE = os.environ.get("GCMAGICC_USE_RUNMODUSE", "all, natural") # "all,natural"
USE_RUNMODUSE = os.environ.get("GCMAGICC_USE_RUNMODUSE", "all") # "all,natural"

# AR6/AR7 workflow selection: 'AR6' | 'AR7' | 'all'
# Can be a comma-separated whitelist (e.g., "AR6,AR7")
USE_WORKFLOW = os.environ.get("GCMAGICC_USE_WORKFLOW", "AR6") # "AR6,AR7"


# ERA5_SPLICED_PREDICTOR_DIR: base directory for MAGICCxERA5 predictors
# New structure: {base}/magicc_based_predictors_*/n_*/{AR6|AR7}/runmode_{all|natural|aerosol|anthropogenic}/predictors/{scenario}/
ERA5_SPLICED_PREDICTOR_DIR = os.environ.get(
    "GCMAGICC_ERA5_SPLICED_PREDICTOR_DIR",
    str(get_newscenario_inputs_root()),
)
# SPLICED_VARIANT_GLOB = os.environ.get("GCMAGICC_SPLICED_VARIANT_GLOB", "magicc_based_predictors_*")
# SPLICED_VARIANT_GLOB = os.environ.get("GCMAGICC_SPLICED_VARIANT_GLOB", "magicc_based_predictors_20260124_044739")
# SPLICED_VARIANT_GLOB = os.environ.get("GCMAGICC_SPLICED_VARIANT_GLOB", "magicc_based_predictors_20260124_044739")
# SPLICED_VARIANT_GLOB = os.environ.get("GCMAGICC_SPLICED_VARIANT_GLOB", "magicc_based_predictors_20260206_032503")
SPLICED_VARIANT_GLOB = os.environ.get("GCMAGICC_SPLICED_VARIANT_GLOB", "magicc_based_predictors_20260309_223935")
# data/newscenario_inputs/magicc_based_predictors_20260206_032503
# SPLICED_VARIANT_GLOB = os.environ.get("GCMAGICC_SPLICED_VARIANT_GLOB", "magicc_based_predictors_20260124_044739")

# data/site_gus/projects/gcmmagicc/data/newscenario_inputs/magicc_based_predictors_20260206_032503/n_100/AR6/runmode_all/predictors/
# data/site_eth/projects/gcmmagicc/data/newscenario_inputs/magicc_based_predictors_20260204_083640
# The variant is the name of the directory that contains the ERA5-spliced predictors. For example, magicc_based_predictors_20251220_061004
# SPLICED_VARIANT = os.environ.get("GCMAGICC_SPLICED_VARIANT", "magicc_based_predictors_20260206_032503")
# SPLICED_VARIANT = os.environ.get("GCMAGICC_SPLICED_VARIANT", "magicc_based_predictors_20260206_032503")
# SPLICED_VARIANT = os.environ.get("GCMAGICC_SPLICED_VARIANT", "magicc_based_predictors_20260124_044739")
SPLICED_VARIANT = os.environ.get("GCMAGICC_SPLICED_VARIANT", "magicc_based_predictors_20260309_223935")

# 
# SPLICED_N = int(os.environ.get("GCMAGICC_SPLICED_N", "20"))
SPLICED_N = int(os.environ.get("GCMAGICC_SPLICED_N", "20"))


ENSEMBLES_D: int = 20            # draws from ERA5-spliced predictors
ENSEMBLES_D = min(ENSEMBLES_D, SPLICED_N)
BIASCORRECT_TO_ERA5_D: bool = True  # effect index fixed to ERA5 (0)

# %% [markdown]
# ## 1) Helper utilities


# %%
@dataclass
class PredictorData:
    X: torch.Tensor
    predictor_names: List[str]
    variables_2predict: List[str]
    model_names: Optional[List[str]] = None
    year: Optional[np.ndarray] = None
    month: Optional[np.ndarray] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    start_month: Optional[int] = None
    end_month: Optional[int] = None


# %%
# ERA5-spliced predictor helpers (Option D analogue)


def _normalize_scenario_name(name: str) -> str:
    return name.replace("clean_", "", 1) if name.startswith("clean_") else name


def filter_scenarios_by_patterns(
    scenarios: List[str],
    patterns: Optional[List[str]],
) -> List[str]:
    """
    Filter scenarios by regex patterns.
    
    Args:
        scenarios: List of scenario names to filter
        patterns: List of regex patterns (e.g., ['ssp.*', 'ssp245']). If None or empty, returns all scenarios.
    
    Returns:
        Filtered list of scenarios that match any of the patterns.
    """
    if not patterns or len(patterns) == 0:
        return scenarios
    
    filtered = []
    for scen in scenarios:
        scen_norm = _normalize_scenario_name(scen)
        for pattern in patterns:
            pattern = pattern.strip()
            if not pattern:
                continue
            try:
                if re.search(pattern, scen_norm):
                    filtered.append(scen)
                    break  # Match found, no need to check other patterns
            except re.error as e:
                print(f"⚠️  Invalid regex pattern '{pattern}': {e}. Skipping this pattern.")
                continue
    
    return filtered


def _has_glob_pattern(value: str) -> bool:
    return any(ch in value for ch in ("*", "?", "["))


def expand_scenario_globs(
    requested: Sequence[str],
    available: Sequence[str],
) -> tuple[List[str], List[str]]:
    """
    Expand glob-style scenario patterns (e.g., "H_*") using available scenario names.

    Returns (expanded, missing_patterns).
    """
    available_norm = [_normalize_scenario_name(s) for s in available]
    expanded: List[str] = []
    missing: List[str] = []
    seen: set[str] = set()
    for req in requested:
        req_norm = _normalize_scenario_name(str(req))
        if _has_glob_pattern(req_norm):
            matches = [s for s in available_norm if fnmatch.fnmatchcase(s, req_norm)]
            if matches:
                for s in matches:
                    if s not in seen:
                        expanded.append(s)
                        seen.add(s)
            else:
                missing.append(str(req))
        else:
            if req_norm not in seen:
                expanded.append(req_norm)
                seen.add(req_norm)
    return expanded, missing


def _split_csv_values(value: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "," in text:
        return [v.strip() for v in text.split(",") if v.strip()]
    return [text]


def normalize_workflow_list(workflow: Optional[Union[str, Sequence[str]]]) -> List[str]:
    vals = _split_csv_values(workflow)
    if not vals:
        return ["AR6"]
    out: List[str] = []
    for v in vals:
        v_up = v.upper()
        if v_up == "ALL":
            return ["AR6", "AR7"]
        if v_up not in {"AR6", "AR7"}:
            raise ValueError(f"Unsupported workflow value '{v}'. Use AR6, AR7, or all.")
        if v_up not in out:
            out.append(v_up)
    return out


def normalize_runmodus_list(runmodus: Optional[Union[str, Sequence[str]]]) -> List[str]:
    vals = _split_csv_values(runmodus)
    if not vals:
        return ["all"]
    out: List[str] = []
    for v in vals:
        v_low = v.lower()
        if v_low not in {"all", "natural", "aerosol", "anthropogenic"}:
            raise ValueError(
                f"Unsupported runmodus value '{v}'. Use all, natural, aerosol, or anthropogenic."
            )
        if v_low not in out:
            out.append(v_low)
    return out


def resolve_scenario_request(scenario: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Map a requested scenario to the predictor source scenario and optional year bounds.
    
    Returns (source_scenario, year_start, year_end). The source_scenario is used for
    locating predictor files, while the requested scenario name is preserved for outputs.
    """
    scen_norm = _normalize_scenario_name(scenario)
    scen_lower = scen_norm.lower()
    if scen_lower == "historical":
        # Use SSP2-4.5 predictors but trim to historical years
        return "ssp245", None, 2015
    return scen_norm, None, None


def _resolve_spliced_variant_root(
    base_dir: Union[str, Path],
    *,
    variant: Optional[str],
    variant_glob: str,
    n_value: Optional[int],
) -> Path:
    base = Path(base_dir)
    if variant:
        var_path = Path(variant)
        if not var_path.is_absolute():
            var_path = base / variant
        if not var_path.is_dir():
            raise FileNotFoundError(f"Requested spliced variant not found: {var_path}")
        return var_path
    if n_value is not None and (base / f"n_{n_value}").is_dir():
        return base
    candidates = sorted([p for p in base.glob(variant_glob) if p.is_dir()])
    if n_value is not None:
        candidates = [p for p in candidates if (p / f"n_{n_value}").is_dir()]
    if candidates:
        return candidates[-1]
    return base


def _resolve_spliced_n_root(variant_root: Path, n_value: Optional[int]) -> Path:
    if variant_root.name.startswith("n_") and variant_root.is_dir():
        return variant_root
    if n_value is None:
        return variant_root
    n_dir = variant_root / f"n_{n_value}"
    if not n_dir.is_dir():
        raise FileNotFoundError(f"n_{n_value} not found under {variant_root}")
    return n_dir


def _resolve_runmode_dir(base: Path, runmodus: str) -> Path:
    for prefix in ("runmode_", "runmodus_"):
        cand = base / f"{prefix}{runmodus}"
        if cand.is_dir():
            return cand
    return base / f"runmode_{runmodus}"


def discover_spliced_scenarios(
    base_dir: Union[str, Path],
    *,
    workflows: Sequence[str],
    runmodus_list: Sequence[str],
    n_value: Optional[int],
    variant: Optional[str],
    variant_glob: str,
) -> List[str]:
    variant_root = _resolve_spliced_variant_root(
        base_dir, variant=variant, variant_glob=variant_glob, n_value=n_value
    )
    try:
        n_root = _resolve_spliced_n_root(variant_root, n_value)
    except FileNotFoundError:
        base = Path(base_dir)
        return sorted([p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")])
    scenarios: set[str] = set()
    for wf in workflows:
        for rm in runmodus_list:
            runmode_dir = _resolve_runmode_dir(n_root / wf, rm)
            pred_root = runmode_dir / "predictors"
            if not pred_root.is_dir():
                continue
            for scen_dir in pred_root.iterdir():
                if scen_dir.is_dir():
                    scenarios.add(scen_dir.name)
    return sorted(scenarios)


def _parse_spliced_run_id(path: Path) -> Optional[int]:
    """
    Extract MAGICC run_id from filenames like:
      Old format: predictors_magicc_scm_ssp245--MESSAGE-GLOBIOM-123---magiccscm-parquet.h5
      New format: predictors_AR6_all_ssp245_r0.h5 or predictors_AR7_natural_ssp245_r42.h5
    """
    # Try new format first: predictors_AR6_all_ssp245_r0.h5
    match = re.search(r"_r([0-9]+)(?:\.(?:h5|csv))?$", path.name)
    if match:
        return int(match.group(1))
    # Fallback to old format: predictors_magicc_scm_ssp245--MESSAGE-GLOBIOM-123---magiccscm-parquet.h5
    match = re.search(r"-([0-9]+)(?=---magiccscm-parquet)", path.name)
    return int(match.group(1)) if match else None


def discover_spliced_predictor_files(
    base_dir: Union[str, Path],
    scenarios: Optional[Sequence[str]] = None,
    workflow: str = "AR6",
    runmodus: str = "all",
    *,
    n_value: Optional[int] = None,
    variant: Optional[str] = None,
    variant_glob: str = SPLICED_VARIANT_GLOB,
    strict: bool = True,
) -> Dict[str, Dict[int, Path]]:
    """
    Return {scenario: {run_id: file_path}} for ERA5-spliced predictor files.
    
    New directory structure:
      {base_dir}/magicc_based_predictors_*/n_*/{AR6|AR7}/runmode_{all|natural|aerosol|anthropogenic}/predictors/{scenario}/
    Legacy directory structure:
      {base_dir}/{scenario}/{AR6|AR7}/runmodus_{all|natural|aerosol|anthropogenic}/
    
    Args:
        base_dir: Base directory containing scenario subdirectories
        scenarios: List of scenario names (None = auto-discover)
        workflow: 'AR6' | 'AR7' | 'all' (if 'all', searches both AR6 and AR7)
        runmodus: 'all' | 'natural' | 'aerosol' | 'anthropogenic'
        n_value: n for n_* (new layout only)
        variant: explicit magicc_based_predictors_* path or name (new layout only)
        variant_glob: glob for magicc_based_predictors_* (new layout only)
        strict: if True, raise on missing scenarios/runmodus; otherwise skip missing
    
    Raises with guidance if a requested scenario is missing.
    """
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(
            f"ERA5-spliced predictor directory not found: {base} "
            " (generate via notebooks/616_*.py in gcmmagicc)."
        )
    
    workflows_to_search = normalize_workflow_list(workflow)

    runmodus_norm = str(runmodus).lower()
    runmodus_dir = f"runmodus_{runmodus_norm}"
    runmode_dirname = f"runmode_{runmodus_norm}"
    
    # Detect new layout (magicc_based_predictors_*/n_*)
    n_value_use = n_value if n_value is not None else SPLICED_N
    variant_use = variant if variant else (SPLICED_VARIANT or None)
    variant_glob_use = variant_glob or SPLICED_VARIANT_GLOB
    variant_root = _resolve_spliced_variant_root(
        base, variant=variant_use, variant_glob=variant_glob_use, n_value=n_value_use
    )
    use_new_layout = False
    n_root: Optional[Path] = None
    try:
        n_root = _resolve_spliced_n_root(variant_root, n_value_use)
        use_new_layout = True
    except FileNotFoundError:
        use_new_layout = False

    # Discover scenarios if not provided
    if scenarios is None:
        if use_new_layout and n_root is not None:
            pred_root = _resolve_runmode_dir(n_root / workflows_to_search[0], runmodus_norm) / "predictors"
            scenarios = [p.name for p in pred_root.iterdir() if p.is_dir()] if pred_root.is_dir() else []
        else:
            scenarios = [p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")]
    
    requested = list(scenarios)
    mapping: Dict[str, Dict[int, Path]] = {}
    
    for scen in requested:
        scen_norm = _normalize_scenario_name(scen)
        run_map: Dict[int, Path] = {}
        
        # Search in each workflow directory
        for wf in workflows_to_search:
            if use_new_layout and n_root is not None:
                runmode_dir = _resolve_runmode_dir(n_root / wf, runmodus_norm)
                scen_dir = runmode_dir / "predictors" / scen_norm
            else:
                scen_dir = base / scen_norm / wf / runmodus_dir
                if not scen_dir.is_dir():
                    scen_dir = base / scen_norm / wf / runmode_dirname
            if not scen_dir.is_dir():
                continue

            # Try new format: predictors_AR6_all_ssp245_r0.h5
            files = list(scen_dir.glob(f"predictors_{wf}_*_{scen_norm}_r*.h5"))
            if not files:
                files = list(scen_dir.glob(f"predictors_{wf}_*_{scen_norm}_r*.csv"))

            # Fallback to old format if new format not found
            if not files:
                files = list(scen_dir.glob("predictors_magicc_scm_*-[0-9]*---magiccscm-parquet.h5"))
                if not files:
                    files = list(scen_dir.glob("predictors_magicc_scm_*-[0-9]*---magiccscm-parquet.csv"))

            # Process files found in this workflow
            for f in files:
                rid = _parse_spliced_run_id(f)
                if rid is None:
                    continue
                existing = run_map.get(rid)
                # Prefer HDF5 if both CSV and H5 exist
                if existing is not None and existing.suffix.lower() == ".h5":
                    continue
                run_map[rid] = f
        
        if not run_map:
            if strict:
                if use_new_layout and n_root is not None:
                    raise FileNotFoundError(
                        f"No predictors for scenario '{scen}' in {n_root} "
                        f"(workflow={workflow}, runmodus={runmodus_norm}). Expected: {runmode_dirname}/predictors/{scen_norm}/"
                    )
                # Legacy checks
                scen_base_dir = base / scen_norm
                if not scen_base_dir.is_dir():
                    raise FileNotFoundError(
                        f"No predictors for scenario '{scen}' in {base}. "
                        f"Expected structure: {base}/{scen_norm}/{{AR6|AR7}}/{runmodus_dir}/"
                    )
                workflows_found = [wf for wf in workflows_to_search if (base / scen_norm / wf).is_dir()]
                if not workflows_found:
                    raise FileNotFoundError(
                        f"No workflow directories found for scenario '{scen}' in {base}/{scen_norm}. "
                        f"Expected: AR6 or AR7 subdirectories."
                    )
                runmodus_found = False
                for wf in workflows_to_search:
                    if (base / scen_norm / wf / runmodus_dir).is_dir():
                        runmodus_found = True
                        break
                if not runmodus_found:
                    raise FileNotFoundError(
                        f"No predictor files found for scenario '{scen}' with workflow={workflow}, runmodus={runmodus_norm} "
                        f"in {base}/{scen_norm}/. Expected: {runmodus_dir} subdirectory."
                    )
        else:
            mapping[scen_norm] = run_map
    
    return mapping


def _assemble_spliced_predictors_from_arrays(
    *,
    year: np.ndarray,
    month: np.ndarray,
    series: Dict[str, np.ndarray],
    meta: dict,
    model_index_name: str = "ERA5",
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> PredictorData:
    """
    Slice to requested years and build PredictorData using meta['variables_X'].
    """
    year = year.astype(int)
    month = month.astype(int)
    if year_start is not None or year_end is not None:
        y0 = year_start if year_start is not None else int(year[0])
        y1 = year_end if year_end is not None else int(year[-1])
        mask = (year >= y0) & (year <= y1)
        year = year[mask]
        month = month[mask]
        series = {k: v[mask] for k, v in series.items()}

    if year.size == 0:
        raise ValueError("ERA5-spliced predictors have no time steps after filtering.")

    nT = int(year.shape[0])
    months_f32 = month.astype(np.float32)
    sin_time = np.sin((months_f32 - 1.0) / 12.0 * 2.0 * np.pi).astype(np.float32)
    cos_time = np.cos((months_f32 - 1.0) / 12.0 * 2.0 * np.pi).astype(np.float32)

    variables_X = meta["variables_X"]
    model_to_index = meta.get("model_to_index", {})
    model_index_val = float(model_to_index.get(model_index_name, 0))

    X_cols: List[np.ndarray] = []
    pred_names: List[str] = []

    for var in variables_X:
        if var == "model_index":
            X_cols.append(np.full((nT, 1), model_index_val, dtype=np.float32))
            pred_names.append("model_index")
        elif var == "month":
            X_cols.append(months_f32.reshape(nT, 1))
            pred_names.append("month")
        elif var == "time":
            X_cols.append(sin_time.reshape(nT, 1))
            pred_names.append("sin_time")
            X_cols.append(cos_time.reshape(nT, 1))
            pred_names.append("cos_time")
        else:
            arr = series.get(var)
            if arr is None:
                arr = np.zeros((nT,), dtype=np.float32)
            X_cols.append(np.asarray(arr, dtype=np.float32).reshape(nT, 1))
            pred_names.append(var)

    X = torch.from_numpy(np.concatenate(X_cols, axis=1)).float()

    start_year = int(year[0])
    end_year = int(year[-1])
    start_month = int(month[0])
    end_month = int(month[-1])

    return PredictorData(
        X=X,
        predictor_names=pred_names,
        variables_2predict=meta.get("variables", []),
        model_names=list(model_to_index.keys()),
        year=year,
        month=month,
        start_year=start_year,
        end_year=end_year,
        start_month=start_month,
        end_month=end_month,
    )


def build_predictors_from_spliced_file(
    predictor_file: Union[str, Path],
    meta: dict,
    *,
    model_index_name: str = "ERA5",
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> Tuple[PredictorData, np.ndarray, np.ndarray]:
    """
    Load ERA5-spliced predictors from HDF5 or CSV and assemble PredictorData.
    Returns (predictor_data, year, month) with any trimming applied.
    """
    path = Path(predictor_file)
    if not path.exists():
        raise FileNotFoundError(f"Predictor file not found: {path}")

    if path.suffix.lower() == ".h5":
        import h5py  # local import

        with h5py.File(path, "r") as h5:
            year = np.asarray(h5["year"], dtype=np.int32).ravel()
            month = np.asarray(h5["month"], dtype=np.int32).ravel()
            series = {}
            for var in meta["variables_X"]:
                if var in ("model_index", "month", "time"):
                    continue
                if var in h5:
                    series[var] = np.asarray(h5[var], dtype=np.float32).ravel()
        pdata = _assemble_spliced_predictors_from_arrays(
            year=year,
            month=month,
            series=series,
            meta=meta,
            model_index_name=model_index_name,
            year_start=year_start,
            year_end=year_end,
        )
        assert pdata.year is not None and pdata.month is not None
        return pdata, np.asarray(pdata.year, dtype=np.int32), np.asarray(pdata.month, dtype=np.int32)

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if data.size == 0 or data.dtype.names is None:
        raise RuntimeError(f"Predictor CSV appears empty or malformed: {path}")

    cols = {name: np.asarray(data[name]) for name in data.dtype.names}
    if "year" not in cols or "month" not in cols:
        raise RuntimeError(f"'year'/'month' columns not found in predictor CSV: {path}")

    series = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in cols.items()
        if k not in ("year", "month")
    }
    pdata = _assemble_spliced_predictors_from_arrays(
        year=np.asarray(cols["year"], dtype=np.int32),
        month=np.asarray(cols["month"], dtype=np.int32),
        series=series,
        meta=meta,
        model_index_name=model_index_name,
        year_start=year_start,
        year_end=year_end,
    )
    assert pdata.year is not None and pdata.month is not None
    return pdata, np.asarray(pdata.year, dtype=np.int32), np.asarray(pdata.month, dtype=np.int32)


# %%
# Time-series helpers


def smooth_annual_series(
    years: np.ndarray,
    values: np.ndarray,
    window: int = 21,
) -> np.ndarray:
    """
    Centered running-mean smoothing over a window of years.
    Uses +/- window//2 years around each year (requires odd window).
    """
    years_arr = np.asarray(years, dtype=int).ravel()
    vals_arr = np.asarray(values, dtype=np.float64).ravel()
    if years_arr.size != vals_arr.size:
        raise ValueError("years and values must have the same length")
    if window <= 0:
        raise ValueError("window must be positive")
    if window % 2 == 0:
        raise ValueError("window must be odd for centered smoothing")
    half = window // 2
    smoothed = np.full(vals_arr.shape, np.nan, dtype=np.float64)
    for i, y in enumerate(years_arr):
        mask = (years_arr >= y - half) & (years_arr <= y + half)
        if not np.any(mask):
            continue
        smoothed[i] = float(np.nanmean(vals_arr[mask]))
    return smoothed


# %%
# Grid helpers


def generate_coordinate_grids(
    nlat: int, nlon: int = 360, lon_convention: str = "360", lat_direction: str = "north_to_south"
) -> tuple[np.ndarray, np.ndarray]:
    lat_res = 180.0 / nlat
    lon_res = 360.0 / nlon
    lat_start = 90.0 - lat_res / 2
    lat_end = -90.0 + lat_res / 2
    if lat_direction == "north_to_south":
        lats = np.linspace(lat_start, lat_end, nlat)
    else:
        lats = np.linspace(lat_end, lat_start, nlat)
    if lon_convention == "180":
        lon_start = -180.0 + lon_res / 2
        lon_end = 180.0 - lon_res / 2
    else:
        lon_start = lon_res / 2
        lon_end = 360.0 - lon_res / 2
    lons = np.linspace(lon_start, lon_end, nlon)
    return lats, lons


def model_version_to_code(model_version: str) -> str:
    """
    Convert MODEL_VERSION to version code for naming.
    
    Args:
        model_version: Model version string (e.g., 'NthreeversT1', 'NxlversA5')
    
    Returns:
        Version code: 'v101' for NthreeversT1, 'v100' for NxlversA5, or empty string if not recognized
    """
    if model_version is None:
        return ""
    
    # Normalize: remove 'model_' prefix if present
    normalized = model_version.replace("model_", "").strip()
    
    if normalized == "NxlversA5":
        return "v100"
    elif normalized == "NthreeversT1":
        return "v101"
    else:
        return ""


def runmodus_to_suffix(runmodus: str) -> str:
    """
    Convert USE_RUNMODUSE to suffix for naming template.
    
    Args:
        runmodus: Runmodus string ('all', 'natural', 'aerosol', 'anthropogenic')
    
    Returns:
        Suffix string: '' for 'all', '-nat' for 'natural', '-aer' for 'aerosol', '-anthrop' for 'anthropogenic'
    """
    runmodus = str(runmodus).lower()
    if runmodus == "all":
        return ""
    elif runmodus == "natural":
        return "-nat"
    elif runmodus == "aerosol":
        return "-aer"
    elif runmodus == "anthropogenic":
        return "-anthrop"
    else:
        return ""


# %%
# Output path helpers


def _resolve_version_code(model_version: str) -> str:
    version_code = model_version_to_code(model_version)
    if version_code:
        return version_code
    if model_version.startswith("v"):
        return model_version
    parts = model_version.split(".")
    if len(parts) >= 3:
        return f"v{parts[0]}{parts[1]}{parts[2]}"
    return f"v{model_version.replace('.', '')}"


def _resolve_experiment_id_for_task(
    *,
    scenario: str,
    runmodus: str,
    experiment_override: Optional[str],
) -> str:
    token = experiment_override if (experiment_override is not None and str(experiment_override).strip()) else scenario
    exp, _ = split_experiment_and_runmodus(token, runmodus_hint=runmodus)
    return exp


def resolve_task_output_directory(
    *,
    output_root: Path,
    model_version: str,
    scenario: str,
    workflow: str,
    runmodus: str,
    canonical_layout: bool,
    canonical_experiment_id: Optional[str],
    canonical_n_ensemble: str,
    canonical_kind: str,
    canonical_run_instance: Optional[str],
) -> tuple[Path, str]:
    version_code = _resolve_version_code(model_version)
    if not canonical_layout:
        workflow_dir = str(workflow or "UNKNOWN").upper()
        return Path(output_root) / version_code / scenario / workflow_dir, version_code

    canonical_runmodus = normalize_runmodus_canonical(runmodus)
    experiment_id = _resolve_experiment_id_for_task(
        scenario=scenario,
        runmodus=canonical_runmodus,
        experiment_override=canonical_experiment_id,
    )
    out = build_era5spliced_dataset_path(
        version=version_code,
        experiment_id=experiment_id,
        arx=str(workflow or "AR6").upper(),
        runmodus=canonical_runmodus,
        n_ensemble=canonical_n_ensemble,
        kind=canonical_kind,
        run_instance=canonical_run_instance,
        root=Path(output_root),
    )
    return out, version_code


# %%
# Load GCMagicc meta


def _resolve_run_general(gcmagicc_path: str) -> Path:
    root = Path(gcmagicc_path).expanduser()
    run_general_path = root / "run_general.py"
    if not run_general_path.exists():
        raise FileNotFoundError(f"run_general.py not found in {root}")
    resolved = run_general_path.parent.resolve()
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    return run_general_path


def _load_run_general_sampler(gcmagicc_path: str):
    run_general_path = _resolve_run_general(gcmagicc_path)
    from run_general import sample_from_combined_model as fn  # type: ignore

    print(f"[worker] Using run_general.py from {run_general_path.parent}")
    return fn


def _infer_date_token(sample_fn) -> Optional[str]:
    try:
        sig = inspect.signature(sample_fn)
        return sig.parameters["DATE"].default
    except Exception:
        return None


def load_meta(gcmagicc_path: str, models_subdir: str) -> Tuple[dict, str, str]:
    run_general_path = _resolve_run_general(gcmagicc_path)
    date_token = None
    try:
        from run_general import sample_from_combined_model

        date_token = _infer_date_token(sample_from_combined_model)
    except Exception:
        pass
    models_dir = run_general_path.parent / models_subdir
    if date_token is None:
        cand = sorted(models_dir.glob("meta_*.pkl"))
        if not cand:
            raise FileNotFoundError(f"No meta_*.pkl in {models_dir}")
        meta_path = cand[0]
        date_guess = meta_path.stem.split("meta_")[-1]
    else:
        meta_path = models_dir / f"meta_{date_token}.pkl"
        date_guess = date_token
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} not found")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta, str(meta_path), date_guess


# %%
# Simple time helpers


def infer_time_from_year_month(
    year: np.ndarray, month: np.ndarray
) -> Tuple[int, int, int, int, np.ndarray]:
    sy, sm = int(year[0]), int(month[0])
    ey, em = int(year[-1]), int(month[-1])
    t = np.array(
        [cftime.Datetime360Day(y, m, 15, has_year_zero=True) for y, m in zip(year, month)]
    )
    return sy, sm, ey, em, t


# %%
# Output writers


def _today_stamp() -> str:
    utc = time.gmtime()
    return f"{utc.tm_year:04d}{utc.tm_mon:02d}{utc.tm_mday:02d}-{utc.tm_hour:02d}{utc.tm_min:02d}"


# %%


def save_simple_nc(
    yhval: torch.Tensor,
    variables: List[str],
    year: np.ndarray,
    month: np.ndarray,
    out_path: Path,
    nlat: int = NLAT,
    lon_convention: str = LON_CONVENTION,
    calendar: str = CAL,
    extra_attrs: Optional[Dict[str, object]] = None,
) -> Path:
    sy, sm, ey, em, tcoord = infer_time_from_year_month(year, month)
    if isinstance(yhval, torch.Tensor):
        reg = yhval.detach().cpu().numpy()
    else:
        reg = yhval
    lats, lons = generate_coordinate_grids(nlat, nlat * 2, lon_convention, "north_to_south")
    ds = xr.Dataset(coords={"time": ("time", tcoord), "lat": ("lat", lats), "lon": ("lon", lons)})
    for i, v in enumerate(variables):
        ds[v] = (("time", "lat", "lon"), reg[:, i, :, :])
    if extra_attrs:
        ds.attrs.update({k: _sanitize_attr_value(v) for k, v in extra_attrs.items()})
    ds.to_netcdf(out_path)
    return out_path


# %%
# CMIP6 writer (per variable)


def _infer_activity_id(experiment_id: str) -> str:
    return "ScenarioMIP" if experiment_id.startswith("ssp") else "CMIP"


REALM_FOR_TABLE = {"Amon": "atmos"}
VAR_TO_TABLE = {"pr": "Amon", "tas": "Amon", "sfcWind": "Amon", "psl": "Amon"}

# %%
# Attribute sanitiser for NetCDF writes


def _sanitize_attr_value(val):
    if val is None:
        return ""
    if isinstance(val, (str, float, int, bool)):
        return val
    if isinstance(val, Path):
        return str(val)
    try:
        return float(val)
    except Exception:
        try:
            return str(val)
        except Exception:
            return ""


def workflow_to_f_flag(workflow: str) -> str:
    """
    Map workflow string to CMIP6 member f-flag.
    Default to f1 if workflow is unknown.
    """
    wf = (workflow or "").strip().lower()
    mapping = {
        "ar6": "f1",
        "ar7": "f2",
        "ar6-conc": "f3",
        "ar6_conc": "f3",
        "ar6_concentration": "f3",
        "ar7-conc": "f4",
        "ar7_conc": "f4",
        "ar7_concentration": "f4",
    }
    return mapping.get(wf, "f1")




def save_cmip6_nc(
    yhval: torch.Tensor,
    variables: List[str],
    year: np.ndarray,
    month: np.ndarray,
    *,
    out_root: Path,
    source_id: str,
    experiment_id: str,
    member_id: str,
    grid_label: str = "gr",
    var_to_table: Optional[dict] = None,
    calendar: str = CAL,
    lon_convention: str = LON_CONVENTION,
    nlat: int = NLAT,
    model_version: str = MODEL_VERSION,
    extra_attrs: Optional[Dict[str, object]] = None,
):
    # Build time axis
    sy, sm, ey, em, tcoord = infer_time_from_year_month(year, month)
    if isinstance(yhval, torch.Tensor):
        reg = yhval.detach().cpu().numpy()
    else:
        reg = yhval
    lats, lons = generate_coordinate_grids(nlat, nlat * 2, lon_convention, "north_to_south")
    pstart, pend = tcoord[0], tcoord[-1]
    time_range = f"{pstart.year:04d}{pstart.month:02d}-{pend.year:04d}{pend.month:02d}"
    v2t = var_to_table or VAR_TO_TABLE
    for iv, v in enumerate(variables):
        table_id = v2t.get(v, "Amon")
        attrs = {
            "Conventions": "CF-1.8 CMIP-6.2",
            "title": f"{source_id} output for {experiment_id} ({member_id}) - {v}",
            "creation_date": cftime.DatetimeGregorian(*time.gmtime()[:6]).isoformat() + "Z",
            "product": "model_output",
            "frequency": "mon",
            "realm": REALM_FOR_TABLE.get(table_id, "atmos"),
            "grid_label": grid_label,
            "grid": "data regridded to 1x1 degree grid",
            "nominal_resolution": "1x1degree",
            "version": f"v{_today_stamp()}",
            "history": "Created by 630_run_newscenarios_fromMAGICCinput.py",
        }
        if extra_attrs:
            attrs.update({k: _sanitize_attr_value(val) for k, val in extra_attrs.items()})
        ds = xr.Dataset(
            coords=dict(time=("time", tcoord), lat=("lat", lats), lon=("lon", lons)),
            data_vars={v: (("time", "lat", "lon"), reg[:, iv, :, :])},
            attrs=attrs,
        )
        outdir = (
            out_root / "CMIP6" / source_id / experiment_id / member_id / table_id / v / grid_label
        )
        outdir.mkdir(parents=True, exist_ok=True)
        fname = (
            f"{v}_{table_id}_{source_id}_{experiment_id}_{member_id}_{grid_label}_{time_range}.nc"
        )
        fpath = outdir / fname
        ds.to_netcdf(fpath)


# %%
# Model runner (run_general with GPU-first behaviour, CPU fallback)


def run_gcmagicc(
    sample_fn,
    predictor_data: PredictorData,
    *,
    dependence: bool,
    usebias_model=None,
    useeffect_model=None,
    device: Optional[str],
    models_dir: Path,
    date_token: Optional[str],
    force_gpu: bool = False,
    amp: bool = False,
    seed: Optional[int] = None,
):
    requested_device = normalize_device_string(device) or DEFAULT_DEVICE
    use_gpu = _device_is_cuda(requested_device)

    if use_gpu and not _cuda_available() and not force_gpu:
        print(f"[worker] ⚠️  Requested GPU '{requested_device}' but torch.cuda.is_available() is False; falling back to CPU.")
        requested_device = "cpu"
        use_gpu = False
    elif use_gpu and not _cuda_available() and force_gpu:
        print(f"[worker] ⚠️  Forcing GPU usage for '{requested_device}' despite torch.cuda.is_available() being False.")

    asnumpy_flag = not use_gpu
    x_input = predictor_data.X
    if isinstance(x_input, torch.Tensor):
        if not use_gpu:
            x_input = x_input.detach().cpu()
    else:
        x_input = torch.from_numpy(np.asarray(x_input))
        if not use_gpu:
            x_input = x_input.cpu()

    kwargs = {
        "x": x_input,
        "device": requested_device,
        "dirname": str(models_dir) + "/",
        "dependence": dependence,
        "rectangular": True,
        "asnumpy": asnumpy_flag,
    }
    if date_token is not None:
        kwargs["DATE"] = date_token

    sig = inspect.signature(sample_fn)
    if "usebias_model" in sig.parameters:
        kwargs["usebias_model"] = usebias_model
    if "useeffect_model" in sig.parameters:
        kwargs["useeffect_model"] = useeffect_model
    if "seed" in sig.parameters and seed is not None:
        kwargs["seed"] = seed
    if "nlat" in sig.parameters:
        kwargs.setdefault("nlat", NLAT)
    if "nsub" in sig.parameters:
        kwargs.setdefault("nsub", 1)
    if "nside" in sig.parameters:
        kwargs.setdefault("nside", 64)

    if use_gpu:
        if "memoryoptimized" in sig.parameters:
            kwargs.setdefault("memoryoptimized", True)
        if "enable_gpu_optimizations" in sig.parameters:
            kwargs.setdefault("enable_gpu_optimizations", True)
    else:
        if "memoryoptimized" in sig.parameters:
            kwargs.setdefault("memoryoptimized", False)
        if "enable_gpu_optimizations" in sig.parameters:
            kwargs.setdefault("enable_gpu_optimizations", False)

    if use_gpu and MIN_FREE_MEMORY_GB > 0 and not force_gpu:
        try:
            idx_check = int(requested_device.split(":", 1)[1]) if ":" in requested_device else 0
        except Exception:
            idx_check = 0
        try:
            free_bytes, _ = torch.cuda.mem_get_info(idx_check)
            free_gb = free_bytes / (1024**3)
            if free_gb < MIN_FREE_MEMORY_GB:
                print(
                    f"[worker] ⚠️  GPU {idx_check} has {free_gb:.2f}GiB free (<{MIN_FREE_MEMORY_GB}GiB); switching to CPU."
                )
                requested_device = "cpu"
                use_gpu = False
                kwargs["device"] = "cpu"
                kwargs["asnumpy"] = True
                kwargs["memoryoptimized"] = False
                kwargs["enable_gpu_optimizations"] = False
        except Exception:
            pass

    if use_gpu:
        _maybe_enable_tf32(DEFAULT_TF32)
        try:
            idx = int(requested_device.split(":", 1)[1]) if ":" in requested_device else torch.cuda.current_device()
        except Exception:
            idx = 0
        try:
            torch.cuda.set_device(idx)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(idx)
        except Exception:
            pass

        if amp:
            amp_dtype = _amp_dtype_cuda()
            with torch.autocast(device_type="cuda", dtype=amp_dtype), torch.inference_mode():
                out = sample_fn(**kwargs)
        else:
            with torch.inference_mode():
                out = sample_fn(**kwargs)

        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
        elif hasattr(out, "cpu"):
            out = out.cpu().numpy()
        try:
            torch.cuda.synchronize(idx)
            torch.cuda.empty_cache()
        except Exception:
            pass
    else:
        with torch.inference_mode():
            out = sample_fn(**kwargs)
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()

    return out


# %% [markdown]
# ## 2) Worker: run a single ensemble member (separate process)

# %%
WORKER_FLAG = "--as-worker"

# %%


def run_single_worker(
    *,
    scenario: str,
    ensemble_id: int,
    workflow: str,
    output_root: Path,
    output_mode: str,
    naming_template: str,
    dependence: bool,
    device: Optional[str],
    nlat: int,
    lon_convention: str,
    calendar: str,
    model_version: str,
    gcmagicc_path: str,
    models_subdir: str,
    resume: bool = True,
    predictor_path: Optional[str] = None,
    magicc_run_id: Optional[int] = None,
    bias_to_era5: Optional[bool] = None,
    seed_override: Optional[int] = None,
    runmodus: str = "all",
    predictor_source_scenario: Optional[str] = None,
    predictor_year_start: Optional[int] = None,
    predictor_year_end: Optional[int] = None,
    canonical_layout: bool = False,
    canonical_experiment_id: Optional[str] = None,
    canonical_n_ensemble: str = "n_20",
    canonical_kind: str = CANONICAL_KIND_ORIGINAL,
    canonical_run_instance: Optional[str] = None,
) -> int:
    """Return 0 on success; non‑zero otherwise."""
    try:
        # Ensure CUDA allocator is set before any torch CUDA usage
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        device_resolved = normalize_device_string(device) or DEFAULT_DEVICE
        if _device_is_cuda(device_resolved) and not _cuda_available() and not _FORCE_GPU:
            print("[worker] ⚠️  CUDA requested but torch.cuda.is_available() is False; downgrading to CPU.")
            device_resolved = "cpu"
        amp_flag = bool(DEFAULT_AMP and _device_is_cuda(device_resolved))

        run_general_path = _resolve_run_general(gcmagicc_path)
        sample_from_combined_model = _load_run_general_sampler(gcmagicc_path)
        date_token = _infer_date_token(sample_from_combined_model)
        models_dir = run_general_path.parent / models_subdir

        # Load predictors from ERA5-spliced file
        if predictor_path is None:
            print("[worker] Missing predictor_path for MAGICCxERA5 run")
            return 2
        meta_d, _, _ = load_meta(str(run_general_path.parent), models_subdir)
        if predictor_source_scenario and predictor_source_scenario != scenario:
            print(
                f"[worker] Using predictors from '{predictor_source_scenario}' for requested scenario '{scenario}'"
            )
        if predictor_year_end is not None or predictor_year_start is not None:
            yr_msg_start = predictor_year_start if predictor_year_start is not None else "start"
            yr_msg_end = predictor_year_end if predictor_year_end is not None else "end"
            print(f"[worker] Trimming predictor years to [{yr_msg_start}, {yr_msg_end}]")
        pdata, year, month = build_predictors_from_spliced_file(
            predictor_path,
            meta_d,
            model_index_name="ERA5",
            year_start=predictor_year_start,
            year_end=predictor_year_end,
        )
        usebias_model = 0 if (BIASCORRECT_TO_ERA5_D if bias_to_era5 is None else bias_to_era5) else None
        useeffect_model = 0
        magicc_ens_flag = magicc_run_id if magicc_run_id is not None else 0

        # Run the model (run_general; GPU preferred)
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
        yhval = run_gcmagicc(
            sample_fn=sample_from_combined_model,
            predictor_data=pdata,
            dependence=dependence,
            usebias_model=usebias_model,
            useeffect_model=useeffect_model,
            device=device_resolved,
            models_dir=models_dir,
            date_token=date_token,
            force_gpu=_FORCE_GPU,
            amp=amp_flag,
            seed=int(seed_override if seed_override is not None else ensemble_id),
        )

        # Save outputs
        dateflag = _today_stamp()
        depflag = "d1" if dependence else "d0"
        version_code = _resolve_version_code(model_version)
        # Remove 'v' prefix for template (version flag)
        version_flag = version_code[1:] if version_code.startswith("v") else version_code
        
        # Format bias and effect flags
        if usebias_model is None:
            bias_flag = "N"
        elif usebias_model == 0:
            bias_flag = "0"  # ERA5
        else:
            bias_flag = str(usebias_model)
            
        if useeffect_model is None:
            effect_flag = "N"
        elif useeffect_model == 0:
            effect_flag = "0"  # ERA5
        else:
            effect_flag = str(useeffect_model)

        f_flag = workflow_to_f_flag(workflow)
        # Format ensemble ID as CMIP6 member ID (e.g., r1i1p1f1)
        # If ensemble_id is just a number, convert to r{id}i1p1f1 format
        if isinstance(ensemble_id, (int, str)) and str(ensemble_id).isdigit():
            ens_flag = f"r{ensemble_id}i1p1{f_flag}"
        else:
            ens_flag = str(ensemble_id)  # Use as-is if already formatted
        
        # Get runmodus suffix
        runmodus_suffix = runmodus_to_suffix(runmodus)
        
        fname_root = naming_template.format(
            version=version_flag,
            dep=depflag,
            bias=bias_flag,
            effect=effect_flag,
            magicc_ens=magicc_ens_flag,
            date=dateflag,
            scenario=scenario,
            runmodus=runmodus_suffix,
            ens=ens_flag,
        )
        canonical_kind_token = str(canonical_kind or CANONICAL_KIND_ORIGINAL).strip().lower()
        if canonical_kind_token not in set(CANONICAL_KIND_CHOICES):
            raise ValueError(
                f"Unsupported canonical kind '{canonical_kind_token}'. "
                f"Choose one of: {', '.join(CANONICAL_KIND_CHOICES)}."
            )
        scen_out, _ = resolve_task_output_directory(
            output_root=Path(output_root),
            model_version=model_version,
            scenario=scenario,
            workflow=str(workflow or "UNKNOWN"),
            runmodus=runmodus,
            canonical_layout=bool(canonical_layout),
            canonical_experiment_id=canonical_experiment_id,
            canonical_n_ensemble=canonical_n_ensemble,
            canonical_kind=canonical_kind_token,
            canonical_run_instance=canonical_run_instance,
        )
        scen_out.mkdir(parents=True, exist_ok=True)

        if output_mode == "simple":
            outfile = scen_out / f"{fname_root}.nc"
            if resume and outfile.exists():
                print(f"[worker] Skipping existing {outfile.name}")
                return 0
            save_simple_nc(
                yhval,
                pdata.variables_2predict,
                year,
                month,
                outfile,
                nlat=nlat,
                lon_convention=lon_convention,
                calendar=calendar,
            )
            print(f"[worker] ✔ wrote {outfile}")
        elif output_mode == "cmip6":
            # infer identifiers
            source_id = f"MLMAGICC_{model_version}"
            experiment_id = scenario if scenario.startswith("ssp") else f"custom-{scenario}"
            member_id = f"r{int(ensemble_id)}i1p1{f_flag}"
            save_cmip6_nc(
                yhval,
                pdata.variables_2predict,
                year,
                month,
                out_root=scen_out,
                source_id=source_id,
                experiment_id=experiment_id,
                member_id=member_id,
                nlat=nlat,
                lon_convention=lon_convention,
                model_version=model_version,
            )
            print(f"[worker] ✔ wrote CMIP6 layout under {scen_out}")
        else:
            raise ValueError(f"Unknown output_mode={output_mode}")

        # Cleanup
        del yhval, pdata
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return 0
    except Exception as e:
        print(f"[worker] ERROR: {e}")
        return 1


# %% [markdown]
# ## 3) Orchestrator (parent): parallel, CPU pinning, resume, CLI overrides

# %%
# CPU slice utility (like in 310_)


def _core_slice(
    worker_id: int, total_workers: int, total_cpus: Optional[int] = None
) -> tuple[str, int]:
    n = total_cpus or (os.cpu_count() or 128)
    total_workers = max(1, total_workers)
    per = max(1, n // total_workers)
    start = worker_id * per
    end = (n - 1) if (worker_id == total_workers - 1) else min(n - 1, start + per - 1)
    if start > end:
        start = end
    return f"{start}-{end}", (end - start + 1)


# %%
def build_spliced_tasks(
    spliced_root: Path,
    scenarios: Optional[List[str]],
    draws_spec: Union[str, int],
    test_one: bool,
    workflow: str = "AR6",
    runmodus: str = "all",
    *,
    n_value: Optional[int] = None,
    variant: Optional[str] = None,
    variant_glob: str = SPLICED_VARIANT_GLOB,
    strict: bool = True,
) -> List[Dict[str, object]]:
    """
    Build work items for MAGICCxERA5 (ERA5-spliced predictors) ensembles.
    
    Args:
        spliced_root: Base directory for ERA5-spliced predictors
        scenarios: List of scenario names (None -> auto-discover)
        draws_spec: Specification for number of draws ('all', 'first', or integer)
        test_one: If True, run only one ensemble per scenario
        workflow: 'AR6' | 'AR7' | 'all' or comma-separated list
        runmodus: 'all' | 'natural' | 'aerosol' | 'anthropogenic' or comma-separated list
        n_value: n for n_* (new layout only)
        variant: explicit magicc_based_predictors_* path or name (new layout only)
        variant_glob: glob for magicc_based_predictors_* (new layout only)
        strict: if True, raise on missing scenarios/runmodus; otherwise skip missing
    """
    workflows = normalize_workflow_list(workflow)
    runmodus_list = normalize_runmodus_list(runmodus)
    n_value_use = n_value if n_value is not None else SPLICED_N
    variant_use = variant if variant else (SPLICED_VARIANT or None)
    if scenarios is None or len(scenarios) == 0:
        scenarios = discover_spliced_scenarios(
            spliced_root,
            workflows=workflows,
            runmodus_list=runmodus_list,
            n_value=n_value_use,
            variant=variant_use,
            variant_glob=variant_glob,
        )
    else:
        if any(_has_glob_pattern(str(s)) for s in scenarios):
            available = discover_spliced_scenarios(
                spliced_root,
                workflows=workflows,
                runmodus_list=runmodus_list,
                n_value=n_value_use,
                variant=variant_use,
                variant_glob=variant_glob,
            )
            expanded, missing = expand_scenario_globs(scenarios, available)
            if missing:
                msg = (
                    "No scenarios match glob pattern(s): "
                    f"{', '.join(missing)} in {spliced_root} "
                    f"(workflows={','.join(workflows)}, runmodus={','.join(runmodus_list)})"
                )
                if strict:
                    raise FileNotFoundError(msg)
                print(f"⚠️  {msg}")
            if expanded:
                if len(expanded) <= 12:
                    print(f"🔍 Expanded scenario globs to {len(expanded)} scenario(s): {', '.join(expanded)}")
                else:
                    print(f"🔍 Expanded scenario globs to {len(expanded)} scenario(s).")
            scenarios = expanded
    scenario_specs = []
    for s in scenarios:
        scen_norm = _normalize_scenario_name(s)
        src_scen, y_start, y_end = resolve_scenario_request(scen_norm)
        scenario_specs.append(
            {
                "requested": scen_norm,
                "source": src_scen,
                "year_start": y_start,
                "year_end": y_end,
            }
        )
    if not scenario_specs:
        return []

    source_scenarios = sorted({spec["source"] for spec in scenario_specs})
    tasks: List[Dict[str, object]] = []

    for wf in workflows:
        for rm in runmodus_list:
            scen_map = discover_spliced_predictor_files(
                spliced_root,
                source_scenarios,
                workflow=wf,
                runmodus=rm,
                n_value=n_value,
                variant=variant,
                variant_glob=variant_glob,
                strict=strict,
            )
            for scen_spec in scenario_specs:
                run_map = scen_map.get(scen_spec["source"], {})
                run_ids = sorted(run_map.keys())
                if not run_ids:
                    scen_norm = scen_spec["requested"]
                    source = scen_spec["source"]
                    hint = ""
                    if source != scen_norm:
                        hint = f" (using predictors from {source})"
                    print(
                        f"⚠️  No ERA5-spliced runs found for scenario {scen_norm}{hint} "
                        f"in {spliced_root} (workflow={wf}, runmodus={rm})"
                    )
                    continue

                if test_one:
                    chosen = run_ids[:1]
                else:
                    if isinstance(draws_spec, str):
                        draws_mode = draws_spec.strip().lower()
                        if draws_mode == "all":
                            chosen = run_ids
                        elif draws_mode == "first":
                            chosen = run_ids[:1]
                        else:
                            try:
                                n_draws = int(draws_mode)
                            except ValueError:
                                n_draws = ENSEMBLES_D
                            n_draws = max(1, n_draws)
                            if n_draws <= len(run_ids):
                                chosen = random.sample(run_ids, k=n_draws)
                            else:
                                chosen = random.choices(run_ids, k=n_draws)
                    else:
                        n_draws = max(1, int(draws_spec))
                        chosen = random.sample(run_ids, k=n_draws) if n_draws <= len(run_ids) else random.choices(run_ids, k=n_draws)

                for idx, rid in enumerate(chosen):
                    tasks.append(
                        {
                            "scenario": scen_spec["requested"],
                            "ensemble_id": idx + 1,
                            "run_id": int(rid),
                            "predictor_path": str(run_map[rid]),
                            "seed_override": int(rid + idx + 1),
                            "runmodus": rm,
                            "workflow": wf,
                            "predictor_source_scenario": scen_spec["source"],
                            "predictor_year_start": scen_spec["year_start"],
                            "predictor_year_end": scen_spec["year_end"],
                        }
                    )
    return tasks


# %%
# CLI
import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run GCMagicc v6 from MAGICC-derived predictors (robust orchestrator)"
    )
    p.add_argument(
        "--scenarios", nargs="*", default=None, help="Scenarios to run (default: auto-discover)"
    )
    p.add_argument(
        "--scenario-whitelist",
        type=str,
        default=None,
        help="Comma-separated regex patterns to filter scenarios (e.g., 'ssp.*,ssp245' or '^ssp'). Applied after auto-discovery.",
    )
    p.add_argument(
        "--ensembles",
        type=str,
        default=None,
        help="Ensemble spec: number/all/first for MAGICCxERA5 draws.",
    )
    p.add_argument("--test-one", action="store_true", help="Run only one ensemble per scenario")
    p.add_argument(
        "--era5-spliced-dir",
        type=str,
        default=None,
        help="Base directory containing magicc_based_predictors_* (or legacy ERA5-spliced layout).",
    )
    p.add_argument(
        "--workflow",
        type=str,
        default=None,
        help="Workflow selection: AR6, AR7, or comma list (e.g., AR6,AR7). 'all' means both.",
    )
    p.add_argument(
        "--runmodus",
        type=str,
        default=None,
        help="Runmodus selection: all, natural, aerosol, anthropogenic, or comma list (e.g., all,natural).",
    )
    p.add_argument(
        "--spliced-n",
        type=int,
        default=None,
        help="n for n_* in magicc_based_predictors (default: from config/env).",
    )
    p.add_argument(
        "--spliced-variant",
        type=str,
        default=None,
        help="Specific magicc_based_predictors_* folder name or full path (default: auto-pick).",
    )
    p.add_argument(
        "--spliced-variant-glob",
        type=str,
        default=None,
        help="Glob for magicc_based_predictors_* folders (default: from config/env).",
    )
    p.add_argument("--bias-to-era5", action="store_true", help="Force bias correction to ERA5 for MAGICCxERA5.")
    p.add_argument("--no-bias-to-era5", action="store_true", help="Disable bias correction to ERA5 for MAGICCxERA5.")
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument(
        "--canonical-layout",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Write outputs using canonical ERA5spliced layout "
            "(<version>/<experiment>/<ARX>/<runmodus>/<n_ensemble>/<kind>). "
            "Enabled by default."
        ),
    )
    p.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Canonical experiment id override. Default resolves from scenario/runmodus.",
    )
    p.add_argument(
        "--n-ensemble-label",
        type=str,
        default=None,
        help="Canonical n_ensemble label (e.g., n_20, n_100). Default derives from --spliced-n.",
    )
    p.add_argument(
        "--canonical-kind",
        type=str,
        default=None,
        choices=list(CANONICAL_KIND_CHOICES),
        help="Canonical kind folder (default: original).",
    )
    p.add_argument(
        "--run-instance",
        type=str,
        default=None,
        help="Optional run-instance suffix under canonical kind.",
    )
    p.add_argument("--output-mode", type=str, choices=["simple", "cmip6"], default=None)
    p.add_argument("--name-template", type=str, default=None)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dependence", type=str, choices=["true", "false"], default=None)
    p.add_argument("--no-cpu-pin", action="store_true")
    p.add_argument("--no-nice", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--auto-consolidate",
        dest="auto_consolidate",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUTO_CONSOLIDATE,
        help="Run scoped 2018 autoconsolidate after successful writes (default: enabled).",
    )
    p.add_argument(
        "--auto-consolidate-config",
        type=str,
        default=None,
        help="Path to 2018 consolidation config JSON (default: gcmmagicc/scripts/2018_consolidate_era5spliced_s3.example.json).",
    )
    p.add_argument(
        "--auto-consolidate-cleanup-local",
        dest="auto_consolidate_cleanup_local",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUTO_CONSOLIDATE_CLEANUP_LOCAL,
        help="Allow local cleanup after verified upload in autoconsolidate (default: enabled).",
    )
    # Internal: worker mode
    p.add_argument(WORKER_FLAG, action="store_true")
    p.add_argument("--worker-args", type=str, default=None, help="JSON with worker kwargs")
    return p


# %%
# Orchestrate with subprocess for true isolation


def launch_worker_subprocess(
    worker_id: int,
    total_workers: int,
    *,
    worker_kwargs: dict,
    cpu_pin: bool = True,
    nice_io: bool = True,
    async_mode: bool = False,
) -> int | subprocess.Popen:
    exe = sys.executable
    cmd = [exe, __file__, WORKER_FLAG, "--worker-args", json.dumps(worker_kwargs)]
    env = os.environ.copy()
    # Threads per run
    tpr = min(16, max(1, THREADS_PER_RUN))
    env.setdefault("THREADS_PER_RUN", str(tpr))
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM",
        "NUMEXPR_MAX_THREADS",
        "TORCH_NUM_THREADS",
    ):
        env.setdefault(k, str(tpr))
    env.setdefault("KMP_BLOCKTIME", "0")
    env.setdefault("OMP_DYNAMIC", "FALSE")
    # Set CUDA memory allocation strategy to avoid fragmentation
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    wrapper: List[str] = []
    if nice_io:
        wrapper += ["nice", "-n", "19", "ionice", "-c2", "-n5"]
    if cpu_pin:
        cores, per = _core_slice(worker_id, total_workers)
        wrapper += ["taskset", "-c", cores]
        print(f"[orchestrator] worker {worker_id} -> cores {cores} (threads {tpr})")
    full_cmd = wrapper + cmd
    if DRY_RUN:
        print("DRY-RUN:", " ".join(shlex.quote(c) for c in full_cmd))
        return 0
    if async_mode:
        try:
            return subprocess.Popen(full_cmd, env=env)  # type: ignore[return-value]
        except Exception as e:
            print(f"subprocess failed to start: {e}")
            return 1
    try:
        cp = subprocess.run(full_cmd, env=env, check=False)
        return cp.returncode
    except Exception as e:
        print(f"subprocess failed: {e}")
        return 1


def _wait_for_one(
    active: List[tuple[subprocess.Popen, str]],
    *,
    block: bool = True,
) -> Optional[int]:
    """
    Wait for the next finished worker in the active list.
    Returns the exit code, or None if nothing finished and block=False.
    """
    while active:
        for i, (proc, label) in enumerate(active):
            rc = proc.poll()
            if rc is None:
                continue
            active.pop(i)
            if rc != 0:
                print(f"[worker {label}] exited with code {rc}")
            return rc
        if not block:
            return None
        time.sleep(0.5)
    return None
    try:
        cp = subprocess.run(full_cmd, env=env, check=False)
        return cp.returncode
    except Exception as e:
        print(f"subprocess failed: {e}")
        return 1


# %% [markdown]
# ## 4) Main

# %%
if __name__ == "__main__":
    parser = build_arg_parser()
    # Use parse_known_args to ignore Jupyter kernel arguments when running in notebook
    try:
        args, _ = parser.parse_known_args()
    except SystemExit:
        # If argparse fails (e.g., in notebook with kernel args), use defaults
        args = argparse.Namespace(
            scenarios=None,
            scenario_whitelist=None,
            ensembles=None,
            test_one=False,
            era5_spliced_dir=None,
            workflow=None,
            runmodus=None,
            spliced_n=None,
            spliced_variant=None,
            spliced_variant_glob=None,
            bias_to_era5=False,
            no_bias_to_era5=False,
            output_root=None,
            canonical_layout=None,
            experiment_id=None,
            n_ensemble_label=None,
            canonical_kind=None,
            run_instance=None,
            output_mode=None,
            name_template=None,
            max_workers=None,
            device=None,
            dependence=None,
            no_cpu_pin=False,
            no_nice=False,
            resume=False,
            no_resume=False,
            dry_run=False,
            auto_consolidate=DEFAULT_AUTO_CONSOLIDATE,
            auto_consolidate_config=None,
            auto_consolidate_cleanup_local=DEFAULT_AUTO_CONSOLIDATE_CLEANUP_LOCAL,
            as_worker=False,
            worker_args=None,
        )

    # Worker mode (child process)
    if getattr(args, WORKER_FLAG.lstrip("-").replace("-", "_")):
        wk = json.loads(args.worker_args)
        rc = run_single_worker(
            scenario=wk["scenario"],
            ensemble_id=wk["ensemble_id"],
            workflow=wk.get("workflow", USE_WORKFLOW),
            output_root=Path(wk["output_root"]),
            output_mode=wk["output_mode"],
            naming_template=wk["naming_template"],
            dependence=wk["dependence"],
            device=wk["device"],
            nlat=wk["nlat"],
            lon_convention=wk["lon_convention"],
            calendar=wk["calendar"],
            model_version=wk["model_version"],
            gcmagicc_path=wk["gcmagicc_path"],
            models_subdir=wk["models_subdir"],
            resume=wk["resume"],
            predictor_path=wk.get("predictor_path"),
            magicc_run_id=wk.get("magicc_run_id"),
            bias_to_era5=wk.get("bias_to_era5"),
            seed_override=wk.get("seed_override"),
            runmodus=wk.get("runmodus", USE_RUNMODUSE),
            predictor_source_scenario=wk.get("predictor_source_scenario"),
            predictor_year_start=wk.get("predictor_year_start"),
            predictor_year_end=wk.get("predictor_year_end"),
            canonical_layout=bool(wk.get("canonical_layout", False)),
            canonical_experiment_id=wk.get("canonical_experiment_id"),
            canonical_n_ensemble=wk.get("canonical_n_ensemble", "n_20"),
            canonical_kind=wk.get("canonical_kind", CANONICAL_KIND_ORIGINAL),
            canonical_run_instance=wk.get("canonical_run_instance"),
        )
        sys.exit(rc)

    # Orchestrator mode
    # Merge CLI overrides into config
    spliced_root = Path(args.era5_spliced_dir) if args.era5_spliced_dir else Path(ERA5_SPLICED_PREDICTOR_DIR)
    requested_output_root = Path(args.output_root) if args.output_root else OUTPUT_ROOT
    output_root = resolve_writable_output_root(
        requested_output_root,
        context="321 output",
        allow_fallback=(args.output_root is None),
    )
    output_mode = args.output_mode or OUTPUT_MODE
    canonical_layout = DEFAULT_CANONICAL_LAYOUT if args.canonical_layout is None else bool(args.canonical_layout)
    canonical_kind = (
        str(args.canonical_kind or DEFAULT_CANONICAL_KIND or CANONICAL_KIND_ORIGINAL).strip().lower()
    )
    if canonical_kind not in set(CANONICAL_KIND_CHOICES):
        print(
            f"Unsupported canonical kind '{canonical_kind}'. "
            f"Choose one of: {', '.join(CANONICAL_KIND_CHOICES)}."
        )
        sys.exit(2)
    name_tpl = args.name_template or NAMING_TEMPLATE
    max_workers_user = args.max_workers
    max_workers = max_workers_user or MAX_WORKERS
    device = normalize_device_string(args.device) or DEVICE
    if _device_is_cuda(device) and not _cuda_available() and not _FORCE_GPU:
        print(f"⚠️  Requested CUDA device '{device}' but torch.cuda.is_available() is False; using CPU instead.")
        device = "cpu"
    elif _device_is_cuda(device) and (_cuda_available() or _FORCE_GPU):
        if not _gpu_preflight(device, warn=not _FORCE_GPU) and not _FORCE_GPU:
            print("⚠️  GPU preflight failed; switching to CPU.")
            device = "cpu"
        else:
            _maybe_enable_tf32(DEFAULT_TF32)
            report_device_status(device)
    # Auto-tune worker count for CPU-only runs (20% safety buffer, ~40GB per job)
    if device == "cpu" and max_workers_user is None:
        max_workers = _estimate_cpu_worker_count(
            per_job_cpus=7,
            per_job_ram_gb=40.0,
            safety_fraction=0.20,
        )
        total_mem_gb, avail_mem_gb = _memory_gb()
        mem_str = ""
        if total_mem_gb > 0:
            mem_str = f"; RAM≈{total_mem_gb:.1f}GB (avail≈{avail_mem_gb:.1f}GB)"
        print(
            f"🧠 CPU parallel planner: {max_workers} worker(s) "
            f"(~7 cores & 40GB per job, 20% safety buffer{mem_str})"
        )
    max_workers = max(1, int(max_workers))
    dependence = DEPENDENCE if (args.dependence is None) else (args.dependence == "true")
    cpu_pin = False if args.no_cpu_pin else CPU_PIN
    nice_io = False if args.no_nice else NICE_IO
    resume_flag = RESUME
    if args.resume:
        resume_flag = True
    if args.no_resume:
        resume_flag = False
    dry_run = DRY_RUN or args.dry_run

    test_one = args.test_one or TEST_ONE
    draws_spec: Union[str, int] = args.ensembles if args.ensembles is not None else ENSEMBLES_D
    bias_to_era5_flag = BIASCORRECT_TO_ERA5_D
    if args.bias_to_era5:
        bias_to_era5_flag = True
    if args.no_bias_to_era5:
        bias_to_era5_flag = False
    
    # Get workflow and runmodus (used for MAGICCxERA5)
    workflow = args.workflow if args.workflow is not None else USE_WORKFLOW
    runmodus = args.runmodus if args.runmodus is not None else USE_RUNMODUSE
    spliced_n = args.spliced_n if args.spliced_n is not None else SPLICED_N
    spliced_variant = args.spliced_variant if args.spliced_variant is not None else (SPLICED_VARIANT or None)
    spliced_variant_glob = (
        args.spliced_variant_glob if args.spliced_variant_glob is not None else SPLICED_VARIANT_GLOB
    )
    n_ensemble_token_raw = args.n_ensemble_label if args.n_ensemble_label is not None else f"n_{spliced_n}"
    try:
        canonical_n_ensemble = normalize_n_ensemble_label(n_ensemble_token_raw)
    except ValueError as exc:
        print(f"Invalid canonical n_ensemble label '{n_ensemble_token_raw}': {exc}")
        sys.exit(2)
    canonical_run_instance = args.run_instance if args.run_instance is not None else None
    try:
        workflows = normalize_workflow_list(workflow)
        runmodus_list = normalize_runmodus_list(runmodus)
    except ValueError as exc:
        print(f"Invalid workflow/runmodus selection: {exc}")
        sys.exit(2)

    # Parse scenario whitelist patterns
    scenario_whitelist_patterns = None
    if args.scenario_whitelist:
        scenario_whitelist_patterns = _split_csv_values(args.scenario_whitelist)
    elif SCENARIO_WHITELIST:
        scenario_whitelist_patterns = SCENARIO_WHITELIST

    # Build task list for MAGICCxERA5
    scen_candidates = args.scenarios or SCENARIOS
    scenarios = [_normalize_scenario_name(s) for s in scen_candidates] if scen_candidates else None
    
    # If scenarios need to be auto-discovered, discover them first
    if scenarios is None:
        scenarios = discover_spliced_scenarios(
            spliced_root,
            workflows=workflows,
            runmodus_list=runmodus_list,
            n_value=spliced_n,
            variant=spliced_variant,
            variant_glob=spliced_variant_glob,
        )
        if scenario_whitelist_patterns:
            scenarios_before_filter = len(scenarios)
            scenarios = filter_scenarios_by_patterns(scenarios, scenario_whitelist_patterns)
            if scenarios_before_filter > len(scenarios):
                print(
                    f"🔍 Scenario whitelist filtered {scenarios_before_filter} scenarios → {len(scenarios)} "
                    f"(patterns: {', '.join(scenario_whitelist_patterns)})"
                )
    elif scenario_whitelist_patterns:
        # Apply whitelist even if scenarios were explicitly provided
        scenarios_before_filter = len(scenarios)
        scenarios = filter_scenarios_by_patterns(scenarios, scenario_whitelist_patterns)
        if scenarios_before_filter > len(scenarios):
            print(
                f"🔍 Scenario whitelist filtered {scenarios_before_filter} scenarios → {len(scenarios)} "
                f"(patterns: {', '.join(scenario_whitelist_patterns)})"
            )
    if not scenarios:
        print("❌ No scenarios to run after filtering. Check scenario whitelist patterns or predictor directories.")
        sys.exit(2)
    
    strict_lookup = bool(scen_candidates) and len(workflows) == 1 and len(runmodus_list) == 1
    tasks = build_spliced_tasks(
        spliced_root,
        scenarios,
        draws_spec,
        test_one,
        workflow=workflows,
        runmodus=runmodus_list,
        n_value=spliced_n,
        variant=spliced_variant,
        variant_glob=spliced_variant_glob,
        strict=strict_lookup,
    )
    if scen_candidates:
        found = {t.get("scenario") for t in tasks if isinstance(t, dict)}
        missing = [s for s in scenarios if s not in found] if scenarios else []
        if missing:
            print(
                f"No ERA5-spliced predictors found for scenarios {missing} in {spliced_root} "
                f"(workflows={workflows}, runmodus={runmodus_list}, n={spliced_n})"
            )
            if strict_lookup:
                sys.exit(2)
    if not tasks:
        print("No MAGICCxERA5 tasks to run (check predictors/draws)")
        sys.exit(0)

    # Meta load once just for a sanity print
    meta, meta_path, DATE = load_meta(GCMagiccpath, MODELS_SUBDIR)
    print(f"Loaded meta {meta_path}; variables -> {meta.get('variables', [])}")

    # Launch workers in a simple round‑robin across worker slots
    n_tasks = len(tasks)
    workflow_msg = f", workflow={','.join(workflows)}, runmodus={','.join(runmodus_list)}, n={spliced_n}"
    canonical_msg = (
        f", canonical_layout={canonical_layout}, kind={canonical_kind}, n_ensemble={canonical_n_ensemble}"
    )
    print(
        f"Planning to run {n_tasks} tasks with {max_workers} workers; "
        f"source=MAGICCxERA5, output mode={output_mode}, bias_to_era5={bias_to_era5_flag}"
        f"{workflow_msg}{canonical_msg}"
    )

    # Launch loop with bounded process concurrency (each worker runs PASS1→PASS2 sequentially)
    active: List[tuple[subprocess.Popen, str]] = []
    exit_codes: List[int] = []

    for idx, task in enumerate(tasks):
        scen = str(task.get("scenario"))
        ens_id = int(task.get("ensemble_id"))
        predictor_path = task.get("predictor_path")
        magicc_run_id = task.get("run_id")
        seed_override = task.get("seed_override")
        task_runmodus = str(task.get("runmodus") or (runmodus_list[0] if runmodus_list else USE_RUNMODUSE))
        task_workflow = str(task.get("workflow") or (workflows[0] if workflows else USE_WORKFLOW))
        task_experiment_id = _resolve_experiment_id_for_task(
            scenario=scen,
            runmodus=task_runmodus,
            experiment_override=args.experiment_id,
        )
        # Skip if resume & output already present
        if resume_flag and output_mode == "simple":
            # Match worker naming (with wildcards for unknown parts) to spot existing outputs
            version_code = _resolve_version_code(MODEL_VERSION)
            version_flag = version_code[1:] if version_code.startswith("v") else version_code
            depflag = "d1" if dependence else "d0"
            if isinstance(ens_id, (int, str)) and str(ens_id).isdigit():
                ens_flag = f"r{ens_id}i1p1{workflow_to_f_flag(task_workflow)}"
            else:
                ens_flag = str(ens_id)
            runmodus_suffix = runmodus_to_suffix(task_runmodus)
            fname_root = name_tpl.format(
                version=version_flag,
                dep=depflag,
                bias="*",
                effect="*",
                magicc_ens="*",
                date="*",
                scenario=scen,
                runmodus=runmodus_suffix,
                ens=ens_flag,
            ).replace("**", "*")
            out_dir, _ = resolve_task_output_directory(
                output_root=Path(output_root),
                model_version=MODEL_VERSION,
                scenario=scen,
                workflow=task_workflow,
                runmodus=task_runmodus,
                canonical_layout=canonical_layout,
                canonical_experiment_id=task_experiment_id,
                canonical_n_ensemble=canonical_n_ensemble,
                canonical_kind=canonical_kind,
                canonical_run_instance=canonical_run_instance,
            )
            already = list(out_dir.glob(f"{fname_root}.nc"))
            if already:
                print(f"[resume] Skip {scen} r{ens_id:03d} (found {already[0].name})")
                continue

        wk = dict(
            scenario=scen,
            ensemble_id=int(ens_id),
            workflow=task_workflow,
            output_root=str(output_root),
            output_mode=output_mode,
            naming_template=name_tpl,
            dependence=bool(dependence),
            device=device,
            nlat=int(NLAT),
            lon_convention=LON_CONVENTION,
            calendar=CAL,
            model_version=MODEL_VERSION,
            gcmagicc_path=GCMagiccpath,
            models_subdir=MODELS_SUBDIR,
            resume=bool(resume_flag),
            predictor_path=predictor_path,
            magicc_run_id=magicc_run_id,
            bias_to_era5=bias_to_era5_flag,
            seed_override=seed_override,
            runmodus=task_runmodus,
            predictor_source_scenario=task.get("predictor_source_scenario"),
            predictor_year_start=task.get("predictor_year_start"),
            predictor_year_end=task.get("predictor_year_end"),
            canonical_layout=canonical_layout,
            canonical_experiment_id=task_experiment_id,
            canonical_n_ensemble=canonical_n_ensemble,
            canonical_kind=canonical_kind,
            canonical_run_instance=canonical_run_instance,
        )

        if dry_run or max_workers <= 1:
            rc = launch_worker_subprocess(
                worker_id=(idx % max_workers),
                total_workers=max_workers,
                worker_kwargs=wk,
                cpu_pin=cpu_pin,
                nice_io=nice_io,
                async_mode=False,
            )
            exit_codes.append(int(rc))
            continue

        proc = launch_worker_subprocess(
            worker_id=(idx % max_workers),
            total_workers=max_workers,
            worker_kwargs=wk,
            cpu_pin=cpu_pin,
            nice_io=nice_io,
            async_mode=True,
        )
        label = f"{scen} r{ens_id:03d}"
        if isinstance(proc, subprocess.Popen):
            active.append((proc, label))
        else:
            try:
                exit_codes.append(int(proc))
            except Exception:
                exit_codes.append(1)
        if len(active) >= max_workers:
            finished = _wait_for_one(active, block=True)
            if finished is not None:
                exit_codes.append(finished)

    # Drain remaining workers
    while active:
        finished = _wait_for_one(active, block=True)
        if finished is not None:
            exit_codes.append(finished)

    # Final GC
    import gc

    gc.collect()

    # Report
    n_fail = sum(1 for c in exit_codes if c != 0)
    if n_fail:
        print(f"Done with {len(exit_codes)} tasks; {n_fail} failed")
        sys.exit(1)
    else:
        print(f"All {len(exit_codes)} tasks finished successfully.")
        if bool(getattr(args, "auto_consolidate", False)):
            if _is_debiasloop_root(output_root):
                source_path = output_root / "debias" if (output_root / "debias").exists() else output_root
                _run_autoconsolidate(
                    source_paths=[source_path],
                    config_path=(
                        Path(args.auto_consolidate_config).expanduser().resolve(strict=False)
                        if getattr(args, "auto_consolidate_config", None)
                        else None
                    ),
                    cleanup_local=bool(getattr(args, "auto_consolidate_cleanup_local", True)),
                )
            else:
                print(
                    "ℹ️  Skipping auto-consolidate: output root is not a debiasloop run tree "
                    f"({output_root})."
                )
