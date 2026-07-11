#!/usr/bin/env python3
# ruff: noqa: E402
# ---
# 320_run_probabilistic_segments_gcmagicc.py
#
# Purpose
# -------
# Run GCMagicc probabilistically (100+ ensembles) while writing only *small*,
# user-selected *segments* (time windows, regions, variables) to an efficient,
# append-only Zarr store. Predictors X can be sourced from (A) CMIP6 .nc input
# (like 300_*.py) or (B) a MAGICC probabilistic SCM output Parquet.
#
# Key differences vs 300_*:
#   • Does NOT write full CMIP6-like .nc files for the whole experiment/model.
#   • Saves only snippets specified by OUTPUTDICTLIST (below) into Zarr groups.
#   • Adds MAGICC-based predictor mode with EFFECT_MODEL_SCHEME.
#   • Adds MAGICC-SAMEPERCMIP6 mode: same MAGICC subset per CMIP6 calibration (35 × N).
#
# Dependencies
# ------------
# core: numpy, torch, xarray, pandas, zarr (via xarray), cftime
# opt : regionmask (for AR6 regions), geopandas (for ISO3 regions; auto-fallback)
#
# Recommended env on HPC:
#   export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Note: PYTHONNOUSERSITE is not set to allow access to packages installed in user site-packages
#
# ---------------------------------------------------------------------------

import os as _os_threadcap
import os
import sys
from pathlib import Path

# --- PATCH: Fix CUDA Fragmentation ---
# This must be set BEFORE torch is imported to take effect
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add the parent directory to Python path to find segment_runner_core
sys.path.insert(0, str(Path(__file__).parent.parent))
import re
import gc
import time
import math
import json
import uuid
import argparse
import datetime
import inspect
import pickle
import random
import traceback
import subprocess
from dataclasses import dataclass, replace
from typing import List, Dict, Tuple, Optional, Sequence, Iterable, Union
from contextlib import contextmanager
from multiprocessing import Pool, Manager, cpu_count, get_context
from functools import partial
from collections import Counter

import numpy as np           # type: ignore
import pandas as pd          # type: ignore
import torch                 # type: ignore
import xarray as xr          # type: ignore
import cftime                # type: ignore
from xarray.coding.times import CFDatetimeCoder  # type: ignore

from segment_runner_core.priority_manager import set_low_priority  # type: ignore
from segment_runner_core.path_utils import get_gcmagicc_path, get_data_path
from scr.validation_helpers.helper_path_utils import (  # type: ignore
    get_data_root as get_shared_data_root,
    get_newscenario_inputs_root,
    get_repo_path,
)

# ---------------------------
# Thread caps / priority (like 300_*)
# ---------------------------
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
# Prefer stable GPU indexing (PCI_BUS_ID gives deterministic device order)
_os_threadcap.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Strip exported bash functions that confuse /bin/sh
for _env_key in tuple(os.environ):
    if _env_key.startswith("BASH_FUNC_"):
        os.environ.pop(_env_key, None)

try:
    set_low_priority(enable=True, cpu_nice=6, io_class="best-effort")
    print("✓ Set low priority for GCMagicc computation process")
except Exception:
    pass

try:
    torch.set_num_threads(_THREADS)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# Try to load HDF5 filter plugins (bitshuffle, zstd, lz4, etc.) if available.
# This is a no-op if the package isn't installed or if the system already provides plugins.
try:  # hdf5plugin registers its filters at import time
    import hdf5plugin  # type: ignore  # noqa: F401
except Exception:
    pass

# =============================================================================
# Device selection helpers
# =============================================================================


def normalize_device_string(device: Optional[str]) -> Optional[str]:
    """
    Map shorthand device strings to torch-compatible identifiers.
    Examples:
        "gpu" -> "cuda"
        "cuda0" -> "cuda:0"
    Returns None if no normalization is possible.
    """
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


def _cuda_available() -> bool:
    try:
        # Do not call torch.cuda.current_device() here; it initializes a CUDA
        # context in the parent, which breaks fork-based multiprocessing and
        # can pin us to GPU:0 unnecessarily. Simply report availability.
        return bool(torch.cuda.is_available())
    except Exception:
        return False

def _explain_cuda_status():
    """Verbose hints for diagnosing CUDA availability."""
    try:
        print(f"   • torch version: {torch.__version__}")
        print(f"   • torch.version.cuda: {getattr(torch.version, 'cuda', None)}")
        print(f"   • torch.version.hip : {getattr(torch.version, 'hip',  None)}")
    except Exception:
        pass
    for k in ("CUDA_VISIBLE_DEVICES","SLURM_JOB_GPUS","SLURM_STEP_GPUS","LOCAL_RANK","SLURM_LOCALID"):
        v = os.environ.get(k)
        if v is not None:
            print(f"   • {k}: {v}")

def _gpu_preflight(device_str: str, warn: bool = True) -> bool:
    """
    Try setting the CUDA device and allocate a tiny tensor. Return True on success.
    """
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
    except Exception as e:
        if warn:
            print(f"⚠️  GPU preflight failed on {device_str}: {e}")
        return False

def _maybe_enable_tf32(enable: bool = True):
    """Enable TF32 for matmul/cudnn on Ampere+ (L40S supports it)."""
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
        pass


# =============================================================================
# CPU sizing helpers (for CPU-only runs)
# =============================================================================


def _read_meminfo_kb() -> Tuple[Optional[int], Optional[int]]:
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


def _memory_gb() -> Tuple[float, float]:
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
    per_job_ram_gb: float = 45.0,
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

    # CPU-limited worker count
    cpu_limited = max(1, cpu_budget // max(1, int(per_job_cpus)))

    # Memory-limited worker count (skip if we couldn't detect memory)
    if per_job_ram_gb > 0 and mem_budget_gb > 0:
        mem_limited = max(1, int(math.floor(mem_budget_gb / per_job_ram_gb)))
    else:
        mem_limited = cpu_limited

    workers = max(1, min(cpu_limited, mem_limited))
    return workers

def _amp_dtype_cuda():
    """Prefer bf16 when supported, else fp16."""
    try:
        return torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
    except Exception:
        return torch.float16


def _find_gpu_index_by_name(name: str) -> Optional[int]:
    """
    Return the first CUDA device index whose name contains the provided string.
    """
    if not name or not _cuda_available():
        return None
    target = name.strip().lower()
    try:
        count = torch.cuda.device_count()
    except Exception:
        return None
    for idx in range(count):
        try:
            dev_name = torch.cuda.get_device_name(idx)
        except Exception:
            continue
        if target in dev_name.lower():
            return idx
    return None


_DEVICE_RESOLUTION_LOG: List[str] = []


def _record_device_note(message: str) -> None:
    _DEVICE_RESOLUTION_LOG.append(message)


def detect_default_device(force_gpu: bool = False) -> str:
    """
    Determine the default device for GCMagicc runs using environment hints.

    Priority:
      1. GCMAGICC_DEVICE (accepts values like "gpu", "cuda:1", "cpu")
      2. GCMAGICC_GPU_NAME (substring match against cuda device names)
      3. CUDA_VISIBLE_DEVICES first entry if CUDA is available
      4. torch.cuda.current_device() / index 0 fallback
      5. "cpu"
    
    Args:
        force_gpu: If True, attempt to use CUDA even if torch.cuda.is_available() returns False
    """
    _DEVICE_RESOLUTION_LOG.clear()
    env_device = normalize_device_string(os.environ.get("GCMAGICC_DEVICE"))
    if env_device:
        if env_device.startswith("cuda"):
            if not _cuda_available():
                if force_gpu:
                    _record_device_note(
                        f"GCMAGICC_DEVICE='{env_device}' requested GPU; forcing GPU usage despite torch.cuda.is_available()=False."
                    )
                    print(f"⚠️  Forcing GPU usage for '{env_device}' despite torch.cuda.is_available()=False. This may fail if PyTorch lacks CUDA support.")
                    return env_device
                else:
                    _record_device_note(
                        f"GCMAGICC_DEVICE='{env_device}' requested GPU but CUDA unavailable; falling back to CPU."
                    )
                    print(f"⚠️  Requested GPU device '{env_device}' but torch.cuda.is_available() is False; falling back to CPU.")
                    return "cpu"
            else:
                _record_device_note(f"Using device override from GCMAGICC_DEVICE='{env_device}'.")
                return env_device
        else:
            _record_device_note(f"Using device override from GCMAGICC_DEVICE='{env_device}'.")
            return env_device

    env_gpu_name = os.environ.get("GCMAGICC_GPU_NAME")
    if env_gpu_name:
        idx = _find_gpu_index_by_name(env_gpu_name)
        if idx is not None:
            _record_device_note(f"Matched GPU name '{env_gpu_name}' to CUDA device index {idx}.")
            return f"cuda:{idx}"
        else:
            print(f"⚠️  Could not match GCMAGICC_GPU_NAME='{env_gpu_name}' to an available GPU; continuing with automatic selection.")
            _record_device_note(f"GCMAGICC_GPU_NAME='{env_gpu_name}' did not match any visible GPU.")

    if _cuda_available() or force_gpu:
        if force_gpu and not _cuda_available():
            _record_device_note("Force GPU enabled; attempting CUDA despite availability check.")
        else:
            _record_device_note("torch reports CUDA available.")
        # If launched under SLURM/torchrun, prefer local rank binding
        lr = os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID")
        if lr and lr.isdigit():
            idx = int(lr)
            _record_device_note(f"Selected CUDA device by LOCAL_RANK/SLURM_LOCALID={idx}.")
            return f"cuda:{idx}"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            first = visible.split(",")[0].strip()
            if first:
                _record_device_note(f"Selected first CUDA_VISIBLE_DEVICES entry '{first}'.")
                return f"cuda:{first}" if first.isdigit() else "cuda"
        # Do not touch torch.cuda.current_device() here; that would create a
        # CUDA context in the parent. Defer exact index selection to workers.
        _record_device_note("Defaulting to CUDA (no specific index).")
        return "cuda"

    _record_device_note("CUDA unavailable: torch.cuda.is_available() returned False.")
    return "cpu"


def get_available_gpus() -> List[int]:
    """
    Return a list of available CUDA device indices.
    Returns empty list if CUDA is not available.
    """
    if not _cuda_available():
        return []
    try:
        count = torch.cuda.device_count()
        return list(range(count))
    except Exception:
        return []

def _classify_gpu_occupancy(indices: List[int]) -> Tuple[List[int], List[int]]:
    """
    Return (idle, busy) GPU indices among the provided list.
    Busy means at least one running process is attached OR memory usage exceeds threshold.
    Falls back to nvidia-smi if pynvml is unavailable.
    
    Memory threshold: GPUs with >10% memory used are considered busy to avoid OOM errors.
    """
    if not indices:
        return ([], [])

    busy: set[int] = set()
    busy_info: Dict[int, List[str]] = {}
    # Memory threshold: consider GPU busy if >10% memory is used
    MEMORY_THRESHOLD_PERCENT = float(os.environ.get("GCMAGICC_GPU_MEMORY_THRESHOLD_PERCENT", "10.0"))

    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        for idx in indices:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            except Exception:
                continue
            
            # Check for running processes
            procs = []
            try:
                procs.extend(pynvml.nvmlDeviceGetComputeRunningProcesses(handle))
            except Exception:
                pass
            try:
                procs.extend(pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle))
            except Exception:
                pass
            
            # Check memory usage
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_mem = mem_info.total
                used_mem = mem_info.used
                free_mem = mem_info.free
                mem_percent = (used_mem / total_mem * 100) if total_mem > 0 else 0
                
                # Consider GPU busy if memory usage exceeds threshold
                if mem_percent > MEMORY_THRESHOLD_PERCENT:
                    busy.add(idx)
                    mem_gb = used_mem / (1024**3)
                    mem_total_gb = total_mem / (1024**3)
                    busy_info.setdefault(idx, []).append(
                        f"memory_used={mem_gb:.2f}GiB/{mem_total_gb:.2f}GiB ({mem_percent:.1f}%)"
                    )
            except Exception:
                pass
            
            if procs:
                busy.add(idx)
                info_list = []
                for p in procs:
                    try:
                        mem = getattr(p, "usedGpuMemory", None)
                        mem_mb = f"{mem/1024/1024:.1f}MiB" if mem is not None else "?"
                    except Exception:
                        mem_mb = "?"
                    info_list.append(f"pid={getattr(p, 'pid', '?')} mem={mem_mb}")
                if info_list:
                    busy_info.setdefault(idx, []).extend(info_list)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    except Exception:
        # Fallback to nvidia-smi
        try:
            # First check for running processes
            out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_index,pid,process_name,used_gpu_memory", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.STDOUT,
            )
            for line in out.splitlines():
                s = line.strip()
                if not s or "No running" in s:
                    continue
                parts = [p.strip() for p in s.split(",")]
                if parts and parts[0].isdigit():
                    gpu_idx = int(parts[0])
                    busy.add(gpu_idx)
                    if len(parts) >= 4:
                        try:
                            mem_part = parts[3]
                        except Exception:
                            mem_part = "?"
                        busy_info.setdefault(gpu_idx, []).append(f"pid={parts[1]} name={parts[2]} mem={mem_part}")
        except Exception:
            pass
        
        # Check memory usage via nvidia-smi
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.STDOUT,
            )
            for line in out.splitlines():
                s = line.strip()
                if not s:
                    continue
                parts = [p.strip() for p in s.split(",")]
                if len(parts) >= 3 and parts[0].isdigit():
                    gpu_idx = int(parts[0])
                    try:
                        used_mem_mb = int(parts[1])
                        total_mem_mb = int(parts[2])
                        mem_percent = (used_mem_mb / total_mem_mb * 100) if total_mem_mb > 0 else 0
                        
                        if mem_percent > MEMORY_THRESHOLD_PERCENT:
                            busy.add(gpu_idx)
                            used_gb = used_mem_mb / 1024
                            total_gb = total_mem_mb / 1024
                            busy_info.setdefault(gpu_idx, []).append(
                                f"memory_used={used_gb:.2f}GiB/{total_gb:.2f}GiB ({mem_percent:.1f}%)"
                            )
                    except (ValueError, IndexError):
                        pass
        except Exception:
            # If we cannot determine occupancy, assume all are idle
            pass

    idle = [g for g in indices if g not in busy]
    if busy_info:
        msg = ", ".join(f"GPU{k}: {', '.join(v)}" for k, v in sorted(busy_info.items()))
        print(f"ℹ️  Detected busy GPUs (processes or memory >{MEMORY_THRESHOLD_PERCENT}%): {msg}")
    return (idle, sorted(busy))


def select_gpus_for_parallel(
    requested_gpus: Optional[int] = None,
    gpu_list: Optional[List[int]] = None,
    device_override: Optional[str] = None,
) -> Tuple[List[int], bool]:
    """
    Determine which GPUs to use for parallel execution.
    
    Args:
        requested_gpus: Number of GPUs to use (None = auto-detect all)
        gpu_list: Specific GPU indices to use (None = auto-detect)
        device_override: Device string override (e.g., "cuda:0", "cpu")
    
    Returns:
        Tuple of (list of GPU indices, use_multiprocessing flag)
        - If CPU mode or single GPU, returns ([], False) for single-process execution
        - If multiple GPUs, returns (list of indices, True) for multiprocessing
    """
    # If CPU mode is explicitly requested, don't use multiprocessing
    if device_override and device_override == "cpu":
        return ([], False)
    
    # Get available GPUs
    available = get_available_gpus()
    if not available:
        return ([], False)

    idle_gpus, busy_gpus = _classify_gpu_occupancy(available)
    preferred_pool = idle_gpus if idle_gpus else available
    if idle_gpus:
        if busy_gpus:
            print(f"🆓 GPU occupancy: idle={idle_gpus}; busy={busy_gpus}. Preferring idle GPUs only.")
        else:
            print(f"🆓 GPU occupancy: idle={idle_gpus}.")
    elif busy_gpus:
        msg = f"⚠️  GPU occupancy: all visible GPUs busy {busy_gpus}"
        print(msg + ("" if ALLOW_BUSY_GPU else "; not proceeding because GCMAGICC_ALLOW_BUSY_GPU is false"))
        if not ALLOW_BUSY_GPU:
            raise RuntimeError("No idle GPUs detected; aborting to avoid running on busy devices. "
                               "Set GCMAGICC_ALLOW_BUSY_GPU=1 to allow using busy GPUs.")

    # If specific GPU list provided, use it
    if gpu_list is not None:
        # Validate all requested GPUs are available
        valid_gpus = [gpu for gpu in gpu_list if gpu in available]
        if not valid_gpus:
            print(f"⚠️  None of requested GPUs {gpu_list} are available. Available: {available}")
            return ([], False)
        # Prefer idle GPUs among the requested list
        preferred_requested = [g for g in valid_gpus if g in preferred_pool]
        if not preferred_requested and not ALLOW_BUSY_GPU:
            busy_requested = [g for g in valid_gpus if g not in preferred_pool]
            if busy_requested:
                raise RuntimeError(
                    f"Requested GPUs {busy_requested} are currently busy; set GCMAGICC_ALLOW_BUSY_GPU=1 to force use or choose idle GPUs {idle_gpus}."
                )
        chosen = preferred_requested if preferred_requested else valid_gpus
        if len(chosen) == 1:
            return (chosen, False)  # Single GPU, no multiprocessing needed
        return (chosen, True)
    
    # If device override specifies a single GPU, use that
    if device_override:
        normalized = normalize_device_string(device_override)
        if normalized and normalized.startswith("cuda"):
            try:
                idx = int(normalized.split(":", 1)[1]) if ":" in normalized else 0
                if idx in available:
                    if idx not in preferred_pool:
                        if not ALLOW_BUSY_GPU:
                            raise RuntimeError(
                                f"Requested GPU override {idx} is busy; set GCMAGICC_ALLOW_BUSY_GPU=1 to override or choose an idle GPU {idle_gpus}."
                            )
                        print(f"⚠️  Requested GPU override {idx} is busy; honoring override due to ALLOW_BUSY_GPU.")
                    return ([idx], False)  # Single GPU
            except Exception:
                pass
    
    # Determine how many GPUs to use
    if requested_gpus is not None:
        n_gpus = min(requested_gpus, len(preferred_pool))
    else:
        # Auto-detect: use all available GPUs
        n_gpus = len(preferred_pool)
    
    if n_gpus <= 1:
        # Single GPU, use first available
        return ([preferred_pool[0]], False)
    
    # Multiple GPUs: use first n_gpus
    selected = preferred_pool[:n_gpus]
    return (selected, True)


# Check for force GPU flag
_FORCE_GPU = bool(os.environ.get("GCMAGICC_FORCE_GPU", ""))
DEFAULT_DEVICE = detect_default_device(force_gpu=_FORCE_GPU)


def get_default_device() -> str:
    """
    Expose the resolved device for external callers/tests.
    """
    return DEFAULT_DEVICE


def report_device_status(device: str) -> None:
    """
    Print diagnostic information about the resolved compute device.
    """
    print(f"🖥️  GCMagicc device: {device}")
    for note in _DEVICE_RESOLUTION_LOG:
        print(f"   • {note}")

    cuda_available = _cuda_available()
    print(f"   • torch.cuda.is_available(): {cuda_available}")
    _explain_cuda_status()
    if cuda_available:
        try:
            count = torch.cuda.device_count()
        except Exception:
            count = None
        if count is not None:
            print(f"   • CUDA device count: {count}")
        if device.startswith("cuda"):
            try:
                idx = int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
            except Exception:
                idx = torch.cuda.current_device()
            try:
                name = torch.cuda.get_device_name(idx)
            except Exception:
                name = "unknown"
            print(f"   • Using CUDA device {idx}: {name}")
        else:
            print("   • CUDA is available but CPU selected. Set GCMAGICC_DEVICE or --device to use GPU.")
    else:
        print("   • CUDA unavailable. Ensure: (1) CUDA-enabled PyTorch build, (2) a GPU is allocated/visible, (3) CUDA drivers/modules are loaded.")

# =============================================================================
# ------------------------- USER SPECIFICATION --------------------------------
# =============================================================================
# 0) Optional model root override. Set this to the absolute path containing
#     run_general.py / model checkpoints if you do not want to rely on env vars.
MODEL_ROOT_OVERRIDE: Optional[str] = None  # e.g. "/scratch/models"
if MODEL_ROOT_OVERRIDE:
    os.environ["GCMAGICC_MODEL_ROOT"] = MODEL_ROOT_OVERRIDE
    print(f"📁 MODEL_ROOT_OVERRIDE set to {MODEL_ROOT_OVERRIDE}")

# 1) Where to source X predictors
SOURCE_X_PREDICTORS = "MAGICCxERA5" # os.environ.get("GCMAGICC_SOURCE_X", "MAGICC-SAMEPERCMIP6")  # 'CMIP6' | 'MAGICC' | 'MAGICC-SAMEPERCMIP6' | 'MAGICCxERA5'

# --- Unified settings (used by all SOURCE_X_PREDICTORS options)
SOURCE_ID_WHITELIST: List[str] = []            # CMIP6 source IDs (e.g., ["UKESM1-0-LL", "MIROC6"])
EXPERIMENT_ID_WHITELIST: List[str] = ["ssp245"] # Experiment/scenario IDs (e.g., ["ssp119", "ssp245", "ssp370"])
                                                # Used for: CMIP6 experiments (Option A) and MAGICC scenarios (Options B & C)

# --- Option A (CMIP6) settings
# SOURCE_ID_WHITELIST: List[str] = ["IPSL-CM6A-LR", "UKESM1-0-LL", "CanESM5", "ACCESS-CM2"]
MAX_N_MEMBERS_PER_SOURCE: int = 3             # 1..100 max r*i*p*f per (source, experiment)
ENSEMBLES_A: int = 3                          # 1..1000 draws per CMIP6 file (reduced from 100 to avoid CUDA OOM)
BIASCORRECT_TO_ERA5_A: bool = False            # if True: usebias_model=0, useeffect_model=source-index

# --- Option B (MAGICC) settings
SCM_RESULTS_DIR = os.environ.get(
    "SCM_PPI_DIR",
    str(get_repo_path("2025magicc") / "output/scm_results_ppi"),
)


SCM_RESULTS_PARQUET = "scm_results_ppi_AR6_clean_ssp119_clean_ssp126_clean_ssp245_and_7_more_20251201-050049.parquet"  # If set, use this file directly; otherwise auto-select from SCM_RESULTS_DIR

SCM_RESULTS_PARQUET = "scm_results_ppi_AR6_clean_ssp119_clean_ssp126_clean_ssp245_and_7_more_RUNMODUS-NATURAL_20251201-050049.parquet"
# note, for attribution studies, use the different RUNMODUS ones.. 
# ANTHROPOGENIC RUN 
# SCM_RESULTS_PARQUET = "scm_results_clean_ssp119_clean_ssp126_clean_ssp245_and_7_more_RUNMODUS-ANTHROPOGENIC_20251113-014654.parquet"  # If set, use this file directly; otherwise auto-select from SCM_RESULTS_DIR
# NATURAL 
# SCM_RESULTS_PARQUET = "scm_results_clean_ssp119_clean_ssp126_clean_ssp245_and_7_more_RUNMODUS-NATURAL_20251113-014654.parquet"  # If set, use this file directly; otherwise auto-select from SCM_RESULTS_DIR
ENSEMBLES_B: int = 25                         # 1..600 (MAGICC ensemble members)
BIASCORRECT_TO_ERA5_B: bool = True             # default True per spec
EFFECT_MODEL_SCHEME: Union[str, int, List[int], List[str]] = 0
# Allowed:
#   "Random" or "all"                -> random choice over all non-zero model indices (equivalent)
#   "allplusERA5"                    -> random choice over all model indices including ERA5 (index 0)
#   7                                -> fixed GCM index
#   [1, 5, 12]                       -> random choice from these indices
#   "UKESM1-0-LL"                    -> fixed by name
#   ["UKESM1-0-LL", "MIROC6"]        -> random choice from these names

# --- Option D (MAGICCxERA5) settings
USE_RUNMODUS = os.environ.get("GCMAGICC_USE_RUNMODUSE", "all")
USE_WORKFLOW = os.environ.get("GCMAGICC_USE_WORKFLOW", "AR6")  # 'AR6' | 'AR7' | 'all'

# ERA5_SPLICED_PREDICTOR_DIR: path to ERA5-spliced predictors base directory
# New structure: {base}/{scenario}/{AR6|AR7}/runmodus_{all|natural|aerosol|anthropogenic}/
ERA5_SPLICED_PREDICTOR_DIR = os.environ.get(
    "GCMAGICC_ERA5_SPLICED_PREDICTOR_DIR",
    str(get_newscenario_inputs_root() / "era5_spliced_predictors_20251216_052212"),
)
ENSEMBLES_D: int = 10                          # default draws from ERA5-spliced predictors
BIASCORRECT_TO_ERA5_D: bool = True             # effect index fixed to 0 (ERA5)

# tas_smoothed – constants (override if you want to lock these)


# Retry behavior for multi-GPU runs. Clamped to [0, 5].
MAX_GPU_RETRIES: int = max(0, min(5, int(os.environ.get("GCMAGICC_MAX_GPU_RETRIES", "2"))))
# Number of work items a GPU worker process should handle before it is torn down
# and respawned. Keeping this small helps drop any leaked CUDA allocations or
# fragmentation between items at the cost of some re-start overhead.
GPU_TASKS_PER_CHILD: int = max(1, int(os.environ.get("GCMAGICC_GPU_TASKS_PER_CHILD", 1)))
# Minimum free memory required on a GPU before we attempt a job (GiB).
# Override via GCMAGICC_MIN_FREE_MEMORY_GB; set to 0 to disable the guard.
MIN_FREE_MEMORY_GB: float = float(os.environ.get("GCMAGICC_MIN_FREE_MEMORY_GB", "20.0"))


# --- Option C (MAGICC-SAMEPERCMIP6) settings
# Select the same subset of MAGICC parameter sets (default 20) for each CMIP6 calibration (non‑zero indices; typically 35),
# resulting in total ensembles ENSEMBLES_C_PER_CMIP6 × N_CMIP6_CAL (≈ 20 × 35).
ENSEMBLES_C_PER_CMIP6: int =  20
BIASCORRECT_TO_ERA5_C: bool = False  # default False
EFFECT_MODEL_SCHEME_C: Union[str, List[int], List[str]] = "allplusERA5"
# Allowed:
#   "all"                              -> use all CMIP6 calibration indices (non-zero, excludes ERA5)
#   "allplusERA5"                     -> use all CMIP6 calibration indices including ERA5 (index 0)
#   [1, 4, 6]                          -> use these specific model indices
#   ["UKESM1-0-LL", "CanESM5", "ACCESS-CM2"] -> use these model names
#   [0, 1, 4] or ["ERA5", "UKESM1-0-LL"] -> can include ERA5 (index 0) in lists



# If ERA5_GMT_REF_K is None, the script will try to find a file or fall back to 288.0 K
ERA5_GMT_REF_K: Optional[float] = None         # mean GLOBAL 2m (or tas) 1990–2020 (K)
HADCRUT5_DELTA_1850_1900_TO_REF_C: Optional[float] = None  # ∆T (°C) 1850–1900 -> 1995–2014
HADCRUT5_SUMMARY_CSV: Optional[str] = os.environ.get(
    "HADCRUT5_SUMMARY_CSV",
    str(Path(__file__).parent.parent / "data" / "HadCRUT5" / "HadCRUT.5.1.0.0.analysis.summary_series.global.annual.csv")
)  # Path to HadCRUT5 summary CSV file (default: data/HadCRUT5/HadCRUT.5.1.0.0.analysis.summary_series.global.annual.csv)

# 2) Output segments to save (independent of A/B)
#    period: (start_year, end_year) inclusive
#    months: 'all' or list of month strings ['Jan','Feb',...,'Dec']
#    region: 'global' | IPCC‑AR6 region name | ISO3 (country)
#    operation: 'grid-points' | 'area-weighted average'
#    variable: one of the emulated output variables (e.g., 'tas','pr','tasmax',...)
# OUTPUTDICTLIST: List[dict] = [
#     # Examples; adjust as desired:
#     {"period": (2081, 2100), "months": "all", "region": "global", "operation": "area-weighted time average", "variable": "tas"},
#     {"period": (2081, 2100), "months": "all", "region": "global", "operation": "area-weighted annual average", "variable": "tas"},
#     {"period": (2081, 2100), "months": "all", "region": "global", "operation": "area-weighted time average", "variable": "tasmax"},
#     {"period": (2081, 2100), "months": "all", "region": "global", "operation": "area-weighted annual average", "variable": "tasmax"},
#     {"period": (2081, 2100), "months": "all", "region": "global", "operation": "area-weighted time average", "variable": "hurs"},
#     {"period": (2081, 2100), "months": "all", "region": "global", "operation": "area-weighted annual average", "variable": "hurs"},
#     {"period": (2081, 2100), "months": ["Jun", "Jul", "Aug"], "region": "Mediterranean", "operation": "grid-points", "variable": "tas"},
#     {"period": (2081, 2100), "months": ["Jun", "Jul", "Aug"], "region": "Mediterranean", "operation": "grid-points", "variable": "tasmax"},
#     {"period": (2081, 2100), "months": ["Jun", "Jul", "Aug"], "region": "Mediterranean", "operation": "grid-points", "variable": "hurs"},
#     {"period": (2081, 2100), "months": ["Jun", "Jul", "Aug"], "region": "PRT", "operation": "grid-points", "variable": "tas"},
#     {"period": (2081, 2100), "months": ["Jun", "Jul", "Aug"], "region": "PRT", "operation": "grid-points", "variable": "tasmax"},
#     {"period": (2081, 2100), "months": ["Jun", "Jul", "Aug"], "region": "PRT", "operation": "grid-points", "variable": "hurs"},
#     {"period": (1995, 2014), "months": "all", "region": "global", "operation": "area-weighted time average", "variable": "tas"},
#     {"period": (1995, 2014), "months": "all", "region": "global", "operation": "area-weighted annual average", "variable": "tas"},
#     {"period": (1995, 2014), "months": "all", "region": "global", "operation": "area-weighted time average", "variable": "tasmax"},
#     {"period": (1995, 2014), "months": "all", "region": "global", "operation": "area-weighted annual average", "variable": "tasmax"},
#     {"period": (1995, 2014), "months": "all", "region": "global", "operation": "area-weighted time average", "variable": "hurs"},
#     {"period": (1995, 2014), "months": "all", "region": "global", "operation": "area-weighted annual average", "variable": "hurs"},
#     {"period": (1995, 2014), "months": ["Jun", "Jul", "Aug"], "region": "Mediterranean", "operation": "grid-points", "variable": "tas"},
#     {"period": (1995, 2014), "months": ["Jun", "Jul", "Aug"], "region": "Mediterranean", "operation": "grid-points", "variable": "tasmax"},
#     {"period": (1995, 2014), "months": ["Jun", "Jul", "Aug"], "region": "Mediterranean", "operation": "grid-points", "variable": "hurs"},
#     {"period": (1995, 2014), "months": ["Jun", "Jul", "Aug"], "region": "PRT", "operation": "grid-points", "variable": "tas"},
#     {"period": (1995, 2014), "months": ["Jun", "Jul", "Aug"], "region": "PRT", "operation": "grid-points", "variable": "tasmax"},
#     {"period": (1995, 2014), "months": ["Jun", "Jul", "Aug"], "region": "PRT", "operation": "grid-points", "variable": "hurs"},

# ]


# OUTPUTDICTLIST: List[dict] = [
#     # Examples; adjust as desired:
#     {"period": (2081, 2100), "months": "all", "region": "global", "operation": "area-weighted time average", "variable": "tas"},
#     {"period": (2081, 2100), "months": "all", "region": "global", "operation": "area-weighted annual average", "variable": "tas"},
#     {"period": (2081, 2100), "months": "all", "region": "global", "operation": "area-weighted time average", "variable": "tasmax"},
#     {"period": (2081, 2100), "months": "all", "region": "global", "operation": "area-weighted annual average", "variable": "tasmax"},
#     {"period": (2081, 2100), "months": "all", "region": "global", "operation": "area-weighted time average", "variable": "hurs"},
#     {"period": (2081, 2100), "months": "all", "region": "global", "operation": "area-weighted annual average", "variable": "hurs"},
#     {"period": (2081, 2100), "months": ["Jun", "Jul", "Aug"], "region": "PRT", "operation": "area-weighted time average", "variable": "tas"},
#     {"period": (2081, 2100), "months": ["Jun", "Jul", "Aug"], "region": "PRT", "operation": "area-weighted time average", "variable": "tasmax"},
#     {"period": (2081, 2100), "months": ["Jun", "Jul", "Aug"], "region": "PRT", "operation": "area-weighted time average", "variable": "hurs"},
#     {"period": (2081, 2100), "months": ["Jun", "Jul", "Aug"], "region": "PRT", "operation": "grid-points", "variable": "tas"},
#     {"period": (2081, 2100), "months": ["Jun", "Jul", "Aug"], "region": "PRT", "operation": "grid-points", "variable": "tasmax"},
#     {"period": (2081, 2100), "months": ["Jun", "Jul", "Aug"], "region": "PRT", "operation": "grid-points", "variable": "hurs"},
#     {"period": (1995, 2014), "months": "all", "region": "global", "operation": "area-weighted time average", "variable": "tas"},
#     {"period": (1995, 2014), "months": "all", "region": "global", "operation": "area-weighted annual average", "variable": "tas"},
#     {"period": (1995, 2014), "months": "all", "region": "global", "operation": "area-weighted time average", "variable": "tasmax"},
#     {"period": (1995, 2014), "months": "all", "region": "global", "operation": "area-weighted annual average", "variable": "tasmax"},
#     {"period": (1995, 2014), "months": "all", "region": "global", "operation": "area-weighted time average", "variable": "hurs"},
#     {"period": (1995, 2014), "months": "all", "region": "global", "operation": "area-weighted annual average", "variable": "hurs"},
#     {"period": (1995, 2014), "months": ["Jun", "Jul", "Aug"], "region": "PRT", "operation": "area-weighted time average", "variable": "tas"},
#     {"period": (1995, 2014), "months": ["Jun", "Jul", "Aug"], "region": "PRT", "operation": "area-weighted time average", "variable": "tasmax"},
#     {"period": (1995, 2014), "months": ["Jun", "Jul", "Aug"], "region": "PRT", "operation": "area-weighted time average", "variable": "hurs"},
#     {"period": (1995, 2014), "months": ["Jun", "Jul", "Aug"], "region": "PRT", "operation": "grid-points", "variable": "tas"},
#     {"period": (1995, 2014), "months": ["Jun", "Jul", "Aug"], "region": "PRT", "operation": "grid-points", "variable": "tasmax"},
#     {"period": (1995, 2014), "months": ["Jun", "Jul", "Aug"], "region": "PRT", "operation": "grid-points", "variable": "hurs"},
# ]

# note that more than 20-year windows seem to crash due to GPU memory constraints. 

OUTPUTDICTLIST: List[dict] = [
    # Examples; adjust as desired:
    {"period": (2081, 2100), "months": "all", "region": "global", "operation": "area-weighted time average", "variable": "tas"},
    {"period": (2081, 2100), "months": "all", "region": "global", "operation": "area-weighted annual average", "variable": "tas"},
    {"period": (2081, 2100), "months": "all", "region": "global", "operation": "area-weighted time average", "variable": "tasmax"},
    {"period": (2081, 2100), "months": "all", "region": "global", "operation": "area-weighted annual average", "variable": "tasmax"},
    {"period": (2081, 2100), "months": "all", "region": "global", "operation": "area-weighted time average", "variable": "pr"},
    {"period": (2081, 2100), "months": "all", "region": "global", "operation": "area-weighted annual average", "variable": "pr"},
    {"period": (2081, 2100), "months": "all", "region": "IRN", "operation": "area-weighted time average", "variable": "tas"},
    {"period": (2081, 2100), "months": "all", "region": "IRN", "operation": "area-weighted time average", "variable": "tasmax"},
    {"period": (2081, 2100), "months": "all", "region": "IRN", "operation": "area-weighted time average", "variable": "hurs"},
    {"period": (2081, 2100), "months": "all", "region": "IRN", "operation": "area-weighted time average", "variable": "tasmin"},
    {"period": (2081, 2100), "months": "all", "region": "IRN", "operation": "area-weighted time average", "variable": "rsds"},
    {"period": (2081, 2100), "months": "all", "region": "IRN", "operation": "area-weighted time average", "variable": "pr"},
    {"period": (2081, 2100), "months": "all", "region": "IRN", "operation": "grid-points", "variable": "tas"},
    {"period": (2081, 2100), "months": "all", "region": "IRN", "operation": "grid-points", "variable": "tasmax"},
    {"period": (2081, 2100), "months": "all", "region": "IRN", "operation": "grid-points", "variable": "hurs"},
    {"period": (2081, 2100), "months": "all", "region": "IRN", "operation": "grid-points", "variable": "tasmin"},
    {"period": (2081, 2100), "months": "all", "region": "IRN", "operation": "grid-points", "variable": "rsds"},
    {"period": (2081, 2100), "months": "all", "region": "IRN", "operation": "grid-points", "variable": "pr"},
    {"period": (1995, 2014), "months": "all", "region": "global", "operation": "area-weighted time average", "variable": "tas"},
    {"period": (1995, 2014), "months": "all", "region": "global", "operation": "area-weighted annual average", "variable": "tas"},
    {"period": (1995, 2014), "months": "all", "region": "global", "operation": "area-weighted time average", "variable": "tasmax"},
    {"period": (1995, 2014), "months": "all", "region": "global", "operation": "area-weighted annual average", "variable": "tasmax"},
    {"period": (1995, 2014), "months": "all", "region": "global", "operation": "area-weighted time average", "variable": "hurs"},
    {"period": (1995, 2014), "months": "all", "region": "global", "operation": "area-weighted annual average", "variable": "hurs"},
    {"period": (1995, 2014), "months": "all", "region": "IRN", "operation": "area-weighted time average", "variable": "tas"},
    {"period": (1995, 2014), "months": "all", "region": "IRN", "operation": "area-weighted time average", "variable": "pr"},
    {"period": (1975, 2025), "months": "all", "region": "IRN", "operation": "area-weighted time average", "variable": "tas"},
    {"period": (1995, 2014), "months": "all", "region": "IRN", "operation": "grid-points", "variable": "tas"},
    {"period": (1975, 2025), "months": "all", "region": "IRN", "operation": "grid-points", "variable": "tas"},
    {"period": (1975, 2025), "months": "all", "region": "IRN", "operation": "grid-points", "variable": "tasmax"},
    {"period": (1975, 2025), "months": "all", "region": "IRN", "operation": "grid-points", "variable": "hurs"},
    {"period": (1975, 2025), "months": "all", "region": "IRN", "operation": "grid-points", "variable": "tasmin"},
    {"period": (1975, 2025), "months": "all", "region": "IRN", "operation": "grid-points", "variable": "rsds"},
    {"period": (1975, 2025), "months": "all", "region": "IRN", "operation": "grid-points", "variable": "pr"},
]

# 3) General I/O and model configuration

# Debug flag: set to True to enable first-success debug logging
# Can also be set via environment variable GCMAGICC_DEBUG_FIRST_SUCCESS or --debug-first-success CLI flag
DEBUG_FIRST_SUCCESS: bool = True  # Set to True here to enable debug logging

def _get_model_version_code(model_version: str = None) -> str:
    """
    Map model_version to short version code.
    
    Args:
        model_version: Model version string (e.g., 'NxlversA5', 'model_NxlversA5', 'data/site_eth/projects/gcmagicc_ensemble_runner/data', 'model_NthreeversT1')
    
    Returns:
        Short version code: 'v100' for NxlversA5, 'v101' for NthreeversT1, or empty string if not recognized
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


def _generate_nameplate(
    option: str,
    bias_to_era5_a: bool = None,
    bias_to_era5_b: bool = None,
    bias_to_era5_c: bool = None,
    source_id_whitelist: List[str] = None,
    experiment_id_whitelist: List[str] = None,
    ensembles_a: int = None,
    ensembles_b: int = None,
    ensembles_c_per_cmip6: int = None,
    model_version: str = None,
) -> str:
    """
    Generate a short auto-generated nameplate based on user-specified options.
    
    Args:
        option: 'CMIP6' or 'MAGICC' or 'MAGICC-SAMEPERCMIP6'
        bias_to_era5_a: BIASCORRECT_TO_ERA5_A flag (for Option A)
        bias_to_era5_b: BIASCORRECT_TO_ERA5_B flag (for Option B)
        bias_to_era5_c: BIASCORRECT_TO_ERA5_C flag (for Option C)
        source_id_whitelist: SOURCE_ID_WHITELIST (for Option A)
        experiment_id_whitelist: EXPERIMENT_ID_WHITELIST (unified for all options)
        ensembles_a: ENSEMBLES_A (for Option A)
        ensembles_b: ENSEMBLES_B (for Option B)
        ensembles_c_per_cmip6: ENSEMBLES_C_PER_CMIP6 (for Option C)
        model_version: Model version string (e.g., 'NxlversA5', 'NthreeversT1')
    
    Returns:
        Short nameplate string (e.g., "A_v100_bE_UKESM1-0-LL_ssp585_E25")
    """
    option = option.upper()
    parts = []
    
    # Get model version code
    version_code = _get_model_version_code(model_version)
    
    if option == "CMIP6":
        parts.append("A")
        # Insert model version code right after option letter
        if version_code:
            parts.append(version_code)
        # Bias correction flag
        if bias_to_era5_a is not None:
            parts.append("bE" if bias_to_era5_a else "bN")
        # Source IDs (abbreviated)
        if source_id_whitelist:
            src_abbrev = []
            for src in source_id_whitelist[:3]:  # Limit to first 3
                # Create short abbreviation (e.g., "UKESM1-0-LL" -> "UKESM1")
                abbrev = src.split("-")[0] if "-" in src else src[:6]
                src_abbrev.append(abbrev)
            parts.append("_".join(src_abbrev))
        # Experiment IDs
        if experiment_id_whitelist:
            exp_abbrev = "_".join(exp[:6] for exp in experiment_id_whitelist[:2])  # Limit to first 2
            parts.append(exp_abbrev)
        # Ensembles count
        if ensembles_a is not None:
            parts.append(f"E{ensembles_a}")
    elif option == "MAGICC":
        parts.append("B")
        # Insert model version code right after option letter
        if version_code:
            parts.append(version_code)
        # Bias correction flag
        if bias_to_era5_b is not None:
            parts.append("bE" if bias_to_era5_b else "bN")
        # Experiment IDs (scenarios)
        if experiment_id_whitelist:
            exp_abbrev = "_".join(exp[:6] for exp in experiment_id_whitelist[:2])  # Limit to first 2
            if exp_abbrev:
                parts.append(exp_abbrev)
        # Ensembles count
        if ensembles_b is not None:
            parts.append(f"E{ensembles_b}")
    elif option == "MAGICC-SAMEPERCMIP6":
        parts.append("C")
        # Insert model version code right after option letter
        if version_code:
            parts.append(version_code)
        # Bias correction flag
        if bias_to_era5_c is not None:
            parts.append("bE" if bias_to_era5_c else "bN")
        # Experiment IDs (scenarios)
        if experiment_id_whitelist:
            exp_abbrev = "_".join(exp[:6] for exp in experiment_id_whitelist[:2])
            if exp_abbrev:
                parts.append(exp_abbrev)
        if ensembles_c_per_cmip6 is not None:
            parts.append(f"E{ensembles_c_per_cmip6}")
    elif option == "MAGICCXERA5":
        parts.append("D")
        # Insert model version code right after option letter
        if version_code:
            parts.append(version_code)
        if bias_to_era5_b is not None:
            parts.append("bE" if bias_to_era5_b else "bN")
        if experiment_id_whitelist:
            exp_abbrev = "_".join(exp[:6] for exp in experiment_id_whitelist[:2])
            if exp_abbrev:
                parts.append(exp_abbrev)
        if ensembles_b is not None:
            parts.append(f"E{ensembles_b}")

    nameplate = "_".join(parts) if parts else "default"
    # Sanitize: remove any characters that might cause filesystem issues
    nameplate = re.sub(r'[^a-zA-Z0-9_-]', '', nameplate)
    return nameplate


def _get_output_folder_name(nameplate: Optional[str] = None) -> str:
    """
    Generate output folder name with nameplate and timestamp.
    
    Format: segments_{nameplate}_{YYYYMonDD_HHMM}
    """
    if nameplate is None:
        # Auto-generate based on current configuration
        option = SOURCE_X_PREDICTORS
        nameplate = _generate_nameplate(
            option=option,
            bias_to_era5_a=BIASCORRECT_TO_ERA5_A,
            bias_to_era5_b=BIASCORRECT_TO_ERA5_B,
            bias_to_era5_c=BIASCORRECT_TO_ERA5_C,
            source_id_whitelist=SOURCE_ID_WHITELIST,
            experiment_id_whitelist=EXPERIMENT_ID_WHITELIST,
            ensembles_a=ENSEMBLES_A,
            ensembles_b=ENSEMBLES_B,
            ensembles_c_per_cmip6=ENSEMBLES_C_PER_CMIP6,
            model_version=MODEL_VERSION,
        )
    
    # Generate timestamp in readable format (e.g., 2025Nov23_1425)
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%b%d_%H%M")
    
    folder_name = f"segments_{nameplate}_{timestamp}"
    return folder_name


#    (inherits path logic from 300_*, but writes a single Zarr store of segments)
# Emulation model identity (compatible with 300_*)
MODEL_VERSION = os.environ.get("GCMAGICC_MODEL_VERSION", "NthreeversT1")   # "NthreeversT1") # "NthreeversT1",, "NxlversA5"
MODEL_NUMBER = os.environ.get("GCMAGICC_MODEL_NUMBER", "v101")

# Default nameplate can be overridden via environment variable or will be auto-generated
DEFAULT_NAMEPLATE = os.environ.get("GCMAGICC_SEGMENTS_NAMEPLATE", None)
OUTPUT_FOLDER_NAME = _get_output_folder_name(DEFAULT_NAMEPLATE)
OUTPUT_STORE_DIR = os.path.join(str(get_data_path("segments")), OUTPUT_FOLDER_NAME)
OUTPUT_STORE_NAME = os.environ.get("GCMAGICC_SEGMENTS_NAME", "segments.zarr")
ZARR_MODE = "a"   # append groups safely
N_LAT = int(os.environ.get("GCMAGICC_N_LAT", "180"))
LON_CONVENTION = "360"  # '360' or '180'
# Fail fast on first emulation/write error (default False to allow queued retries)
FAIL_FAST = os.environ.get("GCMAGICC_FAIL_FAST", "0").strip().lower() not in ("0", "false", "no")
# Force queue+retry behavior even if FAIL_FAST is set (default on)
QUEUE_ON_ERROR = os.environ.get("GCMAGICC_QUEUE_ON_ERROR", "1").strip().lower() not in ("0", "false", "no")
if QUEUE_ON_ERROR:
    FAIL_FAST = False
# Allow falling back to GPUs that already have running processes
ALLOW_BUSY_GPU = os.environ.get("GCMAGICC_ALLOW_BUSY_GPU", "0").strip().lower() not in ("0", "false", "no")
# Optional run log path (jsonl of success/fail)
RUN_LOG_NAME = os.environ.get("GCMAGICC_RUN_LOG_NAME", "run_log.jsonl")
RUN_LOG_NAME = os.environ.get("GCMAGICC_RUN_LOG", "run_log.jsonl")
# Default AMP on CUDA to cut memory footprint; can disable via env/CLI
DEFAULT_AMP = os.environ.get("GCMAGICC_USE_AMP", "1").lower() not in ("0", "false", "no")
# Debug flag to print sample_from_combined_model kwargs (noisy; use sparingly)
DEBUG_SAMPLE_KWARGS: bool = os.environ.get("GCMAGICC_DEBUG_SAMPLE_KWARGS", "0").lower() not in ("0", "false", "no")

# CMIP6-style input directory (for Option A)
GCMAGICC_CMIP6_INPUT_DIR = str(get_shared_data_root() / "out_ETHFOG_10June2025_vetted")
DEFAULT_CMIP6_INPUT_DIR = os.environ.get("GCMAGICC_CMIP6_INPUT_DIR", GCMAGICC_CMIP6_INPUT_DIR)

# =============================================================================
# End user specification
# =============================================================================


# =============================================================================
# Helpers carried over / adapted from 300_* (paths, meta loading, grids)
# =============================================================================

GCMAGICC_OUTPUT_DIR = os.environ.get("GCMAGICC_OUTPUT_DIR", get_gcmagicc_path(MODEL_VERSION))

# --- Model directory resolution (same decision tree as 300_*, condensed)
def _resolve_model_dir(model_version: str) -> Tuple[str, str]:
    server_device = "ada30_server"
    if server_device == "ada30_server":
        if model_version in ["vers2"]:
            base = "../../gcm_firefly_data/model_Nfour_vers2/"
            sub = "models_vers2"
        elif model_version in ["versN3a", "versN3b", "versN3c", "vers4c"]:
            base = f"../../gcm_firefly_data/model_Nfour_{model_version.replace('vers','vers')}/"
            sub = "models"
        elif model_version in ["Nextvers1", "Nextvers2", "Nextvers3"]:
            base = f"../../gcm_firefly_data/model_{model_version}/"
            sub = "models_vers1"
        elif model_version in ["Nextvers5"]:
            base = f"../../gcm_firefly_data/model_{model_version}/"
            sub = "models_vers5"
        elif model_version in ["Nextvers6"]:
            base = f"../../gcm_firefly_data/model_{model_version}/"
            sub = "models_vers6"
        elif model_version in ["NxlversA", "NxlversA2", "NxlversA3", "NxlversA4", "NxlversA5", "NxlversC", "NxlversB", "NthreeversT1","NthreeversT2"]:
            base = f"../../gcm_firefly_data/model_{model_version}/"
            sub = "modelsA"
        else:
            base = f"../../gcm_firefly_data/model_Nfour_{model_version}/"
            sub = "models_vers1"
    else:
        base = "../../gcm_firefly_data/model_Nfour_vers2/"
        sub = "models_vers2"

    # allow full override
    _env_model_dir = os.environ.get("GCMAGICC_MODEL_DIR")
    if _env_model_dir:
        base = _env_model_dir.rstrip("/") + "/"

    # normalize to absolute
    base_abs = str((Path(__file__).resolve().parent / base).resolve()) + os.sep if not os.path.isabs(base) else base.rstrip(os.sep) + os.sep
    return base_abs, sub

GCMagiccpath, subfolder_GCMagiccmodels = _resolve_model_dir(MODEL_VERSION)
GCMagiccmodels = os.path.join(GCMagiccpath, subfolder_GCMagiccmodels)

_run_general_path = Path(GCMagiccpath).resolve() / "run_general.py"
if not _run_general_path.exists():
    raise FileNotFoundError(f"run_general.py not found in {GCMagiccpath} (required for probabilistic segments)")

def _load_sample_from_combined_model():
    """
    Import sampler from run_general.py (required); raise if missing.
    """
    resolved_path = _run_general_path.parent
    resolved_str = str(resolved_path)
    if resolved_str not in sys.path:
        sys.path.insert(0, resolved_str)
    from run_general import sample_from_combined_model as _fn  # type: ignore
    print(f"✅ Using run_general.py from {resolved_path}")
    return _fn

sample_from_combined_model = _load_sample_from_combined_model()

def _load_meta_file(models_dir: Path):
    """
    Resolve META and date token using run_general signature when available,
    otherwise fall back to the first meta_*.pkl in the models directory.
    """
    date_token = None
    try:
        sig = inspect.signature(sample_from_combined_model)
        date_param = sig.parameters.get("DATE")
        if date_param is not None:
            date_token = date_param.default
    except Exception:
        pass

    models_dir = models_dir.resolve()
    if date_token is not None:
        meta_path = models_dir / f"meta_{date_token}.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} not found (DATE derived from run_general)")
    else:
        candidates = sorted(models_dir.glob("meta_*.pkl"))
        if not candidates:
            raise FileNotFoundError(f"No meta_*.pkl found in {models_dir}")
        meta_path = candidates[0]
        date_token = meta_path.stem.split("meta_")[-1]

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta, str(meta_path), str(date_token)

META, METAFILE2USE, DATEOFMETAFILE = _load_meta_file(Path(GCMagiccmodels))

# =============================================================================
# NEW: requested year-window utilities (trim compute to only requested periods)
# =============================================================================
def _collect_requested_year_windows(output_specs: List[dict]) -> List[Tuple[int, int]]:
    """
    Read OUTPUTDICTLIST and return a de-duplicated list of year windows (in input order).
    Overlapping windows are kept separate; only exact duplicates are removed.
    """
    spans: List[Tuple[int,int]] = []
    seen = set()
    for seg in output_specs:
        per = seg.get("period")
        if not per or len(per) != 2:
            continue
        y0, y1 = int(per[0]), int(per[1])
        if y0 > y1:
            y0, y1 = y1, y0
        tup = (y0, y1)
        if tup in seen:
            continue
        seen.add(tup)
        spans.append(tup)
    if not spans:
        return []
    return spans

def _intersect(a: Tuple[int,int], b: Tuple[int,int]) -> Optional[Tuple[int,int]]:
    lo = max(a[0], b[0]); hi = min(a[1], b[1])
    return (lo, hi) if lo <= hi else None

def _infer_time_bounds_from_nc(nc_filename: str) -> Tuple[int,int,int,int]:
    """
    Inspect a NetCDF and return (start_year, start_month, end_year, end_month).
    Simplified version that just uses xarray's default opening.
    """
    # Simplified approach - just use xarray's default opening
    time_coder = CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(nc_filename, decode_times=time_coder) as ds:
        t = ds["time"]
        years = t.dt.year.values
        months = t.dt.month.values
        return int(years[0]), int(months[0]), int(years[-1]), int(months[-1])

# ----------------------------- Robust NetCDF opening -----------------------------
def _engine_wishlist() -> List[str]:
    env = os.environ.get("GCMAGICC_XARRAY_ENGINE") or os.environ.get("XARRAY_ENGINE")
    if env:
        # allow comma-separated list in env var to override order
        return [e.strip() for e in env.split(",") if e.strip()]
    # Try netcdf4 first (works reliably), then h5netcdf (if available), then scipy
    return ["netcdf4", "h5netcdf", "scipy"]

def _mk_cfdate(calendar: str, y: int, m: int, d: int):
    cal = (calendar or "").lower()
    if "360" in cal or "360_day" in cal:
        return cftime.Datetime360Day(y, m, d, has_year_zero=True)
    if "noleap" in cal or "365_day" in cal:
        return cftime.DatetimeNoLeap(y, m, d)
    if "all_leap" in cal or "366_day" in cal:
        return cftime.DatetimeAllLeap(y, m, d)
    if "proleptic_gregorian" in cal:
        return cftime.DatetimeProlepticGregorian(y, m, d)
    if "standard" in cal or "gregorian" in cal:
        return cftime.DatetimeGregorian(y, m, d)
    # For unknown calendars, try to match the calendar type from the dataset
    # Default to 360-day since that's what most climate models use
    return cftime.Datetime360Day(y, m, d, has_year_zero=True)

@contextmanager
def _open_dataset_robust(nc_filename: str, *, decode_times: bool = True):
    """
    Context manager that tries several backends to open a NetCDF file robustly.
    - Prefer 'h5netcdf' (works with HDF5 plugins for modern filters)
    - Fall back to 'netcdf4' and then 'scipy'
    Always decodes with cftime when decode_times=True.
    """
    attempts = []
    last_exc: Optional[BaseException] = None
    for eng in _engine_wishlist():
        try:
            backend_kwargs = {}
            if eng == "h5netcdf":
                # tolerate imperfect netcdf4-ish files produced by various writers
                backend_kwargs = {"invalid_netcdf": True}
            ds = xr.open_dataset(
                nc_filename,
                engine=eng,
                decode_times=False,          # decode ourselves with cftime below
                backend_kwargs=backend_kwargs,
            )
            # Ensure cftime objects for non-standard calendars (e.g., 360_day)
            ds = xr.decode_cf(ds, use_cftime=True) if decode_times else ds
            try:
                yield ds
            finally:
                ds.close()
            return
        except Exception as e:
            last_exc = e
            attempts.append(f"{eng}: {type(e).__name__}: {e}")
            continue
    msg = (
        f"Failed to open NetCDF {nc_filename} with engines "
        f"{' -> '.join(_engine_wishlist())}. "
        "Tip: ensure 'h5netcdf' and 'hdf5plugin' are installed for HDF5-filtered NetCDF4 files."
    )
    if last_exc is not None:
        raise RuntimeError(msg) from last_exc
    raise RuntimeError(msg)

# ---------------------------------------------------------------------------
# Coordinate grid helper (copied from 300_*)
def generate_coordinate_grids(
    nlat: int, nlon: int = 360, lon_convention: str = "360", lat_direction: str = "north_to_south"
) -> tuple[np.ndarray, np.ndarray]:
    lat_res = 180.0 / nlat
    lon_res = 360.0 / nlon
    lat_start = 90.0 - lat_res / 2
    lat_end = -90.0 + lat_res / 2
    lats = np.linspace(lat_start, lat_end, nlat) if lat_direction == "north_to_south" else np.linspace(lat_end, lat_start, nlat)
    if lon_convention == "180":
        lon_start = -180.0 + lon_res / 2
        lon_end = 180.0 - lon_res / 2
    else:
        lon_start = lon_res / 2
        lon_end = 360.0 - lon_res / 2
    lons = np.linspace(lon_start, lon_end, nlon)
    return lats, lons

# ---------------------------------------------------------------------------
# Predictor container (like 300_*)
@dataclass
class PredictorData:
    X: torch.Tensor
    predictor_names: List[str]
    model_names: List[str]
    variables_2predict: List[str]
    # add a minimal time context for 320_* (we store exact months)
    time_index: Optional[pd.DatetimeIndex] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    start_month: int = 1
    end_month: int = 12

# =============================================================================
# CMIP6 predictor extractor (exact copy of 300_* semantics, month 1..12)
# =============================================================================

def extract_predictors_from_nc(
    nc_filename: str,
    timespan: List[int] = [2015, 2100],
    metafile2use: str = METAFILE2USE,
    ) -> PredictorData:
    start_year, end_year = timespan
    
    # Simplified approach - just use xarray's default opening
    time_coder = CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(nc_filename, decode_times=time_coder) as ds:
        return _extract_predictors_from_dataset(ds, start_year, end_year, nc_filename, metafile2use)

def _extract_predictors_from_dataset(ds, start_year, end_year, nc_filename, metafile2use):
    # calendar-aware slicing (keep 360_day where present)
    cal = ds["time"].attrs.get("calendar", "standard")
    
    # If we can't determine the calendar from attributes, infer it from the actual time values
    if cal == "standard" and len(ds["time"].values) > 0:
        first_time = ds["time"].values[0]
        if hasattr(first_time, '__class__'):
            if 'Datetime360Day' in str(type(first_time)):
                cal = "360_day"
            elif 'DatetimeNoLeap' in str(type(first_time)):
                cal = "noleap"
    
    day_end_dec = 30 if "360" in (cal or "").lower() else 31
    t_start = _mk_cfdate(cal, int(start_year), 1, 1)
    t_end   = _mk_cfdate(cal, int(end_year), 12, day_end_dec)

    # keep original bounds for diagnostics
    time_orig = ds["time"].values
    if time_orig.size > 0:
        years_orig = np.array([tv.year for tv in time_orig], dtype=int)
        orig_start_year = int(years_orig[0])
        orig_end_year   = int(years_orig[-1])
    else:
        orig_start_year = orig_end_year = None

    ds = ds.sel(time=slice(t_start, t_end))
    time_vals = ds["time"].values
    nT = time_vals.size

    if nT == 0:
        raise ValueError(
            f"No data available for timespan {start_year}-{end_year} in {Path(nc_filename).name}. "
            f"Available range: {orig_start_year}-{orig_end_year if orig_start_year is not None else 'unknown'}"
        )

    if orig_start_year is not None and nT > 0:
        actual_years = np.array([tv.year for tv in time_vals], dtype=int)
        if actual_years.size > 0:
            actual_start = int(actual_years[0])
            actual_end   = int(actual_years[-1])
            if actual_start > start_year or actual_end < end_year:
                print(f"⚠️  Warning: Requested {start_year}-{end_year}, but got {actual_start}-{actual_end} from {Path(nc_filename).name}")

    with open(metafile2use, "rb") as f:
        meta = pickle.load(f)
    model_to_index = meta["model_to_index"]
    variables_X = meta["variables_X"]

    # parse model name
    base = os.path.basename(nc_filename)
    parts = base.split("_")
    model = parts[1] if len(parts) >= 2 else "unknown"
    model_idx = model_to_index.get(model, -1)

    # months (1..12) – robust across calendars using the month attribute
    month_val = np.array([tv.month for tv in time_vals], dtype=np.int16).astype(np.float32)
    sin_time = np.sin((month_val - 1) / 12 * 2 * np.pi).astype(np.float32)
    cos_time = np.cos((month_val - 1) / 12 * 2 * np.pi).astype(np.float32)

    X_cols: List[np.ndarray] = []
    pred_names: List[str] = []

    for var in variables_X:
        if var == "model_index":
            X_cols.append(np.full((nT, 1), model_idx, dtype=np.float32))
            pred_names.append("model_index")
        elif var == "month":
            X_cols.append(month_val.reshape(nT, 1))
            pred_names.append("month")
        elif var == "time":
            X_cols.append(sin_time.reshape(nT, 1)); pred_names.append("sin_time")
            X_cols.append(cos_time.reshape(nT, 1)); pred_names.append("cos_time")
        else:
            if var not in ds:
                X_cols.append(np.zeros((nT, 1), dtype=np.float32))
            else:
                x = ds[var].values.astype(np.float32)
                if x.ndim > 1:
                    x = np.mean(x, axis=tuple(range(1, x.ndim)))
                if var in ["tas_smoothed", "tas_globalmean"]:
                    x = (x - 273.15) / 10.0
                X_cols.append(x.reshape(nT, 1))
            pred_names.append(var)

    X = torch.from_numpy(np.concatenate(X_cols, axis=1)).float()

    # Extract years/months directly from cftime objects (avoid datetime conversion)
    years  = np.array([t.year  for t in time_vals], dtype=int)
    months = np.array([t.month for t in time_vals], dtype=int)
    start_month, end_month = int(months[0]), int(months[-1])

    return PredictorData(
        X=X,
        predictor_names=pred_names,
        model_names=list(model_to_index.keys()),
        variables_2predict=meta.get("variables", []),
        time_index=None,  # Keep None to avoid datetime conversion for years outside datetime range
        start_year=int(years[0]),
        end_year=int(years[-1]),
        start_month=start_month,
        end_month=end_month,
    )
# =============================================================================
# MAGICC → predictors (Option B)  [ADAPTED FOR WIDE MULTIINDEX PARQUET]
# =============================================================================

MONTHS3 = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MON2NUM = {m:i+1 for i,m in enumerate(MONTHS3)}

def _try_load_hadcrut_delta(ref_csv: Optional[str] = None) -> Optional[float]:
    csv_path = ref_csv or os.environ.get("HADCRUT5_SUMMARY_CSV") or HADCRUT5_SUMMARY_CSV
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        # Find anomaly column (contains "anomaly" or "deg" in name)
        cand_cols = [c for c in df.columns if "anomaly" in c.lower() or "deg" in c.lower()]
        if not cand_cols:
            return None
        vcol = cand_cols[0]
        # Find time/year column (check for "time" or "year" in name, case-insensitive)
        ycol_candidates = [c for c in df.columns if "time" in c.lower() or "year" in c.lower()]
        if not ycol_candidates:
            return None
        ycol = ycol_candidates[0]
        ser = df.set_index(ycol)[vcol].astype(float)
        # Calculate difference: mean(1995-2014) - mean(1850-1900)
        ref = ser.loc[1995:2014].mean()
        base = ser.loc[1850:1900].mean()
        return float(ref - base)
    except Exception:
        return None

def _extract_year_columns_from_wide(columns) -> Tuple[List[int], List[object]]:
    """Return ([years as int sorted], [their column labels in same order])."""
    years = []
    labels = []
    for c in columns:
        if isinstance(c, (int, np.integer)):
            years.append(int(c)); labels.append(c)
        else:
            s = str(c)
            if s.isdigit():
                years.append(int(s)); labels.append(c)
    if not years:
        raise RuntimeError("Could not find numeric year columns in MAGICC parquet.")
    order = np.argsort(years)
    return [years[i] for i in order], [labels[i] for i in order]

def _get_row_by_index_wide(
    df_wide: pd.DataFrame,
    *,
    scenario: str,
    run_id: int,
    variable: str,
    region: str = "World",
) -> Optional[pd.Series]:
    """
    Select a single Series (wide, year columns) by MultiIndex values.
    Index levels (as provided): climate_model, model, region, run_id, scenario, unit, variable
    We match on (region, run_id, scenario, variable). Unit is not forced.
    """
    levs = list(df_wide.index.names)
    mask = np.ones(len(df_wide), dtype=bool)

    def _and(level_name: str, value):
        nonlocal mask
        if level_name in levs:
            mask &= (df_wide.index.get_level_values(level_name) == value)

    _and("region", region)
    _and("run_id", run_id)
    _and("scenario", scenario)
    _and("variable", variable)

    sub = df_wide[mask]
    if sub.shape[0] == 0:
        return None
    # If multiple (different unit/climate_model/model), choose the first consistently
    return sub.iloc[0]

def _get_series_wide(
    df_wide: pd.DataFrame,
    *,
    scenario: str,
    run_id: int,
    variable: str,
    years_sorted: List[int],
    year_labels_sorted: List[object],
    prefer_natural: bool = False,
    region: str = "World",
) -> np.ndarray:
    """
    Extract an annual series for a given MAGICC variable at (scenario, run_id, region).
    If prefer_natural=True, first tries '<scenario>_runmodus_natural', then <scenario>.
    Otherwise the reverse.
    Returns float32 array length=len(years_sorted). Fills zeros if not found.
    """
    scenarios_try: List[str]
    if prefer_natural:
        scenarios_try = [f"{scenario}_runmodus_natural", scenario]
    else:
        scenarios_try = [scenario, f"{scenario}_runmodus_natural"]

    row = None
    for sc in scenarios_try:
        row = _get_row_by_index_wide(
            df_wide, scenario=sc, run_id=run_id, variable=variable, region=region
        )
        if row is not None:
            break

    if row is None:
        # Fill zeros, but warn once for truly missing variables
        return np.zeros((len(years_sorted),), dtype=np.float32)

    vals = row[year_labels_sorted].to_numpy(dtype=np.float32)
    if vals.shape[0] != len(years_sorted):
        # If some years missing, align by reindexing on the fly
        # (rare for these processed files; safe fallback)
        aligned = np.zeros((len(years_sorted),), dtype=np.float32)
        # row index labels might be str/int – map by label dictionary
        s = pd.Series(vals, index=year_labels_sorted)
        for i, col_label in enumerate(year_labels_sorted):
            if col_label in s.index:
                aligned[i] = float(s.loc[col_label])
        vals = aligned.astype(np.float32)
    return vals

def _annual_to_monthly(series_annual: np.ndarray) -> np.ndarray:
    """Replicate each annual value 12 times (360‑day calendar months)."""
    return np.repeat(series_annual.astype(np.float32), 12)

def _build_magicc_predictor_arrays_from_wide(
    df_wide: pd.DataFrame,
    *,
    scenario: str,
    run_id: int,
    era5_ref_K: Optional[float] = ERA5_GMT_REF_K,
    hadcrut_delta_C: Optional[float] = HADCRUT5_DELTA_1850_1900_TO_REF_C,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Tuple[int,int,int,int]]:
    """
    Assemble the predictor dict (names expected by meta['variables_X']) from the
    MAGICC *wide* MultiIndex parquet, keeping variables correlated via the same run_id.
    """
    # 1) Identify year columns
    years_sorted, year_labels_sorted = _extract_year_columns_from_wide(df_wide.columns)
    # Optional trimming window
    if year_start is not None or year_end is not None:
        ys = year_start if year_start is not None else int(years_sorted[0])
        ye = year_end if year_end is not None else int(years_sorted[-1])
    else:
        ys, ye = int(years_sorted[0]), int(years_sorted[-1])
    sy, ey = int(ys), int(ye)
    sm, em = 1, 12  # monthly replication

    # 2) Pull annual series we need
    # names seen in your file (units K or W/m^2) — case sensitive here:
    varnames = {
        "tas_delta": "Surface Air Temperature Change",                 # ∆K rel. 1850–1900
        "heat_uptake": "Heat Uptake",
        "erf_total": "Effective Radiative Forcing",                    # (not directly used, but present)
        "erf_aer": "Effective Radiative Forcing|Aerosols",
        "erf_co2": "Effective Radiative Forcing|CO2",
        "erf_ghg": "Effective Radiative Forcing|Greenhouse Gases",
        "erf_o3_strat": "Effective Radiative Forcing|Stratospheric Ozone",
        "erf_o3_trop": "Effective Radiative Forcing|Tropospheric Ozone",
        "erf_solar": "Effective Radiative Forcing|Solar",
        "erf_volc": "Effective Radiative Forcing|Volcanic",
    }

    # Prefer natural‑mode series for solar/volcanic (if present)
    s_tas = _get_series_wide(df_wide, scenario=scenario, run_id=run_id, variable=varnames["tas_delta"],
                             years_sorted=years_sorted, year_labels_sorted=year_labels_sorted, prefer_natural=False)
    s_heat = _get_series_wide(df_wide, scenario=scenario, run_id=run_id, variable=varnames["heat_uptake"],
                              years_sorted=years_sorted, year_labels_sorted=year_labels_sorted, prefer_natural=False)
    s_erf_strat = _get_series_wide(df_wide, scenario=scenario, run_id=run_id, variable=varnames["erf_o3_strat"],
                                   years_sorted=years_sorted, year_labels_sorted=year_labels_sorted, prefer_natural=False)
    s_erf_solar = _get_series_wide(df_wide, scenario=scenario, run_id=run_id, variable=varnames["erf_solar"],
                                   years_sorted=years_sorted, year_labels_sorted=year_labels_sorted, prefer_natural=True)
    s_erf_volc = _get_series_wide(df_wide, scenario=scenario, run_id=run_id, variable=varnames["erf_volc"],
                                  years_sorted=years_sorted, year_labels_sorted=year_labels_sorted, prefer_natural=True)
    s_erf_trop = _get_series_wide(df_wide, scenario=scenario, run_id=run_id, variable=varnames["erf_o3_trop"],
                                  years_sorted=years_sorted, year_labels_sorted=year_labels_sorted, prefer_natural=False)
    s_erf_ghg  = _get_series_wide(df_wide, scenario=scenario, run_id=run_id, variable=varnames["erf_ghg"],
                                  years_sorted=years_sorted, year_labels_sorted=year_labels_sorted, prefer_natural=False)
    s_erf_aer  = _get_series_wide(df_wide, scenario=scenario, run_id=run_id, variable=varnames["erf_aer"],
                                  years_sorted=years_sorted, year_labels_sorted=year_labels_sorted, prefer_natural=False)
    s_erf_co2  = _get_series_wide(df_wide, scenario=scenario, run_id=run_id, variable=varnames["erf_co2"],
                                  years_sorted=years_sorted, year_labels_sorted=year_labels_sorted, prefer_natural=False)

    # 3) tas_smoothed construction (per spec)
    # tas_smoothed = (MAGICC ∆C + ERA5 K_ref + HadCRUT5 ∆C(1850-1900→1995-2014) + MAGICC 'ALL' offset - 273.15)/10
    # ERA5 reference K
    era5 = float(ERA5_GMT_REF_K) if ERA5_GMT_REF_K is not None else float(os.environ.get("ERA5_GMT_REF_K", "288.0"))
    # HadCRUT delta
    if HADCRUT5_DELTA_1850_1900_TO_REF_C is not None:
        hadc = float(HADCRUT5_DELTA_1850_1900_TO_REF_C)
    else:
        hadc_loaded = _try_load_hadcrut_delta()
        if hadc_loaded is not None:
            hadc = hadc_loaded
        elif os.environ.get("HADCRUT5_DELTA_C") is not None:
            hadc = float(os.environ.get("HADCRUT5_DELTA_C"))
        else:
            csv_path = os.environ.get("HADCRUT5_SUMMARY_CSV") or HADCRUT5_SUMMARY_CSV
            if csv_path:
                raise FileNotFoundError(f"HadCRUT5 file not found or could not be loaded: {csv_path}")
            else:
                raise RuntimeError("HadCRUT5 delta value is required but not provided. Set HADCRUT5_DELTA_1850_1900_TO_REF_C, HADCRUT5_SUMMARY_CSV, or HADCRUT5_DELTA_C environment variable.")
    # 'ALL' offset = value at 1750 minus mean(1850..1900), if available
    # Compute this BEFORE slicing to ensure we have access to the full year range
    try:
        y_full = np.array(years_sorted, dtype=int)
        v_full = s_tas.astype(np.float64)
        if (y_full == 1750).any() and ((y_full >= 1850) & (y_full <= 1900)).any():
            all_offset = float(v_full[y_full == 1750].mean() - v_full[(y_full >= 1850) & (y_full <= 1900)].mean())
        else:
            # Cannot compute from data - check environment variable
            env_offset = os.environ.get("MAGICC_ALL_OFFSET_C")
            if env_offset is not None:
                all_offset = float(env_offset)
            else:
                raise RuntimeError(
                    "MAGICC 'ALL' offset cannot be computed from data (missing 1750 or 1850-1900 data). "
                    "Set MAGICC_ALL_OFFSET_C environment variable to provide the offset value."
                )
    except RuntimeError:
        raise
    except Exception as e:
        # If computation failed, check environment variable
        env_offset = os.environ.get("MAGICC_ALL_OFFSET_C")
        if env_offset is not None:
            all_offset = float(env_offset)
        else:
            raise RuntimeError(
                f"MAGICC 'ALL' offset computation failed: {e}. "
                "Set MAGICC_ALL_OFFSET_C environment variable to provide the offset value."
            )

    # Slice all annual series to [sy..ey] AFTER computing offset
    yarr = np.array(years_sorted, dtype=int)
    mask = (yarr >= sy) & (yarr <= ey)
    def _sl(x): 
        x = x if isinstance(x, np.ndarray) else np.asarray(x, dtype=np.float32)
        return x[mask].astype(np.float32)
    s_tas, s_heat = _sl(s_tas), _sl(s_heat)
    s_erf_strat, s_erf_solar, s_erf_volc = _sl(s_erf_strat), _sl(s_erf_solar), _sl(s_erf_volc)
    s_erf_trop, s_erf_ghg, s_erf_aer, s_erf_co2 = _sl(s_erf_trop), _sl(s_erf_ghg), _sl(s_erf_aer), _sl(s_erf_co2)

    tas_smoothed_annual = ((s_tas + era5 + hadc + all_offset) - 273.15) / 10.0

    # 4) Build predictor dict (annual), then convert to monthly
    predictors_annual = {
        "tas_smoothed": tas_smoothed_annual.astype(np.float32),
        "rtmt_smoothed": s_heat.astype(np.float32),
        "stratO3_ERF": s_erf_strat.astype(np.float32),
        "sol_ERF": s_erf_solar.astype(np.float32),
        "other_ERF": np.zeros_like(s_erf_solar, dtype=np.float32),
        "volc_ERF": s_erf_volc.astype(np.float32),
        "nat_ERF": (s_erf_solar + s_erf_volc).astype(np.float32),
        "totalO3_ERF": s_erf_trop.astype(np.float32),
        "GHG_ERF": s_erf_ghg.astype(np.float32),
        "aer_ERF": s_erf_aer.astype(np.float32),
        "CO2_ERF": s_erf_co2.astype(np.float32),
    }

    predictors_monthly = {k: _annual_to_monthly(v) for k, v in predictors_annual.items()}
    return predictors_monthly, (sy, sm, ey, em)

def _resolve_effect_model_indices(scheme, model_to_index: Dict[str,int], n_draws: int) -> List[int]:
    idx_by_name = model_to_index
    indices_all = sorted(v for v in idx_by_name.values() if v != 0)
    indices_all_plus_era5 = sorted(v for v in idx_by_name.values())  # includes 0 (ERA5)
    def as_idx(name_or_idx):
        if isinstance(name_or_idx, (int, np.integer)):
            if name_or_idx in indices_all_plus_era5:
                return int(name_or_idx)
            raise ValueError(f"Effect model index {name_or_idx} not recognized.")
        if isinstance(name_or_idx, str):
            if name_or_idx == "ERA5":
                return 0
            if name_or_idx in idx_by_name:
                return idx_by_name[name_or_idx]
            raise ValueError(f"Effect model name {name_or_idx!r} not in meta['model_to_index']")
        raise TypeError("Invalid EFFECT_MODEL_SCHEME element")
    if isinstance(scheme, str):
        scheme_lower = scheme.strip().lower()
        if scheme_lower == "random" or scheme_lower == "all":
            # "all" and "Random" are equivalent: randomly choose from all non-zero indices
            return [random.choice(indices_all) for _ in range(n_draws)]
        elif scheme_lower == "allplusera5":
            # "allplusERA5": randomly choose from all indices including ERA5 (index 0)
            return [random.choice(indices_all_plus_era5) for _ in range(n_draws)]
        else:
            idx = as_idx(scheme); return [idx] * n_draws
    elif isinstance(scheme, (int, np.integer)):
        idx = as_idx(int(scheme)); return [idx] * n_draws
    elif isinstance(scheme, list) and scheme:
        pool = [as_idx(x) for x in scheme]
        return [random.choice(pool) for _ in range(n_draws)]
    else:
        raise ValueError("EFFECT_MODEL_SCHEME invalid/empty")

def _resolve_cmip6_calibration_indices(scheme, model_to_index: Dict[str,int]) -> List[int]:
    """
    Resolve which CMIP6 calibration indices to use for Option C based on EFFECT_MODEL_SCHEME_C.
    
    Args:
        scheme: 'all' | 'allplusERA5' | List[int] | List[str]
        model_to_index: Dictionary mapping model names to indices
    
    Returns:
        Sorted list of CMIP6 calibration indices
        - 'all': excludes ERA5 index 0
        - 'allplusERA5': includes ERA5 index 0
    """
    idx_by_name = model_to_index
    indices_all = sorted({int(v) for v in idx_by_name.values() if int(v) != 0})
    indices_all_plus_era5 = sorted({int(v) for v in idx_by_name.values()})  # includes 0 (ERA5)
    
    def as_idx(name_or_idx, allow_era5: bool = False):
        if isinstance(name_or_idx, (int, np.integer)):
            idx = int(name_or_idx)
            if idx == 0 and not allow_era5:
                raise ValueError("ERA5 index (0) cannot be used as a CMIP6 calibration index (use 'allplusERA5' to include it)")
            if allow_era5:
                if idx in indices_all_plus_era5:
                    return idx
            else:
                if idx in indices_all:
                    return idx
            raise ValueError(f"CMIP6 calibration index {idx} not recognized.")
        if isinstance(name_or_idx, str):
            if name_or_idx == "ERA5":
                if not allow_era5:
                    raise ValueError("ERA5 cannot be used as a CMIP6 calibration (use 'allplusERA5' to include it)")
                return 0
            if name_or_idx in idx_by_name:
                idx = idx_by_name[name_or_idx]
                if idx == 0 and not allow_era5:
                    raise ValueError(f"Model {name_or_idx} maps to ERA5 index (0), which cannot be used (use 'allplusERA5' to include it)")
                return idx
            raise ValueError(f"Model name {name_or_idx!r} not in meta['model_to_index']")
        raise TypeError("Invalid EFFECT_MODEL_SCHEME_C element")
    
    if isinstance(scheme, str):
        scheme_lower = scheme.strip().lower()
        if scheme_lower == "all":
            return indices_all
        elif scheme_lower == "allplusera5":
            return indices_all_plus_era5
        else:
            # Single model name
            idx = as_idx(scheme, allow_era5=False)
            return [idx]
    elif isinstance(scheme, list) and scheme:
        # List of indices or names
        # Check if any element is "ERA5" or 0 to determine if ERA5 is allowed
        allow_era5 = any(
            (isinstance(x, str) and x.strip().upper() == "ERA5") or 
            (isinstance(x, (int, np.integer)) and int(x) == 0)
            for x in scheme
        )
        resolved = [as_idx(x, allow_era5=allow_era5) for x in scheme]
        # Remove duplicates while preserving order
        seen = set()
        unique_resolved = []
        for idx in resolved:
            if idx not in seen:
                seen.add(idx)
                unique_resolved.append(idx)
        return sorted(unique_resolved)
    else:
        raise ValueError("EFFECT_MODEL_SCHEME_C must be 'all', 'allplusERA5', a list of indices, or a list of model names")

def build_predictors_from_magicc(
    df_wide: pd.DataFrame,
    scenario: str,
    magicc_member: int,
    model_to_index: Dict[str,int],
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> PredictorData:
    predictors_map, (sy, sm, ey, em) = _build_magicc_predictor_arrays_from_wide(
        df_wide, scenario=scenario, run_id=magicc_member,
        year_start=year_start, year_end=year_end
    )

    variables_X = META["variables_X"]
    # Monthly length: (years * 12)
    nT = len(next(iter(predictors_map.values())))
    # Month 1..12 repeating
    months = np.array([((i % 12) + 1) for i in range(nT)], dtype=np.float32)
    sin_time = np.sin((months - 1) / 12 * 2 * np.pi).astype(np.float32)
    cos_time = np.cos((months - 1) / 12 * 2 * np.pi).astype(np.float32)

    X_cols: List[np.ndarray] = []
    pred_names: List[str] = []

    for var in variables_X:
        if var == "model_index":
            # Placeholder: set per draw to chosen effect_idx later
            X_cols.append(np.full((nT, 1), -1.0, dtype=np.float32))
            pred_names.append("model_index")
        elif var == "month":
            X_cols.append(months.reshape(nT, 1))
            pred_names.append("month")
        elif var == "time":
            X_cols.append(sin_time.reshape(nT, 1)); pred_names.append("sin_time")
            X_cols.append(cos_time.reshape(nT, 1)); pred_names.append("cos_time")
        else:
            arr = predictors_map.get(var, np.zeros((nT,), dtype=np.float32))
            X_cols.append(arr.reshape(nT, 1))
            pred_names.append(var)

    X = torch.from_numpy(np.concatenate(X_cols, axis=1)).float()

    # Keep time_index as None to avoid datetime conversion for years outside datetime range
    # (e.g., 2500). All time information is preserved in start_year, end_year, start_month, end_month

    return PredictorData(
        X=X,
        predictor_names=pred_names,
        model_names=list(model_to_index.keys()),
        variables_2predict=META.get("variables", []),
        time_index=None,  # Avoid pd.to_datetime conversion for years outside datetime range
        start_year=sy,
        end_year=ey,
        start_month=sm,
        end_month=em,
    )


def _assemble_predictors_from_arrays(
    *,
    year: np.ndarray,
    month: np.ndarray,
    series: Dict[str, np.ndarray],
    model_to_index: Dict[str, int],
    model_index_name: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> PredictorData:
    """
    Shared helper: slice to requested years, build PredictorData using META['variables_X'].
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

    nT = int(year.shape[0])
    months = month.astype(np.float32)
    sin_time = np.sin((months - 1.0) / 12.0 * 2.0 * np.pi).astype(np.float32)
    cos_time = np.cos((months - 1.0) / 12.0 * 2.0 * np.pi).astype(np.float32)

    variables_X = META["variables_X"]
    model_index_val = float(model_to_index.get(model_index_name, 0))

    X_cols: List[np.ndarray] = []
    pred_names: List[str] = []

    for var in variables_X:
        if var == "model_index":
            arr = np.full((nT, 1), model_index_val, dtype=np.float32)
            X_cols.append(arr)
            pred_names.append("model_index")
        elif var == "month":
            X_cols.append(months.reshape(nT, 1))
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
        model_names=list(model_to_index.keys()),
        variables_2predict=META.get("variables", []),
        time_index=None,
        start_year=start_year,
        end_year=end_year,
        start_month=start_month,
        end_month=end_month,
    )


def build_predictors_from_spliced_h5(
    predictors_h5: Union[str, Path],
    model_to_index: Dict[str, int],
    model_index_name: str = "ERA5",
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> PredictorData:
    """
    Build PredictorData directly from the ERA5‑spliced HDF5 predictor time series
    written by 616_*.py. Supports optional trimming to [year_start, year_end].
    """
    import h5py  # local import to avoid hard dependency if unused

    predictors_h5 = Path(predictors_h5)
    if not predictors_h5.exists():
        raise FileNotFoundError(f"Predictor HDF5 file not found: {predictors_h5}")

    with h5py.File(predictors_h5, "r") as h5:
        year = np.asarray(h5["year"], dtype=np.int32).ravel()
        month = np.asarray(h5["month"], dtype=np.int32).ravel()
        series = {}
        for var in META["variables_X"]:
            if var in ("model_index", "month", "time"):
                continue
            if var in h5:
                series[var] = np.asarray(h5[var], dtype=np.float32).ravel()
        return _assemble_predictors_from_arrays(
            year=year,
            month=month,
            series=series,
            model_to_index=model_to_index,
            model_index_name=model_index_name,
            year_start=year_start,
            year_end=year_end,
        )


def build_predictors_from_spliced_file(
    predictor_file: Union[str, Path],
    model_to_index: Dict[str, int],
    *,
    model_index_name: str = "ERA5",
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> PredictorData:
    """
    Load ERA5‑spliced predictors from either HDF5 (.h5) or CSV, then assemble PredictorData.
    """
    path = Path(predictor_file)
    if not path.exists():
        raise FileNotFoundError(f"Predictor file not found: {path}")

    if path.suffix.lower() == ".h5":
        return build_predictors_from_spliced_h5(
            path,
            model_to_index,
            model_index_name=model_index_name,
            year_start=year_start,
            year_end=year_end,
        )

    # CSV fallback (no heavy pandas dependency)
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
    return _assemble_predictors_from_arrays(
        year=np.asarray(cols["year"], dtype=np.int32),
        month=np.asarray(cols["month"], dtype=np.int32),
        series=series,
        model_to_index=model_to_index,
        model_index_name=model_index_name,
        year_start=year_start,
        year_end=year_end,
    )


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


def _discover_spliced_predictor_files(
    base_dir: Union[str, Path],
    scenarios: Sequence[str],
    workflow: str = "AR6",
    runmodus: str = "all",
) -> Dict[str, Dict[int, Path]]:
    """
    Return {scenario: {run_id: file_path}} for ERA5-spliced predictor files.
    
    New directory structure:
      {base_dir}/{scenario}/{AR6|AR7}/runmodus_{all|natural|aerosol|anthropogenic}/
    
    Args:
        base_dir: Base directory containing scenario subdirectories
        scenarios: List of scenario names
        workflow: 'AR6' | 'AR7' | 'all' (if 'all', searches both AR6 and AR7)
        runmodus: 'all' | 'natural' | 'aerosol' | 'anthropogenic'
    
    Raises with guidance if a requested scenario is missing.
    """
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(
            f"ERA5-spliced predictor directory not found: {base} "
            "(generate via notebooks/616_*.py in gcmmagicc)."
        )
    
    # Determine which workflows to search
    workflows_to_search = []
    if workflow.upper() == "ALL":
        workflows_to_search = ["AR6", "AR7"]
    else:
        workflows_to_search = [workflow.upper()]
    
    # Normalize runmodus for directory name
    runmodus_dir = f"runmodus_{runmodus}"
    
    mapping: Dict[str, Dict[int, Path]] = {}
    requested = list(scenarios) if scenarios else [p.name for p in base.iterdir() if p.is_dir()]
    
    for scen in requested:
        scen_norm = scen.replace("clean_", "", 1) if scen.startswith("clean_") else scen
        run_map: Dict[int, Path] = {}
        
        # Search in each workflow directory
        for wf in workflows_to_search:
            scen_dir = base / scen_norm / wf / runmodus_dir
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
            # Check if scenario directory exists at all
            scen_base_dir = base / scen_norm
            if not scen_base_dir.is_dir():
                raise FileNotFoundError(
                    f"No predictors for scenario '{scen}' in {base}. "
                    f"Expected structure: {base}/{scen_norm}/{{AR6|AR7}}/{runmodus_dir}/"
                )
            # Check if workflow directories exist
            workflows_found = [wf for wf in workflows_to_search if (base / scen_norm / wf).is_dir()]
            if not workflows_found:
                raise FileNotFoundError(
                    f"No workflow directories found for scenario '{scen}' in {base}/{scen_norm}. "
                    f"Expected: AR6 or AR7 subdirectories."
                )
            # Check if runmodus directory exists
            runmodus_found = False
            for wf in workflows_to_search:
                if (base / scen_norm / wf / runmodus_dir).is_dir():
                    runmodus_found = True
                    break
            if not runmodus_found:
                raise FileNotFoundError(
                    f"No predictor files found for scenario '{scen}' with workflow={workflow}, runmodus={runmodus} "
                    f"in {base}/{scen_norm}/. Expected: {runmodus_dir} subdirectory."
                )
        else:
            mapping[scen_norm] = run_map
    
    return mapping


def _peek_spliced_year_span(predictor_file: Path) -> Tuple[int, int]:
    """
    Read the first/last year quickly from a CSV (cheap) or HDF5 (using h5py) file.
    """
    if predictor_file.suffix.lower() == ".h5":
        import h5py
        with h5py.File(predictor_file, "r") as h5:
            year = np.asarray(h5["year"], dtype=np.int32).ravel()
            return int(year[0]), int(year[-1])
    # CSV fallback: read only relevant lines
    with predictor_file.open("r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise RuntimeError(f"Could not read years from {predictor_file}")
    if lines[0].lower().startswith("year"):
        lines = lines[1:]
    if not lines:
        raise RuntimeError(f"No data rows found in {predictor_file}")
    def _parse_year(line: str) -> int:
        parts = line.split(",")
        return int(parts[0])
    return _parse_year(lines[0]), _parse_year(lines[-1])

# =============================================================================
# Emulation runner (adapted from 300_*)
# =============================================================================

def run_gcmagicc(
    predictor_data: PredictorData,
    *,
    dependence: bool = True,
    usebias_model=None,
    useeffect_model=None,
    device: Optional[str] = None,
    force_gpu: bool = False,
    amp: bool = False,
):
    requested_device = normalize_device_string(device) if device is not None else DEFAULT_DEVICE
    if requested_device is None:
        requested_device = DEFAULT_DEVICE
    # Only downgrade to CPU if CUDA is unavailable AND we're not forcing GPU
    if isinstance(requested_device, str) and requested_device.startswith("cuda") and not _cuda_available() and not force_gpu:
        print(f"⚠️  Requested device '{requested_device}' but CUDA is unavailable; using CPU instead.")
        requested_device = "cpu"
    elif isinstance(requested_device, str) and requested_device.startswith("cuda") and not _cuda_available() and force_gpu:
        print(f"⚠️  Forcing GPU usage for '{requested_device}' despite CUDA availability check.")
    # For GPU, we'll handle numpy conversion ourselves to ensure CPU transfer
    # For CPU, we can use asnumpy=True for efficiency
    use_gpu = isinstance(requested_device, str) and requested_device.startswith("cuda") and (_cuda_available() or force_gpu)

    # Optional guard: if the chosen GPU is low on free memory, fall back to CPU unless force_gpu is set
    if use_gpu and MIN_FREE_MEMORY_GB > 0 and not force_gpu:
        try:
            idx_check = int(requested_device.split(":", 1)[1]) if ":" in requested_device else 0
        except Exception:
            idx_check = 0
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx_check)
            free_gb = free_bytes / (1024 ** 3)
            if free_gb < MIN_FREE_MEMORY_GB:
                print(f"⚠️  GPU {idx_check} has only {free_gb:.2f}GiB free (<{MIN_FREE_MEMORY_GB}GiB). Falling back to CPU. "
                      f"Override threshold with GCMAGICC_MIN_FREE_MEMORY_GB or set --force-gpu to bypass.")
                requested_device = "cpu"
                use_gpu = False
        except Exception:
            pass
    
    # Always pass x as a tensor - sample_from_combined_model handles numpy conversion internally based on asnumpy flag
    # Ensure it's on the correct device (CPU for CPU mode, or appropriate GPU device)
    asnumpy_flag = not use_gpu
    if isinstance(predictor_data.X, torch.Tensor):
        # Ensure tensor is on CPU when using CPU mode
        if not use_gpu:
            x_input = predictor_data.X.detach().cpu()
        else:
            x_input = predictor_data.X
    else:
        # If it's already numpy, convert to tensor
        x_input = torch.from_numpy(np.asarray(predictor_data.X))
        if not use_gpu:
            x_input = x_input.cpu()
    
    kwargs = {
        "x": x_input,
        "device": requested_device,
        "dirname": GCMagiccmodels + "/",
        "DATE": DATEOFMETAFILE,
        "dependence": dependence,
        "rectangular": True,
        # Keep grid sizes consistent with generate_coordinate_grids()
        "nlat": N_LAT,
        "nsub": 1,
        # Healpy interpolation inside sample_from_combined_model expects NumPy
        # For GPU, set False and handle conversion ourselves to ensure CPU transfer
        "asnumpy": asnumpy_flag,
    }
    sig = inspect.signature(sample_from_combined_model)
    if "usebias_model" in sig.parameters:
        kwargs["usebias_model"] = usebias_model
    if "useeffect_model" in sig.parameters:
        kwargs["useeffect_model"] = useeffect_model

    # Ensure GPU-specific optimisations are enabled when running on CUDA
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

    # Optional debug print of key kwargs to verify memoryoptimized is propagated
    if DEBUG_SAMPLE_KWARGS:
        print(
            "🔧 sample_from_combined_model kwargs:",
            {k: kwargs.get(k) for k in ("device", "memoryoptimized", "enable_gpu_optimizations", "asnumpy")}
        )

    # Explicitly set CUDA device if used
    if use_gpu:
        try:
            idx = int(requested_device.split(":", 1)[1]) if ":" in requested_device else torch.cuda.current_device()
            # Aggressive cleanup BEFORE allocation to prevent OOM
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(idx)
        except Exception:
            idx = 0
        try:
            torch.cuda.set_device(idx)
        except Exception:
            pass
        # Optional AMP on CUDA
        if amp:
            amp_dtype = _amp_dtype_cuda()
            with torch.autocast(device_type="cuda", dtype=amp_dtype), torch.inference_mode():
                out = sample_from_combined_model(**kwargs)
        else:
            with torch.inference_mode():
                out = sample_from_combined_model(**kwargs)
        
        # Ensure output is moved to CPU before numpy conversion
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
        elif hasattr(out, 'cpu'):  # Handle any tensor-like object
            out = out.cpu().numpy()
        
        try:
            torch.cuda.synchronize(idx)
        except Exception:
            pass
        try:
            mem_alloc = torch.cuda.memory_allocated(idx) / (1024 ** 2)
            mem_reserved = torch.cuda.memory_reserved(idx) / (1024 ** 2)
            print(f"   • GPU device {idx} usage: allocated {mem_alloc:.1f} MiB, reserved {mem_reserved:.1f} MiB")
        except Exception:
            pass
        # Aggressive memory cleanup for GPU
        try:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(idx)
        except Exception:
            pass
    else:
        with torch.inference_mode():
            out = sample_from_combined_model(**kwargs)
        # Ensure numpy for CPU case too (in case asnumpy didn't work)
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
    
    return out  # expected shape: [T, n_features, nlat, nlon] in lat-lon already

# =============================================================================
# Segments: region masks, weights, and writers
# =============================================================================

def _area_weights_coslat(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Simple cos(lat) weights normalized to mean 1 over the globe.
    """
    cosw = np.cos(np.deg2rad(lats))[:, None] * np.ones((len(lats), len(lons)), dtype=np.float64)
    return (cosw / cosw.mean()).astype(np.float32)

def _mask_for_region(region: str, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask (lat, lon) for region selection.
    - 'global' -> all True
    - Pre-computed mask file (if available) -> loads from data/regionmasks/
    - IPCC AR6 region name -> via regionmask
    - ISO3 country -> via geopandas naturalearth_lowres
    
    Priority order:
    1. 'global' -> returns all True
    2. Pre-computed mask file (if exists)
    3. regionmask (AR6 regions)
    4. geopandas (ISO3 countries)
    """
    if region.lower() == "global":
        return np.ones((len(lats), len(lons)), dtype=bool)

    # Try loading pre-computed mask first (avoids geopandas/regionmask dependency)
    try:
        # Generate a hash/fingerprint from grid geometry
        grid_hash = hash((len(lats), len(lons), float(lats[0]), float(lats[-1]), 
                          float(lons[0]), float(lons[-1]), LON_CONVENTION))
        region_normalized = region.upper().replace(" ", "_")
        mask_dir = Path(get_data_path("regionmasks"))
        mask_file = mask_dir / f"{region_normalized}_nlat{len(lats)}_nlon{len(lons)}_lon{LON_CONVENTION}.npz"
        
        if mask_file.exists():
            data = np.load(str(mask_file))
            saved_lats = data["lats"]
            saved_lons = data["lons"]
            mask = data["mask"]
            
            # Verify grid matches
            if (len(saved_lats) == len(lats) and len(saved_lons) == len(lons) and
                np.allclose(saved_lats, lats, atol=1e-6) and 
                np.allclose(saved_lons, lons, atol=1e-6)):
                return mask.astype(bool)
            else:
                print(f"⚠️  Pre-computed mask for {region} exists but grid doesn't match; recomputing...")
    except Exception as e:
        # If loading fails, continue to fallback methods
        pass

    # Try AR6 regions (regionmask)
    try:
        import regionmask  # type: ignore
        R = regionmask.defined_regions.ar6.all
        # match by name (case-insensitive, allow partial)
        cand = [i for i, n in enumerate(R.names) if n.lower() == region.lower()]
        if not cand:
            # allow matching synonyms (e.g., "Mediterranean")
            cand = [i for i, n in enumerate(R.names) if region.lower() in n.lower()]
        if cand:
            mask = R.mask(lons, lats)  # returns (lat,lon)
            return (mask == cand[0]).values
    except Exception:
        pass

    # Try ISO3 with geopandas
    try:
        import geopandas as gpd  # type: ignore
        from shapely.geometry import Point  # type: ignore
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        iso_col = "iso_a3"
        if iso_col not in world.columns:
            raise RuntimeError("naturalearth_lowres missing iso_a3")
        row = world[world[iso_col].str.upper() == region.upper()]
        if row.empty:
            raise RuntimeError(f"ISO3 {region} not found in naturalearth_lowres.")
        geom = row.iloc[0].geometry
        # vectorized point-in-polygon
        llon, llat = np.meshgrid(lons, lats)
        pts = gpd.GeoSeries.from_xy(llon.ravel(), llat.ravel(), crs=4326)
        inside = pts.within(geom) | pts.intersects(geom)
        return inside.values.reshape(len(lats), len(lons))
    except Exception as e:
        raise RuntimeError(
            f"Region '{region}' not recognized as AR6 or ISO3. "
            f"Pre-computed mask not found. Install 'regionmask' for AR6 or 'geopandas' for ISO3. "
            f"Alternatively, run the mask generation script to create pre-computed masks. Error: {e}"
        )

def _month_filter_indices(time_years: np.ndarray, time_months: np.ndarray, period: Tuple[int,int], months_spec) -> np.ndarray:
    y0, y1 = period
    in_year = (time_years >= y0) & (time_years <= y1)
    if isinstance(months_spec, str) and months_spec.lower() == "all":
        in_month = np.ones_like(in_year, dtype=bool)
    else:
        months_list = months_spec
        mnums = [MON2NUM[m] if isinstance(m, str) else int(m) for m in months_list]
        in_month = np.isin(time_months, np.array(mnums, dtype=int))
    return in_year & in_month

def _make_time_axis(start_year:int, start_month:int, end_year:int, end_month:int) -> np.ndarray:
    out = []
    for y in range(start_year, end_year + 1):
        m0 = start_month if y == start_year else 1
        m1 = end_month if y == end_year else 12
        for m in range(m0, m1 + 1):
            out.append(cftime.Datetime360Day(y, m, 15, has_year_zero=True))
    return np.array(out)

def _segment_key(seg: dict) -> str:
    per = seg["period"]
    months = seg["months"] if isinstance(seg["months"], str) else "-".join(seg["months"])
    return f"{seg['variable']}__{seg['region']}__{seg['operation']}__{per[0]}-{per[1]}__{months}"

def _write_segment_dataset_zarr(
    ds: xr.Dataset,
    *,
    root_dir: str,
    store_name: str,
    group: str,
    mode: str = "a",
    consolidated: bool = False,
):
    path = os.path.join(root_dir, store_name)
    os.makedirs(root_dir, exist_ok=True)
    ds.to_zarr(path, mode=mode, group=group, consolidated=consolidated)

# =============================================================================
# Core: Take yhval and OUTPUTDICTLIST → save per segment
# =============================================================================

def apply_output_specs_and_write(
    *,
    yhval: Union[torch.Tensor, np.ndarray],
    predictors: PredictorData,
    lats: np.ndarray,
    lons: np.ndarray,
    output_specs: List[dict],
    run_meta: Dict[str, Union[str,int]],
    variables_2predict: List[str],
    root_dir: str = OUTPUT_STORE_DIR,
    store_name: str = OUTPUT_STORE_NAME,
):
    # ensure numpy
    if isinstance(yhval, torch.Tensor):
        arr = yhval.detach().cpu().numpy()
    else:
        arr = yhval
    # arr shape: [T, nvar, lat, lon]
    assert arr.ndim == 4, f"yhval shape expected (T,V,lat,lon), got {arr.shape}"
    T, V, nlat, nlon = arr.shape
    assert nlat == len(lats) and nlon == len(lons), "Grid mismatch"

    # Time axis and helpers
    time_cf = _make_time_axis(predictors.start_year, predictors.start_month, predictors.end_year, predictors.end_month)
    years = np.array([t.year for t in time_cf], dtype=int)
    months = np.array([t.month for t in time_cf], dtype=int)

    # weights
    w = _area_weights_coslat(lats, lons).astype(np.float32)

    # loop segments (intersect each segment with the predictor time range)
    pred_span = (int(predictors.start_year), int(predictors.end_year))
    for seg in output_specs:
        var = seg["variable"]
        try:
            v_idx = variables_2predict.index(var)
        except ValueError:
            print(f"⚠️ Skipping segment {_segment_key(seg)}: variable {var!r} not in model outputs {variables_2predict}")
            continue

        # Intersect requested period with available predictor span
        seg_period = tuple(seg["period"])
        ov = _intersect((int(seg_period[0]), int(seg_period[1])), pred_span)
        if ov is None:
            # No overlap with this run window; skip quietly
            continue
        # time filter on the overlapped window
        mask_t = _month_filter_indices(years, months, ov, seg["months"])
        if not mask_t.any():
            # Should not normally happen after intersection; keep a guard.
            seg_y0, seg_y1 = ov
            print(f"⚠️ No time points selected for overlapped segment {var} {ov}; skipping.")
            continue

        sub = arr[mask_t, v_idx, :, :]  # (t, lat, lon)
        t_sel = time_cf[mask_t]

        # region mask
        region = seg["region"]
        rmask = _mask_for_region(region, lats, lons)  # (lat,lon)

        operation = seg["operation"].strip().lower()
        if operation == "area-weighted average":
            # compute time series average (monthly resolution)
            # mask weights
            ww = w * rmask.astype(np.float32)
            ww_sum = ww.sum()
            if ww_sum <= 0:
                print(f"⚠️ Region {region} produced zero area; skipping.")
                continue
            ts = (sub * ww[None, :, :]).sum(axis=(1, 2)) / ww_sum  # (t,)
            ds = xr.Dataset(
                coords={"time": ("time", t_sel)},
                data_vars={
                    var: (("time",), ts.astype(np.float32)),
                },
                attrs={"description": "GCMagicc segment; area-weighted regional average (monthly)"},
            )
            # annotate metadata
            for k, v in run_meta.items():
                ds.attrs[str(k)] = str(v)
            seg_local = dict(seg); seg_local["period"] = ov
            ds.attrs["segment_spec"] = json.dumps(seg_local)

            group = f"runs/{run_meta['run_id']}/{_segment_key(seg)}"
            _write_segment_dataset_zarr(ds, root_dir=root_dir, store_name=store_name, group=group, mode="a")

        elif operation == "area-weighted annual average":
            # compute area-weighted average for each month, then average over months within each year
            ww = w * rmask.astype(np.float32)
            ww_sum = ww.sum()
            if ww_sum <= 0:
                print(f"⚠️ Region {region} produced zero area; skipping.")
                continue
            # First compute monthly area-weighted averages
            ts_monthly = (sub * ww[None, :, :]).sum(axis=(1, 2)) / ww_sum  # (t,)
            
            # Group by year and average over months within each year
            years_sel = np.array([t.year for t in t_sel], dtype=int)
            unique_years = np.unique(years_sel)
            ts_annual = np.zeros(len(unique_years), dtype=np.float32)
            time_annual = []
            
            for i, year in enumerate(unique_years):
                year_mask = (years_sel == year)
                ts_annual[i] = ts_monthly[year_mask].mean()
                # Use mid-year timestamp (e.g., July 1st)
                time_annual.append(cftime.Datetime360Day(year, 7, 15, has_year_zero=True))
            
            ds = xr.Dataset(
                coords={"time": ("time", np.array(time_annual))},
                data_vars={
                    var: (("time",), ts_annual),
                },
                attrs={"description": "GCMagicc segment; area-weighted regional annual average"},
            )
            # annotate metadata
            for k, v in run_meta.items():
                ds.attrs[str(k)] = str(v)
            seg_local = dict(seg); seg_local["period"] = ov
            ds.attrs["segment_spec"] = json.dumps(seg_local)

            group = f"runs/{run_meta['run_id']}/{_segment_key(seg)}"
            _write_segment_dataset_zarr(ds, root_dir=root_dir, store_name=store_name, group=group, mode="a")

        elif operation == "area-weighted time average":
            # compute area-weighted average over space, then average over all time points (single scalar)
            ww = w * rmask.astype(np.float32)
            ww_sum = ww.sum()
            if ww_sum <= 0:
                print(f"⚠️ Region {region} produced zero area; skipping.")
                continue
            # First compute monthly area-weighted averages
            ts_monthly = (sub * ww[None, :, :]).sum(axis=(1, 2)) / ww_sum  # (t,)
            # Then average over all time points
            scalar_value = ts_monthly.mean().astype(np.float32)
            
            # Verify we're creating a scalar (no time dimension)
            # sub shape: (t, lat, lon), ts_monthly shape: (t,), scalar_value: scalar
            assert ts_monthly.ndim == 1, f"ts_monthly should be 1D, got shape {ts_monthly.shape}"
            assert np.isscalar(scalar_value) or (isinstance(scalar_value, np.ndarray) and scalar_value.ndim == 0), \
                f"scalar_value should be scalar, got shape {scalar_value.shape if hasattr(scalar_value, 'shape') else 'N/A'}"
            
            # Create dataset with no time dimension (single scalar)
            ds = xr.Dataset(
                data_vars={
                    var: ((), scalar_value),
                },
                attrs={
                    "description": "GCMagicc segment; area-weighted regional and time-averaged (single scalar)",
                    "time_period_start": str(t_sel[0]),
                    "time_period_end": str(t_sel[-1]),
                    "n_time_points_averaged": len(ts_monthly),  # Debug: number of months averaged
                    "operation_verified": "area-weighted time average",
                },
            )
            # annotate metadata
            for k, v in run_meta.items():
                ds.attrs[str(k)] = str(v)
            seg_local = dict(seg); seg_local["period"] = ov
            ds.attrs["segment_spec"] = json.dumps(seg_local)

            group = f"runs/{run_meta['run_id']}/{_segment_key(seg)}"
            _write_segment_dataset_zarr(ds, root_dir=root_dir, store_name=store_name, group=group, mode="a")

        elif operation == "grid-points":
            # compress region to points
            idxs = np.argwhere(rmask)
            if idxs.size == 0:
                print(f"⚠️ Region {region}: 0 grid cells; skipping.")
                continue
            # flatten selected grid
            vals = sub[:, rmask]  # (t, npoints)
            lat_pts = lats[idxs[:,0]]
            lon_pts = lons[idxs[:,1]]
            ds = xr.Dataset(
                coords={
                    "time": ("time", t_sel),
                    "point": ("point", np.arange(vals.shape[1])),
                    "lat": ("point", lat_pts.astype(np.float32)),
                    "lon": ("point", lon_pts.astype(np.float32)),
                },
                data_vars={
                    var: (("time","point"), vals.astype(np.float32)),
                },
                attrs={"description": "GCMagicc segment; grid points within region polygon"},
            )
            for k, v in run_meta.items():
                ds.attrs[str(k)] = str(v)
            seg_local = dict(seg); seg_local["period"] = ov
            ds.attrs["segment_spec"] = json.dumps(seg_local)

            group = f"runs/{run_meta['run_id']}/{_segment_key(seg)}"
            _write_segment_dataset_zarr(ds, root_dir=root_dir, store_name=store_name, group=group, mode="a")
        else:
            print(f"⚠️ Unknown operation={seg['operation']} (supported: 'area-weighted average', 'area-weighted annual average', 'area-weighted time average', 'grid-points'). Skipping.")

# =============================================================================
# Discovery / Selection (Option A CMIP6)
# =============================================================================

def _discover_cmip6_files(input_dir: str) -> List[str]:
    files = sorted(Path(input_dir).glob("*.nc"))
    return [str(p) for p in files if p.name.startswith("DAT_")]

def _find_magicc_parquet_file(
    directory: str,
    requested_scenarios: List[str],
    explicit_file: Optional[str] = None,
) -> str:
    """
    Find the appropriate MAGICC parquet file from a directory.
    
    If explicit_file is provided and exists, use it.
    Otherwise, scan directory for .parquet files and check which contain the requested scenarios.
    If multiple files match, prompt user to choose.
    If single file matches, auto-select it.
    
    Args:
        directory: Directory to search for .parquet files
        requested_scenarios: List of scenario names to look for
        explicit_file: Optional explicit file path (from SCM_PPI_PARQUET env var or CLI)
    
    Returns:
        Path to selected parquet file
    """
    # If explicit file provided, use it
    if explicit_file:
        if os.path.exists(explicit_file):
            return explicit_file
        else:
            raise FileNotFoundError(f"Explicit MAGICC parquet file not found: {explicit_file}")
    
    # Scan directory for .parquet files
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"MAGICC results directory not found: {directory}")
    
    parquet_files = sorted(dir_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in directory: {directory}")
    
    # If no scenarios requested, return the first file (or most recent)
    if not requested_scenarios:
        print(f"⚠️  No scenarios specified in EXPERIMENT_ID_WHITELIST; using most recent file: {parquet_files[-1].name}")
        return str(parquet_files[-1])
    
    # Check which files contain the requested scenarios
    matching_files: List[Tuple[Path, List[str]]] = []
    
    for parquet_file in parquet_files:
        try:
            # Quick check: read just the index to get scenarios
            # Try pyarrow first, fallback to fastparquet if needed
            try:
                df_sample = pd.read_parquet(parquet_file, engine='pyarrow')
            except Exception:
                df_sample = pd.read_parquet(parquet_file)
            
            # Check if index has 'scenario' level
            if isinstance(df_sample.index, pd.MultiIndex):
                if "scenario" in df_sample.index.names:
                    available_scenarios_raw = sorted(pd.unique(df_sample.index.get_level_values("scenario")))
                else:
                    print(f"⚠️  File {parquet_file.name} does not have 'scenario' in MultiIndex; skipping")
                    continue
            else:
                # Single-level index - check if it's named 'scenario'
                if df_sample.index.name == "scenario":
                    available_scenarios_raw = sorted(pd.unique(df_sample.index))
                else:
                    print(f"⚠️  File {parquet_file.name} does not have 'scenario' index; skipping")
                    continue
            
            # Normalize scenarios: strip 'clean_' prefix for matching
            # Create mapping: normalized -> original
            scenario_normalized_to_original = {}
            for orig_scen in available_scenarios_raw:
                normalized = orig_scen.replace("clean_", "", 1) if orig_scen.startswith("clean_") else orig_scen
                scenario_normalized_to_original[normalized] = orig_scen
            
            # Check which requested scenarios match (after normalization)
            found_scenarios = []
            for req_scen in requested_scenarios:
                if req_scen in scenario_normalized_to_original:
                    # Use the original scenario name from the file
                    found_scenarios.append(scenario_normalized_to_original[req_scen])
            
            if found_scenarios:
                matching_files.append((parquet_file, found_scenarios))
        except Exception as e:
            print(f"⚠️  Could not read {parquet_file.name}: {e}; skipping")
            continue
    
    if not matching_files:
        raise RuntimeError(
            f"None of the requested scenarios {requested_scenarios} were found in any .parquet file in {directory}. "
            f"Available files: {[f.name for f in parquet_files]}"
        )
    
    # If exactly one file matches all scenarios, use it
    perfect_matches = [
        (f, scens) for f, scens in matching_files 
        if set(scens) == set(requested_scenarios)
    ]
    if len(perfect_matches) == 1:
        selected_file, found_scens = perfect_matches[0]
        print(f"✅ Auto-selected parquet file: {selected_file.name}")
        print(f"   Contains all requested scenarios: {found_scens}")
        return str(selected_file)
    
    # If multiple files match, show options and let user choose
    if len(matching_files) == 1:
        selected_file, found_scens = matching_files[0]
        missing = [s for s in requested_scenarios if s not in found_scens]
        if missing:
            print(f"⚠️  Selected file {selected_file.name} contains {found_scens} but missing: {missing}")
        else:
            print(f"✅ Auto-selected parquet file: {selected_file.name}")
        return str(selected_file)
    
    # Multiple matches - prompt user
    print(f"\n📋 Found {len(matching_files)} parquet file(s) containing requested scenarios:")
    for i, (f, scens) in enumerate(matching_files, 1):
        missing = [s for s in requested_scenarios if s not in scens]
        status = "✅" if not missing else "⚠️"
        print(f"   {i}. {status} {f.name}")
        print(f"      Contains: {scens}")
        if missing:
            print(f"      Missing: {missing}")
    
    # Try to auto-select the most recent file that has all scenarios
    if perfect_matches:
        selected_file, found_scens = perfect_matches[-1]  # Most recent
        print(f"\n✅ Auto-selected most recent file with all scenarios: {selected_file.name}")
        return str(selected_file)
    
    # Otherwise, use the most recent file that has at least some scenarios
    selected_file, found_scens = matching_files[-1]
    missing = [s for s in requested_scenarios if s not in found_scens]
    print(f"\n⚠️  Auto-selected most recent file: {selected_file.name}")
    print(f"   Contains: {found_scens}")
    if missing:
        print(f"   Missing: {missing}")
    print(f"   To use a different file, set SCM_PPI_PARQUET environment variable or --scm-parquet CLI argument")
    return str(selected_file)

def _parse_cmip6_name(path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    base = Path(path).name
    parts = base.split("_")
    if len(parts) >= 4:
        return parts[1], parts[2], parts[3]
    return None, None, None

def _is_historical_period(window: Tuple[int, int], historical_cutoff: int = 2015) -> bool:
    """
    Determine if a time window falls in the historical period.
    
    A window is considered historical if its end year is before the cutoff (default 2015).
    """
    return window[1] < historical_cutoff

def _select_cmip6_files(files: Iterable[str], src_whitelist: Sequence[str], exp_whitelist: Sequence[str], max_members: int) -> List[str]:
    """
    Select CMIP6 files matching the whitelist criteria.
    
    If exp_whitelist contains SSP scenarios (starting with 'ssp'), also includes
    'historical' files to allow backward extension for historical periods.
    """
    # Check if any SSP scenarios are requested
    has_ssp = any(exp.lower().startswith('ssp') for exp in exp_whitelist)
    
    grouped: Dict[Tuple[str,str], List[str]] = {}
    for f in files:
        s, e, m = _parse_cmip6_name(f)
        if not s or not e:
            continue
        if src_whitelist and s not in src_whitelist:
            continue
        # Include experiment if it's in whitelist OR if it's 'historical' and we have SSP scenarios
        if exp_whitelist and e not in exp_whitelist:
            if not (has_ssp and e.lower() == 'historical'):
                continue
        grouped.setdefault((s,e), []).append(f)
    selection: List[str] = []
    for (s,e), arr in grouped.items():
        arr_sorted = sorted(arr)
        # keep up to max_members distinct r*i*p*f (file based)
        selection.extend(arr_sorted[:max(1, min(100, max_members))])
    return selection

# =============================================================================
# Multi-GPU worker functions and work item structures
# =============================================================================

@dataclass
class WorkItemA:
    """Work item for Option A (CMIP6)."""
    fpath: str
    draw_idx: int
    window: Tuple[int, int]
    source_id: str
    experiment_id: Optional[str]
    member_id: Optional[str]
    usebias_model: Optional[int]
    useeffect_model: Optional[int]
    seed_base: int
    bias_to_era5: bool
    amp_override: Optional[bool] = None


@dataclass
class WorkItemB:
    """Work item for Option B (MAGICC)."""
    scenario: str
    draw_idx: int
    window: Tuple[int, int]
    magicc_member: int
    effect_idx: int
    seed_base: int
    bias_to_era5: bool
    amp_override: Optional[bool] = None

@dataclass
class WorkItemD:
    """Work item for Option D (MAGICCxERA5 spliced predictors)."""
    scenario: str
    draw_idx: int
    window: Tuple[int, int]
    run_id: int
    predictor_path: str
    seed_base: int
    bias_to_era5: bool
    amp_override: Optional[bool] = None


def _work_item_key(item: Union[WorkItemA, WorkItemB, WorkItemD]) -> Tuple[object, ...]:
    """Stable key for identifying unique work items across retries."""
    if isinstance(item, WorkItemA):
        return (item.fpath, item.draw_idx, item.window, item.useeffect_model, item.bias_to_era5)
    if isinstance(item, WorkItemB):
        return (item.scenario, item.draw_idx, item.window, item.magicc_member, item.effect_idx, item.bias_to_era5)
    return (item.scenario, item.draw_idx, item.window, item.run_id, item.bias_to_era5)


# =============================================================================
# CPU worker helpers (used when --device=cpu or auto-CPU)
# =============================================================================


def _cpu_pool_context():
    """Prefer fork on Linux to avoid re-pickling large globals; fall back otherwise."""
    try:
        return get_context("fork")
    except Exception:
        return get_context()


def _run_cpu_pool(
    tasks,
    worker_fn,
    *,
    workers: int,
    desc: str,
    debug_first_success: bool = False,
):
    """
    Execute tasks in a CPU Pool, returning (successes, failures, retry_items).
    Each worker must return (success, message, retry_item|None).
    """
    if workers <= 1 or not tasks:
        return 0, 0, []

    successes = 0
    failures = 0
    retry_items = []
    first_announced = False
    ctx = _cpu_pool_context()

    with ctx.Pool(processes=workers) as pool:
        for success, msg, retry in _progress_iterable(
            pool.imap_unordered(worker_fn, tasks, chunksize=1),
            total=len(tasks),
            desc=desc,
        ):
            if success:
                successes += 1
                if debug_first_success and not first_announced:
                    print(f"🔍 First-success debug (CPU pool): {msg}")
                    first_announced = True
            else:
                failures += 1
            if retry is not None:
                retry_items.append(retry)

    return successes, failures, retry_items


def _worker_process_cpu_a(args) -> Tuple[bool, str, Optional[WorkItemA]]:
    """
    CPU worker for Option A items. Returns (success, message, retry_item|None).
    """
    (
        work_item,
        amp_default,
        force_gpu_flag,
        model_to_index,
        variables_2predict,
        lats,
        lons,
        output_specs,
        root_dir,
        store_name,
        model_version,
        model_number,
        dateofmetafile,
        debug_trace_emulation,
    ) = args
    device_str = "cpu"
    item_tag = f"{Path(work_item.fpath).name} draw {work_item.draw_idx} window {work_item.window}"
    amp_for_item = work_item.amp_override if work_item.amp_override is not None else amp_default

    try:
        random.seed(work_item.seed_base)
        np.random.seed(work_item.seed_base)
        torch.manual_seed(work_item.seed_base)

        predictors = extract_predictors_from_nc(
            work_item.fpath,
            timespan=[int(work_item.window[0]), int(work_item.window[1])],
        )
    except Exception as e:
        _log_emulation_exception(
            "Option A predictor extraction failure (CPU pool)",
            e,
            debug_trace=debug_trace_emulation,
            context={"file": work_item.fpath, "window": work_item.window},
        )
        retry_item = replace(work_item, amp_override=True) if not amp_for_item else None
        return (
            False,
            f"{item_tag}: predictor extraction failed: {e}",
            retry_item,
        )

    try:
        yhval = run_gcmagicc(
            predictors,
            dependence=True,
            usebias_model=work_item.usebias_model,
            useeffect_model=work_item.useeffect_model,
            device=device_str,
            force_gpu=force_gpu_flag,
            amp=amp_for_item,
        )
    except Exception as e:
        _log_emulation_exception(
            "Option A run_gcmagicc failure (CPU pool)",
            e,
            debug_trace=debug_trace_emulation,
            context={
                "file": work_item.fpath,
                "window": work_item.window,
                "draw_idx": work_item.draw_idx,
                "device": device_str,
                "amp": amp_for_item,
            },
        )
        retry_item = replace(work_item, amp_override=True) if not amp_for_item else None
        return (
            False,
            f"{item_tag}: emulation failed on CPU: {e}",
            retry_item,
        )

    run_id = (
        f"A_{work_item.source_id}_{work_item.experiment_id or 'unknown'}_{work_item.member_id or 'N'}"
        f"__b{0 if work_item.bias_to_era5 else 'N'}e{work_item.useeffect_model if work_item.useeffect_model is not None else 'N'}"
        f"__m{work_item.draw_idx:04d}__win{int(work_item.window[0])}-{int(work_item.window[1])}__{_today_stamp()}__{uuid.uuid4().hex[:8]}"
    )
    run_meta = {
        "run_id": run_id,
        "mode": "CMIP6",
        "source_id": work_item.source_id,
        "experiment_id": work_item.experiment_id or "N",
        "member_id": work_item.member_id or "N",
        "usebias_model": 0 if work_item.bias_to_era5 else "None",
        "useeffect_model": work_item.useeffect_model if work_item.useeffect_model is not None else "None",
        "model_version": model_version,
        "model_number": model_number,
        "model_id": f"{model_version}_{model_number}",
        "date_meta": dateofmetafile,
        "device": device_str,
        "amp": bool(amp_for_item),
    }

    try:
        apply_output_specs_and_write(
            yhval=yhval,
            predictors=predictors,
            lats=lats,
            lons=lons,
            output_specs=output_specs,
            run_meta=run_meta,
            variables_2predict=variables_2predict,
            root_dir=root_dir,
            store_name=store_name,
        )
    except Exception as e:
        _log_emulation_exception(
            "Option A segment write failure (CPU pool)",
            e,
            debug_trace=debug_trace_emulation,
            context={
                "run_id": run_meta.get("run_id"),
                "file": work_item.fpath,
                "window": work_item.window,
            },
        )
        retry_item = replace(work_item, amp_override=True) if not amp_for_item else None
        return (
            False,
            f"{item_tag}: write failed on CPU: {e}",
            retry_item,
        )
    finally:
        del yhval, predictors
        gc.collect()

    return True, f"{item_tag} run_id={run_id}", None


def _worker_process_cpu_b(args) -> Tuple[bool, str, Optional[WorkItemB]]:
    """
    CPU worker for Option B/C items. Returns (success, message, retry_item|None).
    """
    (
        work_item,
        amp_default,
        force_gpu_flag,
        df_wide,
        model_to_index,
        variables_2predict,
        lats,
        lons,
        output_specs,
        root_dir,
        store_name,
        model_version,
        model_number,
        dateofmetafile,
        mode_name,
        effect_model_scheme,
        effect_model_scheme_c,
        debug_trace_emulation,
    ) = args
    device_str = "cpu"
    item_tag = (
        f"{mode_name} {work_item.scenario} run{work_item.magicc_member:03d} "
        f"eff{work_item.effect_idx} draw{work_item.draw_idx} window {work_item.window}"
    )
    amp_for_item = work_item.amp_override if work_item.amp_override is not None else amp_default

    try:
        random.seed(work_item.seed_base)
        np.random.seed(work_item.seed_base)
        torch.manual_seed(work_item.seed_base)

        predictors = build_predictors_from_magicc(
            df_wide,
            work_item.scenario,
            work_item.magicc_member,
            model_to_index,
            year_start=int(work_item.window[0]),
            year_end=int(work_item.window[1]),
        )
    except Exception as e:
        _log_emulation_exception(
            f"{mode_name} predictor build failure (CPU pool)",
            e,
            debug_trace=debug_trace_emulation,
            context={
                "scenario": work_item.scenario,
                "window": work_item.window,
                "magicc_member": work_item.magicc_member,
            },
        )
        retry_item = replace(work_item, amp_override=True) if not amp_for_item else None
        return (
            False,
            f"{item_tag}: predictor build failed: {e}",
            retry_item,
        )

    if "model_index" in predictors.predictor_names:
        mi_col = predictors.predictor_names.index("model_index")
        predictors.X[:, mi_col] = float(work_item.effect_idx)

    usebias_model = 0 if work_item.bias_to_era5 else None
    useeffect_model = work_item.effect_idx

    try:
        yhval = run_gcmagicc(
            predictors,
            dependence=True,
            usebias_model=usebias_model,
            useeffect_model=useeffect_model,
            device=device_str,
            force_gpu=force_gpu_flag,
            amp=amp_for_item,
        )
    except Exception as e:
        _log_emulation_exception(
            f"{mode_name} run_gcmagicc failure (CPU pool)",
            e,
            debug_trace=debug_trace_emulation,
            context={
                "scenario": work_item.scenario,
                "window": work_item.window,
                "magicc_member": work_item.magicc_member,
                "effect_idx": useeffect_model,
                "device": device_str,
                "amp": amp_for_item,
            },
        )
        retry_item = replace(work_item, amp_override=True) if not amp_for_item else None
        return (
            False,
            f"{item_tag}: emulation failed on CPU: {e}",
            retry_item,
        )

    run_id = (
        f"{'B' if mode_name == 'MAGICC' else 'C'}_{work_item.scenario}"
        f"__MAGICCrun{work_item.magicc_member:03d}"
        f"__b{0 if work_item.bias_to_era5 else 'N'}e{useeffect_model}"
        f"__m{work_item.draw_idx:04d}__win{int(work_item.window[0])}-{int(work_item.window[1])}"
        f"__{_today_stamp()}__{uuid.uuid4().hex[:8]}"
    )
    run_meta = {
        "run_id": run_id,
        "mode": mode_name,
        "scenario": work_item.scenario,
        "magicc_run_id": work_item.magicc_member,
        "usebias_model": 0 if work_item.bias_to_era5 else "None",
        "useeffect_model": useeffect_model,
        "model_version": model_version,
        "model_number": model_number,
        "model_id": f"{model_version}_{model_number}",
        "date_meta": dateofmetafile,
        "device": device_str,
        "effect_model_scheme": json.dumps(effect_model_scheme) if isinstance(effect_model_scheme, (list, dict)) else str(effect_model_scheme),
        "amp": bool(amp_for_item),
    }
    if effect_model_scheme_c is not None:
        run_meta["effect_model_scheme_c"] = json.dumps(effect_model_scheme_c) if isinstance(effect_model_scheme_c, (list, dict)) else str(effect_model_scheme_c)

    try:
        apply_output_specs_and_write(
            yhval=yhval,
            predictors=predictors,
            lats=lats,
            lons=lons,
            output_specs=output_specs,
            run_meta=run_meta,
            variables_2predict=variables_2predict,
            root_dir=root_dir,
            store_name=store_name,
        )
    except Exception as e:
        _log_emulation_exception(
            f"{mode_name} segment write failure (CPU pool)",
            e,
            debug_trace=debug_trace_emulation,
            context={
                "run_id": run_meta.get("run_id"),
                "scenario": work_item.scenario,
                "window": work_item.window,
            },
        )
        retry_item = replace(work_item, amp_override=True) if not amp_for_item else None
        return (
            False,
            f"{item_tag}: write failed on CPU: {e}",
            retry_item,
        )
    finally:
        del yhval, predictors
        gc.collect()

    return True, f"{item_tag} run_id={run_id}", None


def _worker_process_cpu_d(args) -> Tuple[bool, str, Optional[WorkItemD]]:
    """
    CPU worker for Option D items. Returns (success, message, retry_item|None).
    """
    (
        work_item,
        amp_default,
        force_gpu_flag,
        model_to_index,
        variables_2predict,
        lats,
        lons,
        output_specs,
        root_dir,
        store_name,
        model_version,
        model_number,
        dateofmetafile,
        era5_effect_idx,
        debug_trace_emulation,
    ) = args
    device_str = "cpu"
    item_tag = (
        f"MAGICCxERA5 {work_item.scenario} run{work_item.run_id:03d} "
        f"draw{work_item.draw_idx} window {work_item.window}"
    )
    amp_for_item = work_item.amp_override if work_item.amp_override is not None else amp_default

    try:
        random.seed(work_item.seed_base)
        np.random.seed(work_item.seed_base)
        torch.manual_seed(work_item.seed_base)

        predictors = build_predictors_from_spliced_file(
            work_item.predictor_path,
            model_to_index,
            model_index_name="ERA5",
            year_start=int(work_item.window[0]),
            year_end=int(work_item.window[1]),
        )
    except Exception as e:
        _log_emulation_exception(
            "MAGICCxERA5 predictor build failure (CPU pool)",
            e,
            debug_trace=debug_trace_emulation,
            context={
                "scenario": work_item.scenario,
                "window": work_item.window,
                "run_id": work_item.run_id,
            },
        )
        retry_item = replace(work_item, amp_override=True) if not amp_for_item else None
        return (
            False,
            f"{item_tag}: predictor build failed: {e}",
            retry_item,
        )

    if "model_index" in predictors.predictor_names:
        mi_col = predictors.predictor_names.index("model_index")
        predictors.X[:, mi_col] = float(era5_effect_idx)

    usebias_model = 0 if work_item.bias_to_era5 else None
    useeffect_model = era5_effect_idx

    try:
        yhval = run_gcmagicc(
            predictors,
            dependence=True,
            usebias_model=usebias_model,
            useeffect_model=useeffect_model,
            device=device_str,
            force_gpu=force_gpu_flag,
            amp=amp_for_item,
        )
    except Exception as e:
        _log_emulation_exception(
            "MAGICCxERA5 run_gcmagicc failure (CPU pool)",
            e,
            debug_trace=debug_trace_emulation,
            context={
                "scenario": work_item.scenario,
                "window": work_item.window,
                "run_id": work_item.run_id,
                "device": device_str,
                "amp": amp_for_item,
            },
        )
        retry_item = replace(work_item, amp_override=True) if not amp_for_item else None
        return (
            False,
            f"{item_tag}: emulation failed on CPU: {e}",
            retry_item,
        )

    run_id = (
        f"D_{work_item.scenario}__ERA5splicedrun{work_item.run_id:03d}"
        f"__b{0 if work_item.bias_to_era5 else 'N'}e{era5_effect_idx}"
        f"__m{work_item.draw_idx:04d}__win{int(work_item.window[0])}-{int(work_item.window[1])}"
        f"__{_today_stamp()}__{uuid.uuid4().hex[:8]}"
    )
    run_meta = {
        "run_id": run_id,
        "mode": "MAGICCxERA5",
        "scenario": work_item.scenario,
        "magicc_run_id": work_item.run_id,
        "usebias_model": 0 if work_item.bias_to_era5 else "None",
        "useeffect_model": era5_effect_idx,
        "model_version": model_version,
        "model_number": model_number,
        "model_id": f"{model_version}_{model_number}",
        "date_meta": dateofmetafile,
        "device": device_str,
        "amp": bool(amp_for_item),
        "predictor_path": work_item.predictor_path,
    }

    try:
        apply_output_specs_and_write(
            yhval=yhval,
            predictors=predictors,
            lats=lats,
            lons=lons,
            output_specs=output_specs,
            run_meta=run_meta,
            variables_2predict=variables_2predict,
            root_dir=root_dir,
            store_name=store_name,
        )
    except Exception as e:
        _log_emulation_exception(
            "MAGICCxERA5 segment write failure (CPU pool)",
            e,
            debug_trace=debug_trace_emulation,
            context={
                "run_id": run_meta.get("run_id"),
                "scenario": work_item.scenario,
                "window": work_item.window,
            },
        )
        retry_item = replace(work_item, amp_override=True) if not amp_for_item else None
        return (
            False,
            f"{item_tag}: write failed on CPU: {e}",
            retry_item,
        )
    finally:
        del yhval, predictors
        gc.collect()

    return True, f"{item_tag} run_id={run_id}", None


def _worker_process_gpu_a(
    work_item: WorkItemA,
    gpu_idx: int,
    *,
    model_to_index: Dict[str, int],
    variables_2predict: List[str],
    lats: np.ndarray,
    lons: np.ndarray,
    output_specs: List[dict],
    root_dir: str,
    store_name: str,
    force_gpu: bool,
    amp_flag: bool,
    model_version: str,
    model_number: str,
    dateofmetafile: str,
    debug_trace_emulation: bool,
) -> Tuple[bool, str]:
    """
    Worker process for Option A (CMIP6) work items.
    Returns (success, message) tuple.
    """
    item_tag = f"{Path(work_item.fpath).name} draw {work_item.draw_idx} window {work_item.window}"
    try:
        # Bind directly to the physical GPU index we were given.
        # We no longer change CUDA_VISIBLE_DEVICES here; with multiprocessing
        # 'spawn' that was too late (CUDA already initialised) and caused all
        # workers to allocate on GPU 0.
        physical_gpu_idx = gpu_idx
        torch_device_idx = physical_gpu_idx
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(torch_device_idx)
            except Exception as e:
                print(
                    f"[worker A] ⚠️  Could not set CUDA device to {torch_device_idx}: {e}",
                    flush=True,
                )
        device_str = f"cuda:{torch_device_idx}"

        # Check GPU memory availability before starting work using nvidia-smi (more reliable)
        min_free_gb = max(0.0, MIN_FREE_MEMORY_GB)
        try:
            # Use nvidia-smi to check memory (works even if PyTorch hasn't initialized CUDA)
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits", "--id", str(physical_gpu_idx)],
                text=True,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            for line in out.splitlines():
                s = line.strip()
                if not s:
                    continue
                parts = [p.strip() for p in s.split(",")]
                if len(parts) >= 3 and parts[0].isdigit() and int(parts[0]) == physical_gpu_idx:
                    try:
                        used_mem_mb = int(parts[1])
                        total_mem_mb = int(parts[2])
                        free_mem_mb = total_mem_mb - used_mem_mb
                        free_mem_gb = free_mem_mb / 1024
                        total_mem_gb = total_mem_mb / 1024
                        used_mem_gb = used_mem_mb / 1024
                        mem_percent = (used_mem_mb / total_mem_mb * 100) if total_mem_mb > 0 else 0

                        if free_mem_gb < min_free_gb:
                            return (False, f"{item_tag}: GPU {physical_gpu_idx} has insufficient free memory: {free_mem_gb:.2f}GiB free (need {min_free_gb}GiB). "
                                          f"Total: {total_mem_gb:.2f}GiB, Used: {used_mem_gb:.2f}GiB ({mem_percent:.1f}%)")

                        # Warn if memory usage is high
                        if mem_percent > 50:
                            print(f"⚠️  GPU {physical_gpu_idx} memory usage is high: {mem_percent:.1f}% ({used_mem_gb:.2f}GiB/{total_mem_gb:.2f}GiB)")
                        break
                    except (ValueError, IndexError) as e:
                        print(f"⚠️  Could not parse GPU {physical_gpu_idx} memory info: {e}")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            # Fallback to PyTorch memory check if nvidia-smi fails
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize(torch_device_idx)
                    mem_info = torch.cuda.mem_get_info(torch_device_idx)
                    free_mem_gb = mem_info[0] / (1024**3)
                    total_mem_gb = mem_info[1] / (1024**3)
                    used_mem_gb = total_mem_gb - free_mem_gb
                    mem_percent = (used_mem_gb / total_mem_gb * 100) if total_mem_gb > 0 else 0

                    if free_mem_gb < min_free_gb:
                        return (False, f"{item_tag}: GPU {physical_gpu_idx} has insufficient free memory: {free_mem_gb:.2f}GiB free (need {min_free_gb}GiB). "
                                      f"Total: {total_mem_gb:.2f}GiB, Used: {used_mem_gb:.2f}GiB ({mem_percent:.1f}%)")
                except Exception as e2:
                    print(f"⚠️  Could not check GPU {physical_gpu_idx} memory availability (nvidia-smi and PyTorch both failed): {e}, {e2}")
            else:
                print(f"⚠️  Could not check GPU {physical_gpu_idx} memory availability: {e}")
        
        # Set RNG seed for this work item
        random.seed(work_item.seed_base)
        np.random.seed(work_item.seed_base)
        torch.manual_seed(work_item.seed_base)
        
        # Build predictors for this window
        try:
            predictors = extract_predictors_from_nc(
                work_item.fpath,
                timespan=[int(work_item.window[0]), int(work_item.window[1])]
            )
        except Exception as e:
            _log_emulation_exception(
                "Option A predictor extraction failure",
                e,
                debug_trace=debug_trace_emulation,
                context={"file": work_item.fpath, "window": work_item.window, "gpu_idx": gpu_idx},
            )
            return (False, f"{item_tag}: predictor extraction failed: {e}")
        
        amp_for_item = work_item.amp_override if work_item.amp_override is not None else amp_flag

        # Run emulation
        try:
            yhval = run_gcmagicc(
                predictors,
                dependence=True,
                usebias_model=work_item.usebias_model,
                useeffect_model=work_item.useeffect_model,
                device=device_str,
                force_gpu=force_gpu,
                amp=amp_for_item,
            )
        except Exception as e:
            # Capture GPU state at time of failure
            try:
                print(f"📸 Dumping GPU state for GPU {gpu_idx} failure...")
                subprocess.run(["nvidia-smi"], check=False)
            except Exception:
                pass

            _log_emulation_exception(
                "Option A run_gcmagicc failure",
                e,
                debug_trace=debug_trace_emulation,
                context={
                    "file": work_item.fpath,
                    "window": work_item.window,
                    "draw_idx": work_item.draw_idx,
                    "gpu_idx": physical_gpu_idx,
                    "device": device_str,
                    "amp": amp_for_item,
                    "usebias_model": work_item.usebias_model,
                    "useeffect_model": work_item.useeffect_model,
                },
            )
            return (False, f"{item_tag}: emulation failed on GPU {physical_gpu_idx}: {e}")
        
        # Generate run metadata
        run_id = (
            f"A_{work_item.source_id}_{work_item.experiment_id or 'unknown'}_{work_item.member_id or 'N'}"
            f"__b{0 if work_item.bias_to_era5 else 'N'}e{work_item.useeffect_model if work_item.useeffect_model is not None else 'N'}"
            f"__m{work_item.draw_idx:04d}__win{int(work_item.window[0])}-{int(work_item.window[1])}__{_today_stamp()}__{uuid.uuid4().hex[:8]}"
        )
        run_meta = {
            "run_id": run_id,
            "mode": "CMIP6",
            "source_id": work_item.source_id,
            "experiment_id": work_item.experiment_id or "N",
            "member_id": work_item.member_id or "N",
            "usebias_model": 0 if work_item.bias_to_era5 else "None",
            "useeffect_model": work_item.useeffect_model if work_item.useeffect_model is not None else "None",
            "model_version": model_version,
            "model_number": model_number,
            "model_id": f"{model_version}_{model_number}",
            "date_meta": dateofmetafile,
            "device": device_str,
            "device_physical": f"cuda:{physical_gpu_idx}",
            "amp": bool(amp_for_item),
        }
        
        # Write output
        try:
            apply_output_specs_and_write(
                yhval=yhval,
                predictors=predictors,
                lats=lats, lons=lons,
                output_specs=output_specs,
                run_meta=run_meta,
                variables_2predict=variables_2predict,
                root_dir=root_dir,
                store_name=store_name,
            )
        except Exception as e:
            _log_emulation_exception(
                "Option A segment write failure",
                e,
                debug_trace=debug_trace_emulation,
                context={
                    "run_id": run_meta.get("run_id"),
                    "file": work_item.fpath,
                    "window": work_item.window,
                },
            )
            return (False, f"{item_tag}: write failed: {e}")
        
        # Aggressive cleanup to prevent memory fragmentation
        del yhval, predictors
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(torch_device_idx)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(torch_device_idx)  # Double sync to ensure cleanup completes
                gc.collect()
            except Exception:
                pass

        return (True, f"GPU {gpu_idx}: {item_tag}")
    
    except Exception as e:
        return (False, f"{item_tag}: GPU {gpu_idx} worker error: {e}")


def _worker_process_gpu_b(
    work_item: WorkItemB,
    gpu_idx: int,
    *,
    df_wide: pd.DataFrame,
    model_to_index: Dict[str, int],
    variables_2predict: List[str],
    lats: np.ndarray,
    lons: np.ndarray,
    output_specs: List[dict],
    root_dir: str,
    store_name: str,
    force_gpu: bool,
    amp_flag: bool,
    model_version: str,
    model_number: str,
    dateofmetafile: str,
    mode_name: str,
    effect_model_scheme: Union[str, int, List[int], List[str]],
    effect_model_scheme_c: Optional[Union[str, List[int], List[str]]] = None,
    debug_trace_emulation: bool = False,
) -> Tuple[bool, str]:
    """
    Worker process for Option B (MAGICC) work items.
    Returns (success, message) tuple.
    """
    item_tag = (f"{work_item.scenario} run{work_item.magicc_member:03d} "
                f"eff{work_item.effect_idx} draw{work_item.draw_idx} window {work_item.window}")
    try:
        # Same strategy as for worker A: bind directly to the physical GPU index.
        physical_gpu_idx = gpu_idx
        torch_device_idx = physical_gpu_idx
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(torch_device_idx)
            except Exception as e:
                print(
                    f"[worker B] ⚠️  Could not set CUDA device to {torch_device_idx}: {e}",
                    flush=True,
                )
        device_str = f"cuda:{torch_device_idx}"

        # Check GPU memory availability before starting work using nvidia-smi (more reliable)
        min_free_gb = max(0.0, MIN_FREE_MEMORY_GB)
        try:
            # Use nvidia-smi to check memory (works even if PyTorch hasn't initialized CUDA)
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits", "--id", str(physical_gpu_idx)],
                text=True,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            for line in out.splitlines():
                s = line.strip()
                if not s:
                    continue
                parts = [p.strip() for p in s.split(",")]
                if len(parts) >= 3 and parts[0].isdigit() and int(parts[0]) == physical_gpu_idx:
                    try:
                        used_mem_mb = int(parts[1])
                        total_mem_mb = int(parts[2])
                        free_mem_mb = total_mem_mb - used_mem_mb
                        free_mem_gb = free_mem_mb / 1024
                        total_mem_gb = total_mem_mb / 1024
                        used_mem_gb = used_mem_mb / 1024
                        mem_percent = (used_mem_mb / total_mem_mb * 100) if total_mem_mb > 0 else 0

                        if free_mem_gb < min_free_gb:
                            return (False, f"{item_tag}: GPU {physical_gpu_idx} has insufficient free memory: {free_mem_gb:.2f}GiB free (need {min_free_gb}GiB). "
                                          f"Total: {total_mem_gb:.2f}GiB, Used: {used_mem_gb:.2f}GiB ({mem_percent:.1f}%)")

                        # Warn if memory usage is high
                        if mem_percent > 50:
                            print(f"⚠️  GPU {physical_gpu_idx} memory usage is high: {mem_percent:.1f}% ({used_mem_gb:.2f}GiB/{total_mem_gb:.2f}GiB)")
                        break
                    except (ValueError, IndexError) as e:
                        print(f"⚠️  Could not parse GPU {physical_gpu_idx} memory info: {e}")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            # Fallback to PyTorch memory check if nvidia-smi fails
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize(torch_device_idx)
                    mem_info = torch.cuda.mem_get_info(torch_device_idx)
                    free_mem_gb = mem_info[0] / (1024**3)
                    total_mem_gb = mem_info[1] / (1024**3)
                    used_mem_gb = total_mem_gb - free_mem_gb
                    mem_percent = (used_mem_gb / total_mem_gb * 100) if total_mem_gb > 0 else 0

                    if free_mem_gb < min_free_gb:
                        return (False, f"{item_tag}: GPU {physical_gpu_idx} has insufficient free memory: {free_mem_gb:.2f}GiB free (need {min_free_gb}GiB). "
                                      f"Total: {total_mem_gb:.2f}GiB, Used: {used_mem_gb:.2f}GiB ({mem_percent:.1f}%)")
                except Exception as e2:
                    print(f"⚠️  Could not check GPU {physical_gpu_idx} memory availability (nvidia-smi and PyTorch both failed): {e}, {e2}")
            else:
                print(f"⚠️  Could not check GPU {physical_gpu_idx} memory availability: {e}")
        
        # Set RNG seed for this work item
        random.seed(work_item.seed_base)
        np.random.seed(work_item.seed_base)
        torch.manual_seed(work_item.seed_base)
        
        # Build predictors for this scenario+member+window
        try:
            predictors = build_predictors_from_magicc(
                df_wide,
                work_item.scenario,
                work_item.magicc_member,
                model_to_index,
                year_start=int(work_item.window[0]),
                year_end=int(work_item.window[1])
            )
        except Exception as e:
            _log_emulation_exception(
                f"{mode_name} predictor build failure",
                e,
                debug_trace=debug_trace_emulation,
                context={
                    "scenario": work_item.scenario,
                    "window": work_item.window,
                    "magicc_member": work_item.magicc_member,
                    "effect_idx": work_item.effect_idx,
                },
            )
            return (False, f"{item_tag}: predictor build failed: {e}")
        
        # Fill 'model_index' to chosen effect_idx
        if "model_index" in predictors.predictor_names:
            mi_col = predictors.predictor_names.index("model_index")
            predictors.X[:, mi_col] = float(work_item.effect_idx)
        
        usebias_model = 0 if work_item.bias_to_era5 else None
        useeffect_model = work_item.effect_idx
        amp_for_item = work_item.amp_override if work_item.amp_override is not None else amp_flag
        
        # Run emulation
        try:
            yhval = run_gcmagicc(
                predictors,
                dependence=True,
                usebias_model=usebias_model,
                useeffect_model=useeffect_model,
                device=device_str,
                force_gpu=force_gpu,
                amp=amp_for_item,
            )
        except Exception as e:
            _log_emulation_exception(
                f"{mode_name} run_gcmagicc failure",
                e,
                debug_trace=debug_trace_emulation,
                context={
                    "scenario": work_item.scenario,
                    "window": work_item.window,
                    "magicc_member": work_item.magicc_member,
                    "effect_idx": work_item.effect_idx,
                    "gpu_idx": physical_gpu_idx,
                    "device": device_str,
                    "amp": amp_for_item,
                },
            )
            return (False, f"{item_tag}: emulation failed on GPU {physical_gpu_idx}: {e}")
        
        # Generate run metadata
        mode_upper = str(mode_name).upper()
        mode_prefix = "B" if mode_upper == "MAGICC" else "C"
        run_id = (
            f"{mode_prefix}_{work_item.scenario}__MAGICCrun{work_item.magicc_member:03d}"
            f"__b{0 if work_item.bias_to_era5 else 'N'}e{useeffect_model}"
            f"__m{work_item.draw_idx:04d}__win{int(work_item.window[0])}-{int(work_item.window[1])}__{_today_stamp()}__{uuid.uuid4().hex[:8]}"
        )
        run_meta = {
            "run_id": run_id,
            "mode": mode_name,
            "scenario": work_item.scenario,
            "magicc_run_id": work_item.magicc_member,
            "usebias_model": 0 if work_item.bias_to_era5 else "None",
            "useeffect_model": useeffect_model,
            "model_version": model_version,
            "model_number": model_number,
            "model_id": f"{model_version}_{model_number}",
            "date_meta": dateofmetafile,
            "device": device_str,
            "device_physical": f"cuda:{physical_gpu_idx}",
            "effect_model_scheme": json.dumps(effect_model_scheme) if isinstance(effect_model_scheme, (list, dict)) else str(effect_model_scheme),
            "amp": bool(amp_for_item),
        }
        if effect_model_scheme_c is not None:
            run_meta["effect_model_scheme_c"] = json.dumps(effect_model_scheme_c) if isinstance(effect_model_scheme_c, (list, dict)) else str(effect_model_scheme_c)
        
        # Write output
        try:
            apply_output_specs_and_write(
                yhval=yhval,
                predictors=predictors,
                lats=lats, lons=lons,
                output_specs=output_specs,
                run_meta=run_meta,
                variables_2predict=variables_2predict,
                root_dir=root_dir,
                store_name=store_name,
            )
        except Exception as e:
            _log_emulation_exception(
                f"{mode_name} segment write failure",
                e,
                debug_trace=debug_trace_emulation,
                context={
                    "run_id": run_meta.get("run_id"),
                    "scenario": work_item.scenario,
                    "window": work_item.window,
                },
            )
            return (False, f"{item_tag}: write failed: {e}")
        
        # Aggressive cleanup to prevent memory fragmentation
        del yhval, predictors
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(torch_device_idx)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(torch_device_idx)  # Double sync to ensure cleanup completes
                gc.collect()
            except Exception:
                pass

        return (True, f"GPU {gpu_idx}: {item_tag}")

    except Exception as e:
        return (False, f"{item_tag}: GPU {gpu_idx} worker error: {e}")


def _worker_process_gpu_d(
    work_item: WorkItemD,
    gpu_idx: int,
    *,
    model_to_index: Dict[str, int],
    variables_2predict: List[str],
    lats: np.ndarray,
    lons: np.ndarray,
    output_specs: List[dict],
    root_dir: str,
    store_name: str,
    force_gpu: bool,
    amp_flag: bool,
    model_version: str,
    model_number: str,
    dateofmetafile: str,
    debug_trace_emulation: bool = False,
) -> Tuple[bool, str]:
    """
    Worker process for Option D (MAGICCxERA5) work items.
    """
    item_tag = (f"{work_item.scenario} run{work_item.run_id:03d} "
                f"draw{work_item.draw_idx} window {work_item.window}")
    try:
        physical_gpu_idx = gpu_idx
        torch_device_idx = physical_gpu_idx
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(torch_device_idx)
            except Exception as e:
                print(
                    f"[worker D] ⚠️  Could not set CUDA device to {torch_device_idx}: {e}",
                    flush=True,
                )
        device_str = f"cuda:{torch_device_idx}"

        # Check GPU memory availability
        min_free_gb = max(0.0, MIN_FREE_MEMORY_GB)
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits", "--id", str(physical_gpu_idx)],
                text=True,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            for line in out.splitlines():
                s = line.strip()
                if not s:
                    continue
                parts = [p.strip() for p in s.split(",")]
                if len(parts) >= 3 and parts[0].isdigit() and int(parts[0]) == physical_gpu_idx:
                    used_mem_mb = int(parts[1])
                    total_mem_mb = int(parts[2])
                    free_mem_mb = total_mem_mb - used_mem_mb
                    free_mem_gb = free_mem_mb / 1024
                    total_mem_gb = total_mem_mb / 1024
                    used_mem_gb = used_mem_mb / 1024
                    mem_percent = (used_mem_mb / total_mem_mb * 100) if total_mem_mb > 0 else 0

                    if free_mem_gb < min_free_gb:
                        return (False, f"{item_tag}: GPU {physical_gpu_idx} has insufficient free memory: {free_mem_gb:.2f}GiB free (need {min_free_gb}GiB). "
                                      f"Total: {total_mem_gb:.2f}GiB, Used: {used_mem_gb:.2f}GiB ({mem_percent:.1f}%)")
                    if mem_percent > 50:
                        print(f"⚠️  GPU {physical_gpu_idx} memory usage is high: {mem_percent:.1f}% ({used_mem_gb:.2f}GiB/{total_mem_gb:.2f}GiB)")
                    break
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize(torch_device_idx)
                    mem_info = torch.cuda.mem_get_info(torch_device_idx)
                    free_mem_gb = mem_info[0] / (1024**3)
                    if free_mem_gb < min_free_gb:
                        return (False, f"{item_tag}: GPU {physical_gpu_idx} has insufficient free memory: {free_mem_gb:.2f}GiB free (need {min_free_gb}GiB).")
                except Exception as e2:
                    print(f"⚠️  Could not check GPU {physical_gpu_idx} memory availability (nvidia-smi and PyTorch both failed): {e}, {e2}")

        random.seed(work_item.seed_base)
        np.random.seed(work_item.seed_base)
        torch.manual_seed(work_item.seed_base)

        try:
            predictors = build_predictors_from_spliced_file(
                work_item.predictor_path,
                model_to_index,
                model_index_name="ERA5",
                year_start=int(work_item.window[0]),
                year_end=int(work_item.window[1]),
            )
        except Exception as e:
            _log_emulation_exception(
                "MAGICCxERA5 predictor build failure (GPU worker)",
                e,
                debug_trace=debug_trace_emulation,
                context={
                    "scenario": work_item.scenario,
                    "window": work_item.window,
                    "run_id": work_item.run_id,
                    "predictor_path": work_item.predictor_path,
                },
            )
            return (False, f"{item_tag}: predictor build failed: {e}")

        if "model_index" in predictors.predictor_names:
            mi_col = predictors.predictor_names.index("model_index")
            predictors.X[:, mi_col] = float(0)

        usebias_model = 0 if work_item.bias_to_era5 else None
        useeffect_model = 0
        amp_for_item = work_item.amp_override if work_item.amp_override is not None else amp_flag

        try:
            yhval = run_gcmagicc(
                predictors,
                dependence=True,
                usebias_model=usebias_model,
                useeffect_model=useeffect_model,
                device=device_str,
                force_gpu=force_gpu,
                amp=amp_for_item,
            )
        except Exception as e:
            _log_emulation_exception(
                "MAGICCxERA5 run_gcmagicc failure (GPU worker)",
                e,
                debug_trace=debug_trace_emulation,
                context={
                    "scenario": work_item.scenario,
                    "window": work_item.window,
                    "run_id": work_item.run_id,
                    "device": device_str,
                    "amp": amp_for_item,
                },
            )
            return (False, f"{item_tag}: emulation failed on GPU {physical_gpu_idx}: {e}")

        run_id = (
            f"D_{work_item.scenario}__ERA5splicedrun{work_item.run_id:03d}"
            f"__b{0 if work_item.bias_to_era5 else 'N'}e0"
            f"__m{work_item.draw_idx:04d}__win{int(work_item.window[0])}-{int(work_item.window[1])}__{_today_stamp()}__{uuid.uuid4().hex[:8]}"
        )
        run_meta = {
            "run_id": run_id,
            "mode": "MAGICCxERA5",
            "scenario": work_item.scenario,
            "magicc_run_id": work_item.run_id,
            "usebias_model": 0 if work_item.bias_to_era5 else "None",
            "useeffect_model": useeffect_model,
            "model_version": model_version,
            "model_number": model_number,
            "model_id": f"{model_version}_{model_number}",
            "date_meta": dateofmetafile,
            "device": device_str,
            "amp": bool(amp_for_item),
            "predictor_path": work_item.predictor_path,
        }

        try:
            apply_output_specs_and_write(
                yhval=yhval,
                predictors=predictors,
                lats=lats, lons=lons,
                output_specs=output_specs,
                run_meta=run_meta,
                variables_2predict=variables_2predict,
                root_dir=root_dir,
                store_name=store_name,
            )
        except Exception as e:
            _log_emulation_exception(
                "Option D segment write failure",
                e,
                debug_trace=debug_trace_emulation,
                context={
                    "run_id": run_meta.get("run_id"),
                    "file": work_item.predictor_path,
                    "window": work_item.window,
                },
            )
            return (False, f"{item_tag}: write failed: {e}")

        del yhval, predictors
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(torch_device_idx)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(torch_device_idx)
                gc.collect()
            except Exception:
                pass

        return (True, f"GPU {gpu_idx}: {item_tag}")

    except Exception as e:
        return (False, f"{item_tag}: GPU {gpu_idx} worker error: {e}")
def _log_emulation_exception(
    prefix: str,
    exc: BaseException,
    *,
    debug_trace: bool,
    context: Optional[Dict[str, Union[str, int, float, Tuple[int, int], Tuple[int, int, int, int]]]] = None,
) -> None:
    """Emit detailed traceback/context for emulation failures when debugging is enabled."""
    if not debug_trace:
        return
    print(f"🟥 {prefix}: {exc}")
    tb_text = traceback.format_exc()
    if tb_text:
        for line in tb_text.rstrip().splitlines():
            print(f"   {line}")
    if context:
        print("   Context:")
        for key, value in context.items():
            print(f"      {key}: {value}")
    sys.stdout.flush()


def _append_run_log(log_path: str, status: str, meta: Dict[str, object], error: Optional[str] = None) -> None:
    """
    Append a single run result to a JSONL log. Best-effort (errors ignored).
    status: "success" | "fail"
    """
    entry = {
        "status": status,
        "meta": meta,
        "error": error,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        # Do not let logging failures break the run
        pass
def _append_log(path: Optional[str], line: str) -> None:
    """Append a single line to a log file; ignore errors silently."""
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write(line.rstrip() + "\n")
    except Exception:
        pass


def _filter_gpus_by_memory(gpu_indices: List[int], min_free_gb: float = None) -> List[int]:
    """
    Filter GPU indices to only include those with sufficient free memory.
    Uses nvidia-smi to check memory availability right before execution.
    
    Default requirement: MIN_FREE_MEMORY_GB (env override GCMAGICC_MIN_FREE_MEMORY_GB).
    """
    min_free_gb = MIN_FREE_MEMORY_GB if min_free_gb is None else min_free_gb
    print(f"🔎 Checking memory availability for GPUs: {gpu_indices} (need >{min_free_gb}GB free)")
    available_gpus = []
    for gpu_idx in gpu_indices:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits", "--id", str(gpu_idx)],
                text=True,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            for line in out.splitlines():
                s = line.strip()
                if not s:
                    continue
                parts = [p.strip() for p in s.split(",")]
                if len(parts) >= 3 and parts[0].isdigit() and int(parts[0]) == gpu_idx:
                    try:
                        used_mem_mb = int(parts[1])
                        total_mem_mb = int(parts[2])
                        free_mem_mb = total_mem_mb - used_mem_mb
                        free_mem_gb = free_mem_mb / 1024
                        used_gb = used_mem_mb / 1024
                        total_gb = total_mem_mb / 1024
                        mem_percent = (used_mem_mb / total_mem_mb * 100) if total_mem_mb > 0 else 0
                        
                        print(f"   • GPU {gpu_idx}: {free_mem_gb:.2f}GiB free / {total_gb:.2f}GiB total ({mem_percent:.1f}% used)")
                        
                        if free_mem_gb >= min_free_gb:
                            available_gpus.append(gpu_idx)
                        else:
                            print(f"⚠️  Skipping GPU {gpu_idx}: insufficient free memory (need {min_free_gb}GiB)")
                        break
                    except (ValueError, IndexError):
                        # If we can't parse, include the GPU (better to try than skip)
                        print(f"⚠️  Could not parse memory info for GPU {gpu_idx}, assuming UNAVAILABLE to be safe")
                        # available_gpus.append(gpu_idx)  <-- Changed to exclude on error
                        break
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            # If nvidia-smi fails, include the GPU (better to try than skip)
            print(f"⚠️  Could not check GPU {gpu_idx} memory ({e}); including it anyway")
            available_gpus.append(gpu_idx)
    
    return available_gpus


def _distribute_work_items(items: List, gpu_indices: List[int]) -> Dict[int, List]:
    """
    Distribute work items across GPUs using round-robin assignment.
    
    Args:
        items: List of work items to distribute
        gpu_indices: List of GPU indices to use
    
    Returns:
        Dictionary mapping GPU index to list of work items
    """
    distribution: Dict[int, List] = {gpu_idx: [] for gpu_idx in gpu_indices}
    for i, item in enumerate(items):
        gpu_idx = gpu_indices[i % len(gpu_indices)]
        distribution[gpu_idx].append(item)
    return distribution


def _append_run_log(root_dir: str, filename: str, line: str) -> None:
    """
    Append a single line to a log file under the output root. Best-effort.
    """
    try:
        os.makedirs(root_dir, exist_ok=True)
        with open(os.path.join(root_dir, filename), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _append_run_log(root_dir: str, payload: Dict[str, object]) -> None:
    """
    Append a JSON line with run status to a log file for reruns/debugging.
    """
    try:
        log_path = os.path.join(root_dir, "run_log.ndjson")
        os.makedirs(root_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        # Logging must not crash the main run
        pass


def _progress_iterable(iterable, *, total: Optional[int] = None, desc: str = ""):
    """
    Lightweight progress wrapper. Uses tqdm if available, otherwise prints periodic updates.
    """
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        total_val = total
        if total_val is None and hasattr(iterable, "__len__"):
            try:
                total_val = len(iterable)  # type: ignore[arg-type]
            except Exception:
                total_val = None
        step = max(1, int(total_val / 50)) if total_val else 10
        def _generator():
            count = 0
            start = time.time()
            for item in iterable:
                yield item
                count += 1
                if total_val:
                    if count == 1 or count == total_val or count % step == 0:
                        elapsed = time.time() - start
                        rate = count / elapsed if elapsed > 0 else 0.0
                        eta = (total_val - count) / rate if rate > 0 else float("nan")
                        eta_str = f"{eta/60:.1f} min" if math.isfinite(eta) else "?"
                        print(f"{desc} {count}/{total_val} ({100*count/total_val:.1f}%), ETA {eta_str}")
                else:
                    if count == 1 or count % step == 0:
                        print(f"{desc} {count} done")
        return _generator()


def _select_retry_gpus(gpu_indices: List[int], success_counts: Counter, fail_counts: Counter) -> List[int]:
    """
    Choose GPUs to use for retries, preferring those with no failures.
    If all GPUs have failures, sort by lowest failure rate first.
    """
    healthy = [g for g in gpu_indices if fail_counts.get(g, 0) == 0]
    if healthy:
        return healthy

    def _score(g: int) -> Tuple[float, int]:
        f = fail_counts.get(g, 0)
        s = success_counts.get(g, 0)
        total = s + f
        rate = f / total if total > 0 else 1.0
        return (rate, f)

    return sorted(gpu_indices, key=_score)


def _run_cmip6_multigpu_with_retries(
    work_items: List[WorkItemA],
    gpu_indices: List[int],
    *,
    model_to_index: Dict[str, int],
    variables_2predict: List[str],
    lats: np.ndarray,
    lons: np.ndarray,
    output_specs: List[dict],
    root_dir: str,
    store_name: str,
    force_gpu_flag: bool,
    amp_flag: bool,
    model_version: str,
    model_number: str,
    dateofmetafile: str,
    debug_first_success: bool = False,
    debug_trace_emulation: bool = False,
    max_retries: int = 0,
    fail_fast: bool = False,
    items_per_gpu_process: int = GPU_TASKS_PER_CHILD,
    amp_retry_queue: Optional[Dict[Tuple[object, ...], WorkItemA]] = None,
) -> Tuple[int, int]:
    """
    Run CMIP6 (Option A) work items across multiple GPUs with optional retries.
    If amp_retry_queue is provided, failed items encountered with AMP disabled
    are added for a later AMP-on retry pass.
    Returns (total_successes, total_failures).
    """
    all_gpus = list(gpu_indices)
    pending = list(work_items)
    attempt = 0
    total_success = 0
    total_fail = 0
    success_counts: Counter = Counter()
    fail_counts: Counter = Counter()
    total_items = len(work_items)
    processed = 0
    amp_queue = amp_retry_queue

    def _queue_amp_retry(item: WorkItemA) -> None:
        if amp_queue is None:
            return
        amp_now = item.amp_override if item.amp_override is not None else amp_flag
        if amp_now:
            return
        key = _work_item_key(item)
        amp_queue[key] = replace(item, amp_override=True)

    def _clear_amp_retry(item: WorkItemA) -> None:
        if amp_queue is None:
            return
        key = _work_item_key(item)
        amp_queue.pop(key, None)

    while pending:
        active_gpus = all_gpus if attempt == 0 else _select_retry_gpus(all_gpus, success_counts, fail_counts)
        if not active_gpus:
            active_gpus = all_gpus

        distribution = _distribute_work_items(pending, active_gpus)
        max_per_child = max(1, items_per_gpu_process)
        max_len = max((len(v) for v in distribution.values()), default=0)
        if max_len == 0:
            break

        failed_next: List[WorkItemA] = []
        successes_round = 0
        failures_round = 0

        wave = 0
        for start in range(0, max_len, max_per_child):
            wave += 1
            tasks = [
                (gpu_idx, distribution.get(gpu_idx, [])[start:start + max_per_child],
                 model_to_index, variables_2predict,
                 lats, lons, output_specs, root_dir, store_name,
                 force_gpu_flag, amp_flag, model_version, model_number, dateofmetafile,
                 debug_first_success and attempt == 0 and wave == 1, debug_trace_emulation)
                for gpu_idx in active_gpus
                if distribution.get(gpu_idx) and distribution[gpu_idx][start:start + max_per_child]
            ]

            if not tasks:
                continue

            with get_context("spawn").Pool(processes=len(tasks)) as pool:
                results_list = pool.starmap(_worker_batch_a, tasks)

            for (task_gpu, items_for_gpu, *_), results in zip(tasks, results_list):
                for item, (success, msg) in zip(items_for_gpu, results):
                    message = f"[GPU {task_gpu}] {msg}"
                    if success:
                        successes_round += 1
                        success_counts[task_gpu] += 1
                        _clear_amp_retry(item)
                    else:
                        failures_round += 1
                        fail_counts[task_gpu] += 1
                        failed_next.append(item)
                        _queue_amp_retry(item)
                        print(f"❌ {message}")
                        if fail_fast:
                            raise RuntimeError(f"Fail-fast: first emulation failure on GPU {task_gpu}: {msg}")

            processed_chunk = sum(len(t[1]) for t in tasks)
            processed += processed_chunk
            if total_items:
                pct = 100 * processed / total_items
                print(f"📈 Progress: {processed}/{total_items} ({pct:.1f}%) after attempt {attempt + 1}, wave {wave} (chunk size {max_per_child})")

        total_success += successes_round
        total_fail += failures_round

        if not failed_next:
            break

        if attempt >= max_retries:
            print(f"⚠️  Max GPU retries reached ({max_retries}); {len(failed_next)} items remain failed.")
            break

        attempt += 1
        pending = failed_next
        next_gpus = _select_retry_gpus(all_gpus, success_counts, fail_counts)
        if not next_gpus:
            next_gpus = all_gpus
        print(f"🔁 Retrying {len(pending)} failed items (attempt {attempt}/{max_retries}) on GPUs {next_gpus}")

    return total_success, total_fail


def _run_magicc_multigpu_with_retries(
    work_items: List[WorkItemB],
    gpu_indices: List[int],
    *,
    df_wide: pd.DataFrame,
    model_to_index: Dict[str, int],
    variables_2predict: List[str],
    lats: np.ndarray,
    lons: np.ndarray,
    output_specs: List[dict],
    root_dir: str,
    store_name: str,
    force_gpu_flag: bool,
    amp_flag: bool,
    model_version: str,
    model_number: str,
    dateofmetafile: str,
    mode_name: str,
    effect_model_scheme: Union[str, int, List[int], List[str]],
    effect_model_scheme_c: Optional[Union[str, List[int], List[str]]] = None,
    debug_first_success: bool = False,
    debug_trace_emulation: bool = False,
    max_retries: int = 0,
    fail_fast: bool = False,
    items_per_gpu_process: int = GPU_TASKS_PER_CHILD,
    amp_retry_queue: Optional[Dict[Tuple[object, ...], WorkItemB]] = None,
) -> Tuple[int, int]:
    """
    Run MAGICC work items across multiple GPUs with optional retries of failed items.
    If amp_retry_queue is provided, failures while running without AMP are queued
    for a later AMP-on retry pass.
    Returns (total_successes, total_failures).
    """
    all_gpus = list(gpu_indices)
    pending = list(work_items)
    attempt = 0
    total_success = 0
    total_fail = 0
    success_counts: Counter = Counter()
    fail_counts: Counter = Counter()
    total_items = len(work_items)
    processed = 0
    amp_queue = amp_retry_queue

    def _queue_amp_retry(item: WorkItemB) -> None:
        if amp_queue is None:
            return
        amp_now = item.amp_override if item.amp_override is not None else amp_flag
        if amp_now:
            return
        key = _work_item_key(item)
        amp_queue[key] = replace(item, amp_override=True)

    def _clear_amp_retry(item: WorkItemB) -> None:
        if amp_queue is None:
            return
        key = _work_item_key(item)
        amp_queue.pop(key, None)

    while pending:
        active_gpus = all_gpus if attempt == 0 else _select_retry_gpus(all_gpus, success_counts, fail_counts)
        if not active_gpus:
            active_gpus = all_gpus

        distribution = _distribute_work_items(pending, active_gpus)
        max_per_child = max(1, items_per_gpu_process)
        max_len = max((len(v) for v in distribution.values()), default=0)
        if max_len == 0:
            break

        failed_next: List[WorkItemB] = []
        successes_round = 0
        failures_round = 0

        wave = 0
        for start in range(0, max_len, max_per_child):
            wave += 1
            tasks = [
                (gpu_idx, distribution.get(gpu_idx, [])[start:start + max_per_child],
                 df_wide, model_to_index, variables_2predict,
                 lats, lons, output_specs, root_dir, store_name,
                 force_gpu_flag, amp_flag, model_version, model_number, dateofmetafile,
                 mode_name, effect_model_scheme, effect_model_scheme_c,
                 debug_first_success and attempt == 0 and wave == 1, debug_trace_emulation)
                for gpu_idx in active_gpus
                if distribution.get(gpu_idx) and distribution[gpu_idx][start:start + max_per_child]
            ]

            if not tasks:
                continue

            with get_context("spawn").Pool(processes=len(tasks)) as pool:
                results_list = pool.starmap(_worker_batch_b, tasks)

            for (task_gpu, items_for_gpu, *_), results in zip(tasks, results_list):
                for item, (success, msg) in zip(items_for_gpu, results):
                    message = f"[GPU {task_gpu}] {msg}"
                    if success:
                        successes_round += 1
                        success_counts[task_gpu] += 1
                        _clear_amp_retry(item)
                    else:
                        failures_round += 1
                        fail_counts[task_gpu] += 1
                        failed_next.append(item)
                        _queue_amp_retry(item)
                        print(f"❌ {message}")
                        if fail_fast:
                            raise RuntimeError(f"Fail-fast: first emulation failure on GPU {task_gpu}: {msg}")

            processed_chunk = sum(len(t[1]) for t in tasks)
            processed += processed_chunk
            if total_items:
                pct = 100 * processed / total_items
                print(f"📈 Progress: {processed}/{total_items} ({pct:.1f}%) after attempt {attempt + 1}, wave {wave} (chunk size {max_per_child})")

        total_success += successes_round
        total_fail += failures_round

        if not failed_next:
            break

        if attempt >= max_retries:
            # Give up after max_retries; keep the failures counted.
            print(f"⚠️  Max GPU retries reached ({max_retries}); {len(failed_next)} items remain failed.")
            break

        attempt += 1
        pending = failed_next
        next_gpus = _select_retry_gpus(all_gpus, success_counts, fail_counts)
        if not next_gpus:
            next_gpus = all_gpus
        print(f"🔁 Retrying {len(pending)} failed items (attempt {attempt}/{max_retries}) on GPUs {next_gpus}")

    return total_success, total_fail


def _run_magiccxera5_multigpu_with_retries(
    work_items: List[WorkItemD],
    gpu_indices: List[int],
    *,
    model_to_index: Dict[str, int],
    variables_2predict: List[str],
    lats: np.ndarray,
    lons: np.ndarray,
    output_specs: List[dict],
    root_dir: str,
    store_name: str,
    force_gpu_flag: bool,
    amp_flag: bool,
    model_version: str,
    model_number: str,
    dateofmetafile: str,
    debug_first_success: bool = False,
    debug_trace_emulation: bool = False,
    max_retries: int = 0,
    fail_fast: bool = False,
    items_per_gpu_process: int = GPU_TASKS_PER_CHILD,
    amp_retry_queue: Optional[Dict[Tuple[object, ...], WorkItemD]] = None,
) -> Tuple[int, int]:
    """
    Run MAGICCxERA5 (Option D) work items across multiple GPUs with optional retries.
    """
    all_gpus = list(gpu_indices)
    pending = list(work_items)
    attempt = 0
    total_success = 0
    total_fail = 0
    success_counts: Counter = Counter()
    fail_counts: Counter = Counter()
    total_items = len(work_items)
    processed = 0
    amp_queue = amp_retry_queue

    def _queue_amp_retry(item: WorkItemD) -> None:
        if amp_queue is None:
            return
        amp_now = item.amp_override if item.amp_override is not None else amp_flag
        if amp_now:
            return
        key = _work_item_key(item)
        amp_queue[key] = replace(item, amp_override=True)

    def _clear_amp_retry(item: WorkItemD) -> None:
        if amp_queue is None:
            return
        key = _work_item_key(item)
        amp_queue.pop(key, None)

    while pending:
        active_gpus = all_gpus if attempt == 0 else _select_retry_gpus(all_gpus, success_counts, fail_counts)
        if not active_gpus:
            active_gpus = all_gpus

        distribution = _distribute_work_items(pending, active_gpus)
        max_per_child = max(1, items_per_gpu_process)
        max_len = max((len(v) for v in distribution.values()), default=0)
        if max_len == 0:
            break

        failed_next: List[WorkItemD] = []
        successes_round = 0
        failures_round = 0

        wave = 0
        for start in range(0, max_len, max_per_child):
            wave += 1
            tasks = [
                (gpu_idx, distribution.get(gpu_idx, [])[start:start + max_per_child],
                 model_to_index, variables_2predict,
                 lats, lons, output_specs, root_dir, store_name,
                 force_gpu_flag, amp_flag, model_version, model_number, dateofmetafile,
                 debug_first_success and attempt == 0 and wave == 1, debug_trace_emulation)
                for gpu_idx in active_gpus
                if distribution.get(gpu_idx) and distribution[gpu_idx][start:start + max_per_child]
            ]

            if not tasks:
                continue

            with get_context("spawn").Pool(processes=len(tasks)) as pool:
                results_list = pool.starmap(_worker_batch_d, tasks)

            for (task_gpu, items_for_gpu, *_), results in zip(tasks, results_list):
                for item, (success, msg) in zip(items_for_gpu, results):
                    message = f"[GPU {task_gpu}] {msg}"
                    if success:
                        successes_round += 1
                        success_counts[task_gpu] += 1
                        _clear_amp_retry(item)
                    else:
                        failures_round += 1
                        fail_counts[task_gpu] += 1
                        failed_next.append(item)
                        _queue_amp_retry(item)
                        print(f"❌ {message}")
                        if fail_fast:
                            raise RuntimeError(f"Fail-fast: first emulation failure on GPU {task_gpu}: {msg}")

            processed_chunk = sum(len(t[1]) for t in tasks)
            processed += processed_chunk
            if total_items:
                pct = 100 * processed / total_items
                print(f"📈 Progress: {processed}/{total_items} ({pct:.1f}%) after attempt {attempt + 1}, wave {wave} (chunk size {max_per_child})")

        total_success += successes_round
        total_fail += failures_round

        if not failed_next:
            break

        if attempt >= max_retries:
            print(f"⚠️  Max GPU retries reached ({max_retries}); {len(failed_next)} items remain failed.")
            break

        attempt += 1
        pending = failed_next

    return total_success, total_fail


def _worker_batch_a(
    gpu_idx: int,
    items: List[WorkItemA],
    model_to_index: Dict[str, int],
    variables_2predict: List[str],
    lats: np.ndarray,
    lons: np.ndarray,
    output_specs: List[dict],
    root_dir: str,
    store_name: str,
    force_gpu: bool,
    amp_flag: bool,
    model_version: str,
    model_number: str,
    dateofmetafile: str,
    debug_first_success: bool,
    debug_trace_emulation: bool,
) -> List[Tuple[bool, str]]:
    """Process a batch of Option A work items on a GPU."""
    # Set environment variable before any PyTorch CUDA operations
    # This helps reduce memory fragmentation in spawned processes
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Bind the process to the physical GPU index we were assigned.
    # We *do not* touch CUDA_VISIBLE_DEVICES here; with the "spawn" start
    # method CUDA is usually initialised before this runs, so changing the
    # env var is ineffective and caused all workers to pile onto GPU 0.
    physical_gpu_idx = gpu_idx
    local_gpu_idx = physical_gpu_idx
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_gpu_idx)
        except Exception as e:
            print(
                f"[worker A batch] ⚠️  Could not set CUDA device to {local_gpu_idx}: {e}",
                flush=True,
            )

    # Initial GPU memory cleanup
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(local_gpu_idx)
            gc.collect()
        except Exception:
            pass
    
    results = []
    debug_logged = False
    for item in items:
        result = _worker_process_gpu_a(
            item, gpu_idx,
            model_to_index=model_to_index,
            variables_2predict=variables_2predict,
            lats=lats, lons=lons,
            output_specs=output_specs,
            root_dir=root_dir, store_name=store_name,
            force_gpu=force_gpu,
            amp_flag=amp_flag,
            model_version=model_version,
            model_number=model_number,
            dateofmetafile=dateofmetafile,
            debug_trace_emulation=debug_trace_emulation,
        )
        results.append(result)
        if debug_first_success and not debug_logged and result[0]:
            print(f"🔍 First-success debug ({gpu_idx=}): {result[1]}")
            sys.stdout.flush()
            debug_logged = True
        
        # Aggressive memory cleanup between work items to prevent fragmentation
        if torch.cuda.is_available():
            try:
                gc.collect()
                torch.cuda.synchronize(local_gpu_idx)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(local_gpu_idx)
                gc.collect()
            except Exception:
                pass

    # Final cleanup before returning
    if torch.cuda.is_available():
        try:
            gc.collect()
            torch.cuda.synchronize(local_gpu_idx)
            torch.cuda.empty_cache()
        except Exception:
            pass

    return results


def _worker_batch_b(
    gpu_idx: int,
    items: List[WorkItemB],
    df_wide: pd.DataFrame,
    model_to_index: Dict[str, int],
    variables_2predict: List[str],
    lats: np.ndarray,
    lons: np.ndarray,
    output_specs: List[dict],
    root_dir: str,
    store_name: str,
    force_gpu: bool,
    amp_flag: bool,
    model_version: str,
    model_number: str,
    dateofmetafile: str,
    mode_name: str,
    effect_model_scheme: Union[str, int, List[int], List[str]],
    effect_model_scheme_c: Optional[Union[str, List[int], List[str]]] = None,
    debug_first_success: bool = False,
    debug_trace_emulation: bool = False,
) -> List[Tuple[bool, str]]:
    """Process a batch of Option B work items on a GPU."""
    # Set environment variable before any PyTorch CUDA operations
    # This helps reduce memory fragmentation in spawned processes
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    physical_gpu_idx = gpu_idx
    local_gpu_idx = physical_gpu_idx
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_gpu_idx)
        except Exception as e:
            print(
                f"[worker B batch] ⚠️  Could not set CUDA device to {local_gpu_idx}: {e}",
                flush=True,
            )

    # Initial GPU memory cleanup
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(local_gpu_idx)
            gc.collect()
        except Exception:
            pass
    
    results = []
    debug_logged = False
    for item in items:
        result = _worker_process_gpu_b(
            item, gpu_idx,
            df_wide=df_wide,
            model_to_index=model_to_index,
            variables_2predict=variables_2predict,
            lats=lats, lons=lons,
            output_specs=output_specs,
            root_dir=root_dir, store_name=store_name,
            force_gpu=force_gpu,
            amp_flag=amp_flag,
            model_version=model_version,
            model_number=model_number,
            dateofmetafile=dateofmetafile,
            mode_name=mode_name,
            effect_model_scheme=effect_model_scheme,
            effect_model_scheme_c=effect_model_scheme_c,
            debug_trace_emulation=debug_trace_emulation,
        )
        results.append(result)
        if debug_first_success and not debug_logged and result[0]:
            print(f"🔍 First-success debug ({gpu_idx=}): {result[1]}")
            sys.stdout.flush()
            debug_logged = True
        
        # Aggressive memory cleanup between work items to prevent fragmentation
        if torch.cuda.is_available():
            try:
                gc.collect()
                torch.cuda.synchronize(local_gpu_idx)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(local_gpu_idx)
                gc.collect()
            except Exception:
                pass

    # Final cleanup before returning
    if torch.cuda.is_available():
        try:
            gc.collect()
            torch.cuda.synchronize(local_gpu_idx)
            torch.cuda.empty_cache()
        except Exception:
            pass

    return results


def _worker_batch_d(
    gpu_idx: int,
    items: List[WorkItemD],
    model_to_index: Dict[str, int],
    variables_2predict: List[str],
    lats: np.ndarray,
    lons: np.ndarray,
    output_specs: List[dict],
    root_dir: str,
    store_name: str,
    force_gpu: bool,
    amp_flag: bool,
    model_version: str,
    model_number: str,
    dateofmetafile: str,
    debug_first_success: bool = False,
    debug_trace_emulation: bool = False,
) -> List[Tuple[bool, str]]:
    """Process a batch of Option D work items on a GPU."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    physical_gpu_idx = gpu_idx
    local_gpu_idx = physical_gpu_idx
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_gpu_idx)
        except Exception as e:
            print(
                f"[worker D batch] ⚠️  Could not set CUDA device to {local_gpu_idx}: {e}",
                flush=True,
            )

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(local_gpu_idx)
            gc.collect()
        except Exception:
            pass

    results = []
    debug_logged = False
    for item in items:
        result = _worker_process_gpu_d(
            item, gpu_idx,
            model_to_index=model_to_index,
            variables_2predict=variables_2predict,
            lats=lats, lons=lons,
            output_specs=output_specs,
            root_dir=root_dir, store_name=store_name,
            force_gpu=force_gpu,
            amp_flag=amp_flag,
            model_version=model_version,
            model_number=model_number,
            dateofmetafile=dateofmetafile,
            debug_trace_emulation=debug_trace_emulation,
        )
        results.append(result)
        if debug_first_success and not debug_logged and result[0]:
            print(f"🔍 First-success debug ({gpu_idx=}): {result[1]}")
            sys.stdout.flush()
            debug_logged = True

        if torch.cuda.is_available():
            try:
                gc.collect()
                torch.cuda.synchronize(local_gpu_idx)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(local_gpu_idx)
                gc.collect()
            except Exception:
                pass

    if torch.cuda.is_available():
        try:
            gc.collect()
            torch.cuda.synchronize(local_gpu_idx)
            torch.cuda.empty_cache()
        except Exception:
            pass

    return results


# =============================================================================
# Main orchestration: per-option execution and segment writing
# =============================================================================

def _today_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%b%d_%H%M")

def main():
    parser = argparse.ArgumentParser(description="Probabilistic segments driver for GCMagicc (320_*)")
    parser.add_argument("--source-x", choices=["CMIP6","MAGICC","MAGICC-SAMEPERCMIP6","MAGICCxERA5"], default=SOURCE_X_PREDICTORS)
    parser.add_argument("--input-dir", default=DEFAULT_CMIP6_INPUT_DIR, help="Dir with DAT_*.nc (Option A)")
    parser.add_argument("--scm-parquet", default=SCM_RESULTS_PARQUET, help="Path to MAGICC parquet (Option B)")
    parser.add_argument("--ensembles", type=int, default=None, help="Override ENSEMBLES_* (per file for A; across members for B)")
    parser.add_argument("--nameplate", type=str, default=None, help="Override auto-generated nameplate (default: auto-generated from config)")
    parser.add_argument("--save-store", default=None, help="Output Zarr store path (default: auto-generated based on nameplate and timestamp)")
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed for reproducibility")
    parser.add_argument("--bias-to-era5", type=int, choices=[0,1], default=None, help="Override BIASCORRECT_TO_ERA5 flag for chosen source")
    parser.add_argument(
        "--workflow",
        type=str,
        choices=["AR6", "AR7", "all"],
        default=None,
        help="Workflow selection for MAGICCxERA5: AR6, AR7, or 'all' to search both (default: from config/env).",
    )
    parser.add_argument(
        "--runmodus",
        type=str,
        choices=["all", "natural", "aerosol", "anthropogenic"],
        default=None,
        help="Runmodus selection for MAGICCxERA5: all, natural, aerosol, or anthropogenic (default: from config/env).",
    )
    parser.add_argument("--device", default=None, help="Override automatic device selection (e.g., 'cuda:0', 'gpu', 'cpu')")
    parser.add_argument("--force-gpu", action="store_true", help="Force GPU usage even if torch.cuda.is_available() returns False")
    parser.add_argument("--amp", action="store_true", help="Enable AMP autocast on CUDA for faster inference")
    parser.add_argument("--no-tf32", dest="tf32", action="store_false", help="Disable TF32 on Ampere+ (enabled by default on CUDA)")
    parser.add_argument("--debug-first-success", action="store_true", help="Print immediately when the first work item finishes successfully")
    parser.add_argument("--debug-trace-emulation", action="store_true", help="Print detailed tracebacks/context when emulation or writing fails")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use for parallel execution (default: auto-detect all available)")
    parser.add_argument("--gpu-list", type=str, default=None, help="Comma-separated list of GPU indices to use (e.g., '0,1,2'). Overrides --gpus.")
    parser.set_defaults(tf32=True)
    args = parser.parse_args()

    # Check debug flag: script variable > CLI flag > environment variable
    debug_first_success = DEBUG_FIRST_SUCCESS  # Start with script-level setting
    if args.debug_first_success:
        debug_first_success = True  # CLI flag overrides script setting
    else:
        # Check environment variable if not set via CLI
        env_debug_flag = os.environ.get("GCMAGICC_DEBUG_FIRST_SUCCESS")
        if env_debug_flag is not None:
            # Handle string values from environment (env vars are always strings)
            normalized = env_debug_flag.strip().lower()
            if normalized not in ("", "0", "false", "no"):
                debug_first_success = True
    if debug_first_success:
        print("🔍 First-success debug logging enabled; reporting after the first successful work item.")

    env_trace_flag = os.environ.get("GCMAGICC_DEBUG_TRACE_EMULATION")
    debug_trace_emulation = bool(args.debug_trace_emulation)
    if env_trace_flag is not None:
        normalized = env_trace_flag.strip().lower()
        if normalized not in ("", "0", "false", "no"):
            debug_trace_emulation = True
    if debug_trace_emulation:
        print("🧪 Emulation-trace debug enabled; stack traces will be printed on first error per worker.")

    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Re-detect device with force_gpu if requested (avoid creating CUDA context)
    force_gpu_flag = args.force_gpu or _FORCE_GPU
    device_for_run = detect_default_device(force_gpu=force_gpu_flag)

    # Parse GPU list from CLI (if any)
    gpu_list_parsed: Optional[List[int]] = None
    if args.gpu_list:
        try:
            gpu_list_parsed = [int(x.strip()) for x in args.gpu_list.split(",")]
        except ValueError:
            print(f"⚠️  Invalid --gpu-list format '{args.gpu_list}'. Expected comma-separated integers (e.g., '0,1,2').")
            return 1

    # Decide parallel GPU usage BEFORE any CUDA preflight to avoid parent context
    # Only honor a single-GPU device override for parallel selection if the user
    # explicitly provided --device. Otherwise, use all visible GPUs by default.
    override_for_parallel: Optional[str] = None
    if args.device is not None:
        override_for_parallel = normalize_device_string(args.device)
        # allow explicit CPU request to force CPU mode
        if override_for_parallel == "cpu":
            pass
        elif override_for_parallel and override_for_parallel.startswith("cuda"):
            if not _cuda_available() and not force_gpu_flag:
                print(f"⚠️  CLI requested device '{override_for_parallel}' but CUDA is unavailable; using CPU instead.")
                override_for_parallel = "cpu"
    else:
        # If auto-detected CPU, force CPU mode; otherwise don't constrain GPUs.
        if device_for_run == "cpu":
            override_for_parallel = "cpu"

    # Determine GPU usage and multiprocessing strategy
    gpu_indices, use_multiprocessing = select_gpus_for_parallel(
        requested_gpus=args.gpus,
        gpu_list=gpu_list_parsed,
        device_override=override_for_parallel,
    )

    # --- PATCH: Strict startup memory check ---
    if gpu_indices:
        print(f"🛡️  Performing strict startup memory check on GPUs {gpu_indices} (need >{MIN_FREE_MEMORY_GB}GB free)...")
        # Filter out any GPUs that are already full (like GPU 0 in your logs)
        gpu_indices = _filter_gpus_by_memory(gpu_indices, min_free_gb=MIN_FREE_MEMORY_GB)
        if not gpu_indices:
            print("❌ No GPUs passed the strict memory check. Exiting to avoid OOM.")
            return 1
        use_multiprocessing = len(gpu_indices) > 1
        print(f"✅ Active GPUs after filter: {gpu_indices}")
    # ------------------------------------------

    # AMP/TF32 flags (env or CLI)
    amp_flag = DEFAULT_AMP or bool(args.amp)
    tf32_flag = bool(os.environ.get("GCMAGICC_TF32", "1").lower() not in ("0","false","no")) and bool(args.tf32)

    cpu_pool_workers = 1  # only used for CPU-only runs

    if use_multiprocessing:
        print(f"🚀 Multi-GPU mode: Using GPUs {gpu_indices} for parallel execution")
        # In multi-GPU mode, avoid pre-initializing CUDA in the parent.
        # We'll enable TF32/AMP inside worker processes where the device is set.
    elif gpu_indices:
        print(f"🖥️  Single-GPU mode: Using GPU {gpu_indices[0]}")
        physical_gpu = gpu_indices[0]
        # Mask visibility to the chosen GPU so models cannot spill onto others
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
            _record_device_note(f"CUDA_VISIBLE_DEVICES masked to GPU {physical_gpu}; internal index becomes 0.")
        except Exception:
            pass
        device_for_run = "cuda:0"
        # If CUDA chosen, try enabling TF32 and preflight (safe in single process)
        if _cuda_available() or force_gpu_flag:
            _maybe_enable_tf32(tf32_flag)
            ok = _gpu_preflight(device_for_run, warn=not force_gpu_flag)
            if not ok and not force_gpu_flag:
                print("⚠️  GPU preflight failed; falling back to CPU.")
                device_for_run = "cpu"
        report_device_status(device_for_run)
    else:
        print(f"💻 CPU mode: Using CPU for computation")
        # Respect explicit --device cpu overrides by forcing device_for_run here
        device_for_run = "cpu"
        report_device_status(device_for_run)
        # Estimate a safe CPU pool size when GPUs are not used
        cpu_pool_workers = _estimate_cpu_worker_count(
            per_job_cpus=7,
            per_job_ram_gb=45.0,
            safety_fraction=0.20,
        )
        if cpu_pool_workers > 1:
            total_mem_gb, avail_mem_gb = _memory_gb()
            mem_str = ""
            if total_mem_gb > 0:
                mem_str = f"; RAM≈{total_mem_gb:.1f}GB (avail≈{avail_mem_gb:.1f}GB)"
            print(
                f"🧠 CPU parallel planner: {cpu_pool_workers} worker(s) "
                f"(~7 cores & 45GB per job, 20% safety buffer{mem_str})"
            )

    # grid
    lats, lons = generate_coordinate_grids(nlat=N_LAT, nlon=N_LAT*2, lon_convention=LON_CONVENTION, lat_direction="north_to_south")

    # meta access
    model_to_index: Dict[str,int] = META["model_to_index"]
    variables_2predict: List[str] = META.get("variables", [])

    # Determine which option will be used and generate nameplate + output path
    option = args.source_x.upper()
    
    # Generate nameplate based on runtime configuration
    # First, get the actual values that will be used (may be overridden by CLI args)
    if option == "CMIP6":
        # Use actual values that will be used (from args or defaults)
        n_draws_val = args.ensembles if args.ensembles is not None else ENSEMBLES_A
        bias_to_era5_val = BIASCORRECT_TO_ERA5_A if args.bias_to_era5 is None else bool(args.bias_to_era5)
        nameplate = args.nameplate if args.nameplate is not None else _generate_nameplate(
            option=option,
            bias_to_era5_a=bias_to_era5_val,
            source_id_whitelist=SOURCE_ID_WHITELIST,
            experiment_id_whitelist=EXPERIMENT_ID_WHITELIST,
            ensembles_a=n_draws_val,
            model_version=MODEL_VERSION,
        )
    elif option == "MAGICC":
        n_draws_val = args.ensembles if args.ensembles is not None else ENSEMBLES_B
        bias_to_era5_val = BIASCORRECT_TO_ERA5_B if args.bias_to_era5 is None else bool(args.bias_to_era5)
        nameplate = args.nameplate if args.nameplate is not None else _generate_nameplate(
            option=option,
            bias_to_era5_b=bias_to_era5_val,
            experiment_id_whitelist=EXPERIMENT_ID_WHITELIST,
            ensembles_b=n_draws_val,
            model_version=MODEL_VERSION,
        )
    elif option == "MAGICC-SAMEPERCMIP6":
        n_draws_val = args.ensembles if args.ensembles is not None else ENSEMBLES_C_PER_CMIP6
        bias_to_era5_val = BIASCORRECT_TO_ERA5_C if args.bias_to_era5 is None else bool(args.bias_to_era5)
        nameplate = args.nameplate if args.nameplate is not None else _generate_nameplate(
            option=option,
            bias_to_era5_c=bias_to_era5_val,
            experiment_id_whitelist=EXPERIMENT_ID_WHITELIST,
            ensembles_c_per_cmip6=n_draws_val,
            model_version=MODEL_VERSION,
        )
    elif option == "MAGICCXERA5":
        n_draws_val = args.ensembles if args.ensembles is not None else ENSEMBLES_D
        bias_to_era5_val = BIASCORRECT_TO_ERA5_D if args.bias_to_era5 is None else bool(args.bias_to_era5)
        nameplate = args.nameplate if args.nameplate is not None else _generate_nameplate(
            option=option,
            bias_to_era5_b=bias_to_era5_val,
            experiment_id_whitelist=EXPERIMENT_ID_WHITELIST,
            ensembles_b=n_draws_val,
            model_version=MODEL_VERSION,
        )
    else:
        raise ValueError("--source-x must be 'CMIP6', 'MAGICC', 'MAGICC-SAMEPERCMIP6', or 'MAGICCxERA5'")
    
    # Generate output folder path if not provided
    if args.save_store is None:
        folder_name = _get_output_folder_name(nameplate)
        # Use dedicated segments folder under data/
        data_dir = str(get_data_path("segments"))
        output_dir = os.path.join(data_dir, folder_name)
        args.save_store = os.path.join(output_dir, OUTPUT_STORE_NAME)
        print(f"📁 Auto-generated output path: {args.save_store}")
    
    # split store path
    root_dir, store_name = os.path.dirname(args.save_store), os.path.basename(args.save_store)
    os.makedirs(root_dir, exist_ok=True)

    # -------------------------------- Option A: CMIP6
    if args.source_x.upper() == "CMIP6":
        input_dir = args.input_dir
        files = _discover_cmip6_files(input_dir)
        files_sel = _select_cmip6_files(files, SOURCE_ID_WHITELIST, EXPERIMENT_ID_WHITELIST, MAX_N_MEMBERS_PER_SOURCE)
        if not files_sel:
            print("No CMIP6 files matching selection. Exiting.")
            return 0
        n_draws = args.ensembles if args.ensembles is not None else ENSEMBLES_A
        n_draws = max(1, min(1000, int(n_draws)))
        bias_to_era5 = BIASCORRECT_TO_ERA5_A if args.bias_to_era5 is None else bool(args.bias_to_era5)

        # NEW: compute merged requested year windows once
        req_windows = _collect_requested_year_windows(OUTPUTDICTLIST)
        if not req_windows:
            print("⚠️ No periods found in OUTPUTDICTLIST; will fall back to file-native span per file.")
        print(f"🟦 Option A / CMIP6: {len(files_sel)} file(s), {n_draws} draws each; bias_to_era5={bias_to_era5}")

        # Build a mapping: (source_id, member_id) -> {experiment_id: file_path}
        # This allows us to find matching historical files for SSP scenarios
        file_map: Dict[Tuple[str, str], Dict[str, str]] = {}
        # Track which experiment_ids are explicitly requested (not just included as fallbacks)
        explicitly_requested_experiments = set(EXPERIMENT_ID_WHITELIST)
        
        for fpath in files_sel:
            source_id, experiment_id, member_id = _parse_cmip6_name(fpath)
            if source_id is None or member_id is None:
                continue
            key = (source_id, member_id)
            if key not in file_map:
                file_map[key] = {}
            file_map[key][experiment_id] = fpath

        # Generate all work items
        work_items: List[WorkItemA] = []
        for fpath in files_sel:
            # NEW: intersect requested windows with file's available years
            try:
                f_sy, f_sm, f_ey, f_em = _infer_time_bounds_from_nc(fpath)
            except Exception as e:
                print(f"❌ Skipping {Path(fpath).name}: cannot infer time bounds: {e}")
                continue
            file_span = (f_sy, f_ey)
            
            # parse identifiers
            source_id, experiment_id, member_id = _parse_cmip6_name(fpath)
            if source_id is None:
                source_id = "unknown"
            if member_id is None:
                member_id = "unknown"

            # Check if this is an SSP scenario
            is_ssp = experiment_id.lower().startswith('ssp') if experiment_id else False
            # Check if this historical file is only included as a fallback (not explicitly requested)
            is_historical_fallback = (experiment_id and experiment_id.lower() == 'historical' and 
                                     'historical' not in explicitly_requested_experiments)

            usebias_model = 0 if bias_to_era5 else None
            useeffect_model = model_to_index.get(source_id, None) if bias_to_era5 else None

            # Skip historical files that are only included as fallbacks (they'll be processed when needed)
            if is_historical_fallback:
                continue

            # Determine which windows to process for this file
            run_windows = []
            if req_windows:
                for w in req_windows:
                    # Check if this window is historical
                    is_historical = _is_historical_period(w)
                    
                    # If window is historical and we have an SSP scenario, try to use historical file
                    if is_historical and is_ssp:
                        file_key = (source_id, member_id)
                        if file_key in file_map and 'historical' in file_map[file_key]:
                            # Use historical file for this window
                            hist_fpath = file_map[file_key]['historical']
                            try:
                                hist_sy, hist_sm, hist_ey, hist_em = _infer_time_bounds_from_nc(hist_fpath)
                                hist_span = (hist_sy, hist_ey)
                                iw = _intersect(w, hist_span)
                                if iw is not None:
                                    # Create work item with historical file
                                    for draw_idx in range(n_draws):
                                        seed_now = (args.seed or 0) + draw_idx + hash((hist_fpath, MODEL_NUMBER)) % 10_000
                                        work_items.append(WorkItemA(
                                            fpath=hist_fpath,
                                            draw_idx=draw_idx,
                                            window=iw,
                                            source_id=source_id,
                                            experiment_id='historical',  # Use historical experiment_id
                                            member_id=member_id,
                                            usebias_model=usebias_model,
                                            useeffect_model=useeffect_model,
                                            seed_base=seed_now,
                                            bias_to_era5=bias_to_era5,
                                        ))
                            except Exception as e:
                                print(f"⚠️  Could not use historical file for {w}: {e}")
                        # Don't add this window to run_windows for the SSP file
                        continue
                    
                    # For non-historical windows or non-SSP scenarios, use current file
                    iw = _intersect(w, file_span)
                    if iw is not None:
                        run_windows.append(iw)
            else:
                run_windows = [file_span]
            
            # Create work items for windows that overlap with this file
            if run_windows:
                for draw_idx in range(n_draws):
                    seed_now = (args.seed or 0) + draw_idx + hash((fpath, MODEL_NUMBER)) % 10_000
                    for (w_sy, w_ey) in run_windows:
                        work_items.append(WorkItemA(
                            fpath=fpath,
                            draw_idx=draw_idx,
                            window=(w_sy, w_ey),
                            source_id=source_id,
                            experiment_id=experiment_id,
                            member_id=member_id,
                            usebias_model=usebias_model,
                            useeffect_model=useeffect_model,
                            seed_base=seed_now,
                            bias_to_era5=bias_to_era5,
                        ))

        if not work_items:
            print("No work items to process. Exiting.")
            return 0

        print(f"📋 Generated {len(work_items)} work items")

        # Process work items
        if use_multiprocessing and len(gpu_indices) > 1:
            # Multi-GPU parallel execution
            # Re-check GPU memory right before execution to catch GPUs that became busy
            gpu_indices_filtered = _filter_gpus_by_memory(gpu_indices, min_free_gb=MIN_FREE_MEMORY_GB)
            if not gpu_indices_filtered:
                raise RuntimeError(f"All GPUs {gpu_indices} have insufficient free memory. Cannot proceed.")
            if len(gpu_indices_filtered) < len(gpu_indices):
                print(f"📊 Filtered GPUs: using {gpu_indices_filtered} (excluded {set(gpu_indices) - set(gpu_indices_filtered)} due to memory constraints)")
                gpu_indices = gpu_indices_filtered
            amp_retry_queue: Dict[Tuple[object, ...], WorkItemA] = {}
            successes, failures = _run_cmip6_multigpu_with_retries(
                work_items, gpu_indices,
                model_to_index=model_to_index,
                variables_2predict=variables_2predict,
                lats=lats, lons=lons,
                output_specs=OUTPUTDICTLIST,
                root_dir=root_dir, store_name=store_name,
                force_gpu_flag=force_gpu_flag,
                amp_flag=amp_flag,
                model_version=MODEL_VERSION, model_number=MODEL_NUMBER, dateofmetafile=DATEOFMETAFILE,
                debug_first_success=debug_first_success,
                debug_trace_emulation=debug_trace_emulation,
                max_retries=MAX_GPU_RETRIES,
                fail_fast=False,
                items_per_gpu_process=GPU_TASKS_PER_CHILD,
                amp_retry_queue=amp_retry_queue,
            )
            if amp_retry_queue:
                fallback_items = list(amp_retry_queue.values())
                print(f"🔁 Queued {len(fallback_items)} items for AMP fallback; retrying with AMP=1")
                amp_success, amp_fail = _run_cmip6_multigpu_with_retries(
                    fallback_items, gpu_indices,
                    model_to_index=model_to_index,
                    variables_2predict=variables_2predict,
                    lats=lats, lons=lons,
                    output_specs=OUTPUTDICTLIST,
                    root_dir=root_dir, store_name=store_name,
                    force_gpu_flag=force_gpu_flag,
                    amp_flag=True,
                    model_version=MODEL_VERSION, model_number=MODEL_NUMBER, dateofmetafile=DATEOFMETAFILE,
                    debug_first_success=False,
                    debug_trace_emulation=debug_trace_emulation,
                    max_retries=max(1, MAX_GPU_RETRIES),
                    fail_fast=False,
                    items_per_gpu_process=GPU_TASKS_PER_CHILD,
                )
                successes += amp_success
                failures += amp_fail
            print(f"✅ Completed: {successes} successful, {failures} failed (max GPU retries={MAX_GPU_RETRIES})")
        else:
            # Single-process execution (single GPU or CPU)
            device_str = device_for_run
            first_success_announced = False
            amp_retry_queue_seq: Dict[Tuple[object, ...], WorkItemA] = {}
            use_cpu_pool = device_str == "cpu" and cpu_pool_workers > 1

            def _process_option_a_seq(items: List[WorkItemA], desc: str, amp_default: bool) -> Tuple[int, int]:
                nonlocal first_success_announced

                if use_cpu_pool:
                    tasks = [
                        (
                            item,
                            amp_default,
                            force_gpu_flag,
                            model_to_index,
                            variables_2predict,
                            lats,
                            lons,
                            OUTPUTDICTLIST,
                            root_dir,
                            store_name,
                            MODEL_VERSION,
                            MODEL_NUMBER,
                            DATEOFMETAFILE,
                            debug_trace_emulation,
                        )
                        for item in items
                    ]
                    successes_local, failures_local, retry_items = _run_cpu_pool(
                        tasks,
                        _worker_process_cpu_a,
                        workers=cpu_pool_workers,
                        desc=desc,
                        debug_first_success=debug_first_success,
                    )
                    for ri in retry_items:
                        amp_retry_queue_seq[_work_item_key(ri)] = ri
                    if debug_first_success and successes_local > 0:
                        first_success_announced = True
                    return successes_local, failures_local

                successes_local = 0
                failures_local = 0
                for item in _progress_iterable(items, total=len(items), desc=desc):
                    amp_for_item = item.amp_override if item.amp_override is not None else amp_default

                    # Set RNG seed
                    random.seed(item.seed_base)
                    np.random.seed(item.seed_base)
                    torch.manual_seed(item.seed_base)
                    
                    # Build predictors
                    try:
                        predictors = extract_predictors_from_nc(
                            item.fpath,
                            timespan=[int(item.window[0]), int(item.window[1])]
                        )
                    except Exception as e:
                        _log_emulation_exception(
                            "Option A predictor extraction failure (sequential)",
                            e,
                            debug_trace=debug_trace_emulation,
                            context={"file": item.fpath, "window": item.window},
                        )
                        msg = f"Predictor extraction failed for {Path(item.fpath).name} window {item.window}: {e}"
                        print(f"❌ {msg}")
                        if not amp_for_item:
                            amp_retry_queue_seq[_work_item_key(item)] = replace(item, amp_override=True)
                        failures_local += 1
                        continue

                    # Run emulation
                    try:
                        yhval = run_gcmagicc(
                            predictors,
                            dependence=True,
                            usebias_model=item.usebias_model,
                            useeffect_model=item.useeffect_model,
                            device=device_str,
                            force_gpu=force_gpu_flag,
                            amp=amp_for_item,
                        )
                    except Exception as e:
                        _log_emulation_exception(
                            "Option A run_gcmagicc failure (sequential)",
                            e,
                            debug_trace=debug_trace_emulation,
                            context={
                                "file": item.fpath,
                                "window": item.window,
                                "draw_idx": item.draw_idx,
                                "device": device_str,
                                "usebias_model": item.usebias_model,
                                "useeffect_model": item.useeffect_model,
                                "amp": amp_for_item,
                            },
                        )
                        msg = f"Run failed for {Path(item.fpath).name} [draw {item.draw_idx+1}/{n_draws}] window {item.window}: {e}"
                        print(f"❌ {msg}")
                        if not amp_for_item:
                            amp_retry_queue_seq[_work_item_key(item)] = replace(item, amp_override=True)
                        failures_local += 1
                        continue

                    # Generate run metadata
                    run_id = (
                        f"A_{item.source_id}_{item.experiment_id or 'unknown'}_{item.member_id or 'N'}"
                        f"__b{0 if item.bias_to_era5 else 'N'}e{item.useeffect_model if item.useeffect_model is not None else 'N'}"
                        f"__m{item.draw_idx:04d}__win{int(item.window[0])}-{int(item.window[1])}__{_today_stamp()}__{uuid.uuid4().hex[:8]}"
                    )
                    run_meta = {
                        "run_id": run_id,
                        "mode": "CMIP6",
                        "source_id": item.source_id,
                        "experiment_id": item.experiment_id or "N",
                        "member_id": item.member_id or "N",
                        "usebias_model": 0 if item.bias_to_era5 else "None",
                        "useeffect_model": item.useeffect_model if item.useeffect_model is not None else "None",
                        "model_version": MODEL_VERSION,
                        "model_number": MODEL_NUMBER,
                        "model_id": f"{MODEL_VERSION}_{MODEL_NUMBER}",
                        "date_meta": DATEOFMETAFILE,
                        "device": device_str,
                        "amp": bool(amp_for_item),
                    }

                    apply_output_specs_and_write(
                        yhval=yhval,
                        predictors=predictors,
                        lats=lats, lons=lons,
                        output_specs=OUTPUTDICTLIST,
                        run_meta=run_meta,
                        variables_2predict=variables_2predict,
                        root_dir=root_dir, store_name=store_name,
                    )

                    if debug_first_success and not first_success_announced:
                        print(f"🔍 First-success debug (CMIP6 sequential): completed {run_id}")
                        sys.stdout.flush()
                        first_success_announced = True

                    del yhval, predictors
                    gc.collect()
                    if isinstance(device_str, str) and device_str.startswith("cuda") and torch.cuda.is_available():
                        try:
                            idx = int(device_str.split(":", 1)[1]) if ":" in device_str else torch.cuda.current_device()
                            torch.cuda.synchronize(idx)
                            torch.cuda.empty_cache()
                            gc.collect()
                        except Exception:
                            pass

                    successes_local += 1

                return successes_local, failures_local

            seq_success, seq_fail = _process_option_a_seq(work_items, "Option A runs", amp_flag)
            if amp_retry_queue_seq:
                amp_items = list(amp_retry_queue_seq.values())
                print(f"🔁 Retrying {len(amp_items)} Option A item(s) with AMP=1")
                add_succ, add_fail = _process_option_a_seq(amp_items, "Option A AMP retry", True)
                seq_success += add_succ
                seq_fail += add_fail

        print(f"✅ Finished Option A; segments stored in {os.path.join(root_dir, store_name)}")
        return 0

    # -------------------------------- Option B: MAGICC
    elif args.source_x.upper() == "MAGICC":
        # Find appropriate parquet file
        parquet_path = args.scm_parquet or SCM_RESULTS_PARQUET
        if parquet_path and os.path.exists(parquet_path):
            # Use explicit file if provided
            print(f"📁 Using explicit MAGICC parquet file: {parquet_path}")
        else:
            # Auto-select from directory
            parquet_path = _find_magicc_parquet_file(
                directory=SCM_RESULTS_DIR,
                requested_scenarios=EXPERIMENT_ID_WHITELIST,
                explicit_file=parquet_path,
            )

        # Load as *wide* MultiIndex table (rows indexed, columns are years)
        df_wide = pd.read_parquet(parquet_path)
        print(f"🟩 Loaded MAGICC parquet: {parquet_path}")
        print(f"   rows={len(df_wide):,}, year columns≈{sum(str(c).isdigit() or isinstance(c,(int,np.integer)) for c in df_wide.columns)}")

        # Choose scenarios
        all_scenarios_raw = sorted(pd.unique(df_wide.index.get_level_values("scenario")))
        # Normalize scenarios: strip 'clean_' prefix for matching
        scenario_normalized_to_original = {}
        for orig_scen in all_scenarios_raw:
            normalized = orig_scen.replace("clean_", "", 1) if orig_scen.startswith("clean_") else orig_scen
            scenario_normalized_to_original[normalized] = orig_scen
        
        if EXPERIMENT_ID_WHITELIST:
            # Match requested scenarios (normalized) to actual scenarios in file
            scen_whitelist = []
            for req_scen in EXPERIMENT_ID_WHITELIST:
                if req_scen in scenario_normalized_to_original:
                    scen_whitelist.append(scenario_normalized_to_original[req_scen])
            if not scen_whitelist:
                raise RuntimeError(f"None of EXPERIMENT_ID_WHITELIST {EXPERIMENT_ID_WHITELIST} are in MAGICC parquet 'scenario' index (after normalizing 'clean_' prefix). Available scenarios: {list(scenario_normalized_to_original.keys())}")
        else:
            # Default: exclude explicit runmodus flavors from primary list
            scen_whitelist = [s for s in all_scenarios_raw if "runmodus" not in s.lower()]

        # Available MAGICC run_ids (0..599 typical)
        ens_ids_all = sorted(pd.unique(df_wide.index.get_level_values("run_id")).tolist())
        if not ens_ids_all:
            raise RuntimeError("No run_id values found in MAGICC dataframe.")
        n_draws = args.ensembles if args.ensembles is not None else ENSEMBLES_B
        n_draws = max(1, min(600, int(n_draws)))
        bias_to_era5 = BIASCORRECT_TO_ERA5_B if args.bias_to_era5 is None else bool(args.bias_to_era5)

        # Choose MAGICC ensemble members (correlated across variables)
        if len(ens_ids_all) < n_draws:
            print(f"⚠️ MAGICC has only {len(ens_ids_all)} run_ids; sampling with replacement to reach {n_draws}.")
            magicc_draws = random.choices(ens_ids_all, k=n_draws)
        else:
            magicc_draws = random.sample(ens_ids_all, k=n_draws)

        # Prepare effect-model indices per draw from scheme
        effect_per_draw = _resolve_effect_model_indices(EFFECT_MODEL_SCHEME, model_to_index, n_draws)

        # NEW: compute merged requested year windows once
        req_windows = _collect_requested_year_windows(OUTPUTDICTLIST)
        if not req_windows:
            print("⚠️ No periods found in OUTPUTDICTLIST; defaulting to full MAGICC year span.")
        print(f"🟩 Option B / MAGICC: scenarios={len(scen_whitelist)}, draws={n_draws}, bias_to_era5={bias_to_era5}")

        # Generate all work items
        work_items: List[WorkItemB] = []
        for scenario in scen_whitelist:
            for i_draw in range(n_draws):
                magicc_member = int(magicc_draws[i_draw])
                effect_idx = int(effect_per_draw[i_draw])

                # Run per requested window (or one full-span run if no windows provided)
                if req_windows:
                    run_windows = req_windows
                else:
                    years_all, _ = _extract_year_columns_from_wide(df_wide.columns)
                    run_windows = [(years_all[0], years_all[-1])]
                
                for (w_sy, w_ey) in run_windows:
                    # Seed per (draw × window)
                    seed_now = (args.seed or 0) + i_draw + hash((scenario, magicc_member, MODEL_NUMBER, w_sy, w_ey)) % 10_000
                    work_items.append(WorkItemB(
                        scenario=scenario,
                        draw_idx=i_draw,
                        window=(w_sy, w_ey),
                        magicc_member=magicc_member,
                        effect_idx=effect_idx,
                        seed_base=seed_now,
                        bias_to_era5=bias_to_era5,
                    ))

        if not work_items:
            print("No work items to process. Exiting.")
            return 0

        print(f"📋 Generated {len(work_items)} work items")

        # Process work items
        if use_multiprocessing and len(gpu_indices) > 1:
            amp_retry_queue: Dict[Tuple[object, ...], WorkItemB] = {}
            successes, failures = _run_magicc_multigpu_with_retries(
                work_items, gpu_indices,
                df_wide=df_wide,
                model_to_index=model_to_index,
                variables_2predict=variables_2predict,
                lats=lats, lons=lons,
                output_specs=OUTPUTDICTLIST,
                root_dir=root_dir, store_name=store_name,
                force_gpu_flag=force_gpu_flag,
                amp_flag=amp_flag,
                model_version=MODEL_VERSION, model_number=MODEL_NUMBER, dateofmetafile=DATEOFMETAFILE,
                mode_name="MAGICC",
                effect_model_scheme=EFFECT_MODEL_SCHEME,
                effect_model_scheme_c=None,
                debug_first_success=debug_first_success,
                debug_trace_emulation=debug_trace_emulation,
                max_retries=MAX_GPU_RETRIES,
                fail_fast=False,
                items_per_gpu_process=GPU_TASKS_PER_CHILD,
                amp_retry_queue=amp_retry_queue,
            )
            if amp_retry_queue:
                fallback_items = list(amp_retry_queue.values())
                print(f"🔁 Queued {len(fallback_items)} items for AMP fallback; retrying with AMP=1")
                amp_success, amp_fail = _run_magicc_multigpu_with_retries(
                    fallback_items, gpu_indices,
                    df_wide=df_wide,
                    model_to_index=model_to_index,
                    variables_2predict=variables_2predict,
                    lats=lats, lons=lons,
                    output_specs=OUTPUTDICTLIST,
                    root_dir=root_dir, store_name=store_name,
                    force_gpu_flag=force_gpu_flag,
                    amp_flag=True,
                    model_version=MODEL_VERSION, model_number=MODEL_NUMBER, dateofmetafile=DATEOFMETAFILE,
                    mode_name="MAGICC",
                    effect_model_scheme=EFFECT_MODEL_SCHEME,
                    effect_model_scheme_c=None,
                    debug_first_success=False,
                    debug_trace_emulation=debug_trace_emulation,
                    max_retries=max(1, MAX_GPU_RETRIES),
                    fail_fast=False,
                    items_per_gpu_process=GPU_TASKS_PER_CHILD,
                )
                successes += amp_success
                failures += amp_fail
            print(f"✅ Completed: {successes} successful, {failures} failed (max GPU retries={MAX_GPU_RETRIES})")
        else:
            # Single-process execution (single GPU or CPU)
            device_str = device_for_run
            first_success_announced = False
            amp_retry_queue_seq: Dict[Tuple[object, ...], WorkItemB] = {}
            use_cpu_pool = device_str == "cpu" and cpu_pool_workers > 1

            def _process_option_b_seq(items: List[WorkItemB], desc: str, amp_default: bool) -> Tuple[int, int]:
                nonlocal first_success_announced

                if use_cpu_pool:
                    tasks = [
                        (
                            item,
                            amp_default,
                            force_gpu_flag,
                            df_wide,
                            model_to_index,
                            variables_2predict,
                            lats,
                            lons,
                            OUTPUTDICTLIST,
                            root_dir,
                            store_name,
                            MODEL_VERSION,
                            MODEL_NUMBER,
                            DATEOFMETAFILE,
                            "MAGICC",
                            EFFECT_MODEL_SCHEME,
                            None,
                            debug_trace_emulation,
                        )
                        for item in items
                    ]
                    successes_local, failures_local, retry_items = _run_cpu_pool(
                        tasks,
                        _worker_process_cpu_b,
                        workers=cpu_pool_workers,
                        desc=desc,
                        debug_first_success=debug_first_success,
                    )
                    for ri in retry_items:
                        amp_retry_queue_seq[_work_item_key(ri)] = ri
                    if debug_first_success and successes_local > 0:
                        first_success_announced = True
                    return successes_local, failures_local

                successes_local = 0
                failures_local = 0
                for item in _progress_iterable(items, total=len(items), desc=desc):
                    amp_for_item = item.amp_override if item.amp_override is not None else amp_default
                    # Set RNG seed
                    random.seed(item.seed_base)
                    np.random.seed(item.seed_base)
                    torch.manual_seed(item.seed_base)
                    
                    # Build predictors
                    try:
                        predictors = build_predictors_from_magicc(
                            df_wide, item.scenario, item.magicc_member, model_to_index,
                            year_start=int(item.window[0]), year_end=int(item.window[1])
                        )
                    except Exception as e:
                        _log_emulation_exception(
                            "MAGICC predictor build failure (sequential)",
                            e,
                            debug_trace=debug_trace_emulation,
                            context={
                                "scenario": item.scenario,
                                "window": item.window,
                                "magicc_member": item.magicc_member,
                            },
                        )
                        msg = f"Predictor build failed [scenario={item.scenario}, run_id={item.magicc_member}, window {item.window}]: {e}"
                        print(f"❌ {msg}")
                        if not amp_for_item:
                            amp_retry_queue_seq[_work_item_key(item)] = replace(item, amp_override=True)
                        failures_local += 1
                        continue

                    # Fill 'model_index' to chosen effect_idx
                    if "model_index" in predictors.predictor_names:
                        mi_col = predictors.predictor_names.index("model_index")
                        predictors.X[:, mi_col] = float(item.effect_idx)

                    usebias_model = 0 if item.bias_to_era5 else None
                    useeffect_model = item.effect_idx

                    # Run emulation
                    try:
                        yhval = run_gcmagicc(
                            predictors,
                            dependence=True,
                            usebias_model=usebias_model,
                            useeffect_model=useeffect_model,
                            device=device_str,
                            force_gpu=force_gpu_flag,
                            amp=amp_for_item,
                        )
                    except Exception as e:
                        _log_emulation_exception(
                            "MAGICC run_gcmagicc failure (sequential)",
                            e,
                            debug_trace=debug_trace_emulation,
                            context={
                                "scenario": item.scenario,
                                "window": item.window,
                                "magicc_member": item.magicc_member,
                                "effect_idx": useeffect_model,
                                "device": device_str,
                                "amp": amp_for_item,
                            },
                        )
                        msg = f"Emulation failed [scenario={item.scenario}, run_id={item.magicc_member}, eff={useeffect_model}, window {item.window}]: {e}"
                        print(f"❌ {msg}")
                        if not amp_for_item:
                            amp_retry_queue_seq[_work_item_key(item)] = replace(item, amp_override=True)
                        failures_local += 1
                        continue

                    # Generate run metadata
                    run_id = (
                        f"B_{item.scenario}__MAGICCrun{item.magicc_member:03d}"
                        f"__b{0 if item.bias_to_era5 else 'N'}e{useeffect_model}"
                        f"__m{item.draw_idx:04d}__win{int(item.window[0])}-{int(item.window[1])}__{_today_stamp()}__{uuid.uuid4().hex[:8]}"
                    )
                    run_meta = {
                        "run_id": run_id,
                        "mode": "MAGICC",
                        "scenario": item.scenario,
                        "magicc_run_id": item.magicc_member,
                        "usebias_model": 0 if item.bias_to_era5 else "None",
                        "useeffect_model": useeffect_model,
                        "model_version": MODEL_VERSION,
                        "model_number": MODEL_NUMBER,
                        "model_id": f"{MODEL_VERSION}_{MODEL_NUMBER}",
                        "date_meta": DATEOFMETAFILE,
                        "device": device_str,
                        "effect_model_scheme": json.dumps(EFFECT_MODEL_SCHEME) if isinstance(EFFECT_MODEL_SCHEME, (list, dict)) else str(EFFECT_MODEL_SCHEME),
                        "amp": bool(amp_for_item),
                    }

                    apply_output_specs_and_write(
                        yhval=yhval,
                        predictors=predictors,
                        lats=lats, lons=lons,
                        output_specs=OUTPUTDICTLIST,
                        run_meta=run_meta,
                        variables_2predict=variables_2predict,
                        root_dir=root_dir, store_name=store_name,
                    )

                    if debug_first_success and not first_success_announced:
                        print(f"🔍 First-success debug (MAGICC sequential): completed {run_id}")
                        sys.stdout.flush()
                        first_success_announced = True

                    del yhval, predictors
                    gc.collect()
                    if isinstance(device_str, str) and device_str.startswith("cuda") and torch.cuda.is_available():
                        try:
                            idx = int(device_str.split(":", 1)[1]) if ":" in device_str else torch.cuda.current_device()
                            torch.cuda.synchronize(idx)
                            torch.cuda.empty_cache()
                            gc.collect()
                        except Exception:
                            pass

                    successes_local += 1

                return successes_local, failures_local

            seq_success, seq_fail = _process_option_b_seq(work_items, "Option B runs", amp_flag)
            if amp_retry_queue_seq:
                amp_items = list(amp_retry_queue_seq.values())
                print(f"🔁 Retrying {len(amp_items)} Option B item(s) with AMP=1")
                add_succ, add_fail = _process_option_b_seq(amp_items, "Option B AMP retry", True)
                seq_success += add_succ
                seq_fail += add_fail

        print(f"✅ Finished Option B; segments stored in {os.path.join(root_dir, store_name)}")
        return 0

    # -------------------------------- Option D: MAGICCxERA5 (spliced predictors)
    elif args.source_x.upper() == "MAGICCXERA5":
        # Configuration specific to ERA5‑spliced MAGICC predictors
        bias_to_era5 = BIASCORRECT_TO_ERA5_D if args.bias_to_era5 is None else bool(args.bias_to_era5)
        n_draws = args.ensembles if args.ensembles is not None else ENSEMBLES_D
        n_draws = max(1, int(n_draws))
        
        # Get workflow and runmodus (CLI override > config > env)
        workflow = args.workflow if args.workflow is not None else USE_WORKFLOW
        runmodus = args.runmodus if args.runmodus is not None else USE_RUNMODUS

        # Discover ERA5‑spliced predictor files written by the 616_* pipeline
        scen_map = _discover_spliced_predictor_files(
            ERA5_SPLICED_PREDICTOR_DIR,
            EXPERIMENT_ID_WHITELIST,
            workflow=workflow,
            runmodus=runmodus,
        )

        # Scenario whitelist: normalise any "clean_" prefixes in EXPERIMENT_ID_WHITELIST
        if EXPERIMENT_ID_WHITELIST:
            scen_whitelist: List[str] = []
            missing_scen: List[str] = []
            for s in EXPERIMENT_ID_WHITELIST:
                s_norm = s.replace("clean_", "", 1) if s.startswith("clean_") else s
                if s_norm in scen_map:
                    scen_whitelist.append(s_norm)
                else:
                    missing_scen.append(s)
            if missing_scen:
                raise RuntimeError(
                    f"No ERA5-spliced predictors found for scenarios {missing_scen} in {ERA5_SPLICED_PREDICTOR_DIR}. "
                    "Run notebooks/616_*.py in the gcmmagicc repo with the appropriate settings."
                )
        else:
            scen_whitelist = list(scen_map.keys())

        # Requested windows from OUTPUTDICTLIST; fall back to full predictor span
        req_windows = _collect_requested_year_windows(OUTPUTDICTLIST)
        if not req_windows:
            # Derive a full span from a representative predictor file
            sample_file = next(iter(next(iter(scen_map.values())).values()))
            sy, ey = _peek_spliced_year_span(sample_file)
            req_windows = [(sy, ey)]
            print("⚠️ No periods found in OUTPUTDICTLIST; defaulting to full ERA5-spliced predictor span.")

        # ERA5 is fixed effect model index 0
        era5_effect_idx = 0
        print(
            f"🟦 Option D / MAGICCxERA5: scenarios={len(scen_whitelist)}, "
            f"draws={n_draws}, windows={len(req_windows)}, bias_to_era5={bias_to_era5}"
        )

        # Build work items: one per scenario × SCM run × ensemble draw × window
        work_items: List[WorkItemD] = []
        for scenario in scen_whitelist:
            run_map = scen_map[scenario]
            run_ids_all = sorted(run_map.keys())
            if not run_ids_all:
                raise RuntimeError(
                    f"No runs found for scenario '{scenario}' in {ERA5_SPLICED_PREDICTOR_DIR}. "
                    "Run notebooks/616_*.py in the gcmmagicc repo with the appropriate settings."
                )
            if len(run_ids_all) < n_draws:
                print(
                    f"⚠️ Scenario {scenario} has only {len(run_ids_all)} runs; "
                    f"sampling with replacement to reach {n_draws}."
                )
                draws = random.choices(run_ids_all, k=n_draws)
            else:
                draws = random.sample(run_ids_all, k=n_draws)

            for i_draw, run_id in enumerate(draws):
                predictor_path = str(run_map[run_id])
                for (w_sy, w_ey) in req_windows:
                    seed_now = (
                        (args.seed or 0)
                        + i_draw
                        + hash((scenario, run_id, MODEL_NUMBER, w_sy, w_ey)) % 10_000
                    )
                    work_items.append(
                        WorkItemD(
                            scenario=scenario,
                            draw_idx=i_draw,
                            window=(int(w_sy), int(w_ey)),
                            run_id=int(run_id),
                            predictor_path=predictor_path,
                            seed_base=int(seed_now),
                            bias_to_era5=bias_to_era5,
                        )
                    )

        if not work_items:
            print("No work items to process. Exiting.")
            return 0

        print(f"📋 Generated {len(work_items)} work items")

        # ------------------------------------------------------------------
        # Dispatch: multi‑GPU path (shared worker/_run_* helpers) vs sequential
        # ------------------------------------------------------------------
        if use_multiprocessing and len(gpu_indices) > 1:
            # Multi-GPU execution mirrors Option B/C, including AMP retry
            amp_retry_queue: Dict[Tuple[object, ...], WorkItemD] = {}
            successes, failures = _run_magiccxera5_multigpu_with_retries(
                work_items,
                gpu_indices,
                model_to_index=model_to_index,
                variables_2predict=variables_2predict,
                lats=lats,
                lons=lons,
                output_specs=OUTPUTDICTLIST,
                root_dir=root_dir,
                store_name=store_name,
                force_gpu_flag=force_gpu_flag,
                amp_flag=amp_flag,
                model_version=MODEL_VERSION,
                model_number=MODEL_NUMBER,
                dateofmetafile=DATEOFMETAFILE,
                debug_first_success=debug_first_success,
                debug_trace_emulation=debug_trace_emulation,
                max_retries=MAX_GPU_RETRIES,
                fail_fast=False,
                items_per_gpu_process=GPU_TASKS_PER_CHILD,
                amp_retry_queue=amp_retry_queue,
            )
            if amp_retry_queue:
                fallback_items = list(amp_retry_queue.values())
                print(
                    f"🔁 Queued {len(fallback_items)} MAGICCxERA5 item(s) for AMP fallback; retrying with AMP=1"
                )
                amp_success, amp_fail = _run_magiccxera5_multigpu_with_retries(
                    fallback_items,
                    gpu_indices,
                    model_to_index=model_to_index,
                    variables_2predict=variables_2predict,
                    lats=lats,
                    lons=lons,
                    output_specs=OUTPUTDICTLIST,
                    root_dir=root_dir,
                    store_name=store_name,
                    force_gpu_flag=force_gpu_flag,
                    amp_flag=True,
                    model_version=MODEL_VERSION,
                    model_number=MODEL_NUMBER,
                    dateofmetafile=DATEOFMETAFILE,
                    debug_first_success=False,
                    debug_trace_emulation=debug_trace_emulation,
                    max_retries=max(1, MAX_GPU_RETRIES),
                    fail_fast=False,
                    items_per_gpu_process=GPU_TASKS_PER_CHILD,
                )
                successes += amp_success
                failures += amp_fail
            print(
                f"✅ Completed: {successes} successful, {failures} failed "
                f"(max GPU retries={MAX_GPU_RETRIES})"
            )
        else:
            # Single-process execution (single GPU or CPU), with optional AMP retry
            device_str = device_for_run
            first_success_announced = False
            amp_retry_queue_seq: Dict[Tuple[object, ...], WorkItemD] = {}
            use_cpu_pool = device_str == "cpu" and cpu_pool_workers > 1

            def _process_option_d_seq(
                items: List[WorkItemD],
                desc: str,
                amp_default: bool,
            ) -> Tuple[int, int]:
                nonlocal first_success_announced

                if use_cpu_pool:
                    tasks = [
                        (
                            item,
                            amp_default,
                            force_gpu_flag,
                            model_to_index,
                            variables_2predict,
                            lats,
                            lons,
                            OUTPUTDICTLIST,
                            root_dir,
                            store_name,
                            MODEL_VERSION,
                            MODEL_NUMBER,
                            DATEOFMETAFILE,
                            era5_effect_idx,
                            debug_trace_emulation,
                        )
                        for item in items
                    ]
                    successes_local, failures_local, retry_items = _run_cpu_pool(
                        tasks,
                        _worker_process_cpu_d,
                        workers=cpu_pool_workers,
                        desc=desc,
                        debug_first_success=debug_first_success,
                    )
                    for ri in retry_items:
                        amp_retry_queue_seq[_work_item_key(ri)] = ri
                    if debug_first_success and successes_local > 0:
                        first_success_announced = True
                    return successes_local, failures_local

                successes_local = 0
                failures_local = 0

                for item in _progress_iterable(items, total=len(items), desc=desc):
                    amp_for_item = (
                        item.amp_override
                        if item.amp_override is not None
                        else amp_default
                    )

                    # Seed per item for reproducibility
                    random.seed(item.seed_base)
                    np.random.seed(item.seed_base)
                    torch.manual_seed(item.seed_base)

                    # Build predictors from ERA5‑spliced file
                    try:
                        predictors = build_predictors_from_spliced_file(
                            item.predictor_path,
                            model_to_index,
                            model_index_name="ERA5",
                            year_start=int(item.window[0]),
                            year_end=int(item.window[1]),
                        )
                    except Exception as e:
                        _log_emulation_exception(
                            "MAGICCxERA5 predictor build failure (sequential)",
                            e,
                            debug_trace=debug_trace_emulation,
                            context={
                                "scenario": item.scenario,
                                "run_id": item.run_id,
                                "window": item.window,
                            },
                        )
                        msg = (
                            f"Predictor build failed "
                            f"[scenario={item.scenario}, run_id={item.run_id}, window {item.window}]: {e}"
                        )
                        print(f"❌ {msg}")
                        if not amp_for_item:
                            amp_retry_queue_seq[_work_item_key(item)] = replace(
                                item, amp_override=True
                            )
                        failures_local += 1
                        continue

                    # Force model_index column to ERA5 (index 0)
                    if "model_index" in predictors.predictor_names:
                        mi_col = predictors.predictor_names.index("model_index")
                        predictors.X[:, mi_col] = float(era5_effect_idx)

                    usebias_model = 0 if item.bias_to_era5 else None
                    useeffect_model = era5_effect_idx

                    # Run emulation using the shared run_gcmagicc wrapper
                    try:
                        yhval = run_gcmagicc(
                            predictors,
                            dependence=True,
                            usebias_model=usebias_model,
                            useeffect_model=useeffect_model,
                            device=device_str,
                            force_gpu=force_gpu_flag,
                            amp=amp_for_item,
                        )
                    except Exception as e:
                        _log_emulation_exception(
                            "MAGICCxERA5 run_gcmagicc failure (sequential)",
                            e,
                            debug_trace=debug_trace_emulation,
                            context={
                                "scenario": item.scenario,
                                "window": item.window,
                                "run_id": item.run_id,
                                "device": device_str,
                                "amp": amp_for_item,
                            },
                        )
                        msg = (
                            f"Emulation failed "
                            f"[scenario={item.scenario}, run_id={item.run_id}, window {item.window}]: {e}"
                        )
                        print(f"❌ {msg}")
                        if not amp_for_item:
                            amp_retry_queue_seq[_work_item_key(item)] = replace(
                                item, amp_override=True
                            )
                        failures_local += 1
                        continue

                    run_id = (
                        f"D_{item.scenario}__ERA5splicedrun{item.run_id:03d}"
                        f"__b{0 if item.bias_to_era5 else 'N'}e{era5_effect_idx}"
                        f"__m{item.draw_idx:04d}__win{int(item.window[0])}-{int(item.window[1])}"
                        f"__{_today_stamp()}__{uuid.uuid4().hex[:8]}"
                    )
                    run_meta = {
                        "run_id": run_id,
                        "mode": "MAGICCxERA5",
                        "scenario": item.scenario,
                        "magicc_run_id": item.run_id,
                        "usebias_model": 0 if item.bias_to_era5 else "None",
                        "useeffect_model": era5_effect_idx,
                        "model_version": MODEL_VERSION,
                        "model_number": MODEL_NUMBER,
                        "model_id": f"{MODEL_VERSION}_{MODEL_NUMBER}",
                        "date_meta": DATEOFMETAFILE,
                        "device": device_str,
                        "amp": bool(amp_for_item),
                        "predictor_path": item.predictor_path,
                    }

                    apply_output_specs_and_write(
                        yhval=yhval,
                        predictors=predictors,
                        lats=lats,
                        lons=lons,
                        output_specs=OUTPUTDICTLIST,
                        run_meta=run_meta,
                        variables_2predict=variables_2predict,
                        root_dir=root_dir,
                        store_name=store_name,
                    )

                    if debug_first_success and not first_success_announced:
                        print(
                            f"🔍 First-success debug (MAGICCxERA5 sequential): completed {run_id}"
                        )
                        sys.stdout.flush()
                        first_success_announced = True

                    # Clean up between runs to reduce fragmentation
                    del yhval, predictors
                    gc.collect()
                    if (
                        isinstance(device_str, str)
                        and device_str.startswith("cuda")
                        and torch.cuda.is_available()
                    ):
                        try:
                            idx = (
                                int(device_str.split(":", 1)[1])
                                if ":" in device_str
                                else torch.cuda.current_device()
                            )
                            torch.cuda.synchronize(idx)
                            torch.cuda.empty_cache()
                            gc.collect()
                        except Exception:
                            pass

                    successes_local += 1

                return successes_local, failures_local

            # First pass with the configured AMP flag
            seq_success, seq_fail = _process_option_d_seq(
                work_items,
                "Option D runs",
                amp_flag,
            )

            # AMP‑enabled retry for failures (if any)
            if amp_retry_queue_seq:
                amp_items = list(amp_retry_queue_seq.values())
                print(
                    f"🔁 Retrying {len(amp_items)} Option D item(s) with AMP=1"
                )
                add_succ, add_fail = _process_option_d_seq(
                    amp_items,
                    "Option D AMP retry",
                    True,
                )
                seq_success += add_succ
                seq_fail += add_fail

        print(
            f"✅ Finished Option D; segments stored in {os.path.join(root_dir, store_name)}"
        )
        return 0

    # -------------------------------- Option C: MAGICC-SAMEPERCMIP6
    elif args.source_x.upper() == "MAGICC-SAMEPERCMIP6":
        # Find appropriate parquet file
        parquet_path = args.scm_parquet or SCM_RESULTS_PARQUET
        if parquet_path and os.path.exists(parquet_path):
            # Use explicit file if provided
            print(f"📁 Using explicit MAGICC parquet file: {parquet_path}")
        else:
            # Auto-select from directory
            parquet_path = _find_magicc_parquet_file(
                directory=SCM_RESULTS_DIR,
                requested_scenarios=EXPERIMENT_ID_WHITELIST,
                explicit_file=parquet_path,
            )

        # Load as *wide* MultiIndex table (rows indexed, columns are years)
        df_wide = pd.read_parquet(parquet_path)
        print(f"🟧 Loaded MAGICC parquet: {parquet_path}")
        print(f"   rows={len(df_wide):,}, year columns≈{sum(str(c).isdigit() or isinstance(c,(int,np.integer)) for c in df_wide.columns)}")

        # Choose scenarios
        all_scenarios_raw = sorted(pd.unique(df_wide.index.get_level_values("scenario")))
        # Normalize scenarios: strip 'clean_' prefix for matching
        scenario_normalized_to_original = {}
        for orig_scen in all_scenarios_raw:
            normalized = orig_scen.replace("clean_", "", 1) if orig_scen.startswith("clean_") else orig_scen
            scenario_normalized_to_original[normalized] = orig_scen
        
        if EXPERIMENT_ID_WHITELIST:
            # Match requested scenarios (normalized) to actual scenarios in file
            scen_whitelist = []
            for req_scen in EXPERIMENT_ID_WHITELIST:
                if req_scen in scenario_normalized_to_original:
                    scen_whitelist.append(scenario_normalized_to_original[req_scen])
            if not scen_whitelist:
                raise RuntimeError(f"None of EXPERIMENT_ID_WHITELIST {EXPERIMENT_ID_WHITELIST} are in MAGICC parquet 'scenario' index (after normalizing 'clean_' prefix). Available scenarios: {list(scenario_normalized_to_original.keys())}")
        else:
            scen_whitelist = [s for s in all_scenarios_raw if "runmodus" not in s.lower()]

        # Available MAGICC run_ids (0..599 typical)
        ens_ids_all = sorted(pd.unique(df_wide.index.get_level_values("run_id")).tolist())
        if not ens_ids_all:
            raise RuntimeError("No run_id values found in MAGICC dataframe.")
        n_draws = args.ensembles if args.ensembles is not None else ENSEMBLES_C_PER_CMIP6
        n_draws = max(1, min(600, int(n_draws)))
        bias_to_era5 = BIASCORRECT_TO_ERA5_C if args.bias_to_era5 is None else bool(args.bias_to_era5)

        # Select the SAME subset of MAGICC ensemble members to be paired with EACH CMIP6 calibration
        if len(ens_ids_all) < n_draws:
            print(f"⚠️ MAGICC has only {len(ens_ids_all)} run_ids; sampling with replacement to reach {n_draws}.")
            magicc_draws = random.choices(ens_ids_all, k=n_draws)
        else:
            magicc_draws = random.sample(ens_ids_all, k=n_draws)

        # Determine the set of CMIP6 calibration indices based on EFFECT_MODEL_SCHEME_C
        try:
            indices_all = _resolve_cmip6_calibration_indices(EFFECT_MODEL_SCHEME_C, model_to_index)
        except Exception as e:
            raise RuntimeError(f"Failed to resolve CMIP6 calibration indices from EFFECT_MODEL_SCHEME_C: {e}")
        n_cmip6 = len(indices_all)
        if n_cmip6 == 0:
            raise RuntimeError("No CMIP6 calibration indices resolved from EFFECT_MODEL_SCHEME_C.")

        # NEW: compute merged requested year windows once
        req_windows = _collect_requested_year_windows(OUTPUTDICTLIST)
        if not req_windows:
            print("⚠️ No periods found in OUTPUTDICTLIST; defaulting to full MAGICC year span.")
            years_all, _ = _extract_year_columns_from_wide(df_wide.columns)
            req_windows = [(int(years_all[0]), int(years_all[-1]))]

        total_runs = len(scen_whitelist) * n_cmip6 * n_draws * len(req_windows)
        if isinstance(EFFECT_MODEL_SCHEME_C, str):
            scheme_lower = EFFECT_MODEL_SCHEME_C.lower()
            if scheme_lower == "all":
                scheme_str = "all"
            elif scheme_lower == "allplusera5":
                scheme_str = "allplusERA5"
            else:
                scheme_str = EFFECT_MODEL_SCHEME_C
        elif isinstance(EFFECT_MODEL_SCHEME_C, list):
            scheme_str = str(EFFECT_MODEL_SCHEME_C)
        else:
            scheme_str = str(EFFECT_MODEL_SCHEME_C)
        print(f"🟧 Option C / MAGICC-SAMEPERCMIP6: scenarios={len(scen_whitelist)}, "
              f"per-CMIP6 parameter sets={n_draws}, CMIP6 calibrations={n_cmip6} (scheme={scheme_str}) "
              f"→ total draws={n_draws*n_cmip6}; windows={len(req_windows)}; bias_to_era5={bias_to_era5} "
              f"(approx total work items: {total_runs})")

        # Generate all work items: same magicc_draws for each effect_idx
        work_items: List[WorkItemB] = []
        for scenario in scen_whitelist:
            for eff_pos, effect_idx in enumerate(indices_all):
                for i_draw, magicc_member in enumerate(magicc_draws):
                    draw_idx_flat = eff_pos * n_draws + i_draw  # unique within scenario
                    for (w_sy, w_ey) in req_windows:
                        seed_now = ((args.seed or 0)
                                    + draw_idx_flat
                                    + (hash((scenario, magicc_member, "C", effect_idx, MODEL_NUMBER, w_sy, w_ey)) % 10_000))
                        work_items.append(WorkItemB(
                            scenario=scenario,
                            draw_idx=draw_idx_flat,
                            window=(int(w_sy), int(w_ey)),
                            magicc_member=int(magicc_member),
                            effect_idx=int(effect_idx),
                            seed_base=int(seed_now),
                            bias_to_era5=bias_to_era5,
                        ))

        if not work_items:
            print("No work items to process. Exiting.")
            return 0

        print(f"📋 Generated {len(work_items)} work items")

        # Process work items
        if use_multiprocessing and len(gpu_indices) > 1:
            amp_retry_queue: Dict[Tuple[object, ...], WorkItemB] = {}
            successes, failures = _run_magicc_multigpu_with_retries(
                work_items, gpu_indices,
                df_wide=df_wide,
                model_to_index=model_to_index,
                variables_2predict=variables_2predict,
                lats=lats, lons=lons,
                output_specs=OUTPUTDICTLIST,
                root_dir=root_dir, store_name=store_name,
                force_gpu_flag=force_gpu_flag,
                amp_flag=amp_flag,
                model_version=MODEL_VERSION, model_number=MODEL_NUMBER, dateofmetafile=DATEOFMETAFILE,
                mode_name="MAGICC-SAMEPERCMIP6",
                effect_model_scheme=f"SamePerCMIP6(n={n_draws})",
                effect_model_scheme_c=EFFECT_MODEL_SCHEME_C,
                debug_first_success=debug_first_success,
                debug_trace_emulation=debug_trace_emulation,
                max_retries=MAX_GPU_RETRIES,
                fail_fast=False,
                items_per_gpu_process=GPU_TASKS_PER_CHILD,
                amp_retry_queue=amp_retry_queue,
            )
            if amp_retry_queue:
                fallback_items = list(amp_retry_queue.values())
                print(f"🔁 Queued {len(fallback_items)} items for AMP fallback; retrying with AMP=1")
                amp_success, amp_fail = _run_magicc_multigpu_with_retries(
                    fallback_items, gpu_indices,
                    df_wide=df_wide,
                    model_to_index=model_to_index,
                    variables_2predict=variables_2predict,
                    lats=lats, lons=lons,
                    output_specs=OUTPUTDICTLIST,
                    root_dir=root_dir, store_name=store_name,
                    force_gpu_flag=force_gpu_flag,
                    amp_flag=True,
                    model_version=MODEL_VERSION, model_number=MODEL_NUMBER, dateofmetafile=DATEOFMETAFILE,
                    mode_name="MAGICC-SAMEPERCMIP6",
                    effect_model_scheme=f"SamePerCMIP6(n={n_draws})",
                    effect_model_scheme_c=EFFECT_MODEL_SCHEME_C,
                    debug_first_success=False,
                    debug_trace_emulation=debug_trace_emulation,
                    max_retries=max(1, MAX_GPU_RETRIES),
                    fail_fast=False,
                    items_per_gpu_process=GPU_TASKS_PER_CHILD,
                )
                successes += amp_success
                failures += amp_fail
            print(f"✅ Completed: {successes} successful, {failures} failed (max GPU retries={MAX_GPU_RETRIES})")
        else:
            # Single-process execution (single GPU or CPU)
            device_str = device_for_run
            first_success_announced = False
            amp_retry_queue_seq: Dict[Tuple[object, ...], WorkItemB] = {}
            use_cpu_pool = device_str == "cpu" and cpu_pool_workers > 1

            def _process_option_c_seq(items: List[WorkItemB], desc: str, amp_default: bool) -> Tuple[int, int]:
                nonlocal first_success_announced

                if use_cpu_pool:
                    tasks = [
                        (
                            item,
                            amp_default,
                            force_gpu_flag,
                            df_wide,
                            model_to_index,
                            variables_2predict,
                            lats,
                            lons,
                            OUTPUTDICTLIST,
                            root_dir,
                            store_name,
                            MODEL_VERSION,
                            MODEL_NUMBER,
                            DATEOFMETAFILE,
                            "MAGICC-SAMEPERCMIP6",
                            f"SamePerCMIP6(n={n_draws})",
                            EFFECT_MODEL_SCHEME_C,
                            debug_trace_emulation,
                        )
                        for item in items
                    ]
                    successes_local, failures_local, retry_items = _run_cpu_pool(
                        tasks,
                        _worker_process_cpu_b,
                        workers=cpu_pool_workers,
                        desc=desc,
                        debug_first_success=debug_first_success,
                    )
                    for ri in retry_items:
                        amp_retry_queue_seq[_work_item_key(ri)] = ri
                    if debug_first_success and successes_local > 0:
                        first_success_announced = True
                    return successes_local, failures_local

                successes_local = 0
                failures_local = 0
                for item in _progress_iterable(items, total=len(items), desc=desc):
                    amp_for_item = item.amp_override if item.amp_override is not None else amp_default
                    # Set RNG seed
                    random.seed(item.seed_base)
                    np.random.seed(item.seed_base)
                    torch.manual_seed(item.seed_base)

                    # Build predictors
                    try:
                        predictors = build_predictors_from_magicc(
                            df_wide, item.scenario, item.magicc_member, model_to_index,
                            year_start=int(item.window[0]), year_end=int(item.window[1])
                        )
                    except Exception as e:
                        _log_emulation_exception(
                            "MAGICC-SAMEPERCMIP6 predictor build failure (sequential)",
                            e,
                            debug_trace=debug_trace_emulation,
                            context={
                                "scenario": item.scenario,
                                "window": item.window,
                                "magicc_member": item.magicc_member,
                            },
                        )
                        msg = f"Predictor build failed [scenario={item.scenario}, run_id={item.magicc_member}, window {item.window}]: {e}"
                        print(f"❌ {msg}")
                        if not amp_for_item:
                            amp_retry_queue_seq[_work_item_key(item)] = replace(item, amp_override=True)
                        failures_local += 1
                        continue

                    # Fill 'model_index' to chosen effect_idx
                    if "model_index" in predictors.predictor_names:
                        mi_col = predictors.predictor_names.index("model_index")
                        predictors.X[:, mi_col] = float(item.effect_idx)

                    usebias_model = 0 if item.bias_to_era5 else None
                    useeffect_model = item.effect_idx

                    # Run emulation
                    try:
                        yhval = run_gcmagicc(
                            predictors,
                            dependence=True,
                            usebias_model=usebias_model,
                            useeffect_model=useeffect_model,
                            device=device_str,
                            force_gpu=force_gpu_flag,
                            amp=amp_for_item,
                        )
                    except Exception as e:
                        _log_emulation_exception(
                            "MAGICC-SAMEPERCMIP6 run_gcmagicc failure (sequential)",
                            e,
                            debug_trace=debug_trace_emulation,
                            context={
                                "scenario": item.scenario,
                                "window": item.window,
                                "magicc_member": item.magicc_member,
                                "effect_idx": useeffect_model,
                                "device": device_str,
                                "amp": amp_for_item,
                            },
                        )
                        msg = f"Emulation failed [scenario={item.scenario}, run_id={item.magicc_member}, eff={useeffect_model}, window {item.window}]: {e}"
                        print(f"❌ {msg}")
                        if not amp_for_item:
                            amp_retry_queue_seq[_work_item_key(item)] = replace(item, amp_override=True)
                        failures_local += 1
                        continue

                    # Generate run metadata
                    run_id = (
                        f"C_{item.scenario}__MAGICCrun{item.magicc_member:03d}"
                        f"__b{0 if item.bias_to_era5 else 'N'}e{useeffect_model}"
                        f"__m{item.draw_idx:04d}__win{int(item.window[0])}-{int(item.window[1])}__{_today_stamp()}__{uuid.uuid4().hex[:8]}"
                    )
                    run_meta = {
                        "run_id": run_id,
                        "mode": "MAGICC-SAMEPERCMIP6",
                        "scenario": item.scenario,
                        "magicc_run_id": item.magicc_member,
                        "usebias_model": 0 if item.bias_to_era5 else "None",
                        "useeffect_model": useeffect_model,
                        "model_version": MODEL_VERSION,
                        "model_number": MODEL_NUMBER,
                        "model_id": f"{MODEL_VERSION}_{MODEL_NUMBER}",
                        "date_meta": DATEOFMETAFILE,
                        "device": device_str,
                        "effect_model_scheme": f"SamePerCMIP6(n={n_draws})",
                        "effect_model_scheme_c": json.dumps(EFFECT_MODEL_SCHEME_C) if isinstance(EFFECT_MODEL_SCHEME_C, (list, dict)) else str(EFFECT_MODEL_SCHEME_C),
                        "amp": bool(amp_for_item),
                    }

                    apply_output_specs_and_write(
                        yhval=yhval,
                        predictors=predictors,
                        lats=lats, lons=lons,
                        output_specs=OUTPUTDICTLIST,
                        run_meta=run_meta,
                        variables_2predict=variables_2predict,
                        root_dir=root_dir, store_name=store_name,
                    )

                    if debug_first_success and not first_success_announced:
                        print(f"🔍 First-success debug (MAGICC-SAMEPERCMIP6 sequential): completed {run_id}")
                        sys.stdout.flush()
                        first_success_announced = True

                    del yhval, predictors
                    gc.collect()
                    if isinstance(device_str, str) and device_str.startswith("cuda") and torch.cuda.is_available():
                        try:
                            idx = int(device_str.split(":", 1)[1]) if ":" in device_str else torch.cuda.current_device()
                            torch.cuda.synchronize(idx)
                            torch.cuda.empty_cache()
                            gc.collect()
                        except Exception:
                            pass

                    successes_local += 1

                return successes_local, failures_local

            seq_success, seq_fail = _process_option_c_seq(work_items, "Option C runs", amp_flag)
            if amp_retry_queue_seq:
                amp_items = list(amp_retry_queue_seq.values())
                print(f"🔁 Retrying {len(amp_items)} Option C item(s) with AMP=1")
                add_succ, add_fail = _process_option_c_seq(amp_items, "Option C AMP retry", True)
                seq_success += add_succ
                seq_fail += add_fail

        print(f"✅ Finished Option C; segments stored in {os.path.join(root_dir, store_name)}")
        return 0

    else:
        raise ValueError("--source-x must be 'CMIP6', 'MAGICC', 'MAGICC-SAMEPERCMIP6', or 'MAGICCxERA5'")

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        gc.collect()
