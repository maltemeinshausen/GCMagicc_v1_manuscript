#!/usr/bin/env python3
"""
331_run_probabilistic_debiasloop_ensembles_gcmagicc.py

Debias-loop driver for full-length GCMagicc ensembles based on 321_*.

The workflow mirrors 330_* (segments) but:
  • Uses the full predictor span instead of short segments.
  • Reuses helpers from 321_run_probabilistic_ensembles_gcmagicc.py.
  • Runs MAGICCxERA5 (Option D) only, adjusting tas_smoothed per ensemble
    by per-year offsets Δ(year) = tas_out_smoothed21(year) - tas_pred(year) computed
    for each year in the experiment period.

Two passes:
  PASS 1 – run emulator with original predictors to estimate per-year Δ for each
           ensemble (global area-weighted tas per year, 21-year smoothed).
  PASS 2 – subtract Δ(year)/10 from tas_smoothed for each year, rerun emulator,
           and write full NetCDF outputs (simple or CMIP6 layout).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from multiprocessing import get_context

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch  # type: ignore


# =============================================================================
# Dynamic import of the 321_* script as module "gcm321"
# =============================================================================


def _load_base321_module() -> object:
    """
    Load 321_run_probabilistic_ensembles_gcmagicc.py as a module named "gcm321".
    """
    here = Path(__file__).resolve()
    base_path = here.with_name("321_run_probabilistic_ensembles_gcmagicc.py")
    if not base_path.exists():
        raise FileNotFoundError(
            f"Cannot find base script '321_run_probabilistic_ensembles_gcmagicc.py' "
            f"next to {here}. Please place this 331_* script in the same folder."
        )

    loader = importlib.machinery.SourceFileLoader("gcm321", str(base_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError("Could not create import spec for gcm321")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


gcm321 = _load_base321_module()

# Aliases / config from 321_*
MODEL_VERSION = gcm321.MODEL_VERSION
MODELS_SUBDIR = gcm321.MODELS_SUBDIR
GCMAGICC_PATH = gcm321.GCMagiccpath
ERA5_SPLICED_PREDICTOR_DIR = gcm321.ERA5_SPLICED_PREDICTOR_DIR
SPLICED_VARIANT_GLOB = getattr(gcm321, "SPLICED_VARIANT_GLOB", "magicc_based_predictors_*")
SPLICED_VARIANT = getattr(gcm321, "SPLICED_VARIANT", "")
SPLICED_N = getattr(gcm321, "SPLICED_N", 100)
ENSEMBLES_D = gcm321.ENSEMBLES_D
BIASCORRECT_TO_ERA5_D = gcm321.BIASCORRECT_TO_ERA5_D
TEST_ONE = getattr(gcm321, "TEST_ONE", False)
N_LAT = gcm321.NLAT
LON_CONVENTION = gcm321.LON_CONVENTION
DEFAULT_DEVICE = gcm321.DEFAULT_DEVICE
DEFAULT_AMP = getattr(gcm321, "DEFAULT_AMP", False)
DEPENDENCE = gcm321.DEPENDENCE
OUTPUT_MODE_DEFAULT = gcm321.OUTPUT_MODE
NAMING_TEMPLATE = gcm321.NAMING_TEMPLATE
USE_RUNMODUSE = gcm321.USE_RUNMODUSE
USE_WORKFLOW = getattr(gcm321, "USE_WORKFLOW", "AR6")
OUTPUT_ROOT_DEFAULT = Path(getattr(gcm321, "OUTPUT_ROOT"))
PER_YEAR_SMOOTH_WINDOW = 21  # years for centered smoothing when computing per-year deltas
PASS1_TEMP_STAGE = "temporary_pass1_origbias"
DELETE_TEMPFOLDER_PASS1 = True  # Delete the temporary PASS1 folder after PASS2 completes
DEBIAS_OUTPUT_PREFIX = os.environ.get("GCMAGICC_DEBIAS_OUTPUT_PREFIX", "debiasloop_100ssp245plusnatv100")

# Output naming helpers (populated after META load)
_MODEL_INDEX_TO_NAME: Dict[int, str] = {}
_DEFAULT_VARLIST: List[str] = []

generate_coordinate_grids = gcm321.generate_coordinate_grids
run_gcmagicc = gcm321.run_gcmagicc
build_predictors_from_spliced_file = gcm321.build_predictors_from_spliced_file
discover_spliced_predictor_files = gcm321.discover_spliced_predictor_files
build_spliced_tasks = gcm321.build_spliced_tasks
discover_spliced_scenarios = gcm321.discover_spliced_scenarios
normalize_device_string = gcm321.normalize_device_string
detect_default_device = gcm321.detect_default_device
report_device_status = gcm321.report_device_status
normalize_workflow_list = gcm321.normalize_workflow_list
normalize_runmodus_list = gcm321.normalize_runmodus_list
model_version_to_code = gcm321.model_version_to_code
runmodus_to_suffix = gcm321.runmodus_to_suffix
save_simple_nc = gcm321.save_simple_nc
save_cmip6_nc = gcm321.save_cmip6_nc
_resolve_run_general = gcm321._resolve_run_general  # type: ignore[attr-defined]
_load_run_general_sampler = gcm321._load_run_general_sampler  # type: ignore[attr-defined]
_infer_date_token = gcm321._infer_date_token  # type: ignore[attr-defined]
load_meta = gcm321.load_meta
estimate_cpu_worker_count = gcm321._estimate_cpu_worker_count  # type: ignore[attr-defined]
smooth_annual_series = gcm321.smooth_annual_series
resolve_writable_output_root = gcm321.resolve_writable_output_root
resolve_task_output_directory = gcm321.resolve_task_output_directory
resolve_experiment_id_for_task = gcm321._resolve_experiment_id_for_task  # type: ignore[attr-defined]
normalize_n_ensemble_label = gcm321.normalize_n_ensemble_label
CANONICAL_KIND_ORIGINAL = gcm321.CANONICAL_KIND_ORIGINAL
CANONICAL_KIND_ORIGINAL_HEALPIX = getattr(gcm321, "CANONICAL_KIND_ORIGINAL_HEALPIX", "original_healpix")
CANONICAL_KIND_DATADERIVATIVES = "dataderivatives"
CANONICAL_KIND_CHOICES = gcm321.CANONICAL_KIND_CHOICES
DEFAULT_CANONICAL_LAYOUT = getattr(gcm321, "DEFAULT_CANONICAL_LAYOUT", True)

# Runtime canonical-output controls (set in main()).
_CANONICAL_LAYOUT_ACTIVE = DEFAULT_CANONICAL_LAYOUT
_CANONICAL_EXPERIMENT_ID_OVERRIDE: Optional[str] = None
_CANONICAL_N_ENSEMBLE_LABEL = "n_20"
_CANONICAL_RUN_INSTANCE: Optional[str] = None
get_repo_path = gcm321.get_repo_path

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
DEFAULT_CREATE_ALSO_HEALPIX_OUTPUT = (
    os.environ.get("GCMAGICC_CREATE_ALSO_HEALPIX_OUTPUT", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_HEALPIX_NSIDE = int(os.environ.get("GCMAGICC_HEALPIX_NSIDE", "64"))
HEALPIX_TO_LATLON_NSUB = int(os.environ.get("GCMAGICC_HEALPIX_TO_LATLON_NSUB", "1"))


def _consolidator_script_path() -> Path:
    return get_repo_path("gcmmagicc") / "scripts" / "2018_consolidate_era5spliced_s3.py"


def _default_autoconsolidate_config() -> Path:
    env = os.environ.get("GCMAGICC_AUTO_CONSOLIDATE_CONFIG", "").strip()
    if env:
        return Path(env).expanduser().resolve(strict=False)
    return get_repo_path("gcmmagicc") / "scripts" / "2018_consolidate_era5spliced_s3.example.json"


def _run_autoconsolidate(
    *,
    source_paths: Sequence[Path],
    config_path: Optional[Path],
    cleanup_local: bool,
) -> None:
    script = _consolidator_script_path().expanduser().resolve(strict=False)
    if not script.exists():
        raise FileNotFoundError(f"Autoconsolidate script not found: {script}")
    cfg = Path(config_path).expanduser().resolve(strict=False) if config_path else _default_autoconsolidate_config()
    if not cfg.exists():
        raise FileNotFoundError(f"Autoconsolidate config not found: {cfg}")

    cmd: List[str] = [
        sys.executable,
        str(script),
        "autoconsolidate",
        "--config",
        str(cfg),
        "--apply",
        "--verify-size-only",
    ]
    if cleanup_local:
        cmd.append("--cleanup-local")
    for src in source_paths:
        cmd.extend(["--source-path", str(Path(src).expanduser().resolve(strict=False))])

    print("🔁 Running auto-consolidate:")
    print("   " + " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Autoconsolidate failed with exit code {proc.returncode}.")


# =============================================================================
# Local helpers and dataclasses
# =============================================================================


@dataclass
class DebiasWorkItem:
    scenario: str
    ensemble_id: int  # 1-based ensemble index (for naming)
    run_id: int  # MAGICC SCM run_id used to create predictor file
    predictor_path: Path
    seed_base: int
    bias_to_era5: bool
    runmodus: str
    workflow: str
    predictor_source: Optional[str] = None
    predictor_year_start: Optional[int] = None
    predictor_year_end: Optional[int] = None


def _today_stamp() -> str:
    """
    Timestamp string consistent with 300_/321_* outputs (YYYYMMDD-HHMM, UTC).
    """
    try:
        return gcm321._today_stamp()  # type: ignore[attr-defined]
    except Exception:
        return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M")


def _workflow_to_f_flag(workflow: str) -> str:
    try:
        return gcm321.workflow_to_f_flag(workflow)  # type: ignore[attr-defined]
    except Exception:
        wf = (workflow or "").strip().lower()
        return "f2" if wf == "ar7" else "f1"


def _build_pass2_attrs(
    item: DebiasWorkItem,
    *,
    tas_pred_pass1: float,
    tas_pred_pass2: float,
    tas_out: float,
    delta_vs_pass1: float,
    delta_vs_pass2: float,
    debias_delta: float,
    debias_units: float,
    offset_type: str,
) -> Dict[str, object]:
    key = _ensemble_key(item)
    return {
        "ensemble_key": key,
        "scenario": item.scenario,
        "workflow": item.workflow,
        "runmodus": item.runmodus,
        "magicc_run_id": int(item.run_id),
        "ensemble_id": int(item.ensemble_id),
        "member_id": f"r{int(item.ensemble_id)}i1p1{_workflow_to_f_flag(item.workflow)}",
        "model_version": MODEL_VERSION,
        "predictor_path": str(item.predictor_path),
        "tas_pred_pass1_C": float(tas_pred_pass1),
        "tas_pred_pass2_C": float(tas_pred_pass2),
        "tas_out_C": float(tas_out),
        "delta_vs_pass1_C": float(delta_vs_pass1),
        "delta_vs_pass2_C": float(delta_vs_pass2),
        "debias_delta_C": float(debias_delta),
        "debias_units": float(debias_units),
        "offset_type": offset_type,
    }


def _ensemble_key(item: DebiasWorkItem) -> str:
    return (
        f"{item.scenario}|{item.workflow}|{item.runmodus}|"
        f"{item.run_id:03d}|{item.ensemble_id:04d}"
    )


def _stable_hash_mod(text: str, modulo: int) -> int:
    if modulo <= 0:
        raise ValueError(f"modulo must be positive, got {modulo}")
    digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def _stable_seed_base(
    *,
    base_seed: int,
    scenario: str,
    workflow: str,
    runmodus: str,
    run_id: int,
    ensemble_id: int,
) -> int:
    seed_material = (
        f"{scenario}|{workflow}|{runmodus}|{int(run_id):03d}|{int(ensemble_id):04d}|{MODEL_VERSION}"
    )
    return int(base_seed) + _stable_hash_mod(seed_material, 10_000)


def _manifest_row_from_item(item: DebiasWorkItem) -> Dict[str, object]:
    return {
        "ensemble_key": _ensemble_key(item),
        "scenario": item.scenario,
        "workflow": item.workflow,
        "runmodus": item.runmodus,
        "run_id": int(item.run_id),
        "ensemble_id": int(item.ensemble_id),
        "predictor_path": str(item.predictor_path),
        "predictor_source_scenario": item.predictor_source,
        "predictor_year_start": int(item.predictor_year_start) if item.predictor_year_start is not None else None,
        "predictor_year_end": int(item.predictor_year_end) if item.predictor_year_end is not None else None,
    }


def _int_or_none(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return int(text)


def _work_item_from_manifest_row(
    row: Dict[str, object],
    *,
    base_seed: int,
    bias_to_era5: bool,
) -> DebiasWorkItem:
    required = ("ensemble_key", "scenario", "workflow", "runmodus", "run_id", "ensemble_id", "predictor_path")
    missing = [k for k in required if k not in row]
    if missing:
        raise RuntimeError(f"Manifest row missing required field(s): {missing}")

    scenario = str(row["scenario"])
    workflow = str(row["workflow"])
    runmodus = str(row["runmodus"])
    run_id = int(row["run_id"])
    ensemble_id = int(row["ensemble_id"])
    predictor_path = Path(str(row["predictor_path"]))
    expected_key = f"{scenario}|{workflow}|{runmodus}|{run_id:03d}|{ensemble_id:04d}"
    provided_key = str(row.get("ensemble_key", ""))
    if provided_key != expected_key:
        raise RuntimeError(
            f"Manifest row ensemble_key mismatch: expected '{expected_key}', found '{provided_key}'."
        )

    return DebiasWorkItem(
        scenario=scenario,
        ensemble_id=ensemble_id,
        run_id=run_id,
        predictor_path=predictor_path,
        seed_base=_stable_seed_base(
            base_seed=base_seed,
            scenario=scenario,
            workflow=workflow,
            runmodus=runmodus,
            run_id=run_id,
            ensemble_id=ensemble_id,
        ),
        bias_to_era5=bias_to_era5,
        runmodus=runmodus,
        workflow=workflow,
        predictor_source=str(row.get("predictor_source_scenario") or "") or None,
        predictor_year_start=_int_or_none(row.get("predictor_year_start")),
        predictor_year_end=_int_or_none(row.get("predictor_year_end")),
    )


def _build_work_manifest_payload(items: Sequence[DebiasWorkItem]) -> Dict[str, object]:
    rows = [_manifest_row_from_item(w) for w in items]
    rows_sorted = sorted(rows, key=lambda r: str(r["ensemble_key"]))
    return {
        "schema": "gcm331_work_manifest/v1",
        "created_at_utc": _timestamp_iso(),
        "count": len(rows_sorted),
        "items": rows_sorted,
    }


def _save_work_manifest(items: Sequence[DebiasWorkItem], path: Path) -> None:
    payload = _build_work_manifest_payload(items)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _load_work_manifest(
    path: Path,
    *,
    base_seed: int,
    bias_to_era5: bool,
) -> List[DebiasWorkItem]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_items: object
    if isinstance(payload, dict):
        raw_items = payload.get("items")
    else:
        raw_items = payload
    if not isinstance(raw_items, list):
        raise RuntimeError(f"Manifest file must contain a JSON list or an object with 'items': {path}")

    work_items: List[DebiasWorkItem] = []
    seen_keys: set[str] = set()
    for i, row in enumerate(raw_items, start=1):
        if not isinstance(row, dict):
            raise RuntimeError(f"Manifest row {i} is not an object: {row!r}")
        item = _work_item_from_manifest_row(row, base_seed=base_seed, bias_to_era5=bias_to_era5)
        key = _ensemble_key(item)
        if key in seen_keys:
            raise RuntimeError(f"Duplicate ensemble_key in manifest: {key}")
        seen_keys.add(key)
        work_items.append(item)

    return sorted(
        work_items,
        key=lambda w: (w.scenario, w.workflow, w.runmodus, int(w.run_id), int(w.ensemble_id)),
    )


def _in_selected_shard(
    ensemble_key: str,
    *,
    shard_count: int,
    shard_index: int,
    shard_strategy: str,
) -> bool:
    if shard_count <= 0:
        raise ValueError(f"shard_count must be >=1, got {shard_count}")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count - 1}], got {shard_index}")
    if shard_strategy != "keyhash":
        raise ValueError(f"Unsupported shard strategy: {shard_strategy}")
    return _stable_hash_mod(ensemble_key, shard_count) == shard_index


def _log_work_item_breakdown(items: Sequence[DebiasWorkItem], *, prefix: str) -> None:
    counts: Dict[Tuple[str, str, str], int] = {}
    for item in items:
        key = (item.scenario, item.workflow, item.runmodus)
        counts[key] = counts.get(key, 0) + 1
    print(f"{prefix} ({len(items)} items)")
    for (scenario, workflow, runmodus), count in sorted(counts.items()):
        print(f"   - {scenario} | {workflow} | {runmodus}: {count}")


def _area_weights_coslat(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Simple cos(lat) weights normalized to mean 1 over the globe."""
    cosw = np.cos(np.deg2rad(lats))[:, None] * np.ones((len(lats), len(lons)), dtype=np.float64)
    return (cosw / cosw.mean()).astype(np.float64)


def _cleanup_cuda_cache() -> None:
    """Best-effort CUDA cache cleanup to recover from OOM spikes."""
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _infer_default_nside(sample_fn=None) -> int:
    """Best-effort detection of nside default used by run_general/run_gcmagicc."""
    for attr in ("NSIDE_DEFAULT", "DEFAULT_NSIDE", "NSIDE"):
        val = getattr(gcm321, attr, None)
        if isinstance(val, int) and val > 0:
            return int(val)

    try:
        src = inspect.getsource(gcm321.run_gcmagicc)
        match = re.search(r"setdefault\(\s*[\"']nside[\"']\s*,\s*([0-9]+)\s*\)", src)
        if match:
            return int(match.group(1))
    except Exception:
        pass

    if sample_fn is not None:
        try:
            sig = inspect.signature(sample_fn)
            param = sig.parameters.get("nside")
            if param and param.default is not inspect._empty and param.default is not None:
                return int(param.default)
        except Exception:
            pass

    return 256


def _estimate_per_job_ram_from_nside(nside: int) -> float:
    """
    Heuristic RAM per job scaling anchored at nside=64->40GB and nside=256->250GB.
    """
    base_nside = 64
    base_ram_gb = 250.0
    target_nside = 128
    target_ram_gb = 1000.0
    if nside <= 0:
        return base_ram_gb
    exponent = math.log(target_ram_gb / base_ram_gb) / math.log(target_nside / base_nside)
    scaled = base_ram_gb * (nside / base_nside) ** exponent
    return max(base_ram_gb, scaled)


def _process_memory_gb() -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (rss_gb, vms_gb) for current process using psutil if available,
    otherwise falls back to resource.ru_maxrss (rss only, rough).
    """
    try:
        import psutil  # type: ignore

        mem = psutil.Process(os.getpid()).memory_info()
        return mem.rss / (1024**3), mem.vms / (1024**3)
    except Exception:
        try:
            import resource  # type: ignore

            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_bytes = rss_kb * 1024.0
            return rss_bytes / (1024**3), None
        except Exception:
            return None, None


def _system_memory_gb() -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (total_gb, available_gb) using psutil if available,
    otherwise parses /proc/meminfo.
    """
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return vm.total / (1024**3), vm.available / (1024**3)
    except Exception:
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
            if total is None or avail is None:
                return None, None
            return total / (1024**2), avail / (1024**2)
        except Exception:
            return None, None


def _fmt_bytes(num: Optional[float]) -> str:
    if num is None or not math.isfinite(num):
        return "n/a"
    step = 1024.0
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    idx = 0
    while num >= step and idx < len(units) - 1:
        num /= step
        idx += 1
    return f"{num:.2f} {units[idx]}"


def _fmt_gb(val: Optional[float]) -> str:
    if val is None or not math.isfinite(val):
        return "n/a"
    return f"{val:.2f} GB"


def _log_memory(prefix: str, key: str, *, est_bytes: Optional[float] = None) -> None:
    rss_gb, vms_gb = _process_memory_gb()
    _, avail_gb = _system_memory_gb()
    est_part = f", est_out={_fmt_bytes(est_bytes)}" if est_bytes is not None else ""
    print(
        f"{prefix} {key} mem rss={_fmt_gb(rss_gb)}, vms={_fmt_gb(vms_gb)}, "
        f"sys_avail={_fmt_gb(avail_gb)}{est_part}"
    )


def _timestamp_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_year_dict(data: Optional[Dict[object, object]]) -> Dict[int, float]:
    """
    Convert year-keyed dicts loaded from JSON (string keys) back to int->float.
    """
    if not isinstance(data, dict):
        return {}
    out: Dict[int, float] = {}
    for k, v in data.items():
        try:
            out[int(k)] = float(v)
        except Exception:
            continue
    return out


def _load_progress_json(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    # Normalize per-year keys back to ints for downstream logic
    if isinstance(data, dict):
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            p1 = entry.get("pass1")
            if isinstance(p1, dict):
                p1["per_year_deltas"] = _coerce_year_dict(p1.get("per_year_deltas"))
                p1["per_year_tas_out"] = _coerce_year_dict(p1.get("per_year_tas_out"))
                p1["per_year_tas_pred"] = _coerce_year_dict(p1.get("per_year_tas_pred"))
                entry["pass1"] = p1
            p2 = entry.get("pass2")
            if isinstance(p2, dict):
                entry["pass2"] = p2
            data[key] = entry
    return data


def _save_progress_json(progress: Dict[str, Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _update_progress_entry(
    progress: Dict[str, Dict[str, object]],
    key: str,
    *,
    status: str,
    pass1: Optional[Dict[str, object]] = None,
    pass2: Optional[Dict[str, object]] = None,
    message: Optional[str] = None,
    output_root: Optional[Path] = None,
    shard_id: Optional[str] = None,
    host: Optional[str] = None,
) -> None:
    entry = progress.get(key, {})
    entry["status"] = status
    entry["updated"] = _timestamp_iso()
    if pass1 is not None:
        entry["pass1"] = pass1
    if pass2 is not None:
        entry["pass2"] = pass2
    if message:
        entry["message"] = message
    if output_root is not None:
        entry["output_root"] = str(output_root)
    if shard_id:
        entry["shard_id"] = str(shard_id)
    if host:
        entry["host"] = str(host)
    progress[key] = entry


def _plan_cpu_workers(per_job_ram_gb: float, requested: Optional[int] = None, safety_fraction: float = 0.20) -> int:
    """
    Memory-aware CPU worker planner that caps workers based on available RAM.
    """
    base = max(
        1,
        int(
            estimate_cpu_worker_count(
                per_job_cpus=25,
                per_job_ram_gb=per_job_ram_gb,
                safety_fraction=safety_fraction,
            )
        ),
    )
    if requested is not None:
        base = min(base, max(1, int(requested)))

    _, avail_gb = _system_memory_gb()
    if avail_gb is not None and per_job_ram_gb > 0:
        mem_limited = max(1, int(math.floor((avail_gb * (1.0 - safety_fraction)) / per_job_ram_gb)))
        base = max(1, min(base, mem_limited))
    return base


def _cpu_pool_context():
    """Multiprocessing context helper (fork fallback like 330_*)."""
    try:
        return gcm321.get_context("fork")  # type: ignore[attr-defined]
    except Exception:
        return get_context("fork")


def _compute_global_tas_mean_C(
    yhval: np.ndarray,
    weights: np.ndarray,
    tas_index: int,
) -> float:
    """
    Compute global area-weighted *time-mean* tas in °C from emulator output.
    """
    if isinstance(yhval, torch.Tensor):
        yhval = yhval.detach().cpu().numpy()

    if yhval.ndim != 4:
        raise ValueError(f"yhval must have shape (T,V,lat,lon), got {yhval.shape}")

    tas_K = yhval[:, tas_index, :, :]
    w_sum = weights.sum()
    if w_sum <= 0:
        raise RuntimeError("Area weights sum to zero; cannot compute global mean.")
    tas_global_K = (tas_K * weights[None, :, :]).sum(axis=(1, 2)) / w_sum
    return float(np.mean(tas_global_K) - 273.15)


def _compute_predictor_tas_mean_C(predictors) -> float:
    """
    Compute time-mean predictor tas_smoothed in °C.
    predictors.X has tas_smoothed column as (K - 273.15)/10 per 321_*.
    """
    names = list(getattr(predictors, "predictor_names", []))
    if "tas_smoothed" not in names:
        raise ValueError(
            f"'tas_smoothed' not found in predictor_names: {predictors.predictor_names}"
        )
    idx = names.index("tas_smoothed")
    x = predictors.X
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    tas_smoothed_units = np.asarray(x, dtype=np.float32)[:, idx]
    tas_C = tas_smoothed_units * 10.0
    return float(np.mean(tas_C))


def _compute_per_year_deltas(
    yhval: np.ndarray,
    predictors,
    weights: np.ndarray,
    tas_index: int,
    year: np.ndarray,
    return_full_data: bool = False,
) -> Union[Dict[int, float], Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]]:
    """
    Compute delta (tas_out - tas_pred) per year to diagnose year-dependent bias.
    tas_out is smoothed with a centered 21-year running mean before differencing
    to align with MAGICC predictor timescale.
    Returns {year: delta_C} dictionary, or if return_full_data=True, returns
    (per_year_deltas, per_year_tas_out, per_year_tas_pred) tuple.
    """
    if isinstance(yhval, torch.Tensor):
        yhval = yhval.detach().cpu().numpy()
    
    if yhval.ndim != 4:
        raise ValueError(f"yhval must have shape (T,V,lat,lon), got {yhval.shape}")
    
    # Compute global tas per time step
    tas_K = yhval[:, tas_index, :, :]
    w_sum = weights.sum()
    if w_sum <= 0:
        raise RuntimeError("Area weights sum to zero; cannot compute global mean.")
    tas_global_K = (tas_K * weights[None, :, :]).sum(axis=(1, 2)) / w_sum
    tas_out_C = tas_global_K - 273.15
    
    # Get predictor tas_smoothed
    names = list(getattr(predictors, "predictor_names", []))
    if "tas_smoothed" not in names:
        raise ValueError(f"'tas_smoothed' not found in predictor_names: {predictors.predictor_names}")
    idx = names.index("tas_smoothed")
    x = predictors.X
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    tas_smoothed_units = np.asarray(x, dtype=np.float32)[:, idx]
    tas_pred_C = tas_smoothed_units * 10.0
    
    # Compute per-year means
    year = np.asarray(year, dtype=int)
    unique_years = np.unique(year)
    unique_years_sorted = np.sort(unique_years)
    per_year_deltas = {}
    per_year_tas_out = {}
    per_year_tas_pred = {}
    per_year_out_list: List[float] = []
    per_year_pred_list: List[float] = []
    
    for y in unique_years_sorted:
        year_mask = (year == y)
        if not year_mask.any():
            continue
        tas_out_year = float(np.mean(tas_out_C[year_mask]))
        tas_pred_year = float(np.mean(tas_pred_C[year_mask]))
        per_year_out_list.append(tas_out_year)
        per_year_pred_list.append(tas_pred_year)

    if per_year_out_list:
        tas_out_smoothed = smooth_annual_series(
            np.asarray(unique_years_sorted, dtype=int),
            np.asarray(per_year_out_list, dtype=np.float64),
            window=PER_YEAR_SMOOTH_WINDOW,
        )
    else:
        tas_out_smoothed = np.array([], dtype=np.float64)

    for i, y in enumerate(unique_years_sorted):
        if i >= len(tas_out_smoothed) or i >= len(per_year_pred_list):
            continue
        tas_out_year = float(tas_out_smoothed[i])
        tas_pred_year = float(per_year_pred_list[i])
        per_year_deltas[int(y)] = float(tas_out_year - tas_pred_year)
        if return_full_data:
            per_year_tas_out[int(y)] = tas_out_year
            per_year_tas_pred[int(y)] = tas_pred_year
    
    if return_full_data:
        return per_year_deltas, per_year_tas_out, per_year_tas_pred
    return per_year_deltas


def _plot_delta_histogram(
    deltas_C: List[float],
    output_dir: Path,
    title_suffix: str = "",
) -> Path:
    """
    Simple histogram of Δ offsets (°C) for all ensembles.
    """
    arr = np.asarray(deltas_C, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return output_dir / "debias_offsets_empty.png"

    mean = float(np.mean(arr))
    std = float(np.std(arr))
    p5, p50, p95 = np.nanpercentile(arr, [5, 50, 95])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(arr, bins=30, alpha=0.7, edgecolor="black")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, label="Δ = 0")
    ax.axvline(mean, color="red", linestyle="-", linewidth=1.2, label=f"mean = {mean:.3f} °C")
    ax.set_xlabel("Δ = tas_out - tas_pred (°C)")
    ax.set_ylabel("Count")

    subtitle = f"n={arr.size}, mean={mean:.3f}, std={std:.3f}, p50={p50:.3f}, [p5,p95]=[{p5:.3f},{p95:.3f}]"
    title = "Debias offsets Δ (global tas)"
    if title_suffix:
        title = f"{title} • {title_suffix}"
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.4)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"debias_offsets_hist_{title_suffix or 'global'}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_delta_timeseries(
    delta_records: List[Dict[str, object]],
    output_dir: Path,
    title_suffix: str = "",
) -> Path:
    """
    Plot timeseries of per-year delta offsets for all ensembles.
    
    Args:
        delta_records: List of records from PASS1, each containing 'per_year_deltas' dict
        output_dir: Directory to save the plot
        title_suffix: Optional suffix for plot title
    
    Returns:
        Path to saved plot file
    """
    if not delta_records:
        return output_dir / "debias_offsets_timeseries_empty.png"
    
    # Collect all per-year deltas
    all_years = set()
    ensemble_data = []
    
    for rec in delta_records:
        per_year_deltas = rec.get("per_year_deltas", {})
        if not per_year_deltas or not isinstance(per_year_deltas, dict):
            continue
        
        # Convert to sorted list of (year, delta) tuples
        year_delta_pairs = sorted([(int(y), float(d)) for y, d in per_year_deltas.items()])
        if not year_delta_pairs:
            continue
        
        years = [y for y, _ in year_delta_pairs]
        deltas = [d for _, d in year_delta_pairs]
        all_years.update(years)
        
        ensemble_key = rec.get("ensemble_key", "unknown")
        scenario = rec.get("scenario", "unknown")
        ensemble_id = rec.get("ensemble_id", 0)
        
        ensemble_data.append({
            "key": ensemble_key,
            "scenario": scenario,
            "ensemble_id": ensemble_id,
            "years": years,
            "deltas": deltas,
        })
    
    if not ensemble_data:
        return output_dir / "debias_offsets_timeseries_empty.png"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each ensemble as a line
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(ensemble_data), 20)))
    for i, ed in enumerate(ensemble_data):
        color = colors[i % len(colors)]
        label = f"{ed['scenario']} r{ed['ensemble_id']}"
        ax.plot(ed["years"], ed["deltas"], marker="o", markersize=3, linewidth=1.5, 
                alpha=0.7, color=color, label=label)
    
    # Add zero line
    if all_years:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5, label="Δ = 0")
    
    # Compute ensemble mean per year (if multiple ensembles)
    if len(ensemble_data) > 1 and all_years:
        years_sorted = sorted(all_years)
        mean_deltas = []
        for year in years_sorted:
            year_deltas = []
            for ed in ensemble_data:
                if year in ed["years"]:
                    idx = ed["years"].index(year)
                    year_deltas.append(ed["deltas"][idx])
            if year_deltas:
                mean_deltas.append(np.mean(year_deltas))
            else:
                mean_deltas.append(np.nan)
        
        # Plot ensemble mean
        valid_mask = ~np.isnan(mean_deltas)
        if valid_mask.any():
            ax.plot(np.array(years_sorted)[valid_mask], np.array(mean_deltas)[valid_mask],
                   color="red", linewidth=2.5, linestyle="-", marker="s", markersize=6,
                   label=f"Ensemble mean (n={len(ensemble_data)})", zorder=10)
    
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Δ = tas_out - tas_pred (°C)", fontsize=12)
    
    title = "Per-year debias offsets Δ (timeseries)"
    if title_suffix:
        title = f"{title} • {title_suffix}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Legend
    if len(ensemble_data) <= 20:
        ax.legend(loc="best", fontsize=9, ncol=2)
    else:
        # Too many ensembles, show only mean
        handles, labels = ax.get_legend_handles_labels()
        mean_handle = [h for h, l in zip(handles, labels) if "Ensemble mean" in l]
        mean_label = [l for l in labels if "Ensemble mean" in l]
        if mean_handle:
            ax.legend(mean_handle + [handles[0]], mean_label + ["Individual ensembles"], 
                     loc="best", fontsize=9)
    
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlim(left=min(all_years) - 1 if all_years else None, 
                right=max(all_years) + 1 if all_years else None)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"debias_offsets_timeseries_{title_suffix or 'all'}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_tas_timeseries_overlay(
    delta_records: List[Dict[str, object]],
    output_dir: Path,
    title_suffix: str = "",
) -> Path:
    """
    Plot timeseries overlay of tas_out, tas_pred_pass1, and delta for each ensemble.
    Creates one plot per ensemble with all three timeseries.
    
    Args:
        delta_records: List of records from PASS1, each containing per-year data
        output_dir: Directory to save the plots
        title_suffix: Optional suffix for plot title
    
    Returns:
        Path to saved plot file (or directory if multiple plots)
    """
    if not delta_records:
        return output_dir / "tas_timeseries_overlay_empty.png"
    
    # Collect data for each ensemble
    ensemble_plots = []
    
    for rec in delta_records:
        per_year_deltas = rec.get("per_year_deltas", {})
        per_year_tas_out = rec.get("per_year_tas_out", {})
        per_year_tas_pred = rec.get("per_year_tas_pred", {})
        
        if not per_year_deltas or not isinstance(per_year_deltas, dict):
            continue
        if not per_year_tas_out or not isinstance(per_year_tas_out, dict):
            continue
        if not per_year_tas_pred or not isinstance(per_year_tas_pred, dict):
            continue
        
        # Get common years
        years = sorted(set(per_year_deltas.keys()) & set(per_year_tas_out.keys()) & set(per_year_tas_pred.keys()))
        if not years:
            continue
        
        ensemble_key = rec.get("ensemble_key", "unknown")
        scenario = rec.get("scenario", "unknown")
        ensemble_id = rec.get("ensemble_id", 0)
        
        # Extract data for common years
        tas_out_vals = [per_year_tas_out[y] for y in years]
        tas_pred_vals = [per_year_tas_pred[y] for y in years]
        delta_vals = [per_year_deltas[y] for y in years]
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(14, 6))
        
        # Plot tas_out and tas_pred on left y-axis
        ax1.plot(years, tas_out_vals, marker="o", markersize=4, linewidth=2, 
                color="blue", label="tas_out", alpha=0.8)
        ax1.plot(years, tas_pred_vals, marker="s", markersize=4, linewidth=2, 
                color="green", label="tas_pred_pass1", alpha=0.8)
        ax1.set_xlabel("Year", fontsize=12)
        ax1.set_ylabel("Temperature (°C)", fontsize=12, color="black")
        ax1.tick_params(axis="y", labelcolor="black")
        ax1.grid(True, linestyle=":", alpha=0.4)
        
        # Create second y-axis for delta
        ax2 = ax1.twinx()
        ax2.plot(years, delta_vals, marker="^", markersize=4, linewidth=2, 
                color="red", label="Δ = tas_out - tas_pred", alpha=0.8)
        ax2.axhline(0.0, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax2.set_ylabel("Δ (°C)", fontsize=12, color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        
        # Title and legend
        title = f"Temperature timeseries: {scenario} r{ensemble_id}"
        if title_suffix:
            title = f"{title} • {title_suffix}"
        ax1.set_title(title, fontsize=13, fontweight="bold")
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)
        
        # Set x-axis limits
        if years:
            ax1.set_xlim(left=min(years) - 1, right=max(years) + 1)
        
        fig.tight_layout()
        
        # Save individual plot
        plot_filename = f"tas_timeseries_{scenario}_r{ensemble_id:04d}.png"
        plot_path = output_dir / plot_filename
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        ensemble_plots.append(plot_path)
    
    if not ensemble_plots:
        return output_dir / "tas_timeseries_overlay_empty.png"
    
    # Return path to first plot (or directory if multiple)
    if len(ensemble_plots) == 1:
        return ensemble_plots[0]
    else:
        # Return directory path if multiple plots
        return output_dir


def _apply_tas_offset(
    predictors, 
    delta_C: float, 
    year: Optional[np.ndarray] = None,
    per_year_deltas: Optional[Dict[int, float]] = None,
    use_per_year: bool = False,
) -> float:
    """
    Apply Δ offset to tas_smoothed so tas_out shifts toward original predictors.
    
    Args:
        predictors: PredictorData object
        delta_C: Overall delta in °C (time-mean)
        year: Optional year array for per-year application
        per_year_deltas: Optional {year: delta_C} dict for per-year offsets
        use_per_year: If True and per_year_deltas provided, apply per-year offsets
    
    Returns:
        Mean applied delta_units in predictor space.
    
    Note: By default, applies constant offset to all time steps (correct for mean bias correction).
    If use_per_year=True and per_year_deltas provided, applies year-specific offsets.
    """
    names = list(getattr(predictors, "predictor_names", []))
    if "tas_smoothed" not in names:
        return 0.0
    idx = names.index("tas_smoothed")
    x = predictors.X
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
        is_tensor = True
    else:
        x_np = np.asarray(x, dtype=np.float32)
        is_tensor = False
    
    if use_per_year and per_year_deltas is not None and year is not None:
        # Apply per-year offsets: each time step gets its year-specific delta
        year_arr = np.asarray(year, dtype=int)
        applied_deltas = []
        unique_years_applied = set()
        for i, y in enumerate(year_arr):
            year_delta_C = per_year_deltas.get(int(y), delta_C)
            delta_units = -(year_delta_C) / 10.0
            x_np[i, idx] = x_np[i, idx] + float(delta_units)
            applied_deltas.append(delta_units)
            unique_years_applied.add(int(y))
        mean_delta_units = float(np.mean(applied_deltas))
        # Verify: check that we're using multiple different year-specific deltas
        unique_deltas_applied = len(set(per_year_deltas.get(int(y), delta_C) for y in unique_years_applied))
        if unique_deltas_applied == 1 and len(unique_years_applied) > 1:
            # This shouldn't happen if per-year deltas vary, but it's possible if all years have same delta
            pass  # All years happen to have the same delta
    else:
        # Apply constant offset (default behavior - correct for mean bias correction)
        delta_units = -(delta_C) / 10.0
        x_np[:, idx] = x_np[:, idx] + float(delta_units)
        mean_delta_units = float(delta_units)
    
    if is_tensor:
        predictors.X[:, idx] = torch.from_numpy(x_np[:, idx]).float()
    else:
        predictors.X = torch.from_numpy(x_np).float()
    
    return mean_delta_units


def _ensure_model_index_era5(predictors) -> None:
    """Force model_index predictor to ERA5 index 0 if present."""
    names = list(getattr(predictors, "predictor_names", []))
    if "model_index" in names:
        mi_col = names.index("model_index")
        predictors.X[:, mi_col] = float(0)


def _effect_source_label(
    useeffect_model: Optional[int],
    model_to_index: Optional[Dict[str, int]] = None,
) -> str:
    """
    Return label for effect model index (ERA5 for 0, mapped model name for >0).
    """
    lookup: Dict[int, str] = _MODEL_INDEX_TO_NAME
    if model_to_index:
        try:
            lookup = {int(v): str(k) for k, v in model_to_index.items()}
        except Exception:
            pass
    try:
        eff_idx = int(useeffect_model) if useeffect_model is not None else None
    except Exception:
        return str(useeffect_model)
    if eff_idx is None:
        return "N"
    if eff_idx == 0:
        return "ERA5"
    return lookup.get(eff_idx, f"model{eff_idx}")


def _variable_string(variable_names: Optional[Sequence[str]]) -> str:
    """Join variable names (in-order) for filename suffix."""
    if not variable_names:
        return ""
    cleaned = [str(v) for v in variable_names if str(v).strip()]
    return "-".join(cleaned)


def _format_output_paths(
    *,
    stage: str,
    output_root: Path,
    item: DebiasWorkItem,
    runmodus: str,
    dependence: bool,
    usebias_model: Optional[int],
    useeffect_model: Optional[int],
    variable_names: Optional[Sequence[str]] = None,
    model_to_index: Optional[Dict[str, int]] = None,
    canonical_kind_override: Optional[str] = None,
) -> Tuple[Path, str, str]:
    """
    Build output directory and filename stem consistent with 300_/321_* naming
    (timestamp, source label, scenario, member, variable list).
    Returns (stage_root, filename_stem, version_code).
    """
    dateflag = _today_stamp()
    depflag = "d1" if dependence else "d0"

    version_code = model_version_to_code(MODEL_VERSION)
    if not version_code:
        if MODEL_VERSION.startswith("v"):
            version_code = MODEL_VERSION
        else:
            parts = MODEL_VERSION.split(".")
            if len(parts) >= 3:
                version_code = f"v{parts[0]}{parts[1]}{parts[2]}"
            else:
                version_code = f"v{MODEL_VERSION.replace('.', '')}"
    version_flag = version_code[1:] if version_code.startswith("v") else version_code

    if usebias_model is None:
        bias_flag = "N"
    elif usebias_model == 0:
        bias_flag = "0"
    else:
        bias_flag = str(usebias_model)

    if useeffect_model is None:
        effect_flag = "N"
    elif useeffect_model == 0:
        effect_flag = "0"
    else:
        effect_flag = str(useeffect_model)

    magicc_ens_flag = item.run_id
    ens_flag = f"r{int(item.ensemble_id)}i1p1{_workflow_to_f_flag(item.workflow)}"
    runmodus_suffix = runmodus_to_suffix(runmodus)

    bias_token = f"b{bias_flag}"
    effect_token = f"e{effect_flag}"
    ensemble_token = f"m{magicc_ens_flag}"
    prefix = f"GCMagicc-v{version_flag}-{depflag}{bias_token}{effect_token}{ensemble_token}-{dateflag}"

    effect_label = _effect_source_label(useeffect_model, model_to_index=model_to_index)
    scenario_part = f"{item.scenario}{runmodus_suffix}"
    var_str = _variable_string(variable_names if variable_names is not None else _DEFAULT_VARLIST)

    parts = [prefix, effect_label, scenario_part, ens_flag]
    if var_str:
        parts.append(var_str)
    fname_root = "_".join(parts)

    workflow_dir = str(item.workflow or "UNKNOWN").upper()
    if _CANONICAL_LAYOUT_ACTIVE:
        canonical_kind = (
            str(canonical_kind_override).strip().lower()
            if canonical_kind_override
            else CANONICAL_KIND_ORIGINAL
            if str(stage).strip().lower() == "debias"
            else CANONICAL_KIND_DATADERIVATIVES
        )
        experiment_id = resolve_experiment_id_for_task(
            scenario=item.scenario,
            runmodus=runmodus,
            experiment_override=_CANONICAL_EXPERIMENT_ID_OVERRIDE,
        )
        stage_root, _ = resolve_task_output_directory(
            output_root=Path(output_root),
            model_version=MODEL_VERSION,
            scenario=item.scenario,
            workflow=workflow_dir,
            runmodus=runmodus,
            canonical_layout=True,
            canonical_experiment_id=experiment_id,
            canonical_n_ensemble=_CANONICAL_N_ENSEMBLE_LABEL,
            canonical_kind=canonical_kind,
            canonical_run_instance=_CANONICAL_RUN_INSTANCE,
        )
        if canonical_kind == CANONICAL_KIND_DATADERIVATIVES:
            stage_root = stage_root / "run_artifacts" / stage
    else:
        legacy_stage = (
            "debias_healpix"
            if canonical_kind_override == CANONICAL_KIND_ORIGINAL_HEALPIX and str(stage).strip().lower() == "debias"
            else stage
        )
        stage_root = output_root / legacy_stage / version_code / item.scenario / workflow_dir
    stage_root.mkdir(parents=True, exist_ok=True)
    return stage_root, fname_root, version_code


def _estimate_output_bytes(time_steps: int, nvars: int, nlat: int, nlon: int, dtype_bytes: int = 4) -> float:
    """Rough estimate of output tensor bytes for one run (T × V × lat × lon)."""
    return float(time_steps) * float(nvars) * float(nlat) * float(nlon) * float(dtype_bytes)


def _planned_output_roots(
    items: Sequence[DebiasWorkItem],
    *,
    output_root: Path,
    canonical_layout: bool,
    canonical_n_ensemble: str,
    create_also_healpix_output: bool,
) -> Dict[str, List[str]]:
    roots: Dict[str, set[str]] = {
        CANONICAL_KIND_ORIGINAL: set(),
        CANONICAL_KIND_DATADERIVATIVES: set(),
    }
    if create_also_healpix_output:
        roots[CANONICAL_KIND_ORIGINAL_HEALPIX] = set()

    for item in items:
        workflow_dir = str(item.workflow or "UNKNOWN").upper()
        if canonical_layout:
            experiment_id = resolve_experiment_id_for_task(
                scenario=item.scenario,
                runmodus=item.runmodus,
                experiment_override=_CANONICAL_EXPERIMENT_ID_OVERRIDE,
            )
            original_root, _ = resolve_task_output_directory(
                output_root=Path(output_root),
                model_version=MODEL_VERSION,
                scenario=item.scenario,
                workflow=workflow_dir,
                runmodus=item.runmodus,
                canonical_layout=True,
                canonical_experiment_id=experiment_id,
                canonical_n_ensemble=canonical_n_ensemble,
                canonical_kind=CANONICAL_KIND_ORIGINAL,
                canonical_run_instance=_CANONICAL_RUN_INSTANCE,
            )
            dataderivatives_root, _ = resolve_task_output_directory(
                output_root=Path(output_root),
                model_version=MODEL_VERSION,
                scenario=item.scenario,
                workflow=workflow_dir,
                runmodus=item.runmodus,
                canonical_layout=True,
                canonical_experiment_id=experiment_id,
                canonical_n_ensemble=canonical_n_ensemble,
                canonical_kind=CANONICAL_KIND_DATADERIVATIVES,
                canonical_run_instance=_CANONICAL_RUN_INSTANCE,
            )
            roots[CANONICAL_KIND_ORIGINAL].add(str(original_root))
            roots[CANONICAL_KIND_DATADERIVATIVES].add(str(dataderivatives_root / "run_artifacts"))
            if create_also_healpix_output:
                healpix_root, _ = resolve_task_output_directory(
                    output_root=Path(output_root),
                    model_version=MODEL_VERSION,
                    scenario=item.scenario,
                    workflow=workflow_dir,
                    runmodus=item.runmodus,
                    canonical_layout=True,
                    canonical_experiment_id=experiment_id,
                    canonical_n_ensemble=canonical_n_ensemble,
                    canonical_kind=CANONICAL_KIND_ORIGINAL_HEALPIX,
                    canonical_run_instance=_CANONICAL_RUN_INSTANCE,
                )
                roots[CANONICAL_KIND_ORIGINAL_HEALPIX].add(str(healpix_root))
        else:
            version_code = model_version_to_code(MODEL_VERSION)
            roots[CANONICAL_KIND_ORIGINAL].add(
                str(Path(output_root) / "debias" / version_code / item.scenario / workflow_dir)
            )
            roots[CANONICAL_KIND_DATADERIVATIVES].add(
                str(Path(output_root) / PASS1_TEMP_STAGE / version_code / item.scenario / workflow_dir)
            )
            if create_also_healpix_output:
                roots[CANONICAL_KIND_ORIGINAL_HEALPIX].add(
                    str(Path(output_root) / "debias_healpix" / version_code / item.scenario / workflow_dir)
                )

    return {kind: sorted(paths) for kind, paths in roots.items()}


def _print_dry_run_summary(
    *,
    work_items_all: Sequence[DebiasWorkItem],
    work_items_selected: Sequence[DebiasWorkItem],
    output_root: Path,
    canonical_layout: bool,
    canonical_n_ensemble: str,
    create_also_healpix_output: bool,
    healpix_nside: int,
    output_mode: str,
    manifest_out_path: Optional[Path],
    shard_runtime_id: str,
    shard_count: int,
    shard_index: int,
) -> None:
    print("🧪 DRY RUN: no PASS1/PASS2 model execution and no output files will be written.")
    print(f"   total work items before sharding: {len(work_items_all)}")
    print(f"   selected work items: {len(work_items_selected)}")
    print(f"   shard: {shard_index}/{shard_count} ({shard_runtime_id})")
    print(f"   output root: {output_root}")
    print(f"   output mode: {output_mode}")
    print(f"   canonical layout: {canonical_layout}")
    print(f"   create HEALPix output: {create_also_healpix_output}")
    if create_also_healpix_output:
        print(f"   HEALPix: nside={int(healpix_nside)}, order=ring, lat/lon regrid nsub={HEALPIX_TO_LATLON_NSUB}")
    if manifest_out_path is not None:
        print(f"   work manifest would be written to: {manifest_out_path}")

    roots = _planned_output_roots(
        work_items_selected,
        output_root=output_root,
        canonical_layout=canonical_layout,
        canonical_n_ensemble=canonical_n_ensemble,
        create_also_healpix_output=create_also_healpix_output,
    )
    for kind, paths in roots.items():
        print(f"   planned {kind} roots ({len(paths)}):")
        for path in paths[:10]:
            print(f"      - {path}")
        if len(paths) > 10:
            print(f"      ... {len(paths) - 10} more")


def _maybe_fallback_device(
    device: str,
    *,
    time_steps: int,
    nvars: int,
    nlat: int,
    nlon: int,
    safety_factor: float = 2.0,
    verbose: bool = True,
) -> str:
    """
    If the estimated output size would exceed available GPU memory, fall back to CPU.
    safety_factor>1 accounts for intermediate buffers during sampling.
    """
    if not isinstance(device, str) or not device.startswith("cuda"):
        return device
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:
        return device

    est_bytes = _estimate_output_bytes(time_steps, nvars, nlat, nlon)
    est_bytes *= max(1.0, safety_factor)
    if est_bytes >= free_bytes:
        if verbose:
            est_gb = est_bytes / (1024**3)
            free_gb = free_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)
            print(
                f"⚠️  Estimated output ~{est_gb:.1f} GiB (incl. safety) exceeds free GPU memory {free_gb:.1f}/{total_gb:.1f} GiB; falling back to CPU."
            )
        return "cpu"
    return device


def _validate_healpix_nside(nside: int) -> int:
    nside_int = int(nside)
    if nside_int <= 0 or (nside_int & (nside_int - 1)) != 0:
        raise RuntimeError(f"--nside must be a positive power of two, got {nside}.")
    return nside_int


def _run_gcmagicc_native_healpix(
    sample_fn,
    predictor_data,
    *,
    dependence: bool,
    usebias_model=None,
    useeffect_model=None,
    device: Optional[str],
    models_dir: Path,
    date_token: Optional[str],
    nside: int,
    force_gpu: bool = False,
    amp: bool = False,
    seed: Optional[int] = None,
):
    """
    Run the underlying sampler in native HEALPix mode.

    This mirrors gcm321.run_gcmagicc but requests rectangular=False so PASS1 and
    PASS2 can derive both native HEALPix and 1x1 lat/lon output from one draw.
    """
    requested_device = normalize_device_string(device) or DEFAULT_DEVICE
    use_gpu = str(requested_device).startswith("cuda")

    if use_gpu and not torch.cuda.is_available() and not force_gpu:
        print(
            f"[worker] ⚠️  Requested GPU '{requested_device}' but torch.cuda.is_available() is False; falling back to CPU."
        )
        requested_device = "cpu"
        use_gpu = False
    elif use_gpu and not torch.cuda.is_available() and force_gpu:
        print(
            f"[worker] ⚠️  Forcing GPU usage for '{requested_device}' despite torch.cuda.is_available() being False."
        )

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
        "rectangular": False,
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
    if "nside" in sig.parameters:
        kwargs["nside"] = int(nside)
    if "nsub" in sig.parameters:
        kwargs.setdefault("nsub", HEALPIX_TO_LATLON_NSUB)

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

    if use_gpu:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            idx = int(str(requested_device).split(":", 1)[1]) if ":" in str(requested_device) else torch.cuda.current_device()
            torch.cuda.set_device(idx)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(idx)
        except Exception:
            idx = 0

        if amp:
            amp_dtype = getattr(gcm321, "_amp_dtype_cuda", lambda: torch.float16)()
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

    return np.asarray(out)


def _healpix_to_latlon_grid(
    hp_cube: Union[np.ndarray, torch.Tensor],
    nlat: int,
    *,
    order: str = "RING",
    nsub: int = HEALPIX_TO_LATLON_NSUB,
) -> np.ndarray:
    """
    Convert (time, variable, cell) native HEALPix maps to (time, variable, lat, lon).

    The math mirrors the nested helper in the GCMAGICC run_general samplers so
    native-first dual output stays numerically aligned with rectangular=True.
    """
    import healpy as hp  # type: ignore

    if isinstance(hp_cube, torch.Tensor):
        cube = hp_cube.detach().cpu().numpy()
    else:
        cube = np.asarray(hp_cube)
    if cube.ndim != 3:
        raise ValueError(f"hp_cube must have shape (time, variable, cell), got {cube.shape}")

    nlat_int = int(nlat)
    nsub_int = max(1, int(nsub))
    lats = 90 - (0.5 + np.arange(nlat_int)) / nlat_int * 180
    lons = (0.5 + np.arange(2 * nlat_int)) / nlat_int * 180

    n_time, n_var, n_pix = cube.shape
    nside_local = hp.npix2nside(int(n_pix))
    if 12 * nside_local * nside_local != int(n_pix):
        raise ValueError(f"NPIX={n_pix} invalid (must be 12*nside^2)")

    nest = str(order).upper() == "NEST"
    offs = (((np.arange(nsub_int) + 0.5) / nsub_int - 0.5) / nlat_int * 180)

    lat_sub = lats[:, None, None, None] + offs[None, None, :, None]
    lon_sub = lons[None, :, None, None] + offs[None, None, None, :]

    theta_sub = np.radians(90.0 - lat_sub)
    phi_sub = np.radians(lon_sub) + np.pi
    theta_sub, phi_sub = np.broadcast_arrays(theta_sub, phi_sub)

    maps = cube.reshape(-1, n_pix)
    vals = hp.get_interp_val(maps, theta_sub.ravel(), phi_sub.ravel(), nest=nest)
    vals = vals.reshape(n_time, n_var, nlat_int, 2 * nlat_int, nsub_int, nsub_int)
    return vals.mean(axis=(-1, -2))


def _healpix_centers(nside: int) -> Tuple[np.ndarray, np.ndarray]:
    import healpy as hp  # type: ignore

    cell = np.arange(hp.nside2npix(int(nside)))
    lon, lat = hp.pix2ang(int(nside), cell, nest=False, lonlat=True)
    return np.asarray(lat, dtype=np.float32), np.asarray(lon, dtype=np.float32)


def _sanitize_nc_attr(value: object) -> object:
    try:
        return gcm321._sanitize_attr_value(value)  # type: ignore[attr-defined]
    except Exception:
        if value is None:
            return ""
        if isinstance(value, (str, float, int, bool)):
            return value
        return str(value)


def save_native_healpix_nc(
    yhval: Union[np.ndarray, torch.Tensor],
    variables: Sequence[str],
    year: np.ndarray,
    month: np.ndarray,
    *,
    out_root: Path,
    filename_stem: str,
    nside: int,
    extra_attrs: Optional[Dict[str, object]] = None,
) -> List[Path]:
    """Write native RING HEALPix output as one per-variable NetCDF file."""
    import xarray as xr  # type: ignore

    if isinstance(yhval, torch.Tensor):
        reg = yhval.detach().cpu().numpy()
    else:
        reg = np.asarray(yhval)
    if reg.ndim != 3:
        raise ValueError(f"HEALPix output must have shape (time, variable, cell), got {reg.shape}")
    if reg.shape[1] != len(variables):
        raise ValueError(f"Variable count mismatch: data has {reg.shape[1]}, names={len(variables)}")

    sy, sm, ey, em, tcoord = gcm321.infer_time_from_year_month(year, month)
    time_range = f"{int(sy):04d}{int(sm):02d}-{int(ey):04d}{int(em):02d}"
    cell_count = int(reg.shape[2])
    cell = np.arange(cell_count, dtype=np.int64)
    cell_lat, cell_lon = _healpix_centers(nside)
    if cell_lat.shape[0] != cell_count:
        raise ValueError(
            f"HEALPix cell count mismatch: data has {cell_count}, nside={nside} has {cell_lat.shape[0]}"
        )

    common_attrs: Dict[str, object] = {
        "frequency": "mon",
        "calendar": gcm321.CAL,
        "grid": "native GCMAGICC HEALPix",
        "healpix_nside": int(nside),
        "healpix_order": "ring",
        "Conventions": "CF-1.8",
    }
    if extra_attrs:
        common_attrs.update(extra_attrs)
    sanitized_attrs = {k: _sanitize_nc_attr(v) for k, v in common_attrs.items()}

    written: List[Path] = []
    for iv, variable in enumerate(variables):
        var_name = str(variable)
        var_dir = Path(out_root) / var_name
        var_dir.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset(
            coords={
                "time": ("time", tcoord),
                "cell": ("cell", cell),
                "cell_lat": ("cell", cell_lat),
                "cell_lon": ("cell", cell_lon),
            },
            data_vars={var_name: (("time", "cell"), reg[:, iv, :])},
            attrs=sanitized_attrs,
        )
        ds["cell_lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
        ds["cell_lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
        ds[var_name].attrs["coordinates"] = "cell_lat cell_lon"
        outfile = var_dir / f"{filename_stem}_{var_name}_healpix-nside{int(nside)}_ring_{time_range}.nc"
        ds.to_netcdf(outfile)
        written.append(outfile)
    return written


# =============================================================================
# Worker functions for CPU pools (mirrors 330_* parallelisation)
# =============================================================================


def _pass1_worker(payload):
    (
        item,
        gcmagicc_path,
        models_subdir,
        meta,
        device_str,
        amp_flag,
        force_gpu,
        dependence,
        weights,
        tas_index,
        output_mode,
        runmodus,
        save_pass1,
        output_root_str,
        pass1_stage,
        create_also_healpix_output,
        healpix_nside,
        debug_trace,
    ) = payload
    import numpy as _np
    import random as _rnd
    import torch as _torch

    _rnd.seed(item.seed_base)
    _np.random.seed(item.seed_base)
    _torch.manual_seed(item.seed_base)

    key = _ensemble_key(item)
    try:
        sample_fn, models_dir, date_token = _ensure_worker_context(gcmagicc_path, models_subdir)
    except Exception as e:
        return (False, key, float("nan"), f"Worker init failed: {e}", None)

    try:
        predictors, year, month = build_predictors_from_spliced_file(
            item.predictor_path,
            meta,
            model_index_name="ERA5",
            year_start=item.predictor_year_start,
            year_end=item.predictor_year_end,
        )
    except Exception as e:
        if debug_trace:
            print("🟥 [PASS1 worker] Predictor build failure:", e)
        return (False, key, float("nan"), f"Predictor build failed: {e}", None)

    _ensure_model_index_era5(predictors)

    try:
        tas_pred_C = _compute_predictor_tas_mean_C(predictors)
    except Exception as e:
        return (False, key, float("nan"), f"tas_pred computation failed: {e}", None)

    est_bytes = _estimate_output_bytes(
        time_steps=len(year),
        nvars=len(predictors.variables_2predict),
        nlat=N_LAT,
        nlon=N_LAT * 2,
    )
    _log_memory("[PASS1 worker start]", key, est_bytes=est_bytes)

    usebias_model = 0 if item.bias_to_era5 else None
    useeffect_model = 0
    device_eff = _maybe_fallback_device(
        device_str,
        time_steps=len(year),
        nvars=len(predictors.variables_2predict),
        nlat=N_LAT,
        nlon=N_LAT * 2,
        verbose=False,
    )

    try:
        if create_also_healpix_output:
            yhval_native = _run_gcmagicc_native_healpix(
                sample_fn=sample_fn,
                predictor_data=predictors,
                dependence=dependence,
                usebias_model=usebias_model,
                useeffect_model=useeffect_model,
                device=device_eff,
                models_dir=models_dir,
                date_token=date_token,
                nside=healpix_nside,
                force_gpu=force_gpu,
                amp=amp_flag,
                seed=item.seed_base,
            )
            yhval = _healpix_to_latlon_grid(
                yhval_native,
                N_LAT,
                nsub=HEALPIX_TO_LATLON_NSUB,
            )
            del yhval_native
        else:
            yhval = run_gcmagicc(
                sample_fn=sample_fn,
                predictor_data=predictors,
                dependence=dependence,
                usebias_model=usebias_model,
                useeffect_model=useeffect_model,
                device=device_eff,
                models_dir=models_dir,
                date_token=date_token,
                force_gpu=force_gpu,
                amp=amp_flag,
                seed=item.seed_base,
            )
    except Exception as e:
        if debug_trace:
            print("🟥 [PASS1 worker] run_gcmagicc failure:", e)
        _cleanup_cuda_cache()
        return (False, key, float("nan"), f"Emulation failed: {e}", None)

    _log_memory("[PASS1 worker after run_gcmagicc]", key)

    try:
        tas_out_C = _compute_global_tas_mean_C(yhval, weights, tas_index)
    except Exception as e:
        if debug_trace:
            print("🟥 [PASS1 worker] Δ computation failure:", e)
        _cleanup_cuda_cache()
        return (False, key, float("nan"), f"Δ computation failed: {e}", None)

    delta_C = float(tas_out_C - tas_pred_C)
    
    # Compute per-year deltas (used for per-year bias correction in PASS2)
    # Also compute tas_out and tas_pred per year for plotting
    try:
        per_year_deltas, per_year_tas_out, per_year_tas_pred = _compute_per_year_deltas(
            yhval, predictors, weights, tas_index, year, return_full_data=True
        )
        per_year_deltas_str = ", ".join([f"{y}:{d:.3f}" for y, d in sorted(per_year_deltas.items())[:5]])
        if len(per_year_deltas) > 5:
            per_year_deltas_str += f" ... ({len(per_year_deltas)} years)"
        # Compute std dev of per-year deltas for diagnostics
        per_year_deltas_values = list(per_year_deltas.values())
        per_year_std = float(_np.std(per_year_deltas_values)) if len(per_year_deltas_values) > 1 else 0.0
    except Exception as e:
        per_year_deltas = {}
        per_year_tas_out = {}
        per_year_tas_pred = {}
        per_year_deltas_str = f"<error: {e}>"
        per_year_std = 0.0

    if save_pass1:
        try:
            stage_root, fname_root, _ = _format_output_paths(
                stage=pass1_stage or PASS1_TEMP_STAGE,
                output_root=Path(output_root_str),
                item=item,
                runmodus=runmodus,
                dependence=dependence,
                usebias_model=usebias_model,
                useeffect_model=useeffect_model,
                variable_names=predictors.variables_2predict,
                model_to_index=meta.get("model_to_index", {}),
            )
            if output_mode == "simple":
                outfile = stage_root / f"{fname_root}.nc"
                save_simple_nc(
                    yhval,
                    predictors.variables_2predict,
                    year,
                    month,
                    outfile,
                    nlat=N_LAT,
                    lon_convention=LON_CONVENTION,
                    calendar=gcm321.CAL,
                )
            else:
                source_id = f"MLMAGICC_{MODEL_VERSION}"
                experiment_id = item.scenario if item.scenario.startswith("ssp") else f"custom-{item.scenario}"
                member_id = f"r{int(item.ensemble_id)}i1p1{_workflow_to_f_flag(item.workflow)}"
                save_cmip6_nc(
                    yhval,
                    predictors.variables_2predict,
                    year,
                    month,
                    out_root=stage_root,
                    source_id=source_id,
                    experiment_id=experiment_id,
                    member_id=member_id,
                    nlat=N_LAT,
                    lon_convention=LON_CONVENTION,
                    model_version=MODEL_VERSION,
                )
        except Exception as e:
            if debug_trace:
                print("🟥 [PASS1 worker] write failure:", e)

    # cleanup
    del yhval
    if isinstance(predictors.X, _torch.Tensor):
        predictors.X = predictors.X.cpu()
    del predictors
    _cleanup_cuda_cache()
    _log_memory("[PASS1 worker post-cleanup]", key)

    msg = (
        f"tas_out={tas_out_C:.3f} °C, tas_pred={tas_pred_C:.3f} °C, Δ={delta_C:.3f} °C"
    )
    record = {
        "ensemble_key": key,
        "scenario": item.scenario,
        "workflow": item.workflow,
        "runmodus": item.runmodus,
        "magicc_run_id": int(item.run_id),
        "ensemble_id": int(item.ensemble_id),
        "predictor_path": str(item.predictor_path),
        "tas_out_C": float(tas_out_C),
        "tas_pred_C": float(tas_pred_C),
        "delta_C": float(delta_C),
        "per_year_deltas": per_year_deltas,  # Used for per-year bias correction in PASS2
        "per_year_tas_out": per_year_tas_out,  # For plotting/resume
        "per_year_tas_pred": per_year_tas_pred,  # For plotting/resume
        "per_year_deltas_str": per_year_deltas_str,
        "per_year_std": per_year_std,  # Std dev of per-year deltas
    }
    return (True, key, delta_C, msg, record)


def _pass2_worker(payload):
    (
        item,
        delta_C,
        per_year_deltas,
        gcmagicc_path,
        models_subdir,
        meta,
        device_str,
        amp_flag,
        force_gpu,
        dependence,
        weights,
        tas_index,
        output_mode,
        runmodus,
        output_root_str,
        create_also_healpix_output,
        healpix_nside,
        debug_trace,
    ) = payload
    import numpy as _np
    import random as _rnd
    import torch as _torch

    _rnd.seed(item.seed_base)
    _np.random.seed(item.seed_base)
    _torch.manual_seed(item.seed_base)

    key = _ensemble_key(item)
    try:
        sample_fn, models_dir, date_token = _ensure_worker_context(gcmagicc_path, models_subdir)
    except Exception as e:
        return (False, key, f"Worker init failed: {e}", None)

    try:
        predictors, year, month = build_predictors_from_spliced_file(
            item.predictor_path,
            meta,
            model_index_name="ERA5",
            year_start=item.predictor_year_start,
            year_end=item.predictor_year_end,
        )
    except Exception as e:
        if debug_trace:
            print("🟥 [PASS2 worker] Predictor build failure:", e)
        return (False, key, f"Predictor build failed: {e}", None)

    _ensure_model_index_era5(predictors)

    try:
        tas_pred_pass1 = _compute_predictor_tas_mean_C(predictors)
    except Exception as e:
        return (False, key, f"tas_pred_pass1 computation failed: {e}", None)

    est_bytes = _estimate_output_bytes(
        time_steps=len(year),
        nvars=len(predictors.variables_2predict),
        nlat=N_LAT,
        nlon=N_LAT * 2,
    )
    _log_memory("[PASS2 worker start]", key, est_bytes=est_bytes)

    # Apply per-year offset (required - per_year_deltas must be available)
    if per_year_deltas is None:
        raise RuntimeError(f"Per-year deltas are required but None for ensemble {key}")
    delta_units = _apply_tas_offset(
        predictors,
        delta_C,
        year=year,
        per_year_deltas=per_year_deltas,
        use_per_year=True,  # Per-year deltas are required
    )
    _ensure_model_index_era5(predictors)
    try:
        tas_pred_pass2 = _compute_predictor_tas_mean_C(predictors)
    except Exception as e:
        return (False, key, f"tas_pred_pass2 computation failed: {e}", None)

    usebias_model = 0 if item.bias_to_era5 else None
    useeffect_model = 0
    device_eff = _maybe_fallback_device(
        device_str,
        time_steps=len(year),
        nvars=len(predictors.variables_2predict),
        nlat=N_LAT,
        nlon=N_LAT * 2,
        verbose=False,
    )

    yhval_native = None
    try:
        if create_also_healpix_output:
            yhval_native = _run_gcmagicc_native_healpix(
                sample_fn=sample_fn,
                predictor_data=predictors,
                dependence=dependence,
                usebias_model=usebias_model,
                useeffect_model=useeffect_model,
                device=device_eff,
                models_dir=models_dir,
                date_token=date_token,
                nside=healpix_nside,
                force_gpu=force_gpu,
                amp=amp_flag,
                seed=item.seed_base,
            )
            yhval = _healpix_to_latlon_grid(
                yhval_native,
                N_LAT,
                nsub=HEALPIX_TO_LATLON_NSUB,
            )
        else:
            yhval = run_gcmagicc(
                sample_fn=sample_fn,
                predictor_data=predictors,
                dependence=dependence,
                usebias_model=usebias_model,
                useeffect_model=useeffect_model,
                device=device_eff,
                models_dir=models_dir,
                date_token=date_token,
                force_gpu=force_gpu,
                amp=amp_flag,
                seed=item.seed_base,
            )
    except Exception as e:
        if debug_trace:
            print("🟥 [PASS2 worker] run_gcmagicc failure:", e)
        _cleanup_cuda_cache()
        return (False, key, f"Emulation failed: {e}", None)

    _log_memory("[PASS2 worker after run_gcmagicc]", key)

    try:
        tas_out_C = _compute_global_tas_mean_C(yhval, weights, tas_index)
    except Exception as e:
        if debug_trace:
            print("🟥 [PASS2 worker] tas_out computation failure:", e)
        _cleanup_cuda_cache()
        return (False, key, f"tas_out computation failed: {e}", None)

    delta_out_vs_pass1 = float(tas_out_C - tas_pred_pass1)
    delta_out_vs_pass2 = float(tas_out_C - tas_pred_pass2)
    offset_type = "per-year" if per_year_deltas is not None else "constant"

    f_flag = _workflow_to_f_flag(item.workflow)
    member_id = f"r{int(item.ensemble_id)}i1p1{f_flag}"
    extra_attrs = _build_pass2_attrs(
        item,
        tas_pred_pass1=tas_pred_pass1,
        tas_pred_pass2=tas_pred_pass2,
        tas_out=tas_out_C,
        delta_vs_pass1=delta_out_vs_pass1,
        delta_vs_pass2=delta_out_vs_pass2,
        debias_delta=delta_C,
        debias_units=delta_units,
        offset_type=offset_type,
    )

    stage_root, fname_root, _ = _format_output_paths(
        stage="debias",
        output_root=Path(output_root_str),
        item=item,
        runmodus=runmodus,
        dependence=dependence,
        usebias_model=usebias_model,
        useeffect_model=useeffect_model,
        variable_names=predictors.variables_2predict,
        model_to_index=meta.get("model_to_index", {}),
    )
    try:
        healpix_written: List[Path] = []
        if create_also_healpix_output and yhval_native is not None:
            healpix_stage_root, healpix_fname_root, _ = _format_output_paths(
                stage="debias",
                output_root=Path(output_root_str),
                item=item,
                runmodus=runmodus,
                dependence=dependence,
                usebias_model=usebias_model,
                useeffect_model=useeffect_model,
                variable_names=predictors.variables_2predict,
                model_to_index=meta.get("model_to_index", {}),
                canonical_kind_override=CANONICAL_KIND_ORIGINAL_HEALPIX,
            )
            healpix_written = save_native_healpix_nc(
                yhval_native,
                predictors.variables_2predict,
                year,
                month,
                out_root=healpix_stage_root,
                filename_stem=healpix_fname_root,
                nside=healpix_nside,
                extra_attrs=extra_attrs,
            )
        if output_mode == "simple":
            outfile = stage_root / f"{fname_root}.nc"
            save_simple_nc(
                yhval,
                predictors.variables_2predict,
                year,
                month,
                outfile,
                nlat=N_LAT,
                lon_convention=LON_CONVENTION,
                calendar=gcm321.CAL,
                extra_attrs=extra_attrs,
            )
        else:
            source_id = f"MLMAGICC_{MODEL_VERSION}"
            experiment_id = item.scenario if item.scenario.startswith("ssp") else f"custom-{item.scenario}"
            save_cmip6_nc(
                yhval,
                predictors.variables_2predict,
                year,
                month,
                out_root=stage_root,
                source_id=source_id,
                experiment_id=experiment_id,
                member_id=member_id,
                nlat=N_LAT,
                lon_convention=LON_CONVENTION,
                model_version=MODEL_VERSION,
                extra_attrs=extra_attrs,
            )
    except Exception as e:
        if debug_trace:
            print("🟥 [PASS2 worker] write failure:", e)
        _cleanup_cuda_cache()
        return (False, key, f"write failed: {e}", None)

    if yhval_native is not None:
        del yhval_native
    del yhval
    if isinstance(predictors.X, _torch.Tensor):
        predictors.X = predictors.X.cpu()
    del predictors
    _cleanup_cuda_cache()
    _log_memory("[PASS2 worker post-cleanup]", key)

    offset_type = "per-year" if per_year_deltas is not None else "constant"
    if per_year_deltas and len(per_year_deltas) > 0:
        per_year_values = list(per_year_deltas.values())
        per_year_min = float(np.min(per_year_values))
        per_year_max = float(np.max(per_year_values))
        per_year_mean = float(np.mean(per_year_values))
        per_year_std = float(np.std(per_year_values))
        per_year_info = f" (Δ range: [{per_year_min:.3f}, {per_year_max:.3f}] °C, mean={per_year_mean:.3f}±{per_year_std:.3f}, n={len(per_year_deltas)} years)"
    else:
        per_year_info = ""
    healpix_info = (
        f" | healpix_nside={int(healpix_nside)} files={len(healpix_written)}"
        if create_also_healpix_output
        else ""
    )
    msg = (
        f"{offset_type} offset{per_year_info} → {stage_root} | "
        f"tas_out={tas_out_C:.3f} °C, tas_pred_pass1={tas_pred_pass1:.3f} °C, "
        f"tas_pred_pass2={tas_pred_pass2:.3f} °C, Δ_vs_pass1={delta_out_vs_pass1:.3f} °C, "
        f"Δ_vs_pass2={delta_out_vs_pass2:.3f} °C, Δ_units_mean={delta_units:.4f}"
        f"{healpix_info}"
    )
    record = {
        "ensemble_key": key,
        "scenario": item.scenario,
        "workflow": item.workflow,
        "runmodus": item.runmodus,
        "magicc_run_id": int(item.run_id),
        "ensemble_id": int(item.ensemble_id),
        "predictor_path": str(item.predictor_path),
        "tas_pred_pass1_C": float(tas_pred_pass1),
        "tas_pred_pass2_C": float(tas_pred_pass2),
        "tas_out_C": float(tas_out_C),
        "delta_vs_pass1_C": float(delta_out_vs_pass1),
        "delta_vs_pass2_C": float(delta_out_vs_pass2),
        "debias_delta_C": float(delta_C),
        "debias_units": float(delta_units),
        "offset_type": offset_type,
        "create_also_healpix_output": bool(create_also_healpix_output),
        "healpix_nside": int(healpix_nside) if create_also_healpix_output else None,
        "healpix_output_files": [str(p) for p in healpix_written] if create_also_healpix_output else [],
    }
    return (True, key, msg, record)


def _pass1_pass2_combined_worker(payload):
    """
    Run PASS1→PASS2 sequentially for one ensemble. Designed for CPU pools.
    """
    item, existing_pass1, cfg = payload
    key = _ensemble_key(item)
    pass1_record = existing_pass1
    pass1_msg = ""

    if pass1_record is None:
        payload1 = (
            item,
            cfg["gcmagicc_path"],
            cfg["models_subdir"],
            cfg["meta"],
            cfg["device_str"],
            cfg["amp_flag"],
            cfg["force_gpu"],
            cfg["dependence"],
            cfg["weights"],
            cfg["tas_index"],
            cfg["output_mode"],
            item.runmodus,
            cfg["save_pass1"],
            cfg["output_root_str"],
            cfg["pass1_stage"],
            cfg["create_also_healpix_output"],
            cfg["healpix_nside"],
            cfg["debug_trace"],
        )
        success1, _, delta_C1, msg1, record1 = _pass1_worker(payload1)
        pass1_msg = msg1 or ""
        if not success1 or record1 is None or not math.isfinite(delta_C1):
            return {
                "key": key,
                "status": "pass1_failed",
                "scenario": item.scenario,
                "workflow": item.workflow,
                "runmodus": item.runmodus,
                "ensemble_id": int(item.ensemble_id),
                "run_id": int(item.run_id),
                "pass1_record": record1,
                "pass2_record": None,
                "message": msg1 or "PASS1 failed",
                "pass1_msg": pass1_msg,
                "pass2_msg": "",
            }
        pass1_record = record1
    else:
        pass1_msg = "PASS1 cached"

    per_year_deltas = _coerce_year_dict(pass1_record.get("per_year_deltas", {})) if pass1_record else {}
    if not per_year_deltas:
        return {
            "key": key,
            "status": "pass1_failed",
            "scenario": item.scenario,
            "workflow": item.workflow,
            "runmodus": item.runmodus,
            "ensemble_id": int(item.ensemble_id),
            "run_id": int(item.run_id),
            "pass1_record": pass1_record,
            "pass2_record": None,
            "message": "Missing per-year deltas; cannot run PASS2.",
            "pass1_msg": pass1_msg,
            "pass2_msg": "",
        }

    delta_C_use = float(pass1_record.get("delta_C", np.mean(list(per_year_deltas.values()))))
    payload2 = (
        item,
        delta_C_use,
        per_year_deltas,
        cfg["gcmagicc_path"],
        cfg["models_subdir"],
        cfg["meta"],
        cfg["device_str"],
        cfg["amp_flag"],
        cfg["force_gpu"],
        cfg["dependence"],
        cfg["weights"],
        cfg["tas_index"],
        cfg["output_mode"],
        item.runmodus,
        cfg["output_root_str"],
        cfg["create_also_healpix_output"],
        cfg["healpix_nside"],
        cfg["debug_trace"],
    )
    success2, _, info2, record2 = _pass2_worker(payload2)
    status = "completed" if success2 else "pass2_failed"
    msg = info2 if isinstance(info2, str) else str(info2)
    return {
        "key": key,
        "status": status,
        "scenario": item.scenario,
        "workflow": item.workflow,
        "runmodus": item.runmodus,
        "ensemble_id": int(item.ensemble_id),
        "run_id": int(item.run_id),
        "pass1_record": pass1_record,
        "pass2_record": record2 if success2 else None,
        "message": msg,
        "pass1_msg": pass1_msg,
        "pass2_msg": info2 if success2 else "",
    }


# =============================================================================
# Worker context helpers for multiprocessing
# =============================================================================

_WORKER_SAMPLE_FN = None
_WORKER_MODELS_DIR = None
_WORKER_DATE_TOKEN = None


def _ensure_worker_context(gcmagicc_path: str, models_subdir: str):
    """
    Lazily load run_general sampler and model paths inside worker processes.
    """
    global _WORKER_SAMPLE_FN, _WORKER_MODELS_DIR, _WORKER_DATE_TOKEN
    if _WORKER_SAMPLE_FN is not None:
        return _WORKER_SAMPLE_FN, _WORKER_MODELS_DIR, _WORKER_DATE_TOKEN
    run_general_path = _resolve_run_general(gcmagicc_path)
    _WORKER_SAMPLE_FN = _load_run_general_sampler(gcmagicc_path)
    _WORKER_DATE_TOKEN = _infer_date_token(_WORKER_SAMPLE_FN)
    _WORKER_MODELS_DIR = run_general_path.parent / models_subdir
    return _WORKER_SAMPLE_FN, _WORKER_MODELS_DIR, _WORKER_DATE_TOKEN


# =============================================================================
# Main debias-loop driver (MAGICCxERA5 ensembles, full length)
# =============================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _CANONICAL_LAYOUT_ACTIVE
    global _CANONICAL_EXPERIMENT_ID_OVERRIDE
    global _CANONICAL_N_ENSEMBLE_LABEL
    global _CANONICAL_RUN_INSTANCE

    parser = argparse.ArgumentParser(
        description=(
            "Debias-loop ensembles driver for GCMagicc based on 321_*, "
            "implemented for source-x=MAGICCxERA5 (Option D)."
        )
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Comma-separated list of scenarios (e.g. 'ssp245,ssp119'). Defaults to SCENARIOS from 321_*.",
    )
    parser.add_argument(
        "--ensembles",
        type=str,
        default=None,
        help="Number of draws (int) or 'all'/'first'. Default: ENSEMBLES_D from 321_*.",
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        help="Workflow selection for ERA5-spliced predictors. Use AR6, AR7, 'all', or comma list.",
    )
    parser.add_argument(
        "--runmodus",
        type=str,
        default=None,
        help="Runmodus selection for ERA5-spliced predictors. Use all, natural, aerosol, anthropogenic, or comma list.",
    )
    parser.add_argument(
        "--era5-spliced-dir",
        type=str,
        default=None,
        help="Base directory for magicc_based_predictors_* (default from 321_*).",
    )
    parser.add_argument(
        "--spliced-n",
        type=int,
        default=None,
        help="n for n_* in magicc_based_predictors (default from 321_*).",
    )
    parser.add_argument(
        "--spliced-variant",
        type=str,
        default=None,
        help="Specific magicc_based_predictors_* folder name or full path (default: auto-pick).",
    )
    parser.add_argument(
        "--spliced-variant-glob",
        type=str,
        default=None,
        help="Glob for magicc_based_predictors_* folders (default from 321_*).",
    )
    parser.add_argument(
        "--bias-to-era5",
        action="store_true",
        help="Force bias correction to ERA5 (usebias_model=0). Default: BIASCORRECT_TO_ERA5_D from 321_*.",
    )
    parser.add_argument(
        "--no-bias-to-era5",
        action="store_true",
        help="Disable bias correction to ERA5 (usebias_model=None).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device, e.g. 'cuda:0', 'cpu'. Default: gcm321.detect_default_device().",
    )
    parser.add_argument(
        "--force-gpu",
        action="store_true",
        help="Force GPU usage even if torch.cuda.is_available() reports False.",
    )
    parser.add_argument("--amp", action="store_true", help="Enable AMP autocast on CUDA.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Base folder for outputs. In canonical mode this should be the ERA5spliced root; "
            "in legacy mode a timestamped debiasloop folder is created under this base."
        ),
    )
    parser.add_argument(
        "--canonical-layout",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Write PASS2 outputs to canonical ERA5spliced layout "
            "(<version>/<experiment>/<ARX>/<runmodus>/<n_ensemble>/original) and "
            "place PASS1/run artifacts under .../dataderivatives/run_artifacts."
        ),
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Canonical experiment id override. Default resolves from scenario/runmodus.",
    )
    parser.add_argument(
        "--n-ensemble-label",
        type=str,
        default=None,
        help="Canonical n_ensemble label (e.g., n_20, n_100). Default derives from --spliced-n.",
    )
    parser.add_argument(
        "--run-instance",
        type=str,
        default=None,
        help="Optional run-instance suffix under canonical kind.",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        choices=["simple", "cmip6"],
        default=None,
        help="Output layout (simple NetCDF per run or CMIP6-style). Default from 321_*.",
    )
    parser.add_argument(
        "--create-also-healpix-output",
        dest="create_also_healpix_output",
        action="store_true",
        default=None,
        help="Also save PASS2 native HEALPix output beside canonical lat/lon output.",
    )
    parser.add_argument(
        "--no-create-also-healpix-output",
        dest="create_also_healpix_output",
        action="store_false",
        help="Disable native HEALPix sidecar output.",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=DEFAULT_HEALPIX_NSIDE,
        help="Native HEALPix nside for --create-also-healpix-output (default: 64).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover tasks and planned output roots, then exit before model execution or file writes.",
    )
    parser.add_argument(
        "--auto-consolidate",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUTO_CONSOLIDATE,
        help="Run scoped 2018 autoconsolidate after PASS2 outputs are written (default: enabled).",
    )
    parser.add_argument(
        "--auto-consolidate-config",
        type=str,
        default=None,
        help="Path to 2018 consolidate config JSON. Default: gcmmagicc/scripts/2018_consolidate_era5spliced_s3.example.json",
    )
    parser.add_argument(
        "--auto-consolidate-cleanup-local",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUTO_CONSOLIDATE_CLEANUP_LOCAL,
        help="Allow local cleanup after verified upload in autoconsolidate (default: enabled).",
    )
    parser.add_argument(
        "--save-pass1",
        action="store_true",
        help=(
            "Also write PASS1 (pre-debias) outputs to a temporary stage folder "
            f"('{PASS1_TEMP_STAGE}', deleted after PASS2 unless DELETE_TEMPFOLDER_PASS1 is False)."
        ),
    )
    parser.add_argument(
        "--delta-csv",
        type=str,
        default=None,
        help="Optional CSV path for saving ensemble Δ offsets. Default: alongside output-root.",
    )
    parser.add_argument(
        "--delta-plot",
        type=str,
        default=None,
        help="Optional PNG path for histogram of Δ offsets. Default: alongside output-root.",
    )
    parser.add_argument(
        "--progress-file",
        type=str,
        default=None,
        help="Path to JSON checkpoint for PASS1/PASS2 progress (default: <output_root>/debias_progress.json).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing progress JSON: reuse completed PASS1 results and skip finished runs.",
    )
    parser.add_argument(
        "--test-one",
        action="store_true",
        help="Run only one ensemble per scenario (for testing). Default: TEST_ONE from 321_*.",
    )
    parser.add_argument(
        "--debug-trace",
        action="store_true",
        help="Print detailed tracebacks/context on failures.",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=None,
        help="Force CPU worker count (override auto planner). Use 1 to disable CPU pool.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Number of deterministic shards across hosts/process groups (default: 1).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based index of this shard (must be < --shard-count).",
    )
    parser.add_argument(
        "--shard-strategy",
        type=str,
        choices=["keyhash"],
        default="keyhash",
        help="Task sharding strategy (default: keyhash).",
    )
    parser.add_argument(
        "--shard-id",
        type=str,
        default=None,
        help="Optional human-readable shard identifier stored in progress metadata.",
    )
    parser.add_argument(
        "--work-manifest-in",
        type=str,
        default=None,
        help="Read canonical work manifest JSON instead of building tasks from predictors.",
    )
    parser.add_argument(
        "--work-manifest-out",
        type=str,
        default=None,
        help="Write canonical work manifest JSON for this run before sharding.",
    )

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Configuration and sanity checks
    # ------------------------------------------------------------------
    source_mode = "MAGICCxERA5"
    if args.shard_count < 1:
        raise RuntimeError(f"--shard-count must be >= 1, got {args.shard_count}")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise RuntimeError(
            f"--shard-index must be in [0, {args.shard_count - 1}], got {args.shard_index}"
        )
    create_also_healpix_output = (
        DEFAULT_CREATE_ALSO_HEALPIX_OUTPUT
        if args.create_also_healpix_output is None
        else bool(args.create_also_healpix_output)
    )
    healpix_nside = _validate_healpix_nside(args.nside)

    # RNG
    base_seed = args.seed if args.seed is not None else 0
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    # Make sure we request expandable_segments before any CUDA allocations
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Device
    if args.device is None:
        device_str = detect_default_device(force_gpu=args.force_gpu)
    else:
        device_str = normalize_device_string(args.device)
        if device_str is None:
            device_str = detect_default_device(force_gpu=args.force_gpu)
    amp_flag = bool(args.amp or (DEFAULT_AMP and isinstance(device_str, str) and device_str.startswith("cuda")))
    print(f"🖥️  Using device: {device_str}")
    report_device_status(device_str)

    # Meta, models, sampler
    run_general_path = _resolve_run_general(GCMAGICC_PATH)
    sample_fn = _load_run_general_sampler(GCMAGICC_PATH)
    date_token = _infer_date_token(sample_fn)
    models_dir = run_general_path.parent / MODELS_SUBDIR
    meta, meta_path, DATEOFMETAFILE = load_meta(GCMAGICC_PATH, MODELS_SUBDIR)
    model_to_index: Dict[str, int] = meta.get("model_to_index", {})
    variables_2predict: List[str] = list(meta.get("variables", []))
    variables_X: List[str] = list(meta.get("variables_X", []))
    # Cache lookups for filename building inside workers.
    global _MODEL_INDEX_TO_NAME, _DEFAULT_VARLIST
    _MODEL_INDEX_TO_NAME = {}
    for name, idx in model_to_index.items():
        try:
            _MODEL_INDEX_TO_NAME[int(idx)] = str(name)
        except Exception:
            continue
    _DEFAULT_VARLIST = list(variables_2predict)
    if "tas_smoothed" not in variables_X:
        print("⚠️  Warning: 'tas_smoothed' not found in META['variables_X']; debias offset might do nothing.")

    # Grid / weights
    lats, lons = generate_coordinate_grids(
        nlat=N_LAT, nlon=N_LAT * 2, lon_convention=LON_CONVENTION, lat_direction="north_to_south"
    )
    weights = _area_weights_coslat(lats, lons)
    try:
        tas_index = variables_2predict.index("tas")
    except ValueError:
        raise RuntimeError(f"'tas' not found in META['variables']: {variables_2predict}")

    nside_effective = healpix_nside if create_also_healpix_output else _infer_default_nside(sample_fn)

    # Scenario discovery (allow auto-discovery from predictors)
    if args.scenarios:
        scen_list = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    else:
        scen_list = list(getattr(gcm321, "SCENARIOS", []) or [])
        if not scen_list:
            scen_list = None

    workflow = args.workflow if args.workflow is not None else USE_WORKFLOW
    runmodus = args.runmodus if args.runmodus is not None else USE_RUNMODUSE
    spliced_root = Path(args.era5_spliced_dir) if args.era5_spliced_dir else Path(ERA5_SPLICED_PREDICTOR_DIR)
    spliced_n = args.spliced_n if args.spliced_n is not None else SPLICED_N
    spliced_variant = args.spliced_variant if args.spliced_variant is not None else (SPLICED_VARIANT or None)
    spliced_variant_glob = (
        args.spliced_variant_glob if args.spliced_variant_glob is not None else SPLICED_VARIANT_GLOB
    )
    canonical_layout_active = (
        DEFAULT_CANONICAL_LAYOUT if args.canonical_layout is None else bool(args.canonical_layout)
    )
    n_ensemble_label_raw = args.n_ensemble_label if args.n_ensemble_label is not None else f"n_{spliced_n}"
    try:
        canonical_n_ensemble = normalize_n_ensemble_label(n_ensemble_label_raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid --n-ensemble-label '{n_ensemble_label_raw}': {exc}") from exc
    _CANONICAL_LAYOUT_ACTIVE = canonical_layout_active
    _CANONICAL_EXPERIMENT_ID_OVERRIDE = (
        str(args.experiment_id).strip() if args.experiment_id is not None and str(args.experiment_id).strip() else None
    )
    _CANONICAL_N_ENSEMBLE_LABEL = canonical_n_ensemble
    _CANONICAL_RUN_INSTANCE = (
        str(args.run_instance).strip() if args.run_instance is not None and str(args.run_instance).strip() else None
    )
    try:
        workflows = normalize_workflow_list(workflow)
        runmodus_list = normalize_runmodus_list(runmodus)
    except ValueError as exc:
        raise RuntimeError(f"Invalid workflow/runmodus selection: {exc}")
    draws_spec: Iterable[str | int] | str | int
    draws_raw = args.ensembles if args.ensembles is not None else ENSEMBLES_D
    try:
        draws_spec = int(draws_raw) if str(draws_raw).isdigit() else draws_raw
    except Exception:
        draws_spec = draws_raw

    if scen_list is None:
        scen_list = discover_spliced_scenarios(
            spliced_root,
            workflows=workflows,
            runmodus_list=runmodus_list,
            n_value=spliced_n,
            variant=spliced_variant,
            variant_glob=spliced_variant_glob,
        )
        if not scen_list:
            raise RuntimeError(
                f"No scenarios found under {spliced_root} for workflows={workflows} and runmodus={runmodus_list}."
            )

    bias_to_era5 = BIASCORRECT_TO_ERA5_D
    if args.bias_to_era5:
        bias_to_era5 = True
    if args.no_bias_to_era5:
        bias_to_era5 = False

    work_items: List[DebiasWorkItem] = []
    if args.work_manifest_in:
        manifest_in_path = Path(args.work_manifest_in).resolve()
        if not manifest_in_path.exists():
            raise FileNotFoundError(f"--work-manifest-in file does not exist: {manifest_in_path}")
        work_items = _load_work_manifest(
            manifest_in_path,
            base_seed=base_seed,
            bias_to_era5=bias_to_era5,
        )
        print(f"📄 Loaded work manifest from {manifest_in_path} ({len(work_items)} items).")
    else:
        # Build work items using 321_*'s helper for MAGICCxERA5 draws
        strict_lookup = len(workflows) == 1 and len(runmodus_list) == 1 and bool(scen_list)
        test_one = args.test_one or TEST_ONE
        tasks = build_spliced_tasks(
            spliced_root,
            scen_list,
            draws_spec,
            test_one=test_one,
            workflow=workflows,
            runmodus=runmodus_list,
            n_value=spliced_n,
            variant=spliced_variant,
            variant_glob=spliced_variant_glob,
            strict=strict_lookup,
        )
        if not tasks:
            print("❌ No MAGICCxERA5 tasks to run (check predictors/draws).")
            return 1

        if test_one:
            print(f"🧪 TEST_ONE mode enabled: running only one ensemble per scenario (total tasks: {len(tasks)})")

        for t in tasks:
            scenario = str(t.get("scenario"))
            ensemble_id = int(t.get("ensemble_id"))
            run_id = int(t.get("run_id"))
            predictor_path = Path(str(t.get("predictor_path")))
            runmodus_val = str(t.get("runmodus") or (runmodus_list[0] if runmodus_list else USE_RUNMODUSE))
            workflow_val = str(t.get("workflow") or (workflows[0] if workflows else USE_WORKFLOW))
            seed_base = _stable_seed_base(
                base_seed=base_seed,
                scenario=scenario,
                workflow=workflow_val,
                runmodus=runmodus_val,
                run_id=run_id,
                ensemble_id=ensemble_id,
            )
            predictor_source = t.get("predictor_source_scenario") or None
            predictor_year_start = t.get("predictor_year_start")
            predictor_year_end = t.get("predictor_year_end")
            work_items.append(
                DebiasWorkItem(
                    scenario=scenario,
                    ensemble_id=ensemble_id,
                    run_id=run_id,
                    predictor_path=predictor_path,
                    seed_base=int(seed_base),
                    bias_to_era5=bias_to_era5,
                    runmodus=runmodus_val,
                    workflow=workflow_val,
                    predictor_source=predictor_source,
                    predictor_year_start=predictor_year_start,
                    predictor_year_end=predictor_year_end,
                )
            )

    if not work_items:
        print("No work items generated; exiting.")
        return 0

    print(
        f"📋 Debias work items: {len(work_items)} "
        f"(scenarios={len(set(w.scenario for w in work_items))}, "
        f"workflows={len(set(w.workflow for w in work_items))}, "
        f"runmodus={len(set(w.runmodus for w in work_items))}, "
        f"draws/scenario+runmodus ≈ {len(work_items) // max(1, len(set((w.scenario, w.workflow, w.runmodus) for w in work_items)))})."
    )

    # Outputs
    if args.output_root is None:
        if args.dry_run:
            output_base = Path(OUTPUT_ROOT_DEFAULT).expanduser().resolve(strict=False)
        else:
            output_base = resolve_writable_output_root(
                OUTPUT_ROOT_DEFAULT,
                context="331 output base",
                allow_fallback=True,
            )
        if _CANONICAL_LAYOUT_ACTIVE:
            output_root = output_base
            print(f"📁 Canonical output root: {output_root}")
        else:
            output_root = output_base / f"{DEBIAS_OUTPUT_PREFIX}_{_today_stamp()}"
            print(f"📁 Auto-generated output root: {output_root}")
    else:
        output_root = Path(args.output_root).expanduser().resolve(strict=False)

    output_mode = args.output_mode or OUTPUT_MODE_DEFAULT
    if not args.dry_run:
        output_root = resolve_writable_output_root(
            output_root,
            context="331 debias outputs",
            allow_fallback=False,
        )
        output_root.mkdir(parents=True, exist_ok=True)

    if _CANONICAL_LAYOUT_ACTIVE:
        artifacts_root = output_root / "_debiasloop_reports" / f"{DEBIAS_OUTPUT_PREFIX}_{_today_stamp()}"
        if not args.dry_run:
            artifacts_root.mkdir(parents=True, exist_ok=True)
    else:
        artifacts_root = output_root

    work_items_sorted_all = sorted(
        work_items,
        key=lambda w: (w.scenario, w.workflow, w.runmodus, int(w.run_id), int(w.ensemble_id)),
    )

    manifest_out_path = Path(args.work_manifest_out).resolve() if args.work_manifest_out else None
    if manifest_out_path is None and args.shard_count > 1 and not args.work_manifest_in:
        manifest_out_path = output_root / "work_manifest.json"
    if manifest_out_path is not None:
        if args.dry_run:
            print(f"🧾 Dry run: work manifest would be written to {manifest_out_path} ({len(work_items_sorted_all)} items).")
        else:
            _save_work_manifest(work_items_sorted_all, manifest_out_path)
            print(f"🧾 Wrote work manifest to {manifest_out_path} ({len(work_items_sorted_all)} items).")

    shard_runtime_id = args.shard_id or (
        f"s{args.shard_index}of{args.shard_count}" if args.shard_count > 1 else "single"
    )
    host_name = os.environ.get("HOSTNAME", "") or socket.gethostname()
    if args.shard_count > 1:
        work_items_selected = [
            w
            for w in work_items_sorted_all
            if _in_selected_shard(
                _ensemble_key(w),
                shard_count=args.shard_count,
                shard_index=args.shard_index,
                shard_strategy=args.shard_strategy,
            )
        ]
        print(
            f"🧩 Sharding enabled: shard {args.shard_index}/{args.shard_count} "
            f"(strategy={args.shard_strategy}, shard_id={shard_runtime_id}, host={host_name})"
        )
        _log_work_item_breakdown(
            work_items_selected,
            prefix=f"🧮 Shard {args.shard_index}/{args.shard_count} workload",
        )
        if not work_items_selected:
            print("✅ No work items selected for this shard. Exiting cleanly.")
            return 0
    else:
        work_items_selected = list(work_items_sorted_all)
    total_runs = len(work_items_selected)

    if args.dry_run:
        _print_dry_run_summary(
            work_items_all=work_items_sorted_all,
            work_items_selected=work_items_selected,
            output_root=output_root,
            canonical_layout=_CANONICAL_LAYOUT_ACTIVE,
            canonical_n_ensemble=canonical_n_ensemble,
            create_also_healpix_output=create_also_healpix_output,
            healpix_nside=healpix_nside,
            output_mode=output_mode,
            manifest_out_path=manifest_out_path,
            shard_runtime_id=shard_runtime_id,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )
        return 0

    if args.delta_csv is None:
        delta_csv_path = artifacts_root / "debias_offsets.csv"
    else:
        delta_csv_path = Path(args.delta_csv).resolve()

    if args.delta_plot is None:
        delta_plot_path = artifacts_root / "debias_offsets_hist.png"
    else:
        delta_plot_path = Path(args.delta_plot).resolve()

    pass1_stage_name = PASS1_TEMP_STAGE
    pass1_temp_stage_root = output_root / pass1_stage_name
    if args.save_pass1:
        delete_note = "will be deleted after PASS2" if DELETE_TEMPFOLDER_PASS1 else "kept after PASS2"
        print(f"📂 PASS1 outputs: {pass1_temp_stage_root} ({delete_note})")

    # CPU parallelisation (mirrors 330_* behaviour: only for CPU device)
    is_cpu_device = str(device_str).lower() == "cpu"
    cpu_workers = 1
    if is_cpu_device:
        per_job_ram_gb = _estimate_per_job_ram_from_nside(nside_effective)
        cpu_workers = _plan_cpu_workers(per_job_ram_gb, requested=args.cpu_workers)
        print(
            f"🧠 CPU planner (mem-aware): up to {cpu_workers} worker(s) "
            f"(~{per_job_ram_gb:.0f}GB/job for nside={nside_effective}, "
            f"available_sys={_fmt_gb(_system_memory_gb()[1])})"
        )
        if cpu_workers > 1:
            print(
                "   PASS1→PASS2 stays sequential inside each run; orchestration can parallelise across ensembles."
            )

    # ------------------------------------------------------------------
    # PASS 1 + PASS 2 combined per run (resume-aware, optional CPU parallelism)
    # ------------------------------------------------------------------
    delta_records_map: Dict[str, Dict[str, object]] = {}
    pass2_records_map: Dict[str, Dict[str, object]] = {}
    debug_trace = bool(args.debug_trace)
    work_items_sorted = list(work_items_selected)

    # Progress / resume handling
    if args.progress_file:
        progress_path = Path(args.progress_file).resolve()
    else:
        if args.shard_count > 1:
            progress_path = artifacts_root / f"debias_progress.s{args.shard_index}of{args.shard_count}.json"
        else:
            progress_path = artifacts_root / "debias_progress.json"
    print(
        f"🗂️  Progress checkpoint file: {progress_path} "
        f"(shard_id={shard_runtime_id}, host={host_name})"
    )
    progress_data: Dict[str, Dict[str, object]] = {}
    if progress_path.exists():
        if args.resume:
            progress_data = _load_progress_json(progress_path)
            print(f"🔁 Loaded progress checkpoint from {progress_path} ({len(progress_data)} entries).")
        else:
            print(f"⚠️  Progress file {progress_path} exists but --resume not set; starting fresh and overwriting later.")

    worker_cfg = {
        "gcmagicc_path": GCMAGICC_PATH,
        "models_subdir": MODELS_SUBDIR,
        "meta": meta,
        "device_str": device_str,
        "amp_flag": amp_flag,
        "force_gpu": args.force_gpu,
        "dependence": DEPENDENCE,
        "weights": weights,
        "tas_index": tas_index,
        "output_mode": output_mode,
        "save_pass1": bool(args.save_pass1),
        "output_root_str": str(output_root),
        "pass1_stage": pass1_stage_name,
        "create_also_healpix_output": bool(create_also_healpix_output),
        "healpix_nside": int(healpix_nside),
        "debug_trace": debug_trace,
    }

    pending_payloads: List[tuple[DebiasWorkItem, Optional[Dict[str, object]], Dict[str, object]]] = []
    for idx, item in enumerate(work_items_sorted, start=1):
        key = _ensemble_key(item)
        existing_entry = progress_data.get(key) if args.resume else None
        pass1_record = None
        if existing_entry:
            status = existing_entry.get("status")
            stored_p1 = existing_entry.get("pass1")
            stored_p2 = existing_entry.get("pass2")
            if isinstance(stored_p1, dict):
                pass1_record = stored_p1
                delta_records_map[key] = stored_p1
            if status == "completed" and isinstance(stored_p2, dict):
                pass2_records_map[key] = stored_p2
                print(f"[RESUME {idx}/{total_runs}] {key} already completed; skipping.")
                continue
            if status == "pass1_done" and pass1_record:
                print(f"[RESUME {idx}/{total_runs}] {key} PASS1 cached; reusing per-year deltas for PASS2.")
        pending_payloads.append((item, pass1_record, worker_cfg))

    if is_cpu_device and cpu_workers > 1:
        print(
            f"🚀 Running PASS1→PASS2 per ensemble with up to {cpu_workers} CPU worker(s) "
            f"(scenario-ordered queue)..."
        )
    else:
        print("🚀 Running PASS1→PASS2 sequentially per ensemble (scenario-ordered)...")

    pending_total = len(pending_payloads)

    def _register_result(res: Dict[str, object], idx_done: int, total: int) -> None:
        key = str(res.get("key"))
        pass1_rec = res.get("pass1_record")
        pass2_rec = res.get("pass2_record")
        status = str(res.get("status"))
        scen = res.get("scenario", "?")
        workflow_val = res.get("workflow", "?")
        runmodus_val = res.get("runmodus", "?")
        run_id_val = res.get("run_id")
        ens_val = res.get("ensemble_id")
        msg = res.get("message", "")

        if isinstance(pass1_rec, dict):
            delta_records_map[key] = pass1_rec
        if isinstance(pass2_rec, dict):
            pass2_records_map[key] = pass2_rec

        try:
            run_id_str = f"{int(run_id_val):03d}"
        except Exception:
            run_id_str = str(run_id_val)
        try:
            ens_str = f"{int(ens_val):04d}"
        except Exception:
            ens_str = str(ens_val)
        prefix = f"{scen}|{workflow_val}|{runmodus_val}|run{run_id_str}|r{ens_str}"

        if status == "completed":
            info = res.get("pass2_msg") or msg
            print(f"[PASS1+PASS2 {idx_done}/{total}] {prefix} {info}")
            _update_progress_entry(
                progress_data,
                key,
                status="completed",
                pass1=pass1_rec if isinstance(pass1_rec, dict) else None,
                pass2=pass2_rec if isinstance(pass2_rec, dict) else None,
                output_root=output_root,
                shard_id=shard_runtime_id,
                host=host_name,
            )
        elif status == "pass2_failed":
            print(f"[PASS2 {idx_done}/{total}] ❌ {prefix}: {msg}")
            _update_progress_entry(
                progress_data,
                key,
                status="failed",
                pass1=pass1_rec if isinstance(pass1_rec, dict) else None,
                message=str(msg),
                output_root=output_root,
                shard_id=shard_runtime_id,
                host=host_name,
            )
        else:
            print(f"[PASS1 {idx_done}/{total}] ❌ {prefix}: {msg}")
            _update_progress_entry(
                progress_data,
                key,
                status="failed",
                pass1=pass1_rec if isinstance(pass1_rec, dict) else None,
                message=str(msg),
                output_root=output_root,
                shard_id=shard_runtime_id,
                host=host_name,
            )
        _save_progress_json(progress_data, progress_path)

    if is_cpu_device and cpu_workers > 1 and pending_payloads:
        ctx = _cpu_pool_context()
        with ctx.Pool(processes=cpu_workers) as pool:
            for i, res in enumerate(
                pool.imap_unordered(_pass1_pass2_combined_worker, pending_payloads, chunksize=1),
                start=1,
            ):
                _register_result(res, i, pending_total)
    else:
        for i, payload in enumerate(pending_payloads, start=1):
            res = _pass1_pass2_combined_worker(payload)
            _register_result(res, i, pending_total)

    delta_records = list(delta_records_map.values())
    pass2_records = list(pass2_records_map.values())

    if not delta_records:
        print("❌ No Δ offsets could be computed; aborting.")
        return 1

    deltas_all = [float(rec["delta_C"]) for rec in delta_records if math.isfinite(rec.get("delta_C", np.nan))]
    mean_delta = float(np.mean(deltas_all))
    std_delta = float(np.std(deltas_all))
    per_year_stds = [r.get("per_year_std", 0.0) for r in delta_records if "per_year_std" in r]
    mean_per_year_std = float(np.mean(per_year_stds)) if per_year_stds else 0.0
    max_per_year_std = float(np.max(per_year_stds)) if per_year_stds else 0.0

    print(
        f"✅ PASS1+PASS2 complete for {len(pass2_records)}/{len(delta_records)} ensembles. "
        f"Mean Δ = {mean_delta:.3f} °C, std = {std_delta:.3f} °C"
    )
    if per_year_stds:
        print(
            f"   Per-year delta consistency: mean std = {mean_per_year_std:.3f} °C, "
            f"max std = {max_per_year_std:.3f} °C"
        )

    # Save Δ offsets to CSV
    try:
        import pandas as pd  # local import

        df_delta = pd.DataFrame(delta_records)
        delta_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_delta.to_csv(delta_csv_path, index=False)
        print(f"💾 Saved Δ offsets to {delta_csv_path}")
    except Exception as e:
        print(f"⚠️  Failed to save Δ CSV ({delta_csv_path}): {e}")

    # Δ histogram
    try:
        delta_plot_path.parent.mkdir(parents=True, exist_ok=True)
        _plot_delta_histogram(deltas_all, delta_plot_path.parent, title_suffix="MAGICCxERA5")
        if delta_plot_path.name != "debias_offsets_hist.png":
            default_plot = delta_plot_path.parent / "debias_offsets_hist_MAGICCxERA5.png"
            if default_plot.exists():
                default_plot.rename(delta_plot_path)
        print(f"📊 Saved Δ histogram around {delta_plot_path}")
    except Exception as e:
        print(f"⚠️  Failed to save Δ histogram: {e}")

    # Save PASS2 records for diagnostics
    if pass2_records:
        try:
            import pandas as pd  # local import

            df_p2 = pd.DataFrame(pass2_records)
            out_csv = artifacts_root / "debias_pass2_records.csv"
            df_p2.to_csv(out_csv, index=False)
            print(f"💾 Saved PASS2 diagnostics to {out_csv}")
        except Exception as e:
            print(f"⚠️  Failed to save PASS2 diagnostics CSV: {e}")

    # Plot timeseries of delta offsets (in info_plots subdirectory)
    info_plots_dir = artifacts_root / "info_plots"
    info_plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        timeseries_plot_path = _plot_delta_timeseries(
            delta_records,
            info_plots_dir,
            title_suffix="MAGICCxERA5",
        )
        print(f"📈 Saved Δ offsets timeseries plot to {timeseries_plot_path}")
    except Exception as e:
        print(f"⚠️  Failed to create Δ offsets timeseries plot: {e}")
        import traceback
        if debug_trace:
            traceback.print_exc()

    # Plot tas_out, tas_pred_pass1, and delta overlay for each ensemble (in main output directory)
    try:
        tas_overlay_path = _plot_tas_timeseries_overlay(
            delta_records,
            artifacts_root,
            title_suffix="MAGICCxERA5",
        )
        if isinstance(tas_overlay_path, Path) and tas_overlay_path.is_dir():
            n_plots = len(list(tas_overlay_path.glob("tas_timeseries_*.png")))
            print(f"📊 Saved {n_plots} tas timeseries overlay plots to {tas_overlay_path}")
        else:
            print(f"📊 Saved tas timeseries overlay plot to {tas_overlay_path}")
    except Exception as e:
        print(f"⚠️  Failed to create tas timeseries overlay plots: {e}")
        import traceback
        if debug_trace:
            traceback.print_exc()

    # Clean up temporary PASS1 outputs (if requested)
    if args.save_pass1 and DELETE_TEMPFOLDER_PASS1:
        try:
            if pass1_temp_stage_root.exists():
                shutil.rmtree(pass1_temp_stage_root)
                print(f"🧹 Removed temporary PASS1 outputs under {pass1_temp_stage_root}")
        except Exception as e:
            print(f"⚠️  Failed to delete PASS1 temp folder {pass1_temp_stage_root}: {e}")

    if bool(getattr(args, "auto_consolidate", False)):
        debias_root = (output_root / "debias").resolve(strict=False)
        if debias_root.exists():
            _run_autoconsolidate(
                source_paths=[debias_root],
                config_path=getattr(args, "auto_consolidate_config", None),
                cleanup_local=bool(getattr(args, "auto_consolidate_cleanup_local", True)),
            )
        else:
            print(f"⚠️  Skipping auto-consolidate: debias root not found: {debias_root}")

    print(f"✅ Debias loop complete. Canonical data root: {output_root}")
    if artifacts_root != output_root:
        print(f"   Run artifacts root: {artifacts_root}")
    print(f"   Δ offsets CSV: {delta_csv_path}")
    print(f"   Δ histogram:   {delta_plot_path}")
    print(f"   Progress checkpoint: {progress_path}")
    print(f"   Δ timeseries:   {info_plots_dir}")
    print(f"   tas overlay plots: {artifacts_root}")
    return 0


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
