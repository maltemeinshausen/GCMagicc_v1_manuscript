# helper_benchmark.py
#
# Central orchestrator for the validation suite
# (structured-record edition: flattened jobs, no nested parallelism)
# --------------------------------------------------------------
#  This version matches the *dictionary-based* interface introduced in
#  helper_bench_metric.py (>= v2):
#
#     . Segments now return **lists of dicts** rather than "name:value" maps.
#     . Database helpers accept those lists directly.
#
#  Consequently:
#     * _process_pair_worker() concatenates lists instead of dict-merging.
#     * run_validation_suite() passes the lists on to the DB helpers.
#     * The two bar-plot utilities build an "id" on the fly from the
#       explicit CSV columns - no legacy metric_name string required.
# --------------------------------------------------------------
from __future__ import annotations
import os
import pkgutil
import multiprocessing as mp
import json
import sqlite3
import subprocess
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Sequence
import time
import traceback
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path
import logging
import threading

import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import psutil  # for accurate RAM
except Exception:
    psutil = None

# Priority / niceness from parallelization_strategies
try:
    from parallelization_strategies import set_low_priority, set_joblib_low_priority
except Exception:

    def set_low_priority(*args, **kwargs):  # type: ignore
        pass

# Import recipes module
import notebooks.recipes as recipes

# local helpers
from .helper_bench_plot import parse_filename, extract_gcmagicc_code, get_common_variables_from_pair
from .helper_bench_plot import get_gcmagicc_prefix_ts
from .helper_bench_metric import (
    write_records_unified,
    discover_other_member_files,
    discover_other_model_files,
    discover_other_gcmagicc_files,
    resolve_cmip6_member_file,
    filter_spatial_vars as _hb_filter_spatial_vars,  # reuse existing dim-check
    jobs_index_mark,
    set_force_overwrite as _metric_set_force_overwrite,
    set_skip_duplication_check as _metric_set_skip_duplication_check,
)
from .helper_bench_metric import _compute_row_key  # for fast in-memory skip
from .helper_bench_metric import _NC_CSV, _CC_CSV, _CZ_CSV, _ZZ_CSV, _NN_CSV, _OC_CSV, _ON_CSV
from .helper_bench_metric import configure_database as _metric_configure_db
from .helper_debug import get_memory_usage_mb, log_stage_memory
from .helper_xarray_engine import (
    configure_mount_aware_xarray_defaults as _configure_xarray_engine_defaults,
    install_xarray_engine_preference as _install_xarray_engine_preference,
)

def set_joblib_low_priority(*args, **kwargs):  # type: ignore
    pass

# =============================================================================
# DEFERRED JOB STORE (persistent crash queue)
# =============================================================================


class DeferredJobStore:
    """Small SQLite-backed queue that persists crashed jobs between runs."""

    def __init__(self, db_path: str | Path | None):
        self.path = Path(db_path).expanduser() if db_path else None
        if self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._ensure_schema()
            except Exception:
                # Disable persistence if initialization fails; runtime falls back to in-memory queue
                self.path = None

    def _ensure_schema(self) -> None:
        if not self.path:
            return
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS failed_jobs_queue (
                    job_id TEXT PRIMARY KEY,
                    job_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    last_error TEXT,
                    exitcode INTEGER,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _job_id(job: Dict) -> str:
        normalized = json.dumps(job, sort_keys=True, default=str)
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

    def record_retry(
        self,
        job_entry: Dict,
        status: str,
        message: str,
        exitcode: Optional[int],
    ) -> None:
        if not self.path:
            return
        job = job_entry["job"]
        job_id = self._job_id(job)
        payload = json.dumps(job, sort_keys=True, default=str)
        attempts = int(job_entry.get("attempt", 0))
        ts = datetime.utcnow().isoformat()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO failed_jobs_queue (job_id, job_json, status, attempts, last_error, exitcode, updated_at)
                VALUES (?, ?, 'pending', ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    job_json=excluded.job_json,
                    status='pending',
                    attempts=excluded.attempts,
                    last_error=excluded.last_error,
                    exitcode=excluded.exitcode,
                    updated_at=excluded.updated_at
                """,
                (job_id, payload, attempts, message, exitcode, ts),
            )

    def mark_resolved(self, job: Dict) -> None:
        if not self.path:
            return
        job_id = self._job_id(job)
        ts = datetime.utcnow().isoformat()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "UPDATE failed_jobs_queue SET status='resolved', updated_at=? WHERE job_id=?",
                (ts, job_id),
            )

    def mark_failed(self, job: Dict, message: str) -> None:
        if not self.path:
            return
        job_id = self._job_id(job)
        ts = datetime.utcnow().isoformat()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                UPDATE failed_jobs_queue
                SET status='failed', last_error=?, updated_at=?
                WHERE job_id=?
                """,
                (message, ts, job_id),
            )

    def load_pending(self) -> List[Dict]:
        if not self.path:
            return []
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT job_json, attempts, last_error FROM failed_jobs_queue WHERE status='pending' ORDER BY updated_at"
            ).fetchall()
        pending: List[Dict] = []
        for job_json, attempts, last_error in rows:
            try:
                job = json.loads(job_json)
            except Exception:
                continue
            pending.append(
                {
                    "job": job,
                    "attempt": int(attempts or 0),
                    "last_error": last_error,
                }
            )
        return pending

# Note: recipes import will be handled by the calling code
# --- colour palette helper (moved to helper_bench_plot) -----------------------
# get_standard_colors() function has been moved to helper_bench_plot.py


def _auto_tune_n_jobs(cfg: dict, *, pairs_count: int) -> int:
    """Choose a safe n_jobs from CPU & RAM. Do **not** use disk space for RAM gating."""
    requested = int(cfg.get("joblib_n_jobs", 1))
    reserve = int(cfg.get("reserve_cpus", 0))
    total_cpu = mp.cpu_count()
    cpu_cap = max(1, min(requested, max(1, total_cpu - reserve)))

    # Real available RAM
    avail_gb = None
    if psutil:
        try:
            avail_gb = psutil.virtual_memory().available / (1024**3)
        except Exception:
            pass

    keep_free_gb = float(cfg.get("keep_free_mem_gb", 32.0))  # OS cushion
    per_job_gb = float(cfg.get("target_mem_per_job_gb", 20.0))
    mem_cap = cpu_cap
    usable_gb = None
    if avail_gb is not None and per_job_gb > 0:
        usable_gb = max(0.0, avail_gb - keep_free_gb)
        mem_cap = max(1, int(usable_gb // per_job_gb))

    # Final: min(CPU cap, MEM cap), with optional min/max floors
    n_jobs = min(cpu_cap, mem_cap)
    min_jobs = int(cfg.get("min_jobs", 1))
    max_jobs = int(cfg.get("max_jobs", cpu_cap))
    n_jobs = max(1, n_jobs)  # never below 1
    n_jobs = min(max_jobs, n_jobs)
    n_jobs = max(min_jobs, n_jobs)  # optional floor

    if n_jobs != requested:
        print(
            f"⚠️  Adjusted joblib jobs from {requested} to {n_jobs} "
            f"(reserve={reserve}, totalCPU={total_cpu}, RAM_avail≈{(avail_gb or -1):.1f}GB, "
            f"keep_free={keep_free_gb}GB, usable≈{(usable_gb if usable_gb is not None else -1):.1f}GB, "
            f"perJob≈{per_job_gb:.1f}GB)"
        )
    print("   🔧 Joblib configuration:")
    print(f"      - Jobs: {n_jobs}")
    print(f"      - Timeout: {cfg.get('timeout_seconds', 'n/a')}s")
    print(f"      - Pairs: {pairs_count}")
    return n_jobs


# --- window-contract helpers -----------------------------------------------
_WINDOW_TOKEN_RE = re.compile(r"(\d{4})to(\d{4})")
_GCMAGICC_TIMESTAMP_RE = re.compile(r"-\d{8}-\d{4}")
_RECIPE_TO_METRICDOMAIN = {
    "ENSOTeleconnections": "ENSOTelecon",
    "IndicatorFrequencies": "IndicatorFreq",
}


def _window_contract_source_core(value: str | None) -> str:
    s = str(value or "").strip()
    if s.startswith("GCMagicc-"):
        return _GCMAGICC_TIMESTAMP_RE.sub("-TIMESTAMP", s)
    return s


def _coerce_forced_windows(raw) -> List[Tuple[int, int, str]]:
    out: List[Tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()
    if not raw:
        return out

    items = raw if isinstance(raw, (list, tuple)) else [raw]
    for item in items:
        y0 = y1 = None
        label = ""
        if isinstance(item, dict):
            y0 = item.get("start_year", item.get("yr0"))
            y1 = item.get("end_year", item.get("yr1"))
            label = str(item.get("label", "") or "").strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            y0, y1 = item[0], item[1]
            if len(item) >= 3:
                label = str(item[2] or "").strip()
        else:
            continue

        try:
            y0i = int(y0)
            y1i = int(y1)
        except Exception:
            continue
        if y1i < y0i:
            y0i, y1i = y1i, y0i
        if not label:
            label = f"{y0i}to{y1i}"
        key = (y0i, y1i, label)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _load_window_contract_entries(path: str | os.PathLike | None) -> List[Dict]:
    text = str(path or "").strip()
    if not text:
        return []
    p = Path(text)
    if not p.exists():
        return []
    if not p.is_file():
        return []
    try:
        payload = json.loads(p.read_text())
    except Exception:
        return []

    entries = payload.get("keys", [])
    if not isinstance(entries, list):
        return []

    out: List[Dict] = []
    for ent in entries:
        if not isinstance(ent, dict):
            continue
        metricdomain = str(ent.get("metricdomain", "") or "").strip()
        source_core = _window_contract_source_core(ent.get("source_core", ""))
        member_id = str(ent.get("member_id", "") or "").strip()
        variable = str(ent.get("variable", "") or "").strip()
        experiment_id = str(ent.get("experiment_id", "") or "").strip()
        comp_source_id = str(ent.get("comp_source_id", "") or "").strip()
        comp_member_id = str(ent.get("comp_member_id", "") or "").strip()
        whitelist = sorted(
            {
                str(m).strip()
                for m in (ent.get("metrictype_whitelist") or [])
                if str(m).strip()
            }
        )
        forced_windows = _coerce_forced_windows(ent.get("forced_windows") or [])
        forced_labels = sorted({w[2] for w in forced_windows if len(w) >= 3})
        if not metricdomain:
            continue
        out.append(
            {
                "metricdomain": metricdomain,
                "source_core": source_core,
                "member_id": member_id,
                "variable": variable,
                "experiment_id": experiment_id,
                "comp_source_id": comp_source_id,
                "comp_member_id": comp_member_id,
                "metrictype_whitelist": whitelist,
                "forced_windows": forced_windows,
                "forced_window_labels": forced_labels,
            }
        )
    return out


def _metricdomain_for_recipe(recipe: str) -> str:
    recipe_s = str(recipe or "").strip()
    return _RECIPE_TO_METRICDOMAIN.get(recipe_s, recipe_s)


def _normalize_experiment_for_contract(scenario: str, cfg: dict) -> str:
    scen = str(scenario or "").strip()
    if scen == "historical-ERA5":
        return "historical"
    if bool(cfg.get("gxe_mode", False)) and scen and scen not in ("historical", "historical-ERA5"):
        return "historical"
    return scen


def _cfg_with_nn_window_contract(cfg: dict, job: dict) -> dict:
    """
    Attach forced window/metrictype controls for GOFNN jobs that are
    observation-side and GOFON-guided.
    """
    if str(job.get("comparison", "")).strip().lower() != "nn":
        return cfg
    if not bool(job.get("gofnn_so_from_gofon", False)):
        return cfg

    strict = bool(cfg.get("gof_window_contract_strict", False))
    entries: List[Dict] = list(cfg.get("window_contract_entries") or [])
    cfg_job = dict(cfg)
    cfg_job["window_contract_active"] = True
    cfg_job["window_contract_required"] = strict
    cfg_job["window_contract_match_count"] = 0

    if not entries:
        return cfg_job

    file_a = str(job.get("file_a", "") or "")
    try:
        cm_model, cm_scen, cm_member = parse_filename(
            os.path.basename(file_a), use_pseudo_member=False
        )
    except Exception:
        return cfg_job

    gcmagicc_code = extract_gcmagicc_code(file_a) or ""
    source_id = f"{gcmagicc_code}_{cm_model}" if gcmagicc_code else cm_model
    source_core = _window_contract_source_core(source_id)
    metricdomain = _metricdomain_for_recipe(str(job.get("recipe", "")))
    member_id = str(cm_member or "").strip()
    experiment_id = _normalize_experiment_for_contract(cm_scen, cfg)
    comp_source_id = "ERA5"

    matches: List[Dict] = []
    for ent in entries:
        if str(ent.get("metricdomain", "") or "").strip() != metricdomain:
            continue
        if str(ent.get("source_core", "") or "").strip() != source_core:
            continue
        if str(ent.get("member_id", "") or "").strip() != member_id:
            continue
        if str(ent.get("experiment_id", "") or "").strip() != experiment_id:
            continue
        if str(ent.get("comp_source_id", "") or "").strip() != comp_source_id:
            continue
        ent_comp_member = str(ent.get("comp_member_id", "") or "").strip()
        if ent_comp_member and ent_comp_member != "r1i1p1f1":
            # GOFNN jobs do not carry ERA5 member metadata; only keep
            # non-default members when explicitly requested.
            continue
        matches.append(ent)

    cfg_job["window_contract_match_count"] = len(matches)
    if not matches:
        return cfg_job

    forced_windows: List[Tuple[int, int, str]] = []
    seen_windows: set[tuple[int, int, str]] = set()
    whitelist_union: set[str] = set()
    whitelist_by_var: Dict[str, set[str]] = defaultdict(set)

    for ent in matches:
        for w in ent.get("forced_windows", []) or []:
            try:
                w_key = (int(w[0]), int(w[1]), str(w[2]))
            except Exception:
                continue
            if w_key in seen_windows:
                continue
            seen_windows.add(w_key)
            forced_windows.append(w_key)
        vals = {str(v).strip() for v in (ent.get("metrictype_whitelist") or []) if str(v).strip()}
        whitelist_union.update(vals)
        var_key = str(ent.get("variable", "") or "").strip()
        if var_key:
            whitelist_by_var[var_key].update(vals)

    cfg_job["forced_windows"] = forced_windows
    cfg_job["forced_window_labels"] = [w[2] for w in forced_windows]
    cfg_job["forced_metrictype_whitelist"] = sorted(whitelist_union)
    cfg_job["forced_metrictype_by_variable"] = {
        k: sorted(v) for k, v in whitelist_by_var.items()
    }
    return cfg_job


def _cfg_with_cc_window_contract(cfg: dict, job: dict) -> dict:
    """
    Attach forced window/metrictype controls for SO-side GOFCC jobs
    that are explicitly associated with GOFOC anchors.
    """
    if str(job.get("comparison", "")).strip().lower() != "cc":
        return cfg
    if not bool(job.get("gofcc_so_from_gofoc", False)):
        return cfg

    strict = bool(cfg.get("gof_oc_window_contract_strict", False))
    entries: List[Dict] = list(cfg.get("oc_window_contract_entries") or [])
    cfg_job = dict(cfg)
    cfg_job["window_contract_active"] = True
    cfg_job["window_contract_required"] = strict
    cfg_job["window_contract_match_count"] = 0
    cfg_job["window_contract_kind"] = "occc"

    if not entries:
        return cfg_job

    file_a = str(job.get("file_a", "") or "")
    try:
        cm_model, cm_scen, cm_member = parse_filename(
            os.path.basename(file_a), use_pseudo_member=False
        )
    except Exception:
        return cfg_job

    source_core = _window_contract_source_core(cm_model)
    metricdomain = _metricdomain_for_recipe(str(job.get("recipe", "")))
    member_id = str(cm_member or "").strip()
    experiment_id = _normalize_experiment_for_contract(cm_scen, cfg)

    matches: List[Dict] = []
    for ent in entries:
        if str(ent.get("metricdomain", "") or "").strip() != metricdomain:
            continue
        if str(ent.get("source_core", "") or "").strip() != source_core:
            continue
        if str(ent.get("member_id", "") or "").strip() != member_id:
            continue
        if str(ent.get("experiment_id", "") or "").strip() != experiment_id:
            continue
        ent_comp_source = str(ent.get("comp_source_id", "") or "").strip()
        if ent_comp_source and ent_comp_source != "ERA5":
            continue
        matches.append(ent)

    cfg_job["window_contract_match_count"] = len(matches)
    if not matches:
        return cfg_job

    forced_windows: List[Tuple[int, int, str]] = []
    seen_windows: set[tuple[int, int, str]] = set()
    whitelist_union: set[str] = set()
    whitelist_by_var: Dict[str, set[str]] = defaultdict(set)

    for ent in matches:
        for w in ent.get("forced_windows", []) or []:
            try:
                w_key = (int(w[0]), int(w[1]), str(w[2]))
            except Exception:
                continue
            if w_key in seen_windows:
                continue
            seen_windows.add(w_key)
            forced_windows.append(w_key)
        vals = {
            str(v).strip()
            for v in (ent.get("metrictype_whitelist") or [])
            if str(v).strip()
        }
        whitelist_union.update(vals)
        var_key = str(ent.get("variable", "") or "").strip()
        if var_key:
            whitelist_by_var[var_key].update(vals)

    cfg_job["forced_windows"] = forced_windows
    cfg_job["forced_window_labels"] = [w[2] for w in forced_windows]
    cfg_job["forced_metrictype_whitelist"] = sorted(whitelist_union)
    cfg_job["forced_metrictype_by_variable"] = {
        k: sorted(v) for k, v in whitelist_by_var.items()
    }
    return cfg_job


def _apply_window_contract_filter(recs: List[Dict], cfg: dict, job: dict) -> List[Dict]:
    """
    Enforce post-plugin whitelist filtering for strict GOF-derived window contracts
    (GOFNN from GOFON, and SO-side GOFCC from GOFOC).
    """
    if not bool(cfg.get("window_contract_active", False)):
        return recs

    strict = bool(cfg.get("window_contract_required", False))
    matches = int(cfg.get("window_contract_match_count", 0) or 0)
    whitelist = {str(v).strip() for v in (cfg.get("forced_metrictype_whitelist") or []) if str(v).strip()}
    by_var = {
        str(k): {str(v).strip() for v in (vals or []) if str(v).strip()}
        for k, vals in (cfg.get("forced_metrictype_by_variable") or {}).items()
    }

    if strict and matches <= 0:
        raise RuntimeError(
            f"window_contract_mismatch:no_contract_match recipe={job.get('recipe')} file_a={os.path.basename(str(job.get('file_a','')))}"
        )
    if not whitelist:
        if strict:
            raise RuntimeError(
                f"window_contract_mismatch:empty_whitelist recipe={job.get('recipe')} file_a={os.path.basename(str(job.get('file_a','')))}"
            )
        return recs

    filtered: List[Dict] = []
    for rec in recs or []:
        mt = str(rec.get("metrictype", "") or "").strip()
        var = str(rec.get("variable", "") or "").strip()
        allow = by_var.get(var, whitelist)
        if mt in allow:
            filtered.append(rec)

    if strict and not filtered:
        raise RuntimeError(
            f"window_contract_mismatch:no_records_after_filter recipe={job.get('recipe')} file_a={os.path.basename(str(job.get('file_a','')))}"
        )
    return filtered


# --- time window generation helper -------------------------------------------
def _generate_time_windows(
    ds,
    cfg: Dict,
    ds_cmp=None,
) -> List[Tuple[int, int, str]]:
    """Generate time windows based on configuration.

    Args:
        ds: xarray Dataset with time dimension
        cfg: Configuration dictionary

    Returns:
        List of (start_year, end_year, window_name) tuples

    Configuration options:
        - window_years: Length of each window (default: 20)
        - window_mode: 'single_end' for just the last window_years years,
                      'single_start' for first window_years years,
                      'single_mid' for middle window_years years,
                      'triple' for start, middle, and end windows (default)
    """
    forced_windows = _coerce_forced_windows(cfg.get("forced_windows"))
    if forced_windows:
        return forced_windows

    yrs = ds.time.dt.year
    yr0, yr1 = int(yrs.min()), int(yrs.max())
    if ds_cmp is not None and "time" in ds_cmp.coords:
        try:
            cyrs = ds_cmp.time.dt.year
            c0, c1 = int(cyrs.min()), int(cyrs.max())
            yr0 = max(yr0, c0)
            yr1 = min(yr1, c1)
        except Exception:
            pass
    win = cfg.get("window_years", 20)
    mode = cfg.get("window_mode", "single_end").lower()

    # Cap maximum year at 2100 for SSP comparisons to ensure compatibility
    # with other scenarios that only go to 2100
    comparison = str(cfg.get("comparison", "nc")).lower()
    if comparison in ["nc", "cc", "cz", "zz", "nn", "oc", "on"]:
        # This is a comparison involving SSP scenarios
        yr1 = min(yr1, 2100)
    if comparison == "nn":
        nn_start_raw = cfg.get("nn_forced_year_start")
        nn_end_raw = cfg.get("nn_forced_year_end")
        if nn_start_raw is not None and str(nn_start_raw).strip() != "":
            yr0 = max(yr0, int(nn_start_raw))
        if nn_end_raw is not None and str(nn_end_raw).strip() != "":
            yr1 = min(yr1, int(nn_end_raw))
    if comparison == "on":
        # Optional cap for GOFON historical-only enforcement.
        hist_end_raw = cfg.get("on_historical_end_year")
        if hist_end_raw is not None and str(hist_end_raw).strip() != "":
            yr1 = min(yr1, int(hist_end_raw))
    if yr1 < yr0:
        yr1 = yr0

    mid = yr0 + (yr1 - yr0) // 2

    windows = {
        "single_start": [(yr0, min(yr1, yr0 + win - 1), f"{yr0}to{min(yr1,yr0+win-1)}")],
        "single_mid": [
            (
                max(yr0, mid - win // 2),
                min(yr1, mid + win // 2 - 1),
                f"{max(yr0,mid-win//2)}to{min(yr1,mid+win//2-1)}",
            )
        ],
        "single_end": [(max(yr0, yr1 - win + 1), yr1, f"{max(yr0,yr1-win+1)}to{yr1}")],
        "triple": [
            (yr0, min(yr1, yr0 + win - 1), f"{yr0}to{min(yr1,yr0+win-1)}"),
            (
                max(yr0, mid - win // 2),
                min(yr1, mid + win // 2 - 1),
                f"{max(yr0,mid-win//2)}to{min(yr1,mid+win//2-1)}",
            ),
            (max(yr0, yr1 - win + 1), yr1, f"{max(yr0,yr1-win+1)}to{yr1}"),
        ],
    }
    return windows.get(mode, windows["single_end"])


# --- xarray engine override (for unstable FUSE/netcdf4 combinations) --------
_XARRAY_ENGINE_OVERRIDE_APPLIED = False


def _install_xarray_engine_override() -> None:
    """
    Install mount-aware xarray engine routing for benchmark workers.
    """
    global _XARRAY_ENGINE_OVERRIDE_APPLIED
    if _XARRAY_ENGINE_OVERRIDE_APPLIED:
        return

    _configure_xarray_engine_defaults(
        mount_engine_default="h5netcdf",
        local_engine_default="netcdf4",
        scope_default="auto",
        fallback_default=True,
        fallback_local_only_default=True,
    )
    _install_xarray_engine_preference(
        mount_engine_default="h5netcdf",
        local_engine_default="netcdf4",
        scope_default="auto",
        fallback_default=True,
        fallback_local_only_default=True,
    )

    _XARRAY_ENGINE_OVERRIDE_APPLIED = True


# --- common variable discovery helpers (shared by many recipes) -------------
def discover_common_spatial_vars_from_open(
    ds_a,
    ds_b,
    restrict: Optional[Sequence[str]] = None,
    *,
    file_a: Optional[str] = None,
    file_b: Optional[str] = None,
) -> List[str]:
    """
    Determine common variables between two OPEN xarray Datasets.
    - If *restrict* is provided: start from that list, keep only present in both.
    - Else: try helper_bench_plot.get_common_variables_from_pair (if file paths provided),
            falling back to name intersection.
    Finally: enforce 3-D spatial variables (must have lat/lon/time) in BOTH datasets.
    """
    # Step 1: candidate names
    if restrict:
        candidates = [v for v in restrict if v in ds_a and v in ds_b]
    else:
        candidates = None
        if file_a and file_b:
            pair = {"cmip6_files": [{"file": file_a}], "gcmagicc_files": [{"file": file_b}]}
            try:
                candidates = get_common_variables_from_pair(pair)
            except Exception:
                candidates = None
        if candidates is None:
            candidates = sorted(set(ds_a.data_vars) & set(ds_b.data_vars))

    # Step 2: require lat/lon/time in BOTH datasets
    cand_a = _hb_filter_spatial_vars(ds_a, candidates)
    cand_b = _hb_filter_spatial_vars(ds_b, candidates)
    return sorted(set(cand_a) & set(cand_b))


def discover_common_spatial_vars_from_files(
    file_a: str,
    file_b: str,
    restrict: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Like discover_common_spatial_vars_from_open but starts from file paths.
    Uses get_common_variables_from_pair for a fast first-pass intersection,
    then verifies lat/lon/time by briefly opening both datasets.
    """
    import xarray as xr
    _install_xarray_engine_override()

    issues = collect_unreadable_job_inputs({"file_a": file_a, "file_b": file_b})
    if issues:
        for _, path, reason in issues:
            _warn_unreadable_input_once("discover_common_spatial_vars", path, reason)
        return []

    if not restrict:
        pair = {"cmip6_files": [{"file": file_a}], "gcmagicc_files": [{"file": file_b}]}
        try:
            base = get_common_variables_from_pair(pair)
        except Exception:
            base = None
    else:
        base = restrict

    with (
        xr.open_dataset(file_a, use_cftime=True) as da,
        xr.open_dataset(file_b, use_cftime=True) as db,
    ):
        return discover_common_spatial_vars_from_open(da, db, base, file_a=file_a, file_b=file_b)


# --- dynamic recipe discovery (cached) --------------------------------------
_RECIPE_CACHE = None


def clear_recipe_cache():
    """Clear the recipe cache to force fresh module loading."""
    global _RECIPE_CACHE
    _RECIPE_CACHE = None
    # Also clear any cached modules from sys.modules
    import sys

    modules_to_remove = []
    for module_name in sys.modules:
        if module_name.startswith("recipes."):
            modules_to_remove.append(module_name)

    for module_name in modules_to_remove:
        del sys.modules[module_name]

    print("✓ Segment cache cleared - fresh modules will be loaded")


def _discover_recipes():
    global _RECIPE_CACHE
    if _RECIPE_CACHE is not None:
        return _RECIPE_CACHE

    # print("Discovering analysis recipes ...")
    recipe_modules = {}
    for _, name, _ in pkgutil.iter_modules(recipes.__path__):
        if name == "template":
            continue

        # Handle hyphenated filenames by loading them directly
        file_path = os.path.join(recipes.__path__[0], f"{name}.py")
        if os.path.exists(file_path):
            try:
                # Use importlib.util to load the module from file path
                import importlib.util
                import sys

                spec = importlib.util.spec_from_file_location(f"recipes.{name}", file_path)
                mod = importlib.util.module_from_spec(spec)
                
                # Set up module properly for dataclasses and other features
                sys.modules[f"recipes.{name}"] = mod
                mod.__module__ = f"recipes.{name}"
                
                spec.loader.exec_module(mod)
                recipe_modules[name] = mod
                # print(f"  - {name}")
            except Exception as e:
                print(f"  ✗ Failed to import {name}: {e}")
        else:
            print(f"  ✗ File not found: {file_path}")

    _RECIPE_CACHE = recipe_modules
    return recipe_modules


def get_recipe_plugins():  # external helper (rarely used)
    return _discover_recipes()


def _pick_oldest_gcmagicc(gcmagicc_candidates):
    """
    Given a list of GCMagicc file dicts or paths, return the *oldest* file.
    Priority sort: by prefix timestamp if present; otherwise by file mtime.
    """
    import os
    from datetime import datetime

    # normalize to list[str]
    paths = []
    for c in gcmagicc_candidates:
        if isinstance(c, dict) and "file" in c:
            paths.append(c["file"])
        elif isinstance(c, str):
            paths.append(c)
    if not paths:
        return None

    def sort_key(p):
        ts = get_gcmagicc_prefix_ts(p)
        return (0, ts) if ts else (1, datetime.fromtimestamp(os.path.getmtime(p)))

    return sorted(paths, key=sort_key)[0]


def _norm_path(path: str | None) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).expanduser().resolve(strict=False))
    except Exception:
        return os.path.abspath(str(path))


def _is_netcdf_like_path(path: str | None) -> bool:
    token = str(path or "").strip().lower()
    return token.endswith((".nc", ".nc4", ".cdf"))


@lru_cache(maxsize=8192)
def probe_netcdf_readable(path: str | None) -> Tuple[bool, str]:
    """
    Best-effort out-of-process readability probe for NetCDF-like inputs.

    This avoids native HDF/netCDF crashes inside xarray by checking files with
    `ncdump -k` before we hand them to recipe code.
    """
    norm = _norm_path(path)
    if not norm:
        return True, ""
    if not _is_netcdf_like_path(norm):
        return True, ""
    if not os.path.exists(norm):
        return False, "missing file"
    try:
        proc = subprocess.run(
            ["ncdump", "-k", norm],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            check=False,
        )
    except FileNotFoundError:
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "ncdump timeout after 20s"

    if int(proc.returncode) == 0:
        return True, ""

    msg = (proc.stderr or "").strip() or f"ncdump rc={proc.returncode}"
    return False, msg.replace("\n", " | ")[:300]


def format_unreadable_input_issues(
    issues: Sequence[Tuple[str, str, str]],
    *,
    include_reason: bool = True,
) -> str:
    parts: List[str] = []
    for label, path, reason in issues:
        token = f"{label}={os.path.basename(path)}"
        if include_reason and reason:
            token = f"{token} ({reason})"
        parts.append(token)
    return "; ".join(parts)


def collect_unreadable_job_inputs(
    job: Dict,
    *,
    extra_paths: Optional[Sequence[Tuple[str, str]]] = None,
) -> List[Tuple[str, str, str]]:
    issues: List[Tuple[str, str, str]] = []
    seen: set[str] = set()
    candidates: List[Tuple[str, str]] = []
    for key in ("file_a", "file_b", "gofon_reference_file", "gofoc_reference_file"):
        path = str(job.get(key, "") or "").strip()
        if path:
            candidates.append((key, path))
    if extra_paths:
        candidates.extend(extra_paths)
    for label, path in candidates:
        norm = _norm_path(path)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        ok, reason = probe_netcdf_readable(norm)
        if not ok:
            issues.append((label, norm, reason))
    return issues


@lru_cache(maxsize=8192)
def _warn_unreadable_input_once(context: str, path: str, reason: str) -> None:
    logging.getLogger("validation_suite").warning(
        "%s: unreadable NetCDF input skipped: %s (%s)",
        context,
        path,
        reason or "unknown error",
    )


def _same_path(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return _norm_path(a) == _norm_path(b)


def _pairing_policy_sc(cfg: dict) -> str:
    return str(cfg.get("pairing_policy_sc", "nc_full_cc_zz_nn_single")).strip().lower()


def _pairing_policy_so(cfg: dict) -> str:
    return str(cfg.get("pairing_policy_so", "reduced")).strip().lower()


@lru_cache(maxsize=1024)
def _overlap_year_bounds(file_a: str, file_b: str) -> Optional[Tuple[int, int]]:
    for path in (file_a, file_b):
        ok, reason = probe_netcdf_readable(path)
        if not ok:
            _warn_unreadable_input_once("overlap_year_bounds", _norm_path(path), reason)
            return None
    try:
        import xarray as xr

        with xr.open_dataset(file_a, use_cftime=True) as ds_a, xr.open_dataset(
            file_b, use_cftime=True
        ) as ds_b:
            if "time" not in ds_a.coords or "time" not in ds_b.coords:
                return None
            yrs_a = ds_a.time.dt.year
            yrs_b = ds_b.time.dt.year
            y0 = max(int(yrs_a.min()), int(yrs_b.min()))
            y1 = min(int(yrs_a.max()), int(yrs_b.max()))
            if y1 < y0:
                return None
            return y0, y1
    except Exception:
        return None


def _cfg_with_nn_gofon_period(cfg: dict, job: dict) -> dict:
    if str(job.get("comparison", "")).strip().lower() != "nn":
        return cfg
    ref = str(job.get("gofon_reference_file", "") or "").strip()
    file_a = str(job.get("file_a", "") or "").strip()
    if not (ref and file_a):
        return cfg
    bounds = _overlap_year_bounds(_norm_path(file_a), _norm_path(ref))
    if not bounds:
        return cfg
    y0, y1 = bounds
    label = f"{int(y0)}to{int(y1)}"
    cfg_job = dict(cfg)
    cfg_job["nn_forced_year_start"] = int(y0)
    cfg_job["nn_forced_year_end"] = int(y1)
    # Ensure NN plugins that key only off forced_windows honor GOFON overlap.
    cfg_job["forced_windows"] = [(int(y0), int(y1), label)]
    cfg_job["forced_window_labels"] = [label]
    return cfg_job


def _build_observation_anchor_member_map(paired_files: List[dict]) -> Dict[Tuple[str, str], str]:
    """
    Pick one deterministic anchor ensemble per (model, scenario) from paired keys.
    Used to reduce observation-side GOFOC/GOFON fan-out.
    """
    anchors: Dict[Tuple[str, str], str] = {}
    for pair in paired_files:
        try:
            model, scenario, ensemble = pair.get("key", ("", "", ""))
        except Exception:
            continue
        model = str(model or "").strip()
        scenario = str(scenario or "").strip()
        ensemble = str(ensemble or "").strip()
        if not (model and scenario and ensemble):
            continue
        key = (model, scenario)
        prev = anchors.get(key)
        if prev is None or ensemble < prev:
            anchors[key] = ensemble
    return anchors


def _apply_gofon_guided_nn_so(jobs: List[dict], cfg: dict) -> List[dict]:
    """
    For observation-side GXE runs, keep only NN jobs that have a matching ON anchor.
    Matching is done per (recipe, source file_a). Kept NN jobs carry a GOFON
    reference file so their period windows can be locked to the GOFON overlap.
    """
    if not jobs:
        return jobs
    if not bool(cfg.get("gofnn_so_follow_gofon", True)):
        return jobs

    on_lookup: Dict[Tuple[str, str], dict] = {}
    for job in jobs:
        if str(job.get("comparison", "")).strip().lower() != "on":
            continue
        key = (str(job.get("recipe", "")), _norm_path(str(job.get("file_a", ""))))
        on_lookup[key] = job

    out: List[dict] = []
    tagged = 0
    pruned = 0
    for job in jobs:
        comp = str(job.get("comparison", "")).strip().lower()
        if comp != "nn":
            out.append(job)
            continue

        pair = job.get("pair", {}) if isinstance(job.get("pair"), dict) else {}
        so_candidate = bool(pair.get("treat_as_historical_for_era5")) or bool(
            cfg.get("gxe_mode", False)
        )
        if not so_candidate:
            out.append(job)
            continue

        key = (str(job.get("recipe", "")), _norm_path(str(job.get("file_a", ""))))
        on_job = on_lookup.get(key)
        if on_job is None:
            pruned += 1
            continue

        tagged_job = dict(job)
        tagged_job["gofnn_so_from_gofon"] = True
        tagged_job["gofon_reference_file"] = on_job.get("file_b", "")
        out.append(tagged_job)
        tagged += 1

    if tagged or pruned:
        print(
            f"[INFO] GOFON-guided GOFNN_SO: tagged={tagged} pruned={pruned} "
            f"(remaining={len(out):,})"
        )

    # Prioritize observation-side dependencies ahead of dependent jobs.
    indexed = list(enumerate(out))
    indexed.sort(
        key=lambda item: (
            0
            if str(item[1].get("comparison", "")).strip().lower() == "on"
            else 1
            if str(item[1].get("comparison", "")).strip().lower() == "oc"
            else 2
            if (
                str(item[1].get("comparison", "")).strip().lower() == "nn"
                and bool(item[1].get("gofnn_so_from_gofon"))
            )
            else 3
            if (
                str(item[1].get("comparison", "")).strip().lower() == "cc"
                and bool(item[1].get("gofcc_so_from_gofoc"))
            )
            else 4,
            item[0],
        )
    )
    return [job for _, job in indexed]


def _apply_gofoc_guided_cc_so(jobs: List[dict], cfg: dict) -> List[dict]:
    """
    Tag SO-side CC jobs that are associated with a GOFOC anchor
    (same recipe and same CMIP6 source file_a).

    Unlike GOFNN pruning, this does not remove any CC jobs: SC CC behavior remains unchanged.
    """
    if not jobs:
        return jobs
    if not bool(cfg.get("gofcc_so_follow_gofoc", True)):
        return jobs

    oc_lookup: Dict[Tuple[str, str], dict] = {}
    for job in jobs:
        if str(job.get("comparison", "")).strip().lower() != "oc":
            continue
        key = (str(job.get("recipe", "")), _norm_path(str(job.get("file_a", ""))))
        oc_lookup[key] = job

    out: List[dict] = []
    tagged = 0
    for job in jobs:
        comp = str(job.get("comparison", "")).strip().lower()
        if comp != "cc":
            out.append(job)
            continue

        key = (str(job.get("recipe", "")), _norm_path(str(job.get("file_a", ""))))
        oc_job = oc_lookup.get(key)
        if oc_job is None:
            out.append(job)
            continue

        tagged_job = dict(job)
        tagged_job["gofcc_so_from_gofoc"] = True
        tagged_job["gofoc_reference_file"] = oc_job.get("file_b", "")
        out.append(tagged_job)
        tagged += 1

    if tagged:
        print(f"[INFO] GOFOC-guided GOFCC_SO: tagged={tagged} (total_jobs={len(out):,})")
    return out


# (Removed legacy per-pair worker; we now fully flatten per-comparison)

# (Removed unused single-recipe runner)


# (Replaced by fully flattened jobs; see _build_jobs_for_pair_recipe/_run_one_job)


def run_validation_suite(
    paired_files: List[dict],
    cfg: dict,
    *,
    return_status: bool = False,
    memory_pressure_event=None,
) -> bool | None:
    """
    Drive the full workflow over *paired_files* using a resilient process pool.
    Each job runs in its own subprocess so failures cannot abort the suite.
    """
    suite_start_time = time.time()

    # Clear recipe cache to ensure fresh module loading
    clear_recipe_cache()

    os.makedirs(cfg["output_folder"], exist_ok=True)

    # Propagate debug + DB backend info to workers via environment
    os.environ["DEBUG_ENABLED"] = "True" if cfg.get("debug", False) else "False"
    if cfg.get("db_backends"):
        os.environ["METRICS_DB_BACKENDS"] = ",".join(cfg["db_backends"])
    if cfg.get("sqlite_path"):
        os.environ["METRICS_DB_SQLITE_PATH"] = cfg["sqlite_path"]
    # NEW: Propagate force-overwrite and skip-dup-check intent to workers
    if cfg.get("force_overwrite") or cfg.get("skip_duplication_check"):
        os.environ["METRICS_FORCE_OVERWRITE"] = "1"
    else:
        os.environ.pop("METRICS_FORCE_OVERWRITE", None)
    if cfg.get("skip_duplication_check"):
        os.environ["METRICS_SKIP_DUP_CHECK"] = "1"
    else:
        os.environ.pop("METRICS_SKIP_DUP_CHECK", None)

    # Optional GOFON-driven window contract for strict GOFNN SO matching.
    contract_path = str(
        cfg.get("gof_window_contract_path")
        or os.environ.get("GOF_WINDOW_CONTRACT_PATH", "")
        or ""
    ).strip()
    strict_cfg = cfg.get("gof_window_contract_strict", None)
    if strict_cfg is None:
        strict_env_raw = os.environ.get(
            "GOF_WINDOW_CONTRACT_STRICT",
            "1" if contract_path else "0",
        )
        contract_strict = str(strict_env_raw).strip().lower() not in (
            "",
            "0",
            "false",
            "no",
            "off",
        )
    else:
        contract_strict = bool(strict_cfg)
    if contract_path:
        entries = _load_window_contract_entries(contract_path)
        cfg["window_contract_entries"] = entries
        cfg["gof_window_contract_path"] = contract_path
        cfg["gof_window_contract_strict"] = contract_strict
        if entries:
            print(
                f"[INFO] Loaded GOF window contract: {len(entries):,} key(s) from {contract_path} "
                f"(strict={contract_strict})"
            )
        else:
            print(
                f"[WARN] GOF window contract path provided but no keys loaded: {contract_path} "
                f"(strict={contract_strict})"
            )

    # Optional GOFOC-driven window contract for SO-side GOFCC matching.
    oc_contract_path = str(
        cfg.get("gof_oc_window_contract_path")
        or os.environ.get("GOF_OC_WINDOW_CONTRACT_PATH", "")
        or ""
    ).strip()
    oc_strict_cfg = cfg.get("gof_oc_window_contract_strict", None)
    if oc_strict_cfg is None:
        oc_strict_env_raw = os.environ.get(
            "GOF_OC_WINDOW_CONTRACT_STRICT",
            "1" if oc_contract_path else "0",
        )
        oc_contract_strict = str(oc_strict_env_raw).strip().lower() not in (
            "",
            "0",
            "false",
            "no",
            "off",
        )
    else:
        oc_contract_strict = bool(oc_strict_cfg)
    if oc_contract_path:
        oc_entries = _load_window_contract_entries(oc_contract_path)
        cfg["oc_window_contract_entries"] = oc_entries
        cfg["gof_oc_window_contract_path"] = oc_contract_path
        cfg["gof_oc_window_contract_strict"] = oc_contract_strict
        if oc_entries:
            print(
                f"[INFO] Loaded GOF OC->CC window contract: {len(oc_entries):,} key(s) from {oc_contract_path} "
                f"(strict={oc_contract_strict})"
            )
        else:
            print(
                f"[WARN] GOF OC->CC window contract path provided but no keys loaded: {oc_contract_path} "
                f"(strict={oc_contract_strict})"
            )

    # Enhanced error handling and debugging
    print(f"🚀 Starting validation suite with {len(paired_files)} pairs")
    print(f"   - Workers: {cfg['n_workers']}")
    print(f"   - Debug mode: {cfg.get('debug', False)}")

    # Respect external decision from 405 if requested
    if cfg.get("joblib_autotune") == "external" and "joblib_n_jobs" in cfg:
        n_jobs = int(cfg["joblib_n_jobs"])
    else:
        # Use the new auto-tuning function for optimal job count
        n_jobs = _auto_tune_n_jobs(cfg, pairs_count=len(paired_files))

    debug = cfg.get("debug", False)

    try:
        # Launch resilient worker pool
        parallel_start = time.time()

        # Add memory monitoring
        initial_memory = log_stage_memory(
            "DEBUG",
            "Initial memory before parallel processing",
            debug,
        )

        def _cap_threads(n: int) -> None:
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                os.environ[k] = str(n)

        _cap_threads(1)  # always 1 per worker (avoid oversubscribe)

        # Lower priority for launcher (children inherit) and set joblib tmp folder
        set_joblib_low_priority(
            cfg.get("worker_cpu_nice", 15),
            cfg.get("worker_ionice_class", "idle"),
            cfg.get("worker_ionice_prio", 7),
        )
        tmpdir = cfg.get("joblib_tmpdir")
        if tmpdir:
            os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(tmpdir))

        # ---------------------------------------------------------------------
        # FULL FLATTENING: enqueue one job per (pair, recipe, comparison)
        # ---------------------------------------------------------------------
        jobs = []
        reduce_so = _pairing_policy_so(cfg) == "reduced"
        so_anchor_members = _build_observation_anchor_member_map(paired_files) if reduce_so else {}
        for pair in paired_files:
            for recipe in cfg["bench_recipes"]:
                # Base jobs (NC/CC/CZ/ZZ/NN)
                jobs.extend(_build_jobs_for_pair_recipe(pair, recipe, cfg))

                # Optional ERA5 comparisons for 'historical'
                model, scen, ens = pair.get("key", ("", "", ""))
                era5_file = cfg.get("era5_file")
                allow_hist = (scen == "historical") or bool(pair.get("treat_as_historical_for_era5", False))
                has_cmip = bool(pair.get("cmip6_files"))
                has_gcm = bool(pair.get("gcmagicc_files"))
                is_so_anchor = True
                if reduce_so:
                    anchor_ens = so_anchor_members.get((str(model), str(scen)))
                    is_so_anchor = bool(anchor_ens and str(ens) == str(anchor_ens))
                # Only map for historical (per user note) and when toggles are on
                if era5_file and allow_hist:
                    # pick primary CMIP & GCMagicc again (each side optional)
                    main_cmip = pair["cmip6_files"][0]["file"] if has_cmip else None
                    main_gcmagicc = (
                        _pick_oldest_gcmagicc(pair["gcmagicc_files"])
                        or pair["gcmagicc_files"][0]["file"]
                    ) if has_gcm else None
                    gxe_mode = bool(cfg.get("gxe_mode", False))
                    gxe_anchor = cfg.get("gxe_anchor_gcmagicc_file")
                    # GOFOC: CMIP6 vs ERA5 (does NOT require a GCMagicc file)
                    if cfg.get("do_gofoc_crunching", False) and main_cmip:
                        if reduce_so and not is_so_anchor:
                            continue
                        jobs.append(
                            {
                                "pair": pair,
                                "recipe": recipe,
                                "comparison": "oc",
                                "file_a": main_cmip,
                                "file_b": era5_file,
                            }
                        )
                    # GOFON: GCMagicc vs ERA5 (requires a GCMagicc file)
                    if cfg.get("do_gofon_crunching", False) and main_gcmagicc:
                        if gxe_mode and gxe_anchor and (not _same_path(main_gcmagicc, gxe_anchor)):
                            # In observation-side GXE mode, GOFON is computed for the anchor member only.
                            continue
                        if reduce_so and not is_so_anchor:
                            continue
                        jobs.append(
                            {
                                "pair": pair,
                                "recipe": recipe,
                                "comparison": "on",
                                "file_a": main_gcmagicc,
                                "file_b": era5_file,
                            }
                        )

        jobs = _apply_gofoc_guided_cc_so(jobs, cfg)
        jobs = _apply_gofon_guided_nn_so(jobs, cfg)

        # --- Fast pruning using the verified jobs_index sentinel ----------------------
        if not cfg.get("skip_duplication_check", False):
            from .helper_bench_metric import jobs_index_has

            kept = []
            pruned_fast = 0
            for jb in jobs:
                if jobs_index_has(jb, cfg):
                    pruned_fast += 1
                else:
                    kept.append(jb)
            if pruned_fast:
                print(f"🧮 Verified sentinel pruned {pruned_fast} job(s) before row‑key checks")
            jobs = kept
        # -----------------------------------------------------------------------------

        # Optional pre-pruning: skip jobs whose records are already complete.
        # Fast existence checker that supports SQLite and CSV. For SQLite, do a per-job batched lookup.
        # Skip this entirely if skip_duplication_check is enabled
        if cfg.get("skip_duplication_check", False):
            print(
                "🔄 Skipping duplication checks (skip_duplication_check=True) - all jobs will run"
            )
            jobs_rr = jobs
        else:

            def _expected_records_for_job(job) -> list:
                issues = collect_unreadable_job_inputs(job)
                if issues:
                    for _, path, reason in issues:
                        _warn_unreadable_input_once("expected_records_preprune", path, reason)
                    return []
                try:
                    plugin = get_recipe_plugins()[job["recipe"]]
                    if hasattr(plugin, "expected_records"):
                        return (
                            plugin.expected_records(
                                job["file_a"], job["file_b"], cfg, comparison=job["comparison"]
                            )
                            or []
                        )
                except Exception:
                    pass
                return []

            import sqlite3
            import pandas as pd

            csv_cache = {
                "nc": None,
                "cc": None,
                "cz": None,
                "zz": None,
                "nn": None,
                "oc": None,
                "on": None,
            }
            csv_paths = {
                "nc": _NC_CSV,
                "cc": _CC_CSV,
                "cz": _CZ_CSV,
                "zz": _ZZ_CSV,
                "nn": _NN_CSV,
                "oc": _OC_CSV,
                "on": _ON_CSV,
            }

            sqlite_path = cfg.get("sqlite_path") or os.environ.get("METRICS_DB_SQLITE_PATH")
            use_sqlite = bool(sqlite_path and os.path.exists(sqlite_path))
            table_by_comp = {
                "nc": "gofnc",
                "cc": "gofcc",
                "cz": "gofcz",
                "zz": "gofzz",
                "nn": "gofnn",
                "oc": "gofoc",
                "on": "gofon",
            }

            def _chunked(seq, size: int = 750):
                for idx in range(0, len(seq), size):
                    yield seq[idx : idx + size]

            con = None
            if use_sqlite:
                try:
                    con = sqlite3.connect(sqlite_path, timeout=60.0)
                except Exception:
                    con = None

            pending: Dict[str, list[tuple[dict, list[str]]]] = defaultdict(list)
            kept_no_template = []
            pruned: list[dict] = []

            for jb in jobs:
                recs_tpl = _expected_records_for_job(jb)
                if not recs_tpl:
                    kept_no_template.append(jb)
                    continue
                comp = jb.get("comparison", "nc")
                try:
                    row_keys = [_compute_row_key(r, comp) for r in recs_tpl]
                except Exception:
                    kept_no_template.append(jb)
                    continue
                if not row_keys:
                    kept_no_template.append(jb)
                    continue
                pending[comp].append((jb, row_keys))

            existing_by_comp: Dict[str, set[str]] = {}
            if con is not None and pending:
                try:
                    for comp, entries in pending.items():
                        table = table_by_comp.get(comp)
                        if not table:
                            continue
                        keys = sorted({rk for _, row_keys in entries for rk in row_keys})
                        if not keys:
                            continue
                        hits: set[str] = set()
                        for chunk in _chunked(keys):
                            qmarks = ",".join(["?"] * len(chunk))
                            sql = f"SELECT row_key FROM {table} WHERE row_key IN ({qmarks})"
                            try:
                                cur = con.execute(sql, chunk)
                                hits.update(r[0] for r in cur.fetchall())
                            except Exception:
                                hits = set()
                                break
                        existing_by_comp[comp] = hits
                finally:
                    try:
                        con.close()
                    except Exception:
                        pass
            elif con is not None:
                try:
                    con.close()
                except Exception:
                    pass

            for comp in pending.keys():
                if comp in existing_by_comp:
                    continue
                if csv_cache.get(comp) is None:
                    p = csv_paths.get(comp)
                    s = set()
                    try:
                        if p and os.path.exists(p) and os.path.getsize(p) > 0:
                            df = pd.read_csv(p, usecols=["row_key"])
                            if "row_key" in df:
                                s = set(df["row_key"].dropna().astype(str).tolist())
                    except Exception:
                        s = set()
                    csv_cache[comp] = s
                existing_by_comp[comp] = csv_cache.get(comp) or set()

            kept = kept_no_template[:]
            for comp, entries in pending.items():
                existing = existing_by_comp.get(comp)
                for jb, row_keys in entries:
                    if existing and all(rk in existing for rk in row_keys):
                        pruned.append(jb)
                    else:
                        kept.append(jb)

            if pruned:
                print(f"🧹 Pre-pruned {len(pruned)} jobs already complete; executing {len(kept)} jobs")
            jobs = kept

            # Round-robin the jobs by comparison type so we don't start with a wall of ZZs.
            by_type = defaultdict(list)
            for j in jobs:
                by_type[j["comparison"]].append(j)
            order = ["nc", "cc", "cz", "zz", "nn", "oc", "on"]
            jobs_rr = []
            while any(by_type[k] for k in order):
                for k in order:
                    if by_type[k]:
                        jobs_rr.append(by_type[k].pop(0))
        # Always assign through a single path to avoid edge cases
        jobs = jobs_rr

        # Show how many jobs per comparison type (useful to see if NN enqueued)
        counts = Counter(j["comparison"] for j in jobs)
        if counts:
            print(
                "🧾 Job counts: "
                + ", ".join(f"{k.upper()}={v}" for k, v in sorted(counts.items()))
            )

        # --- progress accounting per pair (how many jobs per pair key) ---
        def _pair_key(p):
            try:
                return tuple(p["key"])
            except Exception:
                return str(p)

        all_pair_keys = {_pair_key(pair) for pair in paired_files}
        jobs_per_pair = Counter(_pair_key(j["pair"]) for j in jobs)
        pairs_without_jobs = sorted(all_pair_keys - set(jobs_per_pair.keys()))
        if pairs_without_jobs:
            preview = ", ".join(str(pk) for pk in pairs_without_jobs[:3])
            more = "" if len(pairs_without_jobs) <= 3 else f", … (+{len(pairs_without_jobs) - 3} more)"
            print(
                f"🧮 Skipping {len(pairs_without_jobs)} pair(s); metrics already satisfied: {preview}{more}"
            )
        pairs_total = len(all_pair_keys)
        total_jobs = len(jobs)

        max_retries = int(cfg.get("resilient_max_retries", 1))

        pending_primary = deque({"job": job, "attempt": 0, "max_concurrency": n_jobs} for job in jobs)
        pending_deferred: deque[Dict] = deque()
        deferred_store = DeferredJobStore(cfg.get("failed_queue_db_path"))
        persisted = deferred_store.load_pending()
        if persisted:
            print(
                f"↺ Loaded {len(persisted)} deferred job(s) from crash queue; they will run after the primary queue."
            )
            for item in persisted:
                pending_deferred.append(
                    {
                        "job": item["job"],
                        "attempt": int(item.get("attempt", 0)),
                        "max_concurrency": 1,
                    }
                )
        results_list: List[Dict] = []
        failed_jobs: List[Dict] = []

        # Stall watchdog: if we sit with pending jobs but zero active workers for too long, abort
        stall_timeout_seconds = max(
            0,
            int(
                cfg.get(
                    "stall_timeout_seconds",
                    int(os.environ.get("STALL_TIMEOUT_SECONDS", "900")),
                )
            ),
        )
        stall_poll_seconds = max(
            5,
            int(
                cfg.get(
                    "stall_poll_seconds",
                    int(os.environ.get("STALL_POLL_SECONDS", "30")),
                )
            ),
        )
        last_progress_ts = time.time()
        stall_evt = threading.Event()
        stop_evt = threading.Event()

        # Pretty postfix for NC/CC/CZ/ZZ/NN counters (successful jobs only)
        done_by_type = Counter()
        failed_by_type = Counter()
        failed_codes = Counter()
        done_by_pair = Counter()

        ctx = mp.get_context("spawn")
        active = []  # [{"process": proc, "conn": conn, "job_entry": job_entry}]
        current_max_workers = max(1, n_jobs)
        min_workers = max(1, int(cfg.get("min_jobs", 1)))
        current_max_workers = max(min_workers, current_max_workers)
        # --- NEW: hard/soft caps ---
        hard_cap = max(1, int(cfg.get("max_jobs", current_max_workers)))  # immutable ceiling from env/CLI
        soft_cap = max(current_max_workers, hard_cap)  # starts at hard cap; drops on pressure; recovers slowly
        recovery_enabled = bool(cfg.get("worker_recovery_enabled", True))
        recovery_cooldown = max(0, int(cfg.get("worker_recovery_cooldown_jobs", 0)))
        recovery_cooldown_seconds = max(0, int(cfg.get("worker_recovery_cooldown_seconds", 300)))
        recovery_step = max(1, int(cfg.get("worker_recovery_step", 1)))
        throttle_fraction = min(0.5, max(0.05, float(cfg.get("worker_throttle_fraction", 0.25))))
        throttle_min_drop = max(1, int(cfg.get("worker_throttle_min_drop", 1)))
        success_since_throttle = 0
        throttle_active = False
        last_throttle_ts = 0.0
        logger = logging.getLogger("validation_suite")
        pressure_event = memory_pressure_event

        # Watchdog thread: hard-stop if no progress for too long while work remains.
        def _stall_watchdog():
            while not stop_evt.is_set():
                time.sleep(stall_poll_seconds)
                if not stall_timeout_seconds:
                    continue
                stalled_for = time.time() - last_progress_ts
                pending_count = len(pending_primary) + len(pending_deferred) + len(active)
                if pending_count == 0:
                    continue
                if stalled_for >= stall_timeout_seconds:
                    msg = (
                        f"🛑 Stall watchdog: {pending_count} pending job(s), "
                        f"no progress for {int(stalled_for)}s (threshold {stall_timeout_seconds}s); forcing exit."
                    )
                    try:
                        tqdm.write(msg)
                    except Exception:
                        pass
                    logger.error(msg)
                    stall_evt.set()
                    stop_evt.set()
                    try:
                        import signal, os as _os
                        _os.kill(_os.getpid(), signal.SIGTERM)
                    except Exception:
                        pass
                    return

        def _format_counts() -> Dict[str, str]:
            return {
                "NC": f"{done_by_type['nc']}/{counts.get('nc', 0)}",
                "CC": f"{done_by_type['cc']}/{counts.get('cc', 0)}",
                "CZ": f"{done_by_type['cz']}/{counts.get('cz', 0)}",
                "ZZ": f"{done_by_type['zz']}/{counts.get('zz', 0)}",
                "NN": f"{done_by_type['nn']}/{counts.get('nn', 0)}",
                "OC": f"{done_by_type['oc']}/{counts.get('oc', 0)}",
                "ON": f"{done_by_type['on']}/{counts.get('on', 0)}",
            }

        def _job_label(job: Dict) -> str:
            pair_key = job["pair"].get("key", ("?", "?", "?"))
            return f"{job['comparison'].upper()} {job['recipe']} for pair {pair_key}"

        def _launch_job(job_entry: Dict) -> Dict:
            nonlocal last_progress_ts
            parent_conn, child_conn = ctx.Pipe(duplex=False)
            job_entry["attempt"] += 1
            proc = ctx.Process(
                target=_run_job_subprocess,
                args=(job_entry["job"], cfg, child_conn),
            )
            proc.daemon = False
            proc.start()
            child_conn.close()
            active_entry = {
                "process": proc,
                "conn": parent_conn,
                "job_entry": job_entry,
                "start": time.time(),
            }
            active.append(active_entry)
            last_progress_ts = time.time()
            return active_entry

        parallel_time = None
        completed_pairs_counter = len(pairs_without_jobs)
        suite_completed = False

        try:
            parallel_start = time.time()
            # Start watchdog
            watchdog_thread = threading.Thread(target=_stall_watchdog, daemon=True)
            watchdog_thread.start()
            with (
                tqdm(
                    total=total_jobs, desc="Jobs", unit="job", position=0, leave=True
                ) as pbar_jobs,
                tqdm(
                    total=pairs_total, desc="Pairs", unit="pair", position=1, leave=True
                ) as pbar_pairs,
                tqdm(
                    total=soft_cap,
                    desc="Active",
                    unit="wrk",
                    position=2,
                    leave=True,
                ) as pbar_active,
            ):
                progress_path = (
                    Path(str(cfg.get("progress_file_path")))
                    if cfg.get("progress_file_path")
                    else None
                )
                progress_pairs_total = int(cfg.get("progress_pairs_total", pairs_total))
                completed_pairs_counter = len(pairs_without_jobs)

                def _atomic_write_json(path, data):
                    import json, os, tempfile
                    d = os.path.dirname(path)
                    fd, tmp = tempfile.mkstemp(prefix=".tmp_progress_", dir=d or ".")
                    try:
                        with os.fdopen(fd, "w") as f:
                            json.dump(data, f, indent=2)
                        os.replace(tmp, path)
                    finally:
                        try: os.unlink(tmp)
                        except Exception: pass

                def _merge_counts(base: Counter, delta: Counter) -> Dict[str, int]:
                    merged = Counter()
                    merged.update({k: int(v) for k, v in (base or {}).items()})
                    merged.update({k: int(v) for k, v in (delta or {}).items()})
                    return {k: int(v) for k, v in merged.items()}

                def _load_existing_job_stats(path) -> dict:
                    try:
                        if not path or not Path(path).exists():
                            return {}
                        import json
                        with open(path, "r") as f:
                            meta = json.load(f)
                        return {
                            "jobs_total": int(meta.get("jobs_total", meta.get("total_jobs", 0)) or 0),
                            "jobs_completed": int(meta.get("jobs_completed", meta.get("finished_jobs", meta.get("completed_jobs", 0))) or 0),
                            "jobs_failed": int(meta.get("jobs_failed", 0) or 0),
                            "jobs_total_by_type": {str(k): int(v) for k, v in (meta.get("jobs_total_by_type") or meta.get("total_jobs_by_type") or {}).items()},
                            "jobs_completed_by_type": {str(k): int(v) for k, v in (meta.get("jobs_completed_by_type") or meta.get("finished_jobs_by_type") or meta.get("jobs_done_by_type") or {}).items()},
                            "jobs_failed_by_type": {str(k): int(v) for k, v in (meta.get("jobs_failed_by_type") or {}).items()},
                            "jobs_failed_codes": {str(k): int(v) for k, v in (meta.get("jobs_failed_codes") or {}).items()},
                        }
                    except Exception:
                        return {}

                base_stats = _load_existing_job_stats(progress_path)
                _base_jobs_total = int(base_stats.get("jobs_total", 0))
                _base_jobs_completed = int(base_stats.get("jobs_completed", 0))
                _base_jobs_failed = int(base_stats.get("jobs_failed", 0))
                _base_total_by_type = Counter(base_stats.get("jobs_total_by_type") or {})
                _base_completed_by_type = Counter(base_stats.get("jobs_completed_by_type") or {})
                _base_failed_by_type = Counter(base_stats.get("jobs_failed_by_type") or {})
                _base_failed_codes = Counter(base_stats.get("jobs_failed_codes") or {})

                def _write_progress_file(completed: int, status: Optional[str] = None) -> None:
                    if not progress_path:
                        return
                    try:
                        import json

                        if not progress_path.exists():
                            return
                        with progress_path.open("r") as f:
                            data = json.load(f)
                        data["completed_pairs"] = min(completed, progress_pairs_total)
                        data.setdefault("total_pairs", progress_pairs_total)
                        if status is not None:
                            data["status"] = status
                        elif data.get("status") not in ("completed", "failed"):
                            data["status"] = "running"
                        # Preserve and extend job-level stats so wrappers can report failure fractions.
                        finished_by_type = Counter({k: done_by_type.get(k, 0) + failed_by_type.get(k, 0) for k in set(counts.keys()) | set(failed_by_type.keys())})
                        data["jobs_total"] = int(_base_jobs_total + total_jobs)
                        data["jobs_completed"] = int(_base_jobs_completed + int(pbar_jobs.n))
                        data["jobs_failed"] = int(_base_jobs_failed + len(failed_jobs))
                        data["jobs_total_by_type"] = _merge_counts(_base_total_by_type, Counter(counts))
                        data["jobs_completed_by_type"] = _merge_counts(_base_completed_by_type, finished_by_type)
                        data["jobs_failed_by_type"] = _merge_counts(_base_failed_by_type, failed_by_type)
                        data["jobs_failed_codes"] = _merge_counts(_base_failed_codes, failed_codes)
                        _atomic_write_json(str(progress_path), data)
                    except Exception as exc:
                        tqdm.write(
                            f"⚠️  Failed to update progress file {progress_path}: {exc}"
                        )
                if pairs_without_jobs:
                    pbar_pairs.update(len(pairs_without_jobs))
                    _write_progress_file(completed_pairs_counter)

                def _reduce_workers(reason: str, severity: str = "medium", job_entry: Optional[Dict] = None) -> None:
                    nonlocal current_max_workers, throttle_active, soft_cap, success_since_throttle, last_throttle_ts
                    throttle_active = True
                    if job_entry is not None:
                        job_entry["max_concurrency"] = 1
                    prior_workers = current_max_workers
                    if severity == "severe":
                        drop = max(throttle_min_drop, int(round(prior_workers * 0.5)))
                    elif severity == "high":
                        drop = max(throttle_min_drop, int(round(prior_workers * 0.33)))
                    else:
                        drop = max(throttle_min_drop, int(round(prior_workers * throttle_fraction)))
                    new_workers = max(min_workers, max(1, prior_workers - drop))
                    if new_workers < prior_workers:
                        current_max_workers = new_workers
                        soft_cap = max(min_workers, min(soft_cap, new_workers))
                        pbar_active.total = max(soft_cap, pbar_active.n)
                        pbar_active.refresh()
                        msg = (
                            f"⚖️  Reducing workers → cur={current_max_workers}, soft_cap={soft_cap}, "
                            f"hard_cap={hard_cap} after {reason}"
                        )
                        tqdm.write(msg)
                        logger.warning("%s", msg)
                    last_throttle_ts = time.time()
                    success_since_throttle = 0

                def handle_completion(job_entry: Dict, status: str, payload: Dict, exitcode: Optional[int]) -> None:
                    nonlocal current_max_workers, completed_pairs_counter, success_since_throttle, throttle_active, soft_cap, last_progress_ts

                    job = job_entry["job"]
                    pair_key = _pair_key(job["pair"])
                    label = _job_label(job)
                    max_attempts = max_retries + 1

                    def _mark_finished(record: Dict, success: bool) -> None:
                        nonlocal completed_pairs_counter
                        done_by_pair[pair_key] += 1
                        if done_by_pair[pair_key] == jobs_per_pair[pair_key]:
                            pbar_pairs.update(1)
                            completed_pairs_counter += 1
                            _write_progress_file(completed_pairs_counter)
                        pbar_jobs.update(1)
                        if success:
                            done_by_type[job["comparison"]] += 1
                        pbar_jobs.set_postfix(_format_counts(), refresh=False)
                        results_list.append(record)
                        last_progress_ts = time.time()

                    if status == "ok":
                        res = payload.get("result", {})
                        if "pair" not in res:
                            res = {**res, "pair": job["pair"]}
                        if "comparison" not in res:
                            res["comparison"] = job["comparison"]
                        if "recipe" not in res:
                            res["recipe"] = job["recipe"]
                        record = {**res, "status": "ok"}
                        _mark_finished(record, success=True)
                        deferred_store.mark_resolved(job)
                        if recovery_enabled and throttle_active:
                            success_since_throttle += 1
                            # gate on BOTH time and job-count cooldowns
                            time_ok = (recovery_cooldown_seconds == 0) or ((time.time() - last_throttle_ts) >= recovery_cooldown_seconds)
                            jobs_ok = (recovery_cooldown == 0) or (success_since_throttle >= recovery_cooldown)
                            if time_ok and jobs_ok:
                                # First relax the soft cap a bit (10% or +step), then lift current workers up to the soft cap.
                                new_soft = min(hard_cap, max(soft_cap + recovery_step, int(round(soft_cap * 1.10))))
                                if new_soft > soft_cap:
                                    soft_cap = new_soft
                                    pbar_active.total = max(soft_cap, pbar_active.n)
                                    pbar_active.refresh()
                                    logger.info("🪄 Recovery: soft_cap → %d (hard_cap=%d)", soft_cap, hard_cap)
                                new_cur = min(soft_cap, current_max_workers + recovery_step)
                                if new_cur > current_max_workers:
                                    current_max_workers = new_cur
                                    pbar_active.refresh()
                                    msg = (f"⚙️  Increasing workers → cur={current_max_workers}, "
                                           f"soft_cap={soft_cap}, hard_cap={hard_cap} after {success_since_throttle} stable jobs")
                                    tqdm.write(msg)
                                    logger.info(msg)
                                # reset job-count gate; keep time gate for the next lift
                                success_since_throttle = 0
                                if soft_cap >= hard_cap and current_max_workers >= soft_cap:
                                    throttle_active = False
                        else:
                            success_since_throttle = 0
                        tqdm.write(
                            f"✓ {label} ({record.get('n', 0)} records)"
                        )
                        _write_progress_file(completed_pairs_counter)
                        return

                    # Determine retry eligibility
                    message = payload.get("error") or payload.get("message")
                    message = message or (
                        f"exit code {exitcode}" if exitcode is not None else "unknown error"
                    )

                    should_retry = job_entry["attempt"] < max_attempts

                    if should_retry:
                        success_since_throttle = 0
                        if status in {"memory_error", "system_exit", "oom_kill", "signal_exit"}:
                            _reduce_workers(status, severity="severe", job_entry=job_entry)
                        elif status in {"worker_pipe_eof", "worker_pipe_error", "crash"}:
                            _reduce_workers(status, job_entry=job_entry)
                        # Always pin problem jobs to serial and defer them until after the primary queue
                        job_entry["max_concurrency"] = 1
                        pending_deferred.append(job_entry)
                        deferred_store.record_retry(job_entry, status, message, exitcode)
                        retry_msg = (
                            f"↻ Retrying {label} later (attempt {job_entry['attempt']} of {max_attempts}) "
                            f"due to {status}: {message}"
                        )
                        tqdm.write(retry_msg)
                        logger.warning(retry_msg)
                        return

                    failure_record = {
                        "status": status,
                        "recipe": job["recipe"],
                        "comparison": job["comparison"],
                        "pair": job["pair"],
                        "error": message,
                        "exitcode": exitcode,
                        "attempts": job_entry["attempt"],
                    }
                    if "traceback" in payload:
                        failure_record["traceback"] = payload["traceback"]
                    _mark_finished(failure_record, success=False)
                    failed_by_type[job["comparison"]] += 1
                    code_key = None
                    if exitcode is not None:
                        code_key = str(exitcode)
                    elif "signal" in payload:
                        code_key = f"signal_{payload.get('signal')}"
                    else:
                        code_key = payload.get("status") or "unknown"
                    failed_codes[code_key] += 1
                    failed_jobs.append(failure_record)
                    deferred_store.mark_failed(job, message)
                    success_since_throttle = 0
                    fail_msg = f"❌ {label} failed ({status}): {message}"
                    tqdm.write(fail_msg)
                    logger.error(fail_msg)
                    _write_progress_file(completed_pairs_counter)

                def _can_start(job_entry: Dict) -> bool:
                    allowed = min(current_max_workers, job_entry.get("max_concurrency", current_max_workers))
                    if len(active) >= allowed:
                        return False
                    # honor cool-down for memory-heavy retries
                    until = job_entry.get("cooldown_until")
                    if until and time.time() < float(until):
                        return False
                    return True

                # Main scheduling loop
                def _dequeue_startable(queue: deque) -> Optional[Dict]:
                    if not queue:
                        return None
                    rotations = 0
                    length = len(queue)
                    while rotations < length:
                        job_entry = queue[0]
                        if _can_start(job_entry):
                            queue.popleft()
                            return job_entry
                        queue.rotate(-1)
                        rotations += 1
                    return None

                while pending_primary or pending_deferred or active:
                    started = False
                    while len(active) < current_max_workers:
                        if pending_primary:
                            job_entry = _dequeue_startable(pending_primary)
                            if job_entry is None:
                                break
                        else:
                            job_entry = _dequeue_startable(pending_deferred)
                            if job_entry is None:
                                break
                        active_entry = _launch_job(job_entry)
                        pbar_active.n = len(active)
                        pbar_active.refresh()
                        started = True

                    for entry in active[:]:
                        conn = entry["conn"]
                        if "message" not in entry and conn.poll():
                            try:
                                entry["message"] = conn.recv()
                            except EOFError:
                                entry["message"] = {
                                    "status": "worker_pipe_eof",
                                    "error": "Worker pipe closed before status message",
                                }
                                logger.warning(
                                    "Worker pipe EOF for job %s (attempt %d)",
                                    _job_label(entry["job_entry"]["job"]),
                                    entry["job_entry"]["attempt"],
                                )
                            except (BrokenPipeError, ConnectionResetError) as exc:
                                entry["message"] = {
                                    "status": "worker_pipe_error",
                                    "error": f"Worker pipe error: {exc}",
                                }
                                logger.warning(
                                    "Worker pipe error for job %s (attempt %d): %s",
                                    _job_label(entry["job_entry"]["job"]),
                                    entry["job_entry"]["attempt"],
                                    exc,
                                )

                        proc = entry["process"]
                        if not proc.is_alive():
                            proc.join(timeout=0.1)
                            payload = entry.pop("message", None) or {}
                            conn.close()
                            active.remove(entry)
                            pbar_active.n = len(active)
                            pbar_active.refresh()

                            exitcode = proc.exitcode
                            status = payload.get("status")
                            if status is None:
                                if exitcode == 0:
                                    status = "ok"
                                elif exitcode is not None and exitcode < 0:
                                    payload.setdefault(
                                        "error",
                                        f"Process terminated by signal {-exitcode}",
                                    )
                                    payload["signal"] = -exitcode
                                    status = "signal_exit"
                                elif exitcode in (137, 134):
                                    status = "oom_kill" if exitcode == 137 else "abort"
                                elif exitcode == 2:
                                    status = "system_exit"
                                else:
                                    status = "crash"

                            last_progress_ts = time.time()
                            handle_completion(entry["job_entry"], status, payload, exitcode)

                    if pressure_event and pressure_event.is_set():
                        _reduce_workers("memory_pressure_event", severity="severe")
                        pressure_event.clear()
                        continue

                    if stall_evt.is_set():
                        raise RuntimeError("stalled_no_progress_watchdog")

                    # Stall watchdog: no active workers and pending jobs for too long -> bail so wrapper can resume
                    if stall_timeout_seconds and (pending_primary or pending_deferred) and not active:
                        stalled_for = time.time() - last_progress_ts
                        if stalled_for >= stall_timeout_seconds:
                            pending_count = len(pending_primary) + len(pending_deferred)
                            msg = (
                                f"🛑 Detected scheduler stall: {pending_count} pending job(s), "
                                f"0 active for {int(stalled_for)}s. Aborting so wrapper can resume."
                            )
                            tqdm.write(msg)
                            logger.error(msg)
                            raise RuntimeError("stalled_no_workers")

                    if not started:
                        time.sleep(min(0.1, stall_poll_seconds))

                parallel_time = time.time() - parallel_start
                stop_evt.set()
                watchdog_thread.join(timeout=1.0)

        except KeyboardInterrupt:
            tqdm.write("⚠️  Keyboard interrupt received; terminating worker processes...")
            for entry in active:
                try:
                    entry["process"].terminate()
                except Exception:
                    pass
            raise

        if parallel_time is None:
            parallel_time = time.time() - parallel_start
        stop_evt.set()
        try:
            watchdog_thread.join(timeout=1.0)
        except Exception:
            pass

        if debug:
            print(f"⏱️  Parallel processing took {parallel_time:.2f}s")
            log_stage_memory(
                "DEBUG",
                "Memory after parallel processing",
                debug,
                baseline=initial_memory,
            )

        if failed_jobs:
            # Leave the run resumable; do not mark completed
            _write_progress_file(completed_pairs_counter, "running")
            tqdm.write("❗ Incomplete batch: some jobs will be retried by the wrapper.")
            if return_status:
                return False
            else:
                raise RuntimeError(f"{len(failed_jobs)} validation job(s) failed")
        else:
            # Only mark 'completed' if we truly finished every pair we promised
            if completed_pairs_counter >= progress_pairs_total:
                _write_progress_file(progress_pairs_total, "completed")
            else:
                _write_progress_file(completed_pairs_counter, "partial")
            suite_completed = True

        total_time = time.time() - suite_start_time
        print(f"✅ Validation suite completed in {total_time:.2f}s")
        # (Counts are now printed per-job; totals above removed for simplicity)

    except Exception as e:
        print(f"❌ Error in validation suite: {e}")
        try:
            if "_write_progress_file" in locals():
                _write_progress_file(completed_pairs_counter, "running")
        except Exception as progress_exc:
            try:
                (logger if "logger" in locals() else logging.getLogger("validation_suite")).error(
                    "Failed to persist progress after error: %s", progress_exc
                )
            except Exception:
                pass
        import traceback

        traceback.print_exc()
        try:
            (logger if "logger" in locals() else logging.getLogger("validation_suite")).exception(
                "Validation suite crashed: %s", e
            )
        except Exception:
            pass
        if return_status:
            return False
        raise


# (Entire z-score post-processing function removed)


# ---------------------------------------------------------------------------
# Flattened job builder and runner
# ---------------------------------------------------------------------------
def _build_jobs_for_pair_recipe(pair, recipe_name, cfg):
    """Return a list of job dicts for (pair, recipe) across comparison types."""
    jobs = []
    reduce_sc = _pairing_policy_sc(cfg) == "nc_full_cc_zz_nn_single"
    model, scenario, ensemble = pair["key"]
    cmip_root = cfg.get("cmip6folder", "../../data/out_ETHFOG_10June2025_vetted")
    has_cmip = bool(pair.get("cmip6_files"))

    def _cmip_path(path: str) -> str:
        # If the provided path is missing (e.g., stale variable bundle), re-resolve by prefix.
        if os.path.exists(path):
            return path
        return resolve_cmip6_member_file(model, scenario, ensemble, cmip_root, prefer=path)

    main_cmip = _cmip_path(pair["cmip6_files"][0]["file"]) if has_cmip else None
    # Choose the *oldest* GCMagicc file among provided candidates for stable NC baseline
    main_gcmagicc = None
    if pair.get("gcmagicc_files"):
        main_gcmagicc = (
            _pick_oldest_gcmagicc(pair["gcmagicc_files"]) or pair["gcmagicc_files"][0]["file"]
        )

    # (1) CMIP6 × GCMagicc (NC) - only enqueue if requested
    if cfg.get("do_gofnc_crunching", True) and has_cmip and main_gcmagicc:
        jobs.append(
            {
                "pair": pair,
                "recipe": recipe_name,
                "comparison": "nc",
                "file_a": main_cmip,
                "file_b": main_gcmagicc,
            }
        )

    # (2) CMIP6 member-member
    if cfg.get("do_gofcc_crunching", True) and has_cmip and main_cmip:
        cc_candidates = sorted(discover_other_member_files(main_cmip, cmip_root))
        # Optional cap on number of other ensembles for CC
        lim = cfg.get("limit_to_max_CorZ_ens")
        if reduce_sc and not (isinstance(lim, int) and lim > 0):
            lim = 1
        if isinstance(lim, int) and lim > 0:
            cc_candidates = cc_candidates[:lim]
        if cfg.get("debug", False):
            print(f"   🔎 CC candidates for {model} {scenario} {ensemble}: {len(cc_candidates)}")
        for xprime in cc_candidates:
            xm, xs, _ = parse_filename(os.path.basename(xprime))
            if xm == model and xs == scenario:
                jobs.append(
                    {
                        "pair": pair,
                        "recipe": recipe_name,
                        "comparison": "cc",
                        "file_a": main_cmip,
                        "file_b": _cmip_path(xprime),
                    }
                )

    # (3) Same-scenario, different model (CZ)
    if cfg.get("do_gofcz_crunching", True) and has_cmip and main_cmip:
        for z in discover_other_model_files(
            main_cmip, cmip_root, max_models=cfg.get("cz_max_models", 5)
        ):
            zm, zs, _ = parse_filename(os.path.basename(z))
            if zs == scenario and zm != model:
                jobs.append(
                    {
                        "pair": pair,
                        "recipe": recipe_name,
                        "comparison": "cz",
                        "file_a": main_cmip,
                        "file_b": _cmip_path(z),
                    }
                    )

    # (4) Z-within-Z comparisons within scenario (ZZ)
    if cfg.get("do_gofzz_crunching", True) and has_cmip and main_cmip:
        # Get Z models from random_Z_lookup for Z-within-Z comparisons
        z_list = sorted([
            z
            for z in discover_other_model_files(
                main_cmip, cmip_root, max_models=cfg.get("zz_max_models", 5)
            )
            if parse_filename(os.path.basename(z))[1] == scenario
            and parse_filename(os.path.basename(z))[0] != model
        ])

        # Z-within-Z comparisons (same model, different members)
        # Use the same logic as GOFCC but for Z models from random_Z_lookup
        for z_file in z_list:
            z_model, z_scenario, z_ensemble = parse_filename(os.path.basename(z_file))

            # Find other members of the same Z model
            z_other_members = sorted(discover_other_member_files(z_file, cmip_root))
            # Optional cap on number of other ensembles per Z model
            lim = cfg.get("limit_to_max_CorZ_ens")
            if reduce_sc and not (isinstance(lim, int) and lim > 0):
                lim = 1
            if isinstance(lim, int) and lim > 0:
                z_other_members = z_other_members[:lim]

            for z_prime in z_other_members:
                z_prime_model, z_prime_scenario, z_prime_ensemble = parse_filename(
                    os.path.basename(z_prime)
                )

                # Ensure same model and scenario, different ensemble
                if (
                    z_prime_model == z_model
                    and z_prime_scenario == z_scenario
                    and z_prime_ensemble != z_ensemble
                ):
                    jobs.append(
                        {
                            "pair": pair,
                            "recipe": recipe_name,
                            "comparison": "zz",
                            "file_a": _cmip_path(z_file),
                            "file_b": _cmip_path(z_prime),
                        }
                    )

    # (5) GCMagicc member-member
    if cfg.get("do_gofnn_crunching", True) and main_gcmagicc:
        # Ensure NN uses the same oldest GCMagicc as the *baseline*
        main_gcmagicc_file = main_gcmagicc
        gxe_mode = bool(cfg.get("gxe_mode", False))
        gxe_anchor = cfg.get("gxe_anchor_gcmagicc_file")
        gxe_pair = cfg.get("gxe_nn_companion_gcmagicc_file")

        if gxe_mode and gxe_anchor and gxe_pair:
            # Observation-side GXE policy: exactly one directed pair (anchor -> companion).
            if _same_path(main_gcmagicc_file, gxe_anchor):
                _, nscen, _ = parse_filename(os.path.basename(gxe_pair), use_pseudo_member=False)
                if nscen == scenario:
                    jobs.append(
                        {
                            "pair": pair,
                            "recipe": recipe_name,
                            "comparison": "nn",
                            "file_a": main_gcmagicc_file,
                            "file_b": gxe_pair,
                        }
                    )
            elif cfg.get("debug", False):
                print(
                    "   🔎 NN skip (GXE policy): non-anchor baseline "
                    f"{os.path.basename(main_gcmagicc_file)}"
                )
        else:
            nn_max_members = cfg.get("nn_max_members")
            if not isinstance(nn_max_members, int) or nn_max_members <= 0:
                nn_max_members = 1 if reduce_sc else 5
            nn_candidates = discover_other_gcmagicc_files(
                main_gcmagicc_file, max_members=nn_max_members
            )
            if cfg.get("debug", False):
                print(f"   🔎 NN baseline: {os.path.basename(main_gcmagicc_file)}")
                print(f"   🔎 NN candidates found: {len(nn_candidates)}")
                for c in nn_candidates:
                    print(f"      - {os.path.basename(c)}")
            for gcmagicc_other in nn_candidates:
                _, nscen, _ = parse_filename(os.path.basename(gcmagicc_other))
                if nscen != scenario:
                    continue
                jobs.append(
                    {
                        "pair": pair,
                        "recipe": recipe_name,
                        "comparison": "nn",
                        "file_a": main_gcmagicc_file,
                        "file_b": gcmagicc_other,
                    }
                )

    return jobs


def _normalize_records_for_db(recs, job, cfg=None):
    """
    Enforce consistent ID rules across recipes just before DB write.
    Rules:
      * GOFNC: comp_source_id = <GCMagiccPrefix>_<CMIPmodel>; member ids stay plain CMIP (e.g. r1i1p1f2).
      * GOFNN:  source_id     = <GCMagiccPrefixA>_<CMIPmodel>
                comp_source_id= <GCMagiccPrefixB>_<CMIPmodel>
                member_id / comp_member_id = plain CMIP member (no '__GCMagicc...' suffix).
    """
    import os

    fa = os.path.basename(job["file_a"])
    fb = os.path.basename(job["file_b"])

    if job["comparison"] == "nc":
        cm_model, cm_scen, cm_mem_plain = parse_filename(fa, use_pseudo_member=False)
        gcmagicc_prefix = extract_gcmagicc_code(fb) or ""
        for r in recs:
            # Ensure the GCMagicc comp_source_id carries the CMIP model suffix
            comp_sid = r.get("comp_source_id", "")
            if comp_sid.startswith("GCMagicc") and f"_{cm_model}" not in comp_sid:
                r["comp_source_id"] = f"{gcmagicc_prefix}_{cm_model}"
            # Keep plain members (both sides use the same CMIP member for NC)
            r["member_id"] = cm_mem_plain
            r["comp_member_id"] = cm_mem_plain

    elif job["comparison"] == "nn":
        # Both files are GCMagicc; the underlying CMIP model must be the same
        cm_model_a, _, cm_mem_plain_a = parse_filename(fa, use_pseudo_member=False)
        cm_model_b, cm_scen_b, cm_mem_plain_b = parse_filename(fb, use_pseudo_member=False)
        gcmagicc_a = extract_gcmagicc_code(fa) or ""
        gcmagicc_b = extract_gcmagicc_code(fb) or ""
        # In practice cm_model_a == cm_model_b and cm_mem_plain_a == cm_mem_plain_b
        for r in recs:
            r["source_id"] = f"{gcmagicc_a}_{cm_model_a}"
            r["comp_source_id"] = f"{gcmagicc_b}_{cm_model_a}"
            # Strip any '__GCMagicc-...' pseudo suffixes from member ids
            mid = str(r.get("member_id", ""))
            cmid = str(r.get("comp_member_id", ""))
            r["member_id"] = mid.split("__", 1)[0] if "__" in mid else cm_mem_plain_a
            r["comp_member_id"] = cmid.split("__", 1)[0] if "__" in cmid else cm_mem_plain_b

    elif job["comparison"] == "oc":
        # ERA5 × CMIP6
        cm_model, cm_scen, cm_mem_plain = parse_filename(fa, use_pseudo_member=False)
        era_src, era_exp, era_mem = parse_filename(fb, use_pseudo_member=False)
        for r in recs:
            r["source_id"] = cm_model
            r["member_id"] = cm_mem_plain
            r["comp_source_id"] = "ERA5"
            r["comp_member_id"] = era_mem or "r1i1p1f1"
            # normalize experiment label: treat ERA5's 'historical-ERA5' as 'historical'
            r["experiment_id"] = cm_scen if cm_scen != "historical-ERA5" else "historical"

    elif job["comparison"] == "on":
        # ERA5 × GCMagicc
        cm_model_a, cm_scen_a, cm_mem_plain_a = parse_filename(fa, use_pseudo_member=False)
        gcmagicc_a = extract_gcmagicc_code(fa) or ""
        era_src, era_exp, era_mem = parse_filename(fb, use_pseudo_member=False)
        for r in recs:
            r["source_id"] = f"{gcmagicc_a}_{cm_model_a}"
            r["member_id"] = cm_mem_plain_a
            r["comp_source_id"] = "ERA5"
            r["comp_member_id"] = era_mem or "r1i1p1f1"
            if cm_scen_a and not cm_scen_a.startswith("historical"):
                r["experiment_id"] = "historical"
            else:
                r["experiment_id"] = cm_scen_a if cm_scen_a != "historical-ERA5" else "historical"

    return recs


def _run_job_subprocess(job, cfg, conn):
    """Wrapper executed in a child process to isolate crashes/OOMs per job."""
    import sys
    import os
    active_cap_gb = None
    # Apply per-worker memory caps using RLIMIT_AS so only this process is culled.
    try:
        import resource  # POSIX only; raises ImportError on unsupported platforms

        caps_cfg = cfg.get("recipe_memory_caps_gb") or {}
        default_cap = cfg.get("default_worker_memory_cap_gb")
        recipe_key = job.get("recipe")

        # Allow floats or strings, fall back to default when mapping missing.
        configured_cap = caps_cfg.get(recipe_key, default_cap)
        if configured_cap:
            active_cap_gb = float(configured_cap)
            if active_cap_gb > 0:
                cap_bytes = int(active_cap_gb * (1024 ** 3))
                current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
                # Only tighten the limit; never raise above existing hard limit (if set).
                new_soft = min(cap_bytes, current_hard if current_hard > 0 else cap_bytes)
                new_hard = min(cap_bytes, current_hard) if current_hard > 0 else cap_bytes
                resource.setrlimit(resource.RLIMIT_AS, (new_soft, new_hard))
                print(
                    f"🛡️  Worker memory cap active: {active_cap_gb:.1f} GB (RLIMIT_AS) for {recipe_key}",
                    flush=True,
                )
    except ImportError:
        pass  # Non-POSIX platform; RLIMIT_AS unavailable.
    except (ValueError, OSError) as cap_exc:
        print(
            f"⚠️  Could not apply memory cap for {job.get('recipe')} ({cap_exc})",
            flush=True,
        )

    # CRITICAL: Ensure sys.path is set up correctly in worker process
    # This must match the path setup in the main script
    try:
        current_file = __file__
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
    except Exception:
        pass  # Fail silently if path setup doesn't work
    
    try:
        result = _run_one_job(job, cfg)
        
        # CRITICAL: Flush all output before sending status
        # This ensures debug prints are captured even if process dies
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        
        # CRITICAL: Send status message IMMEDIATELY after job completes
        # before any cleanup that might fail or cause OOM
        try:
            conn.send({"status": "ok", "result": result})
            conn.close()
        except (BrokenPipeError, OSError) as e:
            # Parent might have given up - log it but don't crash
            try:
                pair_key = job["pair"].get("key", str(job["pair"]))
                print(f"⚠️  Failed to send result to parent for {pair_key}: {e}", file=sys.stderr)
                sys.stderr.flush()
            except Exception:
                pass
    except SystemExit as exc:
        try:
            conn.send({"status": "system_exit", "exitcode": exc.code})
            conn.close()
        except Exception:
            pass
        raise
    except MemoryError as exc:
        message = str(exc).strip()
        if not message:
            if active_cap_gb:
                message = (
                    f"Worker exceeded memory cap ({active_cap_gb:.1f} GB via RLIMIT_AS)"
                )
            else:
                message = "Worker hit MemoryError"
        try:
            payload = {"status": "memory_error", "error": message}
            if active_cap_gb:
                payload["memory_cap_gb"] = active_cap_gb
            conn.send(payload)
            conn.close()
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover - defensive
        try:
            conn.send(
                {
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            conn.close()
        except Exception:
            pass


def _run_one_job(job, cfg):
    """Execute exactly one comparison job and persist results."""
    import gc
    import os
    import xarray as xr
    _install_xarray_engine_override()

    os.environ["IN_JOBLIB_CONTEXT"] = "1"  # prevent inner parallelism

    # Memory monitoring for individual jobs
    debug = cfg.get("debug", False)

    # Ensure worker low priority (inheritance can be flaky on some schedulers)
    set_low_priority(
        enable=True,
        cpu_nice=int(cfg.get("worker_cpu_nice", 15)),
        io_class=str(cfg.get("worker_ionice_class", "idle")),
        io_priority=int(cfg.get("worker_ionice_prio", 7)),
    )
    # Ensure duplication/overwrite flags are applied in this worker process.
    # Use helper_bench_metric setters so module-level globals update too.
    force_overwrite = bool(cfg.get("force_overwrite") or cfg.get("skip_duplication_check"))
    skip_duplication_check = bool(cfg.get("skip_duplication_check"))
    try:
        _metric_set_force_overwrite(force_overwrite)
    except Exception:
        if force_overwrite:
            os.environ["METRICS_FORCE_OVERWRITE"] = "1"
        else:
            os.environ.pop("METRICS_FORCE_OVERWRITE", None)
    try:
        _metric_set_skip_duplication_check(skip_duplication_check)
    except Exception:
        if skip_duplication_check:
            os.environ["METRICS_SKIP_DUP_CHECK"] = "1"
        else:
            os.environ.pop("METRICS_SKIP_DUP_CHECK", None)

    # Worker RSS guard - bail out before the kernel OOM-kills us
    try:
        import psutil as _ps

        _proc = _ps.Process(os.getpid())
        _rss_limit_mb = int(cfg.get("worker_rss_abort_mb", 0)) or 0

        def _rss_guard(where: str):
            if _rss_limit_mb:
                rss = _proc.memory_info().rss / (1024 * 1024)
                if rss > _rss_limit_mb:
                    print(f"⚠️  Aborting job (RSS {rss:.0f} MB > {_rss_limit_mb} MB) at {where}")
                    raise MemoryError(f"Worker RSS exceeded at {where}")
    except Exception:

        def _rss_guard(where: str):  # no-op if psutil missing
            return

    # Identify / announce job start
    pk = job["pair"].get("key", ("?", "?", "?"))
    job_id = f"{job['recipe']}|{job['comparison']}|{pk[0]}-{pk[1]}-{pk[2]}"
    q = cfg.get("progress_q", None)
    if q:
        try:
            q.put(("start", {"id": job_id}), block=False)
        except Exception:
            pass

    # DEBUG: Add job tracking
    debug = cfg.get("debug", False)
    if debug:
        pair_key = job["pair"].get("key", str(job["pair"]))
        print(
            f"🔍 DEBUG: Starting job - {job['comparison'].upper()} {job['recipe']} for pair {pair_key}"
        )
        print(f"   📁 File A: {os.path.basename(job['file_a'])}")
        print(f"   📁 File B: {os.path.basename(job['file_b'])}")

    job_start_memory = log_stage_memory(
        "DEBUG",
        "Job memory at start",
        debug,
    )

    # Ensure DB backends are configured inside each worker
    try:
        _metric_configure_db(cfg.get("db_backends", []), cfg.get("sqlite_path"))
    except Exception:
        pass

    # Ensure recipe cache is initialized in this worker process
    clear_recipe_cache()
    plugins = get_recipe_plugins()
    if job["recipe"] not in plugins:
        raise KeyError(
            f"Recipe '{job['recipe']}' not found in available plugins: {list(plugins.keys())}"
        )
    plugin = plugins[job["recipe"]]
    # keep xarray file cache tiny to reduce FD & mem churn
    try:
        import xarray as xr

        xr.set_options(file_cache_maxsize=1, keep_attrs=False, cache=False)
    except Exception:
        pass

    # Ensure no pseudo-member leakage into DB columns
    _old_flag = os.environ.pop("PSEUDO_NICOLAI_MEMBER", None)
    try:
        issues = collect_unreadable_job_inputs(job)
        if issues:
            msg = (
                "Skipping job due to unreadable NetCDF input(s): "
                f"{format_unreadable_input_issues(issues)}"
            )
            logging.getLogger("validation_suite").warning("%s", msg)
            raise OSError(msg)

        # DEBUG: Track function call
        if debug:
            print(f"🔍 DEBUG: Calling plugin.gof() for {job['comparison']} comparison")

        cfg_job = _cfg_with_nn_gofon_period(cfg, job)
        cfg_job = _cfg_with_nn_window_contract(cfg_job, job)
        cfg_job = _cfg_with_cc_window_contract(cfg_job, job)
        if debug and cfg_job is not cfg:
            comp_dbg = str(job.get("comparison", "")).strip().lower()
            if comp_dbg == "nn":
                print(
                    "🔍 DEBUG: NN window lock from GOFON period "
                    f"{cfg_job.get('nn_forced_year_start')}..{cfg_job.get('nn_forced_year_end')}"
                )
            if bool(cfg_job.get("window_contract_active", False)):
                print(
                    f"🔍 DEBUG: {comp_dbg.upper()} contract "
                    f"matches={cfg_job.get('window_contract_match_count', 0)} "
                    f"windows={cfg_job.get('forced_window_labels', [])}"
                )

        _rss_guard("before_plugin_call")
        recs = (
            plugin.gof(job["file_a"], job["file_b"], cfg_job, comparison=job["comparison"])
            if hasattr(plugin, "gof")
            else []
        )
        _rss_guard("after_plugin_call")
        recs = _normalize_records_for_db(recs or [], job, cfg_job)
        recs = _apply_window_contract_filter(recs, cfg_job, job)

        # DEBUG: Track results and memory
        if debug:
            print(f"🔍 DEBUG: plugin.gof() returned {len(recs)} records")
            if len(recs) > 0:
                print(f"   📊 Sample record: {recs[0]}")

            # Memory tracking after plugin execution
            job_after_plugin_memory = get_memory_usage_mb()
            if (
                job_after_plugin_memory is not None
                and job_start_memory is not None
            ):
                plugin_memory_change = job_after_plugin_memory - job_start_memory
                print(
                    f"🔍 DEBUG: Memory after plugin execution: {job_after_plugin_memory:.1f} MB (+{plugin_memory_change:+.1f} MB)"
                )

                if plugin_memory_change > 200:  # More than 200MB increase
                    print("⚠️  DEBUG: WARNING: Large memory increase in plugin execution!")

    finally:
        # restore env if there was a previous value
        if _old_flag is not None:
            os.environ["PSEUDO_NICOLAI_MEMBER"] = _old_flag

    if recs:
        # DEBUG: Track database writing
        if debug:
            print(
                f"🔍 DEBUG: Writing {len(recs)} records to database for {job['comparison']} comparison"
            )

        # write will delete+insert when METRICS_FORCE_OVERWRITE=1 (see helper_bench_metric patch)
        write_records_unified(recs, job["comparison"], cfg.get("version_tag"))
        jobs_index_mark(job, cfg, len(recs), version_tag=cfg.get("version_tag"))

        # DEBUG: Verify database write
        if debug:
            from .helper_bench_metric import (
                _NN_CSV,
                _NC_CSV,
                _CC_CSV,
                _CZ_CSV,
                _ZZ_CSV,
                _OC_CSV,
                _ON_CSV,
            )

            db_paths = {
                "nc": _NC_CSV,
                "cc": _CC_CSV,
                "cz": _CZ_CSV,
                "zz": _ZZ_CSV,
                "nn": _NN_CSV,
                "oc": _OC_CSV,
                "on": _ON_CSV,
            }
            db_path = db_paths.get(job["comparison"])
            if db_path and os.path.exists(db_path):
                size = os.path.getsize(db_path)
                print(f"🔍 DEBUG: Database {job['comparison']} size after write: {size} bytes")
            else:
                print(f"🔍 DEBUG: Database {job['comparison']} file does not exist after write")

    # Enhanced cleanup
    plt.close("all")
    xr.backends.file_manager.FILE_CACHE.clear()

    # Clear xarray options to free memory
    try:
        xr.set_options(keep_attrs=False)
    except Exception:
        pass

    # Force multiple garbage collection passes
    gc.collect()
    gc.collect()
    gc.collect()

    # Clear matplotlib cache
    try:
        plt.rcdefaults()
    except Exception:
        pass

    # Memory monitoring for individual jobs
    if debug:
        job_end_memory = get_memory_usage_mb()
        if job_end_memory is not None and job_start_memory is not None:
            job_memory_change = job_end_memory - job_start_memory
            if job_memory_change > 100:  # More than 100MB increase
                print(
                    f"🔍 DEBUG: Job memory change: {job_memory_change:+.1f} MB (start: {job_start_memory:.1f} MB, end: {job_end_memory:.1f} MB)"
                )

    # DEBUG: Final job summary
    if debug:
        pair_key = job["pair"].get("key", str(job["pair"]))
        print(
            f"🔍 DEBUG: Completed job - {job['comparison'].upper()} {job['recipe']} for pair {pair_key} ({len(recs)} records)"
        )

    res = {
        "pair": job["pair"],
        "recipe": job["recipe"],
        "comparison": job["comparison"],
        "n": len(recs),
    }
    if q:
        try:
            q.put(("done", {"id": job_id, "ok": bool(recs)}), block=False)
        except Exception:
            pass
    
    # CRITICAL: Aggressive cleanup before returning to minimize memory during status send
    # Delete large objects that might be lingering
    try:
        del recs
    except Exception:
        pass
    
    # One more GC pass to ensure cleanup before returning
    gc.collect()
    
    return res
