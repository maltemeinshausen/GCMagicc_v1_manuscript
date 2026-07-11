# helper_bench_metric.py
#
# Generic metrics + CSV persistence using an explicit column schema
# -----------------------------------------------------------------
from __future__ import annotations

import os
import glob
import datetime as dt
import logging
import sqlite3
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, filtfilt

import json
import random
import tempfile
import shutil

from .helper_bench_plot import compute_rmse_score, parse_filename
from .helper_debug import get_memory_usage_mb

# -- global debug gate (moved to top so all functions can use it) -------------
DEBUG_ENABLED = os.environ.get("DEBUG_ENABLED", "False").lower() == "true"


def dprint(msg: str, dbg: bool | None = None, level: str = "DEBUG") -> None:
    """Print message only when debug flag is True, with optional logging support."""

    should_print = DEBUG_ENABLED if dbg is None else dbg
    if level == "ERROR":
        should_print = True

    if not should_print:
        return

    try:
        logging.getLogger().log(getattr(logging, level, logging.DEBUG), msg)
    except Exception:
        print(msg, flush=True)


# -- database locations ------------------------------------------
# Use proper path resolution instead of hardcoded relative path
from .helper_path_utils import (
    get_metric_databases_path,
    get_metric_job_sentinels_path,
    get_logs_path,
)
_DB_DIR = get_metric_databases_path()
_NC_CSV = os.path.join(_DB_DIR, "gofnc_database.csv")
_CC_CSV = os.path.join(_DB_DIR, "gofcc_database.csv")
_CZ_CSV = os.path.join(_DB_DIR, "gofcz_database.csv")  # X-Z
_ZZ_CSV = os.path.join(_DB_DIR, "gofzz_database.csv")  # Z-Z′
_NN_CSV = os.path.join(_DB_DIR, "gofnn_database.csv")  # Y-Y′
_OC_CSV = os.path.join(_DB_DIR, "gofoc_database.csv")  # ERA5-CMIP6
_ON_CSV = os.path.join(_DB_DIR, "gofon_database.csv")  # ERA5-GCMagicc
_JOBS_INDEX_CSV = os.path.join(_DB_DIR, "jobs_index.csv")
# SQLite (NEW)
_SQLITE_DB_PATH: str = os.path.join(_DB_DIR, "metrics.sqlite")
_SQLITE_ENABLED: bool = False
# Default to CSV **disabled**; enable explicitly via configure_database(...)
_CSV_ENABLED: bool = False
_SQLITE_CONN: Optional[sqlite3.Connection] = None
# Force-overwrite toggle (can also be set via env: METRICS_FORCE_OVERWRITE=1)
_FORCE_OVERWRITE: bool = str(os.environ.get("METRICS_FORCE_OVERWRITE", "0")).lower() in (
    "1",
    "true",
    "yes",
)
# When true, duplicate checks should report "missing" to force recomputation.
_SKIP_DUPLICATION_CHECK: bool = str(os.environ.get("METRICS_SKIP_DUP_CHECK", "0")).lower() in (
    "1",
    "true",
    "yes",
)
# (z-score database removed; handled in a separate notebook)

# ---------- Fast job sentinel (safe & conservative) ----------
_SENTINEL_DIR = Path(get_metric_job_sentinels_path())
_SENTINEL_DIR.mkdir(parents=True, exist_ok=True)

# Legacy location: migrate any leftover files from logs/job_sentinels once.
_LEGACY_SENTINEL_DIR = Path(get_logs_path()) / "job_sentinels"
if _LEGACY_SENTINEL_DIR.exists() and _LEGACY_SENTINEL_DIR != _SENTINEL_DIR:
    try:
        migrated = 0
        for old_fp in _LEGACY_SENTINEL_DIR.glob("*.json"):
            dest = _SENTINEL_DIR / old_fp.name
            if not dest.exists():
                shutil.move(str(old_fp), str(dest))
                migrated += 1
        # Clean up legacy directory if now empty (best-effort).
        if migrated and not any(_LEGACY_SENTINEL_DIR.iterdir()):
            _LEGACY_SENTINEL_DIR.rmdir()
    except Exception:
        pass

def _code_stamp() -> float:
    """Coarse 'code changed' stamp: max mtime of key modules."""
    try:
        here = Path(__file__).resolve()
        candidates = [
            here,
            here.parent / "helper_benchmark.py",
            here.parent / "helper_bench_plot.py",
        ]
        mt = 0.0
        for p in candidates:
            try:
                mt = max(mt, p.stat().st_mtime)
            except Exception:
                pass
        return mt
    except Exception:
        return 0.0

def _db_stamp(cfg) -> float:
    """DB 'freshness' stamp: sqlite mtime if present, else newest metrics CSV mtime."""
    try:
        sp = cfg.get("sqlite_path") or os.environ.get("METRICS_DB_SQLITE_PATH")
        if sp and os.path.exists(sp):
            return Path(sp).stat().st_mtime
    except Exception:
        pass
    try:
        from .helper_path_utils import get_metric_databases_path
        mdir = Path(get_metric_databases_path())
        if mdir.exists():
            latest = max((p.stat().st_mtime for p in mdir.glob("*.csv")), default=0.0)
            return latest
    except Exception:
        pass
    return 0.0

def _sentinel_key(job, cfg) -> str:
    """Stable filename for a job sentinel."""
    fa = os.path.basename(job.get("file_a", ""))
    fb = os.path.basename(job.get("file_b", ""))
    key = f"{job.get('comparison','?')}|{job.get('recipe','?')}|{fa}|{fb}|{cfg.get('sqlite_path','')}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return f"{job.get('comparison','?')}_{job.get('recipe','?')}_{digest}.json"

# --- Fast "job completion" index (SQLite primary, CSV fallback) -----------------
def _job_sig(job: dict) -> str:
    """
    Deterministic signature per job that does NOT depend on row_keys.
    Distinguishes CC/CZ/ZZ/NN by including file_b, and keeps recipe+comparison.
    """
    model, scen, mem = job["pair"]["key"]
    fa = os.path.basename(job["file_a"])
    fb = os.path.basename(job["file_b"])
    return "|".join([model, scen, mem, job["comparison"], job["recipe"], fa, fb])


def _job_cfg_hash(cfg: dict, recipe: str) -> str:
    """
    Fingerprint only the knobs that affect record semantics.
    (If you add a new recipe knob that changes outputs, include it here.)
    """
    pick = {
        "window_years": cfg.get("window_years"),
        "window_mode": cfg.get("window_mode"),
        "bins": cfg.get("bins"),
        "percentile_for_rmse": cfg.get("percentile_for_rmse"),
        "percentile_for_absdev": cfg.get("percentile_for_absdev"),
        "detimetag_versiontag": cfg.get("detimetag_versiontag"),
        "GlobalTimeseries_vars": cfg.get("GlobalTimeseries_vars"),
        "ZonalMeans_vars": cfg.get("ZonalMeans_vars"),
        "Variability_vars": cfg.get("Variability_vars"),
        "ENSOTeleconnections_vars": cfg.get("ENSOTeleconnections_vars"),
        "Histograms_variables": cfg.get("Histograms_variables"),
        "BiasMaps_restrict2vars": cfg.get("BiasMaps_restrict2vars"),
        "devmaps_triplet_scale": cfg.get("devmaps_triplet_scale"),
        "Histograms_window_years": cfg.get("Histograms_window_years"),
        "ZonalMeans_window_years": cfg.get("ZonalMeans_window_years"),
        "HumidityCoupling_window_years": cfg.get("HumidityCoupling_window_years"),
    }
    s = json.dumps({"recipe": recipe, **pick}, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2s(s.encode("utf-8"), digest_size=12).hexdigest()


def _ensure_jobs_index(sqlite_path: Optional[str]) -> Optional[sqlite3.Connection]:
    if sqlite_path and os.path.exists(sqlite_path):
        con = sqlite3.connect(sqlite_path, timeout=60.0)
        # light read/write performance PRAGMAs
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA temp_store=MEMORY;")
        con.execute(
            "CREATE TABLE IF NOT EXISTS jobs_index ("
            " job_sig TEXT PRIMARY KEY,"
            " model TEXT, scenario TEXT, member TEXT,"
            " comparison TEXT, recipe TEXT,"
            " file_a TEXT, file_b TEXT,"
            " cfg_hash TEXT, version_tag TEXT,"
            " n_records INTEGER NOT NULL,"
            " completed_at TEXT NOT NULL)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS jobs_index_pair "
            " ON jobs_index(model,scenario,member)"
        )
        return con
    return None


def jobs_index_mark(job, cfg, n_records: int, version_tag: str | None = None):
    """Record a fast sentinel that this job produced n_records at the current DB & code state."""
    try:
        fp = _SENTINEL_DIR / _sentinel_key(job, cfg)
        payload = {
            "comparison": job.get("comparison"),
            "recipe": job.get("recipe"),
            "file_a": os.path.basename(job.get("file_a", "")),
            "file_b": os.path.basename(job.get("file_b", "")),
            "n_records": int(n_records),
            "db_stamp": _db_stamp(cfg),
            "code_stamp": _code_stamp(),
            "version_tag": version_tag or cfg.get("version_tag"),
            "created_at": time.time(),
        }
        tmp = fp.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, fp)
    except Exception:
        pass


def jobs_index_has(job, cfg) -> bool:
    """
    Conservative fast check: True only if a fresh sentinel exists *and* code/db haven't
    changed since. Any uncertainty → False (let slow row‑key completeness decide).
    """
    # Honor forced overwrite / redo semantics
    if os.environ.get("METRICS_FORCE_OVERWRITE") == "1":
        return False
    try:
        fp = _SENTINEL_DIR / _sentinel_key(job, cfg)
        if not fp.exists():
            return False
        meta = json.loads(fp.read_text())
        if int(meta.get("n_records", 0)) <= 0:
            return False
        if float(meta.get("code_stamp", 0.0)) < _code_stamp():
            return False
        if float(meta.get("db_stamp", 0.0)) < _db_stamp(cfg):
            return False
        return True
    except Exception:
        return False

# -- unified column schema for all GOF databases --------------------------
# NOTE: add a synthetic, stable key used for all UPSERTs and dedup.
_COLS_GOF = [
    "row_key",
    "metrickey",
    "metricdomain",
    "metrictype",
    "variable",
    "source_id",
    "member_id",
    "experiment_id",
    "comp_source_id",
    "comp_member_id",
    "value",
    "version_tag",
    "timestamp",
]

# (z-score schema removed)

# ensure on-disk DBs exist
# _ensure_db(_NC_CSV,_COLS_GOF)
# _ensure_db(_CC_CSV,_COLS_GOF)
# _ensure_db(_CZ_CSV,_COLS_GOF)
# _ensure_db(_ZZ_CSV,_COLS_GOF)
# _ensure_db(_NN_CSV,_COLS_GOF)

# ----------------------------- CZ & ZZ support --------------------------

# Move random Z lookup to separate directory
from .helper_path_utils import get_data_path
_RAND_Z_DIR = str(get_data_path("random_Z_lookup"))
_RAND_Z_FILE = os.path.join(_RAND_Z_DIR, "_random_Z_lookup.json")

# Cached CMIP file index keyed by absolute root path.
_CMIP_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}


def _normalize_root(path: str) -> str:
    try:
        return str(Path(path).expanduser().resolve(strict=False))
    except Exception:
        return os.path.abspath(path)


def _build_cmip_index(cmip_root: str) -> Dict[str, Any]:
    """
    Build reusable lookup maps from a CMIP root directory.

    Returns dict with:
      - by_model_scen[(model, scenario)] -> List[(ensemble, filepath)]
      - by_triplet[(model, scenario, ensemble)] -> List[filepath]
      - by_exp_source[scenario][model] -> List[filepath]
    """
    by_model_scen: Dict[tuple, List[tuple]] = {}
    by_triplet: Dict[tuple, List[str]] = {}
    by_exp_source: Dict[str, Dict[str, List[str]]] = {}

    patt = os.path.join(cmip_root, "**", "*.nc")
    for f in glob.iglob(patt, recursive=True):
        try:
            model, scen, ens, *_ = parse_filename(os.path.basename(f))
        except Exception:
            continue
        by_model_scen.setdefault((model, scen), []).append((ens, f))
        by_triplet.setdefault((model, scen, ens), []).append(f)
        by_exp_source.setdefault(scen, {}).setdefault(model, []).append(f)

    for key, vals in by_model_scen.items():
        vals.sort(key=lambda t: t[1])
    for key, vals in by_triplet.items():
        vals.sort()
    for exp, model_map in by_exp_source.items():
        for model, vals in model_map.items():
            vals.sort()

    return {
        "by_model_scen": by_model_scen,
        "by_triplet": by_triplet,
        "by_exp_source": by_exp_source,
    }


def _get_cmip_index(cmip_root: str) -> Dict[str, Any]:
    root = _normalize_root(cmip_root)
    cached = _CMIP_INDEX_CACHE.get(root)
    if cached is not None:
        return cached
    idx = _build_cmip_index(root)
    _CMIP_INDEX_CACHE[root] = idx
    return idx


def reset_random_Z_lookup():
    """Reset the random Z lookup file to force regeneration."""
    import os

    print("🔄 Resetting random Z lookup file...")

    # Ensure the directory exists
    os.makedirs(_RAND_Z_DIR, exist_ok=True)

    # Remove the existing file if it exists
    if os.path.exists(_RAND_Z_FILE):
        os.remove(_RAND_Z_FILE)
        print(f"✓ Removed existing lookup file: {_RAND_Z_FILE}")
    else:
        print(f"ℹ️  No existing lookup file found: {_RAND_Z_FILE}")

    print("✅ Random Z lookup reset completed!")
    print("The lookup file will be regenerated on the next validation run.")

    return True


def _resolve_random_z_lookup_path(lookup_path: str | os.PathLike[str] | None = None) -> str:
    """Resolve the on-disk JSON file used for random-Z model selection."""
    if lookup_path is None:
        return _RAND_Z_FILE
    try:
        return str(Path(lookup_path).expanduser().resolve(strict=False))
    except Exception:
        return os.path.abspath(str(lookup_path))


def _load_random_Z_lookup_json(
    lookup_path: str | os.PathLike[str] | None = None,
) -> Dict[str, Dict[str, str]]:
    """Load the persisted random-Z lookup while preserving insertion order."""
    resolved = _resolve_random_z_lookup_path(lookup_path)
    with open(resolved, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"random_Z_lookup JSON must contain an object at the top level: {resolved}")

    lookup: Dict[str, Dict[str, str]] = {}
    for exp, model_map in raw.items():
        if not isinstance(model_map, dict):
            raise ValueError(
                "random_Z_lookup JSON must map experiment_id -> {source_id: path}; "
                f"got {type(model_map).__name__} for {exp!r}"
            )
        lookup[str(exp)] = {str(model): str(path) for model, path in model_map.items()}
    return lookup


def _write_random_Z_lookup_json(
    lookup: Dict[str, Dict[str, str]],
    *,
    lookup_path: str | os.PathLike[str] | None = None,
) -> str:
    """Atomically persist the random-Z lookup using the existing compact JSON style."""
    resolved = _resolve_random_z_lookup_path(lookup_path)
    parent = os.path.dirname(resolved)
    os.makedirs(parent, exist_ok=True)

    fd: int | None = None
    tmp_path = ""
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=".random_z_lookup.", suffix=".tmp", dir=parent)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fd = None
            json.dump(lookup, fh)
        os.replace(tmp_path, resolved)
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return resolved


def augment_random_Z_lookup(
    cmip_root: str,
    *,
    scenarios: Sequence[str],
    lookup_path: str | os.PathLike[str] | None = None,
    write_backup: bool = True,
) -> Dict[str, Any]:
    """
    Append missing scenario/model entries to the persisted random-Z lookup.

    Existing experiment/model assignments are never rewritten. Only missing
    entries are appended, using the same deterministic first-sorted CMIP path
    selection as the initial lookup builder.
    """
    resolved = _resolve_random_z_lookup_path(lookup_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(
            "Cannot augment random_Z_lookup because the lookup file does not exist: "
            f"{resolved}"
        )

    wanted: List[str] = []
    seen: set[str] = set()
    for scenario in scenarios:
        token = str(scenario or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        wanted.append(token)

    lookup = _load_random_Z_lookup_json(resolved)
    idx = _get_cmip_index(cmip_root)
    by_exp_source = idx.get("by_exp_source", {})

    added_by_scenario: Dict[str, List[str]] = {}
    live_model_counts: Dict[str, int] = {}
    total_added = 0

    for scenario in wanted:
        live_models = by_exp_source.get(scenario)
        if not live_models:
            raise ValueError(
                "Requested scenario is absent from the live CMIP6 index and cannot be "
                f"added to random_Z_lookup: {scenario}"
            )

        live_model_counts[scenario] = len(live_models)
        existing_models = lookup.get(scenario)
        if existing_models is None:
            existing_models = {}
            lookup[scenario] = existing_models
        elif not isinstance(existing_models, dict):
            raise ValueError(
                "random_Z_lookup JSON must map experiment_id -> {source_id: path}; "
                f"got {type(existing_models).__name__} for {scenario!r}"
            )

        added_models: List[str] = []
        for model, paths in live_models.items():
            if not paths:
                continue
            current = existing_models.get(model)
            if isinstance(current, str) and current.strip():
                continue
            existing_models[str(model)] = str(paths[0])
            added_models.append(str(model))

        added_by_scenario[scenario] = added_models
        total_added += len(added_models)

    backup_path: str | None = None
    if total_added > 0:
        if write_backup:
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_path = f"{resolved}.{stamp}.bak"
            shutil.copy2(resolved, backup_path)
        _write_random_Z_lookup_json(lookup, lookup_path=resolved)

    return {
        "lookup_path": resolved,
        "backup_path": backup_path,
        "requested_scenarios": wanted,
        "added_by_scenario": added_by_scenario,
        "added_models": total_added,
        "live_model_counts": live_model_counts,
    }


def _ensure_db(path: str, cols: List[str]) -> pd.DataFrame:
    """Ensure database file exists with correct headers."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if not os.path.exists(path):
            # Create new file with headers
            print(f"Creating new database file: {path}")
            df = pd.DataFrame(columns=cols)
            df.to_csv(path, index=False, quoting=1)
            print(f"✓ Created {path} with {len(cols)} columns")
        else:
            # File exists, try to read it
            try:
                df = pd.read_csv(path)
                print(f"✓ Loaded existing database: {path} ({len(df)} rows)")
            except Exception:
                # If CSV is corrupted, recreate it
                print(f"Warning: Corrupted CSV file {path}, recreating...")
                df = pd.DataFrame(columns=cols)
                df.to_csv(path, index=False, quoting=1)
                print(f"✓ Recreated {path} with {len(cols)} columns")

        # Ensure all required columns exist
        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            print(f"Adding missing columns to {path}: {missing_cols}")
            for col in missing_cols:
                df[col] = np.nan
            df.to_csv(path, index=False, quoting=1)
            print(f"✓ Updated {path} with missing columns")

        return df[cols]

    except Exception as e:
        print(f"Error ensuring database {path}: {e}")
        # Create a minimal DataFrame with correct columns as fallback
        print(f"Creating fallback DataFrame for {path}")
        return pd.DataFrame(columns=cols)


# ----------------------------- SQLite helpers (NEW) ---------------------------
_TABLES = {
    "nc": "gofnc",
    "cc": "gofcc",
    "cz": "gofcz",
    "zz": "gofzz",
    "nn": "gofnn",
    "oc": "gofoc",
    "on": "gofon",
}


def _compute_row_key(rec: Dict, comparison: str) -> str:
    """
    Build a stable identity hash for a row. Always includes version_tag so
    metrics from different model tags generate distinct identities.
    """
    import hashlib
    import json

    parts = {
        "metrickey": rec.get("metrickey", ""),
        "metricdomain": rec.get("metricdomain", ""),
        "metrictype": rec.get("metrictype", ""),
        "variable": rec.get("variable", ""),
        "source_id": rec.get("source_id", ""),
        "member_id": rec.get("member_id", ""),
        "experiment_id": rec.get("experiment_id", ""),
        "comp_source_id": rec.get("comp_source_id", ""),
        "comp_member_id": rec.get("comp_member_id", ""),
    }
    parts["version_tag"] = rec.get("version_tag", "") or ""
    j = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(j.encode("utf-8")).hexdigest()


def configure_database(
    backends: List[str] | tuple[str, ...], sqlite_path: Optional[str] = None
) -> None:
    """Select DB backends at runtime. Call early (e.g., from validation_suite)."""
    global _SQLITE_ENABLED, _CSV_ENABLED, _SQLITE_DB_PATH, _SQLITE_CONN
    b = [s.lower() for s in (backends or [])]
    _CSV_ENABLED = "csv" in b
    _SQLITE_ENABLED = "sqlite" in b
    if sqlite_path:
        _SQLITE_DB_PATH = sqlite_path
    if _SQLITE_ENABLED:
        _sqlite_init()
    # if switched off, close any old connection
    if not _SQLITE_ENABLED and _SQLITE_CONN is not None:
        try:
            _SQLITE_CONN.close()
        finally:
            _SQLITE_CONN = None


def set_redo_calcs_if_new_versiontag(flag: bool) -> None:
    """
    Deprecated shim: row_key always includes version_tag now, so this flag
    is ignored. Kept for backward compatibility.
    """
    logging.getLogger(__name__).info(
        "redo_if_new_versiontag is deprecated; row_key always includes version_tag (flag=%s)",
        flag,
    )


def set_force_overwrite(enable: bool) -> None:
    """Force-delete existing rows (by row_key) before writes."""
    global _FORCE_OVERWRITE
    _FORCE_OVERWRITE = bool(enable)
    if enable:
        os.environ["METRICS_FORCE_OVERWRITE"] = "1"
    else:
        os.environ.pop("METRICS_FORCE_OVERWRITE", None)


def set_skip_duplication_check(flag: bool) -> None:
    """Force recomputation by disabling duplicate existence checks."""
    global _SKIP_DUPLICATION_CHECK
    _SKIP_DUPLICATION_CHECK = bool(flag)
    if flag:
        os.environ["METRICS_SKIP_DUP_CHECK"] = "1"
    else:
        os.environ.pop("METRICS_SKIP_DUP_CHECK", None)


def _sqlite_conn() -> sqlite3.Connection:
    global _SQLITE_CONN
    if _SQLITE_CONN is None:
        os.makedirs(os.path.dirname(_SQLITE_DB_PATH), exist_ok=True)
        _SQLITE_CONN = sqlite3.connect(_SQLITE_DB_PATH, timeout=120.0, check_same_thread=False)
        _SQLITE_CONN.execute("PRAGMA journal_mode=WAL;")
        _SQLITE_CONN.execute("PRAGMA synchronous=NORMAL;")
        _SQLITE_CONN.execute("PRAGMA temp_store=MEMORY;")
    return _SQLITE_CONN


def _sqlite_init() -> None:
    con = _sqlite_conn()
    for _comp, t in _TABLES.items():
        # Create table (idempotent)
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {t} (
                row_key        TEXT NOT NULL DEFAULT '',
                metrickey      TEXT,
                metricdomain   TEXT,
                metrictype     TEXT,
                variable       TEXT,
                source_id      TEXT,
                member_id      TEXT,
                experiment_id  TEXT,
                comp_source_id TEXT,
                comp_member_id TEXT,
                value          REAL,
                version_tag    TEXT DEFAULT '',
                timestamp      TEXT
            );
        """)
        # Lightweight migration: ensure row_key column exists (older DBs)
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({t});").fetchall()]
        if "row_key" not in cols:
            con.execute(f"ALTER TABLE {t} ADD COLUMN row_key TEXT NOT NULL DEFAULT '';")
        # Create a single, stable unique index on row_key (no DROP/CREATE races)
        con.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {t}_uniq_rowkey ON {t}(row_key);")
    con.commit()


def _sqlite_upsert_many(comparison: str, records: List[Dict[str, Any]]) -> None:
    """
    Write many metric rows into the SQLite database with robust dedup semantics.
    * If METRICS_FORCE_OVERWRITE=1 -> INSERT OR REPLACE (overwrite duplicates)
    * Else                         -> INSERT OR IGNORE   (keep existing)
    This works regardless of whether uniqueness is enforced on row_key or on a
    legacy composite of columns.
    """
    if not records:
        return

    table = f"gof{comparison.lower()}"
    con = _sqlite_conn()
    cur = con.cursor()

    # Stable column order for writes
    cols = list(_COLS_GOF)
    cols_csv = ", ".join(cols)
    placeholders = ", ".join(["?"] * len(cols))

    # Normalize rows to the full schema (fill missing keys)
    norm = []
    for r in records:
        row = {c: r.get(c, None) for c in cols}
        # canonicalize empties
        if row.get("version_tag") is None:
            row["version_tag"] = ""
        norm.append(tuple(row[c] for c in cols))

    force_overwrite = os.getenv("METRICS_FORCE_OVERWRITE", "0") in (
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    )

    if force_overwrite:
        sql = f"INSERT OR REPLACE INTO {table} ({cols_csv}) VALUES ({placeholders})"
    else:
        sql = f"INSERT OR IGNORE INTO {table} ({cols_csv}) VALUES ({placeholders})"

    try:
        cur.executemany(sql, norm)
        con.commit()
    except sqlite3.Error:
        # As a last resort, fall back to row-wise ops to surface the bad row
        con.rollback()
        for row in norm:
            try:
                cur.execute(sql, row)
            except sqlite3.Error as ee:
                # Print a short hint, then re-raise
                dprint(f"❌ SQLite write failed for table={table}: {ee}")
                raise
        con.commit()


def _sqlite_read_table(comparison: str) -> pd.DataFrame:
    con = _sqlite_conn()
    t = _TABLES[comparison]
    try:
        return pd.read_sql_query(f"SELECT {', '.join(_COLS_GOF)} FROM {t};", con)
    except Exception:
        # empty table
        return pd.DataFrame(columns=_COLS_GOF)


def _safe_write_csv(
    df: pd.DataFrame, path: str, keep_backups: int = 5, unique_subset: Optional[List[str]] = None
) -> None:
    import os
    import fcntl
    import glob
    import pandas as pd

    dprint(f"🔍 DEBUG: _safe_write_csv called for {path}")
    dprint(f"🔍 DEBUG: DataFrame shape: {df.shape}")

    dir_ = os.path.dirname(path)
    os.makedirs(dir_, exist_ok=True)
    lock_path = f"{path}.lock"

    dprint(f"🔍 DEBUG: Using lock file: {lock_path}")

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        subset = unique_subset or ["row_key"]
        # For large CSVs, avoid loading the entire file: read keys only
        big = os.path.exists(path) and os.path.getsize(path) > 100 * 1024 * 1024
        if big:
            try:
                existing_keys = set(
                    pd.read_csv(path, usecols=subset, dtype=str)[subset[0]].astype(str)
                )
                dprint(f"🔍 DEBUG: Existing keys loaded: {len(existing_keys)}")
            except Exception:
                existing_keys = set()
            new_df = df.drop_duplicates(subset=subset, keep="last")
            keep_mask = ~new_df[subset[0]].astype(str).isin(existing_keys)
            df_to_write = new_df.loc[keep_mask]
        else:
            existing = pd.DataFrame()
            if os.path.exists(path) and os.path.getsize(path) > 0:
                existing = pd.read_csv(path)
                dprint(f"🔍 DEBUG: Read existing file with {len(existing)} rows")
            else:
                dprint(f"🔍 DEBUG: No existing file or empty file at {path}")
            df_to_write = pd.concat([existing, df], ignore_index=True).drop_duplicates(
                subset=subset, keep="last"
            )

        dprint(f"🔍 DEBUG: After merge and deduplication: {len(df_to_write)} rows")

        # ---------- snapshot BEFORE overwrite ----------
        if os.path.exists(path):
            ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{path}.{ts}.bak"
            shutil.copy2(path, backup_path)
            dprint(f"🔍 DEBUG: Created backup: {backup_path}")

        # ---------- atomic write ----------
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        dprint(f"🔍 DEBUG: Created temp file: {tmp}")

        with os.fdopen(fd, "w", newline="") as f:
            df_to_write.to_csv(f, index=False, quoting=1)
            f.flush()
            os.fsync(f.fileno())

        dprint("🔍 DEBUG: Wrote data to temp file")

        os.replace(tmp, path)
        os.fsync(os.open(dir_, os.O_DIRECTORY))

        dprint(f"🔍 DEBUG: Replaced {path} with temp file")

        # ---------- rotate old backups ----------
        baks = sorted(glob.glob(f"{path}.*.bak"))
        for b in baks[:-keep_backups]:
            os.remove(b)

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    # Verify the write
    if os.path.exists(path):
        final_size = os.path.getsize(path)
        dprint(f"🔍 DEBUG: Final file size: {final_size} bytes")
        try:
            final_df = pd.read_csv(path)
            dprint(f"🔍 DEBUG: Final file contains {len(final_df)} rows")
        except Exception as e:
            dprint(f"🔍 DEBUG: Error reading final file: {e}")
    else:
        dprint("🔍 DEBUG: ERROR: File does not exist after write!")

    dprint(f"🔍 DEBUG: _safe_write_csv completed for {path}")


def _seed_once_for_repro() -> None:
    """Fix RNG seed once per process so *Z* selection stays deterministic."""
    if not getattr(_seed_once_for_repro, "_done", False):
        random.seed(42)
        _seed_once_for_repro._done = True


def _build_random_Z_lookup(cmip_root: str) -> Dict[str, Dict[str, str]]:
    """
    Return mapping  lookup[experiment_id][source_id] -> *one* member NetCDF path.
    The file is created once and reused afterwards for full reproducibility.
    """
    if os.path.exists(_RAND_Z_FILE):
        return _load_random_Z_lookup_json(_RAND_Z_FILE)

    _seed_once_for_repro()
    lookup: Dict[str, Dict[str, str]] = {}
    idx = _get_cmip_index(cmip_root)
    by_exp_source = idx.get("by_exp_source", {})
    for exp, model_map in by_exp_source.items():
        out = lookup.setdefault(exp, {})
        for model, paths in model_map.items():
            if paths:
                # deterministic first path after sort in _build_cmip_index
                out.setdefault(model, paths[0])

    dprint(f"🔍 Built random Z lookup from index: {len(lookup)} experiments")
    for exp, models in lookup.items():
        dprint(f"   - {exp}: {len(models)} models")

    # Ensure the directory exists before writing
    os.makedirs(_RAND_Z_DIR, exist_ok=True)

    with open(_RAND_Z_FILE, "w") as fh:
        json.dump(lookup, fh)

    dprint(f"✅ Saved lookup to: {_RAND_Z_FILE}")
    return lookup


def discover_other_model_files(
    main_file: str, cmip_root: str, *, max_models: int = 5
) -> List[str]:
    """
    Pick ≤ *max_models* NetCDFs with **different source_id** but same *experiment*
    as *main_file* using the static random-Z lookup table.
    """
    src, exp, mem, *_ = parse_filename(os.path.basename(main_file))
    lookup = _build_random_Z_lookup(cmip_root).get(exp, {})
    others = [p for s, p in lookup.items() if s != src]
    return others[:max_models]


# ----------------------------- NN support -------------------------------
def discover_other_gcmagicc_files(main_file: str, *, max_members: int = 5) -> List[str]:
    """
    Find other GCMagicc ensemble members in the *same* output-folder that
    match source_id & experiment_id of *main_file*.
    """
    f = Path(main_file)
    # Baseline identity uses *plain* member for matching (so NN keeps same r*i*p*f)
    src, exp, mem_plain = parse_filename(f.name, use_pseudo_member=False)
    # Also get baseline pseudo (includes GCMagicc prefix) to exclude the same file
    _, _, mem_pseudo_baseline = parse_filename(f.name, use_pseudo_member=True)
    candidates = list(f.parent.glob(f"*_{exp}_*.nc"))
    others = []

    # Optional targeted debug for NN discovery
    import os

    dbg = os.environ.get("DEBUG_NN", "0") == "1"
    if dbg:
        print(f"[NN] baseline: {f.name}")
        print(f"[NN] src={src} exp={exp} member(plain)={mem_plain}")
        print(f"[NN] candidates in {f.parent}: {len(candidates)}")
        for c in candidates:
            print(f"[NN]   cand: {c.name}")

    for c in candidates:
        if c.name == f.name:
            if dbg:
                print(f"[NN] skip self: {c.name}")
            continue
        # Must match src & exp
        c_src, c_exp, c_mem_plain = parse_filename(c.name, use_pseudo_member=False)
        if c_src != src or c_exp != exp:
            if dbg:
                print(f"[NN] skip model/exp mismatch: {c.name} ({c_src},{c_exp})")
            continue
        # Must keep the SAME plain CMIP6 member (so NN pairs within same ensemble)
        if c_mem_plain != mem_plain:
            if dbg:
                print(f"[NN] skip different plain member: {c.name} ({c_mem_plain} != {mem_plain})")
            continue
        # And must differ by GCMagicc prefix (pseudo member mismatch)
        _, _, c_mem_pseudo = parse_filename(c.name, use_pseudo_member=True)
        if c_mem_pseudo == mem_pseudo_baseline:
            if dbg:
                print(f"[NN] skip same prefix: {c.name}")
            continue
        if dbg:
            print(f"[NN] ✓ accept: {c.name}")
        others.append(str(c))

    # Enforce deterministic ordering before truncation so reduced-member policies
    # produce stable pairings across runs and hosts.
    others = sorted(set(others), key=lambda p: os.path.basename(p))
    return others[:max_members]


# ----------------------------- NN support -------------------------------

# +------------------------------------------------------------------+
# |  IN-MEMORY DATABASE CACHE SYSTEM                                |
# +------------------------------------------------------------------+

# Global cache for *union* of enabled backends to avoid repeated I/O
_DB_CACHE = {
    "nc": None,
    "cc": None,
    "cz": None,
    "zz": None,
    "nn": None,
    "oc": None,
    "on": None,
    "last_modified": {"nc": 0, "cc": 0, "cz": 0, "zz": 0, "nn": 0, "oc": 0, "on": 0},
}


def _get_cached_db(comparison: str) -> pd.DataFrame:
    """
    Get *union view* from enabled backends (CSV, SQLite) with light caching.
    """
    global _DB_CACHE
    import time

    # identify CSV path
    if comparison == "nc":
        csv_path = _NC_CSV
    elif comparison == "cc":
        csv_path = _CC_CSV
    elif comparison == "cz":
        csv_path = _CZ_CSV
    elif comparison == "zz":
        csv_path = _ZZ_CSV
    elif comparison == "nn":
        csv_path = _NN_CSV
    elif comparison == "oc":
        csv_path = _OC_CSV
    elif comparison == "on":
        csv_path = _ON_CSV
    else:
        raise ValueError(
            f"comparison must be 'nc', 'cc', 'cz', 'zz', 'nn', 'oc', or 'on', got {comparison}"
        )

    csv_mtime = None
    if _CSV_ENABLED:
        try:
            csv_mtime = os.path.getmtime(csv_path)
        except OSError:
            csv_mtime = 0
    sql_mtime = None
    if _SQLITE_ENABLED:
        try:
            sql_mtime = os.path.getmtime(_SQLITE_DB_PATH)
        except OSError:
            sql_mtime = 0

    cache_age = time.time() - _DB_CACHE["last_modified"][comparison]
    need_refresh = (
        _DB_CACHE[comparison] is None
        or cache_age > 5
        or (_CSV_ENABLED and csv_mtime and csv_mtime > _DB_CACHE["last_modified"][comparison])
        or (_SQLITE_ENABLED and sql_mtime and sql_mtime > _DB_CACHE["last_modified"][comparison])
    )

    if need_refresh:
        frames = []
        if _CSV_ENABLED:
            frames.append(_ensure_db(csv_path, _COLS_GOF))
        if _SQLITE_ENABLED:
            try:
                frames.append(_sqlite_read_table(comparison))
            except Exception as e:
                print(f"⚠️  SQLite read failed for {comparison}: {e}")
        if frames:
            df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            # drop duplicates ignoring timestamp (keep last)
            df = df.drop_duplicates(subset=[c for c in _COLS_GOF if c != "timestamp"], keep="last")
        else:
            df = pd.DataFrame(columns=_COLS_GOF)
        _DB_CACHE[comparison] = df
        _DB_CACHE["last_modified"][comparison] = time.time()

    return _DB_CACHE[comparison]


def _clear_db_cache():
    """Clear the database cache to force reload from disk."""
    global _DB_CACHE
    _DB_CACHE = {
        "nc": None,
        "cc": None,
        "cz": None,
        "zz": None,
        "nn": None,
        "oc": None,
        "on": None,
        "last_modified": {"nc": 0, "cc": 0, "cz": 0, "zz": 0, "nn": 0, "oc": 0, "on": 0},
    }


def _clear_db_cache_for_comparison(comparison: str):
    """Clear the database cache for a specific comparison type."""
    global _DB_CACHE
    _DB_CACHE[comparison] = None
    _DB_CACHE["last_modified"][comparison] = 0


def _update_db_cache(comparison: str, new_records: List[Dict]):
    """
    Update the in-memory cache with new records.
    This keeps the cache synchronized with disk updates.
    """
    global _DB_CACHE

    # Initialize cache if it's None
    if _DB_CACHE[comparison] is None:
        _DB_CACHE[comparison] = _get_cached_db(comparison)

    new_df = pd.DataFrame(new_records)
    _DB_CACHE[comparison] = pd.concat([_DB_CACHE[comparison], new_df], ignore_index=True)
    # Keep only one row per key in-process (latest wins), to avoid stale views
    if "row_key" in _DB_CACHE[comparison].columns:
        _DB_CACHE[comparison] = _DB_CACHE[comparison].drop_duplicates(
            subset=["row_key"], keep="last"
        )
    _DB_CACHE["last_modified"][comparison] = time.time()


def _clear_db_cache():
    """Clear the database cache to free memory."""
    global _DB_CACHE
    _DB_CACHE.clear()

    # Force garbage collection
    import gc

    gc.collect()

    print("[CACHE] Database cache cleared, memory freed")


# +------------------------------------------------------------------+
# |  OPTIMIZED DATABASE CHECKING                                    |
# +------------------------------------------------------------------+


def check_for_existing_records(record_template: Dict, comparison: str) -> bool:
    """
    Check if a record already exists in the database.
    Uses cached database for better performance.

    Args:
        record_template: Dictionary containing the record structure to check against.
                        Should include: metrickey, metricdomain, metrictype, variable,
                        source_id, member_id, experiment_id, comp_source_id, comp_member_id
        comparison: Either "nc" or "cc" to determine which database to check

    Returns:
        bool: True if the record already exists, False otherwise
    """
    # Hard override: force recomputation
    if _SKIP_DUPLICATION_CHECK:
        dprint("🔄 skip_duplication_check active - treating record as MISSING to force recompute")
        return False
    try:
        db = _get_cached_db(comparison)
        rk = _compute_row_key(record_template, comparison)
        if "row_key" not in db.columns:
            return False
        return (db.row_key == rk).any()

    except Exception:
        print("database check couldn't be done")
        # Continue with computation if we can't check the database
        return False


def check_for_existing_records_batch(record_templates: List[Dict], comparison: str) -> bool:
    """
    Check if all of the records in a batch already exist in the database.
    Uses cached database for better performance.

    Args:
        record_templates: List of dictionaries containing record structures to check against.
                         Each should include: metrickey, metricdomain, metrictype, variable,
                         source_id, member_id, experiment_id, comp_source_id, comp_member_id
                         For GOFNC and GOFNN comparisons, version_tag is also included in the check.
        comparison: Either "nc", "cc", "cz", "zz", or "nn" to determine which database to check

    Returns:
        bool: True if all of the records already exist, False otherwise
    """
    # Hard override: force recomputation
    if _SKIP_DUPLICATION_CHECK:
        dprint(
            "🔄 skip_duplication_check active - treating ALL records as MISSING to force recompute"
        )
        return False
    try:
        db = _get_cached_db(comparison)
        if "row_key" not in db.columns:
            return False
        all_exist = True
        for record_template in record_templates:
            rk = _compute_row_key(record_template, comparison)
            if not (db.row_key == rk).any():
                all_exist = False
                break

        if all_exist:
            dprint(f"🔍 DEBUG: All {len(record_templates)} records already exist in database")
        else:
            dprint("🔍 DEBUG: Some records missing from database, will compute")

        return all_exist

    except Exception as e:
        dprint(f"Database check for duplicates of {record_templates} failed: {e}")
        # Continue with computation if we can't check the database
        return False


def _utc_ts() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds")


# +------------------------------------------------------------------+
# |  WRITE-HELPERS                                                   |
# +------------------------------------------------------------------+


# ------------------------- unified database writer -------------------------------
def write_records_unified(records: List[Dict], comparison: str, version_tag: str = None) -> None:
    """Thread-safe append of *validated* rows to *db_path* with unified schema."""
    if not records:
        dprint(
            f"🔍 DEBUG: write_records_unified called with empty records for {comparison} comparison"
        )
        return

    dprint(
        f"🔍 DEBUG: write_records_unified called with {len(records)} records for {comparison} comparison"
    )

    # -- determine CSV path (if used) -----------------------------------------
    if comparison == "nc":
        db_path = _NC_CSV
    elif comparison == "cc":
        db_path = _CC_CSV
    elif comparison == "cz":
        db_path = _CZ_CSV
    elif comparison == "zz":
        db_path = _ZZ_CSV
    elif comparison == "nn":
        db_path = _NN_CSV
    elif comparison == "oc":
        db_path = _OC_CSV
    elif comparison == "on":
        db_path = _ON_CSV
    else:
        raise ValueError(
            f"comparison must be 'nc','cc','cz','zz','nn','oc','on', got {comparison}"
        )

    dprint(f"🔍 DEBUG: Writing to database path: {db_path}")

    # -- prepare records, add timestamp ------------------------------------
    valid_records = []
    for r in records:
        r.setdefault("metrickey", f"GOF{comparison.upper()}")
        r["timestamp"] = _utc_ts()

        # Add version_tag when GCMagicc is part of comparison (NC, NN, ON)
        if comparison in ["nc", "nn", "on"] and version_tag:
            r["version_tag"] = version_tag
        # Set version_tag to "versionless" for versionless comparisons (CC, CZ, ZZ, OC)
        elif comparison in ["cc", "cz", "zz", "oc"]:
            r["version_tag"] = "versionless"
        # compute stable row_key (after version_tag is set)
        r["row_key"] = _compute_row_key(r, comparison)
        valid_records.append(r)

    dprint(f"🔍 DEBUG: Prepared {len(valid_records)} valid records")
    if len(valid_records) > 0:
        dprint(f"🔍 DEBUG: Sample record: {valid_records[0]}")

    # -- update in-process union cache right away -----------------------------
    _update_db_cache(comparison, valid_records)

    # -- write to configured backends ----------------------------------------
    df = pd.DataFrame(valid_records)
    # normalize version_tag so NaN/None and '' collapse equally
    if "version_tag" in df.columns:
        df["version_tag"] = df["version_tag"].fillna("")

    # CSV de-dup identity: row_key
    uniq_subset = ["row_key"]

    if _CSV_ENABLED:
        _safe_write_csv(df, db_path, unique_subset=uniq_subset)
        dprint(f"🔍 DEBUG: wrote {len(df)} rows to CSV ({db_path})")
    if _SQLITE_ENABLED:
        _sqlite_upsert_many(comparison, valid_records)
        dprint(
            f"🔍 DEBUG: upserted {len(df)} rows to SQLite ({_SQLITE_DB_PATH}:{_TABLES[comparison]})"
        )

    # -- cache is already updated above, no need to clear ---------------------
    # _clear_db_cache_for_comparison(comparison)  # REMOVED: This breaks deduplication

    dprint(f"🔍 DEBUG: write_records_unified completed for {comparison} comparison")


# +------------------------------------------------------------------+
# |  READ-HELPER                                                    |
# +------------------------------------------------------------------+
def get_intracmip6_distribution(template: Dict) -> List[float]:
    """
    Return all CMIP6-vs-CMIP6 GOF values matching *template*
    (same domain, metrictype, variable, model, member, experiment_id).
    """
    # Use the cached union view (CSV and/or SQLite, depending on configured backends)
    db = _get_cached_db("cc")
    mask = (
        (db.metricdomain == template["metricdomain"])
        & (db.metrictype == template["metrictype"])
        & (db.variable == template["variable"])
        & (db.source_id == template["source_id"])
        & (db.member_id == template["member_id"])
        & (db.experiment_id == template["experiment_id"])
        & (db.comp_source_id == template["comp_source_id"])
        & (db.comp_member_id == template["comp_member_id"])
    )
    return db.loc[mask, "value"].astype(float).tolist()


# +------------------------------------------------------------------+
# |  QUICK UTILITIES                                                |
# +------------------------------------------------------------------+
def calculate_cc_rmse(main_file: str, other_file: str, variable: str) -> float:
    ds1 = xr.open_dataset(main_file)[variable]
    ds2 = xr.open_dataset(other_file)[variable]
    ds1, ds2 = xr.align(ds1, ds2, join="inner")
    return compute_rmse_score(ds1, ds2)


def calculate_cc_rmse_consistent(main_file: str, other_file: str, variable: str) -> float:
    """
    Compute RMSE between two CMIP6 files, consistent with GOF database calculation.

    This function matches the calculation method used in the GOF database:
    - First compute temporal mean over the full time period
    - Then compute area-weighted spatial RMSE
    """
    ds1 = xr.open_dataset(main_file)[variable]
    ds2 = xr.open_dataset(other_file)[variable]
    ds1, ds2 = xr.align(ds1, ds2, join="inner")

    # First compute temporal mean (like in GOF database)
    ds1_temporal_mean = ds1.mean(dim="time")
    ds2_temporal_mean = ds2.mean(dim="time")

    # Compute deviation (signed, like in GOF database)
    diff = ds1_temporal_mean - ds2_temporal_mean

    # Area-weighted spatial RMSE (like _weighted_global function)
    weights = np.cos(np.deg2rad(diff.lat))
    squared_diff = diff**2
    spatial_mean_squared = squared_diff.weighted(weights).mean(["lat", "lon"])
    rmse_scalar = np.sqrt(spatial_mean_squared).compute().item()

    return rmse_scalar


def discover_other_member_files(main_file: str, cmip6_root: str) -> List[str]:
    """
    Find *other* CMIP6 members for the same (model, scenario) as *main_file*.
    Searches **recursively** and falls back to a full scan if the primary
    pattern returns nothing (handles mixed directory layouts).
    """
    model, scen, ens = parse_filename(os.path.basename(main_file))
    if model is None:
        return []
    idx = _get_cmip_index(cmip6_root)
    candidates = idx.get("by_model_scen", {}).get((model, scen), [])
    main_base = os.path.basename(main_file)
    result = [p for other_ens, p in candidates if other_ens != ens and os.path.basename(p) != main_base]
    return result


def resolve_cmip6_member_file(
    model: str,
    scenario: str,
    ensemble: str,
    cmip6_root: str,
    *,
    prefer: str | None = None,
) -> str:
    """
    Return an existing CMIP6 file path for a (model, scenario, ensemble) triple.

    This is a small guard against stale hard-coded filenames (e.g., missing a
    variable suffix like `evspsbl`). It searches recursively for
    `DAT_{model}_{scenario}_{ensemble}_*.nc`, optionally preferring a provided
    path if it exists.
    """
    if prefer and os.path.exists(prefer):
        return prefer

    idx = _get_cmip_index(cmip6_root)
    matches = idx.get("by_triplet", {}).get((model, scenario, ensemble), [])
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"No CMIP6 file found for {model}/{scenario}/{ensemble} under {cmip6_root}"
    )


# +---------------------------------------------------------------------------
# | Climate index functions (moved from recipes/ClimateIndices.py)         |
# +---------------------------------------------------------------------------


def _apply_smoothing_and_detrending(da, window=3, cutoff_years=30, order=4):
    """Apply smoothing and detrending to a time series."""
    import time
    start_time = time.time()

    try:
        # Rolling mean smoothing
        da_smoothed = da.rolling(time=window, center=True, min_periods=1).mean()
        # smooth_time = time.time() - smooth_start  # unused variable removed

        # Butterworth filter setup
        fs = 12.0
        fc = 1.0 / cutoff_years
        Wn = fc / (fs / 2)
        b, a = butter(order, Wn, btype="low")
        # filter_setup_time = time.time() - filter_start  # unused variable removed

        # Apply filter
        # --- find the time axis programmatically ----------------------------
        try:  # xarray ≥ 0.18
            t_ax = da_smoothed.get_axis_num("time")
        except AttributeError:  # fallback for very old xarray
            t_ax = list(da_smoothed.dims).index("time")

        data = da_smoothed.values
        trend = filtfilt(b, a, data, axis=t_ax)  # robust to dim order ✅

        da_trend = xr.DataArray(trend, coords=da_smoothed.coords, dims=da_smoothed.dims)
        da_detrended = da_smoothed - da_trend
        # apply_time = time.time() - apply_start  # unused variable removed

        total_time = time.time() - start_time

        # Clean up intermediate variables to free memory
        del da_smoothed, data, trend, da_trend

        # Force garbage collection
        import gc

        gc.collect()

        # Only print if debug is enabled (we'll add a debug parameter later if needed)
        # print(f"[SMOOTH] Smoothing took {smooth_time:.3f}s, filter setup {filter_setup_time:.3f}s, apply {apply_time:.3f}s")
        # print(f"[SMOOTH] Total: {total_time:.3f}s, memory: {memory_change:+.1f} MB")

        return da_detrended
    except Exception as e:
        total_time = time.time() - start_time
        dprint(f"[SMOOTH] Error after {total_time:.3f}s: {e}")
        raise


def detect_longitude_format(ds):
    """Detect the longitude format of a dataset."""
    lon_min = ds.lon.min().values
    lon_max = ds.lon.max().values
    if lon_min >= 0 and lon_max <= 360:
        return "0-360"
    elif lon_min >= -180 and lon_max <= 180:
        return "-180-180"
    else:
        return "0-360"


def get_robust_lon_bounds(ds, target_lon_min, target_lon_max, target_format="0-360"):
    """Get robust longitude bounds for region selection."""
    dataset_format = detect_longitude_format(ds)

    if dataset_format == target_format:
        return target_lon_min, target_lon_max

    if target_format == "0-360":

        def to_0_360(lon):
            return lon if lon >= 0 else lon + 360

        return to_0_360(target_lon_min), to_0_360(target_lon_max)
    else:

        def to_180_180(lon):
            return lon if lon <= 180 else lon - 360

        return to_180_180(target_lon_min), to_180_180(target_lon_max)


def select_robust_region(ds, lat_min, lat_max, lon_min, lon_max, target_format="0-360"):
    """Select a region from a dataset with robust longitude handling."""
    lon_min_adj, lon_max_adj = get_robust_lon_bounds(ds, lon_min, lon_max, target_format)

    # Handle latitude selection with proper ordering
    lat_coord = ds.lat
    if lat_coord.values[0] > lat_coord.values[-1]:  # Descending order
        # For descending order, we need to swap the slice order
        lat_slice = slice(lat_max, lat_min)
    else:  # Ascending order
        lat_slice = slice(lat_min, lat_max)

    return ds.sel(lat=lat_slice, lon=slice(lon_min_adj, lon_max_adj))


def compute_nino34_index(ds, var="ts", debug=False):
    """
    Computes the Niño-3.4 index using the ts field (as a proxy for SST).
    Steps:
      1. Compute monthly anomalies (by subtracting the monthly climatology).
      2. Select the region defined by lat [-5, 5] and lon [190, 240] (0-360 convention).
      3. Compute an area-weighted (cos(lat)) average over that region.
      4. Smooth with a 3-month running mean and remove long-term (>30 yr) variability.
    """
    import time
    print(f"[NINO34] Starting Niño-3.4 computation for {var}")
    start_time = time.time()
    initial_memory = get_memory_usage_mb()

    if debug:
        print(f"[NINO34] Starting Niño-3.4 computation for {var}")
        if initial_memory is not None:
            print(f"[NINO34] Initial memory: {initial_memory:.1f} MB")

    # Get field and compute anomalies
    field_start = time.time()
    tas = ds[var]
    field_time = time.time() - field_start
    if debug:
        print(f"[NINO34] Field extraction took {field_time:.3f}s")

    anom_start = time.time()
    tas_anom = tas.groupby("time.month") - tas.groupby("time.month").mean("time")
    anom_time = time.time() - anom_start
    if debug:
        print(f"[NINO34] Anomaly computation took {anom_time:.3f}s")

    # Select region
    region_start = time.time()
    lon_min, lon_max = 190, 240
    lat_min, lat_max = -5, 5
    try:
        region = select_robust_region(
            tas_anom, lat_min, lat_max, lon_min, lon_max, target_format="0-360"
        )
        region_time = time.time() - region_start
        if debug:
            print(f"[NINO34] Region selection took {region_time:.3f}s")
    except Exception as e:
        if debug:
            dprint(f"[DEBUG] compute_nino34_index: Error selecting region: {e}")
        return None

    if region.size == 0:
        if debug:
            dprint("[DEBUG] compute_nino34_index: ERROR: Selected region is empty!")
        return None

    # --- CHANGED SECTION: proper area weighting --------------------------
    weight_start = time.time()
    weights = np.cos(np.deg2rad(region["lat"]))  # ① cos φ
    raw_index = region.weighted(weights).mean(dim=["lat", "lon"])  # ② xarray-weighted mean
    weight_time = time.time() - weight_start
    if debug:
        print(f"[NINO34] Weighted average computation took {weight_time:.3f}s")
    # ---------------------------------------------------------------------

    # Apply smoothing and detrending
    smooth_start = time.time()
    index_processed = _apply_smoothing_and_detrending(raw_index)
    smooth_time = time.time() - smooth_start
    if debug:
        print(f"[NINO34] Smoothing and detrending took {smooth_time:.3f}s")

    total_time = time.time() - start_time
    final_memory = get_memory_usage_mb()
    memory_change = None
    if final_memory is not None and initial_memory is not None:
        memory_change = final_memory - initial_memory

    if debug:
        print(f"[NINO34] Completed Niño-3.4 computation for {var}")
        if memory_change is not None:
            print(
                f"[NINO34] Total time: {total_time:.3f}s, memory change: {memory_change:+.1f} MB"
            )
        else:
            print(f"[NINO34] Total time: {total_time:.3f}s")

    # Clean up intermediate variables to free memory
    del tas, tas_anom, region, weights, raw_index

    # Force garbage collection multiple times
    import gc

    gc.collect()
    gc.collect()
    gc.collect()

    # Clear xarray cache if available
    try:
        import xarray as xr

        xr.set_options(keep_attrs=False)
    except Exception:
        pass

    # Check memory after cleanup
    if debug:
        cleanup_memory = get_memory_usage_mb()
        if cleanup_memory is not None:
            cleanup_change = 0.0
            if final_memory is not None:
                cleanup_change = cleanup_memory - final_memory
            print(
                f"[NINO34] After cleanup: {cleanup_memory:.1f} MB, change: {cleanup_change:+.1f} MB"
            )

            # Warn if memory is still high
            if cleanup_memory > 5000:  # 5GB threshold
                print(
                    f"⚠️  WARNING: High memory usage after cleanup: {cleanup_memory:.1f} MB"
                )

    return index_processed


def get_optimal_jobs(cfg: dict) -> int:
    """Get optimal number of jobs for joblib based on configuration."""
    import multiprocessing as mp

    # Check if we're in a joblib context to prevent nested parallelization explosion
    if os.environ.get("IN_JOBLIB_CONTEXT") == "1":
        # When already in joblib context, use fewer jobs to avoid over-subscription
        # But be much more aggressive for high-CPU systems
        available_cpus = mp.cpu_count()

        if available_cpus >= 100:  # Very high CPU system (like 128 CPUs)
            if cfg.get("n_workers", 0) > 32:  # Aggressive strategy
                return min(40, available_cpus // 3)  # Use up to 40 jobs (1/3 of CPUs)
            elif cfg.get("n_workers", 0) > 16:  # Balanced strategy
                return min(30, available_cpus // 4)  # Use up to 30 jobs (1/4 of CPUs)
            else:
                return min(20, available_cpus // 6)  # Conservative but still aggressive
        elif available_cpus >= 60:  # High CPU system
            if cfg.get("n_workers", 0) > 32:  # Aggressive strategy
                return min(20, available_cpus // 2)  # Much more aggressive - up to 20 jobs
            elif cfg.get("n_workers", 0) > 16:  # Balanced strategy
                return min(12, available_cpus // 4)  # Moderate for balanced
            else:
                return min(8, available_cpus // 6)  # Conservative for conservative
        else:  # Standard system
            if cfg.get("n_workers", 0) > 32:  # Aggressive strategy
                return min(16, available_cpus // 2)  # Aggressive
            elif cfg.get("n_workers", 0) > 16:  # Balanced strategy
                return min(8, available_cpus // 4)  # Moderate
            else:
                return min(4, available_cpus // 6)  # Conservative
    else:
        # Standalone joblib usage - use much more aggressive settings for high-CPU systems
        available_cpus = mp.cpu_count()

        if available_cpus >= 100:  # Very high CPU system (like 128 CPUs)
            # Target ~40-50 CPUs for standalone usage
            target_cpus = 50
            return min(target_cpus, available_cpus // 2)  # Use 1/2 of available CPUs
        elif available_cpus >= 60:  # High CPU system
            # Target ~30 CPUs for standalone usage
            target_cpus = 30
            return min(target_cpus, available_cpus // 2)  # Use 1/2 of available CPUs
        elif available_cpus >= 30:  # Medium CPU system
            # Target ~20 CPUs for standalone usage
            target_cpus = 20
            return min(target_cpus, available_cpus // 2)  # Use 1/2 of available CPUs
        else:  # Low CPU system
            # Conservative approach
            target_cpus = 10
            return min(target_cpus, available_cpus // 2)  # Conservative approach


def filter_spatial_vars(ds: xr.Dataset, vars_in: list[str]) -> list[str]:
    """Keep variables that have lat×lon×time dims."""
    out = []
    for v in vars_in:
        if v not in ds:
            continue
        dims = ds[v].dims
        if (
            "time" in dims
            and any("lat" in d.lower() for d in dims)
            and any("lon" in d.lower() for d in dims)
        ):
            out.append(v)
    return out


# ----------------------- health check hook for suite (NEW) -------------------
def sqlite_check_health() -> bool:
    if not _SQLITE_ENABLED:
        return True
    try:
        _sqlite_init()
        # roundtrip trivial query
        _sqlite_conn().execute("SELECT 1;").fetchone()
        return True
    except Exception as e:
        print(f"❌ SQLite health check failed: {e}")
        return False


# Initialize only if CSV backend is enabled (avoid creating CSVs when using sqlite-only)
if _CSV_ENABLED:
    _ensure_db(_NC_CSV, _COLS_GOF)
    _ensure_db(_CC_CSV, _COLS_GOF)
    _ensure_db(_CZ_CSV, _COLS_GOF)
    _ensure_db(_ZZ_CSV, _COLS_GOF)
    _ensure_db(_NN_CSV, _COLS_GOF)
    _ensure_db(_OC_CSV, _COLS_GOF)
    _ensure_db(_ON_CSV, _COLS_GOF)

# Optional: auto-configure from environment in case workers import before parent calls configure_database()
_env_backends = os.environ.get("METRICS_DB_BACKENDS")
if _env_backends:
    try:
        _env_list = [b.strip().lower() for b in _env_backends.split(",") if b.strip()]
        _env_sqlite = os.environ.get("METRICS_DB_SQLITE_PATH") or None
        configure_database(_env_list, _env_sqlite)
    except Exception as _e:
        dprint(f"⚠️  Env DB auto-config failed: {_e}")

# Allow env to toggle force-overwrite inside workers
_env_force = os.environ.get("METRICS_FORCE_OVERWRITE")
if _env_force is not None:
    try:
        set_force_overwrite(str(_env_force).lower() in ("true", "1", "yes"))
    except Exception as _e:
        dprint(f"⚠️  Env force-overwrite auto-config failed: {_e}")

# Also allow env to toggle the duplicate-check short-circuit in workers
_env_skipdup = os.environ.get("METRICS_SKIP_DUP_CHECK")
if _env_skipdup is not None:
    try:
        set_skip_duplication_check(str(_env_skipdup).lower() in ("1", "true", "yes"))
    except Exception as _e:
        dprint(f"⚠️  Env skip-duplication-check auto-config failed: {_e}")
