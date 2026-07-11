#!/usr/bin/env python3
"""Drive 815 scenpercentiles from local ERA5spliced staging inputs.

This notebook-visible driver can operate in two modes:
- manual mode: read already-staged local ERA5spliced inputs
- auto-localstaging mode: discover canonical mounted S3 n_20 runs, skip
  already-completed outputs, clean other localstaging n_20 contents, stage the
  exact next run via 101, then export percentiles via 815

Key behavior:
- published outputs live under ERA5spliced_localresults
- per-task symlink views live under localresults `_runmeta/views/`
- `/tmp` staging is never used
- child staging/export subprocesses are launched with `nice -n 15`
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scr.validation_helpers.helper_path_utils import (  # noqa: E402
    STORAGE_ACCESS_MOUNT,
    get_era5spliced_cmip6_localresults_root,
    get_era5spliced_localresults_root,
    get_era5spliced_localstaging_root,
    get_era5spliced_root,
    get_object_bucket,
    get_projects_root,
    resolve_canonical_dataset_root,
)
from scr.validation_helpers.era5spliced_publish_state import (  # noqa: E402
    PERCENTILES_LOCAL_PUBLISH_DIRNAME,
    PERCENTILES_LOCAL_PUBLISH_BASENAME,
    PERCENTILES_R2_BUCKET_DEFAULT,
    PERCENTILES_R2_PREFIX_DEFAULT,
    PERCENTILES_R2_REMOTE_DEFAULT,
    build_percentiles_publish_manifest_payload,
    classify_cmip6_remote_state,
    derive_percentiles_publish_timetag,
    global_annual_ensembles_complete,
    legacy_percentiles_remote_complete,
    percentiles_local_publish_manifest_path,
    percentiles_remote_manifest_path,
    rclone_cat_json,
    rclone_copyto_json,
    remote_percentiles_manifest_matches,
    verify_percentiles_remote_listing,
    write_json_atomic as write_publish_json_atomic,
)
from scr.validation_helpers.scenario_source_resolution_815 import (  # noqa: E402
    dedupe_member_files,
    experiment_token,
    member_identity,
)

AUTOLOCALSTAGING = True
AUTOLOCALSTAGING_VERSIONS = ("v100", "v101")
AUTOLOCALSTAGING_ENSEMBLE = "n_20"

DEFAULT_VERSIONS = ("v100", "v101")
DEFAULT_SCENARIOS = (
    "Current-Policies-GCAM",
    "Current-Policies-MESSAGE",
    "Current-Policies-REMIND",
    "H",
    "HL",
    "L",
    "LN",
    "M",
    "VL",
    "NDC-Trump-high",
    "NDC-Trump-low",
    "NDC-submitted-high",
    "NDC-submitted-low",
    "SSP2-com",
    "ssp119",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp434",
    "ssp460",
    "ssp534-over",
    "ssp585",
)

DEFAULT_VARIABLES = (
    "pr",
    "tas",
    "tasmax",
    "psl",
    "rsds",
    "sfcWind",
    "hurs",
    "tasmin",
    "ts",
    "huss",
)

DEFAULT_SEASONS = (
    "annual",
    "DJF",
    "MAM",
    "JJA",
    "SON",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
DEFAULT_REGIONS = ("AUTO",)

DEFAULT_LOCALSTAGING_ROOT = get_era5spliced_localstaging_root()
DEFAULT_LOCALRESULTS_ROOT = get_era5spliced_localresults_root()
DEFAULT_CMIP6_LOCALRESULTS_ROOT = get_era5spliced_cmip6_localresults_root()
DEFAULT_SOURCE_ROOT = get_era5spliced_root()
DEFAULT_101_SCRIPT = REPO_ROOT / "notebooks" / "101_stage_S3_GCMAGICCfiles.py"
DEFAULT_350_SCRIPT = REPO_ROOT / "notebooks" / "350_convertMultiVarNC_intoCMIP6files.py"
DEFAULT_815_SCRIPT = REPO_ROOT / "notebooks" / "815_simple_plot_SSPprojections.py"
DEFAULT_SCENARIO_PUBLISH_SCRIPT = (
    get_projects_root() / "gcm_firefly_frontend" / "scripts" / "sync_scenario_projection_merge.py"
).resolve(strict=False)
DEFAULT_CMIP6_UPLOAD_PREFIX = "nc/cmip6/era5spliced"
DEFAULT_STAGE101_WORKERS = os.environ.get("GCMAGICC_101_DEFAULT_WORKERS", "10")
DEFAULT_CMIP6_MEMBER_WORKERS = max(1, int(os.environ.get("GCMAGICC_350_MEMBER_WORKERS", "3")))
DEFAULT_CMIP6_FRONTEND_CATALOG = (
    get_projects_root()
    / "gcm_firefly_frontend"
    / "app"
    / "frontend"
    / "public"
    / "data"
    / "gcmagicc_cmip6_data_catalog.json"
).resolve(strict=False)

DEFAULT_META_SUBDIR = "_runmeta"
CMIP6_META_SUBDIR = "_meta"
DEFAULT_R2_REMOTE = PERCENTILES_R2_REMOTE_DEFAULT
DEFAULT_R2_BUCKET = PERCENTILES_R2_BUCKET_DEFAULT
DEFAULT_R2_PREFIX = PERCENTILES_R2_PREFIX_DEFAULT
DEFAULT_SCENARIO_DEFAULT_VERSION = "v101"
DEFAULT_SCENARIO_OBS_HADCRUT_URL = (
    "https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.1.0.0/"
    "analysis/diagnostics/HadCRUT.5.1.0.0.analysis.summary_series.global.annual.csv"
)
DEFAULT_SCENARIO_OBS_BERKELEY_URL = (
    "https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Land_and_Ocean_complete.txt"
)
DEFAULT_SCENARIO_OBS_NORM_START = 1995
DEFAULT_SCENARIO_OBS_NORM_END = 2014
DEFAULT_STALL_HOURS = 6.0
DEFAULT_POLL_SECONDS = 300
EXPECTED_MEMBER_COUNT = 20
TMP_STAGE_PREFIX = "815_stage"
RUN_COMPLETE_BASENAME = "run_complete.json"
RUN_MANIFEST_BASENAME = "run_manifest.json"
UPLOAD_MANIFEST_BASENAME = "upload_manifest.json"
LOCAL_CLEANUP_BASENAME = "local_cleanup.json"
ARX_NAME = "AR6"
RUNMODUS_NAME = "all"
CHILD_NICE_LEVEL = 15

ERROR_PATTERNS = (
    re.compile(r"No such file", re.IGNORECASE),
    re.compile(r"FileNotFoundError", re.IGNORECASE),
    re.compile(r"Input/output error", re.IGNORECASE),
    re.compile(r"\bI/O error\b", re.IGNORECASE),
    re.compile(r"stale file handle", re.IGNORECASE),
    re.compile(r"Permission denied", re.IGNORECASE),
    re.compile(r"\bOSError\b", re.IGNORECASE),
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Unsupported JSON type: {type(obj)!r}")


def normalize_version(version: str) -> str:
    token = str(version or "").strip().lower()
    if token.startswith("v100"):
        return "v100"
    if token.startswith("v101"):
        return "v101"
    if token and not token.startswith("v"):
        token = f"v{token}"
    return token


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=json_default) + "\n")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def merge_rows(
    existing: Sequence[Dict[str, Any]],
    new_rows: Sequence[Dict[str, Any]],
    *,
    key_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    def _key(row: Dict[str, Any]) -> Tuple[str, ...]:
        return tuple(str(row.get(field, "")) for field in key_fields)

    for row in existing:
        if not isinstance(row, dict):
            continue
        merged[_key(row)] = row
    for row in new_rows:
        if not isinstance(row, dict):
            continue
        merged[_key(row)] = row
    return list(merged.values())


def load_completed_tasks_from_ledger(path: Path) -> Set[Tuple[str, str]]:
    completed: Set[Tuple[str, str]] = set()
    if not path.exists():
        return completed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row_text = line.strip()
            if not row_text:
                continue
            try:
                row = json.loads(row_text)
            except Exception:
                continue
            if row.get("event") != "task_end":
                continue
            if row.get("status") != "completed":
                continue
            version = str(row.get("version", "")).strip()
            scenario = str(row.get("scenario", "")).strip()
            if version and scenario:
                completed.add((version, scenario))
    return completed


def read_proc_state_and_ticks(pid: int) -> Tuple[Optional[str], Optional[int]]:
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return None, None
    try:
        raw = stat_path.read_text(encoding="utf-8", errors="replace")
        parts = raw.split()
        if len(parts) < 15:
            return None, None
        state = parts[2]
        ticks = int(parts[13]) + int(parts[14])
        return state, ticks
    except Exception:
        return None, None


def count_percentiles(root: Path) -> int:
    if not root.exists():
        return 0
    count = 0
    for _ in root.rglob("percentiles.json"):
        count += 1
    return count


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _directory_size_bytes(path: Path) -> Tuple[int, int, Optional[float]]:
    total_bytes = 0
    file_count = 0
    latest_mtime: Optional[float] = None
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        try:
            stat = child.stat()
        except OSError:
            continue
        total_bytes += int(stat.st_size)
        file_count += 1
        mtime = float(stat.st_mtime)
        if latest_mtime is None or mtime > latest_mtime:
            latest_mtime = mtime
    return total_bytes, file_count, latest_mtime


def summarize_tmp_stage_dirs() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    tmp_root = Path("/tmp")
    if not tmp_root.exists():
        return out
    for path in sorted(tmp_root.glob(f"{TMP_STAGE_PREFIX}*")):
        if not path.is_dir():
            continue
        total_bytes, file_count, latest_mtime = _directory_size_bytes(path)
        out.append(
            {
                "path": str(path),
                "bytes": total_bytes,
                "file_count": file_count,
                "latest_mtime_utc": (
                    datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat()
                    if latest_mtime is not None
                    else None
                ),
            }
        )
    return out


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TiB"


def _nice_prefix() -> List[str]:
    return ["nice", "-n", str(CHILD_NICE_LEVEL)]


def _extract_new_error_lines(text: str) -> List[str]:
    out: List[str] = []
    if not text:
        return out
    for line in text.splitlines():
        for pat in ERROR_PATTERNS:
            if pat.search(line):
                out.append(line.strip())
                break
    return out


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=60)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=15)
        except Exception:
            return


@dataclass
class Task:
    version: str
    scenario: str
    source_run_root: Path
    source_files: List[Path]
    output_root: Path
    member_identities: List[str] = field(default_factory=list)

    @property
    def run_instance(self) -> str:
        return self.source_run_root.name


@dataclass
class AutoTask:
    version: str
    scenario: str
    ensemble: str
    source_run_root: Path
    run_instance: str
    stage_root: Path
    output_root: Path
    cmip6_output_root: Path
    member_count: int
    completion_state: str = "pending_both"
    percentiles_state: str = "missing"
    percentiles_compute_source: Optional[str] = None
    percentiles_publish_source: Optional[str] = None
    percentiles_publish_timetag: Optional[str] = None
    cmip6_completion_source: Optional[str] = None
    cmip6_state: str = "missing"
    member_identities: List[str] = field(default_factory=list)


@dataclass
class AttemptResult:
    attempt: int
    mode: str
    status: str
    returncode: Optional[int]
    started_at: datetime
    finished_at: datetime
    duration_s: float
    outputs_before: int
    outputs_after: int
    output_delta: int
    process_state_last: Optional[str]
    errors: List[str] = field(default_factory=list)
    log_path: Optional[Path] = None
    note: str = ""


@dataclass
class TaskResult:
    version: str
    scenario: str
    status: str
    source_run_root: Path
    output_root: Path
    cmip6_output_root: Optional[Path] = None
    attempts: List[AttemptResult] = field(default_factory=list)
    reason: str = ""
    stage_status_json: Optional[Path] = None
    stage_manifest_json: Optional[Path] = None
    stage_log_path: Optional[Path] = None
    completion_marker_path: Optional[Path] = None
    cmip6_log_path: Optional[Path] = None
    cmip6_manifest_path: Optional[Path] = None
    cmip6_upload_manifest_path: Optional[Path] = None
    cmip6_local_cleanup_manifest_path: Optional[Path] = None
    cmip6_completion_marker_path: Optional[Path] = None
    completion_state: str = "pending_both"
    percentiles_state: str = "missing"
    percentiles_publish_status: str = ""
    percentiles_publish_reason: str = ""
    percentiles_local_publish_manifest_path: Optional[Path] = None
    percentiles_remote_publish_manifest_path: Optional[str] = None
    percentiles_publish_timetag: Optional[str] = None


@dataclass
class PercentilesPublishResult:
    version: str
    scenario: str
    run_instance: str
    status: str
    reason: str = ""
    publish_timetag: Optional[str] = None
    local_manifest_path: Optional[Path] = None
    remote_manifest_path: Optional[str] = None
    local_deleted: bool = False
    percentiles_file_count: int = 0


@dataclass
class CompletedOutputIndex:
    marker_roots: Set[str] = field(default_factory=set)
    manifest_roots: Set[str] = field(default_factory=set)
    ledger_roots: Set[str] = field(default_factory=set)

    def contains(self, output_root: Path) -> bool:
        key = str(output_root.expanduser().resolve(strict=False))
        return key in self.marker_roots or key in self.manifest_roots or key in self.ledger_roots

    def source_for(self, output_root: Path) -> Optional[str]:
        key = str(output_root.expanduser().resolve(strict=False))
        if key in self.marker_roots:
            return "run_complete.json"
        if key in self.manifest_roots:
            return "final_manifest.json"
        if key in self.ledger_roots:
            return "run_ledger.jsonl"
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Drive 815 scenpercentiles from local ERA5spliced staging runs."
    )
    p.add_argument(
        "--auto-localstaging",
        action=argparse.BooleanOptionalAction,
        default=AUTOLOCALSTAGING,
        help="Auto-discover mounted S3 n_20 tasks, stage them with 101, then export via 815.",
    )
    p.add_argument("--versions", nargs="+", default=None, help="Version filter list (e.g. v100 v101)")
    p.add_argument("--scenarios", nargs="+", default=None, help="Scenario filter list")
    p.add_argument("--timetag", default=utc_stamp(), help="Shared output timetag")
    p.add_argument(
        "--localstaging-root",
        type=Path,
        default=DEFAULT_LOCALSTAGING_ROOT,
        help="Root containing canonical staged originals",
    )
    p.add_argument(
        "--localresults-root",
        type=Path,
        default=DEFAULT_LOCALRESULTS_ROOT,
        help="Root containing canonical localresults outputs",
    )
    p.add_argument(
        "--cmip6-localresults-root",
        type=Path,
        default=DEFAULT_CMIP6_LOCALRESULTS_ROOT,
        help="Root containing CMIP6-style converted local outputs",
    )
    p.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Mounted canonical ERA5splicedS3 root used for auto-localstaging discovery",
    )
    p.add_argument("--python", default=sys.executable, help="Python executable for subprocesses")
    p.add_argument(
        "--script-815",
        type=Path,
        default=DEFAULT_815_SCRIPT,
        help="Path to 815 exporter script",
    )
    p.add_argument(
        "--script-350",
        type=Path,
        default=DEFAULT_350_SCRIPT,
        help="Path to 350 CMIP6 conversion script",
    )
    p.add_argument(
        "--run-cmip6",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In auto-localstaging mode, also run CMIP6 conversion + upload after 815.",
    )
    p.add_argument(
        "--publish-percentiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish completed 815 percentiles to R2 and reclaim local disk after verification.",
    )
    p.add_argument(
        "--backlog-only",
        action="store_true",
        help="Do not stage or recompute. Only offload/clean existing local 815 + CMIP6 outputs.",
    )
    p.add_argument(
        "--scenario-publish-script",
        type=Path,
        default=DEFAULT_SCENARIO_PUBLISH_SCRIPT,
        help="Merge-aware frontend publish script for raw 815 outputs.",
    )
    p.add_argument("--r2-remote", default=DEFAULT_R2_REMOTE, help="Rclone remote name for public R2 publishes.")
    p.add_argument("--r2-bucket", default=DEFAULT_R2_BUCKET, help="Public R2 bucket for 815 percentiles.")
    p.add_argument(
        "--r2-prefix",
        default=DEFAULT_R2_PREFIX,
        help="Public R2 prefix root for 815 percentiles.",
    )
    p.add_argument(
        "--scenario-default-version",
        default=DEFAULT_SCENARIO_DEFAULT_VERSION,
        help="Default version id written into the merged scenario-projection catalog.",
    )
    p.add_argument("--scenario-obs-hadcrut-url", default=DEFAULT_SCENARIO_OBS_HADCRUT_URL)
    p.add_argument("--scenario-obs-berkeley-url", default=DEFAULT_SCENARIO_OBS_BERKELEY_URL)
    p.add_argument("--scenario-obs-norm-start", type=int, default=DEFAULT_SCENARIO_OBS_NORM_START)
    p.add_argument("--scenario-obs-norm-end", type=int, default=DEFAULT_SCENARIO_OBS_NORM_END)
    p.add_argument(
        "--cmip6-upload-prefix",
        default=DEFAULT_CMIP6_UPLOAD_PREFIX,
        help="Target OVH prefix root for converted CMIP6 outputs.",
    )
    p.add_argument(
        "--cmip6-upload-bucket",
        default=get_object_bucket(),
        help="Target OVH bucket for converted CMIP6 outputs.",
    )
    p.add_argument(
        "--cmip6-frontend-catalog",
        type=Path,
        default=DEFAULT_CMIP6_FRONTEND_CATALOG,
        help="Frontend CMIP6 catalog JSON updated incrementally by the 350 worker.",
    )
    p.add_argument(
        "--cmip6-skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --skip-if-complete to the 350 worker.",
    )
    p.add_argument(
        "--cleanup-local-after-upload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After successful CMIP6-like upload/verification, remove the generated local CMIP6-style payload files.",
    )
    p.add_argument(
        "--cmip6-member-workers",
        type=int,
        default=DEFAULT_CMIP6_MEMBER_WORKERS,
        help="350 member-parallelism. Default: 3.",
    )
    p.add_argument("--stall-hours", type=float, default=DEFAULT_STALL_HOURS, help="No-heartbeat stall gate in hours")
    p.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS, help="Polling interval for progress checks")
    p.add_argument("--workers-per-scenario", type=int, default=1, help="815 --workers value per scenario run")
    p.add_argument(
        "--variables",
        nargs="+",
        default=list(DEFAULT_VARIABLES),
        help="Variables forwarded to 815 for compute/backfill targeting.",
    )
    p.add_argument(
        "--seasons",
        nargs="+",
        default=list(DEFAULT_SEASONS),
        help="Seasons forwarded to 815 for compute/backfill targeting.",
    )
    p.add_argument(
        "--regions",
        nargs="+",
        default=list(DEFAULT_REGIONS),
        help="Regions forwarded to 815; use AUTO for canonical global + AR6 + ISO3 discovery.",
    )
    p.add_argument(
        "--also-crunch-cmip6",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass through 815 CMIP6 sidecar crunching (one member per source model for SSP scenarios).",
    )
    p.add_argument(
        "--cmip6-reference-dir",
        type=Path,
        default=None,
        help="Optional explicit vetted CMIP6 directory forwarded to 815 sidecar generation.",
    )
    p.add_argument(
        "--cmip6-stage-root",
        type=Path,
        default=None,
        help="Optional CMIP6 staging-root override forwarded to 815 sidecar generation.",
    )
    p.add_argument(
        "--stage101-workers",
        default=DEFAULT_STAGE101_WORKERS,
        help="101 staging parallel download workers or 'auto'. Default: 10.",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip completed task entries already recorded in the current ledger",
    )
    p.add_argument("--dry-run", action="store_true", help="Resolve tasks and write manifest without executing")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO"], help="Driver verbosity")
    return p.parse_args()


def build_output_root(*, localresults_root: Path, version: str, scenario: str, run_instance: str) -> Path:
    return (
        localresults_root
        / version
        / scenario
        / ARX_NAME
        / RUNMODUS_NAME
        / AUTOLOCALSTAGING_ENSEMBLE
        / "dataderivatives"
        / "815_scenpercentiles"
        / run_instance
    )


def build_cmip6_output_root(
    *,
    cmip6_localresults_root: Path,
    version: str,
    scenario: str,
    run_instance: str,
) -> Path:
    return (
        cmip6_localresults_root
        / version
        / scenario
        / ARX_NAME
        / RUNMODUS_NAME
        / AUTOLOCALSTAGING_ENSEMBLE
        / "original"
        / run_instance
    )


def build_cmip6_remote_run_prefix(
    *,
    upload_prefix: str,
    version: str,
    scenario: str,
    run_instance: str,
) -> str:
    return "/".join(
        [
            str(upload_prefix).strip("/"),
            version,
            scenario,
            ARX_NAME,
            RUNMODUS_NAME,
            AUTOLOCALSTAGING_ENSEMBLE,
            "original",
            run_instance,
        ]
    )


def _files_for_requested_experiment(files: Sequence[Path], scenario: str) -> Tuple[List[Path], List[str]]:
    target = str(scenario).strip().lower()
    matched: List[Path] = []
    discovered: Set[str] = set()
    for path in files:
        token = experiment_token(path)
        if token is None:
            continue
        discovered.add(token)
        if token == target:
            matched.append(path)
    if discovered:
        return matched, sorted(discovered)
    return list(files), []


def build_task_from_run_root(
    *,
    version: str,
    scenario: str,
    source_run_root: Path,
    localresults_root: Path,
) -> Task:
    if not source_run_root.exists():
        raise FileNotFoundError(f"Missing run root: {source_run_root}")
    files = sorted([p for p in source_run_root.glob("*.nc") if p.is_file()])
    scenario_files, discovered_experiments = _files_for_requested_experiment(files, scenario)
    if not scenario_files:
        if discovered_experiments:
            raise RuntimeError(
                f"{version}/{scenario}: no files matching experiment '{scenario}' under "
                f"{source_run_root}. Found experiments={','.join(discovered_experiments)}"
            )
        raise RuntimeError(f"{version}/{scenario}: no .nc files found under {source_run_root}")
    dedup = dedupe_member_files(scenario_files)
    if len(dedup) != EXPECTED_MEMBER_COUNT:
        experiment_note = (
            f" after filtering to experiment '{scenario}'"
            if discovered_experiments
            else ""
        )
        raise RuntimeError(
            f"{version}/{scenario}: expected {EXPECTED_MEMBER_COUNT} deduplicated members under "
            f"{source_run_root}, found {len(dedup)}{experiment_note}"
        )
    return Task(
        version=version,
        scenario=scenario,
        source_run_root=source_run_root,
        source_files=dedup,
        output_root=build_output_root(
            localresults_root=localresults_root,
            version=version,
            scenario=scenario,
            run_instance=source_run_root.name,
        ),
        member_identities=[member_identity(p) for p in dedup],
    )


def _run_dir_sort_key(path: Path) -> Tuple[int, str]:
    try:
        mtime_ns = int(path.stat().st_mtime_ns)
    except OSError:
        mtime_ns = -1
    return (mtime_ns, path.name)


def resolve_task(
    *,
    version: str,
    scenario: str,
    localstaging_root: Path,
    localresults_root: Path,
) -> Task:
    original_root = (
        localstaging_root
        / version
        / scenario
        / ARX_NAME
        / RUNMODUS_NAME
        / AUTOLOCALSTAGING_ENSEMBLE
        / "original"
    )
    if not original_root.exists():
        raise FileNotFoundError(f"Missing original root: {original_root}")

    run_dirs = sorted(
        [p for p in original_root.glob("run_*") if p.is_dir()],
        key=_run_dir_sort_key,
        reverse=True,
    )
    if not run_dirs:
        raise RuntimeError(f"{version}/{scenario}: no run_* directories found under {original_root}")

    details: List[str] = []
    for run_dir in run_dirs:
        files = sorted([p for p in run_dir.glob("*.nc") if p.is_file()])
        scenario_files, discovered_experiments = _files_for_requested_experiment(files, scenario)
        dedup = dedupe_member_files(scenario_files)
        detail = f"{run_dir.name}:{len(dedup)}"
        if discovered_experiments:
            detail += f"[{','.join(discovered_experiments)}]"
        details.append(detail)
        if len(dedup) == EXPECTED_MEMBER_COUNT:
            return build_task_from_run_root(
                version=version,
                scenario=scenario,
                source_run_root=run_dir,
                localresults_root=localresults_root,
            )

    raise RuntimeError(
        f"{version}/{scenario}: no run_* directory has deduplicated member count "
        f"{EXPECTED_MEMBER_COUNT}. Candidates={', '.join(details)}"
    )


def build_manual_tasks(args: argparse.Namespace) -> Tuple[List[Task], List[Dict[str, Any]]]:
    versions = [normalize_version(v) for v in (args.versions or list(DEFAULT_VERSIONS))]
    scenarios = list(dict.fromkeys(args.scenarios or list(DEFAULT_SCENARIOS)))
    tasks: List[Task] = []
    skipped: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for version in versions:
        for scenario in scenarios:
            key = (version, scenario)
            if key in seen:
                continue
            seen.add(key)
            try:
                tasks.append(
                    resolve_task(
                        version=version,
                        scenario=scenario,
                        localstaging_root=args.localstaging_root,
                        localresults_root=args.localresults_root,
                    )
                )
            except Exception as exc:
                skipped.append(
                    {
                        "version": version,
                        "scenario": scenario,
                        "reason": str(exc),
                    }
                )
    return tasks, skipped


def _discover_auto_scenarios(source_root: Path, version: str) -> List[str]:
    version_dir = source_root / version
    if not version_dir.exists():
        return []
    out: List[str] = []
    for scenario_dir in sorted(version_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        original_root = scenario_dir / ARX_NAME / RUNMODUS_NAME / AUTOLOCALSTAGING_ENSEMBLE / "original"
        if original_root.is_dir():
            out.append(scenario_dir.name)
    return out


def resolve_auto_task(
    *,
    source_root: Path,
    localstaging_root: Path,
    localresults_root: Path,
    cmip6_localresults_root: Path,
    version: str,
    scenario: str,
) -> AutoTask:
    source_run_root = Path(
        resolve_canonical_dataset_root(
            version=version,
            experiment_id=scenario,
            arx=ARX_NAME,
            runmodus=RUNMODUS_NAME,
            n_ensemble=AUTOLOCALSTAGING_ENSEMBLE,
            kind="original",
            root=source_root,
        )
    ).expanduser().resolve(strict=False)
    if not source_run_root.exists():
        raise FileNotFoundError(f"Resolved source root does not exist: {source_run_root}")

    files = sorted([p for p in source_run_root.glob("*.nc") if p.is_file()])
    scenario_files, discovered_experiments = _files_for_requested_experiment(files, scenario)
    if not scenario_files:
        if discovered_experiments:
            raise RuntimeError(
                f"{version}/{scenario}: no files matching experiment '{scenario}' under "
                f"{source_run_root}. Found experiments={','.join(discovered_experiments)}"
            )
        raise RuntimeError(f"{version}/{scenario}: no .nc files found under {source_run_root}")
    dedup = dedupe_member_files(scenario_files)
    if len(dedup) != EXPECTED_MEMBER_COUNT:
        experiment_note = (
            f" after filtering to experiment '{scenario}'"
            if discovered_experiments
            else ""
        )
        raise RuntimeError(
            f"{version}/{scenario}: expected {EXPECTED_MEMBER_COUNT} deduplicated mounted members under "
            f"{source_run_root}, found {len(dedup)}{experiment_note}"
        )

    rel = source_run_root.relative_to(source_root)
    stage_root = (localstaging_root / rel).resolve(strict=False)
    run_instance = source_run_root.name
    return AutoTask(
        version=version,
        scenario=scenario,
        ensemble=AUTOLOCALSTAGING_ENSEMBLE,
        source_run_root=source_run_root,
        run_instance=run_instance,
        stage_root=stage_root,
        output_root=build_output_root(
            localresults_root=localresults_root,
            version=version,
            scenario=scenario,
            run_instance=run_instance,
        ),
        cmip6_output_root=build_cmip6_output_root(
            cmip6_localresults_root=cmip6_localresults_root,
            version=version,
            scenario=scenario,
            run_instance=run_instance,
        ),
        member_count=len(dedup),
        member_identities=[member_identity(p) for p in dedup],
    )


def build_completed_output_index(localresults_root: Path) -> CompletedOutputIndex:
    index = CompletedOutputIndex()

    for marker in localresults_root.rglob(RUN_COMPLETE_BASENAME):
        output_root = marker.parent.expanduser().resolve(strict=False)
        index.marker_roots.add(str(output_root))

    meta_base = localresults_root / DEFAULT_META_SUBDIR
    if not meta_base.exists():
        return index

    for manifest_path in sorted(meta_base.glob("815_localstaging_*/final_manifest.json")):
        payload = load_json_file(manifest_path)
        for row in payload.get("results", []):
            if not isinstance(row, dict):
                continue
            if str(row.get("status", "")).strip() != "completed":
                continue
            output_root = str(row.get("output_root", "")).strip()
            if output_root:
                index.manifest_roots.add(str(Path(output_root).expanduser().resolve(strict=False)))

    for ledger_path in sorted(meta_base.glob("815_localstaging_*/run_ledger.jsonl")):
        current_output_root: Dict[Tuple[str, str], str] = {}
        try:
            lines = ledger_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in lines:
            row_text = line.strip()
            if not row_text:
                continue
            try:
                row = json.loads(row_text)
            except Exception:
                continue
            version = str(row.get("version", "")).strip()
            scenario = str(row.get("scenario", "")).strip()
            key = (version, scenario)
            event = str(row.get("event", "")).strip()
            if event == "task_start":
                output_root = str(row.get("output_root", "")).strip()
                if output_root:
                    current_output_root[key] = str(Path(output_root).expanduser().resolve(strict=False))
            elif event == "task_end" and str(row.get("status", "")).strip() == "completed":
                output_root = str(row.get("output_root", "")).strip() or current_output_root.get(key, "")
                if output_root:
                    index.ledger_roots.add(str(Path(output_root).expanduser().resolve(strict=False)))
            elif event == "export815_completed":
                output_root = str(row.get("output_root", "")).strip()
                if output_root:
                    index.ledger_roots.add(str(Path(output_root).expanduser().resolve(strict=False)))

    return index


def _path_exists_with_files(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        next(path.rglob("*"))
    except StopIteration:
        return False
    return True


def percentiles_state_info(
    *,
    args: argparse.Namespace,
    output_root: Path,
    version: str,
    scenario: str,
    run_instance: str,
) -> Dict[str, Any]:
    marker_path = completion_marker_path(output_root)
    local_marker_exists = marker_path.exists()
    local_ensembles_complete = global_annual_ensembles_complete(output_root) if output_root.exists() else False
    local_manifest_path = percentiles_local_publish_manifest_path(output_root)
    local_manifest_payload = load_json_file(local_manifest_path)
    publish_timetag = (
        str(local_manifest_payload.get("publish_timetag") or "").strip()
        if local_manifest_payload
        else ""
    )
    if not publish_timetag and output_root.exists():
        publish_timetag = derive_percentiles_publish_timetag(output_root)

    remote_manifest_path = percentiles_remote_manifest_path(
        version=version,
        scenario=scenario,
        run_instance=run_instance,
        remote=str(args.r2_remote),
        bucket=str(args.r2_bucket),
    )
    try:
        remote_manifest_payload = rclone_cat_json(remote_manifest_path)
        remote_complete = remote_percentiles_manifest_matches(
            remote_manifest_payload,
            version=version,
            scenario=scenario,
            run_instance=run_instance,
            output_root=output_root if output_root.exists() else None,
            publish_timetag=publish_timetag or None,
            prefix_root=str(args.r2_prefix),
        )
    except Exception:
        remote_manifest_payload = {}
        remote_complete = False
    local_compute_complete = local_marker_exists and local_ensembles_complete
    local_publish_complete = local_compute_complete and bool(local_manifest_payload) and remote_percentiles_manifest_matches(
        local_manifest_payload,
        version=version,
        scenario=scenario,
        run_instance=run_instance,
        output_root=output_root if output_root.exists() else None,
        publish_timetag=publish_timetag or None,
        prefix_root=str(args.r2_prefix),
    )
    partial_local = output_root.exists() and not local_compute_complete and (
        _path_exists_with_files(output_root) or count_percentiles(output_root) > 0
    )

    if remote_complete and local_compute_complete:
        status = "done_both"
        compute_source = "run_complete.json"
        publish_source = "remote r2 publish manifest"
    elif remote_complete:
        status = "done_remote_only"
        compute_source = "remote r2 publish manifest"
        publish_source = "remote r2 publish manifest"
    elif local_compute_complete:
        status = "done_local_only"
        compute_source = "run_complete.json"
        publish_source = (
            f"{PERCENTILES_LOCAL_PUBLISH_DIRNAME}/{PERCENTILES_LOCAL_PUBLISH_BASENAME}"
            if local_publish_complete
            else None
        )
    elif partial_local:
        status = "partial_local"
        compute_source = None
        publish_source = None
    else:
        status = "missing"
        compute_source = None
        publish_source = None

    return {
        "status": status,
        "compute_complete": status in {"done_local_only", "done_remote_only", "done_both"},
        "publish_complete": bool(remote_complete),
        "compute_source": compute_source,
        "publish_source": publish_source,
        "local_marker_exists": local_marker_exists,
        "local_ensembles_complete": local_ensembles_complete,
        "local_manifest_path": local_manifest_path if local_manifest_payload else None,
        "remote_manifest_path": remote_manifest_path if remote_manifest_payload else None,
        "publish_timetag": publish_timetag or None,
    }


def cmip6_completion_source(output_root: Path) -> Optional[str]:
    marker = completion_marker_path(output_root)
    upload_manifest = output_root / CMIP6_META_SUBDIR / UPLOAD_MANIFEST_BASENAME
    cleanup_manifest = output_root / CMIP6_META_SUBDIR / LOCAL_CLEANUP_BASENAME
    if marker.exists() and upload_manifest.exists() and cleanup_manifest.exists():
        return "run_complete.json + upload_manifest.json + local_cleanup.json"
    return None


def cmip6_state_info(
    *,
    args: argparse.Namespace,
    output_root: Path,
    version: str,
    scenario: str,
    run_instance: str,
) -> Dict[str, Any]:
    local_source = cmip6_completion_source(output_root)
    remote_prefix = build_cmip6_remote_run_prefix(
        upload_prefix=args.cmip6_upload_prefix,
        version=version,
        scenario=scenario,
        run_instance=run_instance,
    )
    try:
        remote_state = classify_cmip6_remote_state(
            remote_run_prefix=remote_prefix,
            bucket=str(args.cmip6_upload_bucket),
        )
    except Exception:
        remote_state = {"status": "missing", "source": "", "remote_run_prefix": remote_prefix}
    remote_status = str(remote_state.get("status") or "missing")
    partial_local = output_root.exists() and not local_source and _path_exists_with_files(output_root)

    if local_source and remote_status == "done_remote_only":
        status = "done_both"
        source = f"{local_source} + remote completion markers"
    elif remote_status == "done_remote_only":
        status = "done_remote_only"
        source = "remote completion markers"
    elif local_source:
        status = "done_local_only"
        source = local_source
    elif remote_status == "partial_remote":
        status = "partial_remote"
        source = str(remote_state.get("source") or "")
    elif partial_local:
        status = "partial_local"
        source = ""
    else:
        status = "missing"
        source = ""

    return {
        "status": status,
        "compute_complete": status in {"done_local_only", "done_remote_only", "done_both"},
        "completion_source": source or None,
        "remote_prefix": remote_prefix,
        "remote_state": remote_state,
    }


def build_auto_tasks(
    args: argparse.Namespace,
    completed_index: CompletedOutputIndex,
) -> Tuple[List[AutoTask], List[Dict[str, Any]], List[Dict[str, Any]]]:
    versions = [normalize_version(v) for v in (args.versions or list(AUTOLOCALSTAGING_VERSIONS))]
    discovered_rows: List[Dict[str, Any]] = []
    pending: List[AutoTask] = []
    skipped_preflight: List[Dict[str, Any]] = []

    for version in versions:
        scenario_names = (
            list(dict.fromkeys(args.scenarios))
            if args.scenarios
            else _discover_auto_scenarios(args.source_root, version)
        )
        for scenario in scenario_names:
            try:
                auto_task = resolve_auto_task(
                    source_root=args.source_root,
                    localstaging_root=args.localstaging_root,
                    localresults_root=args.localresults_root,
                    cmip6_localresults_root=args.cmip6_localresults_root,
                    version=version,
                    scenario=scenario,
                )
            except Exception as exc:
                skipped_row = {
                    "version": version,
                    "scenario": scenario,
                    "status": "preflight_skipped",
                    "reason": str(exc),
                }
                discovered_rows.append(skipped_row)
                skipped_preflight.append(skipped_row)
                continue

            percentiles_state = percentiles_state_info(
                args=args,
                output_root=auto_task.output_root,
                version=auto_task.version,
                scenario=auto_task.scenario,
                run_instance=auto_task.run_instance,
            )
            cmip6_state = (
                cmip6_state_info(
                    args=args,
                    output_root=auto_task.cmip6_output_root,
                    version=auto_task.version,
                    scenario=auto_task.scenario,
                    run_instance=auto_task.run_instance,
                )
                if args.run_cmip6
                else {"status": "disabled", "completion_source": "disabled", "remote_prefix": "", "remote_state": {}}
            )
            export_completion_source = percentiles_state.get("compute_source")
            cmip6_source = cmip6_state.get("completion_source")
            cmip6_complete = bool(cmip6_state.get("compute_complete")) if args.run_cmip6 else True
            needs_percentiles_publish = bool(args.publish_percentiles) and percentiles_state["status"] == "done_local_only"
            if percentiles_state["compute_complete"] and cmip6_complete and not needs_percentiles_publish:
                status = "done"
            elif needs_percentiles_publish and cmip6_complete:
                status = "pending_percentiles_publish_only"
            elif needs_percentiles_publish:
                status = "pending_percentiles_publish_and_cmip6"
            elif percentiles_state["compute_complete"]:
                status = "pending_cmip6_only"
            elif cmip6_source:
                status = "pending_815_only"
            else:
                status = "pending_both"
            auto_task.percentiles_state = str(percentiles_state.get("status") or "missing")
            auto_task.percentiles_compute_source = export_completion_source
            auto_task.percentiles_publish_source = percentiles_state.get("publish_source")
            auto_task.percentiles_publish_timetag = percentiles_state.get("publish_timetag")
            auto_task.cmip6_completion_source = cmip6_source
            auto_task.cmip6_state = str(cmip6_state.get("status") or "missing")
            auto_task.completion_state = status
            row = {
                "version": auto_task.version,
                "scenario": auto_task.scenario,
                "ensemble": auto_task.ensemble,
                "run_instance": auto_task.run_instance,
                "source_run_root": str(auto_task.source_run_root),
                "stage_root": str(auto_task.stage_root),
                "output_root": str(auto_task.output_root),
                "cmip6_output_root": str(auto_task.cmip6_output_root),
                "member_count": auto_task.member_count,
                "member_identities": auto_task.member_identities,
                "status": status,
                "percentiles_state": auto_task.percentiles_state,
                "reason": (
                    (
                        "815 output already exists; CMIP6 disabled."
                        if (status == "done" and not args.run_cmip6)
                        else "Both 815 and CMIP6 outputs already exist."
                    )
                    if status == "done"
                    else ""
                ),
                "percentiles_compute_source": export_completion_source,
                "percentiles_publish_source": auto_task.percentiles_publish_source,
                "percentiles_publish_timetag": auto_task.percentiles_publish_timetag,
                "percentiles_remote_manifest_path": (
                    percentiles_state.get("remote_manifest_path")
                    if percentiles_state.get("remote_manifest_path")
                    else None
                ),
                "cmip6_completion_source": cmip6_source,
                "cmip6_state": auto_task.cmip6_state,
                "cmip6_remote_prefix": cmip6_state.get("remote_prefix"),
            }
            discovered_rows.append(row)
            if status != "done":
                pending.append(auto_task)

    discovered_rows.sort(
        key=lambda row: (
            str(row.get("version", "")),
            str(row.get("scenario", "")),
            str(row.get("output_root", "")),
        )
    )
    pending.sort(key=lambda row: (row.version, row.scenario))
    return pending, discovered_rows, skipped_preflight


def iter_localstaging_original_roots(localstaging_root: Path) -> List[Path]:
    roots: List[Path] = []
    for version in AUTOLOCALSTAGING_VERSIONS:
        version_dir = localstaging_root / version
        if not version_dir.exists():
            continue
        for scenario_dir in sorted(version_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            original_root = scenario_dir / ARX_NAME / RUNMODUS_NAME / AUTOLOCALSTAGING_ENSEMBLE / "original"
            if original_root.is_dir():
                roots.append(original_root)
    return roots


def cleanup_other_localstaging_contents(
    *,
    localstaging_root: Path,
    active_stage_root: Path,
) -> List[str]:
    removed: List[str] = []
    active_stage_root = active_stage_root.expanduser().resolve(strict=False)
    active_original_root = active_stage_root.parent

    for original_root in iter_localstaging_original_roots(localstaging_root):
        for child in sorted(original_root.iterdir()):
            child_resolved = child.expanduser().resolve(strict=False)
            if child_resolved == active_stage_root:
                continue
            if original_root.expanduser().resolve(strict=False) == active_original_root and child.name == active_stage_root.name:
                continue
            if child.is_symlink() or child.is_file():
                child.unlink(missing_ok=True)
                removed.append(str(child))
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                removed.append(str(child))
    return removed


def completion_marker_path(output_root: Path) -> Path:
    return output_root / RUN_COMPLETE_BASENAME


def write_completion_marker(
    *,
    task: Task,
    timetag: str,
    output_percentiles_file_count: int,
) -> Path:
    marker_path = completion_marker_path(task.output_root)
    payload = {
        "version": task.version,
        "scenario": task.scenario,
        "run_instance": task.run_instance,
        "source_run_root": str(task.source_run_root),
        "output_root": str(task.output_root),
        "timetag": timetag,
        "output_percentiles_file_count": int(output_percentiles_file_count),
        "completed_at_utc": utc_now().isoformat(),
    }
    write_json_atomic(marker_path, payload)
    return marker_path


def create_symlink_view(task: Task, views_root: Path) -> Path:
    scenario_ar6 = views_root / task.version / task.scenario / ARX_NAME
    ensure_clean_dir(scenario_ar6)
    for src in task.source_files:
        dst = scenario_ar6 / src.name
        os.symlink(src, dst)
    return views_root / task.version


def build_815_cmd(
    *,
    args: argparse.Namespace,
    input_root: Path,
    task: Task,
) -> List[str]:
    cmd: List[str] = [
        *_nice_prefix(),
        args.python,
        str(args.script_815),
        "--gcmagicc-dir",
        str(input_root),
        "--version-tag",
        task.version,
        "--output-root",
        str(task.output_root),
        "--output-layout",
        "scenario_run",
        "--timetag",
        args.timetag,
        "--strict-member-count",
        str(EXPECTED_MEMBER_COUNT),
        "--workers",
        str(max(1, int(args.workers_per_scenario))),
        "--min-pixels",
        "9",
        "--save-sig-digits",
        "5",
        "--log-level",
        "INFO",
        "--storage-access",
        STORAGE_ACCESS_MOUNT,
        "--variables",
        *args.variables,
        "--seasons",
        *args.seasons,
        "--regions",
        *args.regions,
        "--scenarios",
        task.scenario,
    ]
    if args.also_crunch_cmip6:
        cmd.append("--also-crunch-cmip6")
    else:
        cmd.append("--no-also-crunch-cmip6")
    if args.cmip6_reference_dir is not None:
        cmd.extend(["--cmip6-dir", str(args.cmip6_reference_dir)])
    if args.cmip6_stage_root is not None:
        cmd.extend(["--cmip6-stage-root", str(args.cmip6_stage_root)])
    if args.resume:
        cmd.append("--skip-existing-scenarios")
    if args.resume:
        cmd.append("--resume")
    return cmd


def build_815_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    py_path = Path(str(args.python)).expanduser().resolve(strict=False)
    candidates = [
        py_path.parents[1] / "share" / "proj" if len(py_path.parents) >= 2 else None,
        REPO_ROOT / ".pixi" / "envs" / "default" / "share" / "proj",
    ]
    proj_dir: Optional[Path] = None
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            proj_dir = candidate
            break
    if proj_dir is not None:
        proj_text = str(proj_dir)
        env["PROJ_LIB"] = proj_text
        env["PROJ_DATA"] = proj_text
        env["PYPROJ_DATADIR"] = proj_text
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["GCMAGICC_STORAGE_ACCESS"] = STORAGE_ACCESS_MOUNT
    env["GCMAGICC_ERA5SPLICED_LOCALSTAGING_ROOT"] = str(args.localstaging_root)
    env["GCMAGICC_ERA5SPLICED_LOCALRESULTS_ROOT"] = str(args.localresults_root)
    env["GCMAGICC_ERA5SPLICED_CMIP6_LOCALRESULTS_ROOT"] = str(args.cmip6_localresults_root)
    return env


def build_350_env(args: argparse.Namespace) -> Dict[str, str]:
    env = build_815_env(args)
    env["GCMAGICC_OBJECT_BUCKET"] = str(args.cmip6_upload_bucket)
    return env


def build_101_cmd(
    *,
    args: argparse.Namespace,
    auto_task: AutoTask,
    status_json: Path,
    manifest_json: Path,
) -> List[str]:
    return [
        *_nice_prefix(),
        args.python,
        str(DEFAULT_101_SCRIPT),
        "--version",
        auto_task.version,
        "--scenario",
        auto_task.scenario,
        "--ensemble",
        auto_task.ensemble,
        "--run-instance",
        auto_task.run_instance,
        "--stage-base",
        str(args.localstaging_root),
        "--workers",
        str(args.stage101_workers),
        "--status-json",
        str(status_json),
        "--manifest-json",
        str(manifest_json),
    ]


def build_350_cmd(
    *,
    args: argparse.Namespace,
    auto_task: AutoTask,
) -> List[str]:
    cmd: List[str] = [
        *_nice_prefix(),
        args.python,
        str(args.script_350),
        "--source-run-root",
        str(auto_task.stage_root),
        "--output-root",
        str(auto_task.cmip6_output_root),
        "--version",
        auto_task.version,
        "--scenario",
        auto_task.scenario,
        "--run-instance",
        auto_task.run_instance,
        "--workflow",
        ARX_NAME,
        "--runmodus",
        RUNMODUS_NAME,
        "--n-ensemble",
        AUTOLOCALSTAGING_ENSEMBLE,
        "--kind",
        "original",
        "--upload-bucket",
        str(args.cmip6_upload_bucket),
        "--upload-prefix",
        str(args.cmip6_upload_prefix),
        "--frontend-catalog",
        str(args.cmip6_frontend_catalog),
        "--member-workers",
        str(max(1, int(args.cmip6_member_workers))),
    ]
    if args.cmip6_skip_existing:
        cmd.append("--skip-if-complete")
    if not args.cleanup_local_after_upload:
        cmd.append("--no-cleanup-local-after-upload")
    return cmd


def build_percentiles_publish_cmd(
    *,
    args: argparse.Namespace,
    version: str,
    scenario: str,
    publish_timetag: str,
    out_path: Path,
    shards_dir: Path,
    observations_path: Path,
) -> List[str]:
    return [
        args.python,
        str(args.scenario_publish_script),
        "--source-root",
        str(args.localresults_root),
        "--source-mode",
        "gus_815_raw",
        "--versions",
        version,
        "--publish-specs",
        f"{version}:{scenario}",
        "--version-timetag",
        f"{version}={publish_timetag}",
        "--prefix",
        str(args.r2_prefix),
        "--out",
        str(out_path),
        "--shards-dir",
        str(shards_dir),
        "--observations-out",
        str(observations_path),
        "--default-version",
        str(args.scenario_default_version),
        "--bucket",
        str(args.r2_bucket),
        "--remote",
        str(args.r2_remote),
        "--hadcrut-url",
        str(args.scenario_obs_hadcrut_url),
        "--berkeley-url",
        str(args.scenario_obs_berkeley_url),
        "--norm-start",
        str(int(args.scenario_obs_norm_start)),
        "--norm-end",
        str(int(args.scenario_obs_norm_end)),
    ]


def _delete_local_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def publish_percentiles_output(
    *,
    args: argparse.Namespace,
    version: str,
    scenario: str,
    run_instance: str,
    output_root: Path,
    ledger_path: Path,
) -> PercentilesPublishResult:
    result = PercentilesPublishResult(
        version=version,
        scenario=scenario,
        run_instance=run_instance,
        status="quarantined",
    )
    if not output_root.exists():
        remote_manifest = rclone_cat_json(
            percentiles_remote_manifest_path(
                version=version,
                scenario=scenario,
                run_instance=run_instance,
                remote=str(args.r2_remote),
                bucket=str(args.r2_bucket),
            )
        )
        if remote_manifest:
            result.status = "completed"
            result.reason = "remote publish manifest already exists"
            result.remote_manifest_path = percentiles_remote_manifest_path(
                version=version,
                scenario=scenario,
                run_instance=run_instance,
                remote=str(args.r2_remote),
                bucket=str(args.r2_bucket),
            )
            result.publish_timetag = str(remote_manifest.get("publish_timetag") or "").strip() or None
            return result
        result.reason = f"Local percentiles output root missing: {output_root}"
        return result

    publish_timetag = derive_percentiles_publish_timetag(output_root)
    result.publish_timetag = publish_timetag
    remote_manifest_path = percentiles_remote_manifest_path(
        version=version,
        scenario=scenario,
        run_instance=run_instance,
        remote=str(args.r2_remote),
        bucket=str(args.r2_bucket),
    )
    result.remote_manifest_path = remote_manifest_path
    existing_remote_manifest = rclone_cat_json(remote_manifest_path)
    if remote_percentiles_manifest_matches(
        existing_remote_manifest,
        version=version,
        scenario=scenario,
        run_instance=run_instance,
        output_root=output_root,
        publish_timetag=publish_timetag,
        prefix_root=str(args.r2_prefix),
    ):
        local_manifest_path = percentiles_local_publish_manifest_path(output_root)
        write_publish_json_atomic(local_manifest_path, existing_remote_manifest)
        _delete_local_tree(output_root)
        result.status = "completed"
        result.reason = "remote publish manifest already exists"
        result.local_manifest_path = local_manifest_path
        result.local_deleted = True
        result.percentiles_file_count = int(existing_remote_manifest.get("percentiles_file_count") or 0)
        return result

    with tempfile.TemporaryDirectory(prefix="815publish_") as temp_dir:
        temp_root = Path(temp_dir)
        out_path = temp_root / "scenario_projection_catalog.json"
        shards_dir = temp_root / "scenario_projection_catalog_shards"
        observations_path = temp_root / "scenario_projection_observations.json"
        cmd = build_percentiles_publish_cmd(
            args=args,
            version=version,
            scenario=scenario,
            publish_timetag=publish_timetag,
            out_path=out_path,
            shards_dir=shards_dir,
            observations_path=observations_path,
        )
        append_jsonl(
            ledger_path,
            {
                "event": "percentiles_publish_started",
                "timestamp_utc": utc_now().isoformat(),
                "version": version,
                "scenario": scenario,
                "run_instance": run_instance,
                "output_root": str(output_root),
                "publish_timetag": publish_timetag,
                "remote_manifest_path": remote_manifest_path,
                "cmd": cmd,
            },
        )
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            cwd=str(args.scenario_publish_script.parent),
            env=os.environ.copy(),
        )
        if proc.returncode != 0:
            result.reason = (
                f"scenario projection publish failed (rc={proc.returncode}): "
                f"{(proc.stderr or proc.stdout or '').strip()[:500]}"
            )
            append_jsonl(
                ledger_path,
                {
                    "event": "percentiles_publish_failed",
                    "timestamp_utc": utc_now().isoformat(),
                    "version": version,
                    "scenario": scenario,
                    "run_instance": run_instance,
                    "output_root": str(output_root),
                    "returncode": proc.returncode,
                    "reason": result.reason,
                },
            )
            return result

        touched_shards = [
            str(path.relative_to(temp_root / "scenario_projection_catalog_shards").as_posix())
            for path in sorted(shards_dir.rglob("*.json"))
            if path.is_file()
        ]
        touched_shards = [
            f"scenario_projection_catalog_shards/{item}"
            for item in touched_shards
        ]
        manifest_payload = build_percentiles_publish_manifest_payload(
            output_root=output_root,
            version=version,
            scenario=scenario,
            run_instance=run_instance,
            publish_timetag=publish_timetag,
            prefix_root=str(args.r2_prefix),
            shard_paths_touched=touched_shards,
            remote=str(args.r2_remote),
            bucket=str(args.r2_bucket),
        )
        if not verify_percentiles_remote_listing(
            output_root=output_root,
            version=version,
            scenario=scenario,
            publish_timetag=publish_timetag,
            remote=str(args.r2_remote),
            bucket=str(args.r2_bucket),
            prefix_root=str(args.r2_prefix),
        ):
            result.reason = "remote percentiles verification failed after publish"
            append_jsonl(
                ledger_path,
                {
                    "event": "percentiles_publish_failed",
                    "timestamp_utc": utc_now().isoformat(),
                    "version": version,
                    "scenario": scenario,
                    "run_instance": run_instance,
                    "reason": result.reason,
                },
            )
            return result

        local_manifest_path = percentiles_local_publish_manifest_path(output_root)
        write_publish_json_atomic(local_manifest_path, manifest_payload)
        rclone_copyto_json(remote_path=remote_manifest_path, payload=manifest_payload)
        remote_manifest_payload = rclone_cat_json(remote_manifest_path)
        if not remote_percentiles_manifest_matches(
            remote_manifest_payload,
            version=version,
            scenario=scenario,
            run_instance=run_instance,
            output_root=output_root,
            publish_timetag=publish_timetag,
            prefix_root=str(args.r2_prefix),
        ):
            result.reason = "remote publish manifest did not verify after upload"
            append_jsonl(
                ledger_path,
                {
                    "event": "percentiles_publish_failed",
                    "timestamp_utc": utc_now().isoformat(),
                    "version": version,
                    "scenario": scenario,
                    "run_instance": run_instance,
                    "reason": result.reason,
                },
            )
            return result

        file_count = count_percentiles(output_root)
        _delete_local_tree(output_root)
        result.status = "completed"
        result.reason = "percentiles published to R2 and local copy deleted"
        result.local_manifest_path = local_manifest_path
        result.local_deleted = True
        result.percentiles_file_count = file_count
        append_jsonl(
            ledger_path,
            {
                "event": "percentiles_publish_completed",
                "timestamp_utc": utc_now().isoformat(),
                "version": version,
                "scenario": scenario,
                "run_instance": run_instance,
                "output_root": str(output_root),
                "remote_manifest_path": remote_manifest_path,
                "publish_timetag": publish_timetag,
                "percentiles_file_count": file_count,
            },
        )
        return result


def run_stage_101(
    *,
    args: argparse.Namespace,
    auto_task: AutoTask,
    log_path: Path,
    status_json: Path,
    manifest_json: Path,
    ledger_path: Path,
) -> TaskResult:
    result = TaskResult(
        version=auto_task.version,
        scenario=auto_task.scenario,
        status="quarantined",
        source_run_root=auto_task.source_run_root,
        output_root=auto_task.output_root,
        stage_log_path=log_path,
        stage_status_json=status_json,
        stage_manifest_json=manifest_json,
    )
    cmd = build_101_cmd(
        args=args,
        auto_task=auto_task,
        status_json=status_json,
        manifest_json=manifest_json,
    )
    started = utc_now()
    append_jsonl(
        ledger_path,
        {
            "event": "stage101_started",
            "timestamp_utc": started.isoformat(),
            "version": auto_task.version,
            "scenario": auto_task.scenario,
            "run_instance": auto_task.run_instance,
            "source_run_root": str(auto_task.source_run_root),
            "stage_root": str(auto_task.stage_root),
            "output_root": str(auto_task.output_root),
            "cmd": cmd,
            "status_json": str(status_json),
            "manifest_json": str(manifest_json),
            "log_path": str(log_path),
        },
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
    finished = utc_now()
    event = "stage101_completed" if proc.returncode == 0 else "stage101_failed"
    append_jsonl(
        ledger_path,
        {
            "event": event,
            "timestamp_utc": finished.isoformat(),
            "version": auto_task.version,
            "scenario": auto_task.scenario,
            "run_instance": auto_task.run_instance,
            "stage_root": str(auto_task.stage_root),
            "output_root": str(auto_task.output_root),
            "returncode": proc.returncode,
            "status_json": str(status_json),
            "manifest_json": str(manifest_json),
            "log_path": str(log_path),
            "duration_s": (finished - started).total_seconds(),
        },
    )
    if proc.returncode == 0:
        result.status = "completed"
        result.reason = "101 staging completed"
        return result

    result.reason = f"101 staging failed (rc={proc.returncode})"
    return result


def run_single_attempt(
    *,
    args: argparse.Namespace,
    task: Task,
    attempt: int,
    mode: str,
    input_root: Path,
    log_path: Path,
    ledger_path: Path,
) -> AttemptResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    task.output_root.mkdir(parents=True, exist_ok=True)
    marker_path = completion_marker_path(task.output_root)
    if marker_path.exists() and not args.resume:
        marker_path.unlink(missing_ok=True)

    outputs_before = count_percentiles(task.output_root)
    started = utc_now()
    cmd = build_815_cmd(args=args, input_root=input_root, task=task)
    env = build_815_env(args)
    append_jsonl(
        ledger_path,
        {
            "event": "attempt_start",
            "timestamp_utc": started.isoformat(),
            "version": task.version,
            "scenario": task.scenario,
            "attempt": attempt,
            "mode": mode,
            "cmd": cmd,
            "source_run_root": str(task.source_run_root),
            "input_root": str(input_root),
            "output_root": str(task.output_root),
            "outputs_before": outputs_before,
        },
    )

    with open(log_path, "a", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

    stall_seconds = max(1.0, float(args.stall_hours) * 3600.0)
    poll_seconds = max(5, int(args.poll_seconds))
    last_heartbeat = time.time()
    last_output_count = outputs_before
    last_ticks: Optional[int] = None
    process_state_last: Optional[str] = None
    errors_found: List[str] = []
    log_offset = 0
    status = "failed"
    rc: Optional[int] = None

    while True:
        now = time.time()
        rc = proc.poll()
        process_state_last, ticks = read_proc_state_and_ticks(proc.pid)
        if ticks is not None and ticks != last_ticks:
            last_heartbeat = now
            last_ticks = ticks

        outputs_now = count_percentiles(task.output_root)
        if outputs_now > last_output_count:
            last_heartbeat = now
            last_output_count = outputs_now

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(log_offset)
                chunk = f.read()
                log_offset = f.tell()
        except Exception:
            chunk = ""

        if chunk:
            last_heartbeat = now
            new_errors = _extract_new_error_lines(chunk)
            if new_errors:
                errors_found.extend(new_errors)
                append_jsonl(
                    ledger_path,
                    {
                        "event": "attempt_errors",
                        "timestamp_utc": utc_now().isoformat(),
                        "version": task.version,
                        "scenario": task.scenario,
                        "attempt": attempt,
                        "errors": new_errors[-20:],
                    },
                )

        if rc is not None:
            status = "success" if rc == 0 else "failed"
            break

        if now - last_heartbeat > stall_seconds:
            status = "stalled"
            _terminate_process(proc)
            rc = proc.returncode
            break

        append_jsonl(
            ledger_path,
            {
                "event": "attempt_poll",
                "timestamp_utc": utc_now().isoformat(),
                "version": task.version,
                "scenario": task.scenario,
                "attempt": attempt,
                "mode": mode,
                "pid": proc.pid,
                "process_state": process_state_last,
                "outputs": last_output_count,
            },
        )
        try:
            proc.wait(timeout=poll_seconds)
        except subprocess.TimeoutExpired:
            pass

    finished = utc_now()
    outputs_after = count_percentiles(task.output_root)
    note = ""
    if status == "stalled":
        note = (
            f"No heartbeat for {args.stall_hours:.2f}h "
            f"(state={process_state_last or '?'}, outputs={last_output_count})."
        )
    result = AttemptResult(
        attempt=attempt,
        mode=mode,
        status=status,
        returncode=rc,
        started_at=started,
        finished_at=finished,
        duration_s=(finished - started).total_seconds(),
        outputs_before=outputs_before,
        outputs_after=outputs_after,
        output_delta=outputs_after - outputs_before,
        process_state_last=process_state_last,
        errors=errors_found[-200:],
        log_path=log_path,
        note=note,
    )
    append_jsonl(
        ledger_path,
        {
            "event": "attempt_end",
            "timestamp_utc": finished.isoformat(),
            "version": task.version,
            "scenario": task.scenario,
            "attempt": attempt,
            "mode": mode,
            "status": status,
            "returncode": rc,
            "duration_s": result.duration_s,
            "outputs_before": outputs_before,
            "outputs_after": outputs_after,
            "output_delta": result.output_delta,
            "process_state_last": process_state_last,
            "note": note,
        },
    )
    return result


def run_task(
    *,
    args: argparse.Namespace,
    task: Task,
    views_root: Path,
    logs_dir: Path,
    ledger_path: Path,
) -> TaskResult:
    result = TaskResult(
        version=task.version,
        scenario=task.scenario,
        status="quarantined",
        source_run_root=task.source_run_root,
        output_root=task.output_root,
    )
    try:
        input_root = create_symlink_view(task, views_root)
    except Exception as exc:
        result.reason = f"Could not build local symlink view: {exc}"
        return result

    log_path = logs_dir / f"{task.version}__{task.scenario}__attempt1_local_symlink.log"
    attempt = run_single_attempt(
        args=args,
        task=task,
        attempt=1,
        mode="local_symlink_view",
        input_root=input_root,
        log_path=log_path,
        ledger_path=ledger_path,
    )
    result.attempts.append(attempt)
    if attempt.status == "success":
        result.status = "completed"
        result.completion_marker_path = write_completion_marker(
            task=task,
            timetag=args.timetag,
            output_percentiles_file_count=count_percentiles(task.output_root),
        )
        return result

    result.reason = f"Attempt 1 ended as {attempt.status} (rc={attempt.returncode})"
    return result


def print_task_overview(tasks: Sequence[Task], skipped: Sequence[Dict[str, Any]]) -> None:
    print(f"[815DRV] Tasks resolved: {len(tasks)}")
    for task in tasks:
        print(
            f"[815DRV]   {task.version}/{task.scenario} | members={len(task.source_files)} | "
            f"source={task.source_run_root}"
        )
    if skipped:
        print(f"[815DRV] Skipped at preflight: {len(skipped)}")
        for row in skipped:
            print(f"[815DRV]   {row.get('version')}/{row.get('scenario')} | reason={row.get('reason')}")


def print_auto_overview(discovered_rows: Sequence[Dict[str, Any]], pending: Sequence[AutoTask]) -> None:
    print(f"[815DRV] Auto tasks discovered: {len(discovered_rows)}")
    for row in discovered_rows:
        print(
            f"[815DRV]   {row.get('version')}/{row.get('scenario')} | "
            f"{row.get('status')} | source={row.get('source_run_root')}"
        )
    if pending:
        print(f"[815DRV] Auto tasks pending stage/export/cmip6: {len(pending)}")


def _task_row(task: Task) -> Dict[str, Any]:
    return {
        "version": task.version,
        "scenario": task.scenario,
        "run_instance": task.run_instance,
        "source_run_root": str(task.source_run_root),
        "output_root": str(task.output_root),
        "member_count": len(task.source_files),
        "member_identities": task.member_identities,
    }


def _result_row(result: TaskResult) -> Dict[str, Any]:
    return {
        "version": result.version,
        "scenario": result.scenario,
        "status": result.status,
        "reason": result.reason,
        "source_run_root": str(result.source_run_root),
        "output_root": str(result.output_root),
        "cmip6_output_root": str(result.cmip6_output_root) if result.cmip6_output_root else None,
        "percentiles_state": result.percentiles_state,
        "percentiles_publish_status": result.percentiles_publish_status,
        "percentiles_publish_reason": result.percentiles_publish_reason,
        "percentiles_publish_timetag": result.percentiles_publish_timetag,
        "percentiles_local_publish_manifest_path": (
            str(result.percentiles_local_publish_manifest_path)
            if result.percentiles_local_publish_manifest_path
            else None
        ),
        "percentiles_remote_publish_manifest_path": result.percentiles_remote_publish_manifest_path,
        "stage_status_json": str(result.stage_status_json) if result.stage_status_json else None,
        "stage_manifest_json": str(result.stage_manifest_json) if result.stage_manifest_json else None,
        "stage_log_path": str(result.stage_log_path) if result.stage_log_path else None,
        "completion_marker_path": str(result.completion_marker_path) if result.completion_marker_path else None,
        "cmip6_log_path": str(result.cmip6_log_path) if result.cmip6_log_path else None,
        "cmip6_manifest_path": str(result.cmip6_manifest_path) if result.cmip6_manifest_path else None,
        "cmip6_upload_manifest_path": (
            str(result.cmip6_upload_manifest_path) if result.cmip6_upload_manifest_path else None
        ),
        "cmip6_local_cleanup_manifest_path": (
            str(result.cmip6_local_cleanup_manifest_path) if result.cmip6_local_cleanup_manifest_path else None
        ),
        "cmip6_completion_marker_path": (
            str(result.cmip6_completion_marker_path) if result.cmip6_completion_marker_path else None
        ),
        "completion_state": result.completion_state,
        "attempts": [
            {
                "attempt": attempt.attempt,
                "mode": attempt.mode,
                "status": attempt.status,
                "returncode": attempt.returncode,
                "started_at": attempt.started_at.isoformat(),
                "finished_at": attempt.finished_at.isoformat(),
                "duration_s": attempt.duration_s,
                "outputs_before": attempt.outputs_before,
                "outputs_after": attempt.outputs_after,
                "output_delta": attempt.output_delta,
                "process_state_last": attempt.process_state_last,
                "errors": attempt.errors,
                "log_path": str(attempt.log_path) if attempt.log_path else None,
                "note": attempt.note,
            }
            for attempt in result.attempts
        ],
    }


def _stage_artifact_paths(meta_root: Path, auto_task: AutoTask) -> Tuple[Path, Path, Path]:
    stage_dir = meta_root / "stage101"
    base = f"{auto_task.version}__{auto_task.scenario}__{auto_task.run_instance}"
    return (
        stage_dir / f"{base}.log",
        stage_dir / f"{base}__status.json",
        stage_dir / f"{base}__manifest.json",
    )


def _cmip6_log_path(meta_root: Path, auto_task: AutoTask) -> Path:
    cmip6_dir = meta_root / "cmip6_350"
    base = f"{auto_task.version}__{auto_task.scenario}__{auto_task.run_instance}"
    return cmip6_dir / f"{base}.log"


def run_cmip6_350(
    *,
    args: argparse.Namespace,
    auto_task: AutoTask,
    log_path: Path,
    ledger_path: Path,
) -> TaskResult:
    result = TaskResult(
        version=auto_task.version,
        scenario=auto_task.scenario,
        status="quarantined",
        source_run_root=auto_task.source_run_root,
        output_root=auto_task.output_root,
        cmip6_output_root=auto_task.cmip6_output_root,
        cmip6_log_path=log_path,
        cmip6_manifest_path=auto_task.cmip6_output_root / CMIP6_META_SUBDIR / RUN_MANIFEST_BASENAME,
        cmip6_upload_manifest_path=auto_task.cmip6_output_root / CMIP6_META_SUBDIR / UPLOAD_MANIFEST_BASENAME,
        cmip6_local_cleanup_manifest_path=auto_task.cmip6_output_root / CMIP6_META_SUBDIR / LOCAL_CLEANUP_BASENAME,
        cmip6_completion_marker_path=completion_marker_path(auto_task.cmip6_output_root),
    )
    cmd = build_350_cmd(args=args, auto_task=auto_task)
    started = utc_now()
    append_jsonl(
        ledger_path,
        {
            "event": "cmip6_started",
            "timestamp_utc": started.isoformat(),
            "version": auto_task.version,
            "scenario": auto_task.scenario,
            "run_instance": auto_task.run_instance,
            "source_run_root": str(auto_task.source_run_root),
            "stage_root": str(auto_task.stage_root),
            "cmip6_output_root": str(auto_task.cmip6_output_root),
            "cmd": cmd,
            "log_path": str(log_path),
        },
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            env=build_350_env(args),
        )
    finished = utc_now()
    status = "completed" if proc.returncode == 0 else "quarantined"
    reason = "350 CMIP6 conversion completed" if proc.returncode == 0 else f"350 CMIP6 conversion failed (rc={proc.returncode})"
    result.status = status
    result.reason = reason
    append_jsonl(
        ledger_path,
        {
            "event": "cmip6_completed" if proc.returncode == 0 else "cmip6_failed",
            "timestamp_utc": finished.isoformat(),
            "version": auto_task.version,
            "scenario": auto_task.scenario,
            "run_instance": auto_task.run_instance,
            "cmip6_output_root": str(auto_task.cmip6_output_root),
            "returncode": proc.returncode,
            "log_path": str(log_path),
            "duration_s": (finished - started).total_seconds(),
            "cmip6_manifest_path": str(result.cmip6_manifest_path),
            "cmip6_upload_manifest_path": str(result.cmip6_upload_manifest_path),
            "cmip6_local_cleanup_manifest_path": str(result.cmip6_local_cleanup_manifest_path),
            "cmip6_completion_marker_path": str(result.cmip6_completion_marker_path),
        },
    )
    return result


def run_auto_task(
    *,
    args: argparse.Namespace,
    auto_task: AutoTask,
    meta_root: Path,
    views_root: Path,
    logs_dir: Path,
    ledger_path: Path,
) -> TaskResult:
    result = TaskResult(
        version=auto_task.version,
        scenario=auto_task.scenario,
        status="quarantined",
        source_run_root=auto_task.source_run_root,
        output_root=auto_task.output_root,
        cmip6_output_root=auto_task.cmip6_output_root,
        completion_state=auto_task.completion_state,
        percentiles_state=auto_task.percentiles_state,
    )
    needs_815_compute = auto_task.percentiles_state in {"missing", "partial_local"}
    needs_cmip6 = args.run_cmip6 and auto_task.cmip6_state in {"missing", "partial_local", "partial_remote"}
    needs_percentiles_publish = bool(args.publish_percentiles) and auto_task.percentiles_state in {"done_local_only"}

    if not needs_815_compute and not needs_cmip6:
        result.status = "completed"
        result.completion_state = "done"
        result.completion_marker_path = completion_marker_path(auto_task.output_root) if auto_task.output_root.exists() else None
        result.percentiles_publish_status = "pending" if needs_percentiles_publish else "skipped"
        result.percentiles_publish_reason = (
            "remote publish pending from local 815 output"
            if needs_percentiles_publish
            else auto_task.percentiles_publish_source or auto_task.percentiles_compute_source or ""
        )
        result.percentiles_publish_timetag = auto_task.percentiles_publish_timetag
        result.percentiles_remote_publish_manifest_path = percentiles_remote_manifest_path(
            version=auto_task.version,
            scenario=auto_task.scenario,
            run_instance=auto_task.run_instance,
            remote=str(args.r2_remote),
            bucket=str(args.r2_bucket),
        )
        result.reason = (
            "percentiles publish pending"
            if needs_percentiles_publish
            else "nothing to recompute"
        )
        return result

    append_jsonl(
        ledger_path,
        {
            "event": "cleanup_started",
            "timestamp_utc": utc_now().isoformat(),
            "version": auto_task.version,
            "scenario": auto_task.scenario,
            "run_instance": auto_task.run_instance,
            "stage_root": str(auto_task.stage_root),
            "output_root": str(auto_task.output_root),
            "cmip6_output_root": str(auto_task.cmip6_output_root),
        },
    )
    removed = cleanup_other_localstaging_contents(
        localstaging_root=args.localstaging_root,
        active_stage_root=auto_task.stage_root,
    )
    append_jsonl(
        ledger_path,
        {
            "event": "cleanup_finished",
            "timestamp_utc": utc_now().isoformat(),
            "version": auto_task.version,
            "scenario": auto_task.scenario,
            "run_instance": auto_task.run_instance,
            "stage_root": str(auto_task.stage_root),
            "output_root": str(auto_task.output_root),
            "cmip6_output_root": str(auto_task.cmip6_output_root),
            "removed_count": len(removed),
            "removed_sample": removed[:25],
        },
    )

    stage_log_path, stage_status_json, stage_manifest_json = _stage_artifact_paths(meta_root, auto_task)
    stage_result = run_stage_101(
        args=args,
        auto_task=auto_task,
        log_path=stage_log_path,
        status_json=stage_status_json,
        manifest_json=stage_manifest_json,
        ledger_path=ledger_path,
    )
    result.stage_log_path = stage_log_path
    result.stage_status_json = stage_status_json
    result.stage_manifest_json = stage_manifest_json
    if stage_result.status != "completed":
        result.reason = stage_result.reason
        return result

    task: Optional[Task] = None
    if needs_815_compute:
        if auto_task.percentiles_state == "partial_local":
            _delete_local_tree(auto_task.output_root)
        task = build_task_from_run_root(
            version=auto_task.version,
            scenario=auto_task.scenario,
            source_run_root=auto_task.stage_root,
            localresults_root=args.localresults_root,
        )
        if str(task.output_root.expanduser().resolve(strict=False)) != str(auto_task.output_root.expanduser().resolve(strict=False)):
            raise RuntimeError(
                f"Resolved staged output root mismatch: expected {auto_task.output_root}, got {task.output_root}"
            )

    if needs_cmip6 and auto_task.cmip6_state == "partial_remote":
        append_jsonl(
            ledger_path,
            {
                "event": "cmip6_partial_remote_resume",
                "timestamp_utc": utc_now().isoformat(),
                "version": auto_task.version,
                "scenario": auto_task.scenario,
                "run_instance": auto_task.run_instance,
                "remote_prefix": build_cmip6_remote_run_prefix(
                    upload_prefix=args.cmip6_upload_prefix,
                    version=auto_task.version,
                    scenario=auto_task.scenario,
                    run_instance=auto_task.run_instance,
                ),
                "note": "Keeping partial remote prefix for resumable 350 retries.",
            },
        )

    export_status = "skipped_existing" if not needs_815_compute else "pending"
    cmip6_status = "disabled"
    pending_publish_after_compute = needs_percentiles_publish
    export_result: Optional[TaskResult] = None
    cmip6_result: Optional[TaskResult] = None

    futures: Dict[Future[TaskResult], str] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        if needs_815_compute and task is not None:
            append_jsonl(
                ledger_path,
                {
                    "event": "export815_started",
                    "timestamp_utc": utc_now().isoformat(),
                    "version": task.version,
                    "scenario": task.scenario,
                    "run_instance": task.run_instance,
                    "source_run_root": str(task.source_run_root),
                    "output_root": str(task.output_root),
                },
            )
            futures[executor.submit(
                run_task,
                args=args,
                task=task,
                views_root=views_root,
                logs_dir=logs_dir,
                ledger_path=ledger_path,
            )] = "815"
        elif auto_task.percentiles_compute_source:
            result.completion_marker_path = completion_marker_path(auto_task.output_root) if auto_task.output_root.exists() else None
            append_jsonl(
                ledger_path,
                {
                    "event": "export815_skipped_existing",
                    "timestamp_utc": utc_now().isoformat(),
                    "version": auto_task.version,
                    "scenario": auto_task.scenario,
                    "run_instance": auto_task.run_instance,
                    "source_run_root": str(auto_task.source_run_root),
                    "output_root": str(auto_task.output_root),
                    "completion_source": auto_task.percentiles_compute_source,
                },
            )

        if args.run_cmip6:
            if needs_cmip6:
                cmip6_log_path = _cmip6_log_path(meta_root, auto_task)
                futures[executor.submit(
                    run_cmip6_350,
                    args=args,
                    auto_task=auto_task,
                    log_path=cmip6_log_path,
                    ledger_path=ledger_path,
                )] = "cmip6"
            elif auto_task.cmip6_completion_source:
                result.cmip6_manifest_path = auto_task.cmip6_output_root / CMIP6_META_SUBDIR / RUN_MANIFEST_BASENAME
                result.cmip6_upload_manifest_path = auto_task.cmip6_output_root / CMIP6_META_SUBDIR / UPLOAD_MANIFEST_BASENAME
                result.cmip6_local_cleanup_manifest_path = (
                    auto_task.cmip6_output_root / CMIP6_META_SUBDIR / LOCAL_CLEANUP_BASENAME
                )
                result.cmip6_completion_marker_path = completion_marker_path(auto_task.cmip6_output_root)
                append_jsonl(
                    ledger_path,
                    {
                        "event": "cmip6_skipped_existing",
                        "timestamp_utc": utc_now().isoformat(),
                        "version": auto_task.version,
                        "scenario": auto_task.scenario,
                        "run_instance": auto_task.run_instance,
                        "cmip6_output_root": str(auto_task.cmip6_output_root),
                        "completion_source": auto_task.cmip6_completion_source,
                    },
                )
                cmip6_status = "skipped_existing"

        for future in as_completed(list(futures.keys())):
            kind = futures[future]
            job_result = future.result()
            if kind == "815":
                export_result = job_result
                result.attempts = export_result.attempts
                result.completion_marker_path = export_result.completion_marker_path
                export_status = export_result.status
                append_jsonl(
                    ledger_path,
                    {
                        "event": "export815_completed" if export_result.status == "completed" else "export815_failed",
                        "timestamp_utc": utc_now().isoformat(),
                        "version": export_result.version,
                        "scenario": export_result.scenario,
                        "run_instance": task.run_instance if task is not None else auto_task.run_instance,
                        "source_run_root": str(export_result.source_run_root),
                        "output_root": str(export_result.output_root),
                        "reason": export_result.reason,
                        "completion_marker_path": (
                            str(export_result.completion_marker_path) if export_result.completion_marker_path else None
                        ),
                    },
                )
                if export_result.status == "completed":
                    pending_publish_after_compute = bool(args.publish_percentiles)
                else:
                    result.reason = export_result.reason or "815 export failed"
            else:
                cmip6_result = job_result
                result.cmip6_log_path = cmip6_result.cmip6_log_path
                result.cmip6_manifest_path = cmip6_result.cmip6_manifest_path
                result.cmip6_upload_manifest_path = cmip6_result.cmip6_upload_manifest_path
                result.cmip6_local_cleanup_manifest_path = cmip6_result.cmip6_local_cleanup_manifest_path
                result.cmip6_completion_marker_path = cmip6_result.cmip6_completion_marker_path
                cmip6_status = cmip6_result.status
                if cmip6_result.status != "completed":
                    result.reason = cmip6_result.reason or "CMIP6 conversion failed"

    if export_result is not None and export_result.status != "completed":
        return result
    if cmip6_result is not None and cmip6_result.status != "completed":
        return result

    result.percentiles_publish_timetag = (
        derive_percentiles_publish_timetag(auto_task.output_root) if auto_task.output_root.exists() else auto_task.percentiles_publish_timetag
    )
    result.percentiles_remote_publish_manifest_path = percentiles_remote_manifest_path(
        version=auto_task.version,
        scenario=auto_task.scenario,
        run_instance=auto_task.run_instance,
        remote=str(args.r2_remote),
        bucket=str(args.r2_bucket),
    )
    result.percentiles_publish_status = "pending" if pending_publish_after_compute and args.publish_percentiles else "skipped"
    result.percentiles_publish_reason = (
        "queued for background publish"
        if result.percentiles_publish_status == "pending"
        else auto_task.percentiles_publish_source or ""
    )

    result.status = "completed"
    result.completion_state = "done"
    result.reason = (
        f"815={export_status}; cmip6={cmip6_status}; percentiles_publish={result.percentiles_publish_status}"
        if args.run_cmip6
        else f"815={export_status}; cmip6=disabled; percentiles_publish={result.percentiles_publish_status}"
    )
    return result


def apply_percentiles_publish_result(task_result: TaskResult, publish_result: PercentilesPublishResult) -> None:
    task_result.percentiles_publish_status = publish_result.status
    task_result.percentiles_publish_reason = publish_result.reason
    task_result.percentiles_publish_timetag = publish_result.publish_timetag
    task_result.percentiles_local_publish_manifest_path = publish_result.local_manifest_path
    task_result.percentiles_remote_publish_manifest_path = publish_result.remote_manifest_path
    if publish_result.status == "completed":
        task_result.percentiles_state = "done_remote_only" if publish_result.local_deleted else "done_both"
    else:
        task_result.status = "quarantined"
        task_result.reason = (
            f"{task_result.reason}; percentiles publish failed: {publish_result.reason}"
            if task_result.reason
            else f"percentiles publish failed: {publish_result.reason}"
        )


def drain_publish_jobs(
    *,
    publish_jobs: List[Tuple[Future[PercentilesPublishResult], TaskResult]],
    wait_for_all: bool,
) -> bool:
    had_failure = False
    remaining: List[Tuple[Future[PercentilesPublishResult], TaskResult]] = []
    for future, task_result in publish_jobs:
        if not wait_for_all and not future.done():
            remaining.append((future, task_result))
            continue
        try:
            publish_result = future.result()
        except Exception as exc:
            publish_result = PercentilesPublishResult(
                version=task_result.version,
                scenario=task_result.scenario,
                run_instance=task_result.output_root.name,
                status="quarantined",
                reason=str(exc),
            )
        apply_percentiles_publish_result(task_result, publish_result)
        if publish_result.status != "completed":
            had_failure = True
    publish_jobs[:] = remaining
    return had_failure


def iter_local_percentiles_run_roots(localresults_root: Path) -> List[Tuple[str, str, str, Path]]:
    rows: List[Tuple[str, str, str, Path]] = []
    for marker in sorted(localresults_root.rglob(RUN_COMPLETE_BASENAME)):
        if DEFAULT_META_SUBDIR in marker.parts:
            continue
        try:
            rel = marker.parent.relative_to(localresults_root)
        except ValueError:
            continue
        if len(rel.parts) != 8:
            continue
        version, scenario, arx, runmodus, ensemble, kind, publish_kind, run_instance = rel.parts
        if (
            arx != ARX_NAME
            or runmodus != RUNMODUS_NAME
            or ensemble != AUTOLOCALSTAGING_ENSEMBLE
            or kind != "dataderivatives"
            or publish_kind != "815_scenpercentiles"
        ):
            continue
        rows.append((version, scenario, run_instance, marker.parent))
    return rows


def iter_local_cmip6_run_roots(cmip6_localresults_root: Path) -> List[Tuple[str, str, str, Path]]:
    roots: Set[Path] = set()
    for path in sorted(cmip6_localresults_root.rglob(RUN_MANIFEST_BASENAME)):
        if path.parent.name != CMIP6_META_SUBDIR:
            continue
        roots.add(path.parent.parent)
    rows: List[Tuple[str, str, str, Path]] = []
    for root in sorted(roots):
        try:
            rel = root.relative_to(cmip6_localresults_root)
        except ValueError:
            continue
        if len(rel.parts) != 7:
            continue
        version, scenario, arx, runmodus, ensemble, kind, run_instance = rel.parts
        if (
            arx != ARX_NAME
            or runmodus != RUNMODUS_NAME
            or ensemble != AUTOLOCALSTAGING_ENSEMBLE
            or kind != "original"
        ):
            continue
        rows.append((version, scenario, run_instance, root))
    return rows


def cleanup_local_cmip6_payload_for_backlog(output_root: Path) -> Dict[str, Any]:
    preserved = {
        RUN_COMPLETE_BASENAME,
        f"{CMIP6_META_SUBDIR}/{RUN_MANIFEST_BASENAME}",
        f"{CMIP6_META_SUBDIR}/{UPLOAD_MANIFEST_BASENAME}",
        f"{CMIP6_META_SUBDIR}/{LOCAL_CLEANUP_BASENAME}",
    }
    deleted_files = 0
    deleted_bytes = 0
    for path in sorted(output_root.rglob("*"), key=lambda p: (p.is_dir(), len(p.parts), str(p)), reverse=True):
        rel = path.relative_to(output_root).as_posix()
        if rel in preserved:
            continue
        if path.is_file():
            try:
                deleted_bytes += int(path.stat().st_size)
            except OSError:
                pass
            path.unlink(missing_ok=True)
            deleted_files += 1
    for path in sorted((p for p in output_root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if path == output_root or path == output_root / CMIP6_META_SUBDIR:
            continue
        try:
            path.rmdir()
        except OSError:
            continue
    cleanup_manifest = output_root / CMIP6_META_SUBDIR / LOCAL_CLEANUP_BASENAME
    payload = load_json_file(cleanup_manifest)
    payload["generated_at_utc"] = utc_now().isoformat()
    payload["deleted_file_count"] = int(payload.get("deleted_file_count") or 0) + deleted_files
    payload["deleted_bytes"] = int(payload.get("deleted_bytes") or 0) + deleted_bytes
    payload["status"] = "completed"
    write_json_atomic(cleanup_manifest, payload)
    return payload


def run_backlog_reclaim(
    *,
    args: argparse.Namespace,
    ledger_path: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if args.publish_percentiles:
        for version, scenario, run_instance, output_root in iter_local_percentiles_run_roots(args.localresults_root):
            state = percentiles_state_info(
                args=args,
                output_root=output_root,
                version=version,
                scenario=scenario,
                run_instance=run_instance,
            )
            row: Dict[str, Any] = {
                "kind": "percentiles",
                "version": version,
                "scenario": scenario,
                "run_instance": run_instance,
                "output_root": str(output_root),
                "state": state["status"],
            }
            if state["status"] == "done_remote_only":
                _delete_local_tree(output_root)
                row["action"] = "deleted_local_copy"
            elif state["status"] == "done_both":
                _delete_local_tree(output_root)
                row["action"] = "deleted_local_copy"
            elif state["status"] == "done_local_only":
                legacy_payload = legacy_percentiles_remote_complete(
                    output_root=output_root,
                    version=version,
                    scenario=scenario,
                    run_instance=run_instance,
                    remote=str(args.r2_remote),
                    bucket=str(args.r2_bucket),
                    prefix_root=str(args.r2_prefix),
                )
                if legacy_payload:
                    local_manifest_path = percentiles_local_publish_manifest_path(output_root)
                    write_publish_json_atomic(local_manifest_path, legacy_payload)
                    rclone_copyto_json(
                        remote_path=percentiles_remote_manifest_path(
                            version=version,
                            scenario=scenario,
                            run_instance=run_instance,
                            remote=str(args.r2_remote),
                            bucket=str(args.r2_bucket),
                        ),
                        payload=legacy_payload,
                    )
                    _delete_local_tree(output_root)
                    row["action"] = "backfilled_remote_manifest_and_deleted_local"
                else:
                    publish_result = publish_percentiles_output(
                        args=args,
                        version=version,
                        scenario=scenario,
                        run_instance=run_instance,
                        output_root=output_root,
                        ledger_path=ledger_path,
                    )
                    row["action"] = publish_result.status
                    row["reason"] = publish_result.reason
            else:
                row["action"] = "skipped"
            rows.append(row)

    for version, scenario, run_instance, output_root in iter_local_cmip6_run_roots(args.cmip6_localresults_root):
        state = cmip6_state_info(
            args=args,
            output_root=output_root,
            version=version,
            scenario=scenario,
            run_instance=run_instance,
        )
        row = {
            "kind": "cmip6",
            "version": version,
            "scenario": scenario,
            "run_instance": run_instance,
            "output_root": str(output_root),
            "state": state["status"],
        }
        if state["status"] in {"done_both", "done_remote_only"} and output_root.exists():
            cleanup_payload = cleanup_local_cmip6_payload_for_backlog(output_root)
            row["action"] = "cleaned_local_payload"
            row["deleted_file_count"] = int(cleanup_payload.get("deleted_file_count") or 0)
            row["deleted_bytes"] = int(cleanup_payload.get("deleted_bytes") or 0)
        else:
            row["action"] = "left_in_place"
        rows.append(row)

    return rows


def main() -> None:
    args = parse_args()
    if not args.script_815.exists():
        raise FileNotFoundError(f"815 script not found: {args.script_815}")
    if args.auto_localstaging and not DEFAULT_101_SCRIPT.exists():
        raise FileNotFoundError(f"101 script not found: {DEFAULT_101_SCRIPT}")
    if args.run_cmip6 and not args.script_350.exists():
        raise FileNotFoundError(f"350 script not found: {args.script_350}")
    if args.publish_percentiles and not args.scenario_publish_script.exists():
        raise FileNotFoundError(f"Scenario publish script not found: {args.scenario_publish_script}")

    args.localstaging_root = args.localstaging_root.expanduser().resolve(strict=False)
    args.localresults_root = args.localresults_root.expanduser().resolve(strict=False)
    args.cmip6_localresults_root = args.cmip6_localresults_root.expanduser().resolve(strict=False)
    args.source_root = args.source_root.expanduser().resolve(strict=False)
    args.script_815 = args.script_815.expanduser().resolve(strict=False)
    args.script_350 = args.script_350.expanduser().resolve(strict=False)
    if args.cmip6_reference_dir is not None:
        args.cmip6_reference_dir = args.cmip6_reference_dir.expanduser().resolve(strict=False)
    if args.cmip6_stage_root is not None:
        args.cmip6_stage_root = args.cmip6_stage_root.expanduser().resolve(strict=False)
    args.cmip6_frontend_catalog = args.cmip6_frontend_catalog.expanduser().resolve(strict=False)
    args.scenario_publish_script = args.scenario_publish_script.expanduser().resolve(strict=False)

    meta_root = args.localresults_root / DEFAULT_META_SUBDIR / f"815_localstaging_{args.timetag}"
    views_root = meta_root / "views"
    logs_dir = meta_root / "logs"
    ledger_path = meta_root / "run_ledger.jsonl"
    manifest_path = meta_root / "final_manifest.json"
    meta_root.mkdir(parents=True, exist_ok=True)
    existing_manifest = load_json_file(manifest_path)

    tmp_stage_summary = summarize_tmp_stage_dirs()
    if tmp_stage_summary:
        total_tmp_stage_bytes = sum(int(row["bytes"]) for row in tmp_stage_summary)
        print(
            "[815DRV] Warning: found existing /tmp/815_stage* directories; "
            f"they are not used by this driver. Total size={_format_bytes(total_tmp_stage_bytes)}"
        )

    discovered_rows: List[Dict[str, Any]] = []
    skipped_preflight: List[Dict[str, Any]] = []
    planned_task_rows: List[Dict[str, Any]] = []
    results: List[TaskResult] = []
    backlog_rows: List[Dict[str, Any]] = []
    stopped_on_failure = False
    publisher_executor: Optional[ThreadPoolExecutor] = None
    publish_jobs: List[Tuple[Future[PercentilesPublishResult], TaskResult]] = []
    if args.publish_percentiles and not args.dry_run and not args.backlog_only:
        publisher_executor = ThreadPoolExecutor(max_workers=1)

    if args.backlog_only:
        backlog_rows = run_backlog_reclaim(args=args, ledger_path=ledger_path)
        for row in backlog_rows:
            append_jsonl(
                ledger_path,
                {
                    "event": "backlog_action",
                    "timestamp_utc": utc_now().isoformat(),
                    **row,
                },
            )
        write_json_atomic(
            manifest_path,
            {
                "generated_at": utc_now().isoformat(),
                "timetag": args.timetag,
                "backlog_only": True,
                "localresults_root": str(args.localresults_root),
                "cmip6_localresults_root": str(args.cmip6_localresults_root),
                "meta_root": str(meta_root),
                "backlog_actions": backlog_rows,
                "summary": {
                    "backlog_actions": len(backlog_rows),
                    "backlog_percentiles": sum(1 for row in backlog_rows if row.get("kind") == "percentiles"),
                    "backlog_cmip6": sum(1 for row in backlog_rows if row.get("kind") == "cmip6"),
                },
            },
        )
        print(f"[815DRV] Backlog reclaim complete. Manifest: {manifest_path}")
        return

    if args.auto_localstaging:
        completed_index = build_completed_output_index(args.localresults_root)
        pending_auto_tasks, discovered_rows, skipped_preflight = build_auto_tasks(args, completed_index)
        print_auto_overview(discovered_rows, pending_auto_tasks)

        for row in discovered_rows:
            append_jsonl(
                ledger_path,
                {
                    "event": "discovered",
                    "timestamp_utc": utc_now().isoformat(),
                    **row,
                },
            )
            if row.get("status") == "done":
                append_jsonl(
                    ledger_path,
                    {
                        "event": "skipped_existing_outputs",
                        "timestamp_utc": utc_now().isoformat(),
                        **row,
                    },
                )

        planned_task_rows = [
            {
                "version": task.version,
                "scenario": task.scenario,
                "ensemble": task.ensemble,
                "run_instance": task.run_instance,
                "source_run_root": str(task.source_run_root),
                "stage_root": str(task.stage_root),
                "output_root": str(task.output_root),
                "cmip6_output_root": str(task.cmip6_output_root),
                "member_count": task.member_count,
                "member_identities": task.member_identities,
                "completion_state": task.completion_state,
                "percentiles_state": task.percentiles_state,
                "percentiles_compute_source": task.percentiles_compute_source,
                "percentiles_publish_source": task.percentiles_publish_source,
                "percentiles_publish_timetag": task.percentiles_publish_timetag,
                "cmip6_completion_source": task.cmip6_completion_source,
                "cmip6_state": task.cmip6_state,
            }
            for task in pending_auto_tasks
        ]

        if not args.dry_run:
            for idx, auto_task in enumerate(pending_auto_tasks, start=1):
                print(f"[815DRV] ({idx}/{len(pending_auto_tasks)}) Auto-running {auto_task.version}/{auto_task.scenario}")
                append_jsonl(
                    ledger_path,
                    {
                        "event": "task_start",
                        "timestamp_utc": utc_now().isoformat(),
                        "version": auto_task.version,
                        "scenario": auto_task.scenario,
                        "source_run_root": str(auto_task.source_run_root),
                        "stage_root": str(auto_task.stage_root),
                        "output_root": str(auto_task.output_root),
                        "cmip6_output_root": str(auto_task.cmip6_output_root),
                        "completion_state": auto_task.completion_state,
                    },
                )
                try:
                    task_result = run_auto_task(
                        args=args,
                        auto_task=auto_task,
                        meta_root=meta_root,
                        views_root=views_root,
                        logs_dir=logs_dir,
                        ledger_path=ledger_path,
                    )
                except Exception as exc:
                    task_result = TaskResult(
                        version=auto_task.version,
                        scenario=auto_task.scenario,
                        status="quarantined",
                        source_run_root=auto_task.source_run_root,
                        output_root=auto_task.output_root,
                        reason=str(exc),
                    )
                results.append(task_result)
                append_jsonl(
                    ledger_path,
                    {
                        "event": "task_end",
                        "timestamp_utc": utc_now().isoformat(),
                        "version": task_result.version,
                        "scenario": task_result.scenario,
                        "output_root": str(task_result.output_root),
                        "cmip6_output_root": str(task_result.cmip6_output_root) if task_result.cmip6_output_root else None,
                        "status": task_result.status,
                        "reason": task_result.reason,
                        "completion_state": task_result.completion_state,
                    },
                )
                print(f"[815DRV]   -> {task_result.status} ({task_result.reason or 'ok'})")
                if (
                    publisher_executor is not None
                    and task_result.status == "completed"
                    and task_result.percentiles_publish_status == "pending"
                ):
                    append_jsonl(
                        ledger_path,
                        {
                            "event": "percentiles_publish_enqueued",
                            "timestamp_utc": utc_now().isoformat(),
                            "version": task_result.version,
                            "scenario": task_result.scenario,
                            "run_instance": task_result.output_root.name,
                            "output_root": str(task_result.output_root),
                        },
                    )
                    publish_jobs.append(
                        (
                            publisher_executor.submit(
                                publish_percentiles_output,
                                args=args,
                                version=task_result.version,
                                scenario=task_result.scenario,
                                run_instance=task_result.output_root.name,
                                output_root=task_result.output_root,
                                ledger_path=ledger_path,
                            ),
                            task_result,
                        )
                    )
                if publish_jobs and drain_publish_jobs(publish_jobs=publish_jobs, wait_for_all=False):
                    stopped_on_failure = True
                    break
                if task_result.status != "completed":
                    stopped_on_failure = True
                    break
    else:
        tasks, skipped_preflight = build_manual_tasks(args)
        resume_precompleted: Set[Tuple[str, str]] = set()
        if args.resume and ledger_path.exists():
            resume_precompleted.update(load_completed_tasks_from_ledger(ledger_path))
            if resume_precompleted:
                before = len(tasks)
                tasks = [t for t in tasks if (t.version, t.scenario) not in resume_precompleted]
                removed = before - len(tasks)
                if removed > 0:
                    print(f"[815DRV] Resume filter removed {removed} pre-completed task(s).")

        print_task_overview(tasks, skipped_preflight)
        planned_task_rows = [_task_row(task) for task in tasks]

        if not args.dry_run:
            for idx, task in enumerate(tasks, start=1):
                print(f"[815DRV] ({idx}/{len(tasks)}) Running {task.version}/{task.scenario}")
                append_jsonl(
                    ledger_path,
                    {
                        "event": "task_start",
                        "timestamp_utc": utc_now().isoformat(),
                        "version": task.version,
                        "scenario": task.scenario,
                        "source_run_root": str(task.source_run_root),
                        "output_root": str(task.output_root),
                    },
                )
                task_result = run_task(
                    args=args,
                    task=task,
                    views_root=views_root,
                    logs_dir=logs_dir,
                    ledger_path=ledger_path,
                )
                results.append(task_result)
                append_jsonl(
                    ledger_path,
                    {
                        "event": "task_end",
                        "timestamp_utc": utc_now().isoformat(),
                        "version": task.version,
                        "scenario": task.scenario,
                        "output_root": str(task.output_root),
                        "status": task_result.status,
                        "reason": task_result.reason,
                    },
                )
                print(f"[815DRV]   -> {task_result.status} ({task_result.reason or 'ok'})")
                if args.publish_percentiles and task_result.status == "completed":
                    task_result.percentiles_publish_status = "pending"
                    task_result.percentiles_state = "done_local_only"
                    task_result.percentiles_publish_timetag = args.timetag
                    if publisher_executor is not None:
                        publish_jobs.append(
                            (
                                publisher_executor.submit(
                                    publish_percentiles_output,
                                    args=args,
                                    version=task_result.version,
                                    scenario=task_result.scenario,
                                    run_instance=task_result.output_root.name,
                                    output_root=task_result.output_root,
                                    ledger_path=ledger_path,
                                ),
                                task_result,
                            )
                        )
                if publish_jobs and drain_publish_jobs(publish_jobs=publish_jobs, wait_for_all=False):
                    stopped_on_failure = True
                    break

    if publisher_executor is not None:
        publisher_executor.shutdown(wait=True)
    if publish_jobs and drain_publish_jobs(publish_jobs=publish_jobs, wait_for_all=True):
        stopped_on_failure = True

    prior_planned_tasks = existing_manifest.get("planned_tasks", [])
    combined_planned_tasks = merge_rows(
        prior_planned_tasks if isinstance(prior_planned_tasks, list) else [],
        planned_task_rows,
        key_fields=("version", "scenario", "output_root"),
    )
    prior_preflight = existing_manifest.get("preflight_skipped", [])
    combined_preflight = merge_rows(
        prior_preflight if isinstance(prior_preflight, list) else [],
        skipped_preflight,
        key_fields=("version", "scenario", "reason"),
    )
    prior_discovered = existing_manifest.get("discovered_tasks", [])
    combined_discovered = merge_rows(
        prior_discovered if isinstance(prior_discovered, list) else [],
        discovered_rows,
        key_fields=("version", "scenario", "output_root"),
    )
    current_results = [_result_row(row) for row in results]
    prior_results = existing_manifest.get("results", [])
    combined_results = merge_rows(
        prior_results if isinstance(prior_results, list) else [],
        current_results,
        key_fields=("version", "scenario", "output_root"),
    )

    skipped_existing_count = sum(
        1 for row in combined_discovered if str(row.get("status", "")).strip() == "done"
    )
    completed_count = sum(1 for row in combined_results if str(row.get("status", "")).strip() == "completed")
    quarantined_count = sum(1 for row in combined_results if str(row.get("status", "")).strip() != "completed")

    base_manifest: Dict[str, Any] = {
        "generated_at": utc_now().isoformat(),
        "timetag": args.timetag,
        "auto_localstaging": bool(args.auto_localstaging),
        "versions": [normalize_version(v) for v in (args.versions or [])],
        "scenarios": list(dict.fromkeys(args.scenarios or [])),
        "source_root": str(args.source_root),
        "localstaging_root": str(args.localstaging_root),
        "localresults_root": str(args.localresults_root),
        "cmip6_localresults_root": str(args.cmip6_localresults_root),
        "meta_root": str(meta_root),
        "views_root": str(views_root),
        "script_101": str(DEFAULT_101_SCRIPT),
        "script_815": str(args.script_815),
        "script_350": str(args.script_350),
        "python": str(args.python),
        "run_cmip6": bool(args.run_cmip6),
        "cmip6_upload_bucket": str(args.cmip6_upload_bucket),
        "cmip6_upload_prefix": str(args.cmip6_upload_prefix),
        "cmip6_frontend_catalog": str(args.cmip6_frontend_catalog),
        "cmip6_skip_existing": bool(args.cmip6_skip_existing),
        "cmip6_member_workers": int(args.cmip6_member_workers),
        "cleanup_local_after_upload": bool(args.cleanup_local_after_upload),
        "publish_percentiles": bool(args.publish_percentiles),
        "scenario_publish_script": str(args.scenario_publish_script),
        "r2_remote": str(args.r2_remote),
        "r2_bucket": str(args.r2_bucket),
        "r2_prefix": str(args.r2_prefix),
        "stall_hours": float(args.stall_hours),
        "poll_seconds": int(args.poll_seconds),
        "workers_per_scenario": int(args.workers_per_scenario),
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
        "tmp_stage_summary": tmp_stage_summary,
        "backlog_actions": backlog_rows,
        "preflight_skipped": combined_preflight,
        "discovered_tasks": combined_discovered,
        "planned_tasks": combined_planned_tasks,
        "results": combined_results,
        "summary": {
            "tasks_discovered": len(combined_discovered),
            "tasks_planned": len(combined_planned_tasks),
            "tasks_completed": completed_count,
            "tasks_quarantined": quarantined_count,
            "tasks_skipped_preflight": len(combined_preflight),
            "tasks_skipped_existing_output": skipped_existing_count,
            "backlog_actions": len(backlog_rows),
            "stopped_on_failure": bool(stopped_on_failure),
        },
    }

    write_json_atomic(manifest_path, base_manifest)
    if args.dry_run:
        print(f"[815DRV] Dry run complete. Manifest: {manifest_path}")
        return

    print(f"[815DRV] Run complete. Manifest: {manifest_path}")
    print(
        f"[815DRV] Completed={completed_count} "
        f"Quarantined={quarantined_count} "
        f"PreflightSkipped={len(combined_preflight)} "
        f"SkippedExisting={skipped_existing_count}"
    )
    if stopped_on_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
