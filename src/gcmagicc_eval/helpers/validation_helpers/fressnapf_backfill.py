"""Sequential Fressnapf backfill runner for canonical GUS data."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

from .era5spliced_publish_state import write_json_atomic
from .helper_path_utils import (
    FRESSNAPF_CMIP6_VETTED_ROOT,
    get_logs_path,
    get_projects_root,
    get_site,
)

DEFAULT_STATE_ROOT = Path(get_logs_path()) / "fressnapf_backfill" / "gus"
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_HEARTBEAT_SECONDS = 300
DEFAULT_WATCH_INTERVAL_SECONDS = 5
DEFAULT_RCLONE_REMOTE = "ovh"
DEFAULT_RCLONE_BUCKET = "gcmagicc-scratch"
FRESSNAPF_ROOT = Path("data/archive")
FRESSNAPF_GCMAGICCOUTPUT_ROOT = FRESSNAPF_ROOT / "GCMAGICCoutput"
FRESSNAPF_CMIP6_ROOT = FRESSNAPF_ROOT / "CMIP6"
FRESSNAPF_ERA5_ROOT = FRESSNAPF_ROOT / "ERA5"
GUS_HOSTNAME_PREFIX = "data/site_gus/projects"
HEALPIX_SOURCE_ENV_CMIP6 = "GCMAGICC_FRESSNAPF_CMIP6_HEALPIX_SOURCE"
HEALPIX_SOURCE_ENV_ERA5 = "GCMAGICC_FRESSNAPF_ERA5_HEALPIX_SOURCE"
CONCURRENCY_LADDER: tuple[tuple[int, int], ...] = ((16, 32), (8, 16), (4, 8))
TERMINAL_STATUSES = {"completed_verified", "failed_fatal", "failed_retryable", "preflight_failed"}


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    source: str
    destination: Path
    exclude_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class DestinationInventory:
    exists: bool
    nonempty: bool
    existing_files: int
    existing_bytes: int


@dataclass
class JobState:
    job_id: str
    source: str
    destination: str
    status: str = "pending"
    attempt: int = 0
    retry_count: int = 0
    pid: int | None = None
    bytes_copied: int = 0
    files_copied: int = 0
    elapsed_seconds: float = 0.0
    seeded_existing: bool = False
    existing_files: int = 0
    existing_bytes: int = 0
    adopted_at: str | None = None
    destination_nonempty: bool = False
    verification_mismatch_count: int = 0
    last_successful_check_time: str | None = None
    last_error: str | None = None
    updated_utc: str = field(default_factory=lambda: _utc_now())

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["updated_utc"] = self.updated_utc
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, job: JobSpec) -> "JobState":
        return cls(
            job_id=str(payload.get("job_id") or job.job_id),
            source=str(payload.get("source") or job.source),
            destination=str(payload.get("destination") or str(job.destination)),
            status=str(payload.get("status") or "pending"),
            attempt=int(payload.get("attempt", 0) or 0),
            retry_count=int(payload.get("retry_count", 0) or 0),
            pid=int(payload["pid"]) if payload.get("pid") not in (None, "") else None,
            bytes_copied=int(payload.get("bytes_copied", 0) or 0),
            files_copied=int(payload.get("files_copied", 0) or 0),
            elapsed_seconds=float(payload.get("elapsed_seconds", 0.0) or 0.0),
            seeded_existing=bool(payload.get("seeded_existing", False)),
            existing_files=int(payload.get("existing_files", 0) or 0),
            existing_bytes=int(payload.get("existing_bytes", 0) or 0),
            adopted_at=_none_if_blank(payload.get("adopted_at")),
            destination_nonempty=bool(payload.get("destination_nonempty", False)),
            verification_mismatch_count=int(payload.get("verification_mismatch_count", 0) or 0),
            last_successful_check_time=_none_if_blank(payload.get("last_successful_check_time")),
            last_error=_none_if_blank(payload.get("last_error")),
            updated_utc=str(payload.get("updated_utc") or _utc_now()),
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _none_if_blank(value: Any) -> str | None:
    token = str(value).strip() if value is not None else ""
    return token or None


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        handle.write(text)
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _format_bytes(num_bytes: int) -> str:
    value = float(max(0, int(num_bytes)))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024.0 or unit == "PiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def _existing_parent(path: Path) -> Path:
    probe = path.expanduser().resolve(strict=False)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


def _is_path_writable(path: Path) -> bool:
    probe = _existing_parent(path)
    return os.access(probe, os.W_OK)


def _run_command(cmd: Sequence[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(cmd),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(
            list(cmd),
            returncode=127,
            stdout="",
            stderr=str(exc),
        )


def _rclone_lsf(path_ref: str) -> tuple[bool, str, str]:
    proc = _run_command(["rclone", "lsf", path_ref, "--max-depth", "1"])
    sample = ""
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            token = line.strip()
            if token:
                sample = token
                break
    return proc.returncode == 0 and bool(sample), sample, (proc.stderr or proc.stdout).strip()


def _tail_text(path: Path, limit: int = 20) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-limit:])


def _load_sibling_module(module_path: Path, *, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load sibling module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_cmipcruncher_firefly_root(projects_root: Path | None = None) -> Path | None:
    root = Path(projects_root or get_projects_root()).expanduser().resolve(strict=False)
    candidate = root / "cmipcruncher_firefly"
    if candidate.exists():
        return candidate
    return None


def _normalize_source_ref(raw: str) -> str:
    token = str(raw or "").strip()
    if not token:
        raise ValueError("Empty source prefix.")
    head = token.split("/", 1)[0]
    if ":" in head:
        return token
    return f"{DEFAULT_RCLONE_REMOTE}:{DEFAULT_RCLONE_BUCKET}/{token.strip('/')}"


def _extract_cmip6_healpix_nametag(source_ref: str) -> str:
    token = source_ref.split(":", 1)[-1]
    marker = "/cmip6_healpix/"
    if marker not in token:
        raise ValueError(f"Cannot extract CMIP6 HEALPix nametag from {source_ref}")
    suffix = token.split(marker, 1)[1].strip("/")
    nametag = suffix.split("/", 1)[0].strip()
    if not nametag:
        raise ValueError(f"Cannot extract CMIP6 HEALPix nametag from {source_ref}")
    return nametag


def _extract_era5_healpix_nametag(source_ref: str) -> str:
    token = source_ref.split(":", 1)[-1]
    marker = "/cmip6_healpix/"
    suffix = token.split(marker, 1)[1].strip("/") if marker in token else ""
    parts = suffix.split("/")
    if len(parts) < 3:
        raise ValueError(f"Cannot extract ERA5 HEALPix nametag from {source_ref}")
    return parts[0]


def resolve_cmip6_healpix_source(projects_root: Path | None = None, *, environ: dict[str, str] | None = None) -> str:
    env = environ or os.environ
    override = str(env.get(HEALPIX_SOURCE_ENV_CMIP6, "")).strip()
    if override:
        return _normalize_source_ref(override)
    sibling_root = find_cmipcruncher_firefly_root(projects_root)
    if sibling_root is None:
        raise RuntimeError(
            "CMIP6 HEALPix source unresolved: sibling repo 'cmipcruncher_firefly' not found under "
            f"{Path(projects_root or get_projects_root()).resolve(strict=False)} and "
            f"{HEALPIX_SOURCE_ENV_CMIP6} is not set."
        )
    module = _load_sibling_module(
        sibling_root / "src" / "cmip6cruncher" / "healpix_defaults.py",
        module_name="_gcmagicc_cmip6_healpix_defaults",
    )
    prefix = module.build_default_cmip6_healpix_prefix(site="gus", version="v100")
    return _normalize_source_ref(prefix)


def resolve_era5_healpix_source(projects_root: Path | None = None, *, environ: dict[str, str] | None = None) -> str:
    env = environ or os.environ
    override = str(env.get(HEALPIX_SOURCE_ENV_ERA5, "")).strip()
    if override:
        return _normalize_source_ref(override)
    sibling_root = find_cmipcruncher_firefly_root(projects_root)
    if sibling_root is None:
        raise RuntimeError(
            "ERA5 HEALPix source unresolved: sibling repo 'cmipcruncher_firefly' not found under "
            f"{Path(projects_root or get_projects_root()).resolve(strict=False)} and "
            f"{HEALPIX_SOURCE_ENV_ERA5} is not set."
        )
    module = _load_sibling_module(
        sibling_root / "src" / "cmip6cruncher" / "healpix_defaults.py",
        module_name="_gcmagicc_era5_healpix_defaults",
    )
    prefix = module.build_default_era5_healpix_day_prefix(site="gus", version="v100")
    return _normalize_source_ref(prefix)


def build_job_specs(projects_root: Path | None = None, *, environ: dict[str, str] | None = None) -> list[JobSpec]:
    cmip6_healpix_source = resolve_cmip6_healpix_source(projects_root, environ=environ)
    era5_healpix_source = resolve_era5_healpix_source(projects_root, environ=environ)
    cmip6_nametag = _extract_cmip6_healpix_nametag(cmip6_healpix_source)
    era5_nametag = _extract_era5_healpix_nametag(era5_healpix_source)
    return [
        JobSpec(
            job_id="era5spliced_localstaging_archive",
            source="ovh:gcmagicc-scratch/nc/consolidated/era5spliced",
            destination=FRESSNAPF_GCMAGICCOUTPUT_ROOT / "ERA5spliced" / "localstaging_archive",
        ),
        JobSpec(
            job_id="cmip6replicas_v101",
            source="ovh:gcmagicc-scratch/nc/eth/v101/gcmagicc",
            destination=FRESSNAPF_GCMAGICCOUTPUT_ROOT / "CMIP6replicas" / "v101",
        ),
        JobSpec(
            job_id="cmip6replicas_v100",
            source="ovh:gcmagicc-scratch/nc/gus/v100/gcmagicc",
            destination=FRESSNAPF_GCMAGICCOUTPUT_ROOT / "CMIP6replicas" / "v100",
        ),
        JobSpec(
            job_id="cmip6_monthly_multivar_1x1",
            source="ovh:gcmagicc-scratch/nc/reference/out_ETHFOG_10June2025_vetted",
            destination=FRESSNAPF_CMIP6_VETTED_ROOT,
        ),
        JobSpec(
            job_id="cmip6_reference_net",
            source="ovh:gcmagicc-scratch/nc/reference/cmip6_ETHFOG_net",
            destination=FRESSNAPF_CMIP6_ROOT / "ETHFOG" / "net",
        ),
        JobSpec(
            job_id="cmip6_daily_healpix",
            source=cmip6_healpix_source,
            destination=FRESSNAPF_CMIP6_ROOT / "healpix" / cmip6_nametag,
            exclude_patterns=("_manifests/**",),
        ),
        JobSpec(
            job_id="era5_daily_healpix",
            source=era5_healpix_source,
            destination=FRESSNAPF_ERA5_ROOT / "healpix" / era5_nametag / "historical-ERA5" / "day",
        ),
    ]


def build_copy_command(job: JobSpec, *, attempt: int, log_path: Path) -> list[str]:
    idx = min(max(0, attempt - 1), len(CONCURRENCY_LADDER) - 1)
    transfers, checkers = CONCURRENCY_LADDER[idx]
    cmd = [
        "rclone",
        "copy",
        job.source,
        str(job.destination),
        "--ignore-existing",
        "--use-json-log",
        "--stats",
        "60s",
        "--stats-log-level",
        "NOTICE",
        "--log-file",
        str(log_path),
        "--transfers",
        str(transfers),
        "--checkers",
        str(checkers),
        "--retries",
        "1",
        "--low-level-retries",
        "1",
        "--retries-sleep",
        "0",
    ]
    for pattern in job.exclude_patterns:
        cmd.extend(["--exclude", pattern])
    return cmd


def build_check_command(job: JobSpec) -> list[str]:
    cmd = [
        "rclone",
        "check",
        job.source,
        str(job.destination),
        "--size-only",
    ]
    for pattern in job.exclude_patterns:
        cmd.extend(["--exclude", pattern])
    return cmd


def parse_stats_from_log_line(line: str) -> dict[str, Any] | None:
    token = str(line).strip()
    if not token:
        return None
    try:
        payload = json.loads(token)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        return None
    return {
        "bytes_copied": int(stats.get("bytes", 0) or 0),
        "files_copied": int(stats.get("transfers", 0) or 0),
        "elapsed_seconds": float(stats.get("elapsedTime", 0.0) or 0.0),
        "errors": int(stats.get("errors", 0) or 0),
        "time": str(payload.get("time") or _utc_now()),
        "level": str(payload.get("level") or ""),
        "message": str(payload.get("msg") or ""),
    }


def classify_error_text(text: str) -> str:
    token = str(text or "").lower()
    fatal_patterns = (
        "permission denied",
        "access is denied",
        "target not writable",
        "not writable",
        "read-only file system",
        "no space left on device",
        "disk full",
        "quota exceeded",
        "failed to create file system",
        "directory not found",
        "source missing",
        "object not found",
        "not found",
    )
    if any(pattern in token for pattern in fatal_patterns):
        return "fatal"
    return "retryable"


def inventory_destination(path: Path) -> DestinationInventory:
    target = path.expanduser().resolve(strict=False)
    if not target.exists():
        return DestinationInventory(False, False, 0, 0)
    existing_files = 0
    existing_bytes = 0
    nonempty = False
    for root, dirs, files in os.walk(target):
        if dirs or files:
            nonempty = True
        for name in files:
            file_path = Path(root) / name
            try:
                existing_bytes += file_path.stat().st_size
            except OSError:
                continue
            existing_files += 1
    return DestinationInventory(True, nonempty, existing_files, existing_bytes)


def assess_adoption_state(inventory: DestinationInventory, *, verification_ok: bool) -> str:
    if not inventory.nonempty:
        return "pending_copy"
    if verification_ok:
        return "completed_verified"
    return "seeded_unverified"


def era5spliced_layout_aligned(root: Path) -> bool:
    target = root.expanduser().resolve(strict=False)
    if not target.exists():
        return False
    for candidate in target.glob("v*/**/original/run_*"):
        if candidate.is_dir():
            return True
    return False


def _job_state_dir(state_root: Path, job_id: str) -> Path:
    return state_root / job_id


def _job_log_path(state_root: Path, job_id: str) -> Path:
    return _job_state_dir(state_root, job_id) / "job.log"


def _job_json_path(state_root: Path, job_id: str) -> Path:
    return _job_state_dir(state_root, job_id) / "job.json"


def _job_history_path(state_root: Path, job_id: str) -> Path:
    return _job_state_dir(state_root, job_id) / "job.history.jsonl"


def _load_state(job: JobSpec, *, state_root: Path) -> JobState:
    payload = _load_json(_job_json_path(state_root, job.job_id))
    if payload:
        return JobState.from_payload(payload, job=job)
    return JobState(job_id=job.job_id, source=job.source, destination=str(job.destination))


def _write_state(job_state: JobState, *, state_root: Path, all_states: Sequence[JobState]) -> None:
    job_state.updated_utc = _utc_now()
    write_json_atomic(_job_json_path(state_root, job_state.job_id), job_state.to_payload())
    _append_jsonl(_job_history_path(state_root, job_state.job_id), job_state.to_payload())
    latest_payload = {
        "generated_utc": _utc_now(),
        "state_root": str(state_root),
        "active_job_id": next((state.job_id for state in all_states if state.pid is not None), None),
        "all_completed_verified": all(state.status == "completed_verified" for state in all_states),
        "jobs": [state.to_payload() for state in all_states],
    }
    write_json_atomic(state_root / "latest.json", latest_payload)
    _write_text_atomic(state_root / "summary.md", build_summary_markdown(all_states, state_root=state_root))


def build_summary_markdown(states: Sequence[JobState], *, state_root: Path) -> str:
    lines = [
        "# Fressnapf Backfill",
        "",
        f"- Generated: {_utc_now()}",
        f"- State root: `{state_root}`",
        "",
        "| Job | Status | Attempt | Seeded | Files | Bytes | Last Error |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for state in states:
        lines.append(
            "| "
            + " | ".join(
                [
                    state.job_id,
                    state.status,
                    str(state.attempt),
                    "yes" if state.seeded_existing else "no",
                    str(state.files_copied or state.existing_files),
                    _format_bytes(state.bytes_copied or state.existing_bytes),
                    (state.last_error or "").replace("\n", " ")[:120],
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _consume_log_updates(log_path: Path, *, start_pos: int, state: JobState) -> tuple[int, bool]:
    pos = start_pos
    changed = False
    if not log_path.exists():
        return pos, changed
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(start_pos)
        while True:
            line = handle.readline()
            if not line:
                break
            pos = handle.tell()
            stats = parse_stats_from_log_line(line)
            if stats is not None:
                state.bytes_copied = int(stats["bytes_copied"])
                state.files_copied = int(stats["files_copied"])
                state.elapsed_seconds = float(stats["elapsed_seconds"])
                changed = True
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                level = str(payload.get("level") or "").lower()
                message = str(payload.get("msg") or "").strip()
                if level in {"error", "fatal"} and message:
                    state.last_error = message
                    changed = True
    return pos, changed


def _run_copy_attempt(
    job: JobSpec,
    *,
    state: JobState,
    all_states: Sequence[JobState],
    state_root: Path,
    heartbeat_seconds: int,
) -> tuple[int, str]:
    log_path = _job_log_path(state_root, job.job_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_pos = log_path.stat().st_size if log_path.exists() else 0
    cmd = build_copy_command(job, attempt=state.attempt, log_path=log_path)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    state.pid = proc.pid
    state.status = "copy_running"
    _write_state(state, state_root=state_root, all_states=all_states)
    last_flush = time.monotonic()
    position = start_pos
    while proc.poll() is None:
        time.sleep(1.0)
        position, changed = _consume_log_updates(log_path, start_pos=position, state=state)
        now = time.monotonic()
        if changed or (now - last_flush) >= float(heartbeat_seconds):
            _write_state(state, state_root=state_root, all_states=all_states)
            last_flush = now
    position, changed = _consume_log_updates(log_path, start_pos=position, state=state)
    stdout, stderr = proc.communicate()
    state.pid = None
    if changed:
        _write_state(state, state_root=state_root, all_states=all_states)
    error_text = "\n".join(part for part in [stderr.strip(), stdout.strip(), _tail_text(log_path)] if part)
    return proc.returncode or 0, error_text


def _run_check(job: JobSpec) -> tuple[int, str]:
    proc = _run_command(build_check_command(job))
    text = "\n".join(part for part in [proc.stdout.strip(), proc.stderr.strip()] if part)
    return proc.returncode, text


def _mount_info(path: Path) -> tuple[bool, str]:
    proc = _run_command(["findmnt", "-T", str(path), "--json"])
    if proc.returncode != 0:
        return False, proc.stderr.strip() or proc.stdout.strip()
    return True, proc.stdout.strip()


def collect_preflight(job_specs: Sequence[JobSpec]) -> dict[str, Any]:
    ok_mount, mount_text = _mount_info(FRESSNAPF_ROOT)
    usage_error = None
    usage_payload: dict[str, int] | None = None
    try:
        usage = shutil.disk_usage(FRESSNAPF_ROOT)
        usage_payload = {"total": int(usage.total), "used": int(usage.used), "free": int(usage.free)}
    except Exception as exc:
        usage_error = str(exc)

    remote_list = _run_command(["rclone", "listremotes"])
    remotes = [line.strip().rstrip(":") for line in remote_list.stdout.splitlines() if line.strip()]
    per_job: list[dict[str, Any]] = []
    errors: list[str] = []
    all_destinations_writable = True

    if shutil.which("rclone") is None:
        errors.append("rclone not found on PATH")
    if DEFAULT_RCLONE_REMOTE not in remotes:
        errors.append("Required rclone remote 'ovh:' not found.")
    if get_site() != "gus":
        errors.append(f"Current host resolves to site '{get_site()}', expected 'gus'.")
    if not ok_mount:
        errors.append(f"Fressnapf mount missing or unreadable: {mount_text}")
    if usage_error:
        errors.append(f"Disk usage query failed: {usage_error}")
    for job in job_specs:
        source_ok, source_sample, source_error = _rclone_lsf(job.source)
        inventory = inventory_destination(job.destination)
        destination_parent = _existing_parent(job.destination)
        destination_writable = _is_path_writable(job.destination)
        all_destinations_writable = all_destinations_writable and destination_writable
        job_payload = {
            "job_id": job.job_id,
            "source": job.source,
            "destination": str(job.destination),
            "source_ok": source_ok,
            "source_sample": source_sample,
            "source_error": source_error or None,
            "destination_exists": inventory.exists,
            "destination_nonempty": inventory.nonempty,
            "existing_files": inventory.existing_files,
            "existing_bytes": inventory.existing_bytes,
            "destination_parent": str(destination_parent),
            "destination_parent_writable": _is_path_writable(destination_parent),
            "destination_writable": destination_writable,
        }
        if job.job_id == "era5spliced_localstaging_archive":
            job_payload["layout_aligned"] = era5spliced_layout_aligned(job.destination)
        per_job.append(job_payload)
        if not source_ok:
            errors.append(f"{job.job_id}: source prefix unreachable or empty ({job.source})")
        if not destination_writable:
            errors.append(f"{job.job_id}: destination parent is not writable ({destination_parent})")

    return {
        "generated_utc": _utc_now(),
        "site": get_site(),
        "mount_ok": ok_mount,
        "mount_info": mount_text,
        "writable": all_destinations_writable,
        "disk_usage": usage_payload,
        "jobs": per_job,
        "errors": errors,
    }


def print_preflight_report(report: dict[str, Any]) -> None:
    print("Fressnapf backfill preflight")
    print("---------------------------")
    print(f"generated_utc : {report.get('generated_utc')}")
    print(f"site          : {report.get('site')}")
    print(f"mount_ok      : {report.get('mount_ok')}")
    print(f"writable      : {report.get('writable')}")
    disk_usage = report.get("disk_usage") or {}
    if disk_usage:
        print(f"disk_free     : {_format_bytes(int(disk_usage.get('free', 0)))}")
    for job in report.get("jobs", []):
        print("")
        print(f"[{job['job_id']}]")
        print(f"source_ok     : {job['source_ok']}")
        print(f"source_sample : {job.get('source_sample') or '<none>'}")
        print(f"destination   : {job['destination']}")
        print(f"nonempty      : {job['destination_nonempty']}")
        print(f"existing_files: {job['existing_files']}")
        print(f"existing_bytes: {_format_bytes(int(job['existing_bytes']))}")
        if "layout_aligned" in job:
            print(f"layout_aligned: {job['layout_aligned']}")
    if report.get("errors"):
        print("")
        print("Errors:")
        for error in report["errors"]:
            print(f"- {error}")


def _verify_seeded_destination(
    job: JobSpec,
    *,
    state: JobState,
    all_states: Sequence[JobState],
    state_root: Path,
) -> bool:
    state.status = "verifying_seeded"
    _write_state(state, state_root=state_root, all_states=all_states)
    rc, text = _run_check(job)
    if rc == 0:
        state.status = "completed_verified"
        state.adopted_at = _utc_now()
        state.last_successful_check_time = state.adopted_at
        state.last_error = None
        _write_state(state, state_root=state_root, all_states=all_states)
        return True
    state.status = "seeded_unverified"
    state.last_error = text or "Seeded verification failed."
    _write_state(state, state_root=state_root, all_states=all_states)
    return False


def _ensure_destination_ready(job: JobSpec) -> None:
    job.destination.mkdir(parents=True, exist_ok=True)


def _next_attempt_number(state: JobState) -> int:
    return max(1, int(state.attempt) + 1)


def _should_skip_completed(state: JobState) -> bool:
    return state.status == "completed_verified"


def _run_job(
    job: JobSpec,
    *,
    state: JobState,
    all_states: Sequence[JobState],
    state_root: Path,
    max_attempts: int,
    heartbeat_seconds: int,
) -> bool:
    inventory = inventory_destination(job.destination)
    state.seeded_existing = inventory.nonempty
    state.destination_nonempty = inventory.nonempty
    state.existing_files = inventory.existing_files
    state.existing_bytes = inventory.existing_bytes
    if inventory.nonempty:
        state.status = "seeded_detected"
        _write_state(state, state_root=state_root, all_states=all_states)
        if _verify_seeded_destination(job, state=state, all_states=all_states, state_root=state_root):
            return True

    attempts_used = max(0, int(state.attempt))
    while attempts_used < int(max_attempts):
        state.attempt = _next_attempt_number(state)
        state.retry_count = max(0, state.attempt - 1)
        attempts_used = state.attempt
        try:
            _ensure_destination_ready(job)
        except Exception as exc:
            state.status = "failed_fatal"
            state.last_error = f"Failed to create destination directory: {exc}"
            _write_state(state, state_root=state_root, all_states=all_states)
            return False

        copy_rc, copy_error = _run_copy_attempt(
            job,
            state=state,
            all_states=all_states,
            state_root=state_root,
            heartbeat_seconds=heartbeat_seconds,
        )
        if copy_rc != 0:
            state.last_error = copy_error or f"rclone copy failed with exit code {copy_rc}"
            classification = classify_error_text(state.last_error)
            if classification == "fatal":
                state.status = "failed_fatal"
                _write_state(state, state_root=state_root, all_states=all_states)
                return False
            if attempts_used >= int(max_attempts):
                state.status = "failed_retryable"
                _write_state(state, state_root=state_root, all_states=all_states)
                return False
            state.status = "retrying"
            _write_state(state, state_root=state_root, all_states=all_states)
            continue

        state.status = "checking"
        _write_state(state, state_root=state_root, all_states=all_states)
        check_rc, check_text = _run_check(job)
        if check_rc == 0:
            state.status = "completed_verified"
            state.last_successful_check_time = _utc_now()
            state.last_error = None
            _write_state(state, state_root=state_root, all_states=all_states)
            return True

        state.verification_mismatch_count += 1
        state.last_error = check_text or "rclone check failed."
        if state.verification_mismatch_count >= 2:
            state.status = "failed_fatal"
            _write_state(state, state_root=state_root, all_states=all_states)
            return False
        if attempts_used >= int(max_attempts):
            state.status = "failed_retryable"
            _write_state(state, state_root=state_root, all_states=all_states)
            return False
        state.status = "retrying"
        _write_state(state, state_root=state_root, all_states=all_states)
    state.status = "failed_retryable"
    _write_state(state, state_root=state_root, all_states=all_states)
    return False


def _first_incomplete_index(states: Sequence[JobState]) -> int:
    for idx, state in enumerate(states):
        if state.status != "completed_verified":
            return idx
    return len(states)


def _csv_to_list(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _filter_job_specs(job_specs: Sequence[JobSpec], selected_job_ids: Sequence[str]) -> list[JobSpec]:
    if not selected_job_ids:
        return list(job_specs)
    jobs_by_id = {job.job_id: job for job in job_specs}
    missing = [job_id for job_id in selected_job_ids if job_id not in jobs_by_id]
    if missing:
        available = ", ".join(sorted(jobs_by_id))
        missing_txt = ", ".join(sorted(dict.fromkeys(missing)))
        raise RuntimeError(f"Unknown --jobs selection: {missing_txt}. Available jobs: {available}")
    ordered: list[JobSpec] = []
    seen: set[str] = set()
    for job_id in selected_job_ids:
        if job_id in seen:
            continue
        ordered.append(jobs_by_id[job_id])
        seen.add(job_id)
    return ordered


def _resolve_job_specs_for_cli(
    projects_root: Path | None,
    *,
    selected_job_ids: Sequence[str] | None = None,
) -> tuple[list[JobSpec] | None, str | None]:
    try:
        jobs = build_job_specs(projects_root)
        return _filter_job_specs(jobs, tuple(selected_job_ids or ())), None
    except RuntimeError as exc:
        return None, str(exc)


def run_backfill(
    *,
    state_root: Path,
    projects_root: Path | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    heartbeat_seconds: int = DEFAULT_HEARTBEAT_SECONDS,
    resume_failed: bool = False,
    selected_job_ids: Sequence[str] | None = None,
) -> int:
    jobs, error = _resolve_job_specs_for_cli(projects_root, selected_job_ids=selected_job_ids)
    if jobs is None:
        print(f"Preflight failed: {error}")
        return 1
    report = collect_preflight(jobs)
    print_preflight_report(report)
    if report.get("errors"):
        return 1

    state_root.mkdir(parents=True, exist_ok=True)
    states = [_load_state(job, state_root=state_root) for job in jobs]
    start_idx = _first_incomplete_index(states) if resume_failed else 0
    for idx, job in enumerate(jobs):
        state = states[idx]
        if idx < start_idx and state.status == "completed_verified":
            continue
        if _should_skip_completed(state):
            _write_state(state, state_root=state_root, all_states=states)
            continue
        ok = _run_job(
            job,
            state=state,
            all_states=states,
            state_root=state_root,
            max_attempts=max_attempts,
            heartbeat_seconds=heartbeat_seconds,
        )
        if not ok:
            return 1
    return 0


def status_command(*, state_root: Path) -> int:
    summary_path = state_root / "summary.md"
    latest_path = state_root / "latest.json"
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))
        return 0
    if latest_path.exists():
        print(json.dumps(_load_json(latest_path), indent=2))
        return 0
    print(f"No backfill state found under {state_root}")
    return 1


def watch_command(*, state_root: Path, interval_seconds: int) -> int:
    summary_path = state_root / "summary.md"
    latest_path = state_root / "latest.json"
    last_seen = ""
    try:
        while True:
            current = ""
            if summary_path.exists():
                current = summary_path.read_text(encoding="utf-8")
            elif latest_path.exists():
                current = json.dumps(_load_json(latest_path), indent=2)
            if current and current != last_seen:
                print(current)
                last_seen = current
            latest = _load_json(latest_path)
            jobs = latest.get("jobs")
            if isinstance(jobs, list) and jobs and all(str(job.get("status")) in TERMINAL_STATUSES for job in jobs):
                return 0
            time.sleep(max(1, int(interval_seconds)))
    except KeyboardInterrupt:
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--projects-root", type=Path, default=None)
    parser.add_argument(
        "--jobs",
        type=str,
        default="",
        help="Optional comma-separated job_id filter (e.g. cmip6replicas_v100,cmip6replicas_v101).",
    )
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--heartbeat-seconds", type=int, default=DEFAULT_HEARTBEAT_SECONDS)
    parser.add_argument("--watch-interval-seconds", type=int, default=DEFAULT_WATCH_INTERVAL_SECONDS)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    subparsers.add_parser("run-gus")
    subparsers.add_parser("status")
    subparsers.add_parser("watch")
    subparsers.add_parser("resume-failed")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    state_root = Path(args.state_root).expanduser().resolve(strict=False)
    projects_root = (
        Path(args.projects_root).expanduser().resolve(strict=False)
        if args.projects_root is not None
        else None
    )
    selected_job_ids = _csv_to_list(args.jobs)
    if args.command == "plan":
        jobs, error = _resolve_job_specs_for_cli(projects_root, selected_job_ids=selected_job_ids)
        if jobs is None:
            print(f"Preflight failed: {error}")
            return 1
        report = collect_preflight(jobs)
        print_preflight_report(report)
        return 1 if report.get("errors") else 0
    if args.command == "run-gus":
        return run_backfill(
            state_root=state_root,
            projects_root=projects_root,
            max_attempts=int(args.max_attempts),
            heartbeat_seconds=int(args.heartbeat_seconds),
            resume_failed=False,
            selected_job_ids=selected_job_ids,
        )
    if args.command == "resume-failed":
        return run_backfill(
            state_root=state_root,
            projects_root=projects_root,
            max_attempts=int(args.max_attempts),
            heartbeat_seconds=int(args.heartbeat_seconds),
            resume_failed=True,
            selected_job_ids=selected_job_ids,
        )
    if args.command == "status":
        return status_command(state_root=state_root)
    if args.command == "watch":
        return watch_command(state_root=state_root, interval_seconds=int(args.watch_interval_seconds))
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
