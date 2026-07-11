"""Shared input resolution helpers for GapFiller-style notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
from shutil import disk_usage
import subprocess
import sys
import time
from typing import Iterable, Sequence

from scr.validation_helpers.helper_path_utils import (
    CANONICAL_KIND_ORIGINAL,
    get_cmip6_vetted_path,
    get_cmip6replicas_localstaging_root,
    get_cmip6replicas_root,
    get_era5spliced_localstaging_root,
    get_era5spliced_root,
    get_gcmagicc_archive_candidates,
    get_object_bucket,
    get_object_remote,
    get_project_root,
    get_version_default,
    normalize_n_ensemble_label,
    resolve_canonical_dataset_root,
    resolve_s3_site_for_version,
    split_experiment_and_runmodus,
)


_VERSION_RE = re.compile(r"^v\d+(?:gxe)?$", re.IGNORECASE)
_CMIP6_GCMAGICC_MEMBER_RE = re.compile(r"^r\d+i\d+p\d+f\d+$")
_CMIP6_GCMAGICC_NAME_RE = re.compile(r"^GCMagicc-.*\.nc$", re.IGNORECASE)
_CMIP6_REBUILD_MODEL_FAMILY = {
    "v100": "NxlversA5",
    "v101": "NthreeversT1",
}


@dataclass(frozen=True)
class GapFillerSource:
    path: Path
    source_kind: str
    version: str
    scenario: str | None = None
    runmodus: str | None = None
    n_ensemble: str | None = None
    run_instance: str | None = None
    remote_prefix: str | None = None


@dataclass(frozen=True)
class GapFillerArchiveFile:
    path: str
    size: int
    source_id: str | None
    scenario: str | None
    member: str | None


@dataclass(frozen=True)
class GapFillerCmip6Preflight:
    status: str
    version: str
    scenario: str
    prefer_staged: bool
    single_member_per_source: bool
    required_scenarios: tuple[str, ...]
    matched_scenarios_upstream: tuple[str, ...]
    matched_scenarios_local: tuple[str, ...]
    available_scenarios_upstream: tuple[str, ...]
    missing_scenarios: tuple[str, ...]
    missing_remote_files: tuple[str, ...]
    missing_local_files: tuple[str, ...]
    recommended_stage_cmd: tuple[str, ...] | None
    resolved_source_path: Path | None
    resolved_source_kind: str | None
    remote_prefix: str
    stage_root: Path
    upstream_file_count: int
    upstream_total_bytes: int
    local_file_count: int
    local_total_bytes: int
    selected_upstream_files: tuple[GapFillerArchiveFile, ...] = ()
    selected_local_files: tuple[GapFillerArchiveFile, ...] = ()


@dataclass(frozen=True)
class GapFillerEra5StagePreflight:
    status: str
    version: str
    scenario: str
    arx: str
    runmodus: str
    n_ensemble: str
    run_instance: str | None
    canonical_source_path: Path
    resolved_source_path: Path
    resolved_source_kind: str
    stage_root: Path
    remote_prefix: str | None
    dry_run_cmd: tuple[str, ...] | None
    stage_cmd: tuple[str, ...] | None
    status_json: Path | None
    manifest_json: Path | None
    dry_run_returncode: int | None
    bytes_total: int
    files_total: int
    free_bytes: int | None
    eta_seconds: float | None
    eta_utc: str | None
    dry_run_stdout: str = ""
    dry_run_stderr: str = ""
    status_payload: dict[str, object] | None = None


def normalize_version_tag(version: str | None) -> str:
    token = str(version or get_version_default()).strip().lower()
    if token.startswith("v100"):
        return "v100"
    if token.startswith("v101"):
        return "v101"
    if token and not token.startswith("v"):
        token = f"v{token}"
    return token or get_version_default().strip().lower()


def infer_version_tag_from_path(path: Path | None) -> str | None:
    if path is None:
        return None
    for part in reversed(Path(path).parts):
        token = str(part).strip()
        if _VERSION_RE.fullmatch(token):
            return token.lower()
    return None


def infer_n_ensemble_from_path(path: Path | None, *, default: str = "n_100") -> str:
    if path is not None:
        for part in reversed(Path(path).parts):
            token = str(part).strip().lower()
            if token.startswith("n_"):
                try:
                    return normalize_n_ensemble_label(token)
                except ValueError:
                    pass
            match = re.search(r"debiasloop_(\d+)", token)
            if match:
                return normalize_n_ensemble_label(match.group(1))
    return normalize_n_ensemble_label(default)


def resolve_requested_version_tag(
    *,
    requested_version: str | None = None,
    preferred_paths: Sequence[Path | None] = (),
) -> str:
    if requested_version and str(requested_version).strip():
        return normalize_version_tag(requested_version)
    for path in preferred_paths:
        inferred = infer_version_tag_from_path(path)
        if inferred:
            return inferred
    return normalize_version_tag(get_version_default())


def _path_under(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve(strict=False).relative_to(Path(root).resolve(strict=False))
        return True
    except ValueError:
        return False


def _safe_exists(path: Path) -> bool:
    try:
        return Path(path).exists()
    except OSError:
        return False


def _safe_is_dir(path: Path) -> bool:
    try:
        return Path(path).is_dir()
    except OSError:
        return False


def _iter_nc_files(root: Path) -> list[Path]:
    if not _safe_is_dir(root):
        return []
    try:
        return sorted(p for p in Path(root).glob("*.nc") if p.is_file())
    except OSError:
        return []


def _parse_archive_filename(path: Path) -> tuple[str | None, str | None, str | None]:
    parts = path.stem.split("_")
    if len(parts) < 4:
        return None, None, None
    source_id = parts[1]
    scenario = parts[2]
    member = parts[3]
    if not _CMIP6_GCMAGICC_MEMBER_RE.fullmatch(member):
        return source_id, scenario, None
    return source_id, scenario, member


def _archive_file_from_path(path: Path) -> GapFillerArchiveFile | None:
    source_id, scenario, member = _parse_archive_filename(path)
    if source_id is None and scenario is None and member is None:
        return None
    try:
        size = int(Path(path).stat().st_size)
    except OSError:
        return None
    return GapFillerArchiveFile(
        path=str(Path(path).as_posix()),
        size=size,
        source_id=source_id,
        scenario=str(scenario).strip().lower() if scenario else None,
        member=member,
    )


def _archive_file_from_remote_row(row: dict[str, object]) -> GapFillerArchiveFile | None:
    rel = str(row.get("Path") or row.get("Name") or "").strip()
    if not rel:
        return None
    source_id, scenario, member = _parse_archive_filename(Path(rel))
    if source_id is None and scenario is None and member is None:
        return None
    return GapFillerArchiveFile(
        path=rel,
        size=int(row.get("Size", 0) or 0),
        source_id=source_id,
        scenario=str(scenario).strip().lower() if scenario else None,
        member=member,
    )


def _rclone_capture_json(cmd: list[str]) -> list[dict[str, object]]:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from rclone lsjson: {exc}") from exc
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected lsjson payload shape.")
    return [row for row in payload if isinstance(row, dict)]


def discover_gapfiller_cmip6_remote_rows(remote_prefix: str) -> list[GapFillerArchiveFile]:
    rows = _rclone_capture_json(["rclone", "lsjson", remote_prefix, "--recursive", "--files-only"])
    out: list[GapFillerArchiveFile] = []
    for row in rows:
        item = _archive_file_from_remote_row(row)
        if item is not None:
            out.append(item)
    return sorted(out, key=lambda item: item.path)


def discover_gapfiller_cmip6_local_rows(root: Path) -> list[GapFillerArchiveFile]:
    if not _safe_is_dir(root):
        return []
    try:
        candidates = sorted(
            p for p in Path(root).rglob("GCMagicc-*.nc") if p.is_file() and _CMIP6_GCMAGICC_NAME_RE.match(p.name)
        )
    except OSError:
        return []
    out: list[GapFillerArchiveFile] = []
    for path in candidates:
        item = _archive_file_from_path(path)
        if item is not None:
            rel = str(path.relative_to(root).as_posix())
            out.append(
                GapFillerArchiveFile(
                    path=rel,
                    size=item.size,
                    source_id=item.source_id,
                    scenario=item.scenario,
                    member=item.member,
                )
            )
    return out


def _dedupe_tokens(tokens: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for token in tokens:
        norm = str(token).strip()
        if norm and norm not in out:
            out.append(norm)
    return tuple(out)


def _required_scenarios_for_gapfiller(scenario: str | None) -> tuple[str, ...]:
    token = str(scenario or "").strip().lower()
    if not token:
        return ()
    if token == "historical":
        return ("historical",)
    return (token, "historical")


def _filter_archive_rows(
    rows: Sequence[GapFillerArchiveFile],
    *,
    source_ids: Sequence[str] | None = None,
    members: Sequence[str] | None = None,
    scenarios: Sequence[str] | None = None,
    single_member_per_source: bool = False,
) -> list[GapFillerArchiveFile]:
    wanted_sources = {str(item).strip() for item in source_ids or [] if str(item).strip()}
    wanted_members = {str(item).strip() for item in members or [] if str(item).strip()}
    wanted_scenarios = {str(item).strip().lower() for item in scenarios or [] if str(item).strip()}
    filtered = []
    for row in rows:
        if wanted_sources and (row.source_id or "") not in wanted_sources:
            continue
        if wanted_members and (row.member or "") not in wanted_members:
            continue
        if wanted_scenarios and (row.scenario or "") not in wanted_scenarios:
            continue
        filtered.append(row)
    filtered = sorted(filtered, key=lambda item: item.path)
    if not single_member_per_source:
        return filtered
    seen: set[tuple[str, str]] = set()
    out: list[GapFillerArchiveFile] = []
    for row in filtered:
        key = ((row.source_id or "").strip(), (row.scenario or "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _scenario_tokens(rows: Sequence[GapFillerArchiveFile]) -> tuple[str, ...]:
    return tuple(sorted({str(row.scenario).strip().lower() for row in rows if row.scenario}))


def _matching_archive_files(
    root: Path,
    *,
    scenarios: Sequence[str] | None = None,
    source_ids: Sequence[str] | None = None,
    members: Sequence[str] | None = None,
) -> list[Path]:
    if not _safe_is_dir(root):
        return []
    wanted_scenarios = {str(item).strip().lower() for item in scenarios or [] if str(item).strip()}
    wanted_sources = {str(item).strip() for item in source_ids or [] if str(item).strip()}
    wanted_members = {str(item).strip() for item in members or [] if str(item).strip()}
    try:
        candidates = sorted(
            p for p in Path(root).rglob("GCMagicc-*.nc") if p.is_file() and _CMIP6_GCMAGICC_NAME_RE.match(p.name)
        )
    except OSError:
        return []

    out: list[Path] = []
    matched_scenarios: set[str] = set()
    for path in candidates:
        source_id, scenario, member = _parse_archive_filename(path)
        if wanted_scenarios and (scenario or "").lower() not in wanted_scenarios:
            continue
        if wanted_sources and (source_id or "") not in wanted_sources:
            continue
        if wanted_members and (member or "") not in wanted_members:
            continue
        if scenario:
            matched_scenarios.add(str(scenario).strip().lower())
        out.append(path)
    if wanted_scenarios and not wanted_scenarios.issubset(matched_scenarios):
        return []
    return out


def matching_gapfiller_cmip6_archive_files(
    root: Path,
    *,
    scenarios: Sequence[str] | None = None,
    source_ids: Sequence[str] | None = None,
    members: Sequence[str] | None = None,
) -> list[Path]:
    return _matching_archive_files(
        root,
        scenarios=scenarios,
        source_ids=source_ids,
        members=members,
    )


def _local_stage_complete(stage_root: Path, source_root: Path) -> bool:
    stage_files = _iter_nc_files(stage_root)
    source_files = _iter_nc_files(source_root)
    if not stage_files or not source_files:
        return False
    stage_map = {p.name: p for p in stage_files}
    source_map = {p.name: p for p in source_files}
    if set(stage_map) != set(source_map):
        return False
    for name, source_path in source_map.items():
        local_path = stage_map[name]
        try:
            if local_path.stat().st_size != source_path.stat().st_size:
                return False
        except OSError:
            return False
    return True


def _gapfiller_stage_throughput_bytes_per_second() -> int:
    return int(
        float(os.environ.get("GCMAGICC_GAPFILLER_STAGE_THROUGHPUT_MIBPS", "150")) * 1024 * 1024
    )


def _existing_parent(path: Path) -> Path:
    probe = Path(path).expanduser().resolve(strict=False)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


def _estimate_stage_eta(total_bytes: int) -> tuple[float | None, str | None]:
    throughput = _gapfiller_stage_throughput_bytes_per_second()
    if total_bytes <= 0 or throughput <= 0:
        return None, None
    eta_seconds = float(total_bytes) / float(throughput)
    eta_utc = datetime.fromtimestamp(time.time() + eta_seconds, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return eta_seconds, eta_utc


def _parse_key_value_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in str(text or "").splitlines():
        token = line.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key.isupper():
            out[key] = value
    return out


def _load_json_file(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(max(0, int(num_bytes)))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024.0 or unit == "PiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def preflight_gapfiller_era5_stage(
    *,
    version: str | None,
    experiment_id: str,
    arx: str = "AR6",
    runmodus: str = "all",
    n_ensemble: str | int = "n_20",
    run_instance: str | None = None,
    canonical_root: Path | None = None,
    stage_root: Path | None = None,
) -> GapFillerEra5StagePreflight:
    version_tag = normalize_version_tag(version)
    scenario_token, runmodus_token = split_experiment_and_runmodus(experiment_id, runmodus_hint=runmodus)
    n_ensemble_token = normalize_n_ensemble_label(n_ensemble)
    canonical_source = resolve_gapfiller_era5_source(
        version=version_tag,
        experiment_id=experiment_id,
        arx=arx,
        runmodus=runmodus,
        n_ensemble=n_ensemble_token,
        run_instance=run_instance,
        canonical_root=canonical_root,
        stage_root=stage_root,
        prefer_staged=False,
    )
    resolved_source = resolve_gapfiller_era5_source(
        version=version_tag,
        experiment_id=experiment_id,
        arx=arx,
        runmodus=runmodus,
        n_ensemble=n_ensemble_token,
        run_instance=run_instance,
        canonical_root=canonical_root,
        stage_root=stage_root,
        prefer_staged=True,
    )
    stage_base = Path(stage_root or get_era5spliced_localstaging_root()).expanduser().resolve(strict=False)
    if _path_under(canonical_source.path, Path(canonical_root or get_era5spliced_root()).expanduser().resolve(strict=False)):
        rel = canonical_source.path.relative_to(Path(canonical_root or get_era5spliced_root()).expanduser().resolve(strict=False))
        stage_candidate = (stage_base / rel).resolve(strict=False)
    else:
        stage_candidate = stage_base

    if resolved_source.source_kind == "staged":
        return GapFillerEra5StagePreflight(
            status="ready",
            version=version_tag,
            scenario=scenario_token,
            arx=str(arx).strip() or "AR6",
            runmodus=runmodus_token,
            n_ensemble=n_ensemble_token,
            run_instance=resolved_source.run_instance,
            canonical_source_path=canonical_source.path,
            resolved_source_path=resolved_source.path,
            resolved_source_kind=resolved_source.source_kind,
            stage_root=resolved_source.path,
            remote_prefix=resolved_source.remote_prefix,
            dry_run_cmd=None,
            stage_cmd=None,
            status_json=None,
            manifest_json=None,
            dry_run_returncode=None,
            bytes_total=0,
            files_total=0,
            free_bytes=int(disk_usage(_existing_parent(resolved_source.path)).free),
            eta_seconds=None,
            eta_utc=None,
            status_payload={},
        )

    dry_run_cmd = build_gapfiller_era5_stage_command(
        version=version_tag,
        scenario=scenario_token,
        ensemble=n_ensemble_token,
        run_instance=resolved_source.run_instance,
        stage_base=stage_base,
        dry_run=True,
    )
    dry_run = subprocess.run([str(item) for item in dry_run_cmd], check=False, capture_output=True, text=True)
    stage_hint = _parse_key_value_lines(dry_run.stdout)
    status_json = Path(stage_hint["STATUS_JSON"]).expanduser().resolve(strict=False) if "STATUS_JSON" in stage_hint else None
    manifest_json = Path(stage_hint["MANIFEST_JSON"]).expanduser().resolve(strict=False) if "MANIFEST_JSON" in stage_hint else None
    stage_root_resolved = Path(stage_hint.get("STAGE_ROOT", stage_candidate)).expanduser().resolve(strict=False)
    status_payload = _load_json_file(status_json)
    bytes_total = int(status_payload.get("bytes_total", 0) or 0)
    files_total = int(status_payload.get("files_total", 0) or 0)
    free_bytes = int(disk_usage(_existing_parent(stage_root_resolved)).free)
    eta_seconds, eta_utc = _estimate_stage_eta(bytes_total)
    status = "stage_required" if dry_run.returncode == 0 else "stage_preflight_failed"

    return GapFillerEra5StagePreflight(
        status=status,
        version=version_tag,
        scenario=scenario_token,
        arx=str(arx).strip() or "AR6",
        runmodus=runmodus_token,
        n_ensemble=n_ensemble_token,
        run_instance=resolved_source.run_instance,
        canonical_source_path=canonical_source.path,
        resolved_source_path=resolved_source.path,
        resolved_source_kind=resolved_source.source_kind,
        stage_root=stage_root_resolved,
        remote_prefix=resolved_source.remote_prefix,
        dry_run_cmd=tuple(str(item) for item in dry_run_cmd),
        stage_cmd=tuple(
            str(item)
            for item in build_gapfiller_era5_stage_command(
                version=version_tag,
                scenario=scenario_token,
                ensemble=n_ensemble_token,
                run_instance=resolved_source.run_instance,
                stage_base=stage_base,
                dry_run=False,
            )
        ),
        status_json=status_json,
        manifest_json=manifest_json,
        dry_run_returncode=int(dry_run.returncode),
        bytes_total=bytes_total,
        files_total=files_total,
        free_bytes=free_bytes,
        eta_seconds=eta_seconds,
        eta_utc=eta_utc,
        dry_run_stdout=str(dry_run.stdout or ""),
        dry_run_stderr=str(dry_run.stderr or ""),
        status_payload=status_payload,
    )


def format_gapfiller_era5_stage_message(
    preflight: GapFillerEra5StagePreflight,
    *,
    include_stage_command: bool = True,
) -> str:
    lines = [
        "ERA5-spliced GapFiller inputs must be locally staged before running 781.",
        f"Version: {preflight.version}",
        f"Scenario: {preflight.scenario}",
        f"Workflow: {preflight.arx}",
        f"Runmodus: {preflight.runmodus}",
        f"Ensemble: {preflight.n_ensemble}",
        f"Canonical source: {preflight.canonical_source_path}",
        f"Resolved source: {preflight.resolved_source_path} ({preflight.resolved_source_kind})",
        f"Stage root: {preflight.stage_root}",
    ]
    if preflight.files_total:
        lines.append(f"Files to stage: {preflight.files_total}")
    if preflight.bytes_total:
        lines.append(f"Bytes to stage: {_format_bytes(preflight.bytes_total)}")
    if preflight.free_bytes is not None:
        lines.append(f"Free disk: {_format_bytes(preflight.free_bytes)}")
    if preflight.eta_seconds is not None:
        eta_bits = f"{int(preflight.eta_seconds)}s"
        if preflight.eta_utc:
            eta_bits += f" (ETA UTC {preflight.eta_utc})"
        lines.append(f"Estimated ETA: {eta_bits}")
    if preflight.status_json is not None:
        lines.append(f"STATUS_JSON: {preflight.status_json}")
    if preflight.manifest_json is not None:
        lines.append(f"MANIFEST_JSON: {preflight.manifest_json}")
    if include_stage_command and preflight.stage_cmd:
        lines.append("Stage command:")
        lines.append("  " + shlex.join(preflight.stage_cmd))
    if preflight.status == "stage_preflight_failed":
        dry_run_detail = preflight.dry_run_stderr.strip() or preflight.dry_run_stdout.strip()
        if dry_run_detail:
            lines.append("Stage dry-run failed:")
            lines.append(dry_run_detail)
    return "\n".join(lines)


def ensure_gapfiller_era5_staged(
    *,
    version: str | None,
    experiment_id: str,
    arx: str = "AR6",
    runmodus: str = "all",
    n_ensemble: str | int = "n_20",
    run_instance: str | None = None,
    canonical_root: Path | None = None,
    stage_root: Path | None = None,
    preflight: GapFillerEra5StagePreflight | None = None,
) -> GapFillerSource:
    if preflight is None:
        preflight = preflight_gapfiller_era5_stage(
            version=version,
            experiment_id=experiment_id,
            arx=arx,
            runmodus=runmodus,
            n_ensemble=n_ensemble,
            run_instance=run_instance,
            canonical_root=canonical_root,
            stage_root=stage_root,
        )
    if preflight.status == "ready":
        return GapFillerSource(
            path=preflight.resolved_source_path,
            source_kind=preflight.resolved_source_kind,
            version=preflight.version,
            scenario=preflight.scenario,
            runmodus=preflight.runmodus,
            n_ensemble=preflight.n_ensemble,
            run_instance=preflight.run_instance,
            remote_prefix=preflight.remote_prefix,
        )
    if preflight.status != "stage_required" or not preflight.stage_cmd:
        raise RuntimeError(format_gapfiller_era5_stage_message(preflight))

    stage_run = subprocess.run([str(item) for item in preflight.stage_cmd], check=False)
    if stage_run.returncode != 0:
        raise RuntimeError(
            format_gapfiller_era5_stage_message(preflight)
            + f"\nStage command exited with code {int(stage_run.returncode)}."
        )

    staged_source = resolve_gapfiller_era5_source(
        version=preflight.version,
        experiment_id=experiment_id,
        arx=arx,
        runmodus=runmodus,
        n_ensemble=preflight.n_ensemble,
        run_instance=preflight.run_instance,
        canonical_root=canonical_root,
        stage_root=stage_root,
        prefer_staged=True,
    )
    if staged_source.source_kind != "staged":
        raise RuntimeError(
            format_gapfiller_era5_stage_message(preflight)
            + "\nStage command completed but the local staged source still does not validate as complete."
        )
    return staged_source


def resolve_gapfiller_era5_source(
    *,
    version: str | None,
    experiment_id: str,
    arx: str = "AR6",
    runmodus: str = "all",
    n_ensemble: str | int = "n_20",
    run_instance: str | None = None,
    canonical_root: Path | None = None,
    stage_root: Path | None = None,
    prefer_staged: bool = True,
) -> GapFillerSource:
    version_tag = normalize_version_tag(version)
    canonical_base = Path(canonical_root or get_era5spliced_root()).expanduser().resolve(strict=False)
    canonical_root = Path(
        resolve_canonical_dataset_root(
            version=version_tag,
            experiment_id=experiment_id,
            arx=arx,
            runmodus=runmodus,
            n_ensemble=n_ensemble,
            kind=CANONICAL_KIND_ORIGINAL,
            run_instance=run_instance,
            root=canonical_base,
        )
    ).expanduser().resolve(strict=False)
    stage_base = Path(stage_root or get_era5spliced_localstaging_root()).expanduser().resolve(strict=False)
    stage_candidate: Path | None = None
    if _path_under(canonical_root, canonical_base):
        rel = canonical_root.relative_to(canonical_base)
        stage_candidate = (stage_base / rel).resolve(strict=False)
        if prefer_staged and _local_stage_complete(stage_candidate, canonical_root):
            return GapFillerSource(
                path=stage_candidate,
                source_kind="staged",
                version=version_tag,
                scenario=split_experiment_and_runmodus(experiment_id, runmodus_hint=runmodus)[0],
                runmodus=split_experiment_and_runmodus(experiment_id, runmodus_hint=runmodus)[1],
                n_ensemble=normalize_n_ensemble_label(n_ensemble),
                run_instance=canonical_root.name if canonical_root.name.startswith("run_") else run_instance,
                remote_prefix=(
                    f"s3://{get_object_bucket()}/nc/consolidated/era5spliced/{rel.as_posix()}"
                    if rel.parts
                    else None
                ),
            )

    if _iter_nc_files(canonical_root):
        return GapFillerSource(
            path=canonical_root,
            source_kind="canonical_mount" if _path_under(canonical_root, canonical_base) else "legacy",
            version=version_tag,
            scenario=split_experiment_and_runmodus(experiment_id, runmodus_hint=runmodus)[0],
            runmodus=split_experiment_and_runmodus(experiment_id, runmodus_hint=runmodus)[1],
            n_ensemble=normalize_n_ensemble_label(n_ensemble),
            run_instance=canonical_root.name if canonical_root.name.startswith("run_") else run_instance,
            remote_prefix=(
                f"s3://{get_object_bucket()}/nc/consolidated/era5spliced/{canonical_root.relative_to(canonical_base).as_posix()}"
                if _path_under(canonical_root, canonical_base)
                else None
            ),
        )

    raise FileNotFoundError(f"Unable to resolve a usable ERA5 GapFiller source for {version_tag}:{experiment_id}:{n_ensemble}")


def build_gapfiller_era5_stage_command(
    *,
    version: str | None,
    scenario: str,
    ensemble: str | int,
    run_instance: str | None = None,
    stage_base: Path | None = None,
    dry_run: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        str(get_project_root() / "notebooks" / "101_stage_S3_GCMAGICCfiles.py"),
        "--version",
        normalize_version_tag(version),
        "--scenario",
        str(scenario).strip(),
        "--ensemble",
        normalize_n_ensemble_label(ensemble),
    ]
    if run_instance:
        cmd.extend(["--run-instance", str(run_instance).strip()])
    if stage_base is not None:
        cmd.extend(["--stage-base", str(Path(stage_base).expanduser().resolve(strict=False))])
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def get_gapfiller_cmip6_remote_prefix(version: str | None) -> str:
    version_tag = normalize_version_tag(version)
    site = resolve_s3_site_for_version(version_tag)
    bucket = get_object_bucket()
    remote = get_object_remote()
    return f"{remote}:{bucket}/nc/{site}/{version_tag}/gcmagicc"


def get_gapfiller_cmip6_rebuild_model_family(version: str | None) -> str | None:
    return _CMIP6_REBUILD_MODEL_FAMILY.get(normalize_version_tag(version))


def resolve_gapfiller_cmip6_rebuild_output_root(version: str | None) -> Path:
    version_tag = normalize_version_tag(version)
    candidates = [
        Path(candidate).expanduser().resolve(strict=False)
        for candidate in get_gcmagicc_archive_candidates(version_tag, include_local_repo=True)
    ]
    for candidate in candidates:
        if _safe_is_dir(candidate):
            return candidate.parent.resolve(strict=False)
    for candidate in candidates:
        parent = candidate.parent.resolve(strict=False)
        if _safe_is_dir(parent):
            return parent
    return (get_project_root() / "data" / "GCMagicc").resolve(strict=False)


def build_gapfiller_cmip6_rebuild_command(
    *,
    version: str | None,
    scenario: str,
    input_root: Path | None = None,
    output_root: Path | None = None,
    requested_workers: int = 4,
    wrap_max_workers: int = 4,
    threads_per_run: int = 16,
    max_ensembles: int = 999,
) -> list[str] | None:
    version_tag = normalize_version_tag(version)
    scenario_token = str(scenario).strip().lower()
    model_family = get_gapfiller_cmip6_rebuild_model_family(version_tag)
    if scenario_token != "ssp245" or model_family is None:
        return None
    vetted_root = Path(input_root or get_cmip6_vetted_path()).expanduser().resolve(strict=False)
    output_base = Path(output_root or resolve_gapfiller_cmip6_rebuild_output_root(version_tag)).expanduser().resolve(strict=False)
    wrapper_path = (get_project_root() / "notebooks" / "310_wrap_300_notebook.py").resolve(strict=False)
    return [
        "env",
        f"GCMAGICC_INPUT_DIR={vetted_root}",
        f"GCMAGICC_OUTPUT_DIR={output_base}",
        f"GCMAGICC_MODEL_VERSION={model_family}",
        f"GCMAGICC_MODEL_NUMBER={version_tag.removeprefix('v')}",
        f"GCMAGICC_MAX_ENSEMBLES={max(1, int(max_ensembles))}",
        "GCMAGICC_CRUNCH_MULTIPLE_ENSEMBLES=0",
        f"GCMAGICC_REQUESTED_WORKERS={max(1, int(requested_workers))}",
        f"GCMAGICC_WRAP_MAX_WORKERS={max(1, int(wrap_max_workers))}",
        f"GCMAGICC_THREADS_PER_RUN_MAX={max(1, int(threads_per_run))}",
        "MAX_DUPLICATES=1",
        "PYTHONUNBUFFERED=1",
        sys.executable,
        str(wrapper_path),
    ]


def build_gapfiller_cmip6_use_rebuilt_archive_command(
    *,
    version: str | None,
    archive_root: Path | None = None,
) -> list[str]:
    version_tag = normalize_version_tag(version)
    output_base = Path(archive_root or resolve_gapfiller_cmip6_rebuild_output_root(version_tag)).expanduser().resolve(strict=False)
    return [
        sys.executable,
        str((get_project_root() / "notebooks" / "781Wrapper_GapFiller.py").resolve(strict=False)),
        "--version-tag",
        version_tag,
        "--",
        "--gcmagicc-cmip6-dir",
        str((output_base / version_tag).resolve(strict=False)),
    ]


def build_gapfiller_cmip6_stage_root(
    *,
    version: str | None,
    scenario: str,
    stage_base: Path | None = None,
    include_historical: bool = True,
) -> Path:
    version_tag = normalize_version_tag(version)
    scenario_token = str(scenario).strip() or "scenario"
    suffix = f"{scenario_token}__historical" if include_historical else scenario_token
    base = Path(stage_base or get_cmip6replicas_localstaging_root()).expanduser().resolve(strict=False)
    return (base / version_tag / suffix).resolve(strict=False)


def preflight_gapfiller_cmip6_inputs(
    *,
    version: str | None,
    scenario: str,
    source_ids: Sequence[str] | None = None,
    members: Sequence[str] | None = None,
    single_member: bool = False,
    prefer_staged: bool = True,
    stage_base: Path | None = None,
    remote_prefix: str | None = None,
) -> GapFillerCmip6Preflight:
    version_tag = normalize_version_tag(version)
    scenario_token = str(scenario).strip().lower()
    required_scenarios = _required_scenarios_for_gapfiller(scenario_token)
    stage_root = build_gapfiller_cmip6_stage_root(
        version=version_tag,
        scenario=scenario_token,
        stage_base=stage_base,
        include_historical="historical" in required_scenarios and scenario_token != "historical",
    )
    remote_prefix_resolved = str(remote_prefix or get_gapfiller_cmip6_remote_prefix(version_tag)).strip()
    upstream_all_rows = discover_gapfiller_cmip6_remote_rows(remote_prefix_resolved)
    upstream_scoped_rows = _filter_archive_rows(
        upstream_all_rows,
        source_ids=source_ids,
        members=members,
        single_member_per_source=single_member,
    )
    available_scenarios_upstream = _scenario_tokens(upstream_scoped_rows)
    selected_upstream_rows = _filter_archive_rows(
        upstream_scoped_rows,
        scenarios=required_scenarios,
    )
    matched_scenarios_upstream = _scenario_tokens(selected_upstream_rows)
    missing_scenarios = tuple(s for s in required_scenarios if s not in set(matched_scenarios_upstream))

    local_all_rows = discover_gapfiller_cmip6_local_rows(stage_root)
    local_scoped_rows = _filter_archive_rows(
        local_all_rows,
        source_ids=source_ids,
        members=members,
        single_member_per_source=single_member,
    )
    selected_local_rows = _filter_archive_rows(
        local_scoped_rows,
        scenarios=required_scenarios,
    )
    matched_scenarios_local = _scenario_tokens(selected_local_rows)

    upstream_map = {row.path: row for row in selected_upstream_rows}
    local_map = {row.path: row for row in selected_local_rows}
    missing_local_files = tuple(
        sorted(
            path
            for path, upstream_row in upstream_map.items()
            if path not in local_map or int(local_map[path].size) != int(upstream_row.size)
        )
    )
    stage_cmd = build_gapfiller_cmip6_stage_command(
        version=version_tag,
        scenario=scenario_token,
        source_ids=source_ids,
        members=members,
        single_member_per_source=bool(single_member),
        stage_base=stage_base,
        dry_run=False,
    )

    status = "ready"
    resolved_source_path: Path | None = stage_root
    resolved_source_kind: str | None = "staged"
    missing_remote_files: tuple[str, ...] = ()
    if missing_scenarios:
        status = "unavailable_upstream"
        resolved_source_path = None
        resolved_source_kind = None
    elif prefer_staged and missing_local_files:
        status = "stageable_missing_runs"
        resolved_source_path = None
        resolved_source_kind = None
    elif not prefer_staged and not selected_local_rows:
        try:
            resolved = resolve_gapfiller_cmip6_source(
                version=version_tag,
                scenario=scenario_token,
                source_ids=source_ids,
                members=members,
                prefer_staged=False,
            )
            resolved_source_path = resolved.path
            resolved_source_kind = resolved.source_kind
        except FileNotFoundError:
            status = "stageable_missing_runs" if not missing_scenarios else "unavailable_upstream"
            resolved_source_path = None
            resolved_source_kind = None

    return GapFillerCmip6Preflight(
        status=status,
        version=version_tag,
        scenario=scenario_token,
        prefer_staged=bool(prefer_staged),
        single_member_per_source=bool(single_member),
        required_scenarios=required_scenarios,
        matched_scenarios_upstream=matched_scenarios_upstream,
        matched_scenarios_local=matched_scenarios_local,
        available_scenarios_upstream=available_scenarios_upstream,
        missing_scenarios=missing_scenarios,
        missing_remote_files=missing_remote_files,
        missing_local_files=missing_local_files,
        recommended_stage_cmd=tuple(stage_cmd) if status == "stageable_missing_runs" else None,
        resolved_source_path=resolved_source_path,
        resolved_source_kind=resolved_source_kind,
        remote_prefix=remote_prefix_resolved,
        stage_root=stage_root,
        upstream_file_count=len(selected_upstream_rows),
        upstream_total_bytes=sum(int(row.size) for row in selected_upstream_rows),
        local_file_count=len(selected_local_rows),
        local_total_bytes=sum(int(row.size) for row in selected_local_rows),
        selected_upstream_files=tuple(selected_upstream_rows),
        selected_local_files=tuple(selected_local_rows),
    )


def format_gapfiller_cmip6_preflight_message(
    preflight: GapFillerCmip6Preflight,
    *,
    include_explicit_override_hint: bool = True,
) -> str:
    lines = [
        "CMIP6-aligned GCMagicc inputs are required for the GapFiller CMIP6 row.",
        f"Version:  {preflight.version}",
        f"Scenario: {preflight.scenario}",
        f"Stage root: {preflight.stage_root}",
    ]
    if preflight.required_scenarios:
        lines.append(f"Required scenarios: {', '.join(preflight.required_scenarios)}")
    if preflight.matched_scenarios_upstream:
        lines.append(f"Matched upstream: {', '.join(preflight.matched_scenarios_upstream)}")
    if preflight.matched_scenarios_local:
        lines.append(f"Matched locally: {', '.join(preflight.matched_scenarios_local)}")

    if preflight.status == "stageable_missing_runs":
        lines.extend(
            [
                "",
                "The upstream archive is valid, but the local staged subset is incomplete.",
                f"Missing local files: {len(preflight.missing_local_files)}",
            ]
        )
        if preflight.recommended_stage_cmd is not None:
            lines.append("Run this exact command to stage the remainder missing runs:")
            lines.append("  " + shlex.join(str(item) for item in preflight.recommended_stage_cmd))
            lines.append("This command stages only the still-missing files.")
    elif preflight.status == "unavailable_upstream":
        lines.extend(
            [
                "",
                "The canonical CMIP6-aligned archive for this version does not contain the requested future scenario.",
            ]
        )
        if preflight.available_scenarios_upstream:
            lines.append(f"Available upstream scenarios: {', '.join(preflight.available_scenarios_upstream)}")
        if preflight.missing_scenarios:
            lines.append(f"Missing scenarios upstream: {', '.join(preflight.missing_scenarios)}")
        rebuild_cmd = build_gapfiller_cmip6_rebuild_command(
            version=preflight.version,
            scenario=preflight.scenario,
        )
        rebuild_model_family = get_gapfiller_cmip6_rebuild_model_family(preflight.version)
        rebuild_output_root = resolve_gapfiller_cmip6_rebuild_output_root(preflight.version)
        if rebuild_cmd is not None and rebuild_model_family is not None:
            lines.extend(
                [
                    "",
                    "This version can be rebuilt locally from the vetted CMIP6 DAT inputs.",
                    f"Rebuild model family: {rebuild_model_family}",
                    f"Rebuild archive target: {(rebuild_output_root / normalize_version_tag(preflight.version)).resolve(strict=False)}",
                    "Exact rebuild command:",
                    "  " + shlex.join(str(item) for item in rebuild_cmd),
                    "After the rebuild, rerun 781 against the rebuilt local archive:",
                    "  " + shlex.join(
                        str(item)
                        for item in build_gapfiller_cmip6_use_rebuilt_archive_command(
                            version=preflight.version,
                            archive_root=rebuild_output_root,
                        )
                    ),
                ]
            )
        lines.append("No staging command exists for this version/scenario pair.")

    if include_explicit_override_hint and preflight.status != "ready":
        lines.append("If you have a different compatible archive, pass --gcmagicc-cmip6-dir explicitly.")
    return "\n".join(lines)


def resolve_gapfiller_cmip6_source(
    *,
    version: str | None,
    scenario: str | None = None,
    source_ids: Sequence[str] | None = None,
    members: Sequence[str] | None = None,
    prefer_staged: bool = True,
) -> GapFillerSource:
    version_tag = normalize_version_tag(version)
    scenarios = [str(scenario).strip(), "historical"] if scenario else None
    if prefer_staged and scenario:
        stage_root = build_gapfiller_cmip6_stage_root(version=version_tag, scenario=scenario)
        if _matching_archive_files(stage_root, scenarios=scenarios, source_ids=source_ids, members=members):
            return GapFillerSource(
                path=stage_root,
                source_kind="staged",
                version=version_tag,
                scenario=str(scenario).strip(),
                remote_prefix=f"s3://{get_object_bucket()}/nc/{resolve_s3_site_for_version(version_tag)}/{version_tag}/gcmagicc",
            )

    mount_root = (get_cmip6replicas_root().expanduser().resolve(strict=False) / version_tag).resolve(strict=False)
    if _matching_archive_files(mount_root, scenarios=scenarios, source_ids=source_ids, members=members):
        return GapFillerSource(
            path=mount_root,
            source_kind="canonical_mount",
            version=version_tag,
            scenario=str(scenario).strip() if scenario else None,
            remote_prefix=f"s3://{get_object_bucket()}/nc/{resolve_s3_site_for_version(version_tag)}/{version_tag}/gcmagicc",
        )

    for candidate in get_gcmagicc_archive_candidates(version_tag, include_local_repo=True):
        resolved = Path(candidate).expanduser().resolve(strict=False)
        if not _safe_exists(resolved):
            continue
        if _matching_archive_files(resolved, scenarios=scenarios, source_ids=source_ids, members=members):
            return GapFillerSource(
                path=resolved,
                source_kind="legacy_archive",
                version=version_tag,
                scenario=str(scenario).strip() if scenario else None,
                remote_prefix=f"s3://{get_object_bucket()}/nc/{resolve_s3_site_for_version(version_tag)}/{version_tag}/gcmagicc",
            )

    raise FileNotFoundError(f"Unable to resolve a usable CMIP6-aligned GapFiller source for {version_tag}:{scenario or 'all'}")


def build_gapfiller_cmip6_stage_command(
    *,
    version: str | None,
    scenario: str | Sequence[str],
    source_ids: Iterable[str] | None = None,
    members: Iterable[str] | None = None,
    single_member_per_source: bool = False,
    stage_base: Path | None = None,
    stage_root: Path | None = None,
    dry_run: bool = False,
) -> list[str]:
    scenarios: Sequence[str]
    if isinstance(scenario, str):
        scenarios = [scenario]
    else:
        scenarios = list(scenario)
    cmd = [
        sys.executable,
        str(get_project_root() / "scripts" / "stage_gapfiller_cmip6_subset.py"),
        "--version",
        normalize_version_tag(version),
    ]
    for scenario_token in scenarios:
        token = str(scenario_token).strip()
        if token:
            cmd.extend(["--scenario", token])
    if stage_base is not None:
        cmd.extend(["--stage-base", str(Path(stage_base).expanduser().resolve(strict=False))])
    if stage_root is not None:
        cmd.extend(["--stage-root", str(Path(stage_root).expanduser().resolve(strict=False))])
    for source_id in source_ids or ():
        token = str(source_id).strip()
        if token:
            cmd.extend(["--source-id", token])
    for member in members or ():
        token = str(member).strip()
        if token:
            cmd.extend(["--member", token])
    if single_member_per_source:
        cmd.append("--single-member-per-source")
    if dry_run:
        cmd.append("--dry-run")
    return cmd
