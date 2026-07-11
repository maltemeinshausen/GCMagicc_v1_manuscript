"""
Shared helpers for scenario-level drought 755 archives.

These helpers keep `760SUPERDRIVER` and `760ORCHESTRATOR` aligned on:
- local/canonical 755 root resolution
- tag completeness checks
- canonical S3 upload verification
- rehydrate back into localresults for 759 resume
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import fnmatch
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from scr.validation_helpers.era5spliced_publish_state import rclone_remote_path
from scr.validation_helpers.helper_path_utils import (
    build_era5spliced_dataset_path,
    get_era5spliced_localresults_root,
    get_era5spliced_root,
    get_object_bucket,
    get_object_remote,
    get_version_default,
)


_base755 = importlib.import_module("notebooks.755_add_SPEI_allLand_to_ensemble_outputs")
_base758_wrapper = importlib.import_module("notebooks.758Wrapper_SPEI_drought")

ARX_NAME = "AR6"
DEFAULT_ENSEMBLE = "n_100"
DEFAULT_ERA5S3_CONSOLIDATED_PREFIX = "nc/consolidated/era5spliced"
STORAGE_LOCAL = "local"
STORAGE_REMOTE = "remote"
STORAGE_CHOICES = (STORAGE_LOCAL, STORAGE_REMOTE)
DEFAULT_SCALE = int(_base755._base754.DEFAULT_SCALE_MONTHS)


@dataclass(frozen=True)
class Drought755ScenarioSpec:
    version: str
    experiment_id: str
    runmodus: str = "all"
    scenario_tag: Optional[str] = None
    ensemble: str = DEFAULT_ENSEMBLE
    arx: str = ARX_NAME

    @property
    def normalized_version(self) -> str:
        return normalize_version(self.version)

    @property
    def requested_scenario(self) -> str:
        return str(self.scenario_tag or self.experiment_id).strip()


@dataclass(frozen=True)
class Drought755ArchiveCandidate:
    spec: Drought755ScenarioSpec
    storage: str
    tag: str
    world_root: Path
    tag_root: Path
    validation: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "spec": asdict(self.spec),
            "storage": self.storage,
            "tag": self.tag,
            "world_root": str(self.world_root),
            "tag_root": str(self.tag_root),
            "validation": self.validation,
        }


def normalize_version(version: str | None) -> str:
    token = str(version or get_version_default()).strip().lower()
    if token and not token.startswith("v"):
        token = f"v{token}"
    if token.startswith("v100"):
        return "v100"
    if token.startswith("v101"):
        return "v101"
    return token or get_version_default()


def scenario_spec(
    *,
    version: str,
    experiment_id: str,
    runmodus: str = "all",
    scenario_tag: Optional[str] = None,
    ensemble: str = DEFAULT_ENSEMBLE,
    arx: str = ARX_NAME,
) -> Drought755ScenarioSpec:
    return Drought755ScenarioSpec(
        version=normalize_version(version),
        experiment_id=str(experiment_id).strip(),
        runmodus=str(runmodus).strip(),
        scenario_tag=str(scenario_tag).strip() if scenario_tag else None,
        ensemble=str(ensemble).strip() or DEFAULT_ENSEMBLE,
        arx=str(arx).strip() or ARX_NAME,
    )


def dataderivatives_root_for_spec(spec: Drought755ScenarioSpec, *, root: Path) -> Path:
    return build_era5spliced_dataset_path(
        version=spec.normalized_version,
        experiment_id=spec.experiment_id,
        arx=spec.arx,
        runmodus=spec.runmodus,
        n_ensemble=spec.ensemble,
        kind="dataderivatives",
        root=Path(root).expanduser().resolve(strict=False),
    ).expanduser().resolve(strict=False)


def world_root_for_spec(spec: Drought755ScenarioSpec, *, root: Path) -> Path:
    return (dataderivatives_root_for_spec(spec, root=root) / _base755.WORLD_SPEIX_DIRNAME).expanduser().resolve(strict=False)


def local_world_root_for_spec(
    spec: Drought755ScenarioSpec,
    *,
    localresults_root: Optional[Path] = None,
) -> Path:
    return world_root_for_spec(
        spec,
        root=Path(localresults_root or get_era5spliced_localresults_root()).expanduser().resolve(strict=False),
    )


def remote_world_root_for_spec(
    spec: Drought755ScenarioSpec,
    *,
    era5s3_root: Optional[Path] = None,
) -> Path:
    return world_root_for_spec(
        spec,
        root=Path(era5s3_root or get_era5spliced_root()).expanduser().resolve(strict=False),
    )


def tag_root_for_world_root(world_root: Path, tag: str) -> Path:
    return (Path(world_root).expanduser().resolve(strict=False) / str(tag).strip()).expanduser().resolve(strict=False)


def forcing_label_for_spec(spec: Drought755ScenarioSpec) -> str:
    return _base755.FORCING_SCENARIO2 if str(spec.runmodus).strip().lower() == "nat" else _base755.FORCING_SCENARIO1


def require_fit_for_spec(spec: Drought755ScenarioSpec) -> bool:
    return str(spec.runmodus).strip().lower() != "nat"


def file_size_map(root: Path) -> Dict[str, int]:
    resolved = Path(root).expanduser().resolve(strict=False)
    if not resolved.exists():
        return {}
    out: Dict[str, int] = {}
    for path in sorted(resolved.rglob("*")):
        if not path.is_file():
            continue
        out[path.relative_to(resolved).as_posix()] = int(path.stat().st_size)
    return out


def s3_remote_path_for_mount_root(remote_root: Path, *, era5s3_root: Optional[Path] = None) -> Optional[str]:
    resolved = Path(remote_root).expanduser().resolve(strict=False)
    canonical_root = Path(era5s3_root or get_era5spliced_root()).expanduser().resolve(strict=False)
    try:
        rel = resolved.relative_to(canonical_root)
    except ValueError:
        return None
    return rclone_remote_path(
        remote=get_object_remote(),
        bucket=get_object_bucket(),
        suffix=f"{DEFAULT_ERA5S3_CONSOLIDATED_PREFIX}/{rel.as_posix()}",
    )


def rclone_file_size_map(remote_path: str) -> Dict[str, int]:
    rows = _rclone_lsjson_rows(remote_path, recursive=True, files_only=True)
    out: Dict[str, int] = {}
    for item in rows:
        rel = str(item.get("Path") or item.get("Name") or "").strip("/")
        if not rel:
            continue
        out[rel] = int(item.get("Size") or 0)
    return out


def _rclone_lsjson_rows(
    remote_path: str,
    *,
    recursive: bool = False,
    files_only: bool = False,
    dirs_only: bool = False,
    max_depth: Optional[int] = None,
) -> List[Dict[str, Any]]:
    cmd = ["rclone", "lsjson", str(remote_path), "--s3-no-check-bucket"]
    if recursive:
        cmd.append("--recursive")
    if files_only:
        cmd.append("--files-only")
    if dirs_only:
        cmd.append("--dirs-only")
    if max_depth is not None:
        cmd.extend(["--max-depth", str(int(max_depth))])
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"rclone lsjson failed for {remote_path}: rc={proc.returncode} stderr={proc.stderr.strip()}"
        )
    try:
        payload = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse rclone lsjson output for {remote_path}") from exc
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected rclone lsjson payload for {remote_path}: {type(payload)!r}")
    return [row for row in payload if isinstance(row, dict)]


def remote_file_size_map(remote_root: Path, *, era5s3_root: Optional[Path] = None) -> Dict[str, int]:
    remote_path = s3_remote_path_for_mount_root(remote_root, era5s3_root=era5s3_root)
    if remote_path:
        return rclone_file_size_map(remote_path)
    return file_size_map(remote_root)


def verify_tree_match(local_root: Path, remote_root: Path, *, era5s3_root: Optional[Path] = None) -> None:
    local_map = file_size_map(local_root)
    remote_map = remote_file_size_map(remote_root, era5s3_root=era5s3_root)
    if not local_map:
        raise RuntimeError(f"Local root has no files: {local_root}")
    if local_map != remote_map:
        remote_path = s3_remote_path_for_mount_root(remote_root, era5s3_root=era5s3_root)
        remote_desc = remote_path if remote_path else str(remote_root)
        raise RuntimeError(
            "Local/remote file listing mismatch.\n"
            f"Local:  {local_root}\n"
            f"Remote: {remote_desc}\n"
            f"Local files={len(local_map)} Remote files={len(remote_map)}"
        )


def prune_empty_parents(path: Path, *, stop_at: Path) -> None:
    current = Path(path).expanduser().resolve(strict=False)
    stop_root = Path(stop_at).expanduser().resolve(strict=False)
    while current != stop_root and stop_root in current.parents:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def safe_rmtree(path: Path, *, allowed_root: Path) -> None:
    resolved = Path(path).expanduser().resolve(strict=False)
    base = Path(allowed_root).expanduser().resolve(strict=False)
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to delete path outside allowed root: {resolved}") from exc
    if not resolved.exists():
        return
    shutil.rmtree(resolved)
    if resolved.exists():
        raise RuntimeError(f"Failed to delete path: {resolved}")
    prune_empty_parents(resolved.parent, stop_at=base)


def copy_tree_files(*, source_root: Path, destination_root: Path) -> None:
    for src in sorted(Path(source_root).rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(source_root)
        dst = Path(destination_root) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue
        shutil.copy2(src, dst)


def build_rclone_upload_cmd(local_tag_root: Path, remote_tag_root: Path, *, era5s3_root: Optional[Path] = None) -> List[str]:
    remote_token = s3_remote_path_for_mount_root(remote_tag_root, era5s3_root=era5s3_root)
    if not remote_token:
        raise RuntimeError(f"Could not map world remote root to object-store path: {remote_tag_root}")
    return [
        "rclone",
        "copy",
        str(Path(local_tag_root).expanduser().resolve(strict=False)),
        str(remote_token),
        "--fast-list",
        "--one-file-system",
        "--s3-force-path-style",
        "--s3-no-check-bucket",
        "--transfers",
        "32",
        "--checkers",
        "64",
        "--s3-upload-concurrency",
        "32",
        "--stats",
        "30s",
    ]


def build_rclone_download_cmd(remote_tag_root: Path, local_tag_root: Path, *, era5s3_root: Optional[Path] = None) -> List[str]:
    remote_token = s3_remote_path_for_mount_root(remote_tag_root, era5s3_root=era5s3_root)
    if not remote_token:
        raise RuntimeError(f"Could not map world remote root to object-store path: {remote_tag_root}")
    return [
        "rclone",
        "copy",
        str(remote_token),
        str(Path(local_tag_root).expanduser().resolve(strict=False)),
        "--fast-list",
        "--s3-force-path-style",
        "--s3-no-check-bucket",
        "--transfers",
        "32",
        "--checkers",
        "64",
        "--stats",
        "30s",
    ]


def _local_dir_names(root: Path) -> List[str]:
    resolved = Path(root).expanduser().resolve(strict=False)
    if not resolved.exists():
        return []
    return sorted(child.name for child in resolved.iterdir() if child.is_dir())


def _remote_dir_names(remote_root: Path, *, era5s3_root: Optional[Path] = None) -> List[str]:
    remote_path = s3_remote_path_for_mount_root(remote_root, era5s3_root=era5s3_root)
    if not remote_path:
        return []
    try:
        rows = _rclone_lsjson_rows(remote_path, dirs_only=True, max_depth=1)
    except RuntimeError:
        return []
    out: List[str] = []
    for row in rows:
        rel = str(row.get("Path") or row.get("Name") or "").strip("/")
        if rel and rel not in out:
            out.append(rel)
    return sorted(out)


def _matching_remote_relpaths(relpaths: Iterable[str], pattern: str) -> List[str]:
    seen: set[str] = set()
    matches: List[str] = []
    for rel in relpaths:
        if fnmatch.fnmatch(rel, pattern) and rel not in seen:
            seen.add(rel)
            matches.append(rel)
    return sorted(matches)


def _validate_world_tag_from_remote_listing(
    spec: Drought755ScenarioSpec,
    *,
    tag: str,
    world_root: Path,
    scale: int = DEFAULT_SCALE,
    era5s3_root: Optional[Path] = None,
) -> Dict[str, Any]:
    output_root = Path(world_root).expanduser().resolve(strict=False)
    tag_root = tag_root_for_world_root(output_root, tag)
    forcing_label = forcing_label_for_spec(spec)
    require_fit = require_fit_for_spec(spec)
    try:
        remote_map = remote_file_size_map(tag_root, era5s3_root=era5s3_root)
    except RuntimeError:
        remote_map = {}
    rel_files = set(remote_map.keys())
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for pet_method in _base758_wrapper.CORE_PET_METHODS:
        manifest_path = _base755._manifest_path(
            output_root,
            output_tag=tag,
            pet_method=pet_method,
            forcing_label=forcing_label,
            scenario_tag=spec.requested_scenario,
            scale=scale,
        )
        manifest_rel = manifest_path.relative_to(tag_root).as_posix()
        manifest_exists = manifest_rel in rel_files
        stacked_matches: List[str] = []
        for label in _base755._stacked_label_candidates(forcing_label, scenario_tag=spec.requested_scenario):
            stacked_dir = _base755._stacked_dir(
                output_root,
                output_tag=tag,
                pet_method=pet_method,
                forcing_label=forcing_label,
                scenario_tag=label,
            )
            pattern = (
                f"{stacked_dir.relative_to(tag_root).as_posix()}/"
                f"{label}__spei{int(scale)}__{_base755.ALLLAND_REGION}__grid__*__all.nc"
            )
            for rel in _matching_remote_relpaths(rel_files, pattern):
                if rel not in stacked_matches:
                    stacked_matches.append(rel)
        fit_matches: List[str] = []
        if require_fit:
            fit_pattern = (
                _base755._pet_root(output_root, output_tag=tag, pet_method=pet_method)
                / "fits"
                / "BASEFIT__*.nc"
            ).relative_to(tag_root).as_posix()
            fit_matches = _matching_remote_relpaths(rel_files, fit_pattern)
        row_missing: List[str] = []
        if not manifest_exists:
            row_missing.append(str(manifest_path))
        if not stacked_matches:
            row_missing.append(
                str(
                    _base755._stacked_dir(
                        output_root,
                        output_tag=tag,
                        pet_method=pet_method,
                        forcing_label=forcing_label,
                        scenario_tag=spec.requested_scenario,
                    )
                )
            )
        if require_fit and not fit_matches:
            row_missing.append(
                str(
                    _base755._pet_root(output_root, output_tag=tag, pet_method=pet_method)
                    / "fits"
                    / "BASEFIT__*.nc"
                )
            )
        missing.extend(row_missing)
        rows.append(
            {
                "pet_method": pet_method,
                "manifest_path": str(manifest_path),
                "manifest_exists": manifest_exists,
                "stacked_count": len(stacked_matches),
                "fit_count": len(fit_matches),
                "complete": not row_missing,
                "missing": row_missing,
            }
        )
    return {
        "complete": not missing,
        "tag": str(tag).strip(),
        "world_root": str(output_root),
        "tag_root": str(tag_root),
        "forcing_label": forcing_label,
        "scenario_tag": spec.requested_scenario,
        "rows": rows,
        "missing": missing,
    }


def validate_world_tag(
    spec: Drought755ScenarioSpec,
    *,
    tag: str,
    world_root: Path,
    scale: int = DEFAULT_SCALE,
    era5s3_root: Optional[Path] = None,
) -> Dict[str, Any]:
    output_root = Path(world_root).expanduser().resolve(strict=False)
    tag_root = tag_root_for_world_root(output_root, tag)
    if s3_remote_path_for_mount_root(tag_root, era5s3_root=era5s3_root):
        return _validate_world_tag_from_remote_listing(
            spec,
            tag=tag,
            world_root=output_root,
            scale=scale,
            era5s3_root=era5s3_root,
        )
    forcing_label = forcing_label_for_spec(spec)
    require_fit = require_fit_for_spec(spec)
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for pet_method in _base758_wrapper.CORE_PET_METHODS:
        manifest_path = _base755._manifest_path(
            output_root,
            output_tag=tag,
            pet_method=pet_method,
            forcing_label=forcing_label,
            scenario_tag=spec.requested_scenario,
            scale=scale,
        )
        stacked_paths = _base755._stacked_glob(
            output_root,
            output_tag=tag,
            pet_method=pet_method,
            forcing_label=forcing_label,
            scenario_tag=spec.requested_scenario,
            scale=scale,
        )
        fit_files: List[Path] = []
        if require_fit:
            fit_files = sorted(
                (_base755._pet_root(output_root, output_tag=tag, pet_method=pet_method) / "fits").glob("BASEFIT__*.nc")
            )
        row_missing: List[str] = []
        if not manifest_path.exists():
            row_missing.append(str(manifest_path))
        if not stacked_paths:
            row_missing.append(
                str(
                    _base755._stacked_dir(
                        output_root,
                        output_tag=tag,
                        pet_method=pet_method,
                        forcing_label=forcing_label,
                        scenario_tag=spec.requested_scenario,
                    )
                )
            )
        if require_fit and not fit_files:
            row_missing.append(
                str(
                    _base755._pet_root(output_root, output_tag=tag, pet_method=pet_method)
                    / "fits"
                    / "BASEFIT__*.nc"
                )
            )
        missing.extend(row_missing)
        rows.append(
            {
                "pet_method": pet_method,
                "manifest_path": str(manifest_path),
                "manifest_exists": bool(manifest_path.exists()),
                "stacked_count": len(stacked_paths),
                "fit_count": len(fit_files),
                "complete": not row_missing,
                "missing": row_missing,
            }
        )
    return {
        "complete": not missing,
        "tag": str(tag).strip(),
        "world_root": str(output_root),
        "tag_root": str(tag_root),
        "forcing_label": forcing_label,
        "scenario_tag": spec.requested_scenario,
        "rows": rows,
        "missing": missing,
    }


def existing_tags(
    world_root: Path,
    *,
    storage: str = STORAGE_LOCAL,
    era5s3_root: Optional[Path] = None,
) -> List[str]:
    if storage == STORAGE_REMOTE:
        return _remote_dir_names(world_root, era5s3_root=era5s3_root)
    return _local_dir_names(world_root)


def inventory_world_root(
    spec: Drought755ScenarioSpec,
    *,
    storage: str,
    world_root: Path,
    tags: Optional[Iterable[str]] = None,
    scale: int = DEFAULT_SCALE,
    era5s3_root: Optional[Path] = None,
) -> Dict[str, Any]:
    tags_to_check = list(tags) if tags is not None else existing_tags(world_root, storage=storage, era5s3_root=era5s3_root)
    rows: List[Dict[str, Any]] = []
    for tag in sorted({str(tag).strip() for tag in tags_to_check if str(tag).strip()}):
        validation = validate_world_tag(spec, tag=tag, world_root=world_root, scale=scale, era5s3_root=era5s3_root)
        rows.append(
            {
                "storage": storage,
                "tag": tag,
                "world_root": str(Path(world_root).expanduser().resolve(strict=False)),
                "tag_root": str(tag_root_for_world_root(world_root, tag)),
                "complete": bool(validation.get("complete")),
                "validation": validation,
            }
        )
    return {
        "spec": asdict(spec),
        "storage": storage,
        "world_root": str(Path(world_root).expanduser().resolve(strict=False)),
        "rows": rows,
        "latest_valid_tag": next((row["tag"] for row in reversed(rows) if bool(row.get("complete"))), None),
    }


def latest_valid_archive_candidate(
    spec: Drought755ScenarioSpec,
    *,
    storage: str,
    localresults_root: Optional[Path] = None,
    era5s3_root: Optional[Path] = None,
    scale: int = DEFAULT_SCALE,
) -> Optional[Drought755ArchiveCandidate]:
    if storage not in STORAGE_CHOICES:
        raise ValueError(f"Unsupported storage '{storage}'. Expected one of: {', '.join(STORAGE_CHOICES)}")
    world_root = (
        local_world_root_for_spec(spec, localresults_root=localresults_root)
        if storage == STORAGE_LOCAL
        else remote_world_root_for_spec(spec, era5s3_root=era5s3_root)
    )
    for tag in reversed(existing_tags(world_root, storage=storage, era5s3_root=era5s3_root)):
        validation = validate_world_tag(spec, tag=tag, world_root=world_root, scale=scale, era5s3_root=era5s3_root)
        if bool(validation.get("complete")):
            return Drought755ArchiveCandidate(
                spec=spec,
                storage=storage,
                tag=tag,
                world_root=world_root,
                tag_root=tag_root_for_world_root(world_root, tag),
                validation=validation,
            )
    return None


def resolve_archive_candidate(
    spec: Drought755ScenarioSpec,
    *,
    explicit_tag: Optional[str] = None,
    prefer_local: bool = True,
    allow_local: bool = True,
    allow_remote: bool = True,
    localresults_root: Optional[Path] = None,
    era5s3_root: Optional[Path] = None,
    scale: int = DEFAULT_SCALE,
) -> Optional[Drought755ArchiveCandidate]:
    local_root = local_world_root_for_spec(spec, localresults_root=localresults_root)
    remote_root = remote_world_root_for_spec(spec, era5s3_root=era5s3_root)
    search_order = [STORAGE_LOCAL, STORAGE_REMOTE] if prefer_local else [STORAGE_REMOTE, STORAGE_LOCAL]
    if not allow_local:
        search_order = [token for token in search_order if token != STORAGE_LOCAL]
    if not allow_remote:
        search_order = [token for token in search_order if token != STORAGE_REMOTE]

    def _candidate_for(storage: str, tag: str) -> Optional[Drought755ArchiveCandidate]:
        world_root = local_root if storage == STORAGE_LOCAL else remote_root
        validation = validate_world_tag(spec, tag=tag, world_root=world_root, scale=scale, era5s3_root=era5s3_root)
        if not bool(validation.get("complete")):
            return None
        return Drought755ArchiveCandidate(
            spec=spec,
            storage=storage,
            tag=tag,
            world_root=world_root,
            tag_root=tag_root_for_world_root(world_root, tag),
            validation=validation,
        )

    if explicit_tag is not None and str(explicit_tag).strip():
        tag = str(explicit_tag).strip()
        for storage in search_order:
            candidate = _candidate_for(storage, tag)
            if candidate is not None:
                return candidate
        return None

    for storage in search_order:
        candidate = latest_valid_archive_candidate(
            spec,
            storage=storage,
            localresults_root=localresults_root,
            era5s3_root=era5s3_root,
            scale=scale,
        )
        if candidate is not None:
            return candidate
    return None


def ensure_archive_uploaded(
    *,
    local_tag_root: Path,
    remote_tag_root: Path,
    runner: Optional[Callable[[Sequence[str]], Any]] = None,
    era5s3_root: Optional[Path] = None,
) -> Dict[str, Any]:
    local_root = Path(local_tag_root).expanduser().resolve(strict=False)
    remote_root = Path(remote_tag_root).expanduser().resolve(strict=False)
    if not local_root.exists():
        raise RuntimeError(f"Local archive root missing: {local_root}")
    try:
        verify_tree_match(local_root, remote_root, era5s3_root=era5s3_root)
    except RuntimeError:
        cmd = build_rclone_upload_cmd(local_root, remote_root, era5s3_root=era5s3_root)
        if runner is None:
            proc = subprocess.run(list(cmd), text=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"Failed to upload local archive to {remote_root}: rc={proc.returncode}")
        else:
            runner(cmd)
        verify_tree_match(local_root, remote_root, era5s3_root=era5s3_root)
        return {
            "uploaded": True,
            "verified": True,
            "local_tag_root": str(local_root),
            "remote_tag_root": str(remote_root),
            "cmd": cmd,
        }
    return {
        "uploaded": False,
        "verified": True,
        "local_tag_root": str(local_root),
        "remote_tag_root": str(remote_root),
        "cmd": build_rclone_upload_cmd(local_root, remote_root, era5s3_root=era5s3_root),
    }


def rehydrate_archive_candidate(
    candidate: Drought755ArchiveCandidate,
    *,
    localresults_root: Optional[Path] = None,
    era5s3_root: Optional[Path] = None,
) -> Path:
    if candidate.storage == STORAGE_LOCAL:
        return Path(candidate.tag_root).expanduser().resolve(strict=False)
    local_world_root = local_world_root_for_spec(candidate.spec, localresults_root=localresults_root)
    local_tag_root = tag_root_for_world_root(local_world_root, candidate.tag)
    remote_token = s3_remote_path_for_mount_root(candidate.tag_root, era5s3_root=era5s3_root)
    if local_tag_root.exists():
        try:
            verify_tree_match(local_tag_root, candidate.tag_root, era5s3_root=era5s3_root)
            return local_tag_root
        except RuntimeError:
            safe_rmtree(local_tag_root, allowed_root=local_world_root)
    if remote_token:
        cmd = build_rclone_download_cmd(candidate.tag_root, local_tag_root, era5s3_root=era5s3_root)
        proc = subprocess.run(list(cmd), text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to rehydrate remote archive from {remote_token}: rc={proc.returncode}")
    else:
        copy_tree_files(source_root=candidate.tag_root, destination_root=local_tag_root)
    verify_tree_match(local_tag_root, candidate.tag_root, era5s3_root=era5s3_root)
    return local_tag_root
