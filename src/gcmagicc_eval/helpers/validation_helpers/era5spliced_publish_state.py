"""Remote publish/completion helpers for ERA5spliced 815 + CMIP6 flows."""

from __future__ import annotations

import configparser
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from .helper_path_utils import (
        get_object_bucket,
        get_object_remote,
        get_rclone_config_path,
        get_s3_storage_options,
    )
except ImportError:  # pragma: no cover - fallback for repo-local script execution
    from scr.validation_helpers.helper_path_utils import (
        get_object_bucket,
        get_object_remote,
        get_rclone_config_path,
        get_s3_storage_options,
    )

PERCENTILES_R2_REMOTE_DEFAULT = "r2"
PERCENTILES_R2_BUCKET_DEFAULT = "gcmagicc-public"
PERCENTILES_R2_PREFIX_DEFAULT = "projection_plots_simple_815/versioned"
PERCENTILES_REMOTE_MANIFEST_PREFIX = "data/scenario_projection_publish_manifests"
PERCENTILES_LOCAL_PUBLISH_DIRNAME = "_publish"
PERCENTILES_LOCAL_PUBLISH_BASENAME = "r2_publish_complete.json"
ENSEMBLES_BASENAME = "ensembles.json"
CMIP6_MEMBERS_BASENAME = "cmip6_members.json"
CMIP6_RUN_COMPLETE_BASENAME = "run_complete.json"
CMIP6_META_SUBDIR = "_meta"
CMIP6_RUN_MANIFEST_BASENAME = "run_manifest.json"
CMIP6_UPLOAD_MANIFEST_BASENAME = "upload_manifest.json"
CMIP6_LOCAL_CLEANUP_BASENAME = "local_cleanup.json"
TIMETAG_RE = re.compile(r"^\d{8}_\d{6}$")
PUBLISH_TIMETAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Unsupported JSON type: {type(obj)!r}")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def rclone_remote_path(remote: str, bucket: str, suffix: str) -> str:
    return f"{remote}:{bucket}/{suffix.strip('/')}"


def _rclone_run(args: Sequence[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["rclone", *args, "--s3-no-check-bucket"],
        text=True,
        capture_output=capture_output,
        check=False,
    )


def rclone_cat_json(remote_path: str) -> Dict[str, Any]:
    result = _rclone_run(["cat", remote_path], capture_output=True)
    if result.returncode != 0:
        return {}
    text = result.stdout.strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def rclone_copyto_json(
    *,
    remote_path: str,
    payload: Dict[str, Any],
    header_upload: Optional[str] = "Cache-Control: public, max-age=300",
) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, default=_json_default)
        handle.write("\n")
    try:
        args: List[str] = ["copyto", str(temp_path), remote_path]
        if header_upload:
            args.extend(["--header-upload", header_upload])
        result = _rclone_run(args, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"rclone copyto failed for {remote_path}: rc={result.returncode} stderr={result.stderr.strip()}"
            )
    finally:
        temp_path.unlink(missing_ok=True)


def rclone_lsf(remote_path: str, *, recursive: bool = False, max_depth: Optional[int] = None) -> List[str]:
    args: List[str] = ["lsf", remote_path, "--files-only"]
    if recursive:
        args.append("--recursive")
    if max_depth is not None:
        args.extend(["--max-depth", str(int(max_depth))])
    result = _rclone_run(args, capture_output=True)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def percentiles_local_publish_manifest_path(output_root: Path) -> Path:
    return output_root / PERCENTILES_LOCAL_PUBLISH_DIRNAME / PERCENTILES_LOCAL_PUBLISH_BASENAME


def percentiles_remote_manifest_key(version: str, scenario: str, run_instance: str) -> str:
    return (
        f"{PERCENTILES_REMOTE_MANIFEST_PREFIX}/"
        f"{str(version).strip()}/{str(scenario).strip()}/{str(run_instance).strip()}.json"
    )


def percentiles_remote_manifest_path(
    *,
    version: str,
    scenario: str,
    run_instance: str,
    remote: str = PERCENTILES_R2_REMOTE_DEFAULT,
    bucket: str = PERCENTILES_R2_BUCKET_DEFAULT,
) -> str:
    return rclone_remote_path(remote, bucket, percentiles_remote_manifest_key(version, scenario, run_instance))


def _first_percentiles_file(output_root: Path) -> Optional[Path]:
    for path in sorted(output_root.rglob("percentiles.json")):
        if path.is_file():
            return path
    return None


def _publish_timetag_from_payload(payload: Dict[str, Any]) -> str:
    candidate = str(payload.get("timetag") or "").strip()
    if candidate and PUBLISH_TIMETAG_RE.match(candidate):
        return candidate
    return ""


def derive_percentiles_publish_timetag(output_root: Path) -> str:
    marker = load_json_file(output_root / "run_complete.json")
    candidate = _publish_timetag_from_payload(marker)
    if candidate:
        return candidate

    sample = _first_percentiles_file(output_root)
    if sample is not None:
        payload = load_json_file(sample)
        candidate = _publish_timetag_from_payload(payload)
        if candidate:
            return candidate

    return datetime.fromtimestamp(
        output_root.stat().st_mtime if output_root.exists() else datetime.now(tz=timezone.utc).timestamp(),
        tz=timezone.utc,
    ).strftime("%Y%m%d_%H%M%S")


def iter_percentiles_files(output_root: Path) -> List[Path]:
    return sorted(path for path in output_root.rglob("percentiles.json") if path.is_file())


def iter_ensembles_files(output_root: Path) -> List[Path]:
    return sorted(path for path in output_root.glob("global/*/annual/ensembles.json") if path.is_file())


def iter_cmip6_sidecar_files(output_root: Path) -> List[Path]:
    return sorted(path for path in output_root.rglob(CMIP6_MEMBERS_BASENAME) if path.is_file())


def global_annual_ensembles_complete(output_root: Path) -> bool:
    percentiles = sorted(path for path in output_root.glob("global/*/annual/percentiles.json") if path.is_file())
    if not percentiles:
        return False
    return all(path.with_name(ENSEMBLES_BASENAME).is_file() for path in percentiles)


def expected_percentiles_remote_keys(
    *,
    output_root: Path,
    version: str,
    scenario: str,
    publish_timetag: str,
    prefix_root: str = PERCENTILES_R2_PREFIX_DEFAULT,
) -> List[str]:
    keys: List[str] = []
    for path in iter_percentiles_files(output_root):
        rel = path.relative_to(output_root)
        if len(rel.parts) != 4:
            continue
        storage_region, variable, season, filename = rel.parts
        if filename != "percentiles.json":
            continue
        keys.append(
            "/".join(
                [
                    prefix_root.strip("/"),
                    str(version).strip(),
                    publish_timetag,
                    variable,
                    season,
                    storage_region,
                    str(scenario).strip(),
                    filename,
                ]
            )
        )
    for path in iter_ensembles_files(output_root):
        rel = path.relative_to(output_root)
        if len(rel.parts) != 4:
            continue
        storage_region, variable, season, filename = rel.parts
        if filename != ENSEMBLES_BASENAME:
            continue
        keys.append(
            "/".join(
                [
                    prefix_root.strip("/"),
                    str(version).strip(),
                    publish_timetag,
                    variable,
                    season,
                    storage_region,
                    str(scenario).strip(),
                    filename,
                ]
            )
        )
    for path in iter_cmip6_sidecar_files(output_root):
        rel = path.relative_to(output_root)
        if len(rel.parts) != 4:
            continue
        storage_region, variable, season, filename = rel.parts
        if filename != CMIP6_MEMBERS_BASENAME:
            continue
        keys.append(
            "/".join(
                [
                    prefix_root.strip("/"),
                    str(version).strip(),
                    publish_timetag,
                    variable,
                    season,
                    storage_region,
                    str(scenario).strip(),
                    filename,
                ]
            )
        )
    return sorted(dict.fromkeys(keys))


def stable_key_hash(keys: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(str(item).strip() for item in keys if str(item).strip()):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_percentiles_publish_manifest_payload(
    *,
    output_root: Path,
    version: str,
    scenario: str,
    run_instance: str,
    publish_timetag: str,
    prefix_root: str,
    shard_paths_touched: Sequence[str],
    remote: str,
    bucket: str,
) -> Dict[str, Any]:
    remote_keys = expected_percentiles_remote_keys(
        output_root=output_root,
        version=version,
        scenario=scenario,
        publish_timetag=publish_timetag,
        prefix_root=prefix_root,
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "scenario": scenario,
        "run_instance": run_instance,
        "output_root": str(output_root),
        "publish_timetag": publish_timetag,
        "r2_remote": remote,
        "r2_bucket": bucket,
        "r2_prefix_root": prefix_root,
        "remote_manifest_key": percentiles_remote_manifest_key(version, scenario, run_instance),
        "percentiles_file_count": len(iter_percentiles_files(output_root)),
        "ensembles_file_count": len(iter_ensembles_files(output_root)),
        "cmip6_members_file_count": len(iter_cmip6_sidecar_files(output_root)),
        "projection_file_count": len(remote_keys),
        "remote_object_key_hash": stable_key_hash(remote_keys),
        "remote_object_key_samples": remote_keys[:3] + remote_keys[-3:] if len(remote_keys) > 6 else remote_keys,
        "catalog_shard_paths_touched": sorted(dict.fromkeys(str(item).strip() for item in shard_paths_touched if str(item).strip())),
    }


def remote_percentiles_manifest_matches(
    payload: Dict[str, Any],
    *,
    version: str,
    scenario: str,
    run_instance: str,
    output_root: Optional[Path] = None,
    publish_timetag: Optional[str] = None,
    prefix_root: str = PERCENTILES_R2_PREFIX_DEFAULT,
) -> bool:
    if not payload:
        return False
    if str(payload.get("version") or "").strip() != str(version).strip():
        return False
    if str(payload.get("scenario") or "").strip() != str(scenario).strip():
        return False
    if str(payload.get("run_instance") or "").strip() != str(run_instance).strip():
        return False
    if publish_timetag is not None and str(payload.get("publish_timetag") or "").strip() != str(publish_timetag).strip():
        return False
    if output_root is None:
        return True
    expected_keys = expected_percentiles_remote_keys(
        output_root=output_root,
        version=version,
        scenario=scenario,
        publish_timetag=publish_timetag or derive_percentiles_publish_timetag(output_root),
        prefix_root=prefix_root,
    )
    expected_percentiles_count = len(iter_percentiles_files(output_root))
    expected_ensembles_count = len(iter_ensembles_files(output_root))
    expected_cmip6_count = len(iter_cmip6_sidecar_files(output_root))
    return (
        int(payload.get("percentiles_file_count") or -1) == expected_percentiles_count
        and int(payload.get("ensembles_file_count") or 0) == expected_ensembles_count
        and int(payload.get("cmip6_members_file_count") or 0) == expected_cmip6_count
        and str(payload.get("remote_object_key_hash") or "").strip() == stable_key_hash(expected_keys)
    )


def verify_percentiles_remote_listing(
    *,
    output_root: Path,
    version: str,
    scenario: str,
    publish_timetag: str,
    remote: str = PERCENTILES_R2_REMOTE_DEFAULT,
    bucket: str = PERCENTILES_R2_BUCKET_DEFAULT,
    prefix_root: str = PERCENTILES_R2_PREFIX_DEFAULT,
) -> bool:
    base_key = "/".join([prefix_root.strip("/"), str(version).strip(), publish_timetag])
    listing = set(rclone_lsf(rclone_remote_path(remote, bucket, base_key), recursive=True))
    if not listing:
        return False
    base_prefix = f"{base_key.strip('/')}/"
    expected_relpaths = []
    for key in expected_percentiles_remote_keys(
        output_root=output_root,
        version=version,
        scenario=scenario,
        publish_timetag=publish_timetag,
        prefix_root=prefix_root,
    ):
        clean_key = str(key).strip("/")
        rel_path = clean_key[len(base_prefix) :] if clean_key.startswith(base_prefix) else clean_key
        expected_relpaths.append(rel_path)
    return all(rel_path in listing for rel_path in expected_relpaths)


def legacy_percentiles_remote_complete(
    *,
    output_root: Path,
    version: str,
    scenario: str,
    run_instance: str,
    remote: str = PERCENTILES_R2_REMOTE_DEFAULT,
    bucket: str = PERCENTILES_R2_BUCKET_DEFAULT,
    prefix_root: str = PERCENTILES_R2_PREFIX_DEFAULT,
) -> Optional[Dict[str, Any]]:
    publish_timetag = derive_percentiles_publish_timetag(output_root)
    remote_index = rclone_cat_json(rclone_remote_path(remote, bucket, "data/scenario_projection_catalog.json"))
    versions = remote_index.get("versions") if isinstance(remote_index, dict) else []
    version_entry = None
    if isinstance(versions, list):
        for row in versions:
            if not isinstance(row, dict):
                continue
            if str(row.get("id") or "").strip() == str(version).strip():
                version_entry = row
                break
    if version_entry is None:
        return None
    scenario_ids = set(str(item).strip() for item in (version_entry.get("scenario_ids") or []) if str(item).strip())
    if scenario not in scenario_ids:
        return None
    if not verify_percentiles_remote_listing(
        output_root=output_root,
        version=version,
        scenario=scenario,
        publish_timetag=publish_timetag,
        remote=remote,
        bucket=bucket,
        prefix_root=prefix_root,
    ):
        return None
    shard_paths = [
        str(row.get("path") or "").strip()
        for row in (version_entry.get("shards") or [])
        if isinstance(row, dict) and str(row.get("path") or "").strip()
    ]
    return build_percentiles_publish_manifest_payload(
        output_root=output_root,
        version=version,
        scenario=scenario,
        run_instance=run_instance,
        publish_timetag=publish_timetag,
        prefix_root=prefix_root,
        shard_paths_touched=shard_paths,
        remote=remote,
        bucket=bucket,
    )


def _build_boto3_client():
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("boto3 and botocore are required for CMIP6 remote state checks") from exc

    opts = dict(get_s3_storage_options())
    if opts.get("anon"):
        raise RuntimeError("Anonymous S3 access is not supported for CMIP6 remote state checks.")

    client_kwargs: Dict[str, Any] = dict(opts.get("client_kwargs") or {})
    config_kwargs = dict(opts.get("config_kwargs") or {})
    if config_kwargs:
        client_kwargs["config"] = Config(**config_kwargs)

    auth_kwargs: Dict[str, Any] = {}
    if opts.get("key"):
        auth_kwargs["aws_access_key_id"] = opts["key"]
    if opts.get("secret"):
        auth_kwargs["aws_secret_access_key"] = opts["secret"]
    if opts.get("token"):
        auth_kwargs["aws_session_token"] = opts["token"]
    return boto3.client("s3", **client_kwargs, **auth_kwargs)


def cmip6_remote_marker_keys(remote_run_prefix: str) -> Dict[str, str]:
    prefix = remote_run_prefix.strip("/")
    return {
        "run_complete": f"{prefix}/{CMIP6_RUN_COMPLETE_BASENAME}",
        "run_manifest": f"{prefix}/{CMIP6_META_SUBDIR}/{CMIP6_RUN_MANIFEST_BASENAME}",
        "upload_manifest": f"{prefix}/{CMIP6_META_SUBDIR}/{CMIP6_UPLOAD_MANIFEST_BASENAME}",
        "local_cleanup": f"{prefix}/{CMIP6_META_SUBDIR}/{CMIP6_LOCAL_CLEANUP_BASENAME}",
    }


def _s3_head_exists(client: Any, *, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
    except Exception:
        return False
    return True


def classify_cmip6_remote_state(
    *,
    remote_run_prefix: str,
    bucket: Optional[str] = None,
) -> Dict[str, Any]:
    client = _build_boto3_client()
    resolved_bucket = str(bucket or get_object_bucket()).strip()
    keys = cmip6_remote_marker_keys(remote_run_prefix)
    marker_hits = {
        name: _s3_head_exists(client, bucket=resolved_bucket, key=key)
        for name, key in keys.items()
    }
    if all(marker_hits.values()):
        return {
            "status": "done_remote_only",
            "source": "remote run_complete.json + remote _meta manifests",
            "bucket": resolved_bucket,
            "remote_run_prefix": remote_run_prefix,
            "markers": marker_hits,
        }

    paginator = client.get_paginator("list_objects_v2")
    has_any_objects = False
    for page in paginator.paginate(Bucket=resolved_bucket, Prefix=remote_run_prefix.strip("/") + "/", PaginationConfig={"MaxItems": 1, "PageSize": 1}):
        if page.get("Contents"):
            has_any_objects = True
            break

    if has_any_objects or any(marker_hits.values()):
        return {
            "status": "partial_remote",
            "source": "remote prefix exists without full completion markers",
            "bucket": resolved_bucket,
            "remote_run_prefix": remote_run_prefix,
            "markers": marker_hits,
        }

    return {
        "status": "missing",
        "source": "",
        "bucket": resolved_bucket,
        "remote_run_prefix": remote_run_prefix,
        "markers": marker_hits,
    }


def delete_s3_prefix(*, remote_run_prefix: str, bucket: Optional[str] = None) -> int:
    client = _build_boto3_client()
    resolved_bucket = str(bucket or get_object_bucket()).strip()
    paginator = client.get_paginator("list_objects_v2")
    deleted = 0
    chunk: List[Dict[str, str]] = []
    for page in paginator.paginate(Bucket=resolved_bucket, Prefix=remote_run_prefix.strip("/") + "/"):
        for row in page.get("Contents") or []:
            key = str(row.get("Key") or "").strip()
            if not key:
                continue
            chunk.append({"Key": key})
            if len(chunk) >= 1000:
                client.delete_objects(Bucket=resolved_bucket, Delete={"Objects": chunk})
                deleted += len(chunk)
                chunk = []
    if chunk:
        client.delete_objects(Bucket=resolved_bucket, Delete={"Objects": chunk})
        deleted += len(chunk)
    return deleted


def object_remote_available(remote: Optional[str] = None) -> bool:
    remote_name = str(remote or get_object_remote()).strip()
    cfg_path = get_rclone_config_path()
    if not cfg_path.exists():
        return False
    parser = configparser.ConfigParser()
    try:
        parser.read(cfg_path, encoding="utf-8")
    except Exception:
        return False
    return parser.has_section(remote_name)
