"""
Shared xarray backend selection for mixed local + mounted storage setups.

The main use-case is avoiding unstable netCDF4/xattr behavior on FUSE-backed
object mounts while still allowing fast local backends for physical disks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class _MountInfo:
    mount_point: str
    fs_type: str
    source: str


_REMOTE_FS_HINTS = {
    "s3fs",
    "goofys",
    "rclonefs",
    "gcsfuse",
    "fuse.s3fs",
    "fuse.goofys",
    "fuse.rclone",
    "fuse.gcsfuse",
}
_REMOTE_SOURCE_HINTS = ("rclone", "s3", "goofys", "s3fs", "gcs")
_SCOPE_CHOICES = {"all", "scratch", "mount", "auto"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _norm_engine(value: object) -> str:
    token = str(value or "").strip().lower()
    if token in {"", "auto", "default", "none"}:
        return ""
    return token


def _decode_mountinfo_token(token: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(token):
        if token[i] == "\\" and i + 3 < len(token) and token[i + 1 : i + 4].isdigit():
            try:
                out.append(chr(int(token[i + 1 : i + 4], 8)))
                i += 4
                continue
            except Exception:
                pass
        out.append(token[i])
        i += 1
    return "".join(out)


def _load_mountinfo() -> list[_MountInfo]:
    entries: list[_MountInfo] = []
    try:
        with open("/proc/self/mountinfo", "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if " - " not in line:
                    continue
                left, right = line.split(" - ", 1)
                left_parts = left.split()
                right_parts = right.split()
                if len(left_parts) < 5 or not right_parts:
                    continue
                mount_point = _decode_mountinfo_token(left_parts[4]).rstrip("/") or "/"
                fs_type = str(right_parts[0]).strip().lower()
                source = str(right_parts[1]).strip().lower() if len(right_parts) > 1 else ""
                entries.append(_MountInfo(mount_point=mount_point, fs_type=fs_type, source=source))
    except Exception:
        return []

    entries.sort(key=lambda item: len(item.mount_point), reverse=True)
    return entries


def _iter_prefixes(raw: str) -> Iterable[str]:
    for token in raw.split(os.pathsep):
        token = str(token).strip()
        if not token:
            continue
        try:
            norm = str(Path(token).expanduser().resolve(strict=False))
        except Exception:
            norm = os.path.abspath(os.path.expanduser(token))
        yield norm.rstrip("/") or "/"


def _path_matches_prefix(path_value: str, prefix: str) -> bool:
    if prefix == "/":
        return True
    return path_value == prefix or path_value.startswith(prefix + os.sep)


def _sample_path(path_obj: object) -> Optional[str]:
    sample = path_obj
    if isinstance(sample, (list, tuple)):
        if not sample:
            return None
        sample = sample[0]
    if isinstance(sample, dict):
        for key in ("file", "path", "filename"):
            if key in sample:
                sample = sample[key]
                break
    try:
        candidate = os.fspath(sample)  # type: ignore[arg-type]
    except Exception:
        return None
    path_str = str(candidate).strip()
    if not path_str:
        return None
    if path_str.startswith("file://"):
        path_str = path_str[7:]
    elif "://" in path_str:
        return None
    return os.path.abspath(os.path.expanduser(path_str))


def _best_mount_for_path(path_value: str, mountinfo: Sequence[_MountInfo]) -> Optional[_MountInfo]:
    for entry in mountinfo:
        if _path_matches_prefix(path_value, entry.mount_point):
            return entry
    return None


@dataclass(frozen=True)
class _Policy:
    mount_engine: str
    local_engine: str
    scope: str
    fallback: bool
    fallback_local_only: bool
    mount_prefixes: Tuple[str, ...]
    local_prefixes: Tuple[str, ...]
    mountinfo: Tuple[_MountInfo, ...]

    def path_is_remote_mount(self, path_obj: object) -> bool:
        path_value = _sample_path(path_obj)
        if not path_value:
            return False
        for prefix in self.local_prefixes:
            if _path_matches_prefix(path_value, prefix):
                return False
        for prefix in self.mount_prefixes:
            if _path_matches_prefix(path_value, prefix):
                return True
        mount = _best_mount_for_path(path_value, self.mountinfo)
        if mount is None:
            return False
        fs_type = mount.fs_type
        if fs_type.startswith("fuse") or fs_type in _REMOTE_FS_HINTS:
            return True
        source = mount.source
        return any(token in source for token in _REMOTE_SOURCE_HINTS)

    def should_use_mount_engine(self, path_obj: object) -> bool:
        path_value = _sample_path(path_obj)
        if self.scope == "all":
            return True
        if self.scope == "scratch":
            return bool(path_value and _path_matches_prefix(path_value, "data/site_gus"))
        if self.scope == "mount":
            return self.path_is_remote_mount(path_obj)
        # auto
        return self.path_is_remote_mount(path_obj)

    def choose_engine(self, path_obj: object, requested_engine: object) -> str:
        requested = _norm_engine(requested_engine)
        # Respect explicit non-netcdf4 requests.
        if requested and requested != "netcdf4":
            return requested
        if self.should_use_mount_engine(path_obj):
            return self.mount_engine or requested
        return self.local_engine or requested


def _build_policy(
    *,
    mount_engine_default: str,
    local_engine_default: str,
    scope_default: str,
    fallback_default: bool,
    fallback_local_only_default: bool,
) -> _Policy:
    mount_engine = _norm_engine(os.environ.get("GCMAGICC_XARRAY_ENGINE_OVERRIDE")) or _norm_engine(
        os.environ.get("GCMAGICC_XARRAY_ENGINE")
    )
    if not mount_engine:
        mount_engine = _norm_engine(mount_engine_default)
    local_engine = _norm_engine(os.environ.get("GCMAGICC_XARRAY_LOCAL_ENGINE")) or _norm_engine(
        local_engine_default
    )
    scope = str(os.environ.get("GCMAGICC_XARRAY_ENGINE_SCOPE", scope_default)).strip().lower()
    if scope not in _SCOPE_CHOICES:
        scope = scope_default if scope_default in _SCOPE_CHOICES else "auto"
    fallback = _env_bool("GCMAGICC_XARRAY_ENGINE_FALLBACK", fallback_default)
    fallback_local_only = _env_bool(
        "GCMAGICC_XARRAY_ENGINE_FALLBACK_LOCAL_ONLY", fallback_local_only_default
    )
    mount_prefixes = tuple(_iter_prefixes(os.environ.get("GCMAGICC_XARRAY_MOUNT_PREFIXES", "")))
    local_prefixes = tuple(_iter_prefixes(os.environ.get("GCMAGICC_XARRAY_LOCAL_PREFIXES", "")))
    return _Policy(
        mount_engine=mount_engine,
        local_engine=local_engine,
        scope=scope,
        fallback=fallback,
        fallback_local_only=fallback_local_only,
        mount_prefixes=mount_prefixes,
        local_prefixes=local_prefixes,
        mountinfo=tuple(_load_mountinfo()),
    )


def configure_mount_aware_xarray_defaults(
    *,
    mount_engine_default: str = "h5netcdf",
    local_engine_default: str = "netcdf4",
    scope_default: str = "auto",
    fallback_default: bool = True,
    fallback_local_only_default: bool = True,
) -> None:
    """
    Set conservative default env knobs without overriding user-provided values.
    """

    legacy_engine = _norm_engine(os.environ.get("GCMAGICC_XARRAY_ENGINE"))
    if legacy_engine and legacy_engine != "auto":
        os.environ.setdefault("GCMAGICC_XARRAY_ENGINE_OVERRIDE", legacy_engine)
        # Historical behavior of GCMAGICC_XARRAY_ENGINE was effectively "force all paths".
        os.environ.setdefault("GCMAGICC_XARRAY_ENGINE_SCOPE", "all")
    else:
        os.environ.setdefault("GCMAGICC_XARRAY_ENGINE_OVERRIDE", mount_engine_default)
        os.environ.setdefault("GCMAGICC_XARRAY_ENGINE_SCOPE", scope_default)

    os.environ.setdefault("GCMAGICC_XARRAY_LOCAL_ENGINE", local_engine_default)
    os.environ.setdefault(
        "GCMAGICC_XARRAY_ENGINE_FALLBACK", "1" if fallback_default else "0"
    )
    os.environ.setdefault(
        "GCMAGICC_XARRAY_ENGINE_FALLBACK_LOCAL_ONLY",
        "1" if fallback_local_only_default else "0",
    )

    if "GCMAGICC_XARRAY_MOUNT_PREFIXES" not in os.environ:
        prefixes: list[str] = []
        if _env_bool("GCMAGICC_XARRAY_ASSUME_DATA_SCRATCH_REMOTE", False):
            prefixes.append("data/site_gus")
        object_mount = os.environ.get("GCMAGICC_OBJECT_MOUNT", "").strip()
        if object_mount:
            prefixes.append(object_mount)
        if prefixes:
            os.environ["GCMAGICC_XARRAY_MOUNT_PREFIXES"] = os.pathsep.join(prefixes)


def install_xarray_engine_preference(
    *,
    mount_engine_default: str = "h5netcdf",
    local_engine_default: str = "netcdf4",
    scope_default: str = "auto",
    fallback_default: bool = True,
    fallback_local_only_default: bool = True,
) -> None:
    """
    Patch xarray open functions so engine selection is path-aware.
    """

    try:
        import xarray as xr  # type: ignore
    except Exception:
        return

    if getattr(xr, "_gcmagicc_engine_pref_installed", False):
        return

    policy = _build_policy(
        mount_engine_default=mount_engine_default,
        local_engine_default=local_engine_default,
        scope_default=scope_default,
        fallback_default=fallback_default,
        fallback_local_only_default=fallback_local_only_default,
    )

    original_open_dataset = xr.open_dataset
    original_open_mfdataset = getattr(xr, "open_mfdataset", None)

    def _run_with_policy(opener, args, kwargs, path_obj):
        requested_engine = kwargs.get("engine")
        chosen_engine = policy.choose_engine(path_obj, requested_engine)
        local_kwargs = dict(kwargs)
        if chosen_engine:
            local_kwargs["engine"] = chosen_engine
        try:
            return opener(*args, **local_kwargs)
        except Exception:
            if not policy.fallback:
                raise
            if policy.fallback_local_only and policy.path_is_remote_mount(path_obj):
                raise
            fallback_kwargs = dict(kwargs)
            if policy.local_engine and policy.local_engine != chosen_engine:
                fallback_kwargs["engine"] = policy.local_engine
            else:
                fallback_kwargs.pop("engine", None)
            return opener(*args, **fallback_kwargs)

    def _patched_open_dataset(*args, **kwargs):
        path_obj = args[0] if args else kwargs.get("filename_or_obj")
        return _run_with_policy(original_open_dataset, args, kwargs, path_obj)

    xr.open_dataset = _patched_open_dataset  # type: ignore[assignment]

    if original_open_mfdataset is not None:

        def _patched_open_mfdataset(*args, **kwargs):
            path_obj = args[0] if args else kwargs.get("paths")
            return _run_with_policy(original_open_mfdataset, args, kwargs, path_obj)

        xr.open_mfdataset = _patched_open_mfdataset  # type: ignore[assignment]

    setattr(xr, "_gcmagicc_engine_pref_installed", True)
    setattr(
        xr,
        "_gcmagicc_engine_pref_summary",
        {
            "mount_engine": policy.mount_engine,
            "local_engine": policy.local_engine,
            "scope": policy.scope,
            "fallback": bool(policy.fallback),
            "fallback_local_only": bool(policy.fallback_local_only),
            "mount_prefixes": list(policy.mount_prefixes),
            "local_prefixes": list(policy.local_prefixes),
        },
    )
