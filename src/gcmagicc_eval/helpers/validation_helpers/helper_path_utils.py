"""
Path utilities for ETH/GUS dual-run deployments.

This module keeps script defaults portable by resolving paths from a small
environment contract and safe auto-detection fallbacks.
"""

from __future__ import annotations

import configparser
import importlib.util
import json
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

ETH_PROJECTS_ROOT = Path("data/site_eth/projects")
GUS_PROJECTS_ROOT = Path("data/site_gus/projects")
ETH_DATA_ROOT = Path("data/site_eth")
GUS_DATA_ROOT = Path("data/site_gus")
FRESSNAPF_DATA_ROOT = Path("data/archive")
# --- Canonical consolidated archive (2026-07 consolidation). On this (gus) host BOTH
# the fressnapf and gus/site_local profiles resolve here; ETH_* left untouched. See
# ~/.claude/plans/can-i-ask-you-vast-raven.md (Old->canonical mapping). ---
FRESSNAPF_GCMAGICC_ROOT = FRESSNAPF_DATA_ROOT / "gcmagicc"
_CANON_ERA5SPLICED_ALLVARS = FRESSNAPF_GCMAGICC_ROOT / "ERA5spliced" / "all-vars-one-file"
_CANON_ERA5SPLICED_ONEVAR = FRESSNAPF_GCMAGICC_ROOT / "ERA5spliced" / "one-var-one-file"
_CANON_CMIP6REPLICAS = FRESSNAPF_GCMAGICC_ROOT / "CMIP6replicas"
_CANON_SCRATCH = FRESSNAPF_GCMAGICC_ROOT / "_scratch"
_CANON_CATALOG = FRESSNAPF_GCMAGICC_ROOT / "_runmeta" / "catalog" / "latest.json"
ETH_GCMAGICC_ARCHIVE_ROOT = ETH_PROJECTS_ROOT / "gcmmagicc" / "data" / "GCMagicc"
GUS_GCMAGICC_ARCHIVE_ROOT = GUS_PROJECTS_ROOT / "gcmmagicc" / "data" / "GCMagicc"
LEGACY_GUS_GCMAGICC_ARCHIVE_ROOT = Path("data/site_gus/legacy/gcmmagicc/data/GCMagicc")
ETH_CREATED_NC_ROOT = ETH_PROJECTS_ROOT / "gcmagicc_ensemble_runner" / "created_nc_files"
GUS_CREATED_NC_ROOT = GUS_PROJECTS_ROOT / "gcmagicc_ensemble_runner" / "created_nc_files"
LEGACY_GUS_CREATED_NC_ROOT = Path("data/site_gus/legacy/gcmagicc_ensemble_runner/created_nc_files")

SITE_ETH = "eth"
SITE_GUS = "gus"
SITE_VALUES = {SITE_ETH, SITE_GUS}
DATA_PROFILE_ENV = "GCMAGICC_DATA_PROFILE"
DATA_PROFILE_FRESSNAPF = "fressnapf"

DEFAULT_VERSION_BY_SITE = {
    SITE_ETH: "v101",
    SITE_GUS: "v100",
}
KNOWN_VERSION_HOME_SITE = {
    "v100": SITE_GUS,
    "v100gxe": SITE_GUS,
    "v101": SITE_ETH,
    "v101gxe": SITE_ETH,
}
DEFAULT_ARCHIVE_VERSIONS = ("v100", "v100gxe", "v101", "v101gxe")
_VERSION_ENV_SANITIZE_RE = re.compile(r"[^0-9A-Za-z]+")

ETH_ERA5SPLICED_S3_ROOT = Path("data/site_eth/GCMAGICCoutput/ERA5splicedS3")
GUS_ERA5SPLICED_S3_ROOT = _CANON_ERA5SPLICED_ALLVARS  # was data/site_gus ERA5splicedS3 mount (retired)
FRESSNAPF_ERA5SPLICED_ROOT = _CANON_ERA5SPLICED_ALLVARS
ETH_ERA5SPLICED_LOCALSTAGING_ROOT = Path(
    "data/site_eth/GCMAGICCoutput/ERA5spliced_localstaging"
)
GUS_ERA5SPLICED_LOCALSTAGING_ROOT = _CANON_SCRATCH  # staging tier retired -> writable scratch
ETH_ERA5SPLICED_LOCALRESULTS_ROOT = Path(
    "data/site_eth/GCMAGICCoutput/ERA5spliced_localresults"
)
GUS_ERA5SPLICED_LOCALRESULTS_ROOT = _CANON_ERA5SPLICED_ALLVARS  # dataderivatives live under all-vars-one-file
ETH_ERA5SPLICED_CMIP6_LOCALRESULTS_ROOT = Path(
    "data/site_eth/GCMAGICCoutput/ERA5spliced_cmip6_localresults"
)
GUS_ERA5SPLICED_CMIP6_LOCALRESULTS_ROOT = _CANON_ERA5SPLICED_ONEVAR
ETH_ERA5SPLICED_CMIP7_LOCALRESULTS_ROOT = Path(
    "data/site_eth/GCMAGICCoutput/ERA5spliced_cmip7_localresults"
)
GUS_ERA5SPLICED_CMIP7_LOCALRESULTS_ROOT = _CANON_ERA5SPLICED_ONEVAR
ETH_CMIP6REPLICAS_ROOT = Path("data/site_eth/GCMAGICCoutput/CMIP6replicas")
GUS_CMIP6REPLICAS_ROOT = _CANON_CMIP6REPLICAS
FRESSNAPF_CMIP6REPLICAS_ROOT = _CANON_CMIP6REPLICAS
ETH_CMIP6REPLICAS_LOCALSTAGING_ROOT = Path(
    "data/site_eth/GCMAGICCoutput/CMIP6replicas_localstaging"
)
GUS_CMIP6REPLICAS_LOCALSTAGING_ROOT = _CANON_SCRATCH
ETH_CMIP6_LOCALSTAGING_ROOT = Path(
    "data/site_eth/out_ETHFOG_10June2025_vetted_localstaging"
)
GUS_CMIP6_LOCALSTAGING_ROOT = Path(
    "data/site_gus/cmip6_ETHFOG/processed/out_ETHFOG_10June2025_vetted_localstaging"
)
ETH_ERA5SPLICED_LEGACY_ROOT = Path("data/site_eth/GCMAGICCoutput/ERA5spliced")
GUS_ERA5SPLICED_LEGACY_ROOT = Path("data/site_gus/GCMAGICCoutput/ERA5spliced")
ETH_ERA5SPLICED_CATALOG = Path("data/site_eth/GCMAGICCoutput/ERA5splicedS3_catalog/latest.json")
GUS_ERA5SPLICED_CATALOG = _CANON_CATALOG
LEGACY_FALLBACK_ENV = "GCMAGICC_ENABLE_LEGACY_FALLBACK"
ETH_ERA5SPLICED_ROOT = ETH_ERA5SPLICED_S3_ROOT
GUS_ERA5SPLICED_ROOT = GUS_ERA5SPLICED_S3_ROOT
FRESSNAPF_CMIP6_ROOT = FRESSNAPF_DATA_ROOT / "CMIP6"
FRESSNAPF_ERA5_ROOT = FRESSNAPF_DATA_ROOT / "ERA5"
FRESSNAPF_CMIP6_VETTED_ROOT = (
    FRESSNAPF_CMIP6_ROOT / "ETHFOG" / "processed" / "out_ETHFOG_10June2025_vetted"
)
FRESSNAPF_CMIP6_VETTED_NFS_COPY_ROOT = (
    FRESSNAPF_CMIP6_ROOT
    / "ETHFOG"
    / "processed"
    / "out_ETHFOG_10June2025_vetted_nfs_copy_before_s3mount_20260428"
)
FRESSNAPF_ERA5_VETTED_ROOT = (
    FRESSNAPF_ERA5_ROOT
    / "processed"
    / "monthly"
    / "multivar_1x1"
    / "out_ERA5_19Feb2026_1degree_vetted"
)
FRESSNAPF_ERA5_025_VETTED_ROOT = (
    FRESSNAPF_ERA5_ROOT
    / "processed"
    / "monthly"
    / "multivar_025x025"
    / "out_ERA5_5July2025_025degree_vetted"
)
FRESSNAPF_CMIP6_HEALPIX_ROOT = FRESSNAPF_CMIP6_ROOT / "healpix"
FRESSNAPF_ERA5_HEALPIX_ROOT = FRESSNAPF_ERA5_ROOT / "healpix"

DATA_PROFILE_SITE_LOCAL = "site_local"
DATA_PROFILE_VALUES = {
    DATA_PROFILE_SITE_LOCAL,
    DATA_PROFILE_FRESSNAPF,
}

RUNMODUS_ALL = "all"
RUNMODUS_NAT = "nat"
RUNMODUS_AER = "aer"
RUNMODUS_CHOICES = (RUNMODUS_ALL, RUNMODUS_NAT, RUNMODUS_AER)

CANONICAL_KIND_ORIGINAL = "original"
CANONICAL_KIND_ORIGINAL_HEALPIX = "original_healpix"
CANONICAL_KIND_DATADERIVATIVES = "dataderivatives"
CANONICAL_KIND_CHOICES = (
    CANONICAL_KIND_ORIGINAL,
    CANONICAL_KIND_ORIGINAL_HEALPIX,
    CANONICAL_KIND_DATADERIVATIVES,
)

STORAGE_ACCESS_MOUNT = "mount"
STORAGE_ACCESS_S3_DIRECT = "s3_direct"
STORAGE_ACCESS_RCLONE_CACHE = "rclone_cache"
STORAGE_ACCESS_CHOICES = (
    STORAGE_ACCESS_MOUNT,
    STORAGE_ACCESS_S3_DIRECT,
    STORAGE_ACCESS_RCLONE_CACHE,
)

DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE = "parallel_run_tree"
DERIVATIVES_LAYOUT_INPLACE = "inplace"
DERIVATIVES_LAYOUT_CHOICES = (
    DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE,
    DERIVATIVES_LAYOUT_INPLACE,
)
DEFAULT_DERIVATIVES_RUN_SUFFIX = "_dataderivatives"
DERIVATIVES_KIND_DATA = "data_derivatives"
DERIVATIVES_KIND_ARCHIVE = "data_derivatives_archive"
DERIVATIVES_KIND_CHOICES = (
    DERIVATIVES_KIND_DATA,
    DERIVATIVES_KIND_ARCHIVE,
)

RCLONE_MOUNT_FSTYPE = "fuse.rclone"
DEFAULT_RCLONE_REMOTE = "BackupStorageS3"


def _norm_site(value: str | None) -> str | None:
    if not value:
        return None
    site = value.strip().lower()
    if site in SITE_VALUES:
        return site
    return None


def _norm_data_profile(value: str | None) -> str | None:
    if not value:
        return None
    token = str(value).strip().lower()
    if token in DATA_PROFILE_VALUES:
        return token
    return None


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve(strict=False)


def _looks_like_repo_root(path: Path) -> bool:
    return (path / "scr").exists() and (path / "notebooks").exists()


def _detect_repo_root() -> Path:
    env_root = _env_path("GCMAGICC_REPO_ROOT")
    if env_root:
        return env_root

    here = Path(__file__).resolve()
    for candidate in [here.parents[2], *here.parents]:
        if _looks_like_repo_root(candidate):
            return candidate

    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if _looks_like_repo_root(candidate):
            return candidate

    return current


def _path_under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _safe_exists_status(path: Path) -> tuple[bool, bool]:
    try:
        return path.exists(), True
    except OSError:
        return False, False


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_iterdir(path: Path) -> list[Path]:
    try:
        return list(path.iterdir())
    except OSError:
        return []


def _safe_glob(path: Path, pattern: str) -> list[Path]:
    try:
        return list(path.glob(pattern))
    except OSError:
        return []


def _detect_site_from_path(path: Path) -> str | None:
    if _path_under(path, ETH_PROJECTS_ROOT):
        return SITE_ETH
    if _path_under(path, GUS_PROJECTS_ROOT):
        return SITE_GUS
    return None


def get_project_root() -> Path:
    """Return repository root (contains `scr/` and `notebooks/`)."""
    return _detect_repo_root()


def get_projects_root() -> Path:
    """
    Return the parent directory that contains sibling repos
    (`gcmmagicc`, `gcmagicc_ensemble_runner`, ...).
    """
    env_root = _env_path("GCMAGICC_PROJECTS_ROOT")
    if env_root:
        return env_root

    repo_root = get_project_root()
    repo_parent = repo_root.parent
    if (repo_parent / "gcmmagicc").exists() and (repo_parent / "gcmagicc_ensemble_runner").exists():
        return repo_parent

    detected_site = _detect_site_from_path(repo_root)
    if detected_site == SITE_ETH:
        return ETH_PROJECTS_ROOT
    if detected_site == SITE_GUS:
        return GUS_PROJECTS_ROOT

    return repo_parent


def get_site() -> str:
    """
    Return active site identifier: `eth` or `gus`.

    Priority:
    1) `GCMAGICC_SITE`
    2) inferred from repo/projects root
    3) filesystem heuristics
    4) ETH fallback
    """
    env_site = _norm_site(os.environ.get("GCMAGICC_SITE"))
    if env_site:
        return env_site

    inferred = _detect_site_from_path(get_projects_root()) or _detect_site_from_path(get_project_root())
    if inferred:
        return inferred

    if GUS_PROJECTS_ROOT.exists() and not ETH_PROJECTS_ROOT.exists():
        return SITE_GUS
    return SITE_ETH


def get_data_profile() -> str:
    """Return active data profile name."""
    return _norm_data_profile(os.environ.get(DATA_PROFILE_ENV)) or DATA_PROFILE_SITE_LOCAL


def is_fressnapf_data_profile() -> bool:
    return get_data_profile() == DATA_PROFILE_FRESSNAPF


def fressnapf_profile_enabled() -> bool:
    """Return True when canonical reads should resolve through Fressnapf."""
    return is_fressnapf_data_profile()


def get_site_scratch_data_root(site: str | None = None) -> Path:
    norm = _norm_site(site) or get_site()
    return ETH_DATA_ROOT if norm == SITE_ETH else GUS_DATA_ROOT


def get_shared_data_root_for_site(site: str | None = None) -> Path:
    env_root = _env_path("GCMAGICC_DATA_ROOT")
    if env_root:
        return env_root
    if is_fressnapf_data_profile():
        return FRESSNAPF_DATA_ROOT
    return get_site_scratch_data_root(site)


def get_data_root() -> Path:
    """Return root directory that stores shared input datasets."""
    return get_shared_data_root_for_site(get_site())


def get_version_default() -> str:
    """Return default output version (`v101` on ETH, `v100` on GUS unless overridden)."""
    return os.environ.get("GCMAGICC_VERSION_DEFAULT", DEFAULT_VERSION_BY_SITE[get_site()]).strip()


def get_repo_path(repo_name: str) -> Path:
    """Return absolute path to a sibling repository under the projects root."""
    return get_projects_root() / repo_name


def get_cmipcruncher_firefly_root() -> Path:
    """Return sibling `cmipcruncher_firefly` repository root."""
    return get_repo_path("cmipcruncher_firefly")


def get_data_path(relative_path: str = "") -> Path:
    """
    Return path inside this repo's `data/` directory.
    """
    base = get_project_root() / "data"
    return (base / relative_path) if relative_path else base


def _load_cmipcruncher_healpix_defaults_module():
    module_path = (
        get_cmipcruncher_firefly_root() / "src" / "cmip6cruncher" / "healpix_defaults.py"
    ).resolve(strict=False)
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("cmip6cruncher_healpix_defaults_ext", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dir_is_writable(path: Path) -> bool:
    """
    Return True when *path* can be created and written to by the current user.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".joblib_write_test_",
            dir=str(path),
            delete=True,
        ):
            pass
        return True
    except Exception:
        return False


def get_joblib_tmpdir_default() -> str:
    """
    Resolve a writable, site-aware default temp directory for joblib payloads.

    Priority:
    1) `JOBLIB_TEMP_FOLDER` environment override
    2) ETH/GUS site-aware scratch candidates
    3) repo-local fallback
    4) `/tmp` user fallback
    """
    env_override = os.environ.get("JOBLIB_TEMP_FOLDER", "").strip()
    if env_override:
        return env_override

    site = get_site()
    data_root = get_site_scratch_data_root(site)
    project_root = get_project_root()

    candidates: list[Path] = []
    if site == SITE_ETH:
        candidates.extend(
            [
                Path("data/tmp_joblib"),
                data_root.parent / "tmp_joblib",
            ]
        )
    elif site == SITE_GUS:
        candidates.extend(
            [
                data_root / "tmp_joblib",
                Path("data/site_gus/tmp_joblib"),
                get_projects_root() / "tmp_joblib",
            ]
        )

    candidates.extend(
        [
            project_root / ".tmp_joblib",
            Path(f"/tmp/gcmagicc_joblib_{os.getuid()}"),
        ]
    )

    for candidate in candidates:
        if _dir_is_writable(candidate):
            return str(candidate)

    # Last-resort deterministic fallback (may still fail at runtime on very strict hosts).
    return str(project_root / ".tmp_joblib")


def get_output_folder(default_relative: str = "reports") -> str:
    return str(get_data_path(default_relative))


def get_sqlite_path(default_relative: str = "metric_databases/metrics.sqlite") -> str:
    return str(get_data_path(default_relative))


def get_gcmagicc_path(version: str = "") -> str:
    base = get_data_path("GCMagicc")
    return str(base / version) if version else str(base)


def _normalize_version_token(version: str | None) -> str:
    return str(version or "").strip().lower()


def is_gxe_version(version: str | None) -> bool:
    return _normalize_version_token(version).endswith("gxe")


def get_version_home_site(version: str | None) -> str | None:
    return KNOWN_VERSION_HOME_SITE.get(_normalize_version_token(version))


def _version_root_env_var(version: str | None) -> str:
    token = _VERSION_ENV_SANITIZE_RE.sub("_", _normalize_version_token(version)).strip("_").upper()
    if not token:
        return "GCMAGICC_VERSION_ROOT"
    return f"GCMAGICC_{token}_ROOT"


def _version_candidates_env_var(version: str | None) -> str:
    return _version_root_env_var(version).replace("_ROOT", "_CANDIDATES")


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        p = Path(raw).expanduser()
        key = str(p.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        out.append(Path(key))
    return out


def _site_order_for_version(version: str | None) -> list[str]:
    preferred = get_version_home_site(version)
    ordered: list[str] = []
    if preferred in SITE_VALUES:
        ordered.append(preferred)
    for site in (SITE_ETH, SITE_GUS):
        if site not in ordered:
            ordered.append(site)
    return ordered


def _archive_roots_for_site(site: str) -> list[Path]:
    norm = _norm_site(site)
    if norm == SITE_ETH:
        return [ETH_GCMAGICC_ARCHIVE_ROOT]
    if norm == SITE_GUS:
        return [GUS_GCMAGICC_ARCHIVE_ROOT, LEGACY_GUS_GCMAGICC_ARCHIVE_ROOT]
    return []


def _created_nc_roots_for_site(site: str) -> list[Path]:
    norm = _norm_site(site)
    if norm == SITE_ETH:
        return [ETH_CREATED_NC_ROOT]
    if norm == SITE_GUS:
        return [GUS_CREATED_NC_ROOT, LEGACY_GUS_CREATED_NC_ROOT]
    return []


def get_gcmagicc_archive_candidates(version: str, *, include_local_repo: bool = True) -> list[Path]:
    """
    Return ordered candidate directories for a specific GCMagicc version.

    This is transition-safe across ETH/GUS while data layouts are being moved.
    Per-version overrides are supported via `GCMAGICC_<VERSION>_ROOT`.
    """
    normalized = _normalize_version_token(version)
    if not normalized:
        return [Path(get_gcmagicc_path())] if include_local_repo else []

    candidates: list[Path] = []

    env_specific = _env_path(_version_root_env_var(normalized))
    if env_specific:
        candidates.append(env_specific)

    # Backward compatibility with existing env knobs used in notebooks.
    if normalized == "v100":
        env_v100 = _env_path("GCMAGICC_V100_ROOT")
        if env_v100:
            candidates.append(env_v100)
    if is_gxe_version(normalized):
        env_gxe = _env_path("GCMAGICC_GXE_FOLDER")
        if env_gxe:
            candidates.append(env_gxe)

    for site in _site_order_for_version(normalized):
        for root in _archive_roots_for_site(site):
            candidates.append(root / normalized)

    if include_local_repo:
        candidates.append(Path(get_gcmagicc_path(normalized)))

    if is_gxe_version(normalized):
        candidates.extend(get_default_gxe_candidate_folders(version=normalized))

    return _dedupe_paths(candidates)


def get_gcmagicc_archive_path(version: str, *, must_exist: bool = True) -> Path:
    """
    Resolve one archive folder for `version`.
    Returns the first existing candidate by default.
    """
    candidates = get_gcmagicc_archive_candidates(version)
    if not candidates:
        return Path(get_gcmagicc_path(version))
    if must_exist:
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return candidates[0]


def get_default_gcmagicc_archive_roots(
    versions: Sequence[str] | None = None,
    *,
    include_base_local: bool = True,
    existing_only: bool = False,
) -> list[Path]:
    """
    Return merged archive roots for typical validation versions across ETH/GUS.
    """
    selected_versions = tuple(versions) if versions is not None else DEFAULT_ARCHIVE_VERSIONS
    roots: list[Path] = []
    if include_base_local:
        roots.append(Path(get_gcmagicc_path()))
    for version in selected_versions:
        roots.extend(get_gcmagicc_archive_candidates(version))
    deduped = _dedupe_paths(roots)
    if existing_only:
        deduped = [p for p in deduped if p.exists()]
    return deduped


def get_gcmagicc_version_path(version: str | None = None) -> Path:
    """Return `data/GCMagicc/<version>`, using site default when omitted."""
    effective = (version or get_version_default()).strip()
    return Path(get_gcmagicc_path(effective))


def get_cmip6_vetted_candidates() -> list[Path]:
    override = _env_path("GCMAGICC_CMIP6_ROOT")
    if override:
        return [override]

    root = get_data_root()
    site = get_site()
    candidates: list[Path] = []

    if is_fressnapf_data_profile():
        processed_root = FRESSNAPF_CMIP6_ROOT / "ETHFOG" / "processed"
        candidates.extend(
            [
                FRESSNAPF_CMIP6_VETTED_NFS_COPY_ROOT,
                FRESSNAPF_CMIP6_VETTED_ROOT,
                processed_root,
                FRESSNAPF_CMIP6_ROOT / "ETHFOG",
            ]
        )
        vetted_children = [
            p.resolve(strict=False)
            for p in sorted(_safe_glob(processed_root, "out_ETHFOG*_vetted"))
            if _safe_is_dir(p)
        ]
        candidates.extend(vetted_children)
        candidates.extend(
            [
                root / "CMIP6" / "ETHFOG" / "processed" / "out_ETHFOG_10June2025_vetted",
                root / "CMIP6" / "ETHFOG" / "processed",
                root / "CMIP6" / "ETHFOG",
                root / "cmip6_ETHFOG" / "processed" / "out_ETHFOG_10June2025_vetted",
                root / "cmip6_ETHFOG" / "processed",
                root / "cmip6_ETHFOG",
            ]
        )
        return _dedupe_paths(candidates)

    if site == SITE_GUS:
        processed_root = root / "cmip6_ETHFOG" / "processed"
        candidates.append(processed_root / "out_ETHFOG_10June2025_vetted")
        vetted_children = [
            p.resolve(strict=False)
            for p in sorted(_safe_glob(processed_root, "out_ETHFOG*_vetted"))
            if _safe_is_dir(p)
        ]
        candidates.extend(vetted_children)
        candidates.extend(
            [
                root / "out_ETHFOG_10June2025_vetted",
                processed_root,
                root / "cmip6_ETHFOG",
            ]
        )
    else:
        candidates.extend(
            [
                root / "out_ETHFOG_10June2025_vetted",
                root / "cmip6_ETHFOG" / "processed" / "out_ETHFOG_10June2025_vetted",
                root / "cmip6_ETHFOG",
            ]
        )

    return _dedupe_paths(candidates)


def get_cmip6_vetted_path() -> Path:
    candidates = get_cmip6_vetted_candidates()
    for candidate in candidates:
        if not _safe_exists(candidate):
            continue
        if _safe_is_dir(candidate) and _safe_glob(candidate, "DAT_*.nc"):
            return candidate
        if _safe_is_dir(candidate) and any(p.is_file() and p.name.startswith("DAT_") for p in _safe_iterdir(candidate)):
            return candidate
    for candidate in candidates:
        if _safe_exists(candidate):
            return candidate
    return candidates[0]


def get_era5_vetted_path() -> Path:
    override = _env_path("GCMAGICC_ERA5_ROOT")
    if override:
        return override
    if is_fressnapf_data_profile():
        candidates = [
            FRESSNAPF_ERA5_VETTED_ROOT,
            FRESSNAPF_ERA5_025_VETTED_ROOT,
            FRESSNAPF_DATA_ROOT / "ERA5" / "processed",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
    root = get_data_root()
    candidates = [
        root / "out_ERA5_19Feb2026_1degree_vetted",
        root / "out_ERA5_4July2025_1degree_vetted",
        root / "ERA5",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_era5_main_file() -> Path:
    override = _env_path("GCMAGICC_ERA5_FILE")
    if override:
        return override
    era5_root = get_era5_vetted_path()
    expected = _era5_expected_main_file(era5_root)
    if expected.exists():
        return expected
    matches = sorted(era5_root.rglob("DAT_ERA5_historical-ERA5*.nc"))
    if matches:
        return matches[0]
    return expected


def _era5_expected_main_file(root: Path) -> Path:
    return Path(root) / (
        "DAT_ERA5_historical-ERA5_r1i1p1f1_clt-day-hurs-huss-month-pr-psl-rlut-"
        "rsds-rsdt-rsnt-rtmt-sfcWind-tas-tasmax-tasmin-ts-year.nc"
    )


def get_era5_025_vetted_path() -> Path:
    override = _env_path("GCMAGICC_ERA5_025_ROOT")
    if override:
        return override
    if is_fressnapf_data_profile():
        candidates = [
            FRESSNAPF_ERA5_025_VETTED_ROOT,
            FRESSNAPF_DATA_ROOT / "ERA5" / "processed" / "monthly" / "multivar_025x025",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
    return FRESSNAPF_ERA5_025_VETTED_ROOT


def get_era5_025_main_file() -> Path:
    override = _env_path("GCMAGICC_ERA5_025_FILE")
    if override:
        return override
    era5_root = get_era5_025_vetted_path()
    expected = _era5_expected_main_file(era5_root)
    if expected.exists():
        return expected
    matches = sorted(era5_root.rglob("DAT_ERA5_historical-ERA5*.nc"))
    if matches:
        return matches[0]
    return expected


def get_created_nc_files_root() -> Path:
    override = _env_path("GCMAGICC_CREATED_NC_ROOT")
    if override:
        return override
    return get_repo_path("gcmagicc_ensemble_runner") / "created_nc_files"


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def legacy_fallback_enabled() -> bool:
    """
    Emergency-only fallback to legacy roots (disabled by default).
    """
    return _env_flag(LEGACY_FALLBACK_ENV, default=False)


def get_era5spliced_legacy_root() -> Path:
    override = _env_path("GCMAGICC_ERA5SPLICED_LEGACY_ROOT")
    if override:
        return override
    return ETH_ERA5SPLICED_LEGACY_ROOT if get_site() == SITE_ETH else GUS_ERA5SPLICED_LEGACY_ROOT


def get_era5spliced_catalog_path() -> Path:
    override = _env_path("GCMAGICC_ERA5SPLICED_CATALOG")
    if override:
        return override
    return ETH_ERA5SPLICED_CATALOG if get_site() == SITE_ETH else GUS_ERA5SPLICED_CATALOG


def get_era5spliced_root() -> Path:
    """
    Return canonical ERA5spliced root for this site.
    Default is the site-local canonical root unless the opt-in Fressnapf data
    profile is active.
    """
    override = _env_path("GCMAGICC_ERA5SPLICED_ROOT")
    if override:
        return override
    if is_fressnapf_data_profile():
        return FRESSNAPF_ERA5SPLICED_ROOT
    return ETH_ERA5SPLICED_ROOT if get_site() == SITE_ETH else GUS_ERA5SPLICED_ROOT


def get_era5spliced_localstaging_root() -> Path:
    """
    Return writable local mirror root for staged ERA5spliced originals.
    """
    override = _env_path("GCMAGICC_ERA5SPLICED_LOCALSTAGING_ROOT")
    if override:
        return override
    return (
        ETH_ERA5SPLICED_LOCALSTAGING_ROOT
        if get_site() == SITE_ETH
        else GUS_ERA5SPLICED_LOCALSTAGING_ROOT
    )


def get_era5spliced_localresults_root() -> Path:
    """
    Return writable local root for ERA5spliced dataderivatives and world outputs.
    """
    override = _env_path("GCMAGICC_ERA5SPLICED_LOCALRESULTS_ROOT")
    if override:
        return override
    return (
        ETH_ERA5SPLICED_LOCALRESULTS_ROOT
        if get_site() == SITE_ETH
        else GUS_ERA5SPLICED_LOCALRESULTS_ROOT
    )


def get_era5spliced_cmip6_localresults_root() -> Path:
    """
    Return writable local root for ERA5spliced CMIP6-style converted outputs.
    """
    override = _env_path("GCMAGICC_ERA5SPLICED_CMIP6_LOCALRESULTS_ROOT")
    if override:
        return override
    return (
        ETH_ERA5SPLICED_CMIP6_LOCALRESULTS_ROOT
        if get_site() == SITE_ETH
        else GUS_ERA5SPLICED_CMIP6_LOCALRESULTS_ROOT
    )


def get_era5spliced_cmip7_localresults_root() -> Path:
    """
    Return writable local root for ERA5spliced CMIP7-style converted outputs.
    """
    override = _env_path("GCMAGICC_ERA5SPLICED_CMIP7_LOCALRESULTS_ROOT")
    if override:
        return override
    return (
        ETH_ERA5SPLICED_CMIP7_LOCALRESULTS_ROOT
        if get_site() == SITE_ETH
        else GUS_ERA5SPLICED_CMIP7_LOCALRESULTS_ROOT
    )


def get_cmip6replicas_root() -> Path:
    """
    Return canonical CMIP6-aligned GCMagicc root for this site/profile.
    """
    override = _env_path("GCMAGICC_CMIP6REPLICAS_ROOT")
    if override:
        return override
    if is_fressnapf_data_profile():
        return FRESSNAPF_CMIP6REPLICAS_ROOT
    return ETH_CMIP6REPLICAS_ROOT if get_site() == SITE_ETH else GUS_CMIP6REPLICAS_ROOT


def get_cmip6replicas_localstaging_root() -> Path:
    """
    Return writable local staging root for GapFiller CMIP6-aligned subsets.
    """
    override = _env_path("GCMAGICC_CMIP6REPLICAS_LOCALSTAGING_ROOT")
    if override:
        return override
    return (
        ETH_CMIP6REPLICAS_LOCALSTAGING_ROOT
        if get_site() == SITE_ETH
        else GUS_CMIP6REPLICAS_LOCALSTAGING_ROOT
    )


def get_cmip6_localstaging_root() -> Path:
    """
    Return writable local staging root for staged raw CMIP6 reference subsets.
    """
    override = _env_path("GCMAGICC_CMIP6_LOCALSTAGING_ROOT")
    if override:
        return override
    return ETH_CMIP6_LOCALSTAGING_ROOT if get_site() == SITE_ETH else GUS_CMIP6_LOCALSTAGING_ROOT


def get_fressnapf_cmip6_healpix_root() -> Path:
    override = _env_path("GCMAGICC_FRESSNAPF_CMIP6_HEALPIX_ROOT")
    if override:
        return override
    return FRESSNAPF_CMIP6_HEALPIX_ROOT


def get_fressnapf_era5_healpix_root() -> Path:
    override = _env_path("GCMAGICC_FRESSNAPF_ERA5_HEALPIX_ROOT")
    if override:
        return override
    return FRESSNAPF_ERA5_HEALPIX_ROOT


def _resolve_healpix_source_prefix_from_sibling(
    *,
    builder_name: str,
    site: str | None = None,
    version: str | None = None,
) -> str | None:
    module = _load_cmipcruncher_healpix_defaults_module()
    if module is None:
        return None
    builder = getattr(module, builder_name, None)
    if builder is None:
        return None
    return str(
        builder(
            site=site or get_site(),
            version=version or get_version_default(),
        )
    ).strip().strip("/")


def resolve_fressnapf_cmip6_healpix_source_prefix(
    *,
    site: str | None = None,
    version: str | None = None,
) -> str:
    override = os.environ.get("GCMAGICC_FRESSNAPF_CMIP6_HEALPIX_SOURCE", "").strip().strip("/")
    if override:
        return override
    resolved = _resolve_healpix_source_prefix_from_sibling(
        builder_name="build_default_cmip6_healpix_prefix",
        site=site,
        version=version,
    )
    if resolved:
        return resolved
    helper_path = get_cmipcruncher_firefly_root() / "src" / "cmip6cruncher" / "healpix_defaults.py"
    raise RuntimeError(
        "Could not resolve the default CMIP6 HEALPix source prefix for Fressnapf. "
        f"Set GCMAGICC_FRESSNAPF_CMIP6_HEALPIX_SOURCE or add the sibling helper at {helper_path}."
    )


def resolve_fressnapf_era5_healpix_source_prefix(
    *,
    site: str | None = None,
    version: str | None = None,
) -> str:
    override = os.environ.get("GCMAGICC_FRESSNAPF_ERA5_HEALPIX_SOURCE", "").strip().strip("/")
    if override:
        return override
    resolved = _resolve_healpix_source_prefix_from_sibling(
        builder_name="build_default_era5_healpix_day_prefix",
        site=site,
        version=version,
    )
    if resolved:
        return resolved
    helper_path = get_cmipcruncher_firefly_root() / "src" / "cmip6cruncher" / "healpix_defaults.py"
    raise RuntimeError(
        "Could not resolve the default ERA5 HEALPix day-source prefix for Fressnapf. "
        f"Set GCMAGICC_FRESSNAPF_ERA5_HEALPIX_SOURCE or add the sibling helper at {helper_path}."
    )


def _sanitize_token(value: str, *, default: str = "unknown") -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    token = re.sub(r"-{2,}", "-", token).strip("-._")
    return token or default


def _parse_findmnt_pairs_line(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in shlex.split(str(line).strip()):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = str(key or "").strip().lower()
        if key:
            fields[key] = value
    return fields


def get_rclone_remote_default() -> str:
    return (
        os.environ.get("GCMAGICC_RCLONE_REMOTE", "").strip()
        or os.environ.get("GCMAGICC_OBJECT_REMOTE", "").strip()
        or DEFAULT_RCLONE_REMOTE
    )


def get_mount_entry_for_path(path: Path | str, *, resolve_path: bool = True) -> dict[str, str] | None:
    target_path = Path(path).expanduser()
    if resolve_path:
        target_path = target_path.resolve(strict=False)
    try:
        proc = subprocess.run(
            ["findmnt", "-n", "--pairs", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS", "-T", str(target_path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    for line in (proc.stdout or "").splitlines():
        parsed = _parse_findmnt_pairs_line(line)
        if parsed:
            return parsed
    return None


def path_uses_rclone_mount(path: Path | str, *, resolve_path: bool = True) -> bool:
    entry = get_mount_entry_for_path(path, resolve_path=resolve_path)
    if entry is None:
        return False
    return str(entry.get("fstype", "")).strip() == RCLONE_MOUNT_FSTYPE


def _normalize_mount_source_to_rclone_ref(
    source: str,
    *,
    remote: str | None = None,
    bucket: str | None = None,
) -> str | None:
    raw = str(source or "").strip()
    if not raw:
        return None

    bucket_value = str(bucket or get_object_bucket()).strip()
    selected_remote = str(remote or "").strip()

    if raw.startswith("s3://"):
        after = raw[5:]
        if "/" in after:
            src_bucket, rel = after.split("/", 1)
        else:
            src_bucket, rel = after, ""
        use_remote = selected_remote or get_rclone_remote_default()
        use_bucket = src_bucket.strip() or bucket_value
        rel_token = rel.strip().strip("/")
        if not rel_token:
            return f"{use_remote}:{use_bucket}"
        return f"{use_remote}:{use_bucket}/{rel_token}"

    if ":" in raw:
        remote_token, rest = raw.split(":", 1)
        remote_token = remote_token.split("{", 1)[0].strip()
        use_remote = selected_remote or remote_token or get_rclone_remote_default()
        token = rest.strip().strip("/")
        if not token:
            return None
        if bucket_value and token.startswith(bucket_value + "/"):
            return f"{use_remote}:{token}"
        if bucket_value:
            return f"{use_remote}:{bucket_value}/{token}"
        return f"{use_remote}:{token}"

    return None


def _mount_source_to_s3_bucket_prefix(
    source: str,
    *,
    bucket: str | None = None,
) -> tuple[str, str] | None:
    raw = str(source or "").strip()
    if not raw:
        return None

    bucket_value = str(bucket or get_object_bucket()).strip()

    if raw.startswith("s3://"):
        after = raw[5:]
        if "/" in after:
            src_bucket, rel = after.split("/", 1)
        else:
            src_bucket, rel = after, ""
        bucket_token = src_bucket.strip() or bucket_value
        if not bucket_token:
            return None
        return bucket_token, rel.strip().strip("/")

    if ":" not in raw:
        return None

    _, rest = raw.split(":", 1)
    token = rest.strip().strip("/")
    if not token:
        return None

    if bucket_value and token.startswith(bucket_value + "/"):
        return bucket_value, token[len(bucket_value) + 1 :].strip("/")

    if "/" in token:
        bucket_token, rel = token.split("/", 1)
        bucket_token = bucket_token.strip() or bucket_value
        if not bucket_token:
            return None
        return bucket_token, rel.strip().strip("/")

    if bucket_value:
        return bucket_value, token

    return token, ""


def _s3_uri_from_mount_entry(
    path: Path | str,
    entry: dict[str, str] | None,
    *,
    bucket: str | None = None,
) -> str | None:
    if entry is None:
        return None
    if str(entry.get("fstype", "")).strip() != RCLONE_MOUNT_FSTYPE:
        return None

    parsed = _mount_source_to_s3_bucket_prefix(str(entry.get("source", "")).strip(), bucket=bucket)
    if parsed is None:
        return None
    bucket_token, prefix = parsed

    local_path = Path(path).expanduser().resolve(strict=False)
    target = Path(str(entry.get("target", "")).strip()).expanduser().resolve(strict=False)
    try:
        rel = local_path.relative_to(target)
    except ValueError:
        return None

    rel_token = rel.as_posix().strip("/")
    key = prefix.rstrip("/")
    if rel_token:
        key = f"{key}/{rel_token}" if key else rel_token
    if not key:
        return f"s3://{bucket_token}"
    return f"s3://{bucket_token}/{key}"


def resolve_rclone_source_ref(
    path: Path | str,
    *,
    remote: str | None = None,
    bucket: str | None = None,
    resolve_path: bool = True,
) -> str | None:
    raw = str(path).strip()
    if not raw:
        return None
    direct = _normalize_mount_source_to_rclone_ref(raw, remote=remote, bucket=bucket)
    if raw.startswith("s3://"):
        return direct

    local_path = Path(raw).expanduser()
    if resolve_path:
        local_path = local_path.resolve(strict=False)

    entry = get_mount_entry_for_path(local_path, resolve_path=False)
    if entry is not None and str(entry.get("fstype", "")).strip() == RCLONE_MOUNT_FSTYPE:
        mount_ref = _normalize_mount_source_to_rclone_ref(
            str(entry.get("source", "")).strip(),
            remote=remote,
            bucket=bucket,
        )
        if mount_ref is not None:
            try:
                rel = local_path.relative_to(Path(str(entry.get("target", "")).strip()))
            except Exception:
                rel = None
            if rel is None or str(rel) in {"", "."}:
                return mount_ref
            return f"{mount_ref.rstrip('/')}/{rel.as_posix()}"

    s3_uri = convert_local_path_to_s3_uri(local_path, bucket=bucket)
    if s3_uri:
        return _normalize_mount_source_to_rclone_ref(s3_uri, remote=remote, bucket=bucket)
    return direct


def copy_remote_file_atomic_via_rclone(
    source_ref: str,
    destination: Path | str,
    *,
    extra_args: Sequence[str] | None = None,
) -> Path:
    target = Path(destination).expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.part"
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass

    cmd = ["rclone", "copyto", str(source_ref), str(tmp)]
    if extra_args:
        cmd.extend(str(x) for x in extra_args if str(x).strip())
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        try:
            tmp.unlink()
        except OSError:
            pass
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"rclone copy failed for {source_ref} -> {target}: {detail}")
    if not tmp.exists():
        raise RuntimeError(f"rclone copy did not produce expected temp file: {tmp}")
    os.replace(tmp, target)
    return target


def normalize_storage_access(value: str | None) -> str:
    token = str(value or STORAGE_ACCESS_MOUNT).strip().lower()
    if token in STORAGE_ACCESS_CHOICES:
        return token
    raise ValueError(
        f"Unsupported storage access '{value}'. "
        f"Choose one of: {', '.join(STORAGE_ACCESS_CHOICES)}."
    )


def get_storage_access_default() -> str:
    return normalize_storage_access(os.environ.get("GCMAGICC_STORAGE_ACCESS", STORAGE_ACCESS_MOUNT))


def get_rclone_config_path() -> Path:
    env_value = os.environ.get("RCLONE_CONFIG", "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve(strict=False)
    return (Path.home() / ".config" / "rclone" / "rclone.conf").expanduser().resolve(strict=False)


def _parse_simple_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    out: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return out

    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw.startswith("export "):
            raw = raw[len("export ") :].strip()
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        out[key] = value
    return out


def get_s3_env_file_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get("GCMAGICC_S3_ENV_FILE", "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve(strict=False))

    candidates.append((Path.home() / ".config" / "gcmagicc" / "ovh_s3.env").expanduser().resolve(strict=False))
    candidates.append((get_projects_root() / "gcm_firefly_frontend" / ".env.ovh_s3").expanduser().resolve(strict=False))

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def get_s3_env_file_values() -> dict[str, str]:
    for path in get_s3_env_file_candidates():
        values = _parse_simple_env_file(path)
        if values:
            return values
    return {}


def get_rclone_remote_config(remote: str | None = None) -> dict[str, str]:
    remote_name = str(remote or get_rclone_remote_default()).strip()
    if not remote_name:
        return {}

    cfg_path = get_rclone_config_path()
    if not cfg_path.exists():
        return {}

    parser = configparser.ConfigParser()
    try:
        parser.read(cfg_path, encoding="utf-8")
    except Exception:
        return {}
    if not parser.has_section(remote_name):
        return {}
    return {str(k).strip(): str(v).strip() for k, v in parser.items(remote_name)}


def get_s3_storage_options(
    *,
    remote: str | None = None,
    force_path_style_default: bool = True,
) -> dict[str, object]:
    opts: dict[str, object] = {}
    env_file_values = get_s3_env_file_values()

    endpoint = (
        os.environ.get("AWS_ENDPOINT_URL_S3")
        or os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("GCMAGICC_S3_ENDPOINT_URL")
    )
    access_key = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_KEY")
    session_token = os.environ.get("AWS_SESSION_TOKEN")
    region_name = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
    endpoint = endpoint or env_file_values.get("AWS_ENDPOINT_URL_S3") or env_file_values.get("AWS_ENDPOINT_URL")
    access_key = access_key or env_file_values.get("AWS_ACCESS_KEY_ID") or env_file_values.get("AWS_ACCESS_KEY")
    secret_key = secret_key or env_file_values.get("AWS_SECRET_ACCESS_KEY") or env_file_values.get("AWS_SECRET_KEY")
    session_token = session_token or env_file_values.get("AWS_SESSION_TOKEN")
    region_name = region_name or env_file_values.get("AWS_DEFAULT_REGION") or env_file_values.get("AWS_REGION")

    if not endpoint or not access_key or not secret_key:
        remote_cfg = get_rclone_remote_config(remote=remote)
        endpoint = endpoint or remote_cfg.get("endpoint")
        access_key = access_key or remote_cfg.get("access_key_id")
        secret_key = secret_key or remote_cfg.get("secret_access_key")
        session_token = session_token or remote_cfg.get("session_token")
        region_name = region_name or remote_cfg.get("region") or remote_cfg.get("location_constraint")

    client_kwargs: dict[str, object] = {}
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    if region_name:
        client_kwargs["region_name"] = region_name
    if client_kwargs:
        opts["client_kwargs"] = client_kwargs

    if os.environ.get("AWS_NO_SIGN_REQUEST", "").strip().lower() in {"1", "true", "yes"}:
        opts["anon"] = True
    else:
        if access_key:
            opts["key"] = access_key
        if secret_key:
            opts["secret"] = secret_key
        if session_token:
            opts["token"] = session_token

    force_path_style = os.environ.get(
        "GCMAGICC_S3_FORCE_PATH_STYLE",
        env_file_values.get(
            "GCMAGICC_S3_FORCE_PATH_STYLE",
            "1" if force_path_style_default else "0",
        ),
    ).strip().lower()
    if force_path_style not in {"0", "false", "no"}:
        opts["config_kwargs"] = {"s3": {"addressing_style": "path"}}

    return opts


def normalize_runmodus_canonical(runmodus: str | None) -> str:
    token = str(runmodus or "").strip().lower()
    if token in {"", "all", "anthropogenic", "anthrop", "anth", "full"}:
        return RUNMODUS_ALL
    if token in {"nat", "natural"}:
        return RUNMODUS_NAT
    if token in {"aer", "aerosol"}:
        return RUNMODUS_AER
    raise ValueError(
        f"Unsupported runmodus '{runmodus}'. "
        f"Choose one of: {', '.join(RUNMODUS_CHOICES)} (aliases: natural->nat, aerosol->aer)."
    )


def _strip_runmodus_suffix(experiment_id: str) -> tuple[str, str | None]:
    token = str(experiment_id or "").strip()
    low = token.lower()
    suffixes = (
        ("_runmodus_natural", RUNMODUS_NAT),
        ("_runmode_natural", RUNMODUS_NAT),
        ("_runmodus_aerosol", RUNMODUS_AER),
        ("_runmode_aerosol", RUNMODUS_AER),
        ("-nat", RUNMODUS_NAT),
        ("_nat", RUNMODUS_NAT),
        ("-aer", RUNMODUS_AER),
        ("_aer", RUNMODUS_AER),
    )
    for suffix, runmodus in suffixes:
        if low.endswith(suffix):
            stripped = token[: len(token) - len(suffix)].strip("-_.")
            return stripped, runmodus
    return token, None


def split_experiment_and_runmodus(
    experiment_id: str,
    *,
    runmodus_hint: str | None = None,
) -> tuple[str, str]:
    """
    Normalize (experiment_id, runmodus) with suffix-aware inference.
    """
    stripped, inferred = _strip_runmodus_suffix(experiment_id)
    runmodus = normalize_runmodus_canonical(runmodus_hint if runmodus_hint is not None else inferred)
    exp = _sanitize_token(stripped or experiment_id, default="experiment")
    return exp, runmodus


def normalize_n_ensemble_label(value: str | int | None) -> str:
    token = str(value or "").strip().lower()
    if not token:
        raise ValueError("n_ensemble label must not be empty.")
    if token.startswith("n_") and token[2:].isdigit():
        return f"n_{int(token[2:])}"
    if token.isdigit():
        return f"n_{int(token)}"
    raise ValueError("n_ensemble must be an integer or n_<integer> (e.g. 100 or n_100).")


def normalize_canonical_kind(kind: str | None) -> str:
    token = str(kind or CANONICAL_KIND_ORIGINAL).strip().lower()
    if token in CANONICAL_KIND_CHOICES:
        return token
    raise ValueError(
        f"Unsupported canonical kind '{kind}'. "
        f"Choose one of: {', '.join(CANONICAL_KIND_CHOICES)}."
    )


def normalize_run_instance(value: str | None) -> str:
    token = _sanitize_token(value or "run_unknown", default="run_unknown")
    return token if token.startswith("run_") else f"run_{token}"


def build_era5spliced_dataset_path(
    *,
    version: str,
    experiment_id: str,
    arx: str,
    runmodus: str,
    n_ensemble: str | int,
    kind: str,
    run_instance: str | None = None,
    root: Path | None = None,
) -> Path:
    """
    Build canonical ERA5spliced dataset path:
      <root>/<version>/<experiment_id>/<ARX>/<runmodus>/<n_ensemble>/<kind>[/<run_instance>]
    """
    base = Path(root or get_era5spliced_root()).expanduser().resolve(strict=False)
    version_token = _sanitize_token(version, default=get_version_default().lower()).lower()
    if not version_token.startswith("v"):
        version_token = f"v{version_token}"
    exp_token, runmodus_token = split_experiment_and_runmodus(experiment_id, runmodus_hint=runmodus)
    arx_token = _sanitize_token(arx.upper(), default="AR6").upper()
    n_label = normalize_n_ensemble_label(n_ensemble)
    kind_token = normalize_canonical_kind(kind)
    out = base / version_token / exp_token / arx_token / runmodus_token / n_label / kind_token
    if run_instance is not None and str(run_instance).strip():
        out = out / normalize_run_instance(run_instance)
    return out


def parse_era5spliced_dataset_path(path: Path, *, root: Path | None = None) -> dict[str, str] | None:
    """
    Parse canonical ERA5spliced path.
    """
    resolved = Path(path).expanduser().resolve(strict=False)
    base = Path(root or get_era5spliced_root()).expanduser().resolve(strict=False)
    try:
        rel = resolved.relative_to(base)
    except ValueError:
        return None
    if len(rel.parts) < 6:
        return None
    version, experiment_id, arx, runmodus, n_ensemble, kind = rel.parts[:6]
    run_instance = rel.parts[6] if len(rel.parts) >= 7 else ""
    try:
        kind_norm = normalize_canonical_kind(kind)
    except ValueError:
        return None
    if kind_norm not in CANONICAL_KIND_CHOICES:
        return None
    try:
        runmodus_norm = normalize_runmodus_canonical(runmodus)
    except ValueError:
        return None
    if not str(n_ensemble).lower().startswith("n_"):
        return None
    return {
        "root": str(base),
        "version": version,
        "experiment_id": experiment_id,
        "arx": arx,
        "runmodus": runmodus_norm,
        "n_ensemble": n_ensemble,
        "kind": kind_norm,
        "run_instance": run_instance,
    }


def _normalize_version_dir(version: str | None) -> str:
    token = str(version or "").strip().lower()
    if token.startswith("v100"):
        return "v100"
    if token.startswith("v101"):
        return "v101"
    if token and not token.startswith("v"):
        token = f"v{token}"
    return token or get_version_default().lower()


def _is_verified_status(status: str) -> bool:
    token = str(status or "").strip().lower()
    return token in {"ok", "verified", "success", "passed", "check_ok"}


def _parse_catalog_entry_relpath(entry: dict) -> dict[str, str] | None:
    rel = str(entry.get("canonical_relative_path", "")).strip().strip("/")
    if not rel:
        return None
    parts = Path(rel).parts
    if len(parts) < 7:
        return None
    version, scenario, workflow, runmodus, n_ensemble, kind, run_instance = parts[:7]
    if not scenario:
        return None
    if not str(workflow).upper().startswith("AR"):
        return None
    if not str(n_ensemble).lower().startswith("n_"):
        return None
    try:
        kind_norm = normalize_canonical_kind(kind)
        runmodus_norm = normalize_runmodus_canonical(runmodus)
        n_norm = normalize_n_ensemble_label(n_ensemble)
    except Exception:
        return None
    return {
        "rel": rel,
        "version": _normalize_version_dir(version),
        "scenario": scenario,
        "workflow": workflow.upper(),
        "runmodus": runmodus_norm,
        "n_ensemble": n_norm,
        "kind": kind_norm,
        "run_instance": normalize_run_instance(run_instance),
        "verification_status": str(entry.get("verification_status", "")).strip(),
        "timestamp_utc": str(entry.get("timestamp_utc", "")).strip(),
    }


def _catalog_candidates(
    *,
    version: str,
    experiment_id: str,
    arx: str,
    runmodus: str,
    n_ensemble: str | int,
    kind: str,
    catalog_path: Path | None = None,
    require_verified: bool = True,
) -> list[dict[str, str]]:
    cat_path = Path(catalog_path or get_era5spliced_catalog_path()).expanduser().resolve(strict=False)
    if not cat_path.exists():
        return []
    try:
        payload = json.loads(cat_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        return []

    exp_norm, runmodus_norm = split_experiment_and_runmodus(experiment_id, runmodus_hint=runmodus)
    key_version = _normalize_version_dir(version)
    key_workflow = _sanitize_token(arx.upper(), default="AR6").upper()
    key_n = normalize_n_ensemble_label(n_ensemble)
    key_kind = normalize_canonical_kind(kind)

    out: list[dict[str, str]] = []
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        parsed = _parse_catalog_entry_relpath(raw)
        if parsed is None:
            continue
        if require_verified and (not _is_verified_status(parsed["verification_status"])):
            continue
        if parsed["version"] != key_version:
            continue
        if parsed["scenario"] != exp_norm:
            continue
        if parsed["workflow"] != key_workflow:
            continue
        if parsed["runmodus"] != runmodus_norm:
            continue
        if parsed["n_ensemble"] != key_n:
            continue
        if parsed["kind"] != key_kind:
            continue
        out.append(parsed)
    out.sort(key=lambda x: (x.get("timestamp_utc", ""), x["run_instance"]))
    return out


def _latest_run_from_existing_dir(base: Path) -> Path | None:
    if not _safe_exists(base) or not _safe_is_dir(base):
        return None
    run_dirs = sorted(
        [p for p in _safe_iterdir(base) if _safe_is_dir(p) and p.name.startswith("run_")],
        key=lambda p: p.name,
    )
    return run_dirs[-1] if run_dirs else None


def _legacy_created_nc_candidates(
    *,
    version: str,
    experiment_id: str,
    arx: str,
    runmodus: str,
    n_ensemble: str | int,
    kind: str,
) -> list[Path]:
    created_root = get_created_nc_files_root().expanduser().resolve(strict=False)
    if not _safe_exists(created_root):
        return []
    exp_norm, runmodus_norm = split_experiment_and_runmodus(experiment_id, runmodus_hint=runmodus)
    arx_norm = _sanitize_token(arx.upper(), default="AR6").upper()
    n_label = normalize_n_ensemble_label(n_ensemble)
    n_digits = n_label.split("_", 1)[1] if "_" in n_label else n_label
    version_norm = _normalize_version_dir(version)

    scenarios: list[str] = [exp_norm]
    if runmodus_norm == RUNMODUS_NAT and not exp_norm.lower().endswith("-nat"):
        scenarios.append(f"{exp_norm}-nat")
    if runmodus_norm == RUNMODUS_AER and not exp_norm.lower().endswith("-aer"):
        scenarios.append(f"{exp_norm}-aer")

    run_glob = f"debiasloop_{n_digits}*"
    if normalize_canonical_kind(kind) == CANONICAL_KIND_DATADERIVATIVES:
        run_glob = f"{run_glob}_dataderivatives*"

    out: list[Path] = []
    seen: set[str] = set()
    for scenario in scenarios:
        try:
            raw_candidates = sorted(
                created_root.glob(f"{run_glob}/debias/{version_norm}/{scenario}/{arx_norm}"),
                key=lambda p: p.parent.parent.parent.name,
            )
        except OSError:
            raw_candidates = []
        for candidate in raw_candidates:
            key = str(candidate.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            out.append(Path(key))
    return out


def resolve_latest_verified_run_root(
    *,
    version: str,
    experiment_id: str,
    arx: str,
    runmodus: str,
    n_ensemble: str | int,
    kind: str,
    root: Path | None = None,
    catalog_path: Path | None = None,
    require_verified: bool = True,
) -> Path | None:
    """
    Resolve latest verified canonical run root from the consolidated catalog.
    """
    base_root = Path(root or get_era5spliced_root()).expanduser().resolve(strict=False)
    candidates = _catalog_candidates(
        version=version,
        experiment_id=experiment_id,
        arx=arx,
        runmodus=runmodus,
        n_ensemble=n_ensemble,
        kind=kind,
        catalog_path=catalog_path,
        require_verified=require_verified,
    )
    if candidates:
        return base_root / candidates[-1]["rel"]
    return None


def resolve_canonical_dataset_root(
    *,
    version: str,
    experiment_id: str,
    arx: str,
    runmodus: str,
    n_ensemble: str | int,
    kind: str,
    run_instance: str | None = None,
    root: Path | None = None,
    catalog_path: Path | None = None,
    require_verified: bool = True,
) -> Path:
    """
    Resolve canonical dataset root (run-level when available).
    """
    base_root = Path(root or get_era5spliced_root()).expanduser().resolve(strict=False)
    exp_norm, runmodus_norm = split_experiment_and_runmodus(experiment_id, runmodus_hint=runmodus)
    key_kind = normalize_canonical_kind(kind)

    if run_instance:
        return build_era5spliced_dataset_path(
            version=version,
            experiment_id=exp_norm,
            arx=arx,
            runmodus=runmodus_norm,
            n_ensemble=n_ensemble,
            kind=key_kind,
            run_instance=run_instance,
            root=base_root,
        )

    latest = resolve_latest_verified_run_root(
        version=version,
        experiment_id=exp_norm,
        arx=arx,
        runmodus=runmodus_norm,
        n_ensemble=n_ensemble,
        kind=key_kind,
        root=base_root,
        catalog_path=catalog_path,
        require_verified=require_verified,
    )
    if latest is not None:
        return latest

    canonical_base = build_era5spliced_dataset_path(
        version=version,
        experiment_id=exp_norm,
        arx=arx,
        runmodus=runmodus_norm,
        n_ensemble=n_ensemble,
        kind=key_kind,
        run_instance=None,
        root=base_root,
    )
    latest_existing = _latest_run_from_existing_dir(canonical_base)
    if latest_existing is not None:
        return latest_existing

    if legacy_fallback_enabled():
        # Legacy canonical tree fallback.
        legacy_base = build_era5spliced_dataset_path(
            version=version,
            experiment_id=exp_norm,
            arx=arx,
            runmodus=runmodus_norm,
            n_ensemble=n_ensemble,
            kind=key_kind,
            run_instance=None,
            root=get_era5spliced_legacy_root(),
        )
        legacy_run = _latest_run_from_existing_dir(legacy_base)
        if legacy_run is not None:
            return legacy_run

        # created_nc_files fallback for emergency rollback mode.
        legacy_created = _legacy_created_nc_candidates(
            version=version,
            experiment_id=exp_norm,
            arx=arx,
            runmodus=runmodus_norm,
            n_ensemble=n_ensemble,
            kind=key_kind,
        )
        if legacy_created:
            return legacy_created[-1]

    return canonical_base


def resolve_dataderivatives_root(
    *,
    version: str,
    experiment_id: str,
    arx: str,
    runmodus: str,
    n_ensemble: str | int,
    run_instance: str | None = None,
    root: Path | None = None,
    catalog_path: Path | None = None,
    require_verified: bool = True,
) -> Path:
    """
    Resolve canonical dataderivatives dataset root.
    """
    return resolve_canonical_dataset_root(
        version=version,
        experiment_id=experiment_id,
        arx=arx,
        runmodus=runmodus,
        n_ensemble=n_ensemble,
        kind=CANONICAL_KIND_DATADERIVATIVES,
        run_instance=run_instance,
        root=root,
        catalog_path=catalog_path,
        require_verified=require_verified,
    )


def resolve_s3_site_for_version(version: str | None, *, fallback_site: str | None = None) -> str:
    token = _normalize_version_token(version)
    preferred = KNOWN_VERSION_HOME_SITE.get(token)
    if preferred in SITE_VALUES:
        return preferred
    if token.startswith("v101"):
        return SITE_ETH
    if token.startswith("v100"):
        return SITE_GUS
    return _norm_site(fallback_site) or get_site()


def convert_local_path_to_s3_uri_candidates(
    path: Path | str,
    *,
    bucket: str | None = None,
    site: str | None = None,
) -> list[str]:
    """
    Best-effort conversion of local canonical/legacy paths to ordered s3:// URI candidates.

    Ordering policy:
    1) consolidated canonical prefix
    2) site/version canonical prefix
    3) created_nc_files fallback prefixes (when source path is under created_nc_files)
    """
    raw = str(path).strip()
    if not raw:
        return []
    if raw.startswith("s3://"):
        return [raw]

    p = Path(raw).expanduser().resolve(strict=False)
    bucket_value = str(bucket or get_object_bucket()).strip()
    if not bucket_value:
        return []

    out: list[str] = []
    seen: set[str] = set()

    def _push(prefix: str) -> None:
        token = str(prefix).strip().strip("/")
        if not token:
            return
        uri = f"s3://{bucket_value}/{token}"
        if uri in seen:
            return
        seen.add(uri)
        out.append(uri)

    canonical = parse_era5spliced_dataset_path(p)
    if canonical is not None:
        rel = p.relative_to(Path(canonical["root"]))
        version = canonical["version"]
        site_value = resolve_s3_site_for_version(version, fallback_site=site)
        rel_posix = rel.as_posix()
        _push(f"nc/consolidated/era5spliced/{rel_posix}")
        _push(f"nc/{site_value}/{version}/era5spliced/{rel_posix}")
        return out

    created_root = get_created_nc_files_root().expanduser().resolve(strict=False)
    try:
        rel = p.relative_to(created_root)
    except ValueError:
        return out

    version = next((part for part in rel.parts if part.lower().startswith("v10")), get_version_default())
    site_value = resolve_s3_site_for_version(version, fallback_site=site)
    rel_posix = rel.as_posix()
    _push(f"nc/consolidated/created_nc_files/{rel_posix}")
    _push(f"nc/{site_value}/{version}/created_nc_files/{rel_posix}")
    return out


def convert_local_path_to_s3_uri(
    path: Path | str,
    *,
    bucket: str | None = None,
    site: str | None = None,
) -> str | None:
    """
    Best-effort conversion of local canonical/legacy paths to s3:// URIs.
    """
    raw = str(path).strip()
    if not raw:
        return None
    if raw.startswith("s3://"):
        return raw

    p = Path(raw).expanduser().resolve(strict=False)
    bucket_value = str(bucket or get_object_bucket()).strip()
    if bucket_value:
        mount_uri = _s3_uri_from_mount_entry(
            p,
            get_mount_entry_for_path(p, resolve_path=False),
            bucket=bucket_value,
        )
        if mount_uri:
            return mount_uri

        canonical = parse_era5spliced_dataset_path(p)
        if canonical is not None:
            rel = p.relative_to(Path(canonical["root"]))
            version = canonical["version"]
            site_value = resolve_s3_site_for_version(version, fallback_site=site)
            prefix = f"nc/{site_value}/{version}/era5spliced/{rel.as_posix()}"
            return f"s3://{bucket_value}/{prefix}"

        created_root = get_created_nc_files_root().expanduser().resolve(strict=False)
        try:
            rel = p.relative_to(created_root)
        except ValueError:
            rel = None
        if rel is not None:
            version = next(
                (part for part in rel.parts if part.lower().startswith("v10")),
                get_version_default(),
            )
            site_value = resolve_s3_site_for_version(version, fallback_site=site)
            prefix = f"nc/{site_value}/{version}/created_nc_files/{rel.as_posix()}"
            return f"s3://{bucket_value}/{prefix}"

    candidates = convert_local_path_to_s3_uri_candidates(
        p,
        bucket=(bucket_value or bucket),
        site=site,
    )
    return candidates[0] if candidates else None


def _normalize_derivatives_layout(layout: str | None) -> str:
    raw = str(layout or DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE).strip().lower()
    if raw in DERIVATIVES_LAYOUT_CHOICES:
        return raw
    raise ValueError(
        f"Unsupported derivatives layout '{layout}'. "
        f"Choose one of: {', '.join(DERIVATIVES_LAYOUT_CHOICES)}."
    )


def _normalize_derivatives_kind(kind: str | None) -> str:
    token = str(kind or DERIVATIVES_KIND_DATA).strip()
    if token in DERIVATIVES_KIND_CHOICES:
        return token
    raise ValueError(
        f"Unsupported derivatives kind '{kind}'. "
        f"Choose one of: {', '.join(DERIVATIVES_KIND_CHOICES)}."
    )


def _normalize_run_suffix(suffix: str | None) -> str:
    token = str(suffix or DEFAULT_DERIVATIVES_RUN_SUFFIX).strip()
    return token or DEFAULT_DERIVATIVES_RUN_SUFFIX


def parse_created_nc_ar_path(base_root: Path) -> dict[str, str] | None:
    """
    Parse a created_nc_files AR-style path.

    Expected shape:
      <created_root>/<run>/debias/<version>/<scenario>/<AR*>
    """
    path = Path(base_root).expanduser().resolve(strict=False)
    created_root = get_created_nc_files_root().expanduser().resolve(strict=False)
    try:
        rel = path.relative_to(created_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 5:
        return None
    run, layer, version, scenario, ar_folder = parts[:5]
    if layer != "debias":
        return None
    if not ar_folder.upper().startswith("AR"):
        return None
    return {
        "run": run,
        "version": version,
        "scenario": scenario,
        "ar_folder": ar_folder,
        "created_root": str(created_root),
    }


def resolve_parallel_derivatives_root(
    base_root: Path,
    *,
    suffix: str = DEFAULT_DERIVATIVES_RUN_SUFFIX,
    kind: str = DERIVATIVES_KIND_DATA,
) -> Path:
    """
    Resolve derivatives root for parallel sibling layout.

    Generic:
      X -> X{suffix}/{kind}

    created_nc AR-specialized:
      <created_root>/<run>/debias/<version>/<scenario>/<AR*> ->
      <created_root>/<run>{suffix}/debias/<version>/<scenario>/<AR*>/{kind}
    """
    resolved = Path(base_root).expanduser().resolve(strict=False)
    norm_kind = _normalize_derivatives_kind(kind)
    norm_suffix = _normalize_run_suffix(suffix)

    canonical = parse_era5spliced_dataset_path(resolved)
    if canonical is not None:
        root = Path(canonical["root"])
        out = (
            root
            / canonical["version"]
            / canonical["experiment_id"]
            / canonical["arx"]
            / canonical["runmodus"]
            / canonical["n_ensemble"]
            / CANONICAL_KIND_DATADERIVATIVES
        )
        run_instance = canonical.get("run_instance", "").strip()
        if run_instance:
            out = out / run_instance
        return out / norm_kind

    parsed = parse_created_nc_ar_path(resolved)
    if parsed is not None:
        created_root = Path(parsed["created_root"])
        run = parsed["run"]
        version = parsed["version"]
        scenario = parsed["scenario"]
        ar_folder = parsed["ar_folder"]
        return (
            created_root
            / f"{run}{norm_suffix}"
            / "debias"
            / version
            / scenario
            / ar_folder
            / norm_kind
        )

    if resolved.name.endswith(norm_suffix):
        return resolved / norm_kind
    return resolved.with_name(f"{resolved.name}{norm_suffix}") / norm_kind


def resolve_derivatives_root(
    base_root: Path,
    *,
    layout: str = DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE,
    suffix: str = DEFAULT_DERIVATIVES_RUN_SUFFIX,
    kind: str = DERIVATIVES_KIND_DATA,
) -> Path:
    """
    Resolve the derivatives root according to layout policy.
    """
    norm_layout = _normalize_derivatives_layout(layout)
    norm_kind = _normalize_derivatives_kind(kind)
    resolved = Path(base_root).expanduser().resolve(strict=False)
    if norm_layout == DERIVATIVES_LAYOUT_INPLACE:
        return resolved / norm_kind
    return resolve_parallel_derivatives_root(resolved, suffix=suffix, kind=norm_kind)


def resolve_speix_root(
    base_root: Path,
    *,
    layout: str = DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE,
    suffix: str = DEFAULT_DERIVATIVES_RUN_SUFFIX,
    kind: str = DERIVATIVES_KIND_DATA,
    fallback_to_inplace: bool = True,
    warn: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Resolve the SPEIx root under derivatives.

    Backward compatibility behavior:
    If a parallel path is requested but missing, and inplace SPEIx exists,
    optionally fall back to inplace and emit a warning.
    """
    norm_layout = _normalize_derivatives_layout(layout)
    norm_kind = _normalize_derivatives_kind(kind)
    primary = resolve_derivatives_root(
        base_root,
        layout=norm_layout,
        suffix=suffix,
        kind=norm_kind,
    ) / "SPEIx"
    if norm_layout != DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE or primary.exists():
        return primary

    inplace = resolve_derivatives_root(
        base_root,
        layout=DERIVATIVES_LAYOUT_INPLACE,
        suffix=suffix,
        kind=norm_kind,
    ) / "SPEIx"
    if fallback_to_inplace and inplace.exists():
        if warn is not None:
            warn(
                "Parallel derivatives path not found; falling back to legacy in-place path: "
                f"{inplace}"
            )
        return inplace
    return primary


def get_newscenario_inputs_root() -> Path:
    override = _env_path("GCMAGICC_NEWSCENARIO_INPUTS_ROOT")
    if override:
        return override
    return get_repo_path("gcmmagicc") / "data" / "newscenario_inputs"


def get_default_gxe_candidate_folders(version: str = "v101gxe") -> list[Path]:
    """
    Return ordered GXE candidate folders (AR6 leaf directories with NetCDF files).

    Overrides:
    - `GCMAGICC_<VERSION>_CANDIDATES` (e.g. `GCMAGICC_V100GXE_CANDIDATES`)
    - `GCMAGICC_GXE_CANDIDATES`
    """
    from os import pathsep

    normalized = _normalize_version_token(version) or "v101gxe"
    base_version = normalized[:-3] if normalized.endswith("gxe") else normalized

    for env_key in (_version_candidates_env_var(normalized), "GCMAGICC_GXE_CANDIDATES"):
        env_raw = os.environ.get(env_key, "").strip()
        if not env_raw:
            continue
        out: list[Path] = []
        for token in env_raw.split(pathsep):
            token = token.strip()
            if not token:
                continue
            p = Path(token).expanduser()
            if not p.is_absolute():
                p = get_created_nc_files_root() / p
            out.append(p.resolve(strict=False))
        deduped = _dedupe_paths(out)
        if deduped:
            return deduped

    canonical_candidates: list[Path] = []
    for runmodus in (RUNMODUS_ALL, RUNMODUS_NAT):
        for n_ensemble in ("n_20", "n_100"):
            canonical_candidates.append(
                resolve_canonical_dataset_root(
                    version=base_version,
                    experiment_id="ssp245",
                    arx="AR6",
                    runmodus=runmodus,
                    n_ensemble=n_ensemble,
                    kind=CANONICAL_KIND_ORIGINAL,
                    root=get_era5spliced_root(),
                )
            )
    canonical_deduped = _dedupe_paths(canonical_candidates)
    roots = [get_created_nc_files_root()]
    for site in _site_order_for_version(normalized):
        roots.extend(_created_nc_roots_for_site(site))
    roots = _dedupe_paths(roots)

    def _is_derivatives_path(path: Path) -> bool:
        return any(part.endswith("_dataderivatives") for part in path.parts)

    preferred: list[Path] = []
    fallback: list[Path] = []
    for root in roots:
        if not _safe_exists(root):
            continue
        preferred.extend(
            p.resolve(strict=False)
            for p in sorted(_safe_glob(root, f"*/debias/{base_version}/ssp245/AR6"), reverse=True)
            if _safe_is_dir(p) and not _is_derivatives_path(p)
        )
        fallback.extend(
            p.resolve(strict=False)
            for p in sorted(_safe_glob(root, f"*/debias/{base_version}/*/AR6"), reverse=True)
            if _safe_is_dir(p) and not _is_derivatives_path(p)
        )
    deduped_preferred = _dedupe_paths(preferred)
    if deduped_preferred:
        return deduped_preferred
    deduped_fallback = _dedupe_paths(fallback)
    if deduped_fallback:
        return deduped_fallback

    canonical_existing: list[Path] = []
    canonical_accessible_missing: list[Path] = []
    for p in canonical_deduped:
        exists, accessible = _safe_exists_status(p)
        if exists:
            canonical_existing.append(p)
        elif accessible:
            canonical_accessible_missing.append(p)
    if canonical_existing:
        return canonical_existing

    if canonical_accessible_missing and not legacy_fallback_enabled():
        return canonical_accessible_missing

    if not legacy_fallback_enabled():
        return []

    # Compatibility fallbacks for historically used folders.
    fallback_roots = _dedupe_paths([get_created_nc_files_root(), ETH_CREATED_NC_ROOT, GUS_CREATED_NC_ROOT])
    static_fallback: list[Path] = []
    for root in fallback_roots:
        static_fallback.extend(
            [
                root / f"debiasloop_ensembles_20260123-0644/debias/{base_version}/ssp245/AR6",
                root / f"debiasloop_20NDCsSSPs_20260123-0644/debias/{base_version}/ssp245/AR6",
                root / f"debiasloop_20NDCssps_20260220-1950/debias/{base_version}/ssp245/AR6",
                root / f"debiasloop_100ssp245plusnatv100_20260223-0301/debias/{base_version}/ssp245/AR6",
            ]
        )
    return _dedupe_paths(static_fallback)


def resolve_v100gxe_so_root() -> Path:
    """
    Resolve the v100gxe S(o) source folder used for GOFON/GOFNN.

    Priority:
    1) `GCMAGICC_V100GXE_SO_ROOT`
    2) fixed campaign default under created_nc_files
    """
    override_raw = os.environ.get("GCMAGICC_V100GXE_SO_ROOT", "").strip()
    if override_raw:
        override_path = Path(override_raw).expanduser()
        if not override_path.is_absolute():
            override_path = get_created_nc_files_root() / override_path
        return override_path.resolve(strict=False)

    return (
        get_created_nc_files_root()
        / "debiasloop_20NDCssps_20260220-1950"
        / "debias"
        / "v100"
        / "ssp245"
        / "AR6"
    ).resolve(strict=False)


def get_object_remote() -> str:
    env_value = os.environ.get("GCMAGICC_OBJECT_REMOTE", "").strip()
    if env_value:
        return env_value
    return get_rclone_remote_default()


def get_object_bucket() -> str:
    env_value = os.environ.get("GCMAGICC_OBJECT_BUCKET", "").strip()
    if env_value:
        return env_value
    return get_s3_env_file_values().get("GCMAGICC_OBJECT_BUCKET", "gcmagicc-scratch").strip()


def get_object_mount() -> Path:
    return Path(os.environ.get("GCMAGICC_OBJECT_MOUNT", "/mnt/gcmagicc-ovh")).expanduser()


def get_object_mode() -> str:
    return os.environ.get("GCMAGICC_OBJECT_MODE", "nc_only").strip()


def get_object_prefix(site: str | None = None, version: str | None = None) -> str:
    site_value = _norm_site(site) or get_site()
    version_value = (version or get_version_default()).strip()
    return f"nc/{site_value}/{version_value}"


# Convenience functions for common local repo paths
def get_reports_path() -> str:
    return get_output_folder("reports")


def get_metric_databases_path() -> str:
    return str(get_data_path("metric_databases"))


def get_metric_job_sentinels_path() -> str:
    return str(get_data_path("metric_databases/job_sentinels"))


def get_archived_databases_path() -> str:
    return str(get_data_path("archived_databases"))


def get_edist_databases_path() -> str:
    return str(get_data_path("edist_databases"))


def get_static_data_path() -> str:
    return str(get_data_path("static"))


def get_breadbasket_path() -> str:
    return str(get_data_path("breadbasket"))


def get_debug_path() -> str:
    return str(get_data_path("debug"))


def get_logs_path() -> str:
    return str(get_project_root() / "logs")


def test_paths() -> None:
    print(f"Project root: {get_project_root()}")
    print(f"Projects root: {get_projects_root()}")
    print(f"Site: {get_site()}")
    print(f"Data profile: {get_data_profile() or 'default'}")
    print(f"Data root: {get_data_root()}")
    print(f"Scratch data root: {get_site_scratch_data_root()}")
    print(f"Default version: {get_version_default()}")
    print(f"GCMagicc path: {get_gcmagicc_path()}")
    print(f"CMIP6 root: {get_cmip6_vetted_path()}")
    print(f"ERA5 root: {get_era5_vetted_path()}")
    print(f"ERA5 file: {get_era5_main_file()}")
    print(f"ERA5 0.25 root: {get_era5_025_vetted_path()}")
    print(f"ERA5 0.25 file: {get_era5_025_main_file()}")
    print(f"created_nc_files root: {get_created_nc_files_root()}")
    print(f"ERA5spliced root: {get_era5spliced_root()}")
    print(f"ERA5spliced local staging root: {get_era5spliced_localstaging_root()}")
    print(f"ERA5spliced localresults root: {get_era5spliced_localresults_root()}")
    print(f"ERA5spliced CMIP6 localresults root: {get_era5spliced_cmip6_localresults_root()}")
    print(f"CMIP6replicas root: {get_cmip6replicas_root()}")
    print(f"CMIP6replicas local staging root: {get_cmip6replicas_localstaging_root()}")
    print(f"CMIP6 local staging root: {get_cmip6_localstaging_root()}")
    print(f"Fressnapf CMIP6 HEALPix root: {get_fressnapf_cmip6_healpix_root()}")
    print(f"Fressnapf ERA5 HEALPix root: {get_fressnapf_era5_healpix_root()}")
    print(f"Object remote: {get_object_remote()}")
    print(f"Object bucket: {get_object_bucket()}")
    print(f"Object mode: {get_object_mode()}")


if __name__ == "__main__":
    test_paths()
