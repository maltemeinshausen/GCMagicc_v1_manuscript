"""
Helpers for stable, version-independent ERA5/CMIP6 overlay caches.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from scr.validation_helpers import helper_path_utils as _paths


OVERLAY_CANONICAL_TAG = (
    os.environ.get("GCMAGICC_OVERLAY_CANONICAL_TAG", "overlay_canonical").strip()
    or "overlay_canonical"
)
DEFAULT_MIN_PIXELS_FOR_COUNTRY = max(
    1,
    int(os.environ.get("GCMAGICC_OVERLAY_MIN_PIXELS_FOR_COUNTRY", "9") or "9"),
)
WORLD_SPEIX_DIRNAME = os.environ.get("GCMAGICC_755_PRODUCT_DIRNAME", "755_SPEIx").strip() or "755_SPEIx"
OVERLAY_SPEIX_DIRNAME = "SPEIx"

SOURCE_ERA5 = "era5"
SOURCE_HISTORICAL = "historical"
SOURCE_HIST_NAT = "hist-nat"
SOURCE_SSP245 = "ssp245"
SOURCE_KEYS = (SOURCE_ERA5, SOURCE_HISTORICAL, SOURCE_HIST_NAT, SOURCE_SSP245)
CMIP6_OVERLAY_EXPECTED_BASELINE_SOURCE_KEY = "historical"
CMIP6_OVERLAY_EXPECTED_BASELINE_POOLING = "per_member"

OVERLAY_MISSING_POLICY_WARN = "warn"
OVERLAY_MISSING_POLICY_CHOICES = (OVERLAY_MISSING_POLICY_WARN,)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILDER_SCRIPT = _REPO_ROOT / "notebooks" / "761_build_overlay_canonical.py"
ETH_REPO_ROOT = (_paths.ETH_PROJECTS_ROOT / "gcmmagicc").expanduser().resolve(strict=False)
_REMOTE_EXISTENCE_CACHE: Dict[Tuple[str, ...], Optional[Path]] = {}


def safe_token(value: str, *, fallback: str = "unknown") -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or fallback


def normalize_pet_method(value: str) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("pet-"):
        token = token[4:]
    return token


def token_has_nat_suffix(value: str) -> bool:
    return str(value or "").strip().lower().endswith("-nat")


def token_mentions_ssp245(value: str) -> bool:
    return "ssp245" in str(value or "").strip().lower()


def default_forcing_label(scenario_tag: str) -> str:
    token = str(scenario_tag or "").strip()
    if token.upper() == "ERA5":
        return "ERA5"
    return "SCENARIO2" if token_has_nat_suffix(token) else "SCENARIO1"


def required_year_for_scenario(scenario_tag: str) -> int:
    return 2050 if token_mentions_ssp245(scenario_tag) else 2000


def stacked_output_label(forcing_label: str, *, scenario_tag: Optional[str] = None) -> str:
    if str(forcing_label).strip().upper() == "ERA5":
        return "ERA5"
    scenario_token = str(scenario_tag or "").strip()
    if scenario_token:
        return safe_token(scenario_token)
    return safe_token(forcing_label)


def stacked_label_candidates(forcing_label: str, *, scenario_tag: Optional[str] = None) -> List[str]:
    labels: List[str] = []
    for candidate in (
        stacked_output_label(forcing_label, scenario_tag=scenario_tag),
        safe_token(forcing_label),
    ):
        if candidate and candidate not in labels:
            labels.append(candidate)
    return labels


def normalize_source_keys(values: Sequence[str] | str | None) -> List[str]:
    if values is None:
        return list(SOURCE_KEYS)
    if isinstance(values, str):
        raw_items = values.split(",")
    else:
        raw_items = list(values)
    out: List[str] = []
    for raw in raw_items:
        token = str(raw or "").strip().lower()
        if not token:
            continue
        if token not in SOURCE_KEYS:
            raise ValueError(
                f"Unsupported overlay source '{raw}'. "
                f"Choose from: {', '.join(SOURCE_KEYS)}."
            )
        if token not in out:
            out.append(token)
    return out or list(SOURCE_KEYS)


def overlay_source_scenario_tag(source_key: str) -> str:
    token = str(source_key or "").strip().lower()
    if token == SOURCE_ERA5:
        return "ERA5"
    if token == SOURCE_HISTORICAL:
        return "historical"
    if token == SOURCE_HIST_NAT:
        return "hist-nat"
    if token == SOURCE_SSP245:
        return "ssp245"
    raise ValueError(f"Unsupported overlay source: {source_key}")


def expected_overlay_baseline(source_key: str) -> Tuple[Optional[str], Optional[str]]:
    token = str(source_key or "").strip().lower()
    if token == SOURCE_ERA5:
        return "era5", "pooled"
    if token in {SOURCE_HISTORICAL, SOURCE_HIST_NAT, SOURCE_SSP245}:
        return (
            CMIP6_OVERLAY_EXPECTED_BASELINE_SOURCE_KEY,
            CMIP6_OVERLAY_EXPECTED_BASELINE_POOLING,
        )
    raise ValueError(f"Unsupported overlay source: {source_key}")


def overlay_metadata_matches_expected(
    metadata: Dict[str, object],
    *,
    source_key: str,
) -> Tuple[bool, str]:
    expected_source, expected_pooling = expected_overlay_baseline(source_key)
    actual_source = str(metadata.get("baseline_source_key") or metadata.get("baseline_source") or "").strip().lower()
    actual_pooling = str(metadata.get("baseline_pooling") or "").strip().lower()
    if expected_source and actual_source != str(expected_source).strip().lower():
        return (
            False,
            f"baseline_source_key={actual_source or '<missing>'} "
            f"(expected {expected_source})",
        )
    if expected_pooling and actual_pooling != str(expected_pooling).strip().lower():
        return (
            False,
            f"baseline_pooling={actual_pooling or '<missing>'} "
            f"(expected {expected_pooling})",
        )
    return True, "ok"


def resolve_overlay_world_root(
    base_root: Path,
    *,
    derivatives_layout: str = _paths.DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE,
    derivatives_run_suffix: str = _paths.DEFAULT_DERIVATIVES_RUN_SUFFIX,
) -> Path:
    return (
        _paths.resolve_derivatives_root(
            base_root,
            layout=derivatives_layout,
            suffix=derivatives_run_suffix,
            kind="data_derivatives",
        )
        / WORLD_SPEIX_DIRNAME
    ).expanduser().resolve(strict=False)


def resolve_overlay_speix_root(
    base_root: Path,
    *,
    derivatives_layout: str = _paths.DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE,
    derivatives_run_suffix: str = _paths.DEFAULT_DERIVATIVES_RUN_SUFFIX,
) -> Path:
    return (
        _paths.resolve_derivatives_root(
            base_root,
            layout=derivatives_layout,
            suffix=derivatives_run_suffix,
            kind="data_derivatives",
        )
        / OVERLAY_SPEIX_DIRNAME
    ).expanduser().resolve(strict=False)


def get_era5_overlay_world_root(
    *,
    derivatives_layout: str = _paths.DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE,
    derivatives_run_suffix: str = _paths.DEFAULT_DERIVATIVES_RUN_SUFFIX,
) -> Path:
    return resolve_overlay_world_root(
        _paths.get_era5_vetted_path(),
        derivatives_layout=derivatives_layout,
        derivatives_run_suffix=derivatives_run_suffix,
    )


def get_era5_overlay_speix_root(
    *,
    derivatives_layout: str = _paths.DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE,
    derivatives_run_suffix: str = _paths.DEFAULT_DERIVATIVES_RUN_SUFFIX,
) -> Path:
    return resolve_overlay_speix_root(
        _paths.get_era5_vetted_path(),
        derivatives_layout=derivatives_layout,
        derivatives_run_suffix=derivatives_run_suffix,
    )


def get_cmip6_overlay_world_root(
    *,
    derivatives_layout: str = _paths.DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE,
    derivatives_run_suffix: str = _paths.DEFAULT_DERIVATIVES_RUN_SUFFIX,
) -> Path:
    return resolve_overlay_world_root(
        _paths.get_cmip6_vetted_path(),
        derivatives_layout=derivatives_layout,
        derivatives_run_suffix=derivatives_run_suffix,
    )


def get_cmip6_overlay_speix_root(
    *,
    derivatives_layout: str = _paths.DERIVATIVES_LAYOUT_PARALLEL_RUN_TREE,
    derivatives_run_suffix: str = _paths.DEFAULT_DERIVATIVES_RUN_SUFFIX,
) -> Path:
    return resolve_overlay_speix_root(
        _paths.get_cmip6_vetted_path(),
        derivatives_layout=derivatives_layout,
        derivatives_run_suffix=derivatives_run_suffix,
    )


def resolve_world_speix_root(root: Path, *, label: str) -> Path:
    candidate = Path(root).expanduser().resolve(strict=False)
    if candidate.name in {OVERLAY_SPEIX_DIRNAME, WORLD_SPEIX_DIRNAME}:
        return candidate
    if (candidate / OVERLAY_SPEIX_DIRNAME).exists():
        return candidate / OVERLAY_SPEIX_DIRNAME
    if (candidate / WORLD_SPEIX_DIRNAME).exists():
        return candidate / WORLD_SPEIX_DIRNAME
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"{label} world SPEIx root not found: {candidate}")


def resolve_world_pet_root(
    root: Path,
    *,
    tag: Optional[str],
    pet_method: str,
    label: str,
) -> Path:
    world_root = resolve_world_speix_root(root, label=label)
    pet_dir_name = f"pet-{normalize_pet_method(pet_method)}"
    if tag and str(tag).strip():
        tagged = world_root / str(tag).strip() / pet_dir_name
        if tagged.exists():
            return tagged
        raise FileNotFoundError(f"{label} world PET root not found: {tagged}")

    direct = world_root / pet_dir_name
    if direct.exists():
        return direct

    tagged_candidates: List[Tuple[str, Path]] = []
    if world_root.exists():
        for child in sorted(world_root.iterdir()):
            if not child.is_dir():
                continue
            pet_dir = child / pet_dir_name
            if pet_dir.exists():
                tagged_candidates.append((child.name, pet_dir))
    if tagged_candidates:
        tagged_candidates.sort(key=lambda item: item[0])
        return tagged_candidates[-1][1]

    raise FileNotFoundError(
        f"{label} world PET root not found under {world_root} "
        f"(expected {pet_dir_name} directly or inside a tag directory)."
    )


def world_file_window_from_name(name: str, *, label_token: str, scale: int) -> Optional[Tuple[int, int]]:
    match = re.match(
        rf"^{re.escape(label_token)}__spei{int(scale)}__ALLLAND__grid__(\d+)-(\d+)__all\.nc$",
        name,
        re.IGNORECASE,
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def discover_world_files(
    pet_root: Path,
    *,
    forcing_label: str,
    scenario_tag: Optional[str],
    scale: int,
    required_year: Optional[int] = None,
) -> List[Path]:
    scored: List[Tuple[int, int, Path]] = []
    seen: set[str] = set()
    for label_token in stacked_label_candidates(forcing_label, scenario_tag=scenario_tag):
        stacked_root = pet_root / "stacked" / label_token
        if not stacked_root.exists():
            continue
        for path in sorted(stacked_root.glob("*.nc")):
            key = str(path.resolve(strict=False))
            if key in seen:
                continue
            window = world_file_window_from_name(path.name, label_token=label_token, scale=scale)
            if window is None:
                continue
            start, end = window
            if required_year is not None and not (start <= int(required_year) <= end):
                continue
            seen.add(key)
            scored.append((int(end - start), int(end), path))
    scored.sort(key=lambda item: (item[0], item[1], item[2].name), reverse=True)
    return [path for _dur, _end, path in scored]


def world_cache_has_content(
    world_root: Path,
    *,
    tag: Optional[str],
    pet_method: str,
    forcing_label: str,
    scenario_tag: Optional[str],
    scale: int,
    required_year: Optional[int] = None,
) -> bool:
    try:
        pet_root = resolve_world_pet_root(
            world_root,
            tag=tag,
            pet_method=pet_method,
            label=forcing_label,
        )
    except Exception:
        return False
    fits_dir = pet_root / "fits"
    world_files = discover_world_files(
        pet_root,
        forcing_label=forcing_label,
        scenario_tag=scenario_tag,
        scale=scale,
        required_year=required_year,
    )
    return bool(world_files) and fits_dir.exists()


def inspect_world_overlay_metadata(
    world_root: Path,
    *,
    tag: Optional[str],
    pet_method: str,
    source_key: str,
    scale: int = 48,
) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "source_key": str(source_key).strip().lower(),
        "world_root": str(Path(world_root).expanduser().resolve(strict=False)),
        "tag": str(tag).strip() if tag is not None else None,
        "pet_method": normalize_pet_method(pet_method),
        "baseline_source_key": None,
        "baseline_pooling": None,
        "baseline_id": None,
        "baseline_strategy": None,
        "world_file": None,
        "valid_for_final_publish": False,
        "reason": "uninitialized",
    }
    try:
        scenario_tag = overlay_source_scenario_tag(source_key)
        forcing_label = "ERA5" if str(source_key).strip().lower() == SOURCE_ERA5 else default_forcing_label(scenario_tag)
        required_year = None if str(source_key).strip().lower() == SOURCE_ERA5 else required_year_for_scenario(scenario_tag)
        pet_root = resolve_world_pet_root(
            world_root,
            tag=tag,
            pet_method=pet_method,
            label=forcing_label,
        )
        world_files = discover_world_files(
            pet_root,
            forcing_label=forcing_label,
            scenario_tag=None if str(source_key).strip().lower() == SOURCE_ERA5 else scenario_tag,
            scale=int(scale),
            required_year=required_year,
        )
        if not world_files:
            metadata["reason"] = "world_file_missing"
            return metadata
        world_file = world_files[0]
        metadata["world_file"] = str(world_file)
        import xarray as xr  # local import to keep helper lightweight

        with xr.open_dataset(world_file, decode_times=False, engine="netcdf4") as ds:
            attrs = dict(ds.attrs)
        metadata["baseline_source_key"] = attrs.get("baseline_source_key") or attrs.get("baseline_source")
        metadata["baseline_pooling"] = attrs.get("baseline_pooling")
        metadata["baseline_id"] = attrs.get("baseline_id")
        metadata["baseline_strategy"] = attrs.get("baseline_strategy")
        ok, reason = overlay_metadata_matches_expected(metadata, source_key=source_key)
        metadata["valid_for_final_publish"] = bool(ok)
        metadata["reason"] = reason
        return metadata
    except Exception as exc:
        metadata["reason"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        return metadata


def stable_region_store_exists(
    speix_root: Path,
    *,
    tag: Optional[str],
    region: Optional[str],
    pet_method: Optional[str],
) -> bool:
    root = Path(speix_root).expanduser().resolve(strict=False)
    if tag and str(tag).strip():
        root = root / str(tag).strip()
    if region:
        region_token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(region).strip()).strip("_") or "REGION"
        root = root / f"region-{region_token}"
    if pet_method:
        root = root / f"pet-{normalize_pet_method(pet_method)}"
    return (root / "segments.zarr").exists()


def _remote_first_existing_path(host: str, candidates: Sequence[Path]) -> Optional[Path]:
    key = (str(host),) + tuple(str(Path(p).expanduser().resolve(strict=False)) for p in candidates)
    if key in _REMOTE_EXISTENCE_CACHE:
        return _REMOTE_EXISTENCE_CACHE[key]

    quoted = " ".join(shlex.quote(str(Path(p).expanduser().resolve(strict=False))) for p in candidates)
    script = (
        f"for p in {quoted}; do "
        'if [ -e "$p" ]; then printf "%s\\n" "$p"; exit 0; fi; '
        "done; exit 1"
    )
    try:
        cp = subprocess.run(
            ["ssh", str(host), script],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        cp = None

    if cp is not None and cp.returncode == 0 and cp.stdout.strip():
        found = Path(cp.stdout.strip().splitlines()[0]).expanduser().resolve(strict=False)
        _REMOTE_EXISTENCE_CACHE[key] = found
        return found

    _REMOTE_EXISTENCE_CACHE[key] = None
    return None


def _overlay_family_and_kind(path: Path) -> Optional[Tuple[str, str, Path]]:
    resolved = Path(path).expanduser().resolve(strict=False)
    roots = (
        (get_era5_overlay_world_root(), "era5", "world"),
        (get_era5_overlay_speix_root(), "era5", "speix"),
        (get_cmip6_overlay_world_root(), "cmip6", "world"),
        (get_cmip6_overlay_speix_root(), "cmip6", "speix"),
    )
    for root, family, kind in roots:
        try:
            rel = resolved.relative_to(root)
            return family, kind, rel
        except ValueError:
            continue
    return None


def _gus_base_candidates_for_family(family: str) -> List[Path]:
    if _paths.is_fressnapf_data_profile():
        if family == "cmip6":
            return [
                _paths.FRESSNAPF_CMIP6_VETTED_ROOT.resolve(strict=False),
                (_paths.FRESSNAPF_DATA_ROOT / "CMIP6" / "ETHFOG").resolve(strict=False),
            ]
        if family == "era5":
            return [
                _paths.FRESSNAPF_ERA5_VETTED_ROOT.resolve(strict=False),
                _paths.FRESSNAPF_ERA5_025_VETTED_ROOT.resolve(strict=False),
                (_paths.FRESSNAPF_DATA_ROOT / "ERA5" / "processed").resolve(strict=False),
            ]
        raise ValueError(f"Unsupported overlay family: {family}")

    if family == "cmip6":
        return [
            (_paths.GUS_DATA_ROOT / "out_ETHFOG_10June2025_vetted").resolve(strict=False),
            (_paths.GUS_DATA_ROOT / "cmip6_ETHFOG").resolve(strict=False),
        ]
    if family == "era5":
        return [
            (_paths.GUS_DATA_ROOT / "out_ERA5_19Feb2026_1degree_vetted").resolve(strict=False),
            (_paths.GUS_DATA_ROOT / "out_ERA5_4July2025_1degree_vetted").resolve(strict=False),
            (_paths.GUS_DATA_ROOT / "ERA5").resolve(strict=False),
        ]
    raise ValueError(f"Unsupported overlay family: {family}")


def gus_base_root_for_overlay_path(path: Path) -> Optional[Path]:
    overlay_info = _overlay_family_and_kind(path)
    if overlay_info is None:
        return None
    family, _kind, _rel = overlay_info
    candidates = _gus_base_candidates_for_family(family)
    return _remote_first_existing_path("gus", candidates) or candidates[0]


def mirror_eth_path_to_gus(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    overlay_info = _overlay_family_and_kind(resolved)
    if overlay_info is not None:
        family, kind, rel = overlay_info
        gus_base_root = gus_base_root_for_overlay_path(resolved)
        if gus_base_root is None:
            raise ValueError(f"Cannot resolve GUS base root for overlay family {family}: {resolved}")
        if kind == "world":
            gus_overlay_root = resolve_overlay_world_root(gus_base_root)
        elif kind == "speix":
            gus_overlay_root = resolve_overlay_speix_root(gus_base_root)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported overlay kind: {kind}")
        return (gus_overlay_root / rel).resolve(strict=False)
    for src_root, dst_root in (
        (_paths.ETH_DATA_ROOT, _paths.get_shared_data_root_for_site(_paths.SITE_GUS)),
        (_paths.ETH_PROJECTS_ROOT, _paths.GUS_PROJECTS_ROOT),
    ):
        try:
            rel = resolved.relative_to(src_root)
            return (dst_root / rel).resolve(strict=False)
        except ValueError:
            continue
    raise ValueError(f"Cannot map ETH path to GUS path: {resolved}")


def overlay_builder_command(
    *,
    sources: Sequence[str],
    sync_gus: bool = False,
    tag: str = OVERLAY_CANONICAL_TAG,
    pet_method: str = "all",
    scale: int = 48,
    cmip6_member_selection: str = "one-per-source",
) -> List[str]:
    cmd = [
        "pixi",
        "run",
        "python",
        str(DEFAULT_BUILDER_SCRIPT.relative_to(_REPO_ROOT)),
        "--sources",
        ",".join(normalize_source_keys(list(sources))),
        "--tag",
        str(tag).strip() or OVERLAY_CANONICAL_TAG,
        "--pet-method",
        str(pet_method).strip() or "all",
        "--scale",
        str(int(scale)),
        "--cmip6-member-selection",
        str(cmip6_member_selection).strip() or "one-per-source",
    ]
    if sync_gus:
        cmd.append("--sync-gus")
    return cmd


def format_builder_command_for_eth(
    *,
    sources: Sequence[str],
    sync_gus: bool = False,
    tag: str = OVERLAY_CANONICAL_TAG,
    pet_method: str = "all",
    scale: int = 48,
    cmip6_member_selection: str = "one-per-source",
) -> str:
    cmd = overlay_builder_command(
        sources=sources,
        sync_gus=sync_gus,
        tag=tag,
        pet_method=pet_method,
        scale=scale,
        cmip6_member_selection=cmip6_member_selection,
    )
    return f"cd {shlex.quote(str(ETH_REPO_ROOT))} && {shlex.join(cmd)}"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    return module


def resolve_overlay_regions(
    *,
    min_pixels: int = DEFAULT_MIN_PIXELS_FOR_COUNTRY,
    era5_file: Optional[Path] = None,
) -> List[str]:
    wrapper_path = _REPO_ROOT / "notebooks" / "754WRAP_SPEI_various_pet_methods.py"
    fallback_regions: List[str] = []
    try:
        module = _load_module(wrapper_path, "_overlay_regions_754wrap")
        fallback_regions = list(getattr(module, "DEFAULT_REGIONS", []))
    except Exception:
        fallback_regions = []

    plot810_path = _REPO_ROOT / "notebooks" / "810_plot_SSPprojections.py"
    try:
        import xarray as xr

        era5_target = Path(era5_file or _paths.get_era5_main_file()).expanduser().resolve(strict=False)
        module_810 = _load_module(plot810_path, "_overlay_regions_810")
        ds = xr.open_dataset(era5_target)
        try:
            regions = module_810.resolve_region_list("ALL", ds["lat"], ds["lon"], int(min_pixels))
        finally:
            ds.close()
        if regions:
            return list(regions)
    except Exception:
        pass

    return list(fallback_regions)
