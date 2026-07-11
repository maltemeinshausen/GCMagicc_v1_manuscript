#!/usr/bin/env python3
"""
759Wrapper_SPEI_drought_fromWorld

Parallel wrapper around 759_GapFiller_SPEI_fromWorld.py.

Differences versus 758Wrapper:
- Reads 755 world/all-land outputs instead of discovering region-scoped 754 stores.
- Uses the canonical expected-region list for --regions auto.
- Does not copy method artifacts; 759 outputs are the final products.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


try:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:  # pragma: no cover
    _cwd = Path.cwd()
    _REPO_ROOT = _cwd.parent if _cwd.name == "notebooks" else _cwd
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_helper_path_utils_file = _REPO_ROOT / "scr" / "validation_helpers" / "helper_path_utils.py"
_helper_path_utils_spec = importlib.util.spec_from_file_location(
    "_gcmagicc_helper_path_utils_wrapper_759",
    _helper_path_utils_file,
)
if _helper_path_utils_spec is None or _helper_path_utils_spec.loader is None:  # pragma: no cover
    raise ImportError(f"Failed to load helper_path_utils from {_helper_path_utils_file}")
_helper_path_utils = importlib.util.module_from_spec(_helper_path_utils_spec)
_helper_path_utils_spec.loader.exec_module(_helper_path_utils)


_base758_wrapper = importlib.import_module("notebooks.758Wrapper_SPEI_drought")
_base758 = importlib.import_module("notebooks.758_GapFiller_SPEI")
_base759 = importlib.import_module("notebooks.759_GapFiller_SPEI_fromWorld")
_base755 = importlib.import_module("notebooks.755_add_SPEI_allLand_to_ensemble_outputs")

build_era5spliced_dataset_path = _helper_path_utils.build_era5spliced_dataset_path
get_era5spliced_localresults_root = _helper_path_utils.get_era5spliced_localresults_root

ENV_759_SSP245_NAT_SCENARIO1_WORLD_ROOT = "GCMAGICC_759_SSP245_NAT_SCENARIO1_WORLD_ROOT"
ENV_759_SSP245_NAT_SCENARIO2_WORLD_ROOT = "GCMAGICC_759_SSP245_NAT_SCENARIO2_WORLD_ROOT"
ENV_759_CURPOL_NDCLOW_SCENARIO1_WORLD_ROOT = "GCMAGICC_759_CURPOL_NDCLOW_SCENARIO1_WORLD_ROOT"
ENV_759_CURPOL_NDCLOW_SCENARIO2_WORLD_ROOT = "GCMAGICC_759_CURPOL_NDCLOW_SCENARIO2_WORLD_ROOT"
ENV_759_ERA5_WORLD_ROOT = "GCMAGICC_759_ERA5_WORLD_ROOT"
ENV_759_ERA5_FILE = "GCMAGICC_759_ERA5_FILE"
ENV_759_OUTPUT_BASE_ROOT = "GCMAGICC_759_OUTPUT_BASE_ROOT"
DEFAULT_MAX_WORKERS = 50
HIGH_MEMORY_BATCH_MAX_WORKERS = 1
HIGH_MEMORY_REGION_CODES = frozenset({"ATA"})
HIGH_MEMORY_REGION_SUBSTRINGS = ("antarctica",)
DEFAULT_UNIFIED_PAYLOAD_MODE = os.environ.get("GCMAGICC_759_UNIFIED_PAYLOAD_MODE", "gzip").strip().lower() or "gzip"
THREAD_LIMIT_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _optional_env_path(name: str) -> Optional[Path]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve(strict=False)


def _thread_limited_env() -> Dict[str, str]:
    env = dict(os.environ)
    for name, value in THREAD_LIMIT_ENV_DEFAULTS.items():
        env.setdefault(name, value)
    return env


def _profile_world_roots(profile_id: str, *, version_tag: str) -> Tuple[Optional[Path], Optional[Path]]:
    localresults_root = Path(str(get_era5spliced_localresults_root())).expanduser().resolve(strict=False)
    if profile_id == "ssp245_nat":
        defaults = (
            build_era5spliced_dataset_path(
                version=version_tag,
                experiment_id="ssp245",
                arx="AR6",
                runmodus="all",
                n_ensemble="n_100",
                kind="dataderivatives",
                root=localresults_root,
            )
            / _base755.WORLD_SPEIX_DIRNAME,
            build_era5spliced_dataset_path(
                version=version_tag,
                experiment_id="ssp245",
                arx="AR6",
                runmodus="nat",
                n_ensemble="n_100",
                kind="dataderivatives",
                root=localresults_root,
            )
            / _base755.WORLD_SPEIX_DIRNAME,
        )
        env_s1 = _optional_env_path(ENV_759_SSP245_NAT_SCENARIO1_WORLD_ROOT)
        env_s2 = _optional_env_path(ENV_759_SSP245_NAT_SCENARIO2_WORLD_ROOT)
    elif profile_id == "curpol_ndclow":
        defaults = (
            build_era5spliced_dataset_path(
                version=version_tag,
                experiment_id="Current-Policies-GCAM",
                arx="AR6",
                runmodus="all",
                n_ensemble="n_100",
                kind="dataderivatives",
                root=localresults_root,
            )
            / _base755.WORLD_SPEIX_DIRNAME,
            build_era5spliced_dataset_path(
                version=version_tag,
                experiment_id="NDC-submitted-low",
                arx="AR6",
                runmodus="all",
                n_ensemble="n_100",
                kind="dataderivatives",
                root=localresults_root,
            )
            / _base755.WORLD_SPEIX_DIRNAME,
        )
        env_s1 = _optional_env_path(ENV_759_CURPOL_NDCLOW_SCENARIO1_WORLD_ROOT)
        env_s2 = _optional_env_path(ENV_759_CURPOL_NDCLOW_SCENARIO2_WORLD_ROOT)
    else:
        defaults = _base758_wrapper._profile_override_roots(profile_id, version_tag=version_tag)
        env_s1 = None
        env_s2 = None
    if defaults is None:
        return env_s1, env_s2
    return env_s1 or defaults[0], env_s2 or defaults[1]


def _build_cmd(
    *,
    script_path: Path,
    region: str,
    pet_method: str,
    scenario1_world_root: Path,
    scenario2_world_root: Path,
    era5_file: Path,
    output_base_root: Path,
    era5_world_root: Optional[Path],
    cmip6_world_root: Optional[Path],
    scenario1_world_tag: Optional[str],
    scenario2_world_tag: Optional[str],
    era5_world_tag: Optional[str],
    overlay_cache_tag: Optional[str],
    overlay_missing_policy: Optional[str],
    scenario1_tag: str,
    scenario2_tag: str,
    scenario1_label: str,
    scenario2_label: str,
    version_tag: str,
    output_timetag: str,
    include_cmip6: bool,
    unified_payload_mode: str,
) -> List[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--scenario1-world-root",
        str(scenario1_world_root),
        "--scenario2-world-root",
        str(scenario2_world_root),
        "--era5-file",
        str(era5_file),
        "--output-base-root",
        str(output_base_root),
        "--scenario1",
        scenario1_tag,
        "--scenario2",
        scenario2_tag,
        "--scenario1-label",
        scenario1_label,
        "--scenario2-label",
        scenario2_label,
        "--region",
        region,
        "--pet-method",
        pet_method,
        "--version-tag",
        version_tag,
        "--output-timetag",
        output_timetag,
        "--unified-payload-mode",
        str(unified_payload_mode).strip().lower(),
    ]
    if era5_world_root is not None:
        cmd += ["--era5-world-root", str(era5_world_root)]
    if cmip6_world_root is not None:
        cmd += ["--cmip6-world-root", str(cmip6_world_root)]
    if scenario1_world_tag is not None:
        cmd += ["--scenario1-world-tag", scenario1_world_tag]
    if scenario2_world_tag is not None:
        cmd += ["--scenario2-world-tag", scenario2_world_tag]
    if era5_world_root is not None and era5_world_tag is not None:
        cmd += ["--era5-world-tag", era5_world_tag]
    if overlay_cache_tag is not None and str(overlay_cache_tag).strip():
        cmd += ["--overlay-cache-tag", str(overlay_cache_tag).strip()]
    if overlay_missing_policy is not None and str(overlay_missing_policy).strip():
        cmd += ["--overlay-missing-policy", str(overlay_missing_policy).strip()]
    cmd += ["--include-cmip6" if include_cmip6 else "--no-include-cmip6"]
    return cmd


def _run_one(cmd: Sequence[str]) -> Tuple[int, str]:
    cp = subprocess.run(
        list(cmd),
        cwd=str(_REPO_ROOT),
        env=_thread_limited_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return int(cp.returncode), cp.stdout or ""


def _is_high_memory_region(region: str) -> bool:
    reg = str(region or "").strip()
    if not reg:
        return False
    if reg.upper() in HIGH_MEMORY_REGION_CODES:
        return True
    reg_lower = reg.lower()
    return any(token in reg_lower for token in HIGH_MEMORY_REGION_SUBSTRINGS)


def _partition_tasks_by_memory(
    tasks: Sequence[Tuple[str, str, List[str]]],
) -> Tuple[List[Tuple[str, str, List[str]]], List[Tuple[str, str, List[str]]]]:
    regular: List[Tuple[str, str, List[str]]] = []
    high_memory: List[Tuple[str, str, List[str]]] = []
    for task in tasks:
        region, _pet_method, _cmd = task
        if _is_high_memory_region(region):
            high_memory.append(task)
        else:
            regular.append(task)
    return regular, high_memory


def _partition_repairs_by_memory(
    repairs: Sequence[Tuple[str, str, List[str], Dict[str, Path], List[str]]],
) -> Tuple[
    List[Tuple[str, str, List[str], Dict[str, Path], List[str]]],
    List[Tuple[str, str, List[str], Dict[str, Path], List[str]]],
]:
    regular: List[Tuple[str, str, List[str], Dict[str, Path], List[str]]] = []
    high_memory: List[Tuple[str, str, List[str], Dict[str, Path], List[str]]] = []
    for item in repairs:
        region, _pet_method, _repair_cmd, _bundle, _full_cmd = item
        if _is_high_memory_region(region):
            high_memory.append(item)
        else:
            regular.append(item)
    return regular, high_memory


def _execute_task_batch(
    tasks: Sequence[Tuple[str, str, List[str]]],
    *,
    batch_label: str,
    max_workers: int,
    total_task_count: int,
    completed_offset: int,
) -> Tuple[List[Tuple[str, str, int, str]], Dict[str, List[str]]]:
    if not tasks:
        return [], {}
    worker_count = max(1, min(int(max_workers), len(tasks)))
    print(f"\nLaunching {len(tasks)} {batch_label} job(s) with max_workers={worker_count}...")

    failures: List[Tuple[str, str, int, str]] = []
    warnings_by_combo: Dict[str, List[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_map = {
            pool.submit(_run_one, cmd): (region, pet, cmd)
            for region, pet, cmd in tasks
        }
        done_count = 0
        for future in concurrent.futures.as_completed(future_map):
            done_count += 1
            region, pet, _cmd = future_map[future]
            try:
                code, output = future.result()
            except Exception as exc:  # pragma: no cover
                code, output = 1, f"wrapper exception: {exc}"
            combo_key = _combo_key(region, pet)
            warning_lines = [
                line.strip()
                for line in str(output or "").splitlines()
                if line.strip().startswith("[WARN]") or line.strip().startswith("⚠️")
            ]
            if warning_lines:
                warnings_by_combo[combo_key] = warning_lines
            print(f"✔ finished {completed_offset + done_count}/{total_task_count} (region={region}, pet={pet})")
            if code != 0:
                failures.append((region, pet, code, output))
    return failures, warnings_by_combo


def _copy_if_missing(src: Path, dst: Path) -> bool:
    if dst.exists() or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _existing_payload_path(path: Path) -> Optional[Path]:
    resolved = _base759._base758._resolve_payload_path(Path(path))
    return resolved if resolved.exists() else None


def _payload_companion_candidates(source_json: Path, suffix: str) -> List[Path]:
    stem = _base759._base758._payload_stem(Path(source_json))
    base = Path(source_json).parent / stem
    return [
        base.with_name(base.name + suffix),
        base.with_name(base.name + suffix + ".gz"),
    ]


def _repair_script_path() -> Path:
    return Path(__file__).with_name("759Repair_safe_outputs_from_json.py")


def _legacy_output_region_token(region: str) -> str:
    """
    Older 759 outputs preserved punctuation like '.' and '&' in the filename token.
    The directory name stayed safe, but the file suffix token did not always match the
    current `_safe_region_tag(...).upper()` convention.
    """
    text = str(region or "").strip().upper().replace(" ", "_").replace("/", "_")
    text = re.sub(r"_+", "_", text).strip("_")
    return text or _base758._output_region_token(region)


def _legacy_variant_path(path: Path, *, region: str) -> Path:
    current_token = _base758._output_region_token(region)
    legacy_token = _legacy_output_region_token(region)
    if legacy_token == current_token:
        return Path(path)
    return Path(path).with_name(Path(path).name.replace(current_token, legacy_token))


def _required_output_variants(
    *,
    output_base_root: Path,
    version_tag: str,
    scenario_pair_tag: str,
    timetag: str,
    region: str,
    pet_method: str,
) -> Dict[str, List[Path]]:
    paths = _base759._expected_output_paths(
        output_base_root=output_base_root,
        version_tag=version_tag,
        scenario_pair_tag=scenario_pair_tag,
        timetag=timetag,
        region=region,
        pet_method=pet_method,
    )
    out: Dict[str, List[Path]] = {}
    for key in (
        "json",
        "panel_a_json",
        "panel_i_json",
        "unified_png",
        "unified_pdf",
        "maps_png",
        "maps_pdf",
        "times_png",
        "times_pdf",
        "hist_png",
        "hist_pdf",
    ):
        canonical = Path(paths[key])
        if key == "json":
            variants = list(_base758._payload_json_variants(canonical))
            legacy = _legacy_variant_path(canonical, region=region)
            if legacy != canonical:
                variants.extend(_base758._payload_json_variants(legacy))
        else:
            variants = [canonical]
            legacy = _legacy_variant_path(canonical, region=region)
            if legacy != canonical:
                variants.append(legacy)
        out[key] = variants
    return out


def _payload_json_candidates(
    *,
    out_dir: Path,
    pet_method: str,
    timetag: str,
) -> List[Path]:
    pet_tag = _base755._normalize_pet_method(pet_method)
    pattern = f"SPEI_UNIFIED_{pet_tag}_ERA5_GCMAGICC_{timetag}_*.json*"
    candidates: List[Path] = []
    for path in out_dir.rglob(pattern):
        payload_stem = _base758._payload_stem(path)
        if payload_stem.endswith("_panelA_map") or payload_stem.endswith("_panelI_timeseries"):
            continue
        if path.is_file():
            candidates.append(path)
    return sorted(
        {path.resolve(strict=False) for path in candidates},
        key=lambda path: (len(path.parts), path.name),
    )


def _candidate_with_suffix(source_json: Path, suffix: str) -> Optional[Path]:
    for candidate in _payload_companion_candidates(source_json, suffix):
        if candidate.exists():
            return candidate
    return None


def _find_existing_payload_bundle(
    *,
    output_base_root: Path,
    version_tag: str,
    scenario_pair_tag: str,
    timetag: str,
    region: str,
    pet_method: str,
) -> Optional[Dict[str, Path]]:
    paths = _base759._expected_output_paths(
        output_base_root=output_base_root,
        version_tag=version_tag,
        scenario_pair_tag=scenario_pair_tag,
        timetag=timetag,
        region=region,
        pet_method=pet_method,
    )
    out_dir = Path(paths["out_dir"])
    candidates = _payload_json_candidates(out_dir=out_dir, pet_method=pet_method, timetag=timetag)
    safe_json = Path(paths["json"])
    source_json: Optional[Path] = None
    if candidates:
        candidates = sorted(
            candidates,
            key=lambda path: (
                0 if path.name != safe_json.name else 1,
                len(path.parts),
                path.name,
            ),
        )
        source_json = candidates[0]
    else:
        source_json = _existing_payload_path(safe_json)
    if source_json is None:
        return None
    panel_a = _candidate_with_suffix(source_json, "_panelA_map.json")
    panel_i = _candidate_with_suffix(source_json, "_panelI_timeseries.json")
    return {
        "source_json": source_json,
        "source_panel_a_json": panel_a or Path(paths["panel_a_json"]),
        "source_panel_i_json": panel_i or Path(paths["panel_i_json"]),
        "safe_json": Path(paths["json"]),
        "safe_panel_a_json": Path(paths["panel_a_json"]),
        "safe_panel_i_json": Path(paths["panel_i_json"]),
    }


def _build_repair_cmd(
    *,
    source_json: Path,
    safe_json: Path,
    source_panel_a_json: Path,
    safe_panel_a_json: Path,
    source_panel_i_json: Path,
    safe_panel_i_json: Path,
    output_base_root: Path,
    output_timetag: str,
) -> List[str]:
    return [
        sys.executable,
        str(_repair_script_path()),
        "--source-json",
        str(source_json),
        "--safe-json",
        str(safe_json),
        "--source-panel-a-json",
        str(source_panel_a_json),
        "--safe-panel-a-json",
        str(safe_panel_a_json),
        "--source-panel-i-json",
        str(source_panel_i_json),
        "--safe-panel-i-json",
        str(safe_panel_i_json),
        "--output-base-root",
        str(output_base_root),
        "--output-timetag",
        str(output_timetag),
    ]


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _manifest_path(*, output_base_root: Path, version_tag: str, scenario_pair_tag: str, timetag: str) -> Path:
    return (
        Path(output_base_root).expanduser().resolve(strict=False)
        / str(version_tag).strip()
        / str(scenario_pair_tag).strip()
        / str(timetag).strip()
        / _base759.PUBLISH_MANIFEST_BASENAME
    )


def _combo_key(region: str, pet_method: str) -> str:
    return f"{_base758_wrapper._safe_region_tag(region)}/{_base755._normalize_pet_method(pet_method)}"


def _combo_completion(
    *,
    output_base_root: Path,
    version_tag: str,
    scenario_pair_tag: str,
    timetag: str,
    region: str,
    pet_method: str,
) -> Dict[str, object]:
    paths = _base759._expected_output_paths(
        output_base_root=output_base_root,
        version_tag=version_tag,
        scenario_pair_tag=scenario_pair_tag,
        timetag=timetag,
        region=region,
        pet_method=pet_method,
    )
    out_dir = paths["out_dir"]
    stats_paths = sorted(Path(out_dir).glob("SPEI_STATS_*.json"))
    required_variants = _required_output_variants(
        output_base_root=output_base_root,
        version_tag=version_tag,
        scenario_pair_tag=scenario_pair_tag,
        timetag=timetag,
        region=region,
        pet_method=pet_method,
    )
    canonical_missing: List[str] = []
    missing: List[str] = []
    all_variants_present = True
    complete_canonical = True
    for key, variants in required_variants.items():
        canonical = variants[0]
        if not canonical.exists():
            canonical_missing.append(str(canonical))
            complete_canonical = False
        if not any(path.exists() for path in variants):
            missing.append(str(canonical))
            all_variants_present = False
    if not stats_paths:
        stats_glob = str(Path(out_dir) / "SPEI_STATS_*.json")
        missing.append(stats_glob)
        canonical_missing.append(stats_glob)
        all_variants_present = False
        complete_canonical = False
    complete = all_variants_present and bool(stats_paths)
    needs_repair = complete and not complete_canonical
    return {
        "complete": complete,
        "complete_canonical": complete_canonical,
        "needs_repair": needs_repair,
        "missing": missing,
        "canonical_missing": canonical_missing,
        "out_dir": str(out_dir),
        "stats_paths": [str(path) for path in stats_paths],
    }


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel wrapper for 759_GapFiller_SPEI_fromWorld.py")
    p.add_argument("--profile", default="curpol_ndclow", choices=sorted(_base758_wrapper.PROFILE_MAP.keys()))
    p.add_argument("--scenario1-world-root", type=Path, default=None, help="Scenario 1 755 world root (run root or SPEIx root).")
    p.add_argument("--scenario2-world-root", type=Path, default=None, help="Scenario 2 755 world root.")
    p.add_argument("--era5-world-root", type=Path, default=_optional_env_path(ENV_759_ERA5_WORLD_ROOT), help="Optional stable ERA5 overlay world-root override.")
    p.add_argument("--cmip6-world-root", type=Path, default=None, help="Optional stable CMIP6 overlay world-root override.")
    p.add_argument("--era5-file", type=Path, default=_optional_env_path(ENV_759_ERA5_FILE) or _base759.DEFAULT_ERA5_FILE, help="ERA5 file used to locate the canonical 758 SPEIx source.")
    p.add_argument("--output-base-root", type=Path, default=_optional_env_path(ENV_759_OUTPUT_BASE_ROOT) or _base759.DEFAULT_OUTPUT_BASE_ROOT, help="Base directory for published outputs.")
    p.add_argument("--world-tag", default=None, help="Shared 755 world tag.")
    p.add_argument("--scenario1-world-tag", default=None, help="Optional Scenario 1 tag override.")
    p.add_argument("--scenario2-world-tag", default=None, help="Optional Scenario 2 tag override.")
    p.add_argument("--era5-world-tag", default=None, help="Optional ERA5 tag override.")
    p.add_argument("--overlay-cache-tag", default=_base759.DEFAULT_OVERLAY_CACHE_TAG, help=f"Stable reusable overlay cache tag (default: {_base759.DEFAULT_OVERLAY_CACHE_TAG}).")
    p.add_argument("--overlay-missing-policy", default=_base759.DEFAULT_OVERLAY_MISSING_POLICY, choices=[_base759.DEFAULT_OVERLAY_MISSING_POLICY], help=f"Behavior when reusable overlay caches are missing (default: {_base759.DEFAULT_OVERLAY_MISSING_POLICY}).")
    p.add_argument("--regions", nargs="*", default=["auto"], help="Region list, or 'auto' for the canonical expected list.")
    p.add_argument("--pet-methods", nargs="+", default=list(_base758_wrapper.CORE_PET_METHODS), help="PET methods to sweep.")
    p.add_argument("--version-tag", default=None, help="Optional version tag override.")
    p.add_argument("--output-timetag", default=None, help="Shared timetag for all launched jobs.")
    p.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Parallel subprocess cap (default: {DEFAULT_MAX_WORKERS}).",
    )
    p.add_argument("--include-cmip6", action=argparse.BooleanOptionalAction, default=None, help="Override the profile CMIP6 setting.")
    p.add_argument("--skip-if-complete", action=argparse.BooleanOptionalAction, default=True, help="Skip region/PET jobs whose final outputs already exist for the selected timetag.")
    p.add_argument("--repair-from-existing-json", action=argparse.BooleanOptionalAction, default=True, help="Repair incomplete combos from already-written unified JSON bundles before launching a full 759 recompute.")
    p.add_argument(
        "--unified-payload-mode",
        default=DEFAULT_UNIFIED_PAYLOAD_MODE,
        choices=list(_base759.UNIFIED_PAYLOAD_MODE_CHOICES),
        help="How 759 should retain the large main SPEI_UNIFIED payload after rendering (default: gzip).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return p.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    profile = _base758_wrapper.PROFILE_MAP[args.profile]
    script_path = Path(__file__).with_name("759_GapFiller_SPEI_fromWorld.py")
    version_tag = (
        str(args.version_tag).strip()
        if args.version_tag and str(args.version_tag).strip()
        else _base758_wrapper._DEFAULT_VERSION_TAG
    )

    default_s1_root, default_s2_root = _profile_world_roots(args.profile, version_tag=version_tag)
    scenario1_world_root = (
        Path(args.scenario1_world_root).expanduser().resolve(strict=False)
        if args.scenario1_world_root is not None
        else default_s1_root
    )
    scenario2_world_root = (
        Path(args.scenario2_world_root).expanduser().resolve(strict=False)
        if args.scenario2_world_root is not None
        else default_s2_root
    )
    era5_file = Path(args.era5_file).expanduser().resolve(strict=False)
    output_base_root = Path(args.output_base_root).expanduser().resolve(strict=False)
    era5_world_root = (
        Path(args.era5_world_root).expanduser().resolve(strict=False)
        if args.era5_world_root is not None
        else None
    )
    cmip6_world_root = (
        Path(args.cmip6_world_root).expanduser().resolve(strict=False)
        if args.cmip6_world_root is not None
        else None
    )
    if scenario1_world_root is None or scenario2_world_root is None:
        raise SystemExit(
            "Scenario world roots are not fully configured. Pass --scenario1-world-root/--scenario2-world-root "
            "or set the matching GCMAGICC_759_* environment variables."
        )

    include_cmip6 = profile.include_cmip6 if args.include_cmip6 is None else bool(args.include_cmip6)
    scenario1_tag = profile.scenario1
    scenario2_tag = profile.scenario2
    scenario1_label = profile.scenario1_label
    scenario2_label = profile.scenario2_label
    scenario_pair_id = _base758_wrapper._scenario_pair_tag(scenario1_tag, scenario2_tag)

    regions = _base758_wrapper._expected_regions() if (len(args.regions) == 1 and args.regions[0].lower() == "auto") else list(args.regions)
    if not regions:
        raise SystemExit("No regions selected.")

    pet_methods = [_base755._normalize_pet_method(pet) for pet in args.pet_methods]
    timetag = str(args.output_timetag).strip() if args.output_timetag else datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario1_world_tag = args.scenario1_world_tag if args.scenario1_world_tag is not None else args.world_tag
    scenario2_world_tag = args.scenario2_world_tag if args.scenario2_world_tag is not None else args.world_tag
    era5_world_tag = args.era5_world_tag if args.era5_world_tag is not None else args.overlay_cache_tag

    tasks: List[Tuple[str, str, List[str]]] = []
    for region in regions:
        for pet_method in pet_methods:
            tasks.append(
                (
                    region,
                    pet_method,
                    _build_cmd(
                        script_path=script_path,
                        region=region,
                        pet_method=pet_method,
                        scenario1_world_root=scenario1_world_root,
                        scenario2_world_root=scenario2_world_root,
                        era5_file=era5_file,
                        output_base_root=output_base_root,
                        era5_world_root=era5_world_root,
                        cmip6_world_root=cmip6_world_root,
                        scenario1_world_tag=scenario1_world_tag,
                        scenario2_world_tag=scenario2_world_tag,
                        era5_world_tag=era5_world_tag,
                        overlay_cache_tag=args.overlay_cache_tag,
                        overlay_missing_policy=args.overlay_missing_policy,
                        scenario1_tag=scenario1_tag,
                        scenario2_tag=scenario2_tag,
                        scenario1_label=scenario1_label,
                        scenario2_label=scenario2_label,
                        version_tag=version_tag,
                        output_timetag=timetag,
                        include_cmip6=include_cmip6,
                        unified_payload_mode=str(args.unified_payload_mode).strip().lower(),
                    ),
                )
            )

    print(f"Base script:      {script_path}")
    print(f"Profile:          {args.profile}")
    print(f"Scenario pair id: {scenario_pair_id}")
    print(f"Scenario1 root:   {scenario1_world_root}")
    print(f"Scenario2 root:   {scenario2_world_root}")
    print(f"ERA5 file:        {era5_file}")
    print(f"ERA5 world root:  {era5_world_root if era5_world_root is not None else '(using 758 ERA5 source logic)'}")
    print(f"CMIP6 world root: {cmip6_world_root if cmip6_world_root is not None else '(using 759 stable default)'}")
    print(f"Output base root: {output_base_root}")
    print(f"Version tag:      {version_tag}")
    print(f"Timetag:          {timetag}")
    print(f"Regions ({len(regions)}): {', '.join(regions)}")
    print(f"PET methods:      {', '.join(pet_methods)}")
    print(f"CMIP6 overlays:   {include_cmip6}")
    print(f"Overlay tag:      {args.overlay_cache_tag}")
    print(f"Missing policy:   {args.overlay_missing_policy}")
    print(f"Skip complete:    {bool(args.skip_if_complete)}")
    print(f"Payload mode:     {str(args.unified_payload_mode).strip().lower()}")

    expected_combo_keys = [_combo_key(region, pet) for region, pet, _cmd in tasks]
    skipped_completed: List[Dict[str, object]] = []
    repaired_from_existing: List[Dict[str, object]] = []
    repair_candidates: List[Tuple[str, str, List[str], Dict[str, Path], List[str]]] = []
    tasks_to_run: List[Tuple[str, str, List[str]]] = []
    for region, pet_method, cmd in tasks:
        completion = _combo_completion(
            output_base_root=output_base_root,
            version_tag=version_tag,
            scenario_pair_tag=scenario_pair_id,
            timetag=timetag,
            region=region,
            pet_method=pet_method,
        )
        if bool(completion["complete"]):
            if bool(completion.get("needs_repair")) and args.repair_from_existing_json:
                bundle = _find_existing_payload_bundle(
                    output_base_root=output_base_root,
                    version_tag=version_tag,
                    scenario_pair_tag=scenario_pair_id,
                    timetag=timetag,
                    region=region,
                    pet_method=pet_method,
                )
                if bundle is not None:
                    repair_candidates.append(
                        (
                            region,
                            pet_method,
                            _build_repair_cmd(
                                source_json=bundle["source_json"],
                                safe_json=bundle["safe_json"],
                                source_panel_a_json=bundle["source_panel_a_json"],
                                safe_panel_a_json=bundle["safe_panel_a_json"],
                                source_panel_i_json=bundle["source_panel_i_json"],
                                safe_panel_i_json=bundle["safe_panel_i_json"],
                                output_base_root=output_base_root,
                                output_timetag=timetag,
                            ),
                            bundle,
                            cmd,
                        )
                    )
                    continue
            if args.skip_if_complete:
                skipped_completed.append(
                    {
                        "region": region,
                        "pet_method": pet_method,
                        "combo_key": _combo_key(region, pet_method),
                        "out_dir": completion["out_dir"],
                        "needs_repair": bool(completion.get("needs_repair")),
                    }
                )
                continue
            tasks_to_run.append((region, pet_method, cmd))
            continue
        if args.repair_from_existing_json:
            bundle = _find_existing_payload_bundle(
                output_base_root=output_base_root,
                version_tag=version_tag,
                scenario_pair_tag=scenario_pair_id,
                timetag=timetag,
                region=region,
                pet_method=pet_method,
            )
            if bundle is not None:
                repair_candidates.append(
                    (
                        region,
                        pet_method,
                        _build_repair_cmd(
                            source_json=bundle["source_json"],
                            safe_json=bundle["safe_json"],
                            source_panel_a_json=bundle["source_panel_a_json"],
                            safe_panel_a_json=bundle["safe_panel_a_json"],
                            source_panel_i_json=bundle["source_panel_i_json"],
                            safe_panel_i_json=bundle["safe_panel_i_json"],
                            output_base_root=output_base_root,
                            output_timetag=timetag,
                        ),
                        bundle,
                        cmd,
                    )
                )
                continue
        tasks_to_run.append((region, pet_method, cmd))

    manifest_path = _manifest_path(
        output_base_root=output_base_root,
        version_tag=version_tag,
        scenario_pair_tag=scenario_pair_id,
        timetag=timetag,
    )

    if args.dry_run:
        if repair_candidates:
            print("\nRepair-from-existing-json commands:")
            for region, pet_method, repair_cmd, bundle, _full_cmd in repair_candidates:
                print(f"  # {_combo_key(region, pet_method)} from {bundle['source_json']}")
                print("  " + " ".join(repair_cmd))
        print("\nDry-run commands:")
        for _region, _pet, cmd in tasks_to_run:
            print("  " + " ".join(cmd))
        if skipped_completed:
            print("\nAlready complete:")
            for row in skipped_completed:
                print(f"  {_combo_key(str(row['region']), str(row['pet_method']))} -> {row['out_dir']}")
        return

    base_manifest: Dict[str, object] = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "scenario_pair_tag": scenario_pair_id,
        "version_tag": version_tag,
        "timetag": timetag,
        "scenario1_world_root": str(scenario1_world_root),
        "scenario2_world_root": str(scenario2_world_root),
        "era5_file": str(era5_file),
        "era5_world_root": str(era5_world_root) if era5_world_root is not None else None,
        "output_base_root": str(output_base_root),
        "world_tag": args.world_tag,
        "scenario1_world_tag": scenario1_world_tag,
        "scenario2_world_tag": scenario2_world_tag,
        "include_cmip6": include_cmip6,
        "expected_combos": expected_combo_keys,
        "skipped_preexisting": skipped_completed,
        "repaired_from_existing": repaired_from_existing,
    }

    if repair_candidates:
        repair_failures: List[Tuple[str, str, int, str]] = []
        regular_repairs, high_memory_repairs = _partition_repairs_by_memory(repair_candidates)
        completed_repair_offset = 0
        for batch_label, batch_repairs, batch_max_workers in (
            ("regular", regular_repairs, int(args.max_workers)),
            ("high-memory region", high_memory_repairs, HIGH_MEMORY_BATCH_MAX_WORKERS),
        ):
            if not batch_repairs:
                continue
            worker_count = max(1, min(int(batch_max_workers), len(batch_repairs)))
            print(
                f"\nRepairing {len(batch_repairs)} {batch_label} combo(s) from existing JSON bundles "
                f"with max_workers={worker_count}..."
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
                future_map = {
                    pool.submit(_run_one, repair_cmd): (region, pet_method, repair_cmd, bundle, full_cmd)
                    for region, pet_method, repair_cmd, bundle, full_cmd in batch_repairs
                }
                done_count = 0
                for future in concurrent.futures.as_completed(future_map):
                    done_count += 1
                    region, pet_method, _repair_cmd, bundle, full_cmd = future_map[future]
                    try:
                        code, output = future.result()
                    except Exception as exc:  # pragma: no cover
                        code, output = 1, f"repair wrapper exception: {exc}"
                    completion = _combo_completion(
                        output_base_root=output_base_root,
                        version_tag=version_tag,
                        scenario_pair_tag=scenario_pair_id,
                        timetag=timetag,
                        region=region,
                        pet_method=pet_method,
                    )
                    if bool(completion["complete"]):
                        repaired_from_existing.append(
                            {
                                "region": region,
                                "pet_method": pet_method,
                                "combo_key": _combo_key(region, pet_method),
                                "out_dir": completion["out_dir"],
                                "source_json": str(bundle["source_json"]),
                            }
                        )
                        print(
                            f"↺ repaired {completed_repair_offset + done_count}/{len(repair_candidates)} "
                            f"(region={region}, pet={pet_method})"
                        )
                        continue
                    tasks_to_run.append((region, pet_method, full_cmd))
                    if code != 0:
                        repair_failures.append((region, pet_method, code, output))
                    print(
                        f"↺ repair incomplete {completed_repair_offset + done_count}/{len(repair_candidates)} "
                        f"(region={region}, pet={pet_method}) -> queued full 759 run"
                    )
            completed_repair_offset += len(batch_repairs)
        if repair_failures:
            print(f"Repair stage had {len(repair_failures)} non-zero subprocess exit(s); unresolved combos were queued for full 759 recompute.")

    if not tasks_to_run:
        completed_rows = []
        for region in regions:
            for pet_method in pet_methods:
                completion = _combo_completion(
                    output_base_root=output_base_root,
                    version_tag=version_tag,
                    scenario_pair_tag=scenario_pair_id,
                    timetag=timetag,
                    region=region,
                    pet_method=pet_method,
                )
                if bool(completion["complete"]):
                    completed_rows.append(
                        {
                            "region": region,
                            "pet_method": pet_method,
                            "combo_key": _combo_key(region, pet_method),
                            "out_dir": completion["out_dir"],
                        }
                    )
        manifest_payload = dict(base_manifest)
        manifest_payload["repaired_from_existing"] = repaired_from_existing
        manifest_payload["completed_combos"] = completed_rows
        manifest_payload["failed_combos"] = []
        manifest_payload["summary"] = {
            "expected": len(expected_combo_keys),
            "completed": len(completed_rows),
            "failed": 0,
            "skipped_preexisting": len(skipped_completed),
            "repaired_from_existing": len(repaired_from_existing),
            "launched": 0,
        }
        _write_json_atomic(manifest_path, manifest_payload)
        print("\nAll requested combos are already complete.")
        print(f"Manifest: {manifest_path}")
        return

    regular_tasks, high_memory_tasks = _partition_tasks_by_memory(tasks_to_run)
    failures: List[Tuple[str, str, int, str]] = []
    warnings_by_combo: Dict[str, List[str]] = {}
    completed_offset = 0
    regular_failures, regular_warnings = _execute_task_batch(
        regular_tasks,
        batch_label="regular",
        max_workers=int(args.max_workers),
        total_task_count=len(tasks_to_run),
        completed_offset=completed_offset,
    )
    failures.extend(regular_failures)
    warnings_by_combo.update(regular_warnings)
    completed_offset += len(regular_tasks)
    heavy_failures, heavy_warnings = _execute_task_batch(
        high_memory_tasks,
        batch_label="high-memory region",
        max_workers=HIGH_MEMORY_BATCH_MAX_WORKERS,
        total_task_count=len(tasks_to_run),
        completed_offset=completed_offset,
    )
    failures.extend(heavy_failures)
    warnings_by_combo.update(heavy_warnings)

    completed_rows = []
    failed_rows = []
    for region in regions:
        for pet_method in pet_methods:
            combo_key = _combo_key(region, pet_method)
            completion = _combo_completion(
                output_base_root=output_base_root,
                version_tag=version_tag,
                scenario_pair_tag=scenario_pair_id,
                timetag=timetag,
                region=region,
                pet_method=pet_method,
            )
            row = {
                "region": region,
                "pet_method": pet_method,
                "combo_key": combo_key,
                "out_dir": completion["out_dir"],
                "warnings": warnings_by_combo.get(combo_key, []),
            }
            if bool(completion["complete"]):
                completed_rows.append(row)
            else:
                row["missing"] = completion["missing"]
                failed_rows.append(row)

    manifest_payload = dict(base_manifest)
    manifest_payload["repaired_from_existing"] = repaired_from_existing
    manifest_payload["completed_combos"] = completed_rows
    manifest_payload["failed_combos"] = failed_rows
    manifest_payload["summary"] = {
        "expected": len(expected_combo_keys),
        "completed": len(completed_rows),
        "failed": len(failed_rows),
        "skipped_preexisting": len(skipped_completed),
        "repaired_from_existing": len(repaired_from_existing),
        "launched": len(tasks_to_run),
    }
    _write_json_atomic(manifest_path, manifest_payload)
    print(f"Manifest:         {manifest_path}")

    if failures:
        for region, pet, code, output in failures:
            print(f"\n[FAIL {code}] region={region} pet={pet}")
            if output.strip():
                print(output.strip())
        raise SystemExit(f"{len(failures)} job(s) failed.")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
