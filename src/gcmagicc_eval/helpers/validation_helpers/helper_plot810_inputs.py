"""Input resolution and staging helpers for 810 SSP projection plotting."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
import sys
from typing import Iterable, Sequence

from scr.validation_helpers.helper_gapfiller_inputs import (
    build_gapfiller_era5_stage_command,
    build_gapfiller_cmip6_stage_command,
    matching_gapfiller_cmip6_archive_files,
    normalize_version_tag,
    resolve_gapfiller_era5_source,
)
from scr.validation_helpers.helper_path_utils import (
    CANONICAL_KIND_ORIGINAL,
    get_cmip6_localstaging_root,
    get_cmip6_vetted_candidates,
    get_cmip6replicas_root,
    get_cmip6replicas_localstaging_root,
    get_era5spliced_localstaging_root,
    get_era5spliced_root,
    get_gcmagicc_archive_candidates,
    get_object_bucket,
    get_object_remote,
    get_project_root,
    normalize_n_ensemble_label,
    parse_era5spliced_dataset_path,
    path_uses_rclone_mount,
    resolve_canonical_dataset_root,
)


@dataclass(frozen=True)
class Plot810ScenarioSource:
    scenario: str
    path: Path
    source_kind: str
    remote_prefix: str | None = None
    stage_root: Path | None = None
    mount_root: Path | None = None
    explicit: bool = False
    run_instance: str | None = None


@dataclass(frozen=True)
class Plot810Source:
    path: Path
    source_kind: str
    label: str
    remote_prefix: str | None = None
    stage_root: Path | None = None
    mount_root: Path | None = None
    explicit: bool = False
    available_scenarios: tuple[str, ...] = ()
    missing_scenarios: tuple[str, ...] = ()
    matched_variables: tuple[str, ...] = ()
    missing_variables: tuple[str, ...] = ()
    scenario_sources: dict[str, Plot810ScenarioSource] = field(default_factory=dict)
    run_instance: str | None = None


_SAFE_TOKEN_RE = re.compile(r"[^0-9A-Za-z._-]+")


def _normalize_items(items: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        token = str(item or "").strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(token)
    return tuple(out)


def _safe_token(value: str) -> str:
    token = _SAFE_TOKEN_RE.sub("-", str(value or "").strip()).strip("-._")
    return token or "item"


def _compound_token(prefix: str, values: Sequence[str]) -> str:
    normalized = _normalize_items(values)
    joined = "__".join(_safe_token(value) for value in normalized) or "all"
    digest = hashlib.sha1("|".join(normalized).encode("utf-8")).hexdigest()[:10]
    if len(joined) > 80:
        joined = joined[:60].strip("-._")
    return f"{prefix}__{joined}__{digest}"


def _filename_mentions_var(path: Path, var: str) -> bool:
    stem = path.stem.lower()
    token = str(var or "").strip().lower()
    tokens = stem.replace("-", "_").split("_")
    return token in tokens or token in stem


def _parse_cmip6_meta(path: Path) -> tuple[str | None, str | None, str | None]:
    name = path.name
    if not name.startswith("DAT_"):
        return None, None, None
    parts = name.replace("DAT_", "").split("_")
    if len(parts) < 3:
        return None, None, None
    return parts[0], parts[1], parts[2]


def _parse_archive_filename(path: Path) -> tuple[str | None, str | None, str | None]:
    parts = Path(path).stem.split("_")
    if len(parts) < 4:
        return None, None, None
    return parts[1], parts[2], parts[3]


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


def _iter_nc_files(root: Path) -> list[Path]:
    resolved = Path(root).expanduser().resolve(strict=False)
    if not resolved.is_dir():
        return []
    try:
        return sorted(p for p in resolved.glob("*.nc") if p.is_file())
    except OSError:
        return []


def _latest_run_root(root: Path) -> Path | None:
    resolved = Path(root).expanduser().resolve(strict=False)
    if _iter_nc_files(resolved):
        return resolved
    run_dirs = sorted(
        p for p in _safe_iterdir(resolved) if p.is_dir() and p.name.startswith("run_")
    )
    for candidate in reversed(run_dirs):
        if _iter_nc_files(candidate):
            return candidate.resolve(strict=False)
    return None


def _normalize_gapfiller_source_kind(value: str) -> str:
    token = str(value or "").strip().lower()
    if token == "canonical_mount":
        return "mounted"
    if token == "legacy":
        return "legacy_archive"
    return token or "missing"


def _cmip6_inventory(
    root: Path,
    *,
    scenarios: Sequence[str],
    variables: Sequence[str],
    include_historical: bool = True,
) -> dict[str, object]:
    resolved_root = Path(root).expanduser().resolve(strict=False)
    if not resolved_root.is_dir():
        wanted_scenarios = set(item.lower() for item in _normalize_items(scenarios))
        wanted_variables = set(item.lower() for item in _normalize_items(variables))
        return {
            "files": [],
            "available_scenarios": (),
            "missing_scenarios": tuple(sorted(wanted_scenarios)),
            "matched_variables": (),
            "missing_variables": tuple(sorted(wanted_variables)),
        }

    wanted_scenarios = {item.lower() for item in _normalize_items(scenarios)}
    if include_historical:
        wanted_scenarios.add("historical")
    wanted_variables = {item.lower() for item in _normalize_items(variables)}

    try:
        candidates = sorted(p for p in resolved_root.glob("DAT_*.nc") if p.is_file())
    except OSError:
        candidates = []

    selected: list[Path] = []
    matched_scenarios: set[str] = set()
    matched_variables: set[str] = set()
    for path in candidates:
        _, scenario, _ = _parse_cmip6_meta(path)
        scenario_key = str(scenario or "").strip().lower()
        if not scenario_key or (wanted_scenarios and scenario_key not in wanted_scenarios):
            continue
        matched_here = {var for var in wanted_variables if _filename_mentions_var(path, var)}
        if wanted_variables and not matched_here:
            continue
        matched_scenarios.add(scenario_key)
        matched_variables.update(matched_here)
        selected.append(path)

    return {
        "files": selected,
        "available_scenarios": tuple(sorted(matched_scenarios)),
        "missing_scenarios": tuple(sorted(wanted_scenarios - matched_scenarios)),
        "matched_variables": tuple(sorted(matched_variables)),
        "missing_variables": tuple(sorted(wanted_variables - matched_variables)),
    }


def matching_plot810_cmip6_files(
    root: Path,
    *,
    scenarios: Sequence[str],
    variables: Sequence[str],
    include_historical: bool = True,
    require_complete: bool = False,
) -> list[Path]:
    inventory = _cmip6_inventory(
        root,
        scenarios=scenarios,
        variables=variables,
        include_historical=include_historical,
    )
    if require_complete and (
        inventory["missing_scenarios"] or inventory["missing_variables"]
    ):
        return []
    return list(inventory["files"])


def _classify_root_with_files(path: Path, *, mounted_label: str, local_label: str) -> str:
    resolved = Path(path).expanduser().resolve(strict=False)
    if path_uses_rclone_mount(resolved, resolve_path=False):
        return mounted_label
    return local_label


def build_plot810_gcmagicc_stage_root(
    *,
    version: str | None,
    scenarios: Sequence[str],
    stage_base: Path | None = None,
) -> Path:
    version_tag = normalize_version_tag(version)
    base = Path(stage_base or get_cmip6replicas_localstaging_root()).expanduser().resolve(strict=False)
    token = _compound_token("plot810", tuple(sorted(_normalize_items(scenarios), key=str.lower)))
    return (base / version_tag / token).resolve(strict=False)


def build_plot810_cmip6_stage_root(
    *,
    scenarios: Sequence[str],
    variables: Sequence[str],
    stage_base: Path | None = None,
) -> Path:
    base = Path(stage_base or get_cmip6_localstaging_root()).expanduser().resolve(strict=False)
    scenario_token = _compound_token("scenarios", tuple(sorted(_normalize_items(scenarios), key=str.lower)))
    variable_token = _compound_token("vars", tuple(sorted(_normalize_items(variables), key=str.lower)))
    return (base / scenario_token / variable_token).resolve(strict=False)


def get_plot810_raw_cmip6_remote_prefix() -> str:
    return f"{get_object_remote()}:{get_object_bucket()}/nc/reference/out_ETHFOG_10June2025_vetted"


def build_plot810_cmip6_stage_command(
    *,
    scenarios: Sequence[str],
    variables: Sequence[str],
    stage_base: Path | None = None,
    stage_root: Path | None = None,
    max_per_model: int | None = None,
    dry_run: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        str(get_project_root() / "scripts" / "stage_plot810_cmip6_subset.py"),
    ]
    for scenario in _normalize_items(scenarios):
        cmd.extend(["--scenario", scenario])
    for variable in _normalize_items(variables):
        cmd.extend(["--variable", variable])
    if stage_base is not None:
        cmd.extend(["--stage-base", str(Path(stage_base).expanduser().resolve(strict=False))])
    if stage_root is not None:
        cmd.extend(["--stage-root", str(Path(stage_root).expanduser().resolve(strict=False))])
    if max_per_model is not None:
        cmd.extend(["--max-per-model", str(int(max_per_model))])
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def resolve_plot810_gcmagicc_source(
    *,
    version: str | None,
    scenarios: Sequence[str],
    explicit_dir: Path | None = None,
    stage_base: Path | None = None,
    source_ids: Sequence[str] | None = None,
    members: Sequence[str] | None = None,
    prefer_staged: bool = True,
) -> Plot810Source:
    version_tag = normalize_version_tag(version)
    wanted_scenarios = tuple(sorted({*{item.lower() for item in _normalize_items(scenarios)}, "historical"}))
    remote_prefix = f"s3://{get_object_bucket()}/nc/{'eth' if version_tag.startswith('v101') else 'gus'}/{version_tag}/gcmagicc"
    mount_root = (get_cmip6replicas_root().expanduser().resolve(strict=False) / version_tag).resolve(strict=False)
    stage_root = build_plot810_gcmagicc_stage_root(version=version_tag, scenarios=scenarios, stage_base=stage_base)

    if explicit_dir is not None:
        resolved = Path(explicit_dir).expanduser().resolve(strict=False)
        matches = matching_gapfiller_cmip6_archive_files(
            resolved,
            scenarios=wanted_scenarios,
            source_ids=source_ids,
            members=members,
        )
        source_kind = _classify_root_with_files(resolved, mounted_label="mounted", local_label="local") if matches else "missing"
        matched_scenarios = sorted(
            {
                str((_parse_archive_filename(path)[1] or "")).strip().lower()
                for path in matches
                if _parse_archive_filename(path)[1]
            }
        )
        return Plot810Source(
            path=resolved,
            source_kind=source_kind,
            label="gcmagicc_cmip6",
            remote_prefix=remote_prefix,
            stage_root=stage_root,
            mount_root=mount_root,
            explicit=True,
            available_scenarios=tuple(matched_scenarios),
            missing_scenarios=tuple(sorted(set(wanted_scenarios) - set(matched_scenarios))),
        )

    if prefer_staged:
        stage_matches = matching_gapfiller_cmip6_archive_files(
            stage_root,
            scenarios=wanted_scenarios,
            source_ids=source_ids,
            members=members,
        )
        if stage_matches:
            matched_scenarios = sorted(
                {
                    str((_parse_archive_filename(path)[1] or "")).strip().lower()
                    for path in stage_matches
                    if _parse_archive_filename(path)[1]
                }
            )
            return Plot810Source(
                path=stage_root,
                source_kind="staged",
                label="gcmagicc_cmip6",
                remote_prefix=remote_prefix,
                stage_root=stage_root,
                mount_root=mount_root,
                available_scenarios=tuple(matched_scenarios),
                missing_scenarios=tuple(sorted(set(wanted_scenarios) - set(matched_scenarios))),
            )

    mount_matches = matching_gapfiller_cmip6_archive_files(
        mount_root,
        scenarios=wanted_scenarios,
        source_ids=source_ids,
        members=members,
    )
    if mount_matches:
        matched_scenarios = sorted(
            {
                str((_parse_archive_filename(path)[1] or "")).strip().lower()
                for path in mount_matches
                if _parse_archive_filename(path)[1]
            }
        )
        return Plot810Source(
            path=mount_root,
            source_kind=_classify_root_with_files(mount_root, mounted_label="mounted", local_label="local"),
            label="gcmagicc_cmip6",
            remote_prefix=remote_prefix,
            stage_root=stage_root,
            mount_root=mount_root,
            available_scenarios=tuple(matched_scenarios),
            missing_scenarios=tuple(sorted(set(wanted_scenarios) - set(matched_scenarios))),
        )

    for candidate in get_gcmagicc_archive_candidates(version_tag, include_local_repo=True):
        resolved = Path(candidate).expanduser().resolve(strict=False)
        matches = matching_gapfiller_cmip6_archive_files(
            resolved,
            scenarios=wanted_scenarios,
            source_ids=source_ids,
            members=members,
        )
        if matches:
            matched_scenarios = sorted(
                {
                    str((_parse_archive_filename(path)[1] or "")).strip().lower()
                    for path in matches
                    if _parse_archive_filename(path)[1]
                }
            )
            return Plot810Source(
                path=resolved,
                source_kind="legacy_archive",
                label="gcmagicc_cmip6",
                remote_prefix=remote_prefix,
                stage_root=stage_root,
                mount_root=mount_root,
                available_scenarios=tuple(matched_scenarios),
                missing_scenarios=tuple(sorted(set(wanted_scenarios) - set(matched_scenarios))),
            )

    return Plot810Source(
        path=mount_root,
        source_kind="missing",
        label="gcmagicc_cmip6",
        remote_prefix=remote_prefix,
        stage_root=stage_root,
        mount_root=mount_root,
        available_scenarios=(),
        missing_scenarios=tuple(sorted(set(wanted_scenarios))),
    )


def resolve_plot810_cmip6_source(
    *,
    scenarios: Sequence[str],
    variables: Sequence[str],
    explicit_dir: Path | None = None,
    stage_base: Path | None = None,
    prefer_staged: bool = True,
) -> Plot810Source:
    remote_prefix = f"s3://{get_object_bucket()}/nc/reference/out_ETHFOG_10June2025_vetted"
    stage_root = build_plot810_cmip6_stage_root(
        scenarios=scenarios,
        variables=variables,
        stage_base=stage_base,
    )
    mount_root = get_cmip6_vetted_candidates()[0]

    if explicit_dir is not None:
        resolved = Path(explicit_dir).expanduser().resolve(strict=False)
        inventory = _cmip6_inventory(resolved, scenarios=scenarios, variables=variables)
        matches = list(inventory["files"])
        source_kind = _classify_root_with_files(resolved, mounted_label="mounted", local_label="local") if matches else "missing"
        return Plot810Source(
            path=resolved,
            source_kind=source_kind,
            label="cmip6_raw",
            remote_prefix=remote_prefix,
            stage_root=stage_root,
            mount_root=mount_root,
            explicit=True,
            available_scenarios=tuple(inventory["available_scenarios"]),
            missing_scenarios=tuple(inventory["missing_scenarios"]),
            matched_variables=tuple(inventory["matched_variables"]),
            missing_variables=tuple(inventory["missing_variables"]),
        )

    if prefer_staged:
        stage_inventory = _cmip6_inventory(stage_root, scenarios=scenarios, variables=variables)
        stage_matches = list(stage_inventory["files"])
        if stage_matches:
            return Plot810Source(
                path=stage_root,
                source_kind="staged",
                label="cmip6_raw",
                remote_prefix=remote_prefix,
                stage_root=stage_root,
                mount_root=mount_root,
                available_scenarios=tuple(stage_inventory["available_scenarios"]),
                missing_scenarios=tuple(stage_inventory["missing_scenarios"]),
                matched_variables=tuple(stage_inventory["matched_variables"]),
                missing_variables=tuple(stage_inventory["missing_variables"]),
            )

    for candidate in get_cmip6_vetted_candidates():
        resolved = Path(candidate).expanduser().resolve(strict=False)
        inventory = _cmip6_inventory(resolved, scenarios=scenarios, variables=variables)
        matches = list(inventory["files"])
        if matches:
            return Plot810Source(
                path=resolved,
                source_kind=_classify_root_with_files(resolved, mounted_label="mounted", local_label="local"),
                label="cmip6_raw",
                remote_prefix=remote_prefix,
                stage_root=stage_root,
                mount_root=mount_root,
                available_scenarios=tuple(inventory["available_scenarios"]),
                missing_scenarios=tuple(inventory["missing_scenarios"]),
                matched_variables=tuple(inventory["matched_variables"]),
                missing_variables=tuple(inventory["missing_variables"]),
            )

    empty_inventory = _cmip6_inventory(mount_root, scenarios=scenarios, variables=variables)
    return Plot810Source(
        path=mount_root,
        source_kind="missing",
        label="cmip6_raw",
        remote_prefix=remote_prefix,
        stage_root=stage_root,
        mount_root=mount_root,
        available_scenarios=tuple(empty_inventory["available_scenarios"]),
        missing_scenarios=tuple(empty_inventory["missing_scenarios"]),
        matched_variables=tuple(empty_inventory["matched_variables"]),
        missing_variables=tuple(empty_inventory["missing_variables"]),
    )


def _infer_scenario_from_run_root(path: Path) -> str | None:
    resolved = Path(path).expanduser().resolve(strict=False)
    if resolved.name.startswith("run_") and len(resolved.parts) >= 6:
        try:
            return resolved.parents[4].name
        except IndexError:
            return None
    return None


def _resolve_explicit_gxe_run_root(
    *,
    explicit_dir: Path,
    version_tag: str,
    scenario: str,
    gxe_ensemble: str,
) -> Path | None:
    resolved = Path(explicit_dir).expanduser().resolve(strict=False)
    if _iter_nc_files(resolved):
        inferred_scenario = _infer_scenario_from_run_root(resolved)
        return resolved if inferred_scenario in {None, str(scenario).strip()} else None

    candidates: list[Path] = []
    scenario_token = str(scenario).strip()
    if resolved.name == version_tag:
        candidates.append(resolved / scenario_token / "AR6" / "all" / gxe_ensemble / "original")
    candidates.append(resolved / version_tag / scenario_token / "AR6" / "all" / gxe_ensemble / "original")
    candidates.append(resolved / scenario_token / "AR6" / "all" / gxe_ensemble / "original")
    candidates.append(resolved / scenario_token / "AR6")
    for candidate in candidates:
        run_root = _latest_run_root(candidate)
        if run_root is not None:
            return run_root
    return None


def build_plot810_gxe_stage_command(
    *,
    version: str | None,
    scenario: str,
    ensemble: str | int,
    run_instance: str | None = None,
    stage_base: Path | None = None,
    dry_run: bool = False,
) -> list[str]:
    return build_gapfiller_era5_stage_command(
        version=version,
        scenario=scenario,
        ensemble=ensemble,
        run_instance=run_instance,
        stage_base=stage_base,
        dry_run=dry_run,
    )


def resolve_plot810_gxe_source(
    *,
    version: str | None,
    scenarios: Sequence[str],
    gxe_ensemble: str | int = "n_20",
    explicit_dir: Path | None = None,
    stage_base: Path | None = None,
    prefer_staged: bool = True,
) -> Plot810Source:
    version_tag = normalize_version_tag(version)
    ensemble_token = normalize_n_ensemble_label(gxe_ensemble)
    requested_scenarios = tuple(sorted(item.lower() for item in _normalize_items(scenarios)))
    stage_root = Path(stage_base or get_era5spliced_localstaging_root()).expanduser().resolve(strict=False)
    mount_root = Path(get_era5spliced_root()).expanduser().resolve(strict=False)
    scenario_sources: dict[str, Plot810ScenarioSource] = {}
    missing_scenarios: list[str] = []
    overall_path = Path(explicit_dir).expanduser().resolve(strict=False) if explicit_dir is not None else (
        mount_root / version_tag
    ).resolve(strict=False)

    for scenario in requested_scenarios:
        try:
            if explicit_dir is not None:
                run_root = _resolve_explicit_gxe_run_root(
                    explicit_dir=Path(explicit_dir),
                    version_tag=version_tag,
                    scenario=scenario,
                    gxe_ensemble=ensemble_token,
                )
                if run_root is None:
                    raise FileNotFoundError(f"No explicit GXE run root found for {scenario}")
                meta = parse_era5spliced_dataset_path(run_root, root=run_root.parents[6] if len(run_root.parts) >= 7 else None)
                scenario_sources[scenario] = Plot810ScenarioSource(
                    scenario=scenario,
                    path=run_root,
                    source_kind=_classify_root_with_files(run_root, mounted_label="mounted", local_label="local"),
                    remote_prefix=None,
                    stage_root=stage_root,
                    mount_root=mount_root,
                    explicit=True,
                    run_instance=(meta or {}).get("run_instance") or (run_root.name if run_root.name.startswith("run_") else None),
                )
                continue

            resolved = resolve_gapfiller_era5_source(
                version=version_tag,
                experiment_id=scenario,
                arx="AR6",
                runmodus="all",
                n_ensemble=ensemble_token,
                run_instance=None,
                canonical_root=mount_root,
                stage_root=stage_root,
                prefer_staged=prefer_staged,
            )
            scenario_sources[scenario] = Plot810ScenarioSource(
                scenario=scenario,
                path=resolved.path,
                source_kind=_normalize_gapfiller_source_kind(resolved.source_kind),
                remote_prefix=resolved.remote_prefix,
                stage_root=stage_root,
                mount_root=mount_root,
                explicit=False,
                run_instance=resolved.run_instance,
            )
        except FileNotFoundError:
            missing_scenarios.append(scenario)

    available_scenarios = tuple(sorted(scenario_sources))
    overall_kind = "missing"
    run_instance = None
    if scenario_sources:
        kinds = {item.source_kind for item in scenario_sources.values()}
        overall_kind = kinds.pop() if len(kinds) == 1 else "mixed"
        run_instances = {item.run_instance for item in scenario_sources.values() if item.run_instance}
        if len(run_instances) == 1:
            run_instance = next(iter(run_instances))

    return Plot810Source(
        path=overall_path,
        source_kind=overall_kind,
        label="gcmagicc_gxe",
        remote_prefix=None,
        stage_root=stage_root,
        mount_root=mount_root,
        explicit=explicit_dir is not None,
        available_scenarios=available_scenarios,
        missing_scenarios=tuple(sorted(set(requested_scenarios) - set(available_scenarios))),
        scenario_sources=scenario_sources,
        run_instance=run_instance,
    )


def build_plot810_gcmagicc_stage_command(
    *,
    version: str | None,
    scenarios: Sequence[str],
    stage_base: Path | None = None,
    stage_root: Path | None = None,
    source_ids: Iterable[str] | None = None,
    members: Iterable[str] | None = None,
    dry_run: bool = False,
) -> list[str]:
    return build_gapfiller_cmip6_stage_command(
        version=version,
        scenario=list(_normalize_items(scenarios)),
        source_ids=source_ids,
        members=members,
        stage_base=stage_base,
        stage_root=stage_root,
        dry_run=dry_run,
    )
