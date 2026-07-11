"""Utilities for resolving scenario source roots for the 815 exporter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote

from scr.validation_helpers.helper_path_utils import get_site

_MEMBER_RE = re.compile(r"(r\d+i\d+p\d+f\d+)", re.IGNORECASE)
_MODEL_RE = re.compile(r"(d\d+b\d+e\d+m\d+)", re.IGNORECASE)
_RUN_TS_RE = re.compile(r"(\d{8}-\d{4}|\d{8}_\d{6}|\d{8})")
_N_FOLDER_RE = re.compile(r"n_(\d+)", re.IGNORECASE)
_DERIVATIVE_DIR_TOKENS = {"data_derivatives", "data_derivatives_archive", "dataderivatives"}


@dataclass(frozen=True)
class SourceRoot:
    path: Path
    tag: str
    priority: int = 0
    origin: str = ""


@dataclass(frozen=True)
class ScenarioCandidate:
    scenario: str
    root: Path
    tag: str
    priority: int
    member_count: int
    files: List[Path]
    source_run_path: Optional[Path] = None
    source_run_name: Optional[str] = None


@dataclass(frozen=True)
class ScenarioSelection:
    scenario: str
    chosen: Optional[ScenarioCandidate]
    candidates: List[ScenarioCandidate]
    excluded_reason: Optional[str] = None


def storage_region_id(region: str) -> str:
    """Create a path-safe region token that preserves reversibility."""
    return quote(str(region), safe=".-_")


def experiment_token(path: Path) -> str | None:
    """Extract the scenario/experiment token from a GCMAGICC member filename."""
    stem = path.stem
    if "_ERA5_" not in stem:
        return None
    tail = stem.split("_ERA5_", 1)[1]
    member = _MEMBER_RE.search(tail)
    if member:
        token = tail[: member.start()].rstrip("_").strip()
        return token.lower() if token else None
    token = tail.split("_", 1)[0].strip()
    return token.lower() if token else None


def _normalise_experiment_token(value: str) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _filter_files_for_experiment(files: Sequence[Path], scenario: str) -> List[Path]:
    requested = _normalise_experiment_token(scenario)
    if not requested:
        return sorted(files)
    matches: List[Path] = []
    saw_token = False
    for path in files:
        token = experiment_token(path)
        if token is None:
            continue
        saw_token = True
        if _normalise_experiment_token(token) == requested:
            matches.append(path)
    if matches:
        return sorted(matches)
    return [] if saw_token else sorted(files)


def member_identity(path: Path) -> str:
    """Return a stable member identity used for deduplication across roots."""
    stem = path.stem
    model = _MODEL_RE.search(stem)
    member = _MEMBER_RE.search(stem)
    experiment = experiment_token(path)
    parts: list[str] = []
    if model:
        parts.append(model.group(1).lower())
    if member:
        parts.append(member.group(1).lower())
    if experiment:
        parts.append(experiment)
    if parts:
        return "|".join(parts)
    return stem.lower()


def dedupe_member_files(files: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()
    for p in sorted(files):
        key = member_identity(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _extract_run_token(path: Path) -> str:
    matches: List[str] = []
    for part in path.parts:
        matches.extend(_RUN_TS_RE.findall(part))
    if not matches:
        return ""
    # Lexicographic sort works for these timestamp token formats.
    return sorted(matches)[-1]


def _candidate_rank_key(candidate: ScenarioCandidate) -> tuple[int, str, str]:
    run_path = candidate.source_run_path or candidate.root
    token = _extract_run_token(run_path)
    return (int(candidate.priority), token, str(run_path))


def _source_run_path_for_files(files: Sequence[Path]) -> Optional[Path]:
    if not files:
        return None
    parents = {p.parent for p in files}
    if len(parents) == 1:
        return next(iter(parents))
    try:
        return Path(os.path.commonpath([str(p.parent) for p in files]))
    except ValueError:
        return None


def _expand_path_token(raw: str) -> str:
    return os.path.expandvars(os.path.expanduser(str(raw)))


def _path_from_env_spec(env_key: str, suffix: Any = None) -> Optional[Path]:
    env_raw = os.environ.get(str(env_key), "").strip()
    if not env_raw:
        return None
    path = Path(_expand_path_token(env_raw))
    suffix_raw = str(suffix or "").strip()
    if suffix_raw:
        suffix_path = Path(_expand_path_token(suffix_raw).lstrip("/"))
        path = path / suffix_path
    return path


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _safe_iterdir(path: Path) -> List[Path]:
    try:
        return list(path.iterdir())
    except OSError:
        return []


def _safe_glob(pattern: str) -> List[str]:
    try:
        return glob.glob(pattern)
    except OSError:
        return []


def _safe_path_glob(path: Path, pattern: str) -> List[Path]:
    try:
        return list(path.glob(pattern))
    except OSError:
        return []


def _safe_path_rglob(path: Path, pattern: str) -> List[Path]:
    try:
        return list(path.rglob(pattern))
    except OSError:
        return []


def _candidate_specs_for_site(roots_cfg: Any, site: str) -> List[Dict[str, Any]]:
    if isinstance(roots_cfg, list):
        return [dict(item) for item in roots_cfg if isinstance(item, dict)]
    if isinstance(roots_cfg, dict):
        selected: List[Dict[str, Any]] = []
        for key in (site, "all"):
            block = roots_cfg.get(key)
            if isinstance(block, list):
                selected.extend(dict(item) for item in block if isinstance(item, dict))
        return selected
    return []


def expand_source_roots(config: Dict[str, Any], *, version_family: str, site: Optional[str] = None) -> List[SourceRoot]:
    families = config.get("families", {}) if isinstance(config, dict) else {}
    if not isinstance(families, dict):
        raise ValueError("Invalid source config: missing 'families' object")
    if version_family not in families:
        raise KeyError(f"Version family '{version_family}' not found in source config")

    family_cfg = families[version_family]
    if not isinstance(family_cfg, dict):
        raise ValueError(f"Invalid family block for '{version_family}'")

    active_site = str(site or get_site()).strip().lower()
    specs = _candidate_specs_for_site(family_cfg.get("roots"), active_site)

    roots: List[SourceRoot] = []
    seen: set[str] = set()
    for idx, spec in enumerate(specs):
        if not isinstance(spec, dict):
            continue
        priority = int(spec.get("priority", 0) or 0)
        tag = str(spec.get("tag") or family_cfg.get("source_tags", [version_family])[0]).strip() or version_family
        origin = str(spec.get("name") or spec.get("origin") or f"candidate_{idx}")

        path_entries: List[Path] = []
        raw_path = spec.get("path")
        raw_glob = spec.get("glob")
        raw_path_env = spec.get("path_env")

        if raw_path:
            path_entries.append(Path(_expand_path_token(str(raw_path))))
        if raw_path_env:
            env_path = _path_from_env_spec(str(raw_path_env), spec.get("path_suffix"))
            if env_path is not None:
                path_entries.append(env_path)
        if raw_glob:
            for raw in sorted(_safe_glob(_expand_path_token(str(raw_glob)))):
                path_entries.append(Path(raw))

        for path in path_entries:
            resolved = path.expanduser().resolve(strict=False)
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            roots.append(SourceRoot(path=resolved, tag=tag, priority=priority, origin=origin))

    roots.sort(key=lambda item: (-item.priority, str(item.path)))
    return roots


def discover_scenarios_for_root(root: Path) -> List[str]:
    if not _safe_exists(root) or not _safe_is_dir(root):
        return []
    return sorted(p.name for p in _safe_iterdir(root) if _safe_is_dir(p))


def _path_uses_derivative_dir(path: Path) -> bool:
    return any(part in _DERIVATIVE_DIR_TOKENS for part in path.parts)


def _sorted_nc_files_under(root: Path) -> List[Path]:
    return sorted(
        p for p in _safe_path_rglob(root, "*.nc") if _safe_is_file(p) and not _path_uses_derivative_dir(p.relative_to(root))
    )


def _direct_nc_files_under(root: Path) -> List[Path]:
    return sorted(p for p in _safe_path_glob(root, "*.nc") if _safe_is_file(p))


def _n_folder_size(path: Path) -> int:
    for part in path.parts:
        match = _N_FOLDER_RE.fullmatch(part)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return 10**9
    return 10**9


def _original_dir_sort_key(path: Path) -> tuple[int, str]:
    return (_n_folder_size(path), str(path))


def _file_group_sort_key(files: Sequence[Path]) -> tuple[int, str]:
    parent = files[0].parent if files else Path("")
    return (-len(dedupe_member_files(files)), str(parent))


def _nc_file_groups_for_original_dir(original_dir: Path) -> List[List[Path]]:
    # Canonical n_* original directories may contain multiple run_* children.
    # Keep those runs separate so strict n_20 resolution does not merge a
    # complete 20-member run with later one-member incremental landings.
    grouped: List[List[Path]] = []
    child_dirs = sorted(
        (
            p
            for p in _safe_iterdir(original_dir)
            if _safe_is_dir(p) and not _path_uses_derivative_dir(p.relative_to(original_dir))
        ),
        key=str,
    )
    for child in child_dirs:
        files = _sorted_nc_files_under(child)
        if files:
            grouped.append(files)

    direct_files = _direct_nc_files_under(original_dir)
    if direct_files:
        grouped.append(direct_files)

    if grouped:
        return sorted(grouped, key=_file_group_sort_key)

    files = _sorted_nc_files_under(original_dir)
    return [files] if files else []


def _candidate_scenario_dirs(root: Path, scenario: str) -> List[Path]:
    raw = str(scenario).strip()
    if not raw:
        return []
    candidates = [root / raw]
    normalised = _normalise_experiment_token(raw)
    if normalised.endswith("-nat"):
        base = raw[: -len("-nat")]
        if base:
            candidates.append(root / base)
    seen: set[str] = set()
    existing: List[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _safe_exists(candidate) and _safe_is_dir(candidate):
            existing.append(candidate)
    return existing


def scenario_nc_file_groups(root: Path, scenario: str) -> List[List[Path]]:
    grouped_files: List[List[Path]] = []
    for scen_dir in _candidate_scenario_dirs(root, scenario):
        dir_groups: List[List[Path]] = []
        nc_dir = scen_dir / "AR6" if _safe_is_dir(scen_dir / "AR6") else scen_dir
        files = _filter_files_for_experiment(
            sorted(p for p in _safe_path_glob(nc_dir, "*.nc") if _safe_is_file(p)),
            scenario,
        )
        if files:
            dir_groups.append(files)
            grouped_files.extend(dir_groups)
            continue

        # Canonical layouts store members below .../AR6/<runmodus>/n_*/original/.
        # Prefer those trees and keep derivative folders out of the discovery set.
        canonical_original_dirs = sorted(
            (p for p in _safe_path_glob(nc_dir, "*/n_*/original") if _safe_is_dir(p)),
            key=_original_dir_sort_key,
        )
        for canonical_original in canonical_original_dirs:
            for group in _nc_file_groups_for_original_dir(canonical_original):
                filtered = _filter_files_for_experiment(group, scenario)
                if filtered:
                    dir_groups.append(filtered)
        if dir_groups:
            grouped_files.extend(dir_groups)
            continue

        original_dirs = sorted((p for p in _safe_path_rglob(nc_dir, "original") if _safe_is_dir(p)), key=_original_dir_sort_key)
        for original_dir in original_dirs:
            files = _filter_files_for_experiment(_sorted_nc_files_under(original_dir), scenario)
            if files:
                dir_groups.append(files)
        if dir_groups:
            grouped_files.extend(dir_groups)
            continue

        fallback = _filter_files_for_experiment(_sorted_nc_files_under(nc_dir), scenario)
        if fallback:
            dir_groups.append(fallback)
            grouped_files.extend(dir_groups)
    if grouped_files:
        return grouped_files
    return []


def scenario_nc_files(root: Path, scenario: str) -> List[Path]:
    groups = scenario_nc_file_groups(root, scenario)
    if not groups:
        return []
    return groups[0]


def resolve_scenarios(
    roots: Sequence[SourceRoot],
    *,
    scenarios: Optional[Sequence[str]] = None,
    strict_member_count: Optional[int] = None,
    required_source_run_name: Optional[str] = None,
) -> List[ScenarioSelection]:
    required_run_name = str(required_source_run_name or "").strip() or None
    scenario_names: List[str]
    if scenarios:
        scenario_names = list(dict.fromkeys(str(s) for s in scenarios if str(s).strip()))
    else:
        discovered: set[str] = set()
        for root in roots:
            discovered.update(discover_scenarios_for_root(root.path))
        scenario_names = sorted(discovered)

    selections: List[ScenarioSelection] = []

    for scenario in scenario_names:
        candidates: List[ScenarioCandidate] = []
        available_candidates: List[ScenarioCandidate] = []
        for root in roots:
            file_groups = scenario_nc_file_groups(root.path, scenario)
            if not file_groups:
                continue

            root_candidates: List[ScenarioCandidate] = []
            for files in file_groups:
                dedup = dedupe_member_files(files)
                if not dedup:
                    continue
                source_run_path = _source_run_path_for_files(dedup)
                root_candidates.append(
                    ScenarioCandidate(
                        scenario=scenario,
                        root=root.path,
                        tag=root.tag,
                        priority=root.priority,
                        member_count=len(dedup),
                        files=dedup,
                        source_run_path=source_run_path,
                        source_run_name=source_run_path.name if source_run_path is not None else None,
                    )
                )
            if not root_candidates:
                continue

            available_candidates.extend(root_candidates)
            if required_run_name is not None:
                root_candidates = [c for c in root_candidates if c.source_run_name == required_run_name]
                if not root_candidates:
                    continue

            if strict_member_count is not None:
                eligible_in_root = [c for c in root_candidates if c.member_count == int(strict_member_count)]
                if eligible_in_root:
                    candidates.append(sorted(eligible_in_root, key=_candidate_rank_key, reverse=True)[0])
                    continue

            candidates.append(sorted(root_candidates, key=_candidate_rank_key, reverse=True)[0])

        if not candidates:
            reason = "missing_in_all_roots"
            if available_candidates and required_run_name is not None:
                reason = f"source_run_name_not_found:{required_run_name}"
            selections.append(
                ScenarioSelection(
                    scenario=scenario,
                    chosen=None,
                    candidates=sorted(available_candidates, key=_candidate_rank_key, reverse=True),
                    excluded_reason=reason,
                )
            )
            continue

        eligible = candidates
        if strict_member_count is not None:
            eligible = [c for c in candidates if c.member_count == int(strict_member_count)]

        if not eligible:
            count_values = sorted({c.member_count for c in candidates})
            if strict_member_count is not None and len(count_values) > 1:
                reason = f"mixed_member_counts:{','.join(str(v) for v in count_values)}"
            elif strict_member_count is not None:
                reason = f"member_count_not_{int(strict_member_count)}:{','.join(str(v) for v in count_values)}"
            else:
                reason = "no_eligible_candidates"
            selections.append(
                ScenarioSelection(
                    scenario=scenario,
                    chosen=None,
                    candidates=candidates,
                    excluded_reason=reason,
                )
            )
            continue

        chosen = sorted(eligible, key=_candidate_rank_key, reverse=True)[0]
        selections.append(
            ScenarioSelection(
                scenario=scenario,
                chosen=chosen,
                candidates=sorted(candidates, key=_candidate_rank_key, reverse=True),
                excluded_reason=None,
            )
        )

    return selections


def selections_manifest(
    selections: Sequence[ScenarioSelection],
    *,
    version_family: str,
    version_tag: str,
    strict_member_count: Optional[int],
    source_config: Optional[Path],
    timetag: str,
    output_root: Path,
    required_source_run_name: Optional[str] = None,
) -> Dict[str, Any]:
    chosen = [s for s in selections if s.chosen is not None]
    skipped = [s for s in selections if s.chosen is None]

    chosen_rows: List[Dict[str, Any]] = []
    for row in chosen:
        assert row.chosen is not None
        chosen_rows.append(
            {
                "scenario": row.scenario,
                "root": str(row.chosen.root),
                "source_tag": row.chosen.tag,
                "priority": row.chosen.priority,
                "member_count": row.chosen.member_count,
                "source_run_name": row.chosen.source_run_name,
                "source_run_path": str(row.chosen.source_run_path) if row.chosen.source_run_path else None,
                "member_identities": [member_identity(p) for p in row.chosen.files],
                "files": [str(p) for p in row.chosen.files],
            }
        )

    skipped_rows: List[Dict[str, Any]] = []
    for row in skipped:
        skipped_rows.append(
            {
                "scenario": row.scenario,
                "reason": row.excluded_reason,
                "candidates": [
                    {
                        "root": str(c.root),
                        "source_tag": c.tag,
                        "priority": c.priority,
                        "member_count": c.member_count,
                        "source_run_name": c.source_run_name,
                        "source_run_path": str(c.source_run_path) if c.source_run_path else None,
                    }
                    for c in row.candidates
                ],
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version_family": version_family,
        "version_tag": version_tag,
        "strict_member_count": strict_member_count,
        "required_source_run_name": str(required_source_run_name or "") or None,
        "source_config": str(source_config) if source_config else None,
        "timetag": timetag,
        "output_root": str(output_root),
        "summary": {
            "scenarios_considered": len(selections),
            "scenarios_selected": len(chosen_rows),
            "scenarios_skipped": len(skipped_rows),
            "selected_member_total": sum(int(r["member_count"]) for r in chosen_rows),
        },
        "selected": chosen_rows,
        "skipped": skipped_rows,
    }


def load_source_config(path: Path) -> Dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve(strict=False)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid source config payload in {cfg_path}")
    return payload


def family_defaults(config: Dict[str, Any], version_family: str) -> Dict[str, Any]:
    families = config.get("families", {}) if isinstance(config, dict) else {}
    if not isinstance(families, dict) or version_family not in families:
        raise KeyError(f"Version family '{version_family}' not found")
    block = families[version_family]
    if not isinstance(block, dict):
        raise ValueError(f"Invalid family block for '{version_family}'")
    return {
        "version_tag": str(block.get("version_tag") or version_family),
        "source_tags": list(block.get("source_tags") or [version_family]),
        "output_layout": str(block.get("output_layout") or "versioned"),
        "public_version": str(block.get("public_version") or version_family),
    }


def selected_roots_from_manifest(manifest: Dict[str, Any]) -> List[Path]:
    selected = manifest.get("selected", []) if isinstance(manifest, dict) else []
    out: List[Path] = []
    seen: set[str] = set()
    for row in selected:
        if not isinstance(row, dict):
            continue
        root = Path(str(row.get("root", ""))).expanduser().resolve(strict=False)
        key = str(root)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out
