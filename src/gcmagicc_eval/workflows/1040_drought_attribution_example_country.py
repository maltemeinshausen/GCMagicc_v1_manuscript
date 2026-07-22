#!/usr/bin/env python3
"""
1040_drought_attribution_example_country
==============================================

Build a condensed manuscript drought-attribution figure for a country drought-attribution
example from the published 758/759 numeric payloads.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
import shlex
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, MultipleLocator


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scr.validation_helpers.helper_fonts import ResolvedFontFamily, apply_sans_font_rcparams
from scr.validation_helpers.helper_path_utils import get_data_path

_base758 = importlib.import_module("notebooks.758_GapFiller_SPEI")


DEFAULT_VERSION_TAG = "v100"
DEFAULT_SCENARIO1 = "ssp245"
DEFAULT_SCENARIO2 = "ssp245-nat"
DEFAULT_COUNTRY_ISO3 = "IRN"
DEFAULT_PET_METHOD = "penman-monteith"
DEFAULT_SOURCE_TIMETAG = "auto"
DEFAULT_LOCAL_PUBLISH_ROOT = Path(get_data_path("drought_attribution_758")).expanduser().resolve(strict=False)
DEFAULT_GUS_HOST = "gus"
DEFAULT_GUS_PUBLISH_ROOT = Path("data/site_gus/projects/gcmmagicc/data/drought_attribution_758")
DEFAULT_DROUGHT_ROOT = REPO_ROOT / "data" / "manuscript_figures" / "drought_attribution"
DEFAULT_CACHE_ROOT = DEFAULT_DROUGHT_ROOT / "cache"
DEFAULT_OUTDIR = DEFAULT_DROUGHT_ROOT
DEFAULT_STEM = "auto"
VERSION_LABELS = {
    "v100": "GCMagicc",
    "v101": "GCMagicc-CE",
}
MM_PER_INCH = 25.4
DEFAULT_WIDTH_MM = 183.0
DEFAULT_WIDTH_IN = DEFAULT_WIDTH_MM / MM_PER_INCH
DEFAULT_HEIGHT_IN = 5.35
DEFAULT_HEIGHT_MM = DEFAULT_HEIGHT_IN * MM_PER_INCH
DEFAULT_DPI = 300
DEFAULT_JOURNAL_TARGET = "nature-full-width"
SSH_CONNECT_TIMEOUT = 12

FONT_FAMILY_CANDIDATES = (
    "Helvetica",
    "Arial",
    "Nimbus Sans",
    "Liberation Sans",
)

PANEL_LABEL_GID = "nature-panel-label"
PANEL_LABEL_PT = 8.0
BODY_FONT_PT = 5.6
AXIS_LABEL_PT = 6.0
TITLE_FONT_PT = 6.2
TICK_FONT_PT = 5.2
MIN_TEXT_PT = 5.0
MAX_TEXT_PT = 7.0
LEGEND_FONT_PT = 5.0
INLINE_HEADER_PT = 5.4

MANUSCRIPT_TRACE_LW = 0.24
MANUSCRIPT_TRACE_ALPHA = 0.08
MANUSCRIPT_MEDIAN_LW = 1.0
MANUSCRIPT_MEDIAN_ALPHA = 0.16
MANUSCRIPT_MEAN_LW = 1.45
MANUSCRIPT_MEAN_ALPHA = 0.24
MANUSCRIPT_ZERO_LINE_LW = 0.65
MANUSCRIPT_CMIP6_LW = 0.55
MANUSCRIPT_CMIP6_ALPHA = 0.28
MANUSCRIPT_ERA5_LW = 1.7
MANUSCRIPT_KEUNE_MARKER_SIZE = 1.45

_ACTIVE_FONT_FAMILY = "DejaVu Sans"
_ACTIVE_FONT_PATH = ""

REMOTE_LIST_TIMETAGS_SCRIPT = """
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.exists():
    print("[]")
    raise SystemExit(0)
items = sorted([p.name for p in root.iterdir() if p.is_dir()], reverse=True)
print(json.dumps(items))
"""

REMOTE_COMBO_STATUS_SCRIPT = """
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
combo_key = sys.argv[2]
combo_dir = Path(sys.argv[3])
canonical_json = Path(sys.argv[4])
panel_a_path = Path(sys.argv[5])
panel_i_path = Path(sys.argv[6])

def payload_variants(path):
    out = [path]
    name = path.name
    if name.endswith(".json"):
        out.append(path.with_name(name + ".gz"))
    elif name.endswith(".json.gz"):
        out.append(path.with_name(name[:-3]))
    return out

payload_path = ""
for candidate in payload_variants(canonical_json):
    if candidate.exists():
        payload_path = str(candidate)
        break

stats_paths = sorted(str(p) for p in combo_dir.glob("SPEI_STATS_*.json") if p.is_file())
manifest_has_combo = False
if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        manifest = None
    if isinstance(manifest, dict):
        completed = manifest.get("completed_combos", [])
        if isinstance(completed, list):
            manifest_has_combo = any(
                isinstance(row, dict) and str(row.get("combo_key", "")).strip() == combo_key
                for row in completed
            )

core_files_ok = bool(payload_path) and panel_a_path.exists() and panel_i_path.exists()
print(
    json.dumps(
        {
            "manifest_has_combo": bool(manifest_has_combo),
            "core_files_ok": bool(core_files_ok),
            "payload_path": payload_path,
            "panel_a_path": str(panel_a_path),
            "panel_i_path": str(panel_i_path),
            "stats_paths": stats_paths,
        }
    )
)
"""


@dataclass(frozen=True)
class SourceBundle:
    source_type: str
    version_tag: str
    scenario_pair_tag: str
    timetag: str
    combo_dir: Path
    manifest_path: Path
    payload_path: Path
    panel_a_json: Path
    panel_i_json: Path
    stats_paths: Tuple[Path, ...]
    origin_manifest_path: str
    origin_payload_path: str


@dataclass(frozen=True)
class RemoteBundle:
    version_tag: str
    scenario_pair_tag: str
    timetag: str
    combo_dir: str
    manifest_path: str
    payload_path: str
    panel_a_json: str
    panel_i_json: str
    stats_paths: Tuple[str, ...]


@dataclass(frozen=True)
class LoadedFigurePayload:
    payload: Mapping[str, Any]
    map_series: List[_base758.SPEISeries]
    map_titles: List[str]
    era5_series: _base758.SPEISeries
    era5drought_keune_series: List[_base758.SPEISeries]
    all_list: List[_base758.SPEISeries]
    nat_list: List[_base758.SPEISeries]
    cmip6_hist_list: List[_base758.SPEISeries]
    cmip6_hist_nat_list: List[_base758.SPEISeries]
    cmip6_ssp245_list: List[_base758.SPEISeries]
    region: str
    region_long_name: str
    scenario1_tag: str
    scenario2_tag: str
    scenario1_label: str
    scenario2_label: str
    pet_method: str
    scale: int
    hist_window_current: Tuple[int, int]
    hist_window_future: Tuple[int, int]


def _run_command(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    cp = subprocess.run(
        list(args),
        check=False,
        text=True,
        capture_output=True,
    )
    if check and cp.returncode != 0:
        stderr = cp.stderr.strip() or cp.stdout.strip()
        raise RuntimeError(f"Command failed ({cp.returncode}): {' '.join(args)}\n{stderr}")
    return cp


def _normalize_country_iso3(value: str) -> str:
    token = str(value).strip().upper()
    if len(token) != 3 or not token.isalpha():
        raise ValueError(f"Country ISO3 code must be three letters, got {value!r}")
    return token


def _country_iso3_arg(value: str) -> str:
    try:
        return _normalize_country_iso3(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _normalize_version_tag(value: str) -> str:
    token = str(value).strip().lower()
    if token not in VERSION_LABELS:
        allowed = ", ".join(sorted(VERSION_LABELS))
        raise ValueError(f"Version tag must be one of {allowed}, got {value!r}")
    return token


def _version_tag_arg(value: str) -> str:
    try:
        return _normalize_version_tag(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _version_label(version_tag: str) -> str:
    return VERSION_LABELS.get(str(version_tag).strip().lower(), str(version_tag).strip())


def _safe_filename_token(value: str) -> str:
    token = str(value).strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in token)


def _default_output_stem(*, country_iso3: str, version_tag: str) -> str:
    return "drought_attribution_{country}_{version}_{model}".format(
        country=_safe_filename_token(_normalize_country_iso3(country_iso3)),
        version=_safe_filename_token(_normalize_version_tag(version_tag)),
        model=_safe_filename_token(_version_label(version_tag)),
    )


def _resolve_output_stem(stem: str, *, country_iso3: str, version_tag: str) -> str:
    token = str(stem).strip()
    if not token or token.lower() == "auto":
        return _default_output_stem(country_iso3=country_iso3, version_tag=version_tag)
    return token


def _missing_source_warning(args: argparse.Namespace) -> str:
    version_tag = _normalize_version_tag(args.version_tag)
    country_iso3 = _normalize_country_iso3(args.region)
    return (
        "No published drought-attribution payload was found for "
        f"{version_tag} ({_version_label(version_tag)}) / "
        f"{args.scenario1} vs {args.scenario2} / {country_iso3} / {args.pet_method}. "
        "Either the 760SUPER* publication is not complete yet, or a specifically "
        "targeted 754/758 run must be executed for this combo."
    )


def _ssh_json(host: str, script: str, *args: object) -> Any:
    remote_cmd = "python3 -c {script} {args}".format(
        script=shlex.quote(str(script)),
        args=" ".join(shlex.quote(str(arg)) for arg in args),
    ).strip()
    cp = _run_command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
            str(host),
            remote_cmd,
        ]
    )
    text = cp.stdout.strip()
    if not text:
        raise RuntimeError(f"Remote JSON response from {host} was empty.")
    return json.loads(text)


def _rsync_args(host: str, remote_path: str, local_path: Path) -> List[str]:
    dst = Path(local_path).expanduser().resolve(strict=False)
    return [
        "rsync",
        "-az",
        "--partial",
        "--timeout=60",
        "-e",
        f"ssh -o BatchMode=yes -o ConnectTimeout={SSH_CONNECT_TIMEOUT}",
        f"{host}:{remote_path}",
        str(dst),
    ]


def _rsync_command_text(host: str, remote_path: str, local_path: Path) -> str:
    return shlex.join(_rsync_args(host, remote_path, local_path))


def _rsync_file(host: str, remote_path: str, local_path: Path) -> None:
    dst = Path(local_path).expanduser().resolve(strict=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cache] rsync: {_rsync_command_text(host, remote_path, dst)}", flush=True)
    _run_command(_rsync_args(host, remote_path, dst))


def _scp_file(host: str, remote_path: str, local_path: Path) -> None:
    dst = Path(local_path).expanduser().resolve(strict=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
            f"{host}:{remote_path}",
            str(dst),
        ]
    )


def _ssh_copy_file(host: str, remote_path: str, local_path: Path) -> None:
    dst = Path(local_path).expanduser().resolve(strict=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "import pathlib, sys; "
        "sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())"
    )
    remote_cmd = "python3 -c {script} {path}".format(
        script=shlex.quote(script),
        path=shlex.quote(str(remote_path)),
    )
    cp = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
            str(host),
            remote_cmd,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if cp.returncode != 0:
        stderr = cp.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Command failed ({cp.returncode}): ssh {host} <binary-copy>\n{stderr}"
        )
    dst.write_bytes(cp.stdout)


def _copy_remote_file(
    host: str,
    remote_path: str,
    local_path: Path,
    *,
    allow_fallbacks: bool = True,
) -> None:
    dst = Path(local_path).expanduser().resolve(strict=False)
    if dst.exists() and dst.is_file():
        return
    try:
        _rsync_file(host, remote_path, dst)
    except RuntimeError as rsync_exc:
        if not allow_fallbacks:
            raise RuntimeError(
                f"{rsync_exc}\nManual retry:\n  {_rsync_command_text(host, remote_path, dst)}"
            ) from rsync_exc
        try:
            _scp_file(host, remote_path, dst)
        except RuntimeError as scp_exc:
            try:
                _ssh_copy_file(host, remote_path, dst)
            except RuntimeError as ssh_exc:
                raise RuntimeError(
                    f"{ssh_exc}\nManual retry:\n  {_rsync_command_text(host, remote_path, dst)}"
                ) from scp_exc


def _scenario_pair_tag(scenario1: str, scenario2: str) -> str:
    return _base758._scenario_pair_tag(scenario1, scenario2)


def _combo_key(region: str, pet_method: str) -> str:
    return f"{_base758._safe_region_tag(region)}/{_base758._normalize_pet_method(pet_method)}"


def _publish_root(base_root: Path, *, version_tag: str, scenario_pair_tag: str) -> Path:
    return (
        Path(base_root).expanduser().resolve(strict=False)
        / str(version_tag).strip()
        / str(scenario_pair_tag).strip()
    )


def _manifest_path(base_root: Path, *, version_tag: str, scenario_pair_tag: str, timetag: str) -> Path:
    return _publish_root(base_root, version_tag=version_tag, scenario_pair_tag=scenario_pair_tag) / str(timetag).strip() / "759_publish_manifest.json"


def _combo_dir(
    base_root: Path,
    *,
    version_tag: str,
    scenario_pair_tag: str,
    timetag: str,
    region: str,
    pet_method: str,
) -> Path:
    return (
        _publish_root(base_root, version_tag=version_tag, scenario_pair_tag=scenario_pair_tag)
        / str(timetag).strip()
        / _base758._safe_region_tag(region)
        / _base758._normalize_pet_method(pet_method)
    )


def _canonical_payload_path(combo_dir: Path, *, region: str, pet_method: str, timetag: str) -> Path:
    pet_tag = _base758._normalize_pet_method(pet_method)
    region_token = _base758._output_region_token(region)
    return combo_dir / f"SPEI_UNIFIED_{pet_tag}_ERA5_GCMAGICC_{timetag}_{region_token}.json"


def _panel_a_path(main_payload_path: Path) -> Path:
    return _base758._payload_with_suffix(main_payload_path, "_panelA_map")


def _panel_i_path(main_payload_path: Path) -> Path:
    return _base758._payload_with_suffix(main_payload_path, "_panelI_timeseries")


def _load_manifest_payload(path: Path) -> Optional[Dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve(strict=False)
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_has_combo(payload: Optional[Mapping[str, Any]], combo_key: str) -> bool:
    if not payload:
        return False
    completed = payload.get("completed_combos", [])
    if not isinstance(completed, list):
        return False
    return any(
        isinstance(row, dict) and str(row.get("combo_key", "")).strip() == combo_key
        for row in completed
    )


def _local_bundle_status(
    *,
    base_root: Path,
    version_tag: str,
    scenario_pair_tag: str,
    timetag: str,
    region: str,
    pet_method: str,
) -> Dict[str, Any]:
    combo_dir = _combo_dir(
        base_root,
        version_tag=version_tag,
        scenario_pair_tag=scenario_pair_tag,
        timetag=timetag,
        region=region,
        pet_method=pet_method,
    )
    main_payload = _canonical_payload_path(combo_dir, region=region, pet_method=pet_method, timetag=timetag)
    try:
        payload_path = _base758._resolve_payload_json_path(main_payload)
    except FileNotFoundError:
        payload_path = None
    panel_a_json = _panel_a_path(main_payload)
    panel_i_json = _panel_i_path(main_payload)
    stats_paths = tuple(sorted(combo_dir.glob("SPEI_STATS_*.json")))
    manifest_path = _manifest_path(base_root, version_tag=version_tag, scenario_pair_tag=scenario_pair_tag, timetag=timetag)
    manifest_payload = _load_manifest_payload(manifest_path)
    core_files_ok = payload_path is not None and panel_a_json.exists() and panel_i_json.exists()
    return {
        "combo_dir": combo_dir,
        "manifest_path": manifest_path,
        "payload_path": payload_path,
        "panel_a_json": panel_a_json,
        "panel_i_json": panel_i_json,
        "stats_paths": stats_paths,
        "manifest_has_combo": _manifest_has_combo(manifest_payload, _combo_key(region, pet_method)),
        "core_files_ok": core_files_ok,
    }


def _local_timetags(
    *,
    base_root: Path,
    version_tag: str,
    scenario_pair_tag: str,
    source_timetag: str,
) -> List[str]:
    if str(source_timetag).strip().lower() != "auto":
        return [str(source_timetag).strip()]
    pair_root = _publish_root(base_root, version_tag=version_tag, scenario_pair_tag=scenario_pair_tag)
    if not pair_root.exists():
        return []
    return sorted([path.name for path in pair_root.iterdir() if path.is_dir()], reverse=True)


def _find_local_bundle(
    *,
    base_root: Path,
    version_tag: str,
    scenario_pair_tag: str,
    source_timetag: str,
    region: str,
    pet_method: str,
) -> Optional[SourceBundle]:
    for timetag in _local_timetags(
        base_root=base_root,
        version_tag=version_tag,
        scenario_pair_tag=scenario_pair_tag,
        source_timetag=source_timetag,
    ):
        status = _local_bundle_status(
            base_root=base_root,
            version_tag=version_tag,
            scenario_pair_tag=scenario_pair_tag,
            timetag=timetag,
            region=region,
            pet_method=pet_method,
        )
        if not status["core_files_ok"]:
            continue
        if not (status["manifest_has_combo"] or status["core_files_ok"]):
            continue
        payload_path = status["payload_path"]
        if payload_path is None:
            continue
        return SourceBundle(
            source_type="local",
            version_tag=version_tag,
            scenario_pair_tag=scenario_pair_tag,
            timetag=timetag,
            combo_dir=status["combo_dir"],
            manifest_path=status["manifest_path"],
            payload_path=payload_path,
            panel_a_json=status["panel_a_json"],
            panel_i_json=status["panel_i_json"],
            stats_paths=tuple(status["stats_paths"]),
            origin_manifest_path=str(status["manifest_path"]),
            origin_payload_path=str(payload_path),
        )
    return None


def _remote_timetags(
    *,
    host: str,
    base_root: Path,
    version_tag: str,
    scenario_pair_tag: str,
    source_timetag: str,
) -> List[str]:
    if str(source_timetag).strip().lower() != "auto":
        return [str(source_timetag).strip()]
    pair_root = _publish_root(base_root, version_tag=version_tag, scenario_pair_tag=scenario_pair_tag)
    payload = _ssh_json(host, REMOTE_LIST_TIMETAGS_SCRIPT, pair_root)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected remote timetag response from {host}: {payload!r}")
    return [str(item).strip() for item in payload if str(item).strip()]


def _remote_bundle_status(
    *,
    host: str,
    base_root: Path,
    version_tag: str,
    scenario_pair_tag: str,
    timetag: str,
    region: str,
    pet_method: str,
) -> Dict[str, Any]:
    combo_dir = _combo_dir(
        base_root,
        version_tag=version_tag,
        scenario_pair_tag=scenario_pair_tag,
        timetag=timetag,
        region=region,
        pet_method=pet_method,
    )
    manifest_path = _manifest_path(base_root, version_tag=version_tag, scenario_pair_tag=scenario_pair_tag, timetag=timetag)
    main_payload = _canonical_payload_path(combo_dir, region=region, pet_method=pet_method, timetag=timetag)
    panel_a_json = _panel_a_path(main_payload)
    panel_i_json = _panel_i_path(main_payload)
    payload = _ssh_json(
        host,
        REMOTE_COMBO_STATUS_SCRIPT,
        manifest_path,
        _combo_key(region, pet_method),
        combo_dir,
        main_payload,
        panel_a_json,
        panel_i_json,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected remote combo-status response from {host}: {payload!r}")
    return {
        "combo_dir": str(combo_dir),
        "manifest_path": str(manifest_path),
        "payload_path": str(payload.get("payload_path") or "").strip(),
        "panel_a_json": str(payload.get("panel_a_path") or "").strip(),
        "panel_i_json": str(payload.get("panel_i_path") or "").strip(),
        "stats_paths": tuple(str(item) for item in payload.get("stats_paths", []) if str(item).strip()),
        "manifest_has_combo": bool(payload.get("manifest_has_combo")),
        "core_files_ok": bool(payload.get("core_files_ok")),
    }


def _find_remote_bundle(
    *,
    host: str,
    base_root: Path,
    version_tag: str,
    scenario_pair_tag: str,
    source_timetag: str,
    region: str,
    pet_method: str,
) -> Optional[RemoteBundle]:
    for timetag in _remote_timetags(
        host=host,
        base_root=base_root,
        version_tag=version_tag,
        scenario_pair_tag=scenario_pair_tag,
        source_timetag=source_timetag,
    ):
        status = _remote_bundle_status(
            host=host,
            base_root=base_root,
            version_tag=version_tag,
            scenario_pair_tag=scenario_pair_tag,
            timetag=timetag,
            region=region,
            pet_method=pet_method,
        )
        if not status["core_files_ok"]:
            continue
        if not (status["manifest_has_combo"] or status["core_files_ok"]):
            continue
        payload_path = str(status["payload_path"]).strip()
        if not payload_path:
            continue
        return RemoteBundle(
            version_tag=version_tag,
            scenario_pair_tag=scenario_pair_tag,
            timetag=timetag,
            combo_dir=str(status["combo_dir"]),
            manifest_path=str(status["manifest_path"]),
            payload_path=payload_path,
            panel_a_json=str(status["panel_a_json"]),
            panel_i_json=str(status["panel_i_json"]),
            stats_paths=tuple(status["stats_paths"]),
        )
    return None


def _cache_remote_bundle(
    *,
    host: str,
    cache_root: Path,
    remote_bundle: RemoteBundle,
    region: str,
    pet_method: str,
) -> SourceBundle:
    local_manifest_path = _manifest_path(
        cache_root,
        version_tag=remote_bundle.version_tag,
        scenario_pair_tag=remote_bundle.scenario_pair_tag,
        timetag=remote_bundle.timetag,
    )
    local_combo_dir = _combo_dir(
        cache_root,
        version_tag=remote_bundle.version_tag,
        scenario_pair_tag=remote_bundle.scenario_pair_tag,
        timetag=remote_bundle.timetag,
        region=region,
        pet_method=pet_method,
    )
    local_payload_path = local_combo_dir / Path(remote_bundle.payload_path).name
    local_panel_a_json = local_combo_dir / Path(remote_bundle.panel_a_json).name
    local_panel_i_json = local_combo_dir / Path(remote_bundle.panel_i_json).name
    copied_stats_paths: List[Path] = []

    _copy_remote_file(host, remote_bundle.manifest_path, local_manifest_path)
    _copy_remote_file(host, remote_bundle.payload_path, local_payload_path)
    _copy_remote_file(host, remote_bundle.panel_a_json, local_panel_a_json)
    _copy_remote_file(host, remote_bundle.panel_i_json, local_panel_i_json)
    for remote_stat in remote_bundle.stats_paths:
        local_stat = local_combo_dir / Path(remote_stat).name
        try:
            _copy_remote_file(host, remote_stat, local_stat, allow_fallbacks=False)
        except Exception as exc:
            warnings.warn(
                f"Unable to cache optional stats payload {remote_stat} from {host}: {exc}",
                RuntimeWarning,
            )
            continue
        copied_stats_paths.append(local_stat)

    return SourceBundle(
        source_type="gus",
        version_tag=remote_bundle.version_tag,
        scenario_pair_tag=remote_bundle.scenario_pair_tag,
        timetag=remote_bundle.timetag,
        combo_dir=local_combo_dir,
        manifest_path=local_manifest_path,
        payload_path=local_payload_path,
        panel_a_json=local_panel_a_json,
        panel_i_json=local_panel_i_json,
        stats_paths=tuple(copied_stats_paths),
        origin_manifest_path=remote_bundle.manifest_path,
        origin_payload_path=remote_bundle.payload_path,
    )


def _bundle_with_remote_origin(
    bundle: SourceBundle,
    *,
    remote_publish_root: Path,
    region: str,
    pet_method: str,
) -> SourceBundle:
    remote_manifest_path = _manifest_path(
        remote_publish_root,
        version_tag=bundle.version_tag,
        scenario_pair_tag=bundle.scenario_pair_tag,
        timetag=bundle.timetag,
    )
    remote_combo_dir = _combo_dir(
        remote_publish_root,
        version_tag=bundle.version_tag,
        scenario_pair_tag=bundle.scenario_pair_tag,
        timetag=bundle.timetag,
        region=region,
        pet_method=pet_method,
    )
    return SourceBundle(
        source_type="gus",
        version_tag=bundle.version_tag,
        scenario_pair_tag=bundle.scenario_pair_tag,
        timetag=bundle.timetag,
        combo_dir=bundle.combo_dir,
        manifest_path=bundle.manifest_path,
        payload_path=bundle.payload_path,
        panel_a_json=bundle.panel_a_json,
        panel_i_json=bundle.panel_i_json,
        stats_paths=bundle.stats_paths,
        origin_manifest_path=str(remote_manifest_path),
        origin_payload_path=str(remote_combo_dir / bundle.payload_path.name),
    )


def _resolve_source_bundle(args: argparse.Namespace) -> SourceBundle:
    args.version_tag = _normalize_version_tag(args.version_tag)
    args.region = _normalize_country_iso3(args.region)
    pair_tag = _scenario_pair_tag(args.scenario1, args.scenario2)
    local_publish_root = Path(args.local_publish_root).expanduser().resolve(strict=False)
    local_bundle = _find_local_bundle(
        base_root=local_publish_root,
        version_tag=args.version_tag,
        scenario_pair_tag=pair_tag,
        source_timetag=args.source_timetag,
        region=args.region,
        pet_method=args.pet_method,
    )
    if local_bundle is not None:
        return local_bundle

    cache_root = Path(args.cache_root).expanduser().resolve(strict=False)
    cached_bundle = _find_local_bundle(
        base_root=cache_root,
        version_tag=args.version_tag,
        scenario_pair_tag=pair_tag,
        source_timetag=args.source_timetag,
        region=args.region,
        pet_method=args.pet_method,
    )
    if cached_bundle is not None:
        return _bundle_with_remote_origin(
            cached_bundle,
            remote_publish_root=Path(args.gus_publish_root).expanduser().resolve(strict=False),
            region=args.region,
            pet_method=args.pet_method,
        )

    remote_error: Optional[Exception] = None
    try:
        remote_bundle = _find_remote_bundle(
            host=args.gus_host,
            base_root=Path(args.gus_publish_root).expanduser().resolve(strict=False),
            version_tag=args.version_tag,
            scenario_pair_tag=pair_tag,
            source_timetag=args.source_timetag,
            region=args.region,
            pet_method=args.pet_method,
        )
    except Exception as exc:
        remote_bundle = None
        remote_error = exc

    if remote_bundle is not None:
        return _cache_remote_bundle(
            host=args.gus_host,
            cache_root=Path(args.cache_root).expanduser().resolve(strict=False),
            remote_bundle=remote_bundle,
            region=args.region,
            pet_method=args.pet_method,
        )

    if remote_error is not None:
        raise SystemExit(f"{_missing_source_warning(args)} Remote probe failed: {remote_error}")
    raise SystemExit(_missing_source_warning(args))


def _load_figure_payload(bundle: SourceBundle) -> LoadedFigurePayload:
    payload = _base758._load_payload_json(bundle.payload_path)
    series = payload.get("series", {})
    hist_windows = payload.get("hist_windows", {})
    map_series = [_base758._series_from_dict(item) for item in series.get("map", [])]
    if not map_series:
        raise RuntimeError(f"No map payload found in {bundle.payload_path}")
    return LoadedFigurePayload(
        payload=payload,
        map_series=map_series,
        map_titles=list(payload.get("map_titles", [])),
        era5_series=_base758._series_from_dict(series["era5"]),
        era5drought_keune_series=[_base758._series_from_dict(item) for item in series.get("era5drought_keune", [])],
        all_list=[_base758._series_from_dict(item) for item in series.get("all", [])],
        nat_list=[_base758._series_from_dict(item) for item in series.get("nat", [])],
        cmip6_hist_list=[_base758._series_from_dict(item) for item in series.get("cmip6_hist", [])],
        cmip6_hist_nat_list=[_base758._series_from_dict(item) for item in series.get("cmip6_hist_nat", [])],
        cmip6_ssp245_list=[_base758._series_from_dict(item) for item in series.get("cmip6_ssp245", [])],
        region=str(payload["region"]),
        region_long_name=str(payload.get("region_long_name") or payload["region"]),
        scenario1_tag=str(payload["scenario1"]),
        scenario2_tag=str(payload["scenario2"]),
        scenario1_label=str(payload["scenario1_label"]),
        scenario2_label=str(payload["scenario2_label"]),
        pet_method=str(payload["pet_method"]),
        scale=int(payload["scale"]),
        hist_window_current=tuple(hist_windows.get("current", (2021, 2025))),  # type: ignore[arg-type]
        hist_window_future=tuple(hist_windows.get("future", (2041, 2060))),  # type: ignore[arg-type]
    )


def _active_font_family() -> str:
    return str(_ACTIVE_FONT_FAMILY or "DejaVu Sans")


def _active_font_path() -> str:
    return str(_ACTIVE_FONT_PATH or "")


def _set_style(savefig_dpi: int) -> ResolvedFontFamily:
    global _ACTIVE_FONT_FAMILY, _ACTIVE_FONT_PATH
    resolved = apply_sans_font_rcparams(
        candidates=FONT_FAMILY_CANDIDATES,
        rc_updates={
            "figure.dpi": 150,
            "savefig.dpi": int(savefig_dpi),
            "font.size": BODY_FONT_PT,
            "axes.labelsize": AXIS_LABEL_PT,
            "axes.titlesize": AXIS_LABEL_PT,
            "xtick.labelsize": TICK_FONT_PT,
            "ytick.labelsize": TICK_FONT_PT,
            "legend.fontsize": LEGEND_FONT_PT,
            "axes.linewidth": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.frameon": False,
            "mathtext.default": "regular",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        },
    )
    _ACTIVE_FONT_FAMILY = resolved.family
    _ACTIVE_FONT_PATH = resolved.path
    return resolved


def _warn_if_cartopy_unavailable() -> None:
    if _base758.ccrs is not None and _base758.cfeature is not None:
        return
    warnings.warn(
        "Cartopy is not available in this Python environment, so drought-attribution figure map panel 'a' "
        "will use a simplified fallback instead of the full 758 coastline/ocean styling. "
        "Run the script via 'pixi run python ...' to get the same Cartopy coastline path as 758/759.",
        RuntimeWarning,
    )


def _month_values(series: _base758.SPEISeries, start_year: int, end_year: int) -> np.ndarray:
    vals = _base758._region_mean(series.values)
    mask = (series.years >= start_year) & (series.years <= end_year) & np.isfinite(vals)
    return np.asarray(vals[mask], dtype=float)


def _ensemble_month_values(series_list: Sequence[_base758.SPEISeries], start_year: int, end_year: int) -> np.ndarray:
    arrays = [_month_values(series, start_year, end_year) for series in series_list]
    arrays = [arr for arr in arrays if arr.size > 0]
    if not arrays:
        return np.asarray([], dtype=float)
    return np.concatenate(arrays)


def _build_hist_bins(*arrays: np.ndarray) -> np.ndarray:
    valid = [arr for arr in arrays if arr.size > 0]
    if not valid:
        return np.arange(-3.0, 3.01, 0.2)
    combined = np.concatenate(valid)
    span_min = float(np.nanmin(combined))
    span_max = float(np.nanmax(combined))
    bin_min = np.floor(span_min / 0.2) * 0.2 - 0.2
    bin_max = np.ceil(span_max / 0.2) * 0.2 + 0.2
    return np.arange(bin_min, bin_max + 0.0001, 0.2)


def _latest_era5_point(era5_series: _base758.SPEISeries) -> Dict[str, Optional[float]]:
    era5_vals = _base758._region_mean(era5_series.values)
    mask = np.isfinite(era5_vals)
    if not np.any(mask):
        return {
            "value": None,
            "year": None,
            "month": None,
            "time": None,
            "tag": None,
            "series": era5_vals,
        }
    idx = int(np.where(mask)[0][-1])
    year = int(era5_series.years[idx])
    month = int(era5_series.months[idx])
    return {
        "value": float(era5_vals[idx]),
        "year": float(year),
        "month": float(month),
        "time": float(era5_series.time[idx]),
        "tag": f"{year}-{month:02d}",
        "series": era5_vals,
    }


def _latest_era5_map_series(era5_series: _base758.SPEISeries) -> _base758.SPEISeries:
    if era5_series.values.ndim != 2 or era5_series.values.shape[0] == 0:
        raise RuntimeError("ERA5 series does not contain a spatial time axis for map reconstruction.")
    region_mean = _base758._region_mean(era5_series.values)
    mask = np.isfinite(region_mean)
    if not np.any(mask):
        raise RuntimeError("ERA5 series does not contain a finite timestep for map reconstruction.")
    idx = int(np.where(mask)[0][-1])
    return _base758.SPEISeries(
        label=era5_series.label,
        source=era5_series.source,
        time=np.asarray([float(era5_series.time[idx])], dtype=float),
        years=np.asarray([int(era5_series.years[idx])], dtype=int),
        months=np.asarray([int(era5_series.months[idx])], dtype=int),
        values=np.asarray(era5_series.values[idx : idx + 1], dtype=float),
        lat=None if era5_series.lat is None else np.asarray(era5_series.lat, dtype=float),
        lon=None if era5_series.lon is None else np.asarray(era5_series.lon, dtype=float),
        pet_method=era5_series.pet_method,
        baseline_source=era5_series.baseline_source,
        baseline_pooling=era5_series.baseline_pooling,
        baseline_strategy=era5_series.baseline_strategy,
        baseline_start_year=era5_series.baseline_start_year,
        baseline_end_year=era5_series.baseline_end_year,
        baseline_fit_file=era5_series.baseline_fit_file,
    )


def _add_panel_label(ax: plt.Axes, label: str, *, inside: bool) -> Text:
    x = 0.012 if inside else 0.0
    y = 0.988 if inside else 1.01
    va = "top" if inside else "bottom"
    artist = ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va=va,
        fontsize=PANEL_LABEL_PT,
        fontweight="bold",
        fontstyle="normal",
        clip_on=False,
    )
    artist.set_gid(PANEL_LABEL_GID)
    return artist


def _add_inline_panel_header(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.085,
        0.968,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=INLINE_HEADER_PT,
        fontweight="normal",
    )


def _figure_page_size_mm(fig: plt.Figure) -> Dict[str, float]:
    width_in, height_in = fig.get_size_inches()
    return {
        "width": round(float(width_in) * MM_PER_INCH, 3),
        "height": round(float(height_in) * MM_PER_INCH, 3),
    }


def _figure_text_audit(fig: plt.Figure) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for artist in fig.findobj(match=Text):
        text = str(artist.get_text() or "")
        if not text.strip():
            continue
        kind = "panel_label" if artist.get_gid() == PANEL_LABEL_GID else "text"
        out.append(
            {
                "text": text,
                "kind": kind,
                "fontsize": float(artist.get_fontsize()),
                "fontfamily": artist.get_fontfamily(),
            }
        )
    return out


def _wrap_lon(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def _lon_extent(lon: np.ndarray, pad: float = 2.0) -> Tuple[float, float]:
    lon_wrapped = _wrap_lon(lon)
    if lon_wrapped.size == 0:
        return -180.0, 180.0
    lon_sorted = np.sort(lon_wrapped)
    diffs = np.diff(np.concatenate([lon_sorted, lon_sorted[:1] + 360.0]))
    gap_idx = int(np.argmax(diffs))
    start = lon_sorted[(gap_idx + 1) % lon_sorted.size]
    span = 360.0 - diffs[gap_idx]
    lon_min = start - pad
    lon_max = start + span + pad
    while lon_min > 180.0:
        lon_min -= 360.0
        lon_max -= 360.0
    while lon_max <= -180.0:
        lon_min += 360.0
        lon_max += 360.0
    return float(lon_min), float(lon_max)


def _grid_from_points(lon: np.ndarray, lat: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_wrapped = _wrap_lon(np.asarray(lon, dtype=float))
    lat = np.asarray(lat, dtype=float)
    values = np.asarray(values, dtype=float)
    lon_u = np.unique(lon_wrapped)
    lat_u = np.unique(lat)
    grid = np.full((lat_u.size, lon_u.size), np.nan, dtype=float)
    present = np.zeros((lat_u.size, lon_u.size), dtype=bool)
    lon_to_idx = {value: idx for idx, value in enumerate(lon_u)}
    lat_to_idx = {value: idx for idx, value in enumerate(lat_u)}
    for x_value, y_value, value in zip(lon_wrapped, lat, values):
        yi = lat_to_idx[y_value]
        xi = lon_to_idx[x_value]
        grid[yi, xi] = value
        present[yi, xi] = True

    def _edges(arr: np.ndarray) -> np.ndarray:
        arr = np.sort(arr)
        if arr.size == 1:
            step = 0.5
            return np.array([arr[0] - step, arr[0] + step], dtype=float)
        mid = (arr[1:] + arr[:-1]) / 2.0
        first = arr[0] - (arr[1] - arr[0]) / 2.0
        last = arr[-1] + (arr[-1] - arr[-2]) / 2.0
        return np.concatenate([[first], mid, [last]]).astype(float)

    return _edges(lon_u), _edges(lat_u), grid, present


def _plot_map_panel(
    ax: plt.Axes,
    cax: plt.Axes,
    *,
    series: _base758.SPEISeries,
    label: str,
    start_year: int,
    end_year: int,
    region: str,
    scale: int,
    title_x: float = 0.0,
) -> None:
    years = series.years
    mask = (years >= start_year) & (years <= end_year)
    vals = series.values[mask]
    if vals.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        cax.axis("off")
        return

    mean_map = np.nanmean(vals, axis=0)
    ax.set_title(label, fontsize=TITLE_FONT_PT, loc="left", pad=3.0, x=title_x)
    transform = _base758.ccrs.PlateCarree() if _base758.ccrs is not None and hasattr(ax, "projection") else None
    mappable = None

    if series.lat is not None and series.lon is not None and len(series.lat) == mean_map.size == len(series.lon):
        lon_edges, lat_edges, grid, present = _grid_from_points(series.lon, series.lat, mean_map)
        data_layer = np.ma.array(grid, mask=(~present) | (~np.isfinite(grid)))
        mesh_kwargs = {
            "cmap": _base758.SPEI_CMAP,
            "vmin": -3.0,
            "vmax": 3.0,
            "alpha": 0.7,
            "shading": "auto",
            "edgecolors": "#f5f5f5",
            "linewidth": 0.18,
            "antialiased": False,
        }
        if transform is not None:
            mesh_kwargs["transform"] = transform
        mappable = ax.pcolormesh(lon_edges, lat_edges, data_layer, **mesh_kwargs)

        nan_mask = (~present) | np.isfinite(grid)
        if np.any(~nan_mask):
            nan_layer = np.ma.array(np.ones_like(grid, dtype=float), mask=nan_mask)
            nan_kwargs = {
                "cmap": ListedColormap([_base758.ERA5DROUGHT_NAN_COLOR]),
                "vmin": 0.0,
                "vmax": 1.0,
                "alpha": 0.95,
                "shading": "auto",
                "edgecolors": "none",
                "linewidth": 0.0,
                "antialiased": False,
            }
            if transform is not None:
                nan_kwargs["transform"] = transform
            ax.pcolormesh(lon_edges, lat_edges, nan_layer, **nan_kwargs)

        lon_min, lon_max = _lon_extent(series.lon)
        lat_min = float(lat_edges.min()) - 2.0
        lat_max = float(lat_edges.max()) + 2.0
        ax.set_aspect("equal", adjustable="box")
        if transform is not None:
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=transform)
        else:
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
            ax.set_facecolor("#e8f7ff")
            lon_u = np.unique(_wrap_lon(np.asarray(series.lon, dtype=float)))
            lat_u = np.unique(np.asarray(series.lat, dtype=float))
            try:
                ax.contour(
                    lon_u,
                    lat_u,
                    present.astype(float),
                    levels=[0.5],
                    colors="#4a4a4a",
                    linewidths=0.65,
                    zorder=3.2,
                )
            except Exception:
                pass

        if region and _base758._is_ipcc_ar6_region(region) and _base758.regionmask is not None:
            try:
                ar6 = _base758.regionmask.defined_regions.ar6.all
                rid = ar6.map_keys(region.upper())
                poly = ar6.polygons[rid]
                polys = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
                for geom in polys:
                    if getattr(geom, "is_empty", False):
                        continue
                    xvals, yvals = geom.exterior.xy
                    line_kwargs = {
                        "color": "#8a8a8a",
                        "linewidth": 0.55,
                        "linestyle": "--",
                        "alpha": 0.95,
                        "zorder": 4,
                    }
                    if transform is not None:
                        ax.plot(xvals, yvals, transform=transform, **line_kwargs)
                    else:
                        ax.plot(xvals, yvals, **line_kwargs)
            except Exception:
                pass

        if _base758.cfeature is not None and _base758.ccrs is not None and hasattr(ax, "projection"):
            ax.set_facecolor("#e8f7ff")
            ax.add_feature(_base758.cfeature.LAND, facecolor="white", edgecolor="none", zorder=0.1)
            ax.add_feature(_base758.cfeature.COASTLINE, linewidth=0.5, edgecolor="#333", zorder=1)
            ax.add_feature(_base758.cfeature.BORDERS, linewidth=0.45, edgecolor="#333", zorder=1)
    else:
        mappable = ax.imshow(mean_map.reshape(-1, 1).T, aspect="auto", cmap=_base758.SPEI_CMAP, vmin=-3.0, vmax=3.0)
        ax.set_yticks([])
        ax.set_xticks([])

    ax.grid(alpha=0.08, linewidth=0.3)
    if FuncFormatter is not None:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}°"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}°"))
    ax.tick_params(axis="both", labelsize=TICK_FONT_PT)
    ax.set_xlabel("Lon (°)", fontsize=AXIS_LABEL_PT)
    ax.set_ylabel("Lat (°)", fontsize=AXIS_LABEL_PT)

    map_box = ax.get_position()
    cax_box = cax.get_position()
    cbar_gap = 0.004
    cax.set_position([cax_box.x0, map_box.y0 - cbar_gap - cax_box.height, cax_box.width, cax_box.height])

    color_mappable = mappable if mappable is not None else plt.cm.ScalarMappable(norm=Normalize(-3.0, 3.0), cmap=_base758.SPEI_CMAP)
    cbar = plt.colorbar(color_mappable, cax=cax, orientation="horizontal")
    cbar.set_label(f"SPEI{scale}", fontsize=BODY_FONT_PT, labelpad=1.0)
    cbar.set_ticks([-2.33, -1.65, -1.28, -0.84, 0.0, 0.84, 1.28, 1.65, 2.33])
    cbar.ax.tick_params(labelsize=MIN_TEXT_PT, length=2.0, pad=0.8, labelrotation=35)
    cbar.ax.text(0.0, -1.85, "dry", transform=cbar.ax.transAxes, ha="left", va="top", fontsize=MIN_TEXT_PT)
    cbar.ax.text(1.0, -1.85, "wet", transform=cbar.ax.transAxes, ha="right", va="top", fontsize=MIN_TEXT_PT)


def _add_pair_ylabel(fig: plt.Figure, axes: Sequence[plt.Axes], label: str, x_pad: float) -> None:
    visible_axes = [ax for ax in axes if ax.axison]
    if not visible_axes:
        return
    boxes = [ax.get_position() for ax in visible_axes]
    x0 = min(box.x0 for box in boxes)
    y0 = min(box.y0 for box in boxes)
    y1 = max(box.y1 for box in boxes)
    fig.text(
        x0 - x_pad,
        0.5 * (y0 + y1),
        label,
        rotation="vertical",
        ha="center",
        va="center",
        fontsize=AXIS_LABEL_PT,
    )


def _plot_hist_panel(
    ax: plt.Axes,
    *,
    bins: np.ndarray,
    all_values: np.ndarray,
    nat_values: np.ndarray,
    era5_values: np.ndarray,
    scenario1_label: str,
    scenario2_label: str,
    scale: int,
    era5_label: str,
    latest_era5_value: Optional[float],
    latest_era5_tag: Optional[str],
    show_xlabel: bool,
    show_legend: bool,
) -> None:
    def _legend_label(base_label: str, values: np.ndarray) -> str:
        return f"{base_label} (n={int(values.size)})"

    def _draw_hist(values: np.ndarray, *, label: str, color: str, filled: bool, scale_factor: float = 1.0) -> None:
        if values.size == 0:
            return
        weights = np.full(values.shape, float(scale_factor) / float(values.size), dtype=float)
        if filled:
            ax.hist(
                values,
                bins=bins,
                weights=weights,
                color=color,
                alpha=0.55,
                edgecolor="white",
                linewidth=0.6,
                label=label,
            )
            return
        ax.hist(
            values,
            bins=bins,
            weights=weights,
            histtype="step",
            linestyle="--",
            linewidth=1.0,
            color=color,
            label=label,
        )

    _draw_hist(
        all_values,
        label=_legend_label(scenario1_label, all_values),
        color=_base758.ROW_COLORS["all"],
        filled=True,
    )
    _draw_hist(
        nat_values,
        label=_legend_label(scenario2_label, nat_values),
        color=_base758.ROW_COLORS["nat"],
        filled=True,
    )
    _draw_hist(
        era5_values,
        label=_legend_label(era5_label, era5_values),
        color=_base758.ROW_COLORS["era5"],
        filled=False,
        scale_factor=1.0 / 3.0,
    )

    if latest_era5_value is not None and np.isfinite(latest_era5_value):
        latest_label = f"Latest ERA5 ({latest_era5_tag})" if latest_era5_tag else "Latest ERA5"
        ax.axvline(
            latest_era5_value,
            color="#0f5e5e",
            linestyle="-",
            linewidth=1.0,
            alpha=0.95,
            label=latest_label,
        )

    ax.set_xlim(float(bins[0]), float(bins[-1]))

    if show_xlabel:
        ax.set_xlabel(f"SPEI{scale} index")
    else:
        ax.tick_params(axis="x", labelbottom=False)

    ax.grid(False)

    density_maxima = [
        np.nanmax(np.histogram(values, bins=bins, weights=np.full(values.shape, scale_factor / float(values.size), dtype=float))[0])
        for values, scale_factor in (
            (all_values, 1.0),
            (nat_values, 1.0),
            (era5_values, 1.0 / 3.0),
        )
        if values.size > 0
    ]
    y_max = float(max(density_maxima)) if density_maxima else 1.0
    ax.set_ylim(0.0, y_max * (1.42 if show_legend else 1.12))
    ax.tick_params(axis="y", labelleft=False)

    handles, labels = ax.get_legend_handles_labels()
    if show_legend and handles:
        ax.legend(
            handles,
            labels,
            loc="upper right",
            frameon=False,
            fontsize=LEGEND_FONT_PT,
            ncol=1,
            handlelength=1.8,
            borderaxespad=0.18,
            labelspacing=0.18,
            handletextpad=0.36,
        )


def _scenario_distributions_by_year_month(
    series_list: Sequence[_base758.SPEISeries],
    *,
    start_year: int,
    end_year: int,
) -> Tuple[Dict[Tuple[int, int], List[float]], Dict[Tuple[int, int], float]]:
    out: Dict[Tuple[int, int], List[float]] = {}
    time_lookup: Dict[Tuple[int, int], float] = {}
    for series in series_list:
        vals = _base758._region_mean(series.values)
        mask = (series.years >= start_year) & (series.years <= end_year) & np.isfinite(vals)
        for value, year, month, time_value in zip(vals[mask], series.years[mask], series.months[mask], np.asarray(series.time[mask], dtype=float)):
            key = (int(year), int(month))
            out.setdefault(key, []).append(float(value))
            time_lookup.setdefault(key, float(time_value))
    return out, time_lookup


def _build_risk_line_data(
    distributions: Mapping[Tuple[int, int], Sequence[float]],
    time_lookup: Mapping[Tuple[int, int], float],
    *,
    return_periods: Sequence[int],
    start_year: int,
    end_year: int,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    keys = sorted(
        key for key, values in distributions.items() if start_year <= key[0] <= end_year and len(values) > 0
    )
    x = np.asarray(
        [time_lookup.get(key, float(key[0]) + (float(key[1]) - 0.5) / 12.0) for key in keys],
        dtype=float,
    )
    lines: Dict[int, np.ndarray] = {}
    for return_period in return_periods:
        percentile = 100.0 / float(return_period)
        line_values: List[float] = []
        for key in keys:
            arr = np.asarray(distributions.get(key, []), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                line_values.append(np.nan)
            else:
                line_values.append(float(np.nanpercentile(arr, percentile)))
        lines[int(return_period)] = np.asarray(line_values, dtype=float)
    return x, lines


def _risk_band_specs() -> List[Tuple[float, float, str]]:
    return [
        (-4.00, -2.33, "extremely dry"),
        (-2.33, -1.65, "severely dry"),
        (-1.65, -1.28, "moderately dry"),
        (-1.28, -0.84, "mildly dry"),
        (-0.84, 0.00, "near-normal"),
    ]


def _risk_band_handles() -> List[Patch]:
    norm = Normalize(-3.0, 3.0)
    handles: List[Patch] = []
    for y0, y1, label in _risk_band_specs():
        mid = 0.5 * (y0 + y1)
        handles.append(
            Patch(
                facecolor=_base758.SPEI_CMAP(norm(float(np.clip(mid, -3.0, 3.0)))),
                edgecolor="none",
                alpha=0.33,
                label=label,
            )
        )
    return handles


def _risk_line_styles(return_periods: Sequence[int]) -> Dict[int, Any]:
    return {
        rp: ("-" if idx % 2 == 0 else (0, (1.8, 1.2)))
        for idx, rp in enumerate(return_periods)
    }


def _risk_line_width(return_period: int) -> float:
    widths = {
        1: 0.85,
        2: 0.70,
        10: 0.55,
        20: 0.45,
    }
    return float(widths.get(int(return_period), 0.45))


def _plot_risk_panel(
    ax: plt.Axes,
    *,
    era5_series: _base758.SPEISeries,
    distributions: Mapping[Tuple[int, int], Sequence[float]],
    time_lookup: Mapping[Tuple[int, int], float],
    scenario_label: str,
    scale: int,
    latest_era5: Mapping[str, Optional[float]],
    rp_colors: Mapping[int, str],
    show_xlabel: bool,
    show_legend: bool,
    legend_y: float = 1.0,
) -> None:
    risk_start_year = 1960
    risk_end_year = 2100
    return_periods = (1, 2, 10, 20)
    x, lines = _build_risk_line_data(
        distributions,
        time_lookup,
        return_periods=return_periods,
        start_year=risk_start_year,
        end_year=risk_end_year,
    )
    if x.size == 0:
        ax.text(0.5, 0.5, "No scenario data in 1960-2100", ha="center", va="center")
        ax.set_axis_off()
        return

    norm = Normalize(-3.0, 3.0)
    cutoff_rp = int(max(return_periods))
    cutoff_line = np.clip(lines.get(cutoff_rp, np.asarray([])), -4.0, 0.0)
    cutoff_line = cutoff_line if cutoff_line.size == x.size else None
    for y0, y1, label in _risk_band_specs():
        mid = 0.5 * (y0 + y1)
        band_color = _base758.SPEI_CMAP(norm(float(np.clip(mid, -3.0, 3.0))))
        if cutoff_line is None:
            ax.axhspan(y0, y1, color=band_color, alpha=0.33, zorder=0.05)
        else:
            band_floor = np.maximum(float(y0), cutoff_line)
            band_ceiling = np.full_like(band_floor, float(y1))
            band_mask = np.isfinite(band_floor) & np.isfinite(band_ceiling) & (band_floor < band_ceiling)
            if np.any(band_mask):
                ax.fill_between(
                    x,
                    band_floor,
                    band_ceiling,
                    where=band_mask,
                    color=band_color,
                    alpha=0.33,
                    zorder=0.05,
                    linewidth=0.0,
                )
        if label in {"extremely dry", "severely dry", "moderately dry"}:
            ax.text(
                risk_start_year + 0.8,
                mid,
                label,
                fontsize=MIN_TEXT_PT,
                color="#2f2f2f",
                va="center",
                ha="left",
                alpha=0.72,
                zorder=0.3,
            )

    line_styles = _risk_line_styles(return_periods)
    for return_period in return_periods:
        yvals = np.clip(lines.get(return_period, np.asarray([])), -4.0, 0.0)
        if yvals.size == 0:
            continue
        mask = np.isfinite(yvals)
        if not np.any(mask):
            continue
        ax.plot(
            x[mask],
            yvals[mask],
            linestyle=line_styles[return_period],
            linewidth=_risk_line_width(return_period),
            color=rp_colors[return_period],
            label=f"1 every {return_period} year",
            zorder=1.5,
        )

    era5_region_vals = np.asarray(latest_era5["series"], dtype=float)
    era_mask = (
        (era5_series.years >= risk_start_year)
        & (era5_series.years <= risk_end_year)
        & np.isfinite(era5_region_vals)
    )
    if np.any(era_mask):
        ax.plot(
            np.asarray(era5_series.time[era_mask], dtype=float),
            np.clip(era5_region_vals[era_mask], -4.0, 0.0),
            color="#0f5e5e",
            linewidth=1.9,
            label="ERA5 mean",
            zorder=2.2,
        )

    latest_time = latest_era5.get("time")
    latest_value = latest_era5.get("value")
    latest_year = latest_era5.get("year")
    latest_tag = latest_era5.get("tag")
    if (
        latest_time is not None
        and latest_value is not None
        and latest_year is not None
        and risk_start_year <= int(latest_year) <= risk_end_year
    ):
        ax.scatter(
            [float(latest_time)],
            [float(np.clip(latest_value, -4.0, 0.0))],
            color="#0f5e5e",
            s=36,
            linewidths=0.6,
            edgecolors="white",
            zorder=2.5,
            label=f"Latest ERA5 ({latest_tag})",
        )

    ax.set_xlim(float(risk_start_year), float(risk_end_year))
    ax.set_ylim(0.0, -4.0)
    if show_xlabel:
        ax.set_xlabel("Year")
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda xvalue, _pos: f"{int(round(xvalue))}"))

    for boundary in (-0.84, -1.28, -1.65, -2.33):
        ax.axhline(
            float(boundary),
            color="#cfcfcf",
            linewidth=0.45,
            linestyle=(0, (2.0, 2.0)),
            zorder=0.35,
        )

    if show_legend:
        ax.legend(
            fontsize=LEGEND_FONT_PT,
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(1.0, legend_y),
            ncol=2,
            handlelength=1.8,
            columnspacing=0.55,
            labelspacing=0.12,
            handletextpad=0.28,
            borderaxespad=0.18,
        )


def _plot_keune_overlays(ax: plt.Axes, series_list: Sequence[_base758.SPEISeries]) -> None:
    for series in series_list:
        vals = _base758._region_mean(series.values)
        if vals.size == 0:
            continue
        ax.plot(
            series.time,
            vals,
            linestyle="None",
            marker="o",
            markersize=MANUSCRIPT_KEUNE_MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.28,
            markevery=12,
            zorder=20.0,
        )


def _plot_manuscript_timeseries(
    ax: plt.Axes,
    series: _base758.SPEISeries,
    color: str,
    *,
    mean_color: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    values = series.values
    if values.ndim != 2:
        return
    n_traces = min(values.shape[1], _base758.MAX_GRID_TRACES)
    step = max(1, values.shape[1] // n_traces)
    for col in range(0, values.shape[1], step):
        ax.plot(
            series.time,
            values[:, col],
            color=color,
            alpha=MANUSCRIPT_TRACE_ALPHA,
            linewidth=MANUSCRIPT_TRACE_LW,
            zorder=0.75,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    mc = mean_color or color
    ax.plot(
        series.time,
        np.nanmedian(values, axis=1),
        color=mc,
        alpha=MANUSCRIPT_MEDIAN_ALPHA,
        linewidth=MANUSCRIPT_MEDIAN_LW,
        zorder=0.88,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    ax.plot(
        series.time,
        np.nanmean(values, axis=1),
        color=mc,
        alpha=MANUSCRIPT_MEAN_ALPHA,
        linewidth=MANUSCRIPT_MEAN_LW,
        zorder=0.96,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    ax.axhline(0.0, color="#999", linewidth=MANUSCRIPT_ZERO_LINE_LW, linestyle="--", zorder=0.6)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(round(value))}"))
    ax.tick_params(axis="x", rotation=0, labelsize=TICK_FONT_PT)
    ax.tick_params(axis="y", labelsize=TICK_FONT_PT)


def _panel_legend(
    *,
    cmip6_label: str,
    gcmagicc_color: str,
    include_cmip6: bool,
    include_keune: bool,
) -> List[Line2D]:
    handles = [
        Line2D([0], [0], color=gcmagicc_color, lw=MANUSCRIPT_MEAN_LW, alpha=0.45, label="GCMAGICC"),
        Line2D([0], [0], color=_base758.ROW_MEAN_COLORS["era5"], lw=MANUSCRIPT_ERA5_LW, alpha=0.9, label="ERA5"),
    ]
    if include_cmip6:
        handles.append(
            Line2D([0], [0], color=_base758.CMIP6_PANEL_COLOR, lw=MANUSCRIPT_CMIP6_LW, alpha=0.7, label=cmip6_label)
        )
    if include_keune:
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="o",
                markersize=2.3,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.35,
                alpha=0.95,
                label="Keune et al. 2025",
            )
        )
    return handles


def _plot_timeseries_panel(
    ax: plt.Axes,
    *,
    series_list: Sequence[_base758.SPEISeries],
    cmip6_list: Sequence[_base758.SPEISeries],
    era5_series: _base758.SPEISeries,
    keune_series: Sequence[_base758.SPEISeries],
    scenario_label: str,
    gcmagicc_key: str,
    cmip6_label: str,
    scale: int,
    show_xlabel: bool,
    show_legend: bool,
) -> None:
    xlim_shared = (1850.0, 2101.0)
    if not series_list:
        ax.text(0.5, 0.5, f"No data for {scenario_label}", ha="center", va="center")
        ax.set_axis_off()
        return

    for series in series_list:
        _plot_manuscript_timeseries(
            ax,
            series,
            _base758.ROW_COLORS[gcmagicc_key],
            mean_color=_base758.ROW_MEAN_COLORS[gcmagicc_key],
            xlim=xlim_shared,
        )

    _base758._overlay_cmip6_individual_lines(
        ax,
        list(cmip6_list),
        color=_base758.CMIP6_PANEL_COLOR,
        label="CMIP6",
        linewidth=MANUSCRIPT_CMIP6_LW,
        linestyle="-",
        alpha=MANUSCRIPT_CMIP6_ALPHA,
        zorder=2.2,
        xlim=xlim_shared,
    )
    ax.plot(
        era5_series.time,
        np.nanmedian(era5_series.values, axis=1),
        color=_base758.ROW_MEAN_COLORS["era5"],
        alpha=0.75,
        linewidth=MANUSCRIPT_ERA5_LW,
        zorder=4.0,
    )
    if keune_series:
        _plot_keune_overlays(ax, list(keune_series))

    ax.set_xlim(xlim_shared)
    ax.set_ylim(-5.0, 5.0)
    if show_xlabel:
        ax.set_xlabel("Year")
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", labelsize=TICK_FONT_PT)
    ax.tick_params(axis="x", labelsize=TICK_FONT_PT, pad=1.0)
    ax.grid(alpha=0.10, linewidth=0.35)
    if show_legend:
        ax.legend(
            handles=_panel_legend(
                cmip6_label=cmip6_label,
                gcmagicc_color=_base758.ROW_MEAN_COLORS[gcmagicc_key],
                include_cmip6=bool(cmip6_list),
                include_keune=bool(keune_series),
            ),
            loc="upper right",
            ncol=2,
            frameon=False,
            fontsize=LEGEND_FONT_PT,
            handlelength=1.5,
            columnspacing=0.55,
            labelspacing=0.16,
            borderaxespad=0.12,
        )


def _build_condensed_figure(data: LoadedFigurePayload) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    fig = plt.figure(figsize=(DEFAULT_WIDTH_IN, DEFAULT_HEIGHT_IN))
    outer = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.16, 1.0],
        left=0.058,
        right=0.988,
        bottom=0.082,
        top=0.972,
        hspace=0.16,
    )
    top = outer[0].subgridspec(
        2,
        3,
        width_ratios=[29.0, 5.5, 65.5],
        wspace=0.0,
        hspace=0.04,
    )
    top_left = top[:, 0].subgridspec(2, 1, height_ratios=[1.0, 0.045], hspace=0.02)

    if _base758.ccrs is not None:
        projection = _base758.ccrs.PlateCarree()
        ax_map = fig.add_subplot(top_left[0, 0], projection=projection)
    else:
        ax_map = fig.add_subplot(top_left[0, 0])
    cax_map = fig.add_subplot(top_left[1, 0])
    ax_ts_all = fig.add_subplot(top[0, 2])
    ax_ts_nat = fig.add_subplot(top[1, 2], sharex=ax_ts_all)

    fig.canvas.draw()
    top_map_box = ax_map.get_position()
    top_right_box = ax_ts_all.get_position()
    bottom_box = outer[1].get_position(fig)
    bottom_mid_y = bottom_box.y0 + 0.5 * bottom_box.height

    ref_xlim = (1850.0, 2101.0)
    risk_xlim = (1960.0, 2100.0)
    ref_width = top_right_box.x1 - top_right_box.x0
    risk_x0 = top_right_box.x0 + ((risk_xlim[0] - ref_xlim[0]) / (ref_xlim[1] - ref_xlim[0])) * ref_width
    risk_x1 = top_right_box.x0 + ((risk_xlim[1] - ref_xlim[0]) / (ref_xlim[1] - ref_xlim[0])) * ref_width
    hist_x0 = top_map_box.x0
    target_gap_width = 0.24 * (top_right_box.x1 - top_map_box.x0)
    hist_x1 = max(hist_x0 + 0.22, risk_x0 - target_gap_width)
    hist_width = max(0.01, hist_x1 - hist_x0)
    risk_width = max(0.01, risk_x1 - risk_x0)
    row_height = 0.5 * bottom_box.height

    ax_hist_current = fig.add_axes([hist_x0, bottom_mid_y, hist_width, row_height])
    ax_hist_future = fig.add_axes([hist_x0, bottom_box.y0, hist_width, row_height], sharex=ax_hist_current)
    ax_risk_all = fig.add_axes([risk_x0, bottom_mid_y, risk_width, row_height])
    ax_risk_nat = fig.add_axes([risk_x0, bottom_box.y0, risk_width, row_height], sharex=ax_risk_all)

    panel_axes = {
        "a": ax_map,
        "b": ax_ts_all,
        "c": ax_ts_nat,
        "d": ax_hist_current,
        "e": ax_hist_future,
        "f": ax_risk_all,
        "g": ax_risk_nat,
    }
    for label, ax in panel_axes.items():
        _add_panel_label(ax, label, inside=(label != "a"))

    map_series = _latest_era5_map_series(data.era5_series)
    if map_series.years.size and map_series.months.size:
        map_title = f"ERA5 {int(map_series.years[-1])}-{int(map_series.months[-1]):02d}"
    else:
        map_title = data.map_titles[0] if data.map_titles else f"{data.region_long_name} ERA5 map"
    map_year_min = int(np.nanmin(map_series.years)) if map_series.years.size else 0
    map_year_max = int(np.nanmax(map_series.years)) if map_series.years.size else map_year_min
    _plot_map_panel(
        ax_map,
        cax_map,
        series=map_series,
        label=map_title,
        start_year=map_year_min,
        end_year=map_year_max,
        region=data.region,
        scale=data.scale,
        title_x=0.14,
    )

    cmip6_j = list(data.cmip6_hist_list)
    if _base758._token_mentions_ssp245(data.scenario1_tag):
        cmip6_j.extend(data.cmip6_ssp245_list)
    cmip6_j_label = "CMIP6 historical + ssp245" if data.cmip6_ssp245_list else "CMIP6 historical"
    _plot_timeseries_panel(
        ax_ts_all,
        series_list=data.all_list,
        cmip6_list=cmip6_j,
        era5_series=data.era5_series,
        keune_series=data.era5drought_keune_series,
        scenario_label=data.scenario1_label,
        gcmagicc_key="all",
        cmip6_label=cmip6_j_label,
        scale=data.scale,
        show_xlabel=False,
        show_legend=True,
    )

    cmip6_k = list(data.cmip6_hist_nat_list) if _base758._token_has_nat_suffix(data.scenario2_tag) else []
    _plot_timeseries_panel(
        ax_ts_nat,
        series_list=data.nat_list,
        cmip6_list=cmip6_k,
        era5_series=data.era5_series,
        keune_series=data.era5drought_keune_series,
        scenario_label=data.scenario2_label,
        gcmagicc_key="nat",
        cmip6_label="CMIP6 hist-nat",
        scale=data.scale,
        show_xlabel=True,
        show_legend=False,
    )

    current_values = {
        "era5": _month_values(data.era5_series, *data.hist_window_current),
        "all": _ensemble_month_values(data.all_list, *data.hist_window_current),
        "nat": _ensemble_month_values(data.nat_list, *data.hist_window_current),
    }
    future_values = {
        "era5": current_values["era5"],
        "all": _ensemble_month_values(data.all_list, *data.hist_window_future),
        "nat": _ensemble_month_values(data.nat_list, *data.hist_window_future),
    }
    hist_bins = _build_hist_bins(
        current_values["era5"],
        current_values["all"],
        current_values["nat"],
        future_values["all"],
        future_values["nat"],
    )
    latest_era5 = _latest_era5_point(data.era5_series)
    _plot_hist_panel(
        ax_hist_current,
        bins=hist_bins,
        all_values=current_values["all"],
        nat_values=current_values["nat"],
        era5_values=current_values["era5"],
        scenario1_label=data.scenario1_label,
        scenario2_label=data.scenario2_label,
        scale=data.scale,
        era5_label=f"ERA5 {data.hist_window_current[0]}-{data.hist_window_current[1]}",
        latest_era5_value=latest_era5["value"],
        latest_era5_tag=str(latest_era5["tag"]) if latest_era5["tag"] is not None else None,
        show_xlabel=False,
        show_legend=False,
    )
    _plot_hist_panel(
        ax_hist_future,
        bins=hist_bins,
        all_values=future_values["all"],
        nat_values=future_values["nat"],
        era5_values=future_values["era5"],
        scenario1_label=data.scenario1_label,
        scenario2_label=data.scenario2_label,
        scale=data.scale,
        era5_label=f"ERA5 {data.hist_window_current[0]}-{data.hist_window_current[1]}",
        latest_era5_value=latest_era5["value"],
        latest_era5_tag=str(latest_era5["tag"]) if latest_era5["tag"] is not None else None,
        show_xlabel=True,
        show_legend=False,
    )

    dist_all, time_lookup_all = _scenario_distributions_by_year_month(data.all_list, start_year=1960, end_year=2100)
    dist_nat, time_lookup_nat = _scenario_distributions_by_year_month(data.nat_list, start_year=1960, end_year=2100)
    _plot_risk_panel(
        ax_risk_all,
        era5_series=data.era5_series,
        distributions=dist_all,
        time_lookup=time_lookup_all,
        scenario_label=data.scenario1_label,
        scale=data.scale,
        latest_era5=latest_era5,
        rp_colors={1: "#7A4A1A", 2: "#A0662A", 10: "#C58B4A", 20: "#E2B97A"},
        show_xlabel=False,
        show_legend=False,
    )
    _plot_risk_panel(
        ax_risk_nat,
        era5_series=data.era5_series,
        distributions=dist_nat,
        time_lookup=time_lookup_nat,
        scenario_label=data.scenario2_label,
        scale=data.scale,
        latest_era5=latest_era5,
        rp_colors={1: "#0B3C6D", 2: "#1F5C99", 10: "#4A86C5", 20: "#9EC3E6"},
        show_xlabel=True,
        show_legend=False,
        legend_y=0.82,
    )

    _add_inline_panel_header(ax_ts_all, data.scenario1_label)
    _add_inline_panel_header(ax_ts_nat, data.scenario2_label)
    _add_inline_panel_header(ax_hist_current, f"{data.hist_window_current[0]}-{data.hist_window_current[1]}")
    _add_inline_panel_header(ax_hist_future, f"{data.hist_window_future[0]}-{data.hist_window_future[1]}")
    _add_inline_panel_header(ax_risk_all, data.scenario1_label)
    _add_inline_panel_header(ax_risk_nat, data.scenario2_label)

    for ax in (ax_ts_all, ax_ts_nat, ax_hist_current, ax_hist_future, ax_risk_all, ax_risk_nat):
        ax.set_ylabel("")
    for ax in (ax_hist_current, ax_hist_future):
        ax.tick_params(axis="y", labelleft=False)
    ax_ts_nat.set_xlabel("")
    ax_risk_nat.set_xlabel("")
    ax_ts_nat.tick_params(axis="x", labelsize=5.2, pad=1.0)
    ax_risk_nat.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: "" if np.isclose(value, -4.0) else f"{int(round(value))}")
    )

    gap_x0 = hist_x1 + 0.007
    gap_width = max(0.01, risk_x0 - hist_x1 - 0.012)
    legend_hist_ax = fig.add_axes([gap_x0, bottom_box.y0 + 0.56 * bottom_box.height, gap_width, 0.15 * bottom_box.height])
    legend_risk_ax = fig.add_axes([gap_x0, bottom_box.y0 + 0.08 * bottom_box.height, gap_width, 0.34 * bottom_box.height])
    legend_hist_ax.axis("off")
    legend_risk_ax.axis("off")

    hist_handles, _hist_labels = ax_hist_future.get_legend_handles_labels()
    risk_indent = 0.08
    legend_hist_ax.text(0.0, 1.02, "Legend panels d,e:", ha="left", va="bottom", fontsize=LEGEND_FONT_PT, fontweight="bold")
    legend_risk_ax.text(
        risk_indent,
        1.02,
        "Legend panels f,g:",
        ha="left",
        va="bottom",
        fontsize=LEGEND_FONT_PT,
        fontweight="bold",
    )
    if hist_handles:
        hist_labels = [
            f"{data.scenario1_label} (n={current_values['all'].size}|{future_values['all'].size})",
            f"{data.scenario2_label} (n={current_values['nat'].size}|{future_values['nat'].size})",
            f"ERA5 {data.hist_window_current[0]}-{data.hist_window_current[1]} (n={current_values['era5'].size}|{current_values['era5'].size})",
            f"Latest ERA5 ({latest_era5['tag']})" if latest_era5.get("tag") is not None else "Latest ERA5",
        ][: len(hist_handles)]
        legend_hist_ax.legend(
            hist_handles,
            hist_labels,
            loc="upper left",
            bbox_to_anchor=(0.0, 0.88),
            frameon=False,
            fontsize=LEGEND_FONT_PT,
            handlelength=1.7,
            labelspacing=0.18,
            handletextpad=0.35,
            borderaxespad=0.0,
        )
    risk_line_colors_all = {1: "#7A4A1A", 2: "#A0662A", 10: "#C58B4A", 20: "#E2B97A"}
    risk_line_colors_nat = {1: "#0B3C6D", 2: "#1F5C99", 10: "#4A86C5", 20: "#9EC3E6"}
    risk_return_periods = (1, 2, 10, 20)
    risk_line_styles = _risk_line_styles(risk_return_periods)
    latest_risk_label = f"Latest ERA5 ({latest_era5['tag']})" if latest_era5.get("tag") is not None else "Latest ERA5"
    risk_band_handles = _risk_band_handles()
    paired_handles: List[Tuple[Line2D, Line2D]] = []
    paired_labels: List[str] = []
    for rp in risk_return_periods:
        paired_handles.append(
            (
                Line2D([0], [0], color=risk_line_colors_all[rp], linewidth=_risk_line_width(rp), linestyle=risk_line_styles[rp]),
                Line2D([0], [0], color=risk_line_colors_nat[rp], linewidth=_risk_line_width(rp), linestyle=risk_line_styles[rp]),
            )
        )
        paired_labels.append(f"1 every {rp} year" if rp == 1 else f"1 every {rp} years")
    risk_primary_legend = legend_risk_ax.legend(
        paired_handles
        + [
            Line2D([0], [0], color="#0f5e5e", linewidth=1.8, label="ERA5 mean"),
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="o",
                markersize=4.4,
                markerfacecolor="#0f5e5e",
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=latest_risk_label,
            ),
        ],
        paired_labels + ["ERA5 mean", latest_risk_label],
        loc="upper left",
        bbox_to_anchor=(risk_indent, 0.88),
        frameon=False,
        fontsize=LEGEND_FONT_PT,
        handlelength=2.0,
        labelspacing=0.12,
        handletextpad=0.4,
        borderaxespad=0.0,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
    )
    legend_risk_ax.add_artist(risk_primary_legend)
    legend_risk_ax.legend(
        risk_band_handles,
        [handle.get_label() for handle in risk_band_handles],
        loc="upper left",
        bbox_to_anchor=(risk_indent, 0.16),
        frameon=False,
        fontsize=LEGEND_FONT_PT,
        ncol=1,
        handlelength=1.0,
        labelspacing=0.10,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    fig.canvas.draw()
    _add_pair_ylabel(fig, [ax_ts_all, ax_ts_nat], f"SPEI{data.scale}", x_pad=0.036)
    _add_pair_ylabel(fig, [ax_hist_current, ax_hist_future], "Normalized frequency", x_pad=0.028)
    _add_pair_ylabel(fig, [ax_risk_all, ax_risk_nat], f"SPEI{data.scale} drought severity", x_pad=0.044)

    return fig, panel_axes


def _write_provenance(
    *,
    fig: plt.Figure,
    outdir: Path,
    stem: str,
    bundle: SourceBundle,
) -> Path:
    provenance_path = outdir / f"{stem}_provenance.json"
    saved_page_mm = _figure_page_size_mm(fig)
    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_type": bundle.source_type,
        "version_tag": bundle.version_tag,
        "scenario_pair_tag": bundle.scenario_pair_tag,
        "timetag": bundle.timetag,
        "manifest_path": str(bundle.manifest_path),
        "payload_path": str(bundle.payload_path),
        "panel_a_json": str(bundle.panel_a_json),
        "panel_i_json": str(bundle.panel_i_json),
        "stats_paths": [str(path) for path in bundle.stats_paths],
        "origin_manifest_path": bundle.origin_manifest_path,
        "origin_payload_path": bundle.origin_payload_path,
        "journal_target": DEFAULT_JOURNAL_TARGET,
        "target_width_mm": round(DEFAULT_WIDTH_MM, 3),
        "target_height_mm": round(DEFAULT_HEIGHT_MM, 3),
        "saved_pdf_page_mm": saved_page_mm,
        "resolved_font_family": _active_font_family(),
        "resolved_font_path": _active_font_path(),
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def _append_path_parts_once(base_path: Path, parts: Sequence[str]) -> Path:
    outdir = Path(base_path)
    normalized_parts = tuple(str(part).strip() for part in parts if str(part).strip())
    if not normalized_parts:
        return outdir

    existing_parts = tuple(outdir.parts)
    max_match = min(len(existing_parts), len(normalized_parts))
    for n_match in range(max_match, 0, -1):
        if existing_parts[-n_match:] == normalized_parts[:n_match]:
            for part in normalized_parts[n_match:]:
                outdir = outdir / part
            return outdir

    for part in normalized_parts:
        outdir = outdir / part
    return outdir


def _effective_output_dir(base_outdir: Path, *, version_tag: str, country_iso3: str, timetag: str) -> Path:
    outdir = Path(base_outdir).expanduser().resolve(strict=False)
    return _append_path_parts_once(
        outdir,
        [
            _normalize_version_tag(version_tag),
            _normalize_country_iso3(country_iso3),
            str(timetag).strip(),
        ],
    )


def _current_output_timetag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_outputs(
    *,
    fig: plt.Figure,
    outdir: Path,
    stem: str,
    dpi: int,
    bundle: SourceBundle,
) -> Tuple[Path, Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / f"{stem}.png"
    pdf_path = outdir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=int(dpi))
    fig.savefig(pdf_path, dpi=int(dpi))
    provenance_path = _write_provenance(fig=fig, outdir=outdir, stem=stem, bundle=bundle)
    return png_path, pdf_path, provenance_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build manuscript drought-attribution figure for an example-country drought attribution.")
    p.add_argument(
        "--version-tag",
        "--gcmagicc-version",
        type=_version_tag_arg,
        default=DEFAULT_VERSION_TAG,
        help="GCMagicc version to plot: v100 (GCMagicc) or v101 (GCMagicc-CE).",
    )
    p.add_argument("--scenario1", default=DEFAULT_SCENARIO1)
    p.add_argument("--scenario2", default=DEFAULT_SCENARIO2)
    p.add_argument(
        "--country-iso3",
        "--region",
        dest="region",
        type=_country_iso3_arg,
        default=DEFAULT_COUNTRY_ISO3,
        help="Three-letter country ISO3 code. Default: BRA.",
    )
    p.add_argument("--pet-method", default=DEFAULT_PET_METHOD)
    p.add_argument("--source-timetag", default=DEFAULT_SOURCE_TIMETAG, help="Published source timetag, or 'auto'.")
    p.add_argument("--local-publish-root", type=Path, default=DEFAULT_LOCAL_PUBLISH_ROOT)
    p.add_argument("--gus-host", default=DEFAULT_GUS_HOST)
    p.add_argument("--gus-publish-root", type=Path, default=DEFAULT_GUS_PUBLISH_ROOT)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument(
        "--stem",
        default=DEFAULT_STEM,
        help="Output filename stem. Use 'auto' to include country, version, and GCMagicc label.",
    )
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    args.version_tag = _normalize_version_tag(args.version_tag)
    args.region = _normalize_country_iso3(args.region)
    resolved_font = _set_style(args.dpi)
    _warn_if_cartopy_unavailable()
    bundle = _resolve_source_bundle(args)
    payload = _load_figure_payload(bundle)
    fig, _panel_axes = _build_condensed_figure(payload)
    output_timetag = _current_output_timetag()
    effective_outdir = _effective_output_dir(
        Path(args.outdir).expanduser().resolve(strict=False),
        version_tag=args.version_tag,
        country_iso3=args.region,
        timetag=output_timetag,
    )
    output_stem = _resolve_output_stem(
        str(args.stem),
        country_iso3=args.region,
        version_tag=args.version_tag,
    )
    png_path, pdf_path, provenance_path = _save_outputs(
        fig=fig,
        outdir=effective_outdir,
        stem=output_stem,
        dpi=int(args.dpi),
        bundle=bundle,
    )
    plt.close(fig)
    print(f"Source:           {bundle.source_type}")
    print(f"Country ISO3:     {args.region}")
    print(f"GCMagicc version: {args.version_tag} ({_version_label(args.version_tag)})")
    print(f"Source timetag:   {bundle.timetag}")
    print(f"Output timetag:   {output_timetag}")
    print(f"Journal target:   {DEFAULT_JOURNAL_TARGET}")
    print(f"Figure page mm:   {_figure_page_size_mm(fig)}")
    print(f"Resolved font:    {resolved_font.family}")
    print(f"Payload:          {bundle.payload_path}")
    print(f"PNG:              {png_path}")
    print(f"PDF:              {pdf_path}")
    print(f"Provenance:       {provenance_path}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
