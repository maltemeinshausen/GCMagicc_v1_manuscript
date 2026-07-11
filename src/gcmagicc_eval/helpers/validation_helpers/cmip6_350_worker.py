"""Convert staged ERA5spliced multi-var NetCDF runs into CMIP6-style files."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import sys
from concurrent.futures import CancelledError, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cftime
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scr.validation_helpers.helper_path_utils import (  # noqa: E402
    get_cmip6_vetted_path,
    get_era5spliced_cmip6_localresults_root,
    get_object_bucket,
    get_projects_root,
    get_s3_storage_options,
)

DEFAULT_UPLOAD_PREFIX = "nc/cmip6/era5spliced"
DEFAULT_WORKFLOW = "AR6"
DEFAULT_RUNMODUS = "all"
DEFAULT_N_ENSEMBLE = "n_20"
DEFAULT_KIND = "original"
RUN_COMPLETE_BASENAME = "run_complete.json"
RUN_MANIFEST_BASENAME = "run_manifest.json"
UPLOAD_MANIFEST_BASENAME = "upload_manifest.json"
LOCAL_CLEANUP_BASENAME = "local_cleanup.json"
TIMEVAR_SUBDIR = "CMIP6_timevars"
META_SUBDIR = "_meta"
COMPRESS_OPTS = {"zlib": True, "complevel": 4}
EARTH_RADIUS_M = 6_371_000.0
HEIGHT_COORDS = {
    "tas": ("height2m", 2.0),
    "tasmax": ("height2m", 2.0),
    "tasmin": ("height2m", 2.0),
    "hurs": ("height2m", 2.0),
    "huss": ("height2m", 2.0),
    "sfcWind": ("height10m", 10.0),
}
DEFAULT_UNITS = {
    "tas": "K",
    "tasmax": "K",
    "tasmin": "K",
    "ts": "K",
    "pr": "kg m-2 s-1",
    "psl": "Pa",
    "sfcWind": "m s-1",
    "hurs": "%",
    "huss": "1",
    "rsds": "W m-2",
    "clt": "%",
}
DEFAULT_STANDARD_NAME = {
    "tas": "air_temperature",
    "tasmax": "maximum_air_temperature",
    "tasmin": "minimum_air_temperature",
    "ts": "surface_temperature",
    "pr": "precipitation_flux",
    "psl": "air_pressure_at_sea_level",
    "sfcWind": "wind_speed",
    "hurs": "relative_humidity",
    "huss": "specific_humidity",
    "rsds": "surface_downwelling_shortwave_flux_in_air",
    "clt": "cloud_area_fraction",
}
CMIP6_DEFAULT_TABLE = "Amon"
VAR_TO_TABLE = {
    "clt": "Amon",
    "evspsbl": "Amon",
    "hurs": "Amon",
    "huss": "Amon",
    "pr": "Amon",
    "psl": "Amon",
    "rlut": "Amon",
    "rsds": "Amon",
    "rsdt": "Amon",
    "rsut": "Amon",
    "rtmt": "Amon",
    "sfcWind": "Amon",
    "tas": "Amon",
    "tasmax": "Amon",
    "tasmin": "Amon",
    "ts": "Amon",
    "mrso": "Lmon",
}
REALM_FOR_TABLE = {"Amon": "atmos", "Lmon": "land"}
ACTIVITY_ID_LOOKUP = {"historical": "CMIP", "amip": "CMIP", "abrupt4xCO2": "CMIP"}
EXPERIMENT_LONG_NAME = {
    "historical": "all-forcing simulation of the recent past",
    "amip": "AMIP atmosphere-only run (1979-2014)",
    "abrupt4xCO2": "abrupt 4xCO2",
}


@dataclass
class VariableMeta:
    id: str
    table: str
    activity_id: str
    institution_id: str
    source_id: str
    experiment_id: str
    member_id: str
    grid_label: str
    version: str
    time_range: str
    path: str
    size_mb: float
    format: str = "netcdf"


@dataclass
class DatasetMeta:
    id: str
    label: str
    source_id: str
    experiment_id: str
    member_id: str
    workflow: str
    institution_id: str
    grid_label: str
    activity_id: str
    time_range: str
    variables: List[VariableMeta]
    zarr: Optional[dict[str, Any]] = None


@dataclass
class UploadRecord:
    local_path: str
    remote_key: str
    size_bytes: int
    etag: Optional[str]
    content_type: str


@dataclass
class RunConversionResult:
    source_run_root: Path
    output_root: Path
    completion_marker_path: Path
    run_manifest_path: Path
    upload_manifest_path: Optional[Path]
    local_cleanup_manifest_path: Optional[Path]
    frontend_catalog_path: Optional[Path]
    dataset_count: int
    uploaded_count: int
    uploaded_bytes: int
    cleanup_deleted_file_count: int = 0
    cleanup_deleted_bytes: int = 0
    skipped_existing: bool = False


@dataclass
class ConversionContext:
    output_root: Path
    remote_run_prefix: str
    cmor_version: str
    version: str
    scenario: str
    workflow: str
    runmodus: str
    n_ensemble: str
    kind: str
    run_instance: str


@dataclass
class ConvertedDataset:
    dataset_meta: DatasetMeta
    relative_output_paths: List[str]
    source_file: str


@dataclass
class MemberWorkResult:
    source_index: int
    source_file: str
    dataset_meta: DatasetMeta
    relative_output_paths: List[str]
    uploaded_records: List[UploadRecord]
    uploaded_bytes: int
    cleanup_deleted_file_count: int
    cleanup_deleted_bytes: int


@dataclass
class ResumeProgress:
    source: str
    dataset_metas: List[DatasetMeta]
    uploaded_records: List[UploadRecord]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Unsupported JSON type: {type(obj)!r}")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1024 / 1024, 2)


def _preferred_netcdf_engine() -> str | None:
    if importlib.util.find_spec("netCDF4") is not None:
        return "netcdf4"
    if importlib.util.find_spec("h5netcdf") is not None:
        return "h5netcdf"
    return None


def _encoding_for_engine(encoding: Dict[str, Dict[str, Any]], engine: str | None) -> Dict[str, Dict[str, Any]]:
    if engine in {"netcdf4", "h5netcdf"}:
        return encoding
    cleaned: Dict[str, Dict[str, Any]] = {}
    for key, value in encoding.items():
        if not isinstance(value, dict):
            cleaned[key] = value
            continue
        cleaned[key] = {
            sub_key: sub_value
            for sub_key, sub_value in value.items()
            if sub_key not in {"zlib", "complevel"}
        }
    return cleaned


def write_netcdf_compat(ds: xr.Dataset, path: Path, *, encoding: Dict[str, Dict[str, Any]]) -> None:
    engine = _preferred_netcdf_engine()
    ds.to_netcdf(
        path,
        engine=engine,
        encoding=_encoding_for_engine(encoding, engine),
    )


def _slug(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    token = re.sub(r"-{2,}", "-", token).strip("-._")
    return token or "unknown"


def completion_marker_path(output_root: Path) -> Path:
    return output_root / RUN_COMPLETE_BASENAME


def run_manifest_path(output_root: Path) -> Path:
    return output_root / META_SUBDIR / RUN_MANIFEST_BASENAME


def upload_manifest_path(output_root: Path) -> Path:
    return output_root / META_SUBDIR / UPLOAD_MANIFEST_BASENAME


def local_cleanup_manifest_path(output_root: Path) -> Path:
    return output_root / META_SUBDIR / LOCAL_CLEANUP_BASENAME


def is_run_complete(output_root: Path) -> bool:
    return (
        completion_marker_path(output_root).exists()
        and upload_manifest_path(output_root).exists()
        and local_cleanup_manifest_path(output_root).exists()
    )


def _variable_meta_from_dict(payload: Dict[str, Any]) -> VariableMeta:
    return VariableMeta(
        id=str(payload.get("id") or ""),
        table=str(payload.get("table") or ""),
        activity_id=str(payload.get("activity_id") or ""),
        institution_id=str(payload.get("institution_id") or ""),
        source_id=str(payload.get("source_id") or ""),
        experiment_id=str(payload.get("experiment_id") or ""),
        member_id=str(payload.get("member_id") or ""),
        grid_label=str(payload.get("grid_label") or ""),
        version=str(payload.get("version") or ""),
        time_range=str(payload.get("time_range") or ""),
        path=str(payload.get("path") or ""),
        size_mb=float(payload.get("size_mb") or 0.0),
        format=str(payload.get("format") or "netcdf"),
    )


def _dataset_meta_from_dict(payload: Dict[str, Any]) -> DatasetMeta:
    variables_raw = payload.get("variables") or []
    return DatasetMeta(
        id=str(payload.get("id") or ""),
        label=str(payload.get("label") or ""),
        source_id=str(payload.get("source_id") or ""),
        experiment_id=str(payload.get("experiment_id") or ""),
        member_id=str(payload.get("member_id") or ""),
        workflow=str(payload.get("workflow") or ""),
        institution_id=str(payload.get("institution_id") or ""),
        grid_label=str(payload.get("grid_label") or ""),
        activity_id=str(payload.get("activity_id") or ""),
        time_range=str(payload.get("time_range") or ""),
        variables=[
            _variable_meta_from_dict(item)
            for item in variables_raw
            if isinstance(item, dict)
        ],
        zarr=dict(payload.get("zarr")) if isinstance(payload.get("zarr"), dict) else None,
    )


def _load_dataset_meta_list(manifest_path: Path) -> List[DatasetMeta]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: List[DatasetMeta] = []
    for item in payload.get("datasets") or []:
        if not isinstance(item, dict):
            continue
        # Backward compatibility: older manifests stored only a subset of dataset fields.
        if "variables" not in item:
            continue
        out.append(_dataset_meta_from_dict(item))
    return out


def _upload_record_from_dict(payload: Dict[str, Any]) -> UploadRecord:
    return UploadRecord(
        local_path=str(payload.get("local_path") or ""),
        remote_key=str(payload.get("remote_key") or ""),
        size_bytes=int(payload.get("size_bytes") or 0),
        etag=str(payload.get("etag") or "").strip() or None,
        content_type=str(payload.get("content_type") or "application/octet-stream"),
    )


def _load_upload_records(manifest_path: Path) -> List[UploadRecord]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: List[UploadRecord] = []
    for item in payload.get("files") or []:
        if not isinstance(item, dict):
            continue
        out.append(_upload_record_from_dict(item))
    return out


def _time_range_human_to_token(raw: str) -> str:
    match = re.fullmatch(r"(\d{4})-(\d{2}) to (\d{4})-(\d{2})", str(raw or "").strip())
    if not match:
        return ""
    return f"{match.group(1)}{match.group(2)}-{match.group(3)}{match.group(4)}"


def _dataset_expected_remote_keys(dataset: DatasetMeta, remote_run_prefix: str) -> set[str]:
    keys = {
        str(item.path).strip("/")
        for item in dataset.variables
        if str(item.path).strip("/")
    }
    time_range_token = _time_range_human_to_token(dataset.time_range)
    if time_range_token:
        keys.add(
            "/".join(
                [
                    remote_run_prefix.strip("/"),
                    TIMEVAR_SUBDIR,
                    (
                        f"timevars_Amon_{dataset.source_id}_{dataset.experiment_id}_"
                        f"{dataset.member_id}_{dataset.grid_label}_{time_range_token}.csv"
                    ),
                ]
            )
        )
    if dataset.variables:
        version = str(dataset.variables[0].version or "").strip()
        if version:
            keys.add(
                "/".join(
                    [
                        remote_run_prefix.strip("/"),
                        "CMIP6",
                        dataset.activity_id,
                        dataset.institution_id,
                        dataset.source_id,
                        dataset.experiment_id,
                        dataset.member_id,
                        "fx",
                        "areacella",
                        dataset.grid_label,
                        version,
                        (
                            f"areacella_fx_{dataset.source_id}_{dataset.experiment_id}_"
                            f"{dataset.member_id}_{dataset.grid_label}.nc"
                        ),
                    ]
                )
            )
    return {item for item in keys if item}


def _filter_completed_progress(
    *,
    dataset_metas: Sequence[DatasetMeta],
    uploaded_records: Sequence[UploadRecord],
    remote_run_prefix: str,
) -> tuple[List[DatasetMeta], List[UploadRecord], set[str]]:
    uploaded_keys = {str(item.remote_key).strip("/") for item in uploaded_records if str(item.remote_key).strip("/")}
    kept_datasets: List[DatasetMeta] = []
    completed_keys: set[str] = set()
    completed_members: set[str] = set()
    for dataset in dataset_metas:
        expected = _dataset_expected_remote_keys(dataset, remote_run_prefix)
        if expected and expected.issubset(uploaded_keys):
            kept_datasets.append(dataset)
            completed_keys.update(expected)
            completed_members.add(str(dataset.member_id))
    filtered_records = [
        item
        for item in uploaded_records
        if str(item.remote_key).strip("/") in completed_keys
    ]
    return kept_datasets, filtered_records, completed_members


def _cleanup_local_payload(
    *,
    output_root: Path,
    cleanup_enabled: bool,
) -> tuple[Path, Dict[str, Any]]:
    cleanup_path = local_cleanup_manifest_path(output_root)
    preserved = {
        RUN_COMPLETE_BASENAME,
        f"{META_SUBDIR}/{RUN_MANIFEST_BASENAME}",
        f"{META_SUBDIR}/{UPLOAD_MANIFEST_BASENAME}",
        f"{META_SUBDIR}/{LOCAL_CLEANUP_BASENAME}",
    }
    deleted_files = 0
    deleted_bytes = 0

    if cleanup_enabled:
        for path in sorted(output_root.rglob("*"), key=lambda p: (p.is_dir(), len(p.parts), str(p)), reverse=True):
            rel = path.relative_to(output_root).as_posix()
            if rel in preserved:
                continue
            if path.is_file():
                try:
                    deleted_bytes += int(path.stat().st_size)
                except OSError:
                    pass
                path.unlink(missing_ok=True)
                deleted_files += 1

        for path in sorted(
            [p for p in output_root.rglob("*") if p.is_dir()],
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            if path == output_root or path == output_root / META_SUBDIR:
                continue
            try:
                path.rmdir()
            except OSError:
                continue

    payload = {
        "generated_at_utc": utc_now().isoformat(),
        "output_root": str(output_root),
        "cleanup_enabled": bool(cleanup_enabled),
        "deleted_file_count": int(deleted_files),
        "deleted_bytes": int(deleted_bytes),
        "status": "completed",
    }
    write_json_atomic(cleanup_path, payload)
    return cleanup_path, payload


def _delete_relative_payload_paths(
    *,
    output_root: Path,
    relative_paths: Iterable[str],
) -> tuple[int, int]:
    deleted_files = 0
    deleted_bytes = 0
    for rel_path in sorted(dict.fromkeys(str(item).strip("/") for item in relative_paths if str(item).strip())):
        path = output_root / rel_path
        if not path.is_file():
            continue
        try:
            deleted_bytes += int(path.stat().st_size)
        except OSError:
            pass
        path.unlink(missing_ok=True)
        deleted_files += 1
        parent = path.parent
        while parent != output_root and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    return deleted_files, deleted_bytes


def build_cmip6_local_run_root(
    *,
    base_root: Path,
    version: str,
    scenario: str,
    run_instance: str,
    workflow: str = DEFAULT_WORKFLOW,
    runmodus: str = DEFAULT_RUNMODUS,
    n_ensemble: str = DEFAULT_N_ENSEMBLE,
    kind: str = DEFAULT_KIND,
) -> Path:
    return (
        base_root.expanduser().resolve(strict=False)
        / version
        / scenario
        / workflow
        / runmodus
        / n_ensemble
        / kind
        / run_instance
    )


def build_cmip6_remote_run_prefix(
    *,
    upload_prefix: str,
    version: str,
    scenario: str,
    run_instance: str,
    workflow: str = DEFAULT_WORKFLOW,
    runmodus: str = DEFAULT_RUNMODUS,
    n_ensemble: str = DEFAULT_N_ENSEMBLE,
    kind: str = DEFAULT_KIND,
) -> str:
    return "/".join(
        [
            upload_prefix.strip("/"),
            version,
            scenario,
            workflow,
            runmodus,
            n_ensemble,
            kind,
            run_instance,
        ]
    )


def default_frontend_catalog_path() -> Path:
    return (
        get_projects_root()
        / "gcm_firefly_frontend"
        / "app"
        / "frontend"
        / "public"
        / "data"
        / "gcmagicc_cmip6_data_catalog.json"
    ).resolve(strict=False)


def _creation_date() -> str:
    return utc_now().isoformat(timespec="seconds") + "Z"


def cmor_version_from_run_instance(run_instance: str) -> str:
    match = re.search(r"(\d{8})-(\d{4})$", str(run_instance))
    if match:
        return f"v{match.group(1)}{match.group(2)}"
    return utc_now().strftime("v%Y%m%d%H%M")


def _infer_activity_id(exp: str) -> str:
    return "ScenarioMIP" if str(exp).startswith("ssp") else ACTIVITY_ID_LOOKUP.get(exp, "CMIP")


def _experiment_long_name(exp_id: str) -> str:
    return EXPERIMENT_LONG_NAME.get(exp_id, exp_id)


def parse_cmip_parts(fname: str) -> tuple[str, str, str]:
    stem = Path(fname).stem
    if not stem.startswith("DAT_"):
        raise ValueError("Not a DAT_* file")
    parts = stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"DAT file name missing expected parts: {fname}")
    return parts[2], parts[3], parts[4]


def parse_gcmagicc_parts(fname: str) -> tuple[str, str, str, str]:
    parts = Path(fname).stem.split("_")
    if len(parts) < 4 or not parts[0].startswith("GCMagicc"):
        raise ValueError("Not a GCMagicc_* file")
    gcmagicc_tag, base_source_id, exp, mem = parts[:4]
    composite_source_id = f"{gcmagicc_tag}-{base_source_id}"
    return composite_source_id, base_source_id, exp, mem


def _cmip6_fog_root() -> Path:
    explicit = os.environ.get("GCMAGICC_CMIP6_FOG_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve(strict=False)
    return Path(get_cmip6_vetted_path()).expanduser().resolve(strict=False)


def find_fog_institution(source_id: str) -> str | None:
    base = _cmip6_fog_root() / "historical" / "Amon" / "tas" / source_id
    if not base.exists():
        return None
    for member_dir in sorted(base.iterdir()):
        if not member_dir.is_dir():
            continue
        for grid_dir in sorted(member_dir.iterdir()):
            try:
                tas_file = next(grid_dir.glob("tas_Amon_*.nc"))
            except StopIteration:
                continue
            with xr.open_dataset(tas_file, decode_times=False) as ds:
                value = ds.attrs.get("institution_id")
                if value:
                    return str(value)
    return None


def find_fog_source_type(source_id: str) -> str | None:
    base = _cmip6_fog_root() / "historical" / "Amon" / "tas" / source_id
    if not base.exists():
        return None
    for member_dir in sorted(base.iterdir()):
        if not member_dir.is_dir():
            continue
        for grid_dir in sorted(member_dir.iterdir()):
            try:
                tas_file = next(grid_dir.glob("tas_Amon_*.nc"))
            except StopIteration:
                continue
            with xr.open_dataset(tas_file, decode_times=False) as ds:
                value = ds.attrs.get("source_type")
                if value:
                    return str(value)
    return None


def time_axis_info(ds: xr.Dataset) -> tuple[Sequence[Any], Any, Any, str]:
    units = ds["time"].attrs.get("units", "days since 1850-01-01 00:00:00")
    cal = ds["time"].attrs.get("calendar", "standard")
    dates = cftime.num2date(
        ds["time"].values,
        units,
        calendar=cal,
        only_use_cftime_datetimes=True,
    )
    p0, p1 = dates[0], dates[-1]
    tr = f"{p0.year:04d}{p0.month:02d}-{p1.year:04d}{p1.month:02d}"
    return dates, p0, p1, tr


def humanize_time_range(token: str) -> str:
    match = re.fullmatch(r"(\d{4})(\d{2})-(\d{4})(\d{2})", str(token or ""))
    if not match:
        return str(token or "")
    return f"{match.group(1)}-{match.group(2)} to {match.group(3)}-{match.group(4)}"


def global_attrs(table: str, var: str, meta: dict[str, Any], p0: Any, p1: Any, tr: str, grid: str) -> dict[str, Any]:
    return {
        "Conventions": "CF-1.8 CMIP-6.2",
        "title": f"{meta['source_id']} output for {meta['experiment_id']} ({meta['member_id']}) - {var}",
        "creation_date": _creation_date(),
        "tracking_id": f"hdl:21.14100/{os.urandom(8).hex()}",
        "activity_id": meta["activity_id"],
        "mip_era": "CMIP6",
        "institution_id": meta["institution_id"],
        "source_id": meta["source_id"],
        "experiment_id": meta["experiment_id"],
        "experiment": _experiment_long_name(meta["experiment_id"]),
        "member_id": meta["member_id"],
        "variant_label": meta["member_id"],
        "product": "model_output",
        "source_type": meta.get("source_type", "AOGCM"),
        "sub_experiment": "none",
        "sub_experiment_id": "none",
        "table_id": table,
        "variable_id": var,
        "frequency": "mon",
        "realm": REALM_FOR_TABLE.get(table, "atmos"),
        "standard_name": meta.get("standard_name", ""),
        "grid_label": grid,
        "grid": "gr9: 1x1deg (regridded Jun-2025)",
        "nominal_resolution": "1x1 degree",
        "version": meta["version"],
        "start_time": p0.isoformat(),
        "end_time": p1.isoformat(),
        "time_range": tr,
        "history": "Reformatted by notebook 350 from multi-var files, M. Meinshausen",
    }


def _fill_coord_attrs(coord: xr.DataArray, kind: str) -> None:
    defaults = {
        "lat": {
            "standard_name": "latitude",
            "long_name": "Latitude",
            "units": "degrees_north",
            "axis": "Y",
        },
        "lon": {
            "standard_name": "longitude",
            "long_name": "Longitude",
            "units": "degrees_east",
            "axis": "X",
        },
    }
    if kind not in defaults:
        return
    for key, value in defaults[kind].items():
        cur = coord.attrs.get(key)
        if not cur or str(cur).lower() == "unknown":
            coord.attrs[key] = value


def global_attrs_fx(meta: dict[str, Any], grid: str, grid_desc: str = "regular lat-lon grid") -> dict[str, Any]:
    return {
        "Conventions": "CF-1.8 CMIP-6.2",
        "title": f"{meta['source_id']} grid-cell areas for {meta['experiment_id']} ({meta['member_id']})",
        "creation_date": _creation_date(),
        "tracking_id": f"hdl:21.14100/{os.urandom(8).hex()}",
        "activity_id": meta["activity_id"],
        "mip_era": "CMIP6",
        "institution_id": meta["institution_id"],
        "source_id": meta["source_id"],
        "experiment_id": meta["experiment_id"],
        "experiment": _experiment_long_name(meta["experiment_id"]),
        "member_id": meta["member_id"],
        "variant_label": meta["member_id"],
        "product": "model_output",
        "source_type": meta.get("source_type", "AOGCM"),
        "sub_experiment": "none",
        "sub_experiment_id": "none",
        "table_id": "fx",
        "variable_id": "areacella",
        "frequency": "fx",
        "realm": "atmos",
        "grid_label": grid,
        "grid": grid_desc,
        "nominal_resolution": "1x1 degree",
        "version": meta["version"],
        "history": "Generated by notebook 350 (areacella helper)",
    }


def _coord_bounds(coord: xr.DataArray) -> np.ndarray:
    vals = np.asarray(coord.values, dtype=float)
    if vals.ndim != 1:
        raise ValueError(f"Expect 1-D coordinate for bounds, got {coord.name}")
    if vals.size < 2:
        raise ValueError(f"Need at least two points to build bounds for {coord.name}")
    diffs = np.diff(vals)
    first_edge = vals[0] - diffs[0] / 2
    last_edge = vals[-1] + diffs[-1] / 2
    edges = np.concatenate(([first_edge], vals[:-1] + diffs / 2, [last_edge]))
    return np.stack([edges[:-1], edges[1:]], axis=1)


def _cell_areas(lat_bnds: np.ndarray, lon_bnds: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(lat_bnds)
    lon_rad = np.deg2rad(lon_bnds)
    lat_height = np.abs(np.sin(lat_rad[:, 1]) - np.sin(lat_rad[:, 0]))
    lon_width = np.abs(lon_rad[:, 1] - lon_rad[:, 0])
    return (EARTH_RADIUS_M**2) * lat_height[:, None] * lon_width[None, :]


def _get_bounds(ds: xr.Dataset, coord_name: str) -> np.ndarray:
    bname = f"{coord_name}_bnds"
    if bname in ds:
        existing = ds[bname]
        try:
            return existing.transpose(coord_name, "bnds").values
        except Exception:
            pass
    return _coord_bounds(ds[coord_name])


def _add_height_coord(vds: xr.Dataset, var_name: str) -> None:
    if var_name not in HEIGHT_COORDS:
        return
    coord_name, value = HEIGHT_COORDS[var_name]
    if coord_name in vds:
        return
    vds[coord_name] = xr.DataArray(
        value,
        attrs={
            "units": "m",
            "axis": "Z",
            "positive": "up",
            "long_name": "height",
            "standard_name": "height",
        },
    )
    coords_attr = vds[var_name].attrs.get("coordinates", "").split()
    if coord_name not in coords_attr:
        coords_attr.append(coord_name)
    vds[var_name].attrs["coordinates"] = " ".join(c for c in coords_attr if c)


def _timeseries_unit(vname: str, ds: xr.Dataset) -> str:
    if vname.endswith("_ERF"):
        return "W m-2"
    if vname.endswith(("_smoothed", "_globalmean")):
        base = vname.split("_")[0]
        if base in ds.data_vars:
            return str(ds[base].attrs.get("units", ""))
    return str(ds[vname].attrs.get("units", ""))


def csv_path_for_run(output_root: Path, meta: dict[str, Any], tr: str) -> Path:
    fname = (
        f"timevars_Amon_{meta['source_id']}_{meta['experiment_id']}"
        f"_{meta['member_id']}_{meta['grid_label']}_{tr}.csv"
    )
    return output_root / TIMEVAR_SUBDIR / fname


def write_csv_row(csvfile: Path, header: list[str], row: dict[str, Any]) -> None:
    csvfile.parent.mkdir(parents=True, exist_ok=True)
    new = not csvfile.exists()
    with open(csvfile, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if new:
            writer.writeheader()
        writer.writerow(row)


def _experiment_id_from_name(path: Path) -> str | None:
    parts = path.stem.split("_")
    if len(parts) >= 3 and parts[0].startswith("GCMagicc"):
        return parts[2]
    if path.stem.startswith("DAT_"):
        parsed = path.stem.split("_")
        if len(parsed) >= 4:
            return parsed[3]
    return None


def extract_meta(fp: Path) -> dict[str, Any]:
    with xr.open_dataset(fp, decode_times=False) as ds:
        meta = {
            key: ds.attrs.get(key)
            for key in ("source_id", "experiment_id", "member_id", "institution_id")
        }

    if fp.name.startswith("DAT_"):
        src, exp, mem = parse_cmip_parts(fp.name)
        meta["source_id"] = src
        meta["experiment_id"] = exp
        meta["member_id"] = mem
        meta["institution_id"] = meta.get("institution_id") or "CR"
    elif fp.name.startswith("GCMagicc"):
        composite_src, base_src, exp, mem = parse_gcmagicc_parts(fp.name)
        meta["source_id"] = composite_src
        meta["experiment_id"] = exp
        meta["member_id"] = mem
        meta["institution_id"] = "GCMAGICC"
        meta["source_type"] = find_fog_source_type(base_src) or "AOGCM"
    else:
        if not meta.get("source_id"):
            meta["source_id"] = re.split(r"_", fp.stem)[1]
        meta["experiment_id"] = meta.get("experiment_id") or "historical"
        meta["member_id"] = meta.get("member_id") or "r1i1p1f1"
        src = str(meta["source_id"])
        meta["institution_id"] = find_fog_institution(src) or meta.get("institution_id") or "UNKNOWN"
        meta["source_type"] = find_fog_source_type(src) or "AOGCM"

    meta["activity_id"] = _infer_activity_id(str(meta["experiment_id"]))
    return meta


def maybe_write_areacella(ds: xr.Dataset, meta: dict[str, Any], context: ConversionContext) -> Path | None:
    if "lat" not in ds or "lon" not in ds:
        return None

    lat_bnds = _get_bounds(ds, "lat")
    lon_bnds = _get_bounds(ds, "lon")
    area = _cell_areas(lat_bnds, lon_bnds).astype("float32")

    areacella = xr.DataArray(area, dims=("lat", "lon"), coords={"lat": ds["lat"], "lon": ds["lon"]})
    areacella.attrs.update(
        {
            "standard_name": "cell_area",
            "long_name": "Grid-Cell Area for Atmospheric Grid Variables",
            "units": "m2",
            "cell_methods": "area: sum",
            "comment": "Approximated on a spherical Earth assuming regular lat-lon grid",
            "missing_value": np.float32(1e20),
        }
    )
    areacella.attrs.pop("_FillValue", None)

    lat_bnds_da = xr.DataArray(lat_bnds, dims=("lat", "bnds"), coords={"lat": ds["lat"]})
    lon_bnds_da = xr.DataArray(lon_bnds, dims=("lon", "bnds"), coords={"lon": ds["lon"]})
    _fill_coord_attrs(ds["lat"], "lat")
    _fill_coord_attrs(ds["lon"], "lon")
    _fill_coord_attrs(lat_bnds_da["lat"], "lat")  # type: ignore[index]
    _fill_coord_attrs(lon_bnds_da["lon"], "lon")  # type: ignore[index]
    lat_attrs = dict(ds["lat"].attrs)
    lon_attrs = dict(ds["lon"].attrs)
    lat_attrs["bounds"] = "lat_bnds"
    lon_attrs["bounds"] = "lon_bnds"

    grid_desc = ds.attrs.get("grid", "regular latitude-longitude grid")
    ads = xr.Dataset(
        data_vars={
            "areacella": areacella,
            "lat_bnds": lat_bnds_da,
            "lon_bnds": lon_bnds_da,
        },
        coords={"lat": ds["lat"], "lon": ds["lon"]},
        attrs=global_attrs_fx(meta, meta["grid_label"], str(grid_desc)),
    )
    ads["lat"].attrs = lat_attrs
    ads["lon"].attrs = lon_attrs

    outdir = (
        context.output_root
        / "CMIP6"
        / meta["activity_id"]
        / meta["institution_id"]
        / meta["source_id"]
        / meta["experiment_id"]
        / meta["member_id"]
        / "fx"
        / "areacella"
        / meta["grid_label"]
        / meta["version"]
    )
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / (
        f"areacella_fx_{meta['source_id']}_{meta['experiment_id']}_{meta['member_id']}_{meta['grid_label']}.nc"
    )
    if not path.exists():
        write_netcdf_compat(
            ads,
            path,
            encoding={
                "areacella": {**COMPRESS_OPTS, "_FillValue": np.float32(1e20)},
                "lat_bnds": {"_FillValue": None},
                "lon_bnds": {"_FillValue": None},
            },
        )
    return path


def convert_source_file(nc: Path, context: ConversionContext) -> ConvertedDataset:
    meta = extract_meta(nc)
    meta["version"] = context.cmor_version

    variable_entries: List[VariableMeta] = []
    generated_rel_paths: List[str] = []

    with xr.open_dataset(nc, decode_times=False) as ds:
        dates, p0, p1, tr = time_axis_info(ds)
        time_range_human = humanize_time_range(tr)
        meta["time_start"] = p0.isoformat()
        meta["time_end"] = p1.isoformat()
        meta["grid_label"] = ds.attrs.get("grid_label", "gr9")

        areacella_path = maybe_write_areacella(ds, meta, context)
        if areacella_path is not None:
            generated_rel_paths.append(areacella_path.relative_to(context.output_root).as_posix())

        csvfile = csv_path_for_run(context.output_root, meta, tr)
        base = [
            "institution_id",
            "source_id",
            "experiment_id",
            "member_id",
            "grid_label",
            "time_start",
            "time_end",
            "variable",
            "unit",
            "domain",
            "filepath",
        ]
        header = base + [d.isoformat() for d in dates]

        for var_name in ds.data_vars:
            var = ds[var_name]
            if {"lat", "lon"}.issubset(var.dims):
                table = VAR_TO_TABLE.get(var_name, CMIP6_DEFAULT_TABLE)
                vds = var.to_dataset(name=var_name)
                if "time" in vds:
                    vds["time"].attrs["calendar"] = ds["time"].attrs.get("calendar", "standard")
                if "lat" in vds:
                    _fill_coord_attrs(vds["lat"], "lat")
                    lat_bnds = _get_bounds(ds, "lat")
                    vds["lat_bnds"] = xr.DataArray(lat_bnds, dims=("lat", "bnds"), coords={"lat": vds["lat"]})
                    vds["lat"].attrs["bounds"] = "lat_bnds"
                if "lon" in vds:
                    _fill_coord_attrs(vds["lon"], "lon")
                    lon_bnds = _get_bounds(ds, "lon")
                    vds["lon_bnds"] = xr.DataArray(lon_bnds, dims=("lon", "bnds"), coords={"lon": vds["lon"]})
                    vds["lon"].attrs["bounds"] = "lon_bnds"

                units_attr = vds[var_name].attrs.get("units")
                if not units_attr or str(units_attr).lower() == "unknown":
                    vds[var_name].attrs["units"] = DEFAULT_UNITS.get(var_name, "")
                std_attr = vds[var_name].attrs.get("standard_name")
                if not std_attr:
                    vds[var_name].attrs["standard_name"] = DEFAULT_STANDARD_NAME.get(var_name, "")

                _add_height_coord(vds, var_name)
                vds.attrs = global_attrs(table, var_name, meta, p0, p1, tr, meta["grid_label"])

                outdir = (
                    context.output_root
                    / "CMIP6"
                    / meta["activity_id"]
                    / meta["institution_id"]
                    / meta["source_id"]
                    / meta["experiment_id"]
                    / meta["member_id"]
                    / table
                    / var_name
                    / meta["grid_label"]
                    / meta["version"]
                )
                outdir.mkdir(parents=True, exist_ok=True)
                ncfpath = outdir / (
                    f"{var_name}_{table}_{meta['source_id']}_{meta['experiment_id']}"
                    f"_{meta['member_id']}_{meta['grid_label']}_{tr}.nc"
                )
                write_netcdf_compat(vds, ncfpath, encoding={var_name: COMPRESS_OPTS})

                rel_path = ncfpath.relative_to(context.output_root).as_posix()
                generated_rel_paths.append(rel_path)
                variable_entries.append(
                    VariableMeta(
                        id=var_name,
                        table=table,
                        activity_id=str(meta["activity_id"]),
                        institution_id=str(meta["institution_id"]),
                        source_id=str(meta["source_id"]),
                        experiment_id=str(meta["experiment_id"]),
                        member_id=str(meta["member_id"]),
                        grid_label=str(meta["grid_label"]),
                        version=str(meta["version"]),
                        time_range=time_range_human,
                        path=f"{context.remote_run_prefix}/{rel_path}",
                        size_mb=size_mb(ncfpath),
                    )
                )

                row = {col: np.nan for col in header}
                row.update(
                    {
                        "institution_id": meta["institution_id"],
                        "source_id": meta["source_id"],
                        "experiment_id": meta["experiment_id"],
                        "member_id": meta["member_id"],
                        "grid_label": meta["grid_label"],
                        "time_start": meta["time_start"],
                        "time_end": meta["time_end"],
                        "variable": var_name,
                        "unit": var.attrs.get("units", ""),
                        "domain": "grid",
                        "filepath": rel_path,
                    }
                )
                write_csv_row(csvfile, header, row)
            else:
                row = {
                    "institution_id": meta["institution_id"],
                    "source_id": meta["source_id"],
                    "experiment_id": meta["experiment_id"],
                    "member_id": meta["member_id"],
                    "grid_label": meta["grid_label"],
                    "time_start": meta["time_start"],
                    "time_end": meta["time_end"],
                    "variable": var_name,
                    "unit": _timeseries_unit(var_name, ds),
                    "domain": "timeseries",
                    "filepath": "thisfile",
                }
                for d, val in zip(dates, var.values):
                    row[d.isoformat()] = np.asarray(val).item()
                write_csv_row(csvfile, header, row)

    generated_rel_paths.append(csvfile.relative_to(context.output_root).as_posix())
    dataset_id = "-".join(
        [
            _slug(context.version),
            _slug(context.scenario),
            _slug(context.run_instance),
            _slug(str(meta["member_id"])),
        ]
    )
    dataset_meta = DatasetMeta(
        id=dataset_id,
        label=f"{meta['source_id']} | {meta['experiment_id']} | {context.workflow} | {meta['member_id']}",
        source_id=str(meta["source_id"]),
        experiment_id=str(meta["experiment_id"]),
        member_id=str(meta["member_id"]),
        workflow=context.workflow,
        institution_id=str(meta["institution_id"]),
        grid_label=str(meta["grid_label"]),
        activity_id=str(meta["activity_id"]),
        time_range=time_range_human,
        variables=sorted(variable_entries, key=lambda item: (item.table, item.id)),
        zarr=None,
    )
    return ConvertedDataset(
        dataset_meta=dataset_meta,
        relative_output_paths=sorted(dict.fromkeys(generated_rel_paths)),
        source_file=str(nc),
    )


def _content_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".nc":
        return "application/x-netcdf"
    if suffix == ".csv":
        return "text/csv; charset=utf-8"
    if suffix == ".json":
        return "application/json; charset=utf-8"
    return "application/octet-stream"


def _build_boto3_client():
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("boto3 and botocore are required for CMIP6 upload support") from exc

    opts = dict(get_s3_storage_options())
    if opts.get("anon"):
        raise RuntimeError("Anonymous S3 upload is not supported for CMIP6 publishing.")

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


def _remote_manifest_key(remote_run_prefix: str, basename: str) -> str:
    return f"{remote_run_prefix.strip('/')}/{META_SUBDIR}/{basename}"


def _download_remote_json(client: Any, *, bucket: str, key: str) -> Dict[str, Any] | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except Exception:
        return None
    try:
        body = response["Body"].read()
    except Exception:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def _load_local_resume_progress(output_root: Path, remote_run_prefix: str) -> ResumeProgress | None:
    manifest = run_manifest_path(output_root)
    upload_manifest = upload_manifest_path(output_root)
    marker = completion_marker_path(output_root)
    if marker.exists() or not manifest.exists() or not upload_manifest.exists():
        return None
    dataset_metas = _load_dataset_meta_list(manifest)
    uploaded_records = _load_upload_records(upload_manifest)
    dataset_metas, uploaded_records, _ = _filter_completed_progress(
        dataset_metas=dataset_metas,
        uploaded_records=uploaded_records,
        remote_run_prefix=remote_run_prefix,
    )
    if not dataset_metas:
        return None
    return ResumeProgress(source="local", dataset_metas=dataset_metas, uploaded_records=uploaded_records)


def _load_remote_resume_progress(
    *,
    client: Any,
    bucket: str,
    remote_run_prefix: str,
) -> ResumeProgress | None:
    run_complete_key = f"{remote_run_prefix.strip('/')}/{RUN_COMPLETE_BASENAME}"
    try:
        client.head_object(Bucket=bucket, Key=run_complete_key)
    except Exception:
        pass
    else:
        return None

    run_payload = _download_remote_json(
        client,
        bucket=bucket,
        key=_remote_manifest_key(remote_run_prefix, RUN_MANIFEST_BASENAME),
    )
    upload_payload = _download_remote_json(
        client,
        bucket=bucket,
        key=_remote_manifest_key(remote_run_prefix, UPLOAD_MANIFEST_BASENAME),
    )
    if not isinstance(run_payload, dict) or not isinstance(upload_payload, dict):
        return None
    dataset_metas = [
        _dataset_meta_from_dict(item)
        for item in (run_payload.get("datasets") or [])
        if isinstance(item, dict) and "variables" in item
    ]
    uploaded_records = [
        _upload_record_from_dict(item)
        for item in (upload_payload.get("files") or [])
        if isinstance(item, dict)
    ]
    dataset_metas, uploaded_records, _ = _filter_completed_progress(
        dataset_metas=dataset_metas,
        uploaded_records=uploaded_records,
        remote_run_prefix=remote_run_prefix,
    )
    if not dataset_metas:
        return None
    return ResumeProgress(source="remote", dataset_metas=dataset_metas, uploaded_records=uploaded_records)


def _choose_resume_progress(*candidates: ResumeProgress | None) -> ResumeProgress | None:
    valid = [item for item in candidates if item is not None and item.dataset_metas]
    if not valid:
        return None
    return max(valid, key=lambda item: (len(item.dataset_metas), 1 if item.source == "local" else 0))


def _source_member_id(path: Path) -> str:
    try:
        if path.name.startswith("GCMagicc"):
            return parse_gcmagicc_parts(path.name)[3]
        if path.name.startswith("DAT_"):
            return parse_cmip_parts(path.name)[2]
    except Exception:
        pass
    return str(extract_meta(path).get("member_id") or "")


def upload_output_files(
    *,
    output_root: Path,
    relative_paths: Iterable[str],
    bucket: str,
    remote_run_prefix: str,
    client: Any | None = None,
) -> tuple[list[UploadRecord], int]:
    client = client or _build_boto3_client()
    uploaded: List[UploadRecord] = []
    total_bytes = 0
    for rel_path in sorted(dict.fromkeys(str(item).strip("/") for item in relative_paths if str(item).strip())):
        local_path = output_root / rel_path
        if not local_path.is_file():
            continue
        key = f"{remote_run_prefix}/{rel_path}"
        content_type = _content_type_for_path(local_path)
        client.upload_file(
            str(local_path),
            bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )
        head = client.head_object(Bucket=bucket, Key=key)
        local_size = int(local_path.stat().st_size)
        remote_size = int(head.get("ContentLength") or 0)
        if remote_size != local_size:
            raise RuntimeError(
                f"Uploaded size mismatch for {key}: local={local_size} remote={remote_size}"
            )
        uploaded.append(
            UploadRecord(
                local_path=str(local_path),
                remote_key=key,
                size_bytes=local_size,
                etag=str(head.get("ETag") or "").strip('"') or None,
                content_type=content_type,
            )
        )
        total_bytes += local_size
    return uploaded, total_bytes


def _write_upload_manifest(
    *,
    manifest_path: Path,
    bucket: str,
    remote_run_prefix: str,
    uploaded_records: Sequence[UploadRecord],
) -> None:
    write_json_atomic(
        manifest_path,
        {
            "generated_at_utc": utc_now().isoformat(),
            "bucket": bucket,
            "s3_run_prefix": remote_run_prefix,
            "uploaded_count": len(uploaded_records),
            "uploaded_bytes": int(sum(int(item.size_bytes) for item in uploaded_records)),
            "files": [asdict(item) for item in uploaded_records],
        },
    )


def _upload_progress_metadata(
    *,
    output_root: Path,
    manifest_path: Path,
    upload_manifest_local: Path,
    bucket: str,
    remote_run_prefix: str,
    client: Any | None = None,
) -> None:
    upload_output_files(
        output_root=output_root,
        relative_paths=[
            manifest_path.relative_to(output_root).as_posix(),
            upload_manifest_local.relative_to(output_root).as_posix(),
        ],
        bucket=bucket,
        remote_run_prefix=remote_run_prefix,
        client=client,
    )


def _merge_catalog_datasets(existing: Sequence[dict[str, Any]], incoming: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    def dataset_key(item: dict[str, Any]) -> str:
        dataset_id = str(item.get("id") or "").strip()
        if dataset_id:
            return dataset_id
        return "|".join(
            [
                str(item.get("source_id") or ""),
                str(item.get("experiment_id") or ""),
                str(item.get("member_id") or ""),
                str(item.get("workflow") or ""),
                str(item.get("time_range") or ""),
            ]
        )

    for item in existing:
        key = dataset_key(item)
        if key not in order:
            order.append(key)
        merged[key] = item
    for item in incoming:
        key = dataset_key(item)
        if key not in order:
            order.append(key)
        merged[key] = item
    return [merged[key] for key in order]


def _time_window_from_datasets(datasets: Sequence[dict[str, Any]]) -> dict[str, Any]:
    starts: List[str] = []
    ends: List[str] = []
    for dataset in datasets:
        raw = str(dataset.get("time_range") or "")
        match = re.fullmatch(r"(\d{4}-\d{2}) to (\d{4}-\d{2})", raw)
        if not match:
            continue
        starts.append(match.group(1))
        ends.append(match.group(2))
    if not starts or not ends:
        return {}
    start = min(starts)
    end = max(ends)
    start_year, start_month = start.split("-")
    end_year, end_month = end.split("-")
    length_months = (int(end_year) - int(start_year)) * 12 + (int(end_month) - int(start_month)) + 1
    return {"start": start, "end": end, "length_months": length_months}


def write_frontend_catalog(datasets: Sequence[DatasetMeta], catalog_path: Path) -> Path:
    incoming = [asdict(item) for item in datasets]
    merged_datasets = incoming
    note = (
        "Community-facing CMIP6-style ERA5spliced downloads. "
        "Each variable is published as its own NetCDF file, served from the public GCMagicc data origin (data.gcmagicc.org)."
    )

    if catalog_path.exists():
        try:
            existing_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            existing_datasets = existing_payload.get("datasets") or []
            if isinstance(existing_datasets, list):
                merged_datasets = _merge_catalog_datasets(existing_datasets, incoming)
            existing_note = existing_payload.get("note")
            if isinstance(existing_note, str) and existing_note.strip():
                note = existing_note
        except Exception:
            pass

    payload = {
        "generated_at": utc_now().isoformat(),
        "note": note,
        "time": _time_window_from_datasets(merged_datasets),
        "datasets": merged_datasets,
    }
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(catalog_path, payload)
    return catalog_path


def _write_run_manifest(
    *,
    manifest_path: Path,
    source_run_root: Path,
    output_root: Path,
    remote_run_prefix: str,
    version: str,
    scenario: str,
    workflow: str,
    runmodus: str,
    n_ensemble: str,
    kind: str,
    run_instance: str,
    cmor_version: str,
    source_file_count: int,
    datasets: Sequence[DatasetMeta],
    incremental_index: int | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "generated_at_utc": utc_now().isoformat(),
        "source_run_root": str(source_run_root),
        "output_root": str(output_root),
        "s3_run_prefix": remote_run_prefix,
        "version": version,
        "scenario": scenario,
        "workflow": workflow,
        "runmodus": runmodus,
        "n_ensemble": n_ensemble,
        "kind": kind,
        "run_instance": run_instance,
        "cmor_version": cmor_version,
        "source_file_count": source_file_count,
        "datasets": [asdict(item) for item in datasets],
    }
    if incremental_index is not None:
        payload["incremental_source_index"] = int(incremental_index)
    write_json_atomic(manifest_path, payload)


def _sorted_member_results(results_by_index: Dict[int, MemberWorkResult]) -> List[MemberWorkResult]:
    return [results_by_index[index] for index in sorted(results_by_index)]


def _collect_progress_state(
    results_by_index: Dict[int, MemberWorkResult],
) -> tuple[List[DatasetMeta], List[UploadRecord], int, int]:
    ordered_results = _sorted_member_results(results_by_index)
    dataset_metas = [item.dataset_meta for item in ordered_results]
    uploaded_records: List[UploadRecord] = []
    cleanup_deleted_file_count = 0
    cleanup_deleted_bytes = 0
    for item in ordered_results:
        uploaded_records.extend(item.uploaded_records)
        cleanup_deleted_file_count += int(item.cleanup_deleted_file_count)
        cleanup_deleted_bytes += int(item.cleanup_deleted_bytes)
    return dataset_metas, uploaded_records, cleanup_deleted_file_count, cleanup_deleted_bytes


def _process_member_job(
    *,
    source_file: str,
    source_index: int,
    context: ConversionContext,
    upload_bucket: str | None,
    cleanup_local_after_upload: bool,
) -> MemberWorkResult:
    item = convert_source_file(Path(source_file), context)
    uploaded_records: List[UploadRecord] = []
    uploaded_bytes = 0
    cleanup_deleted_file_count = 0
    cleanup_deleted_bytes = 0
    if upload_bucket:
        uploaded_records, uploaded_bytes = upload_output_files(
            output_root=context.output_root,
            relative_paths=item.relative_output_paths,
            bucket=upload_bucket,
            remote_run_prefix=context.remote_run_prefix,
        )
        if cleanup_local_after_upload:
            cleanup_deleted_file_count, cleanup_deleted_bytes = _delete_relative_payload_paths(
                output_root=context.output_root,
                relative_paths=item.relative_output_paths,
            )
    return MemberWorkResult(
        source_index=source_index,
        source_file=item.source_file,
        dataset_meta=item.dataset_meta,
        relative_output_paths=item.relative_output_paths,
        uploaded_records=uploaded_records,
        uploaded_bytes=uploaded_bytes,
        cleanup_deleted_file_count=cleanup_deleted_file_count,
        cleanup_deleted_bytes=cleanup_deleted_bytes,
    )


def convert_run(
    *,
    source_run_root: Path,
    output_root: Path,
    version: str,
    scenario: str,
    run_instance: str,
    upload_bucket: str | None = None,
    upload_prefix: str = DEFAULT_UPLOAD_PREFIX,
    skip_if_complete: bool = False,
    workflow: str = DEFAULT_WORKFLOW,
    runmodus: str = DEFAULT_RUNMODUS,
    n_ensemble: str = DEFAULT_N_ENSEMBLE,
    kind: str = DEFAULT_KIND,
    frontend_catalog_path: Path | None = None,
    update_frontend_catalog: bool = True,
    cleanup_local_after_upload: bool = True,
    member_workers: int = 1,
) -> RunConversionResult:
    source_run_root = Path(source_run_root).expanduser().resolve(strict=False)
    output_root = Path(output_root).expanduser().resolve(strict=False)
    marker_path = completion_marker_path(output_root)
    manifest_path = run_manifest_path(output_root)
    upload_manifest_local = upload_manifest_path(output_root)
    cleanup_manifest = local_cleanup_manifest_path(output_root)
    remote_run_prefix = build_cmip6_remote_run_prefix(
        upload_prefix=upload_prefix,
        version=version,
        scenario=scenario,
        workflow=workflow,
        runmodus=runmodus,
        n_ensemble=n_ensemble,
        kind=kind,
        run_instance=run_instance,
    )
    cmor_version = cmor_version_from_run_instance(run_instance)

    if not source_run_root.exists():
        raise FileNotFoundError(f"Missing staged source run root: {source_run_root}")
    if source_run_root.name != run_instance:
        raise ValueError(
            f"Run-instance mismatch: source root name is {source_run_root.name!r}, expected {run_instance!r}"
        )

    if skip_if_complete and is_run_complete(output_root):
        uploaded_count = 0
        uploaded_bytes = 0
        deleted_file_count = 0
        deleted_bytes = 0
        if upload_manifest_local.exists():
            payload = json.loads(upload_manifest_local.read_text(encoding="utf-8"))
            uploaded_count = int(payload.get("uploaded_count") or 0)
            uploaded_bytes = int(payload.get("uploaded_bytes") or 0)
        if cleanup_manifest.exists():
            payload = json.loads(cleanup_manifest.read_text(encoding="utf-8"))
            deleted_file_count = int(payload.get("deleted_file_count") or 0)
            deleted_bytes = int(payload.get("deleted_bytes") or 0)
        return RunConversionResult(
            source_run_root=source_run_root,
            output_root=output_root,
            completion_marker_path=marker_path,
            run_manifest_path=manifest_path,
            upload_manifest_path=upload_manifest_local,
            local_cleanup_manifest_path=cleanup_manifest,
            frontend_catalog_path=frontend_catalog_path,
            dataset_count=0,
            uploaded_count=uploaded_count,
            uploaded_bytes=uploaded_bytes,
            cleanup_deleted_file_count=deleted_file_count,
            cleanup_deleted_bytes=deleted_bytes,
            skipped_existing=True,
        )

    upload_client: Any | None = _build_boto3_client() if upload_bucket else None

    context = ConversionContext(
        output_root=output_root,
        remote_run_prefix=remote_run_prefix,
        cmor_version=cmor_version,
        version=version,
        scenario=scenario,
        workflow=workflow,
        runmodus=runmodus,
        n_ensemble=n_ensemble,
        kind=kind,
        run_instance=run_instance,
    )
    source_files: List[Path] = []
    dataset_metas: List[DatasetMeta] = []
    uploaded_count = 0
    uploaded_bytes = 0
    uploaded_records: List[UploadRecord] = []
    progress_deleted_file_count = 0
    progress_deleted_bytes = 0
    member_worker_count = max(1, int(member_workers or 1))
    source_files = sorted(path for path in source_run_root.glob("*.nc") if path.is_file())
    filtered = [path for path in source_files if _experiment_id_from_name(path) == scenario]
    if filtered:
        source_files = filtered
    if not source_files:
        raise RuntimeError(f"No staged source NetCDF files found for scenario {scenario!r} under {source_run_root}")

    local_progress = _load_local_resume_progress(output_root, remote_run_prefix)
    remote_progress = (
        _load_remote_resume_progress(
            client=upload_client,
            bucket=str(upload_bucket),
            remote_run_prefix=remote_run_prefix,
        )
        if upload_bucket and upload_client is not None
        else None
    )
    resume_progress = _choose_resume_progress(local_progress, remote_progress)

    if output_root.exists() and resume_progress is None:
        shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    source_member_by_index = {
        index: _source_member_id(path)
        for index, path in enumerate(source_files, start=1)
    }
    source_index_by_member = {
        member_id: index
        for index, member_id in source_member_by_index.items()
        if member_id
    }
    base_dataset_metas: List[DatasetMeta] = []
    base_uploaded_records: List[UploadRecord] = []
    completed_member_ids: set[str] = set()
    if resume_progress is not None:
        dataset_by_member = {
            str(item.member_id): item
            for item in resume_progress.dataset_metas
            if str(item.member_id)
        }
        base_dataset_metas = [
            dataset_by_member[member_id]
            for member_id in (
                source_member_by_index[index]
                for index in range(1, len(source_files) + 1)
            )
            if member_id in dataset_by_member
        ]
        completed_member_ids = {str(item.member_id) for item in base_dataset_metas if str(item.member_id)}
        completed_remote_keys: set[str] = set()
        for dataset in base_dataset_metas:
            completed_remote_keys.update(_dataset_expected_remote_keys(dataset, remote_run_prefix))
        base_uploaded_records = [
            item
            for item in resume_progress.uploaded_records
            if str(item.remote_key).strip("/") in completed_remote_keys
        ]
        uploaded_records = list(base_uploaded_records)
        dataset_metas = list(base_dataset_metas)
        uploaded_count = len(uploaded_records)
        uploaded_bytes = int(sum(int(item.size_bytes) for item in uploaded_records))
        if cleanup_local_after_upload:
            progress_deleted_file_count = len(uploaded_records)
            progress_deleted_bytes = int(sum(int(item.size_bytes) for item in uploaded_records))
        _write_run_manifest(
            manifest_path=manifest_path,
            source_run_root=source_run_root,
            output_root=output_root,
            remote_run_prefix=remote_run_prefix,
            version=version,
            scenario=scenario,
            workflow=workflow,
            runmodus=runmodus,
            n_ensemble=n_ensemble,
            kind=kind,
            run_instance=run_instance,
            cmor_version=context.cmor_version,
            source_file_count=len(source_files),
            datasets=dataset_metas,
            incremental_index=None,
        )
        if upload_bucket:
            _write_upload_manifest(
                manifest_path=upload_manifest_local,
                bucket=str(upload_bucket),
                remote_run_prefix=remote_run_prefix,
                uploaded_records=uploaded_records,
            )

    remaining_sources = [
        (index, path)
        for index, path in enumerate(source_files, start=1)
        if source_member_by_index.get(index) not in completed_member_ids
    ]

    results_by_index: Dict[int, MemberWorkResult] = {}

    def _record_progress(*, incremental_index: int | None) -> None:
        nonlocal dataset_metas, uploaded_records, uploaded_count, uploaded_bytes
        nonlocal progress_deleted_file_count, progress_deleted_bytes
        ordered_results = _sorted_member_results(results_by_index)
        dataset_metas = list(base_dataset_metas) + [item.dataset_meta for item in ordered_results]
        uploaded_records = list(base_uploaded_records)
        progress_deleted_file_count = len(base_uploaded_records) if cleanup_local_after_upload else 0
        progress_deleted_bytes = int(sum(int(item.size_bytes) for item in base_uploaded_records)) if cleanup_local_after_upload else 0
        for item in ordered_results:
            uploaded_records.extend(item.uploaded_records)
            progress_deleted_file_count += int(item.cleanup_deleted_file_count)
            progress_deleted_bytes += int(item.cleanup_deleted_bytes)
        uploaded_count = len(uploaded_records)
        uploaded_bytes = int(sum(int(item.size_bytes) for item in uploaded_records))
        _write_run_manifest(
            manifest_path=manifest_path,
            source_run_root=source_run_root,
            output_root=output_root,
            remote_run_prefix=remote_run_prefix,
            version=version,
            scenario=scenario,
            workflow=workflow,
            runmodus=runmodus,
            n_ensemble=n_ensemble,
            kind=kind,
            run_instance=run_instance,
            cmor_version=context.cmor_version,
            source_file_count=len(source_files),
            datasets=dataset_metas,
            incremental_index=incremental_index,
        )
        if upload_bucket:
            _write_upload_manifest(
                manifest_path=upload_manifest_local,
                bucket=str(upload_bucket),
                remote_run_prefix=remote_run_prefix,
                uploaded_records=uploaded_records,
            )
            if upload_client is not None:
                _upload_progress_metadata(
                    output_root=output_root,
                    manifest_path=manifest_path,
                    upload_manifest_local=upload_manifest_local,
                    bucket=str(upload_bucket),
                    remote_run_prefix=remote_run_prefix,
                    client=upload_client,
                )

    if remaining_sources:
        if member_worker_count == 1 or len(remaining_sources) == 1:
            for index, path in remaining_sources:
                item = convert_source_file(path, context)
                streamed_records: List[UploadRecord] = []
                streamed_bytes = 0
                deleted_files = 0
                deleted_bytes = 0
                if upload_bucket:
                    streamed_records, streamed_bytes = upload_output_files(
                        output_root=output_root,
                        relative_paths=item.relative_output_paths,
                        bucket=str(upload_bucket),
                        remote_run_prefix=remote_run_prefix,
                        client=upload_client,
                    )
                    if cleanup_local_after_upload:
                        deleted_files, deleted_bytes = _delete_relative_payload_paths(
                            output_root=output_root,
                            relative_paths=item.relative_output_paths,
                        )
                results_by_index[index] = MemberWorkResult(
                    source_index=index,
                    source_file=str(path),
                    dataset_meta=item.dataset_meta,
                    relative_output_paths=item.relative_output_paths,
                    uploaded_records=streamed_records,
                    uploaded_bytes=streamed_bytes,
                    cleanup_deleted_file_count=deleted_files,
                    cleanup_deleted_bytes=deleted_bytes,
                )
                _record_progress(incremental_index=index)
        else:
            futures = {}
            first_failure: tuple[int, Path, Exception] | None = None
            with ProcessPoolExecutor(max_workers=min(member_worker_count, len(remaining_sources))) as executor:
                for index, path in remaining_sources:
                    future = executor.submit(
                        _process_member_job,
                        source_file=str(path),
                        source_index=index,
                        context=context,
                        upload_bucket=upload_bucket,
                        cleanup_local_after_upload=cleanup_local_after_upload,
                    )
                    futures[future] = (index, path)

                for future in as_completed(list(futures.keys())):
                    index, path = futures[future]
                    try:
                        member_result = future.result()
                    except CancelledError:
                        continue
                    except Exception as exc:
                        if first_failure is None:
                            first_failure = (index, path, exc)
                            for other in futures:
                                if other is not future:
                                    other.cancel()
                        continue
                    results_by_index[index] = member_result
                    _record_progress(incremental_index=None)

            if first_failure is not None:
                failed_index, failed_path, failed_exc = first_failure
                raise RuntimeError(
                    f"CMIP6 member conversion failed for {failed_path.name} (index {failed_index}): {failed_exc}"
                ) from failed_exc

    upload_manifest_path_local: Path | None = None

    if upload_bucket:
        upload_manifest_path_local = upload_manifest_local
        _write_upload_manifest(
            manifest_path=upload_manifest_path_local,
            bucket=str(upload_bucket),
            remote_run_prefix=remote_run_prefix,
            uploaded_records=uploaded_records,
        )

    resolved_catalog_path: Path | None = None
    if update_frontend_catalog:
        resolved_catalog_path = (
            Path(frontend_catalog_path).expanduser().resolve(strict=False)
            if frontend_catalog_path is not None
            else default_frontend_catalog_path()
        )
        if dataset_metas:
            write_frontend_catalog(dataset_metas, resolved_catalog_path)

    cleanup_manifest_local: Path | None = None
    cleanup_payload: Dict[str, Any] | None = None
    if upload_bucket:
        if cleanup_manifest.exists():
            cleanup_manifest_local = cleanup_manifest
            cleanup_payload = json.loads(cleanup_manifest_local.read_text(encoding="utf-8"))
        else:
            cleanup_manifest_local, cleanup_payload = _cleanup_local_payload(
                output_root=output_root,
                cleanup_enabled=bool(cleanup_local_after_upload),
            )
            if cleanup_payload is not None:
                cleanup_payload["deleted_file_count"] = int(cleanup_payload.get("deleted_file_count") or 0) + progress_deleted_file_count
                cleanup_payload["deleted_bytes"] = int(cleanup_payload.get("deleted_bytes") or 0) + progress_deleted_bytes
                write_json_atomic(cleanup_manifest_local, cleanup_payload)

    write_json_atomic(
        marker_path,
        {
            "completed_at_utc": utc_now().isoformat(),
            "source_run_root": str(source_run_root),
            "output_root": str(output_root),
            "version": version,
            "scenario": scenario,
            "workflow": workflow,
            "runmodus": runmodus,
            "n_ensemble": n_ensemble,
            "kind": kind,
            "run_instance": run_instance,
            "cmor_version": context.cmor_version,
            "dataset_count": len(dataset_metas),
            "uploaded_count": uploaded_count,
            "uploaded_bytes": uploaded_bytes,
            "upload_manifest_path": str(upload_manifest_path_local) if upload_manifest_path_local else None,
            "local_cleanup_manifest_path": str(cleanup_manifest_local) if cleanup_manifest_local else None,
            "local_cleanup_deleted_file_count": (
                int(cleanup_payload.get("deleted_file_count") or 0) if cleanup_payload else 0
            ),
            "local_cleanup_deleted_bytes": (
                int(cleanup_payload.get("deleted_bytes") or 0) if cleanup_payload else 0
            ),
            "local_cleanup_completed_at_utc": (
                cleanup_payload.get("generated_at_utc") if cleanup_payload else None
            ),
            "frontend_catalog_path": str(resolved_catalog_path) if resolved_catalog_path else None,
        },
    )

    if upload_bucket:
        metadata_relative_paths = [
            manifest_path.relative_to(output_root).as_posix(),
            upload_manifest_local.relative_to(output_root).as_posix() if upload_manifest_path_local else "",
            cleanup_manifest_local.relative_to(output_root).as_posix() if cleanup_manifest_local else "",
            marker_path.relative_to(output_root).as_posix(),
        ]
        upload_output_files(
            output_root=output_root,
            relative_paths=[item for item in metadata_relative_paths if item],
            bucket=str(upload_bucket),
            remote_run_prefix=remote_run_prefix,
            client=upload_client,
        )

    return RunConversionResult(
        source_run_root=source_run_root,
        output_root=output_root,
        completion_marker_path=marker_path,
        run_manifest_path=manifest_path,
        upload_manifest_path=upload_manifest_path_local,
        local_cleanup_manifest_path=cleanup_manifest_local,
        frontend_catalog_path=resolved_catalog_path,
        dataset_count=len(dataset_metas),
        uploaded_count=uploaded_count,
        uploaded_bytes=uploaded_bytes,
        cleanup_deleted_file_count=int(cleanup_payload.get("deleted_file_count") or 0) if cleanup_payload else 0,
        cleanup_deleted_bytes=int(cleanup_payload.get("deleted_bytes") or 0) if cleanup_payload else 0,
        skipped_existing=False,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert one staged ERA5spliced n_20/original run into CMIP6-style NetCDF files."
    )
    parser.add_argument("--source-run-root", type=Path, required=True, help="Staged run root under ERA5spliced_localstaging.")
    parser.add_argument("--output-root", type=Path, default=None, help="Final local CMIP6 run root. Defaults under ERA5spliced_cmip6_localresults.")
    parser.add_argument("--version", required=True, help="Canonical version (e.g. v100 or v101).")
    parser.add_argument("--scenario", required=True, help="Canonical scenario identifier.")
    parser.add_argument("--run-instance", required=True, help="Run directory name (e.g. run_debiasloop_... ).")
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW, help="Workflow token (default: AR6).")
    parser.add_argument("--runmodus", default=DEFAULT_RUNMODUS, help="Runmodus token (default: all).")
    parser.add_argument("--n-ensemble", default=DEFAULT_N_ENSEMBLE, help="Ensemble token (default: n_20).")
    parser.add_argument("--kind", default=DEFAULT_KIND, help="Canonical kind (default: original).")
    parser.add_argument("--upload-bucket", default=get_object_bucket(), help="Target OVH bucket for converted outputs.")
    parser.add_argument("--upload-prefix", default=DEFAULT_UPLOAD_PREFIX, help="Target OVH prefix root.")
    parser.add_argument(
        "--skip-if-complete",
        action="store_true",
        help="Skip when run_complete.json, upload_manifest.json, and local_cleanup.json already exist.",
    )
    parser.add_argument("--frontend-catalog", type=Path, default=default_frontend_catalog_path(), help="Frontend CMIP6 catalog JSON to update incrementally.")
    parser.add_argument("--no-update-frontend-catalog", action="store_true", help="Do not touch the frontend CMIP6 catalog.")
    parser.add_argument("--no-upload", action="store_true", help="Convert locally only; skip OVH upload and verification.")
    parser.add_argument(
        "--member-workers",
        type=int,
        default=1,
        help="Parallel member workers. Default: 1 for standalone 350 runs.",
    )
    parser.add_argument(
        "--cleanup-local-after-upload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After verified upload, remove local converted payload files and keep only run metadata.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_root = (
        args.output_root.expanduser().resolve(strict=False)
        if args.output_root is not None
        else build_cmip6_local_run_root(
            base_root=get_era5spliced_cmip6_localresults_root(),
            version=str(args.version),
            scenario=str(args.scenario),
            workflow=str(args.workflow),
            runmodus=str(args.runmodus),
            n_ensemble=str(args.n_ensemble),
            kind=str(args.kind),
            run_instance=str(args.run_instance),
        )
    )

    result = convert_run(
        source_run_root=args.source_run_root,
        output_root=output_root,
        version=str(args.version),
        scenario=str(args.scenario),
        run_instance=str(args.run_instance),
        upload_bucket=None if args.no_upload else str(args.upload_bucket or "").strip(),
        upload_prefix=str(args.upload_prefix),
        skip_if_complete=bool(args.skip_if_complete),
        workflow=str(args.workflow),
        runmodus=str(args.runmodus),
        n_ensemble=str(args.n_ensemble),
        kind=str(args.kind),
        frontend_catalog_path=args.frontend_catalog,
        update_frontend_catalog=not bool(args.no_update_frontend_catalog),
        cleanup_local_after_upload=bool(args.cleanup_local_after_upload),
        member_workers=max(1, int(args.member_workers or 1)),
    )

    print(
        json.dumps(
            {
                "source_run_root": str(result.source_run_root),
                "output_root": str(result.output_root),
                "completion_marker_path": str(result.completion_marker_path),
                "run_manifest_path": str(result.run_manifest_path),
                "upload_manifest_path": str(result.upload_manifest_path) if result.upload_manifest_path else None,
                "local_cleanup_manifest_path": (
                    str(result.local_cleanup_manifest_path) if result.local_cleanup_manifest_path else None
                ),
                "frontend_catalog_path": str(result.frontend_catalog_path) if result.frontend_catalog_path else None,
                "dataset_count": result.dataset_count,
                "uploaded_count": result.uploaded_count,
                "uploaded_bytes": result.uploaded_bytes,
                "cleanup_deleted_file_count": result.cleanup_deleted_file_count,
                "cleanup_deleted_bytes": result.cleanup_deleted_bytes,
                "skipped_existing": result.skipped_existing,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
