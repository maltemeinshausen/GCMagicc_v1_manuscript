"""Convert staged ERA5spliced multi-var NetCDF runs into CMIP7-style files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from concurrent.futures import CancelledError, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scr.validation_helpers import cmip6_350_worker as cmip6  # noqa: E402
from scr.validation_helpers.helper_path_utils import (  # noqa: E402
    get_era5spliced_cmip7_localresults_root,
    get_object_bucket,
    get_projects_root,
)

DEFAULT_UPLOAD_PREFIX = "nc/cmip7/era5spliced"
DEFAULT_WORKFLOW = cmip6.DEFAULT_WORKFLOW
DEFAULT_RUNMODUS = cmip6.DEFAULT_RUNMODUS
DEFAULT_N_ENSEMBLE = cmip6.DEFAULT_N_ENSEMBLE
DEFAULT_KIND = cmip6.DEFAULT_KIND
DEFAULT_REGION = "glb"
CMIP7_DRS_SPECS = "MIP-DRS7"
CMIP7_DATA_SPECS_VERSION = "MIP-DS7.1.0.0"
CMIP7_CONVENTIONS = "CF-1.12"
CMIP7_MIP_ERA = "CMIP7"
CMIP7_DEFAULT_PRODUCT = "model-output"
CMIP7_DEFAULT_LICENSE_ID = "cc-by-4-0"
RUN_COMPLETE_BASENAME = cmip6.RUN_COMPLETE_BASENAME
RUN_MANIFEST_BASENAME = cmip6.RUN_MANIFEST_BASENAME
UPLOAD_MANIFEST_BASENAME = cmip6.UPLOAD_MANIFEST_BASENAME
LOCAL_CLEANUP_BASENAME = cmip6.LOCAL_CLEANUP_BASENAME
TIMEVAR_SUBDIR = "CMIP7_timevars"
META_SUBDIR = cmip6.META_SUBDIR
COMPRESS_OPTS = cmip6.COMPRESS_OPTS

write_json_atomic = cmip6.write_json_atomic
utc_now = cmip6.utc_now
size_mb = cmip6.size_mb
write_netcdf_compat = cmip6.write_netcdf_compat
completion_marker_path = cmip6.completion_marker_path
run_manifest_path = cmip6.run_manifest_path
upload_manifest_path = cmip6.upload_manifest_path
local_cleanup_manifest_path = cmip6.local_cleanup_manifest_path
UploadRecord = cmip6.UploadRecord
RunConversionResult = cmip6.RunConversionResult
upload_output_files = cmip6.upload_output_files


@dataclass(frozen=True)
class BrandedVariableMeta:
    variable_id: str
    branding_suffix: str
    temporal_label: str
    vertical_label: str
    horizontal_label: str
    area_label: str
    realm: str
    frequency: str = "mon"
    cell_methods: str = "time: mean"

    @property
    def branded_variable(self) -> str:
        return f"{self.variable_id}_{self.branding_suffix}"


CMIP7_VARIABLES: Dict[str, BrandedVariableMeta] = {
    "areacella": BrandedVariableMeta("areacella", "ti-u-hxy-u", "ti", "u", "hxy", "u", "atmos", "fx", "area: sum"),
    "clt": BrandedVariableMeta("clt", "tavg-u-hxy-u", "tavg", "u", "hxy", "u", "atmos"),
    "evspsbl": BrandedVariableMeta("evspsbl", "tavg-u-hxy-u", "tavg", "u", "hxy", "u", "atmos"),
    "hurs": BrandedVariableMeta("hurs", "tavg-h2m-hxy-u", "tavg", "h2m", "hxy", "u", "atmos"),
    "huss": BrandedVariableMeta("huss", "tavg-h2m-hxy-u", "tavg", "h2m", "hxy", "u", "atmos"),
    "mrso": BrandedVariableMeta("mrso", "tavg-u-hxy-lnd", "tavg", "u", "hxy", "lnd", "land"),
    "pr": BrandedVariableMeta("pr", "tavg-u-hxy-u", "tavg", "u", "hxy", "u", "atmos"),
    "psl": BrandedVariableMeta("psl", "tavg-u-hxy-u", "tavg", "u", "hxy", "u", "atmos"),
    "rlut": BrandedVariableMeta("rlut", "tavg-u-hxy-u", "tavg", "u", "hxy", "u", "atmos"),
    "rsds": BrandedVariableMeta("rsds", "tavg-u-hxy-u", "tavg", "u", "hxy", "u", "atmos"),
    "rsdt": BrandedVariableMeta("rsdt", "tavg-u-hxy-u", "tavg", "u", "hxy", "u", "atmos"),
    "rsut": BrandedVariableMeta("rsut", "tavg-u-hxy-u", "tavg", "u", "hxy", "u", "atmos"),
    "rtmt": BrandedVariableMeta("rtmt", "tavg-u-hxy-u", "tavg", "u", "hxy", "u", "atmos"),
    "sfcWind": BrandedVariableMeta("sfcWind", "tavg-h10m-hxy-u", "tavg", "h10m", "hxy", "u", "atmos"),
    "tas": BrandedVariableMeta("tas", "tavg-h2m-hxy-u", "tavg", "h2m", "hxy", "u", "atmos"),
    "tasmax": BrandedVariableMeta("tasmax", "tmax-h2m-hxy-u", "tmax", "h2m", "hxy", "u", "atmos", "mon", "time: max"),
    "tasmin": BrandedVariableMeta("tasmin", "tmin-h2m-hxy-u", "tmin", "h2m", "hxy", "u", "atmos", "mon", "time: min"),
    "ts": BrandedVariableMeta("ts", "tavg-u-hxy-u", "tavg", "u", "hxy", "u", "atmos"),
}


@dataclass
class VariableMeta:
    id: str
    branded_variable: str
    branding_suffix: str
    frequency: str
    region: str
    activity_id: str
    institution_id: str
    source_id: str
    experiment_id: str
    member_id: str
    variant_label: str
    grid_label: str
    directory_date: str
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
    variant_label: str
    workflow: str
    institution_id: str
    grid_label: str
    activity_id: str
    region: str
    directory_date: str
    time_range: str
    variables: List[VariableMeta]
    relative_output_paths: List[str]
    zarr: Optional[dict[str, Any]] = None


@dataclass
class ConversionContext:
    output_root: Path
    remote_run_prefix: str
    directory_date: str
    version: str
    scenario: str
    workflow: str
    runmodus: str
    n_ensemble: str
    kind: str
    run_instance: str
    region: str
    strict_cv: bool = False


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


def _slug(value: str) -> str:
    return cmip6._slug(value)


def _cmip7_variable_meta(var_name: str) -> BrandedVariableMeta:
    try:
        return CMIP7_VARIABLES[var_name]
    except KeyError as exc:
        raise ValueError(
            f"No CMIP7 branded-variable mapping configured for {var_name!r}. "
            "Add it to CMIP7_VARIABLES before writing CMIP7 outputs."
        ) from exc


def _split_variant_label(variant_label: str) -> dict[str, str]:
    match = re.fullmatch(r"(r[^i]+)(i[^p]+)(p[^f]+)(f.+)", str(variant_label or "").strip())
    if not match:
        raise ValueError(f"CMIP7 variant_label must look like r<i>i<i>p<i>f<i>, got {variant_label!r}")
    return {
        "realization_index": match.group(1),
        "initialization_index": match.group(2),
        "physics_index": match.group(3),
        "forcing_index": match.group(4),
    }


def _validate_cmip7_shape(*, grid_label: str, directory_date: str, region: str, strict_cv: bool) -> None:
    if not re.fullmatch(r"v\d{8}", str(directory_date or "")):
        raise ValueError(f"CMIP7 directory_date must look like vYYYYMMDD, got {directory_date!r}")
    if not str(region or "").strip():
        raise ValueError("CMIP7 region must not be empty")
    if strict_cv and not re.fullmatch(r"g\d{3,}", str(grid_label or "")):
        raise ValueError(
            f"CMIP7 strict mode requires a registered-style grid_label like g121, got {grid_label!r}"
        )


def is_run_complete(output_root: Path) -> bool:
    return (
        completion_marker_path(output_root).exists()
        and upload_manifest_path(output_root).exists()
        and local_cleanup_manifest_path(output_root).exists()
    )


def build_cmip7_local_run_root(
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


def build_cmip7_remote_run_prefix(
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
        / "gcmagicc_cmip7_data_catalog.json"
    ).resolve(strict=False)


def directory_date_from_run_instance(run_instance: str) -> str:
    match = re.search(r"(\d{8})-(\d{4})$", str(run_instance))
    if match:
        return f"v{match.group(1)}"
    return utc_now().strftime("v%Y%m%d")


def cmor_version_from_run_instance(run_instance: str) -> str:
    return directory_date_from_run_instance(run_instance)


def _cmip7_directory(
    *,
    output_root: Path,
    meta: dict[str, Any],
    variable: BrandedVariableMeta,
    region: str,
    grid_label: str,
    directory_date: str,
) -> Path:
    return (
        output_root
        / CMIP7_DRS_SPECS
        / CMIP7_MIP_ERA
        / str(meta["activity_id"])
        / str(meta["institution_id"])
        / str(meta["source_id"])
        / str(meta["experiment_id"])
        / str(meta["member_id"])
        / region
        / variable.frequency
        / variable.variable_id
        / variable.branding_suffix
        / grid_label
        / directory_date
    )


def _cmip7_filename(
    *,
    meta: dict[str, Any],
    variable: BrandedVariableMeta,
    region: str,
    grid_label: str,
    time_range: str | None,
) -> str:
    parts = [
        variable.variable_id,
        variable.branding_suffix,
        variable.frequency,
        region,
        grid_label,
        str(meta["source_id"]),
        str(meta["experiment_id"]),
        str(meta["member_id"]),
    ]
    if time_range:
        parts.append(time_range)
    return "_".join(parts) + ".nc"


def csv_path_for_run(output_root: Path, meta: dict[str, Any], tr: str, *, region: str) -> Path:
    fname = (
        f"timevars_CMIP7_{meta['source_id']}_{meta['experiment_id']}"
        f"_{meta['member_id']}_{region}_{meta['grid_label']}_{tr}.csv"
    )
    return output_root / TIMEVAR_SUBDIR / fname


def _write_csv_row(csvfile: Path, header: list[str], row: dict[str, Any]) -> None:
    csvfile.parent.mkdir(parents=True, exist_ok=True)
    new = not csvfile.exists()
    with open(csvfile, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if new:
            writer.writeheader()
        writer.writerow(row)


def _creation_date() -> str:
    return utc_now().isoformat(timespec="seconds") + "Z"


def _cmip7_global_attrs(
    *,
    variable: BrandedVariableMeta,
    meta: dict[str, Any],
    p0: Any | None,
    p1: Any | None,
    tr: str | None,
    grid: str,
    region: str,
    grid_desc: str = "regular latitude-longitude grid",
) -> dict[str, Any]:
    variant_parts = _split_variant_label(str(meta["member_id"]))
    attrs: dict[str, Any] = {
        "Conventions": CMIP7_CONVENTIONS,
        "activity_id": str(meta["activity_id"]),
        "area_label": variable.area_label,
        "branded_variable": variable.branded_variable,
        "branding_suffix": variable.branding_suffix,
        "creation_date": _creation_date(),
        "data_specs_version": CMIP7_DATA_SPECS_VERSION,
        "drs_specs": CMIP7_DRS_SPECS,
        "experiment_id": str(meta["experiment_id"]),
        "experiment": cmip6._experiment_long_name(str(meta["experiment_id"])),
        "frequency": variable.frequency,
        "grid_label": grid,
        "grid": grid_desc,
        "horizontal_label": variable.horizontal_label,
        "institution_id": str(meta["institution_id"]),
        "license_id": CMIP7_DEFAULT_LICENSE_ID,
        "mip_era": CMIP7_MIP_ERA,
        "nominal_resolution": "1x1 degree",
        "product": CMIP7_DEFAULT_PRODUCT,
        "realm": variable.realm,
        "region": region,
        "source_id": str(meta["source_id"]),
        "temporal_label": variable.temporal_label,
        "tracking_id": f"hdl:21.14100/{os.urandom(8).hex()}",
        "variable_id": variable.variable_id,
        "variant_label": str(meta["member_id"]),
        "vertical_label": variable.vertical_label,
        "history": "Reformatted by notebook 351 from multi-var files, M. Meinshausen",
        **variant_parts,
    }
    if p0 is not None:
        attrs["start_time"] = p0.isoformat()
    if p1 is not None:
        attrs["end_time"] = p1.isoformat()
    if tr:
        attrs["time_range"] = tr
    return attrs


def maybe_write_areacella(ds: xr.Dataset, meta: dict[str, Any], context: ConversionContext) -> Path | None:
    if "lat" not in ds or "lon" not in ds:
        return None

    variable = _cmip7_variable_meta("areacella")
    lat_bnds = cmip6._get_bounds(ds, "lat")
    lon_bnds = cmip6._get_bounds(ds, "lon")
    area = cmip6._cell_areas(lat_bnds, lon_bnds).astype("float32")

    areacella = xr.DataArray(area, dims=("lat", "lon"), coords={"lat": ds["lat"], "lon": ds["lon"]})
    areacella.attrs.update(
        {
            "standard_name": "cell_area",
            "long_name": "Grid-Cell Area for Atmospheric Grid Variables",
            "units": "m2",
            "cell_methods": variable.cell_methods,
            "comment": "Approximated on a spherical Earth assuming regular lat-lon grid",
            "missing_value": np.float32(1e20),
        }
    )
    areacella.attrs.pop("_FillValue", None)

    lat_bnds_da = xr.DataArray(lat_bnds, dims=("lat", "bnds"), coords={"lat": ds["lat"]})
    lon_bnds_da = xr.DataArray(lon_bnds, dims=("lon", "bnds"), coords={"lon": ds["lon"]})
    cmip6._fill_coord_attrs(ds["lat"], "lat")
    cmip6._fill_coord_attrs(ds["lon"], "lon")
    cmip6._fill_coord_attrs(lat_bnds_da["lat"], "lat")  # type: ignore[index]
    cmip6._fill_coord_attrs(lon_bnds_da["lon"], "lon")  # type: ignore[index]
    lat_attrs = dict(ds["lat"].attrs)
    lon_attrs = dict(ds["lon"].attrs)
    lat_attrs["bounds"] = "lat_bnds"
    lon_attrs["bounds"] = "lon_bnds"

    grid_desc = str(ds.attrs.get("grid", "regular latitude-longitude grid"))
    ads = xr.Dataset(
        data_vars={
            "areacella": areacella,
            "lat_bnds": lat_bnds_da,
            "lon_bnds": lon_bnds_da,
        },
        coords={"lat": ds["lat"], "lon": ds["lon"]},
        attrs=_cmip7_global_attrs(
            variable=variable,
            meta=meta,
            p0=None,
            p1=None,
            tr=None,
            grid=str(meta["grid_label"]),
            region=context.region,
            grid_desc=grid_desc,
        ),
    )
    ads["lat"].attrs = lat_attrs
    ads["lon"].attrs = lon_attrs

    outdir = _cmip7_directory(
        output_root=context.output_root,
        meta=meta,
        variable=variable,
        region=context.region,
        grid_label=str(meta["grid_label"]),
        directory_date=context.directory_date,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / _cmip7_filename(
        meta=meta,
        variable=variable,
        region=context.region,
        grid_label=str(meta["grid_label"]),
        time_range=None,
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
    meta = cmip6.extract_meta(nc)
    meta["directory_date"] = context.directory_date

    variable_entries: List[VariableMeta] = []
    generated_rel_paths: List[str] = []

    with xr.open_dataset(nc, decode_times=False) as ds:
        dates, p0, p1, tr = cmip6.time_axis_info(ds)
        time_range_human = cmip6.humanize_time_range(tr)
        meta["time_start"] = p0.isoformat()
        meta["time_end"] = p1.isoformat()
        meta["grid_label"] = str(ds.attrs.get("grid_label", "gr9"))
        _validate_cmip7_shape(
            grid_label=str(meta["grid_label"]),
            directory_date=context.directory_date,
            region=context.region,
            strict_cv=context.strict_cv,
        )

        areacella_path = maybe_write_areacella(ds, meta, context)
        if areacella_path is not None:
            generated_rel_paths.append(areacella_path.relative_to(context.output_root).as_posix())

        csvfile = csv_path_for_run(context.output_root, meta, tr, region=context.region)
        base = [
            "institution_id",
            "source_id",
            "experiment_id",
            "variant_label",
            "region",
            "grid_label",
            "time_start",
            "time_end",
            "variable",
            "branded_variable",
            "unit",
            "domain",
            "filepath",
        ]
        header = base + [d.isoformat() for d in dates]

        for source_var_name in ds.data_vars:
            var = ds[source_var_name]
            if {"lat", "lon"}.issubset(var.dims):
                variable = _cmip7_variable_meta(source_var_name)
                target_var_name = variable.variable_id
                vds = var.to_dataset(name=target_var_name)
                if "time" in vds:
                    vds["time"].attrs["calendar"] = ds["time"].attrs.get("calendar", "standard")
                if "lat" in vds:
                    cmip6._fill_coord_attrs(vds["lat"], "lat")
                    lat_bnds = cmip6._get_bounds(ds, "lat")
                    vds["lat_bnds"] = xr.DataArray(lat_bnds, dims=("lat", "bnds"), coords={"lat": vds["lat"]})
                    vds["lat"].attrs["bounds"] = "lat_bnds"
                if "lon" in vds:
                    cmip6._fill_coord_attrs(vds["lon"], "lon")
                    lon_bnds = cmip6._get_bounds(ds, "lon")
                    vds["lon_bnds"] = xr.DataArray(lon_bnds, dims=("lon", "bnds"), coords={"lon": vds["lon"]})
                    vds["lon"].attrs["bounds"] = "lon_bnds"

                units_attr = vds[target_var_name].attrs.get("units")
                if not units_attr or str(units_attr).lower() == "unknown":
                    vds[target_var_name].attrs["units"] = cmip6.DEFAULT_UNITS.get(target_var_name, "")
                std_attr = vds[target_var_name].attrs.get("standard_name")
                if not std_attr:
                    vds[target_var_name].attrs["standard_name"] = cmip6.DEFAULT_STANDARD_NAME.get(target_var_name, "")
                if not vds[target_var_name].attrs.get("cell_methods"):
                    vds[target_var_name].attrs["cell_methods"] = variable.cell_methods

                cmip6._add_height_coord(vds, target_var_name)
                vds.attrs = _cmip7_global_attrs(
                    variable=variable,
                    meta=meta,
                    p0=p0,
                    p1=p1,
                    tr=tr,
                    grid=str(meta["grid_label"]),
                    region=context.region,
                    grid_desc=str(ds.attrs.get("grid", "regular latitude-longitude grid")),
                )

                outdir = _cmip7_directory(
                    output_root=context.output_root,
                    meta=meta,
                    variable=variable,
                    region=context.region,
                    grid_label=str(meta["grid_label"]),
                    directory_date=context.directory_date,
                )
                outdir.mkdir(parents=True, exist_ok=True)
                ncfpath = outdir / _cmip7_filename(
                    meta=meta,
                    variable=variable,
                    region=context.region,
                    grid_label=str(meta["grid_label"]),
                    time_range=tr,
                )
                write_netcdf_compat(vds, ncfpath, encoding={target_var_name: COMPRESS_OPTS})

                rel_path = ncfpath.relative_to(context.output_root).as_posix()
                generated_rel_paths.append(rel_path)
                variable_entries.append(
                    VariableMeta(
                        id=target_var_name,
                        branded_variable=variable.branded_variable,
                        branding_suffix=variable.branding_suffix,
                        frequency=variable.frequency,
                        region=context.region,
                        activity_id=str(meta["activity_id"]),
                        institution_id=str(meta["institution_id"]),
                        source_id=str(meta["source_id"]),
                        experiment_id=str(meta["experiment_id"]),
                        member_id=str(meta["member_id"]),
                        variant_label=str(meta["member_id"]),
                        grid_label=str(meta["grid_label"]),
                        directory_date=context.directory_date,
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
                        "variant_label": meta["member_id"],
                        "region": context.region,
                        "grid_label": meta["grid_label"],
                        "time_start": meta["time_start"],
                        "time_end": meta["time_end"],
                        "variable": target_var_name,
                        "branded_variable": variable.branded_variable,
                        "unit": var.attrs.get("units", ""),
                        "domain": "grid",
                        "filepath": rel_path,
                    }
                )
                _write_csv_row(csvfile, header, row)
            else:
                row = {
                    "institution_id": meta["institution_id"],
                    "source_id": meta["source_id"],
                    "experiment_id": meta["experiment_id"],
                    "variant_label": meta["member_id"],
                    "region": context.region,
                    "grid_label": meta["grid_label"],
                    "time_start": meta["time_start"],
                    "time_end": meta["time_end"],
                    "variable": source_var_name,
                    "branded_variable": "",
                    "unit": cmip6._timeseries_unit(source_var_name, ds),
                    "domain": "timeseries",
                    "filepath": "thisfile",
                }
                for d, val in zip(dates, var.values):
                    row[d.isoformat()] = np.asarray(val).item()
                _write_csv_row(csvfile, header, row)

    generated_rel_paths.append(csvfile.relative_to(context.output_root).as_posix())
    relative_output_paths = sorted(dict.fromkeys(generated_rel_paths))
    dataset_id = "-".join(
        [
            _slug(context.version),
            _slug(context.scenario),
            _slug(context.run_instance),
            _slug(str(meta["member_id"])),
            _slug(context.region),
        ]
    )
    dataset_meta = DatasetMeta(
        id=dataset_id,
        label=f"{meta['source_id']} | {meta['experiment_id']} | {context.workflow} | {meta['member_id']} | CMIP7",
        source_id=str(meta["source_id"]),
        experiment_id=str(meta["experiment_id"]),
        member_id=str(meta["member_id"]),
        variant_label=str(meta["member_id"]),
        workflow=context.workflow,
        institution_id=str(meta["institution_id"]),
        grid_label=str(meta["grid_label"]),
        activity_id=str(meta["activity_id"]),
        region=context.region,
        directory_date=context.directory_date,
        time_range=time_range_human,
        variables=sorted(variable_entries, key=lambda item: (item.frequency, item.id, item.branding_suffix)),
        relative_output_paths=relative_output_paths,
        zarr=None,
    )
    return ConvertedDataset(
        dataset_meta=dataset_meta,
        relative_output_paths=relative_output_paths,
        source_file=str(nc),
    )


def _variable_meta_from_dict(payload: Dict[str, Any]) -> VariableMeta:
    return VariableMeta(
        id=str(payload.get("id") or ""),
        branded_variable=str(payload.get("branded_variable") or ""),
        branding_suffix=str(payload.get("branding_suffix") or ""),
        frequency=str(payload.get("frequency") or ""),
        region=str(payload.get("region") or ""),
        activity_id=str(payload.get("activity_id") or ""),
        institution_id=str(payload.get("institution_id") or ""),
        source_id=str(payload.get("source_id") or ""),
        experiment_id=str(payload.get("experiment_id") or ""),
        member_id=str(payload.get("member_id") or ""),
        variant_label=str(payload.get("variant_label") or payload.get("member_id") or ""),
        grid_label=str(payload.get("grid_label") or ""),
        directory_date=str(payload.get("directory_date") or ""),
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
        variant_label=str(payload.get("variant_label") or payload.get("member_id") or ""),
        workflow=str(payload.get("workflow") or ""),
        institution_id=str(payload.get("institution_id") or ""),
        grid_label=str(payload.get("grid_label") or ""),
        activity_id=str(payload.get("activity_id") or ""),
        region=str(payload.get("region") or DEFAULT_REGION),
        directory_date=str(payload.get("directory_date") or ""),
        time_range=str(payload.get("time_range") or ""),
        variables=[
            _variable_meta_from_dict(item)
            for item in variables_raw
            if isinstance(item, dict)
        ],
        relative_output_paths=[
            str(item)
            for item in (payload.get("relative_output_paths") or [])
            if str(item).strip()
        ],
        zarr=dict(payload.get("zarr")) if isinstance(payload.get("zarr"), dict) else None,
    )


def _load_dataset_meta_list(manifest_path: Path) -> List[DatasetMeta]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        _dataset_meta_from_dict(item)
        for item in (payload.get("datasets") or [])
        if isinstance(item, dict) and "variables" in item
    ]


def _dataset_expected_remote_keys(dataset: DatasetMeta, remote_run_prefix: str) -> set[str]:
    keys = {
        str(item.path).strip("/")
        for item in dataset.variables
        if str(item.path).strip("/")
    }
    for rel_path in dataset.relative_output_paths:
        rel = str(rel_path).strip("/")
        if rel:
            keys.add(f"{remote_run_prefix.strip('/')}/{rel}")
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


def _load_local_resume_progress(output_root: Path, remote_run_prefix: str) -> ResumeProgress | None:
    manifest = run_manifest_path(output_root)
    upload_manifest = upload_manifest_path(output_root)
    marker = completion_marker_path(output_root)
    if marker.exists() or not manifest.exists() or not upload_manifest.exists():
        return None
    dataset_metas = _load_dataset_meta_list(manifest)
    uploaded_records = cmip6._load_upload_records(upload_manifest)
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

    run_payload = cmip6._download_remote_json(
        client,
        bucket=bucket,
        key=cmip6._remote_manifest_key(remote_run_prefix, RUN_MANIFEST_BASENAME),
    )
    upload_payload = cmip6._download_remote_json(
        client,
        bucket=bucket,
        key=cmip6._remote_manifest_key(remote_run_prefix, UPLOAD_MANIFEST_BASENAME),
    )
    if not isinstance(run_payload, dict) or not isinstance(upload_payload, dict):
        return None
    dataset_metas = [
        _dataset_meta_from_dict(item)
        for item in (run_payload.get("datasets") or [])
        if isinstance(item, dict) and "variables" in item
    ]
    uploaded_records = [
        cmip6._upload_record_from_dict(item)
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


def _write_upload_manifest(
    *,
    manifest_path: Path,
    bucket: str,
    remote_run_prefix: str,
    uploaded_records: Sequence[UploadRecord],
) -> None:
    cmip6._write_upload_manifest(
        manifest_path=manifest_path,
        bucket=bucket,
        remote_run_prefix=remote_run_prefix,
        uploaded_records=uploaded_records,
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
    cmip6._upload_progress_metadata(
        output_root=output_root,
        manifest_path=manifest_path,
        upload_manifest_local=upload_manifest_local,
        bucket=bucket,
        remote_run_prefix=remote_run_prefix,
        client=client,
    )


def _delete_relative_payload_paths(
    *,
    output_root: Path,
    relative_paths: Iterable[str],
) -> tuple[int, int]:
    return cmip6._delete_relative_payload_paths(output_root=output_root, relative_paths=relative_paths)


def _cleanup_local_payload(
    *,
    output_root: Path,
    cleanup_enabled: bool,
) -> tuple[Path, Dict[str, Any]]:
    return cmip6._cleanup_local_payload(output_root=output_root, cleanup_enabled=cleanup_enabled)


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
    directory_date: str,
    region: str,
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
        "directory_date": directory_date,
        "region": region,
        "drs_specs": CMIP7_DRS_SPECS,
        "mip_era": CMIP7_MIP_ERA,
        "data_specs_version": CMIP7_DATA_SPECS_VERSION,
        "source_file_count": source_file_count,
        "datasets": [asdict(item) for item in datasets],
    }
    if incremental_index is not None:
        payload["incremental_source_index"] = int(incremental_index)
    write_json_atomic(manifest_path, payload)


def _sorted_member_results(results_by_index: Dict[int, MemberWorkResult]) -> List[MemberWorkResult]:
    return [results_by_index[index] for index in sorted(results_by_index)]


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


def _experiment_id_from_name(path: Path) -> str | None:
    return cmip6._experiment_id_from_name(path)


def _source_member_id(path: Path) -> str:
    return cmip6._source_member_id(path)


def write_frontend_catalog(datasets: Sequence[DatasetMeta], catalog_path: Path) -> Path:
    incoming = [asdict(item) for item in datasets]
    merged_datasets = incoming
    note = (
        "Community-facing CMIP7-style ERA5spliced downloads. "
        "Each variable is published as its own branded-variable NetCDF file, served from the public GCMagicc data origin (data.gcmagicc.org)."
    )

    if catalog_path.exists():
        try:
            existing_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            existing_datasets = existing_payload.get("datasets") or []
            if isinstance(existing_datasets, list):
                merged_datasets = cmip6._merge_catalog_datasets(existing_datasets, incoming)
            existing_note = existing_payload.get("note")
            if isinstance(existing_note, str) and existing_note.strip():
                note = existing_note
        except Exception:
            pass

    payload = {
        "generated_at": utc_now().isoformat(),
        "note": note,
        "time": cmip6._time_window_from_datasets(merged_datasets),
        "drs_specs": CMIP7_DRS_SPECS,
        "mip_era": CMIP7_MIP_ERA,
        "data_specs_version": CMIP7_DATA_SPECS_VERSION,
        "datasets": merged_datasets,
    }
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(catalog_path, payload)
    return catalog_path


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
    region: str = DEFAULT_REGION,
    directory_date: str | None = None,
    strict_cv: bool = False,
) -> RunConversionResult:
    source_run_root = Path(source_run_root).expanduser().resolve(strict=False)
    output_root = Path(output_root).expanduser().resolve(strict=False)
    marker_path = completion_marker_path(output_root)
    manifest_path = run_manifest_path(output_root)
    upload_manifest_local = upload_manifest_path(output_root)
    cleanup_manifest = local_cleanup_manifest_path(output_root)
    resolved_directory_date = directory_date or directory_date_from_run_instance(run_instance)
    _validate_cmip7_shape(
        grid_label="g000" if strict_cv else "gr9",
        directory_date=resolved_directory_date,
        region=region,
        strict_cv=False,
    )
    remote_run_prefix = build_cmip7_remote_run_prefix(
        upload_prefix=upload_prefix,
        version=version,
        scenario=scenario,
        workflow=workflow,
        runmodus=runmodus,
        n_ensemble=n_ensemble,
        kind=kind,
        run_instance=run_instance,
    )

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

    upload_client: Any | None = cmip6._build_boto3_client() if upload_bucket else None

    context = ConversionContext(
        output_root=output_root,
        remote_run_prefix=remote_run_prefix,
        directory_date=resolved_directory_date,
        version=version,
        scenario=scenario,
        workflow=workflow,
        runmodus=runmodus,
        n_ensemble=n_ensemble,
        kind=kind,
        run_instance=run_instance,
        region=str(region),
        strict_cv=bool(strict_cv),
    )
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
            directory_date=context.directory_date,
            region=context.region,
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
            directory_date=context.directory_date,
            region=context.region,
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
                    f"CMIP7 member conversion failed for {failed_path.name} (index {failed_index}): {failed_exc}"
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
            "directory_date": context.directory_date,
            "region": context.region,
            "drs_specs": CMIP7_DRS_SPECS,
            "mip_era": CMIP7_MIP_ERA,
            "data_specs_version": CMIP7_DATA_SPECS_VERSION,
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
        description="Convert one staged ERA5spliced n_20/original run into CMIP7-style NetCDF files."
    )
    parser.add_argument("--source-run-root", type=Path, required=True, help="Staged run root under ERA5spliced_localstaging.")
    parser.add_argument("--output-root", type=Path, default=None, help="Final local CMIP7 run root. Defaults under ERA5spliced_cmip7_localresults.")
    parser.add_argument("--version", required=True, help="Canonical version (e.g. v100 or v101).")
    parser.add_argument("--scenario", required=True, help="Canonical scenario identifier.")
    parser.add_argument("--run-instance", required=True, help="Run directory name (e.g. run_debiasloop_... ).")
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW, help="Workflow token (default: AR6).")
    parser.add_argument("--runmodus", default=DEFAULT_RUNMODUS, help="Runmodus token (default: all).")
    parser.add_argument("--n-ensemble", default=DEFAULT_N_ENSEMBLE, help="Ensemble token (default: n_20).")
    parser.add_argument("--kind", default=DEFAULT_KIND, help="Canonical kind (default: original).")
    parser.add_argument("--region", default=DEFAULT_REGION, help="CMIP7 region token (default: glb).")
    parser.add_argument("--directory-date", default=None, help="CMIP7 directoryDateDD folder (vYYYYMMDD). Defaults from run-instance date.")
    parser.add_argument("--strict-cv", action="store_true", help="Enable stricter local checks for registered-looking CMIP7 labels.")
    parser.add_argument("--upload-bucket", default=get_object_bucket(), help="Target OVH bucket for converted outputs.")
    parser.add_argument("--upload-prefix", default=DEFAULT_UPLOAD_PREFIX, help="Target OVH prefix root.")
    parser.add_argument(
        "--skip-if-complete",
        action="store_true",
        help="Skip when run_complete.json, upload_manifest.json, and local_cleanup.json already exist.",
    )
    parser.add_argument("--frontend-catalog", type=Path, default=default_frontend_catalog_path(), help="Frontend CMIP7 catalog JSON to update incrementally.")
    parser.add_argument("--no-update-frontend-catalog", action="store_true", help="Do not touch the frontend CMIP7 catalog.")
    parser.add_argument("--no-upload", action="store_true", help="Convert locally only; skip OVH upload and verification.")
    parser.add_argument(
        "--member-workers",
        type=int,
        default=1,
        help="Parallel member workers. Default: 1 for standalone 351 runs.",
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
        else build_cmip7_local_run_root(
            base_root=get_era5spliced_cmip7_localresults_root(),
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
        region=str(args.region),
        directory_date=str(args.directory_date) if args.directory_date else None,
        strict_cv=bool(args.strict_cv),
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
