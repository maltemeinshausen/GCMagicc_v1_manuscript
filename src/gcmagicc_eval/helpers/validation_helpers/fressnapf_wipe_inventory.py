"""Generate an evidence-backed pre-wipe inventory for data/archive."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import Callable, Iterable, Sequence


DEFAULT_DATASETS_ROOT = Path("data/archive")
DEFAULT_OUTPUT_DIR = Path("~/.claude/plans/fressnapf-raid-wipe-artifacts").expanduser()
DEFAULT_LOCAL_BACKUP_DIR = Path("/data/fressnapf_pre_raid_backup_20260422")
LOCAL_BACKUP_THRESHOLD_BYTES = 100 * 1024**3
IGNORE_EXACT = {".codex_write_test"}
IGNORE_PREFIXES = ("read-16j.",)


@dataclass(frozen=True)
class InventorySpec:
    rel_path: str
    backup_now: str
    restore_source: str
    restore_locator: str
    evidence_kind: str
    evidence_ref: str
    recommended_temp_target: str
    notes: str = ""
    aggregate: bool = False
    aggregate_children: tuple[str, ...] = ()
    needs_eth_prompt: bool = False
    eth_candidates: tuple[str, ...] = ()
    eth_search_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class InventoryRow:
    path: str
    size_bytes: int
    size_human: str
    backup_now: str
    restore_source: str
    restore_locator: str
    evidence_kind: str
    evidence_ref: str
    recommended_temp_target: str
    notes: str
    aggregate: bool
    needs_eth_prompt: bool
    eth_candidates: tuple[str, ...]
    eth_search_terms: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
    ) as handle:
        handle.write(text)
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def _format_bytes(num_bytes: int) -> str:
    value = float(max(0, int(num_bytes)))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024.0 or unit == "PiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def _looks_ignored(name: str) -> bool:
    return name in IGNORE_EXACT or any(name.startswith(prefix) for prefix in IGNORE_PREFIXES)


def _run_command(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), check=False, capture_output=True, text=True)


def _du_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    proc = _run_command(["du", "-sB1", str(path)])
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"du -sB1 failed for {path}: {detail}")
    token = proc.stdout.strip().split()[0]
    return int(token)


def _df_available_bytes(path: Path) -> int:
    proc = _run_command(["df", "-B1", str(path)])
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"df failed for {path}: {detail}")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Unexpected df output for {path}: {proc.stdout!r}")
    return int(lines[1].split()[3])


def _tsv_escape(value: str) -> str:
    return str(value).replace("\t", " ").replace("\n", " ").strip()


def _candidate_block(candidates: Sequence[str]) -> str:
    if not candidates:
        return "(none)"
    return "\n".join(f"- `{candidate}`" for candidate in candidates)


def _build_find_command(search_terms: Sequence[str]) -> str:
    base_paths = ["data/site_eth/projects", "data/site_eth"]
    if not search_terms:
        pattern = "-path '*'"
    else:
        pieces = [f"-path '*{term}*'" for term in search_terms]
        pattern = " -o ".join(pieces)
    joined_bases = " ".join(shlex.quote(path) for path in base_paths)
    return (
        f"find {joined_bases} -maxdepth 7 \\( {pattern} \\) 2>/dev/null | sort"
    )


def _build_du_command(candidates: Sequence[str]) -> str:
    if candidates:
        joined = " ".join(shlex.quote(path) for path in candidates)
        return f"du -sh {joined} 2>/dev/null"
    return "echo 'No concrete ETH candidate paths yet.'"


def _build_ls_command(candidates: Sequence[str]) -> str:
    if candidates:
        joined = " ".join(shlex.quote(path) for path in candidates)
        return f"ls -lah {joined} 2>/dev/null"
    return "echo 'No concrete ETH candidate paths yet.'"


def build_inventory_specs() -> list[InventorySpec]:
    return [
        InventorySpec(
            rel_path="CMIP6",
            backup_now="no",
            restore_source="s3",
            restore_locator=(
                "ovh:gcmagicc-scratch/nc/reference/out_ETHFOG_10June2025_vetted ; "
                "ovh:gcmagicc-scratch/nc/reference/cmip6_ETHFOG_net ; "
                "ovh:gcmagicc-scratch/nc/gus/cmip6_healpix"
            ),
            evidence_kind="repo_test+live_rclone",
            evidence_ref=(
                "tests/test_fressnapf_backfill.py ; "
                "2001_instructions_s3_operations.md ; live rclone lsd"
            ),
            recommended_temp_target="skip",
            notes="aggregate_overview_only; actionable rows below carry the precise restore targets",
            aggregate=True,
            aggregate_children=(
                "CMIP6/ETHFOG/processed/out_ETHFOG_10June2025_vetted",
                "CMIP6/ETHFOG/processed/out_ETHFOG_10June2025_vetted_dataderivatives",
                "CMIP6/ETHFOG/net",
                "CMIP6/ETHFOG/healpix_daily_sample",
            ),
        ),
        InventorySpec(
            rel_path="CMIP6/ETHFOG/processed/out_ETHFOG_10June2025_vetted",
            backup_now="no",
            restore_source="s3",
            restore_locator="ovh:gcmagicc-scratch/nc/reference/out_ETHFOG_10June2025_vetted",
            evidence_kind="repo_test+live_rclone_size",
            evidence_ref=(
                "tests/test_fressnapf_backfill.py ; "
                "scripts/ovh_nc_copy.sh ; "
                "rclone size ovh:gcmagicc-scratch/nc/reference/out_ETHFOG_10June2025_vetted"
            ),
            recommended_temp_target="skip",
            notes="canonical vetted ETHFOG reference bundle",
        ),
        InventorySpec(
            rel_path="CMIP6/ETHFOG/processed/out_ETHFOG_10June2025_vetted_dataderivatives",
            backup_now="no",
            restore_source="s3",
            restore_locator="ovh:gcmagicc-scratch/nc/reference/out_ETHFOG_10June2025_vetted",
            evidence_kind="repo_script+companion_bundle",
            evidence_ref=(
                "scripts/ovh_nc_copy.sh ; "
                "local processed companion tree under data/archive/CMIP6/ETHFOG/processed"
            ),
            recommended_temp_target="skip",
            notes="treated as part of the mirrored ETHFOG reference bundle",
        ),
        InventorySpec(
            rel_path="CMIP6/ETHFOG/net",
            backup_now="no",
            restore_source="s3",
            restore_locator="ovh:gcmagicc-scratch/nc/reference/cmip6_ETHFOG_net",
            evidence_kind="repo_test+live_rclone",
            evidence_ref="tests/test_fressnapf_backfill.py ; rclone lsd ovh:gcmagicc-scratch/nc/reference/cmip6_ETHFOG_net",
            recommended_temp_target="skip",
            notes="reference CMIP6 net tree",
        ),
        InventorySpec(
            rel_path="CMIP6/ETHFOG/healpix_daily_sample",
            backup_now="no",
            restore_source="s3",
            restore_locator="ovh:gcmagicc-scratch/nc/gus/cmip6_healpix",
            evidence_kind="live_rclone_prefix",
            evidence_ref="rclone lsd ovh:gcmagicc-scratch/nc/gus/cmip6_healpix",
            recommended_temp_target="skip",
            notes="healpix samples are already present under the gus cmip6_healpix prefixes",
        ),
        InventorySpec(
            rel_path="Caravan",
            backup_now="no",
            restore_source="public",
            restore_locator="https://doi.org/10.5281/zenodo.10968091",
            evidence_kind="public_dataset",
            evidence_ref="Caravan public Zenodo DOI 10.5281/zenodo.10968091",
            recommended_temp_target="skip",
            notes="aggregate_overview_only; actionable rows below",
            aggregate=True,
            aggregate_children=("Caravan/v1.4", "Caravan/v1.6"),
        ),
        InventorySpec(
            rel_path="Caravan/v1.4",
            backup_now="no",
            restore_source="public",
            restore_locator="https://doi.org/10.5281/zenodo.10968091",
            evidence_kind="public_dataset",
            evidence_ref="Caravan public Zenodo DOI 10.5281/zenodo.10968091",
            recommended_temp_target="skip",
            notes="publicly re-downloadable release",
        ),
        InventorySpec(
            rel_path="Caravan/v1.6",
            backup_now="no",
            restore_source="public",
            restore_locator="https://doi.org/10.5281/zenodo.10968091",
            evidence_kind="public_dataset",
            evidence_ref="Caravan public Zenodo DOI 10.5281/zenodo.10968091",
            recommended_temp_target="skip",
            notes="placeholder / empty local version directory",
        ),
        InventorySpec(
            rel_path="ERA5",
            backup_now="no",
            restore_source="public",
            restore_locator="Copernicus CDS ERA5 / ERA5-Land re-download and monthly re-derivation",
            evidence_kind="public_dataset+repo_script",
            evidence_ref="scripts/2161_fressnapf_eth_push.py ; local ERA5 processed tree",
            recommended_temp_target="skip",
            notes="aggregate_overview_only; actionable rows below",
            aggregate=True,
            aggregate_children=("ERA5/processed",),
        ),
        InventorySpec(
            rel_path="ERA5/processed",
            backup_now="no",
            restore_source="public",
            restore_locator="Copernicus CDS ERA5 / ERA5-Land re-download and monthly re-derivation",
            evidence_kind="public_dataset+repo_script",
            evidence_ref="scripts/2161_fressnapf_eth_push.py ; local processed/monthly tree",
            recommended_temp_target="skip",
            notes="small derived branch; public source + local derivation recipe",
        ),
        InventorySpec(
            rel_path="GCMAGICCoutput",
            backup_now="no",
            restore_source="s3",
            restore_locator="mixed object-store restores across S3 and R2; see child rows",
            evidence_kind="repo_test+live_rclone",
            evidence_ref=(
                "tests/test_fressnapf_backfill.py ; "
                "2001_instructions_s3_operations.md ; "
                "live rclone lsd/rclone size"
            ),
            recommended_temp_target="skip",
            notes="aggregate_overview_only; child rows capture the precise restore path and local-backup exceptions",
            aggregate=True,
            aggregate_children=(
                "GCMAGICCoutput/CMIP6replicas",
                "GCMAGICCoutput/ERA5spliced",
                "GCMAGICCoutput/drought_attribution_758",
            ),
        ),
        InventorySpec(
            rel_path="GCMAGICCoutput/CMIP6replicas",
            backup_now="no",
            restore_source="s3",
            restore_locator="ovh:gcmagicc-scratch/nc/gus/v100/gcmagicc ; ovh:gcmagicc-scratch/nc/eth/v101/gcmagicc",
            evidence_kind="repo_test+live_rclone_size",
            evidence_ref=(
                "tests/test_fressnapf_backfill.py ; "
                "rclone size ovh:gcmagicc-scratch/nc/gus/v100/gcmagicc ; "
                "rclone size ovh:gcmagicc-scratch/nc/eth/v101/gcmagicc"
            ),
            recommended_temp_target="skip",
            notes="aggregate_overview_only; keep child metadata exceptions separately",
            aggregate=True,
            aggregate_children=(
                "GCMAGICCoutput/CMIP6replicas/v100",
                "GCMAGICCoutput/CMIP6replicas/v101",
                "GCMAGICCoutput/CMIP6replicas/_runmeta",
                "GCMAGICCoutput/CMIP6replicas/_space_relief_20260414",
            ),
        ),
        InventorySpec(
            rel_path="GCMAGICCoutput/CMIP6replicas/v100",
            backup_now="no",
            restore_source="s3",
            restore_locator="ovh:gcmagicc-scratch/nc/gus/v100/gcmagicc",
            evidence_kind="repo_test+live_rclone_size",
            evidence_ref="tests/test_fressnapf_backfill.py ; rclone size ovh:gcmagicc-scratch/nc/gus/v100/gcmagicc",
            recommended_temp_target="skip",
            notes="canonical gus v100 replica tree",
        ),
        InventorySpec(
            rel_path="GCMAGICCoutput/CMIP6replicas/v101",
            backup_now="no",
            restore_source="s3",
            restore_locator="ovh:gcmagicc-scratch/nc/eth/v101/gcmagicc",
            evidence_kind="repo_test+live_rclone_size",
            evidence_ref="tests/test_fressnapf_backfill.py ; rclone size ovh:gcmagicc-scratch/nc/eth/v101/gcmagicc",
            recommended_temp_target="skip",
            notes="canonical eth v101 replica tree",
        ),
        InventorySpec(
            rel_path="GCMAGICCoutput/CMIP6replicas/_runmeta",
            backup_now="yes",
            restore_source="local_backup:/data",
            restore_locator=str(DEFAULT_LOCAL_BACKUP_DIR),
            evidence_kind="local_operational_metadata",
            evidence_ref="current _runmeta tree under data/archive/GCMAGICCoutput/CMIP6replicas",
            recommended_temp_target="AUTO_LOCAL_DATA",
            notes="operational logs / sqlite manifests; preserve locally now and also query ETH",
            needs_eth_prompt=True,
            eth_candidates=(
                "data/site_eth/GCMAGICCoutput/CMIP6replicas/_runmeta",
                "data/site_eth/projects/gcmmagicc/data/metric_databases",
            ),
            eth_search_terms=("CMIP6replicas", "_runmeta", "metric_databases", "v100_nc_nn_20260414"),
        ),
        InventorySpec(
            rel_path="GCMAGICCoutput/CMIP6replicas/_space_relief_20260414",
            backup_now="yes",
            restore_source="local_backup:/data",
            restore_locator=str(DEFAULT_LOCAL_BACKUP_DIR),
            evidence_kind="local_space_relief_snapshot",
            evidence_ref="current _space_relief_20260414 tree under data/archive/GCMAGICCoutput/CMIP6replicas",
            recommended_temp_target="AUTO_LOCAL_DATA",
            notes="space-relief metric-db snapshot set; size in this report is on-disk usage, not apparent size",
            needs_eth_prompt=True,
            eth_candidates=(
                "data/site_eth/GCMAGICCoutput/CMIP6replicas/_space_relief_20260414",
                "data/site_eth/projects/gcmmagicc/data/metric_databases",
            ),
            eth_search_terms=("_space_relief_20260414", "metrics_snapshot", "dryruns_metric_databases"),
        ),
        InventorySpec(
            rel_path="GCMAGICCoutput/ERA5spliced",
            backup_now="no",
            restore_source="s3",
            restore_locator="ovh:gcmagicc-scratch/nc/consolidated/era5spliced ; ovh:gcmagicc-scratch/nc/cmip6/era5spliced/v101",
            evidence_kind="repo_docs+live_rclone",
            evidence_ref="2001_instructions_s3_operations.md ; rclone lsd ovh:gcmagicc-scratch/nc/consolidated/era5spliced ; rclone lsd ovh:gcmagicc-scratch/nc/cmip6/era5spliced/v101",
            recommended_temp_target="skip",
            notes="aggregate_overview_only; localresults metadata kept separately",
            aggregate=True,
            aggregate_children=(
                "GCMAGICCoutput/ERA5spliced/localstaging_archive",
                "GCMAGICCoutput/ERA5spliced/localresults",
            ),
        ),
        InventorySpec(
            rel_path="GCMAGICCoutput/ERA5spliced/localstaging_archive",
            backup_now="no",
            restore_source="s3",
            restore_locator="ovh:gcmagicc-scratch/nc/consolidated/era5spliced",
            evidence_kind="repo_test+live_rclone",
            evidence_ref="tests/test_fressnapf_backfill.py ; rclone lsd ovh:gcmagicc-scratch/nc/consolidated/era5spliced",
            recommended_temp_target="skip",
            notes="consolidated era5spliced archive tree",
        ),
        InventorySpec(
            rel_path="GCMAGICCoutput/ERA5spliced/localresults",
            backup_now="no",
            restore_source="s3",
            restore_locator="ovh:gcmagicc-scratch/nc/cmip6/era5spliced/v101",
            evidence_kind="live_rclone_prefix",
            evidence_ref="rclone lsd ovh:gcmagicc-scratch/nc/cmip6/era5spliced/v101",
            recommended_temp_target="skip",
            notes="aggregate_overview_only; keep _runmeta separately",
            aggregate=True,
            aggregate_children=(
                "GCMAGICCoutput/ERA5spliced/localresults/v101",
                "GCMAGICCoutput/ERA5spliced/localresults/_runmeta",
            ),
        ),
        InventorySpec(
            rel_path="GCMAGICCoutput/ERA5spliced/localresults/v101",
            backup_now="no",
            restore_source="s3",
            restore_locator="ovh:gcmagicc-scratch/nc/cmip6/era5spliced/v101",
            evidence_kind="live_rclone_prefix",
            evidence_ref="rclone lsd ovh:gcmagicc-scratch/nc/cmip6/era5spliced/v101",
            recommended_temp_target="skip",
            notes="localresults v101 runs already mirrored to object storage prefixes",
        ),
        InventorySpec(
            rel_path="GCMAGICCoutput/ERA5spliced/localresults/_runmeta",
            backup_now="yes",
            restore_source="local_backup:/data",
            restore_locator=str(DEFAULT_LOCAL_BACKUP_DIR),
            evidence_kind="local_operational_metadata",
            evidence_ref="current localresults/_runmeta tree under data/archive/GCMAGICCoutput/ERA5spliced",
            recommended_temp_target="AUTO_LOCAL_DATA",
            notes="drought / publish run ledgers and logs; preserve locally now and also query ETH",
            needs_eth_prompt=True,
            eth_candidates=(
                "data/site_eth/GCMAGICCoutput/ERA5spliced_localresults/_runmeta",
                "data/site_eth/GCMAGICCoutput/ERA5spliced/_runmeta",
            ),
            eth_search_terms=("ERA5spliced", "_runmeta", "760_droughtn100", "final_manifest.json"),
        ),
        InventorySpec(
            rel_path="GCMAGICCoutput/drought_attribution_758",
            backup_now="no",
            restore_source="r2",
            restore_locator="r2:gcmagicc-public/drought_attribution_758",
            evidence_kind="live_rclone_prefix+publish_manifest",
            evidence_ref="rclone lsd r2:gcmagicc-public/drought_attribution_758 ; local 759_publish_manifest.json files",
            recommended_temp_target="skip",
            notes="aggregate_overview_only; actionable rows below",
            aggregate=True,
            aggregate_children=(
                "GCMAGICCoutput/drought_attribution_758/v100",
                "GCMAGICCoutput/drought_attribution_758/v101",
            ),
        ),
        InventorySpec(
            rel_path="GCMAGICCoutput/drought_attribution_758/v100",
            backup_now="no",
            restore_source="r2",
            restore_locator="r2:gcmagicc-public/drought_attribution_758/v100",
            evidence_kind="live_rclone_prefix+publish_manifest",
            evidence_ref="rclone lsd r2:gcmagicc-public/drought_attribution_758/v100 ; local v100 759_publish_manifest.json",
            recommended_temp_target="skip",
            notes="v100 drought publish tree mirrored to public R2",
        ),
        InventorySpec(
            rel_path="GCMAGICCoutput/drought_attribution_758/v101",
            backup_now="no",
            restore_source="r2",
            restore_locator="r2:gcmagicc-public/drought_attribution_758/v101",
            evidence_kind="live_rclone_prefix+publish_manifest",
            evidence_ref="rclone lsd r2:gcmagicc-public/drought_attribution_758/v101 ; local v101 759_publish_manifest.json",
            recommended_temp_target="skip",
            notes="v101 drought publish tree mirrored to public R2",
        ),
        InventorySpec(
            rel_path="ISIMIP3b",
            backup_now="no",
            restore_source="public",
            restore_locator="https://data.isimip.org/api/v1/datasets/",
            evidence_kind="public_dataset",
            evidence_ref="ISIMIP public dataset API and files.isimip.org",
            recommended_temp_target="skip",
            notes="aggregate_overview_only; actionable row below",
            aggregate=True,
            aggregate_children=("ISIMIP3b/OutputData/water_global",),
        ),
        InventorySpec(
            rel_path="ISIMIP3b/OutputData/water_global",
            backup_now="no",
            restore_source="public",
            restore_locator="https://data.isimip.org/api/v1/datasets/",
            evidence_kind="public_dataset",
            evidence_ref="ISIMIP public dataset API and files.isimip.org",
            recommended_temp_target="skip",
            notes="publicly accessible ISIMIP3b water_global data",
        ),
        InventorySpec(
            rel_path="Koppen",
            backup_now="no",
            restore_source="public",
            restore_locator="https://doi.org/10.6084/m9.figshare.6396959.v1",
            evidence_kind="public_dataset",
            evidence_ref="Beck et al. 2018 Figshare bundle for KG climate classification",
            recommended_temp_target="skip",
            notes="aggregate_overview_only; actionable row below",
            aggregate=True,
            aggregate_children=("Koppen/beck2018",),
        ),
        InventorySpec(
            rel_path="Koppen/beck2018",
            backup_now="no",
            restore_source="public",
            restore_locator="https://doi.org/10.6084/m9.figshare.6396959.v1",
            evidence_kind="public_dataset+local_provenance",
            evidence_ref="local Beck_KG_V1.zip provenance + public Figshare bundle",
            recommended_temp_target="skip",
            notes="publicly re-downloadable archive",
        ),
        InventorySpec(
            rel_path="RISKATLAS",
            backup_now="no",
            restore_source="rebuild_from_retained_source",
            restore_locator="Retain RISKATLAS/T0_source and small metadata; rebuild the rest from the retained source bundle",
            evidence_kind="current_layout+rebuild_policy",
            evidence_ref="current data/archive/RISKATLAS layout and runbook policy",
            recommended_temp_target="skip",
            notes="aggregate_overview_only; child rows below carry the actionable keep/rebuild decisions",
            aggregate=True,
            aggregate_children=(
                "RISKATLAS/T0_source",
                "RISKATLAS/catalogs",
                "RISKATLAS/logs",
                "RISKATLAS/datamodel_benchmark",
                "RISKATLAS/T1_virtual",
                "RISKATLAS/T2_canonical",
                "RISKATLAS/T3_serving",
                "RISKATLAS/T4_export",
                "RISKATLAS/scratch",
            ),
        ),
        InventorySpec(
            rel_path="RISKATLAS/T0_source",
            backup_now="yes",
            restore_source="local_backup:/data",
            restore_locator=str(DEFAULT_LOCAL_BACKUP_DIR),
            evidence_kind="retained_source_bundle",
            evidence_ref="current data/archive/RISKATLAS/T0_source tree",
            recommended_temp_target="AUTO_LOCAL_DATA",
            notes="retain reference assets, but rebuilds still require the CMIP6/ERA5/GCMAGICC climate roots to be rehydrated first",
            needs_eth_prompt=True,
            eth_candidates=(
                "data/site_eth/projects/riskatlas_datamanagement/data/T0_source",
                "data/site_eth/projects/riskatlas_datamanagement/data/source",
                "data/site_eth/projects/riskatlas_dm/data/T0_source",
            ),
            eth_search_terms=("T0_source", "overture_places", "vida_buildings", "wri_gppd"),
        ),
        InventorySpec(
            rel_path="RISKATLAS/catalogs",
            backup_now="yes",
            restore_source="local_backup:/data",
            restore_locator=str(DEFAULT_LOCAL_BACKUP_DIR),
            evidence_kind="small_operational_metadata",
            evidence_ref="current data/archive/RISKATLAS/catalogs tree",
            recommended_temp_target="AUTO_LOCAL_DATA",
            notes="tiny STAC catalog snapshot; preserve for convenience",
            needs_eth_prompt=True,
            eth_candidates=(
                "data/site_eth/projects/riskatlas_datamanagement/data/catalogs",
                "data/site_eth/projects/riskatlas_datamanagement/data/RISKATLAS/catalogs",
            ),
            eth_search_terms=("catalogs", "stac", "catalog.json", "_summary.json"),
        ),
        InventorySpec(
            rel_path="RISKATLAS/logs",
            backup_now="yes",
            restore_source="local_backup:/data",
            restore_locator=str(DEFAULT_LOCAL_BACKUP_DIR),
            evidence_kind="small_operational_metadata",
            evidence_ref="current data/archive/RISKATLAS/logs tree",
            recommended_temp_target="AUTO_LOCAL_DATA",
            notes="small ingest/download logs; preserve for operator context",
            needs_eth_prompt=True,
            eth_candidates=(
                "data/site_eth/projects/riskatlas_datamanagement/logs",
                "data/site_eth/projects/riskatlas_datamanagement/data/logs",
            ),
            eth_search_terms=("download_caravan.log", "ingest_isimip3b", "download_koppen.log"),
        ),
        InventorySpec(
            rel_path="RISKATLAS/datamodel_benchmark",
            backup_now="no",
            restore_source="rebuild_from_retained_source",
            restore_locator="Rebuild from RISKATLAS/T0_source via benchmark pipeline after the RAID rebuild",
            evidence_kind="current_tree+symlinked_sources",
            evidence_ref="current datamodel_benchmark tree with source-linked inputs",
            recommended_temp_target="skip",
            notes="benchmark outputs can be regenerated only after the CMIP6/ERA5/GCMAGICC climate roots are rehydrated; T0_source alone is not sufficient",
        ),
        InventorySpec(
            rel_path="RISKATLAS/T1_virtual",
            backup_now="no",
            restore_source="rebuild_from_retained_source",
            restore_locator="Rebuild from RISKATLAS/T0_source via risk-atlas ingestion pipeline",
            evidence_kind="pipeline_layout",
            evidence_ref="current risk-atlas tier layout",
            recommended_temp_target="skip",
            notes="re-materialise only after the CMIP6/ERA5/GCMAGICC climate roots are available again",
        ),
        InventorySpec(
            rel_path="RISKATLAS/T2_canonical",
            backup_now="no",
            restore_source="rebuild_from_retained_source",
            restore_locator="Rebuild from RISKATLAS/T0_source via risk-atlas ingestion pipeline",
            evidence_kind="pipeline_layout",
            evidence_ref="current risk-atlas tier layout",
            recommended_temp_target="skip",
            notes="canonical derived tier; rebuild only after the climate inputs are rehydrated",
        ),
        InventorySpec(
            rel_path="RISKATLAS/T3_serving",
            backup_now="no",
            restore_source="rebuild_from_retained_source",
            restore_locator="Rebuild from RISKATLAS/T0_source via risk-atlas ingestion pipeline",
            evidence_kind="pipeline_layout",
            evidence_ref="current risk-atlas tier layout",
            recommended_temp_target="skip",
            notes="serving/export derivatives come after climate rehydrate and risk-atlas rebuild steps",
        ),
        InventorySpec(
            rel_path="RISKATLAS/T4_export",
            backup_now="no",
            restore_source="rebuild_from_retained_source",
            restore_locator="Rebuild from retained source if needed; currently empty placeholder",
            evidence_kind="pipeline_layout",
            evidence_ref="current risk-atlas tier layout",
            recommended_temp_target="skip",
            notes="empty export placeholder",
        ),
        InventorySpec(
            rel_path="RISKATLAS/scratch",
            backup_now="no",
            restore_source="rebuild_from_retained_source",
            restore_locator="Recreate scratch area after RAID rebuild; currently empty",
            evidence_kind="pipeline_layout",
            evidence_ref="current risk-atlas tier layout",
            recommended_temp_target="skip",
            notes="empty scratch placeholder",
        ),
    ]


def _materialize_rows(
    specs: Sequence[InventorySpec],
    *,
    datasets_root: Path,
    size_lookup: Callable[[Path], int],
    local_backup_target: str,
) -> list[InventoryRow]:
    spec_map = {spec.rel_path: spec for spec in specs}
    size_cache: dict[str, int] = {}

    def compute_size(spec: InventorySpec) -> int:
        cached = size_cache.get(spec.rel_path)
        if cached is not None:
            return cached
        abs_path = datasets_root / spec.rel_path
        if not abs_path.exists():
            size_cache[spec.rel_path] = 0
            return 0
        if spec.aggregate and spec.aggregate_children:
            total = 0
            for child_rel in spec.aggregate_children:
                child_spec = spec_map.get(child_rel)
                if child_spec is None:
                    continue
                total += compute_size(child_spec)
            size_cache[spec.rel_path] = total
            return total
        value = size_lookup(abs_path)
        size_cache[spec.rel_path] = value
        return value

    rows: list[InventoryRow] = []
    for spec in specs:
        abs_path = datasets_root / spec.rel_path
        if not abs_path.exists():
            continue
        recommended = spec.recommended_temp_target
        if recommended == "AUTO_LOCAL_DATA":
            recommended = local_backup_target
        rows.append(
            InventoryRow(
                path=spec.rel_path,
                size_bytes=compute_size(spec),
                size_human="",
                backup_now=spec.backup_now,
                restore_source=spec.restore_source,
                restore_locator=spec.restore_locator,
                evidence_kind=spec.evidence_kind,
                evidence_ref=spec.evidence_ref,
                recommended_temp_target=recommended,
                notes=spec.notes,
                aggregate=spec.aggregate,
                needs_eth_prompt=spec.needs_eth_prompt,
                eth_candidates=spec.eth_candidates,
                eth_search_terms=spec.eth_search_terms,
            )
        )
    return [
        InventoryRow(
            path=row.path,
            size_bytes=row.size_bytes,
            size_human=_format_bytes(row.size_bytes),
            backup_now=row.backup_now,
            restore_source=row.restore_source,
            restore_locator=row.restore_locator,
            evidence_kind=row.evidence_kind,
            evidence_ref=row.evidence_ref,
            recommended_temp_target=row.recommended_temp_target,
            notes=row.notes,
            aggregate=row.aggregate,
            needs_eth_prompt=row.needs_eth_prompt,
            eth_candidates=row.eth_candidates,
            eth_search_terms=row.eth_search_terms,
        )
        for row in rows
    ]


def _validate_top_level_coverage(rows: Sequence[InventoryRow], datasets_root: Path) -> list[str]:
    actual = sorted(
        entry.name
        for entry in datasets_root.iterdir()
        if not _looks_ignored(entry.name)
    )
    covered = sorted(row.path for row in rows if "/" not in row.path)
    missing = sorted(set(actual) - set(covered))
    if missing:
        raise RuntimeError(
            "Missing top-level inventory coverage for: " + ", ".join(missing)
        )
    return actual


def _ignored_entries_summary(datasets_root: Path) -> tuple[list[str], int]:
    entries: list[str] = []
    total = 0
    for entry in sorted(datasets_root.iterdir(), key=lambda item: item.name):
        if not _looks_ignored(entry.name):
            continue
        entries.append(entry.name)
        try:
            if entry.is_file():
                total += entry.stat().st_size
            else:
                total += _du_size_bytes(entry)
        except OSError:
            continue
    return entries, total


def render_inventory_tsv(rows: Sequence[InventoryRow]) -> str:
    header = [
        "path",
        "size_bytes",
        "size_human",
        "backup_now",
        "restore_source",
        "restore_locator",
        "evidence_kind",
        "evidence_ref",
        "recommended_temp_target",
        "notes",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    _tsv_escape(row.path),
                    str(row.size_bytes),
                    _tsv_escape(row.size_human),
                    row.backup_now,
                    row.restore_source,
                    _tsv_escape(row.restore_locator),
                    _tsv_escape(row.evidence_kind),
                    _tsv_escape(row.evidence_ref),
                    _tsv_escape(row.recommended_temp_target),
                    _tsv_escape(row.notes),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _actionable_rows(rows: Sequence[InventoryRow]) -> list[InventoryRow]:
    return [row for row in rows if not row.aggregate]


def render_summary_md(
    rows: Sequence[InventoryRow],
    *,
    datasets_root: Path,
    top_level_entries: Sequence[str],
    available_data_bytes: int,
) -> str:
    actionable = _actionable_rows(rows)
    yes_rows = [row for row in actionable if row.backup_now == "yes"]
    no_rows = [row for row in actionable if row.backup_now == "no"]
    local_rows = [
        row
        for row in yes_rows
        if row.restore_source == "local_backup:/data"
    ]
    eth_rows = [row for row in actionable if row.needs_eth_prompt]
    largest = sorted(actionable, key=lambda row: row.size_bytes, reverse=True)[:12]
    skip_candidates = sorted(no_rows, key=lambda row: row.size_bytes, reverse=True)[:12]
    ignored_entries, ignored_total = _ignored_entries_summary(datasets_root)

    def bullet_rows(source_rows: Iterable[InventoryRow]) -> str:
        lines = []
        for row in source_rows:
            lines.append(
                f"- `{row.path}` — {row.size_human} — `backup_now={row.backup_now}` — "
                f"`restore_source={row.restore_source}` — {row.notes or 'no extra notes'}"
            )
        return "\n".join(lines) if lines else "- none"

    lines = [
        "# Fressnapf RAID Wipe Inventory Summary",
        "",
        f"- Generated: `{_utc_now()}`",
        f"- Root: `{datasets_root}`",
        f"- `/data` free at generation time: `{_format_bytes(available_data_bytes)}`",
        f"- Top-level entries covered: {', '.join(f'`{name}`' for name in top_level_entries)}",
        f"- ETH audit prompts prepared: `{len(eth_rows)}` dataset families",
        "",
        "## Restore Order",
        "",
        "1. Restore or rehydrate the core climate inputs first: `CMIP6/ETHFOG`, `ERA5/processed`, `GCMAGICCoutput/CMIP6replicas`, `GCMAGICCoutput/ERA5spliced`, and `GCMAGICCoutput/drought_attribution_758` from their recorded S3/R2/public locators.",
        "2. Restore the local-backup subset from `/data` for the small retained metadata and source bundle rows.",
        "3. Only then rebuild `RISKATLAS/datamodel_benchmark`, `T1_virtual`, `T2_canonical`, and `T3_serving`, because those tiers depend on the climate-input trees and not just `RISKATLAS/T0_source`.",
        "",
        "## Totals by `backup_now`",
        "",
        "| backup_now | rows | total_size |",
        "| --- | ---: | ---: |",
        f"| `no` | {len(no_rows)} | {_format_bytes(sum(row.size_bytes for row in no_rows))} |",
        f"| `yes` | {len(yes_rows)} | {_format_bytes(sum(row.size_bytes for row in yes_rows))} |",
        "",
        "## Largest Actionable Space Consumers",
        "",
        bullet_rows(largest),
        "",
        "## Immediate Skip / Rehydrate Candidates",
        "",
        bullet_rows(skip_candidates),
        "",
        "## Category II Temporary Placement Totals",
        "",
        f"- Total `backup_now=yes`: `{_format_bytes(sum(row.size_bytes for row in yes_rows))}`",
        f"- Total staged to `/data`: `{_format_bytes(sum(row.size_bytes for row in local_rows))}`",
        f"- `/data` headroom check: `{'fits' if sum(row.size_bytes for row in local_rows) < LOCAL_BACKUP_THRESHOLD_BYTES else 'exceeds'}` the 100 GiB soft limit",
        "",
        bullet_rows(local_rows),
        "",
        "## Ignored / Out-of-Scope Entries",
        "",
        f"- Ignored entries at dataset root: {', '.join(f'`{name}`' for name in ignored_entries) if ignored_entries else 'none'}",
        f"- Ignored total size: `{_format_bytes(ignored_total)}`",
        "- `read-16j.*` remains excluded from backup planning per user instruction, but it is physically present at the dataset root.",
        "",
        "## ETH Audit Prompts (Non-blocking)",
        "",
        "- These rows are already scheduled for local `/data` backup; the ETH prompts are only to check for a second copy on ETH, not to unblock the wipe.",
        "",
        bullet_rows(eth_rows),
        "",
        "## Coverage Notes",
        "",
        "- Aggregate overview rows are included in `inventory.tsv` for every top-level dataset family.",
        "- Totals in this summary use only actionable rows, so aggregate parents do not double-count child space.",
        "- Hidden scratch blobs and excluded non-dataset files are omitted from the inventory by design.",
        "",
    ]
    return "\n".join(lines)


def render_eth_prompts_md(rows: Sequence[InventoryRow]) -> str:
    prompt_rows = [row for row in _actionable_rows(rows) if row.needs_eth_prompt]
    lines = [
        "# ETH Audit Prompts",
        "",
        "These prompts are optional ETH-side audits for the retained local-backup rows and operational metadata below.",
        "The current plan does not depend on ETH for these rows, but a second ETH-side copy can still be useful for operator confidence.",
        "",
    ]
    for row in sorted(prompt_rows, key=lambda item: item.path):
        find_cmd = _build_find_command(row.eth_search_terms)
        du_cmd = _build_du_command(row.eth_candidates)
        ls_cmd = _build_ls_command(row.eth_candidates)
        lines.extend(
            [
                f"## `{row.path}`",
                "",
                f"- Local size on `data/archive`: `{row.size_human}`",
                f"- Local classification: `backup_now={row.backup_now}`, `restore_source={row.restore_source}`",
                "- Likely ETH candidate paths:",
                _candidate_block(row.eth_candidates),
                "",
                "```text",
                (
                    f"Please inspect whether the dataset family `{row.path}` already exists on ETH.\n\n"
                    f"Local reference path on gus: `data/archive/{row.path}`\n"
                    f"Local size on gus: {row.size_human}\n\n"
                    "Run these exact commands on the ETH side:\n"
                    f"1. {find_cmd}\n"
                    f"2. {du_cmd}\n"
                    f"3. {ls_cmd}\n\n"
                    "Reply in this exact structure:\n"
                    "- found: yes/no\n"
                    "- best_path: <single best ETH path or none>\n"
                    "- du_sh: <verbatim du output>\n"
                    "- notable_children: <3-10 important child entries>\n"
                    "- recommended_action: keep_eth_as_restore_source / keep_local_backup / both / discard\n"
                    "- notes: <anything surprising>\n"
                ),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def generate_reports(
    *,
    datasets_root: Path = DEFAULT_DATASETS_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    local_backup_dir: Path = DEFAULT_LOCAL_BACKUP_DIR,
    size_lookup: Callable[[Path], int] | None = None,
    available_data_bytes: int | None = None,
) -> dict[str, Path]:
    output_dir = output_dir.expanduser().resolve(strict=False)
    datasets_root = datasets_root.expanduser().resolve(strict=False)
    local_backup_dir = local_backup_dir.expanduser().resolve(strict=False)
    if size_lookup is None:
        size_lookup = _du_size_bytes
    if available_data_bytes is None:
        available_data_bytes = _df_available_bytes(local_backup_dir.parent)

    specs = build_inventory_specs()
    rows = _materialize_rows(
        specs,
        datasets_root=datasets_root,
        size_lookup=size_lookup,
        local_backup_target=str(local_backup_dir),
    )
    top_level_entries = _validate_top_level_coverage(rows, datasets_root)

    actionable = _actionable_rows(rows)
    local_total = sum(
        row.size_bytes
        for row in actionable
        if row.backup_now == "yes" and row.restore_source == "local_backup:/data"
    )
    if local_total >= LOCAL_BACKUP_THRESHOLD_BYTES:
        raise RuntimeError(
            "Local backup set exceeds the 100 GiB soft limit: "
            f"{_format_bytes(local_total)} >= {_format_bytes(LOCAL_BACKUP_THRESHOLD_BYTES)}"
        )

    inventory_path = output_dir / "inventory.tsv"
    summary_path = output_dir / "summary.md"
    prompts_path = output_dir / "eth_prompts.md"

    _write_text_atomic(inventory_path, render_inventory_tsv(rows))
    _write_text_atomic(
        summary_path,
        render_summary_md(
            rows,
            datasets_root=datasets_root,
            top_level_entries=top_level_entries,
            available_data_bytes=available_data_bytes,
        ),
    )
    _write_text_atomic(prompts_path, render_eth_prompts_md(rows))

    return {
        "inventory": inventory_path,
        "summary": summary_path,
        "eth_prompts": prompts_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a wipe inventory for data/archive."
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=DEFAULT_DATASETS_ROOT,
        help="Datasets root to inventory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that will receive inventory.tsv, summary.md, and eth_prompts.md.",
    )
    parser.add_argument(
        "--local-backup-dir",
        type=Path,
        default=DEFAULT_LOCAL_BACKUP_DIR,
        help="Temporary local backup target used for Category II rows.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    paths = generate_reports(
        datasets_root=args.datasets_root,
        output_dir=args.output_dir,
        local_backup_dir=args.local_backup_dir,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


__all__ = [
    "DEFAULT_DATASETS_ROOT",
    "DEFAULT_LOCAL_BACKUP_DIR",
    "DEFAULT_OUTPUT_DIR",
    "InventoryRow",
    "InventorySpec",
    "build_inventory_specs",
    "generate_reports",
    "main",
    "render_eth_prompts_md",
    "render_inventory_tsv",
    "render_summary_md",
]
