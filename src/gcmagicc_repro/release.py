# SPDX-License-Identifier: Apache-2.0
"""Fetch, verify, smoke, and figure-dispatch operations."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

from .kernels import area_weighted_mean, corrected_tas_predictor, modified_hargreaves_monthly_mm, moving_block_resample


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_MANIFEST = ROOT / "data" / "external_data_manifest.json"

FIGURES = {
    "workflow": ("1120_gcmagicc_workflow_schematic.py", []),
    "turkiye": ("1100_turkiye_regional_application.py", []),
    "drought": ("1040_drought_attribution_example_country.py", []),
    "drought-common-protocol": ("1090_drought_common_protocol.py", []),
    "drought-main-figure": ("1130_drought_attribution_synthesis.py", []),
    "resolution": ("1050_resolution_sensitivity.py", ["gcmagicc-checkpoints"]),
    "emergent": ("1060_emergent_constraints.py", []),
    "aerosol": ("1070_aerosol_pattern.py", ["gcmagicc-checkpoints"]),
    "xs": ("1080_gcmagicc_xs_prediction_skill.py", []),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_external() -> dict:
    return json.loads(EXTERNAL_MANIFEST.read_text(encoding="utf-8"))


def _csv_structure_failures(path: Path) -> list[str]:
    """Return structural CSV errors without interpreting scientific values."""
    failures: list[str] = []
    try:
        display_path = path.relative_to(ROOT)
    except ValueError:
        display_path = path
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, None)
            if not header:
                return [f"empty CSV: {display_path}"]
            width = len(header)
            for line_number, row in enumerate(reader, start=2):
                if len(row) != width:
                    failures.append(
                        f"malformed CSV row in {display_path}:{line_number}: "
                        f"expected {width} columns, found {len(row)}"
                    )
    except (csv.Error, UnicodeDecodeError) as error:
        failures.append(f"invalid CSV {display_path}: {error}")
    return failures


def _safe_members(
    tar: "tarfile.TarFile",
    root: Path,
    subtree: Path | None = None,
    expected: set[str] | None = None,
):
    """Return regular-file members confined to the declared release subtree."""
    root = root.resolve()
    confine = (subtree or root).resolve()
    seen: set[str] = set()
    safe: list[tarfile.TarInfo] = []
    for member in tar.getmembers():
        archive_path = PurePosixPath(member.name)
        if archive_path.is_absolute() or ".." in archive_path.parts:
            raise RuntimeError(f"refusing unsafe tar member {member.name!r}")
        resolved = (root / Path(*archive_path.parts)).resolve()
        if not resolved.is_relative_to(confine):
            raise RuntimeError(f"refusing out-of-tree tar member {member.name!r}")
        if not member.isfile():
            raise RuntimeError(f"refusing non-regular tar member {member.name!r}")
        if member.name in seen:
            raise RuntimeError(f"refusing duplicate tar member {member.name!r}")
        if expected is not None and member.name not in expected:
            raise RuntimeError(f"refusing undeclared tar member {member.name!r}")
        seen.add(member.name)
        safe.append(member)
    if expected is not None and seen != expected:
        missing = sorted(expected - seen)
        raise RuntimeError(f"archive is missing {len(missing)} declared member(s): {missing[:5]}")
    return safe


def extract_archive(obj: dict, archive: Path) -> None:
    """Extract a checkpoint tarball and verify every file against its recorded SHA-256.

    data/checkpoint_manifest.json is the authority: a checkpoint that does not match its
    recorded hash is not the checkpoint the paper used, so a mismatch is a hard error.
    """
    # Archive member names are repository-root-relative (see scripts/pack_checkpoints.py),
    # so extraction happens at ROOT. extract_to declares the subtree the archive is
    # allowed to write into, and is enforced rather than used as the extraction directory.
    manifest_path = ROOT / obj.get("per_file_manifest", "data/checkpoint_manifest.json")
    if not manifest_path.is_file():
        raise RuntimeError(f"missing checkpoint manifest {manifest_path}")
    per_file_object_id = obj.get("per_file_object_id", obj["id"])
    entries = next(
        (o["files"] for o in json.loads(manifest_path.read_text(encoding="utf-8"))["objects"]
         if o["id"] == per_file_object_id),
        [],
    )
    if not entries:
        raise RuntimeError(f"no per-file checkpoint records for {obj['id']}")
    expected = {entry["path"] for entry in entries}
    subtree = (ROOT / obj["extract_to"]).resolve()
    subtree.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(ROOT, members=_safe_members(tar, ROOT, subtree, expected))

    bad: list[str] = []
    for entry in entries:
        path = ROOT / entry["path"]
        if not path.is_file():
            bad.append(f"missing {entry['path']}")
        elif path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            bad.append(f"hash or size mismatch {entry['path']}")
    if bad:
        raise RuntimeError(
            f"{obj['id']}: {len(bad)} extracted file(s) failed verification: " + "; ".join(bad[:5])
        )
    print(f"  {obj['id']}: extracted and verified {len(entries)} files into {obj['extract_to']}")
    archive.unlink()


def fetch() -> int:
    pending: list[str] = []
    for obj in load_external()["objects"]:
        if obj["status"] == "not-redistributed":
            continue
        if obj["status"] != "published":
            pending.append(f"{obj['id']} ({obj['status']})")
            continue
        destination = ROOT / obj["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temp = Path(handle.name)
        try:
            urllib.request.urlretrieve(obj["url"], temp)
            if temp.stat().st_size != obj["bytes"] or sha256(temp) != obj["sha256"]:
                raise RuntimeError(f"checksum or size mismatch for {obj['id']}")
            temp.replace(destination)
        finally:
            if temp.exists():
                temp.unlink()
        if obj.get("extract_to"):
            extract_archive(obj, destination)
    if pending:
        print("External release objects still pending:")
        for item in pending:
            print(f"  {item}")
        return 2
    return 0


def smoke() -> int:
    cfg = json.loads((ROOT / "configs" / "smoke.json").read_text(encoding="utf-8"))
    h = cfg["hargreaves"]
    pet = modified_hargreaves_monthly_mm(
        h["temperature_c"], h["temperature_min_c"], h["temperature_max_c"], h["rsds_mj_m2_day"], h["days"]
    )
    corrected = corrected_tas_predictor([1.0, 1.0, 1.0], cfg["two_pass_delta_c"])
    weighted = area_weighted_mean([-1.0, -2.0, -3.0], [0.0, 45.0, 70.0])
    boot = moving_block_resample(list(range(20)), block=5, seed=cfg["seed"])
    result = {
        "seed": cfg["seed"],
        "modified_hargreaves_monthly_mm": round(pet, 10),
        "corrected_tas_predictor": corrected,
        "area_weighted_mean": round(weighted, 10),
        "bootstrap_prefix": boot[:10],
        "december_samples_per_year": 1,
        "bootstrap_replicates_release_protocol": 10000,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def reproduce(figure: str, dry_run: bool, extra: list[str]) -> int:
    if figure not in FIGURES:
        raise ValueError(f"unknown figure {figure!r}; choose from {', '.join(sorted(FIGURES))}")
    script_name, required = FIGURES[figure]
    objects = {item["id"]: item for item in load_external()["objects"]}
    blocked = [item for item in required if objects[item]["status"] != "published"]
    script = ROOT / "src" / "gcmagicc_eval" / "workflows" / script_name
    command = [sys.executable, str(script), *extra]
    print(" ".join(command))
    if blocked:
        print("Blocked by unpublished external objects: " + ", ".join(blocked))
        return 2
    if dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def verify() -> int:
    failures: list[str] = []
    forbidden_machine_paths = (
        "/data/projects/",
        "/scratch2/",
        "/mnt/fressnapf/",
        "/data/scratch/",
        "/r/scratch/",
        "/home/",
    )
    placeholder_pattern = re.compile(
        r"(?:Review" + r"Placeholder|\bTO" + r"DO\b|\bTB" + r"D\b|10\.0000/)"
    )
    listed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    public_paths = [ROOT / item.decode("utf-8") for item in listed.split(b"\0") if item]
    for path in public_paths:
        if not path.is_file():
            continue
        if path.stat().st_size > 50 * 1024 * 1024:
            failures.append(f"file exceeds 50 MiB: {path.relative_to(ROOT)}")
        if path.suffix.lower() == ".csv":
            failures.extend(_csv_structure_failures(path))
        if (
            path.suffix.lower() in {"", ".cff", ".csv", ".json", ".md", ".py", ".sh", ".tex", ".toml", ".txt", ".yaml", ".yml"}
            and path.resolve() != Path(__file__).resolve()
        ):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
            ):
                # Generated JSON can contain multi-megabyte encoded payloads. Such lines
                # are checksummed as data rather than interpreted as release prose.
                if len(line) > 20_000:
                    continue
                for prefix in forbidden_machine_paths:
                    if prefix in line:
                        failures.append(
                            f"machine-specific path in {path.relative_to(ROOT)}:{line_number}: {prefix}"
                        )
                match = placeholder_pattern.search(line)
                if match:
                    failures.append(
                        f"placeholder token in {path.relative_to(ROOT)}:{line_number}: {match.group(0)}"
                    )
    for required in ("LICENSES/Apache-2.0.txt", "LICENSES/CC-BY-4.0.txt", "NOTICE", ".reuse/dep5"):
        if not (ROOT / required).is_file():
            failures.append(f"missing {required}")
    manifest = ROOT / "provenance" / "manifest.csv"
    if not manifest.is_file():
        failures.append("missing provenance/manifest.csv")
    else:
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                destination = ROOT / row["destination"]
                if not destination.is_file():
                    failures.append(f"missing frozen file {row['destination']}")
                elif sha256(destination) != row["sha256"]:
                    failures.append(f"hash mismatch {row['destination']}")
    forcing = ROOT / "data" / "natural_forcing_ssp245_ar6_run0_1850-2100.csv"
    if not forcing.is_file() or sha256(forcing) != "089372873cc283e8188c94dc9818cd0b75b694e4f38457f2925d516169c3e801":
        failures.append("natural-forcing artifact missing or changed")
    external = load_external()
    if external.get("schema") != "gcmagicc-external-data/v1":
        failures.append("invalid external manifest schema")
    objects = external.get("objects", [])
    ids = [item.get("id") for item in objects]
    if len(ids) != len(set(ids)):
        failures.append("duplicate external object id")
    deposit_files = [item.get("deposit_file") for item in objects]
    destinations = [item.get("destination") for item in objects]
    if len(deposit_files) != len(set(deposit_files)):
        failures.append("duplicate external deposit filename")
    if len(destinations) != len(set(destinations)):
        failures.append("duplicate external fetch destination")
    objects_by_deposit = {item.get("deposit_file"): item for item in objects}
    any_published = any(item.get("status") == "published" for item in objects)
    if any_published:
        if not external.get("zenodo_record") or not external.get("record_doi"):
            failures.append("published objects require Zenodo record and DOI metadata")
    for obj in objects:
        obj_id = obj.get("id", "<missing-id>")
        if not isinstance(obj.get("bytes"), int) or obj["bytes"] <= 0:
            failures.append(f"external object has invalid byte size: {obj_id}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(obj.get("sha256", ""))):
            failures.append(f"external object has invalid SHA-256: {obj_id}")
        deposit_file = str(obj.get("deposit_file", ""))
        if not deposit_file or PurePosixPath(deposit_file).name != deposit_file:
            failures.append(f"external object has unsafe deposit filename: {obj_id}")
        destination = PurePosixPath(str(obj.get("destination", "")))
        if not destination.parts or destination.is_absolute() or ".." in destination.parts:
            failures.append(f"external object has unsafe fetch destination: {obj_id}")
        licenses = obj.get("licenses", [obj.get("license")])
        if not licenses or any(item not in {"Apache-2.0", "CC-BY-4.0"} for item in licenses):
            failures.append(f"external object has invalid licence metadata: {obj_id}")
        status = obj.get("status")
        if status == "not-redistributed":
            if obj.get("deposit") is not False:
                failures.append(f"non-redistributed object must set deposit=false: {obj_id}")
            if obj.get("url") is not None:
                failures.append(f"non-redistributed object must not have a URL: {obj_id}")
            if not obj.get("note"):
                failures.append(f"non-redistributed object requires an explanatory note: {obj_id}")
        elif status != "published":
            failures.append(f"external object is not published: {obj_id} ({obj.get('status')})")
        elif not str(obj.get("url", "")).startswith("https://"):
            failures.append(f"external object has no HTTPS URL: {obj_id}")

    registry = ROOT / "provenance" / "figure_registry.csv"
    if registry.is_file():
        known_external = set(ids)
        registered_roles: set[tuple[str, str]] = set()
        with registry.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                role = (row["semantic_role"], row["manuscript_role"])
                if role in registered_roles:
                    failures.append(f"duplicate figure-registry role {role[0]} ({role[1]})")
                registered_roles.add(role)
                artifact = ROOT / row["artifact_path"]
                if not artifact.exists():
                    failures.append(f"missing figure-registry artifact {row['artifact_path']}")
                prepared = row["prepared_data"]
                if (
                    prepared
                    and prepared not in {"release-native", "metadata-only"}
                    and not (ROOT / prepared).exists()
                ):
                    failures.append(f"missing prepared data {prepared} for {row['semantic_role']}")
                for dependency in filter(None, row["raw_dependency"].split(";")):
                    if dependency not in known_external:
                        failures.append(
                            f"unknown external dependency {dependency} for {row['semantic_role']}"
                        )

    for relative in (
        "misc/NIC-AER/data/EXTERNAL_MANIFEST.csv",
        "misc/NIC-PM/data/EXTERNAL_MANIFEST.csv",
        "misc/NIC-RES/data/EXTERNAL_MANIFEST.csv",
        "misc/NIC-XS/data/EXTERNAL_MANIFEST.csv",
    ):
        path = ROOT / relative
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            expected_fields = {"name", "role", "byte_size", "sha256", "public_url", "status"}
            if set(reader.fieldnames or []) != expected_fields:
                failures.append(f"invalid columns in {relative}")
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    failures.append(f"malformed CSV row in {relative}:{line_number}")
                    continue
                if row["byte_size"] or row["sha256"] or row["public_url"]:
                    obj = objects_by_deposit.get(row["name"])
                    if obj is None:
                        failures.append(
                            f"undeclared external file in {relative}:{line_number}: {row['name']}"
                        )
                        continue
                    if row["byte_size"] != str(obj["bytes"]):
                        failures.append(f"external byte-size mismatch in {relative}:{line_number}")
                    if row["sha256"] != obj["sha256"]:
                        failures.append(f"external SHA-256 mismatch in {relative}:{line_number}")
                    if obj.get("status") == "published":
                        if row["public_url"] != obj.get("url"):
                            failures.append(f"external URL mismatch in {relative}:{line_number}")
                        if row["status"] != "published":
                            failures.append(f"external status mismatch in {relative}:{line_number}")
                    elif obj.get("status") == "not-redistributed":
                        if row["public_url"]:
                            failures.append(f"non-redistributed URL in {relative}:{line_number}")
                        if not row["status"].startswith("not redistributed"):
                            failures.append(f"external status mismatch in {relative}:{line_number}")
    if failures:
        print("VERIFY: FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("VERIFY: PASS")
    return 0
