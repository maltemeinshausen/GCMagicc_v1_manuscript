#!/usr/bin/env python3
"""Stage and verify the external objects for the Zenodo web upload."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import zipfile
from pathlib import Path
from pathlib import PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "external_data_manifest.json"
ZENODO_LIMIT_BYTES = 50_000_000_000
ZENODO_LIMIT_FILES = 100


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def stage(source: Path, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size == source.stat().st_size and sha256(destination) == sha256(source):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    temporary.unlink()
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def verify_sqlite(path: Path) -> None:
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        result = connection.execute("PRAGMA integrity_check").fetchone()
    finally:
        connection.close()
    if result != ("ok",):
        raise RuntimeError(f"{path}: SQLite integrity check failed: {result}")


def verify_csv(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or len(header) != len(set(header)):
            raise RuntimeError(f"{path}: missing or duplicate CSV columns")
        width = len(header)
        for line_number, row in enumerate(reader, start=2):
            if len(row) != width:
                raise RuntimeError(
                    f"{path}:{line_number}: expected {width} columns, found {len(row)}"
                )


def verify_npz(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if not names or len(names) != len(set(names)):
            raise RuntimeError(f"{path}: empty archive or duplicate members")
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if name.is_absolute() or ".." in name.parts or member.is_dir():
                raise RuntimeError(f"{path}: unsafe member {member.filename}")
            if stat.S_IFMT(mode) not in {0, stat.S_IFREG}:
                raise RuntimeError(f"{path}: non-regular member {member.filename}")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"{path}: CRC failure in {bad_member}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-metrics", required=True, type=Path)
    parser.add_argument("--current-metrics", required=True, type=Path)
    parser.add_argument("--energy-distance", required=True, type=Path)
    parser.add_argument("--xs-regenerated", required=True, type=Path)
    parser.add_argument("--xs-member-draws", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "zenodo")
    args = parser.parse_args()

    local_sources = {
        "gcmagicc-checkpoints": ROOT / "dist/gcmagicc-checkpoints.tar.gz",
        "gcmagicc-ce-checkpoints": ROOT / "dist/gcmagicc-ce-checkpoints.tar.gz",
        "nic-aer-release-bundle": ROOT / "misc/zenodo/NIC-AER.zip",
        "gcmagicc-pm-bundle": ROOT / "misc/zenodo/NIC-PM.zip",
        "nic-res-release-bundle": ROOT / "misc/zenodo/NIC-RES.zip",
        "gcmagicc-xs-bundle": ROOT / "misc/zenodo/NIC-XS.zip",
        "validation-metrics-sqlite": args.historical_metrics,
        "validation-diagnostics-metrics-sqlite-v20260821": args.current_metrics,
        "validation-diagnostics-edist-sqlite-v20260821": args.energy_distance,
        "gcmagicc-xs-regenerated-intermediate-v20260713": args.xs_regenerated,
        "gcmagicc-xs-member-draws-v20260713": args.xs_member_draws,
    }
    objects = {
        item["id"]: item
        for item in json.loads(MANIFEST.read_text(encoding="utf-8"))["objects"]
        if item.get("deposit", True)
    }
    if set(local_sources) != set(objects):
        raise RuntimeError("staging source IDs do not match the external-data manifest")

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    total = 0
    for object_id, source in local_sources.items():
        record = objects[object_id]
        if not source.is_file():
            raise RuntimeError(f"{object_id}: missing source {source}")
        if source.stat().st_size != record["bytes"] or sha256(source) != record["sha256"]:
            raise RuntimeError(f"{object_id}: source does not match the external-data manifest")
        if object_id in {
            "validation-metrics-sqlite",
            "validation-diagnostics-metrics-sqlite-v20260821",
            "validation-diagnostics-edist-sqlite-v20260821",
        }:
            verify_sqlite(source)
        elif object_id in {
            "gcmagicc-xs-regenerated-intermediate-v20260713",
        }:
            verify_csv(source)
        elif object_id == "gcmagicc-xs-member-draws-v20260713":
            verify_npz(source)
        destination = args.out / record["deposit_file"]
        stage(source, destination)
        rows.append(f"{record['sha256']}  {record['deposit_file']}\n")
        total += record["bytes"]

    checksum = args.out / "SHA256SUMS.txt"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.out, delete=False) as handle:
        handle.write("".join(rows))
        temporary_checksum = Path(handle.name)
    temporary_checksum.replace(checksum)
    file_count = len(rows) + 1
    total += checksum.stat().st_size
    expected_files = {item["deposit_file"] for item in objects.values()} | {checksum.name}
    actual_files = {path.name for path in args.out.iterdir() if path.is_file()}
    if actual_files != expected_files:
        unexpected = sorted(actual_files - expected_files)
        missing = sorted(expected_files - actual_files)
        raise RuntimeError(f"staging directory mismatch; unexpected={unexpected}, missing={missing}")
    if file_count > ZENODO_LIMIT_FILES or total > ZENODO_LIMIT_BYTES:
        raise RuntimeError(f"Zenodo staging exceeds a release limit: {file_count} files, {total} bytes")
    print(f"staged {file_count} files ({total} bytes) in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
