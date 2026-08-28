#!/usr/bin/env python3
"""Pack the GCMagicc / GCMagicc-CE trained checkpoints for public deposit.

Reads data/checkpoint_manifest.json, verifies every source file against its recorded
SHA-256, and writes one deterministic .tar.gz per public model. Refuses to pack if any
file is missing or its hash does not match, so a tarball can never contain a checkpoint
other than the one the paper used.

Usage:
    python scripts/pack_checkpoints.py --source /path/to/gcm_firefly_data --out dist/

The --source directory is the internal snapshot root and is NOT part of this release;
this script exists so the deposit is reproducible by whoever holds those files.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "checkpoint_manifest.json"

# destination path in the release  ->  path within the internal snapshot
SNAPSHOT_SUBDIR = {"GCMagicc": "model_NxlversA5", "GCMagicc-CE": "model_NthreeversT1"}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(source_root: Path, public_model: str, entry: dict) -> Path:
    snap = source_root / SNAPSHOT_SUBDIR[public_model]
    if entry["role"] == "normalization":
        return snap / Path(entry["path"]).name
    return snap / "modelsA" / entry["variant"] / Path(entry["path"]).name


def verify_archive(archive: Path, entries: list[dict]) -> None:
    """Verify deterministic metadata, membership, sizes, and payload hashes."""
    if not archive.is_file():
        raise RuntimeError(f"missing archive {archive}")
    with archive.open("rb") as handle:
        header = handle.read(10)
    if len(header) != 10 or header[:3] != b"\x1f\x8b\x08":
        raise RuntimeError(f"{archive}: invalid gzip header")
    if header[4:8] != b"\0\0\0\0":
        raise RuntimeError(f"{archive}: gzip mtime is not zero")
    # FLG must not advertise an original filename or comment, both of which would
    # make the byte stream depend on the temporary build path.
    if header[3] & 0x18:
        raise RuntimeError(f"{archive}: gzip header contains filename/comment metadata")

    expected = {entry["path"]: entry for entry in entries}
    seen: set[str] = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.name in seen:
                raise RuntimeError(f"{archive}: duplicate member {member.name}")
            if member.name not in expected:
                raise RuntimeError(f"{archive}: unexpected member {member.name}")
            if not member.isfile():
                raise RuntimeError(f"{archive}: non-regular member {member.name}")
            if (
                member.mtime != 0
                or member.uid != 0
                or member.gid != 0
                or member.uname
                or member.gname
                or member.pax_headers
            ):
                raise RuntimeError(f"{archive}: non-deterministic metadata for {member.name}")
            entry = expected[member.name]
            if member.size != entry["bytes"]:
                raise RuntimeError(f"{archive}: size mismatch for {member.name}")
            extracted = tar.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"{archive}: cannot read {member.name}")
            digest = hashlib.sha256()
            for block in iter(lambda: extracted.read(1 << 20), b""):
                digest.update(block)
            if digest.hexdigest() != entry["sha256"]:
                raise RuntimeError(f"{archive}: SHA-256 mismatch for {member.name}")
            seen.add(member.name)
    if seen != set(expected):
        missing = sorted(set(expected) - seen)
        raise RuntimeError(f"{archive}: missing {len(missing)} member(s): {missing[:5]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="internal snapshot root containing model_NxlversA5/ and model_NthreeversT1/")
    parser.add_argument("--out", default=ROOT / "dist", type=Path)
    parser.add_argument("--verify-only", action="store_true",
                        help="check hashes and report, without writing tarballs")
    parser.add_argument("--verify-archives", action="store_true",
                        help="also verify existing archive members, hashes, and deterministic metadata")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    results: list[dict] = []

    for obj in manifest["objects"]:
        model = obj["public_model"]
        print(f"== {obj['id']}  ({obj['file_count']} files, "
              f"{obj['total_bytes'] / 1073741824:.2f} GB)")
        members: list[tuple[Path, str]] = []
        for entry in obj["files"]:
            src = source_path(args.source, model, entry)
            if not src.is_file():
                failures.append(f"{obj['id']}: missing {src}")
                continue
            if src.stat().st_size != entry["bytes"]:
                failures.append(f"{obj['id']}: size mismatch {src}")
                continue
            actual = sha256(src)
            if actual != entry["sha256"]:
                failures.append(f"{obj['id']}: sha256 mismatch {src}")
                continue
            members.append((src, entry["path"]))
        print(f"   verified {len(members)}/{len(obj['files'])} files")

        archive = args.out / obj["archive_file"]
        if not failures and args.verify_archives:
            verify_archive(archive, obj["files"])
            print(f"   verified archive {archive}")

        if failures or args.verify_only:
            continue

        # Normalise both the gzip header and each tar member. tarfile's "w:gz"
        # mode otherwise records the build time in the gzip header.
        with tempfile.NamedTemporaryFile(dir=args.out, delete=False) as temporary:
            temp_path = Path(temporary.name)
        try:
            with temp_path.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                        for src, arcname in sorted(members, key=lambda pair: pair[1]):
                            info = tar.gettarinfo(str(src), arcname=arcname)
                            info.mtime = 0
                            info.uid = info.gid = 0
                            info.uname = info.gname = ""
                            info.pax_headers = {}
                            with src.open("rb") as handle:
                                tar.addfile(info, handle)
            temp_path.replace(archive)
        finally:
            temp_path.unlink(missing_ok=True)
        results.append({"id": obj["id"], "archive_file": obj["archive_file"],
                        "tarball_bytes": archive.stat().st_size,
                        "tarball_sha256": sha256(archive)})
        print(f"   wrote {archive}  {archive.stat().st_size / 1073741824:.2f} GB")

    if failures:
        print("\nFAILED - nothing packed:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1

    if results:
        print("\nInsert into data/external_data_manifest.json after deposit:")
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
