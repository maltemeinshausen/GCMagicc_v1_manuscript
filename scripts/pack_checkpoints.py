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
import hashlib
import json
import sys
import tarfile
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="internal snapshot root containing model_NxlversA5/ and model_NthreeversT1/")
    parser.add_argument("--out", default=ROOT / "dist", type=Path)
    parser.add_argument("--verify-only", action="store_true",
                        help="check hashes and report, without writing tarballs")
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

        if failures or args.verify_only:
            continue

        archive = args.out / obj["archive_file"]
        # mtime/uid/gid are normalised so the tarball is byte-reproducible.
        with tarfile.open(archive, "w:gz") as tar:
            for src, arcname in sorted(members, key=lambda pair: pair[1]):
                info = tar.gettarinfo(str(src), arcname=arcname)
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with src.open("rb") as handle:
                    tar.addfile(info, handle)
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
