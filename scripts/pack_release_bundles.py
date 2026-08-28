#!/usr/bin/env python3
"""Build deterministic, public-safe ZIPs for the four NIC release bundles."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT / "misc"
OUTPUT_ROOT = BUNDLE_ROOT / "zenodo"
BUNDLES = ("NIC-AER", "NIC-PM", "NIC-RES", "NIC-XS")
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def excluded(path: Path) -> bool:
    return (
        path.name == "PROVENANCE_INTERNAL.md"
        or path.name == "SHA256SUMS.txt"
        or path.suffix == ".pyc"
        or "__pycache__" in path.parts
    )


def bundle_files(bundle: Path) -> list[Path]:
    return sorted(
        (path for path in bundle.rglob("*") if path.is_file() and not excluded(path)),
        key=lambda path: path.relative_to(bundle).as_posix(),
    )


def checksum_text(bundle: Path, files: list[Path]) -> str:
    return "".join(f"{sha256(path)}  {path.relative_to(bundle).as_posix()}\n" for path in files)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def zip_info(name: str, mode: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (mode & 0xFFFF) << 16
    return info


def build_zip(bundle: Path, destination: Path) -> None:
    files = bundle_files(bundle)
    checksum = bundle / "SHA256SUMS.txt"
    files_with_manifest = sorted([*files, checksum], key=lambda path: path.relative_to(bundle).as_posix())
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files_with_manifest:
                relative = path.relative_to(bundle).as_posix()
                mode = 0o755 if os.access(path, os.X_OK) else 0o644
                archive.writestr(zip_info(f"{bundle.name}/{relative}", mode), path.read_bytes())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def verify_zip(bundle: Path, archive_path: Path) -> None:
    expected = {
        f"{bundle.name}/{path.relative_to(bundle).as_posix()}"
        for path in [*bundle_files(bundle), bundle / "SHA256SUMS.txt"]
    }
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError(f"{archive_path}: duplicate members")
        if set(names) != expected:
            raise RuntimeError(f"{archive_path}: member set does not match sanitized bundle")
        for name in names:
            if name.endswith("PROVENANCE_INTERNAL.md") or "__pycache__" in name or name.endswith(".pyc"):
                raise RuntimeError(f"{archive_path}: private or cached member {name}")
        prefix = f"{bundle.name}/"
        checksums = archive.read(prefix + "SHA256SUMS.txt").decode("utf-8").splitlines()
        for line in checksums:
            expected_hash, relative = line.split("  ", 1)
            actual = hashlib.sha256(archive.read(prefix + relative)).hexdigest()
            if actual != expected_hash:
                raise RuntimeError(f"{archive_path}: checksum mismatch for {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-manifests", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    release_rows: list[str] = []
    for name in BUNDLES:
        bundle = BUNDLE_ROOT / name
        manifest = bundle / "SHA256SUMS.txt"
        rendered = checksum_text(bundle, bundle_files(bundle))
        if args.update_manifests:
            atomic_write(manifest, rendered)
        elif not manifest.is_file() or manifest.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"{manifest}: stale; rerun with --update-manifests")

        archive = OUTPUT_ROOT / f"{name}.zip"
        if not args.verify_only:
            build_zip(bundle, archive)
        verify_zip(bundle, archive)
        release_rows.append(f"{sha256(archive)}  {archive.name}\n")
        print(f"verified {archive.relative_to(ROOT)} ({archive.stat().st_size} bytes)")

    if args.update_manifests:
        atomic_write(OUTPUT_ROOT / "SHA256SUMS.zip.txt", "".join(release_rows))
    elif (OUTPUT_ROOT / "SHA256SUMS.zip.txt").read_text(encoding="utf-8") != "".join(release_rows):
        raise RuntimeError("misc/zenodo/SHA256SUMS.zip.txt is stale")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
