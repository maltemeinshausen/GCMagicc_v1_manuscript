# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from gcmagicc_repro.release import _csv_structure_failures, _safe_members


def archive_with(member: tarfile.TarInfo) -> tarfile.TarFile:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        if member.isfile():
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
        else:
            archive.addfile(member)
    payload.seek(0)
    return tarfile.open(fileobj=payload, mode="r")


def test_safe_members_accepts_only_declared_regular_files(tmp_path: Path) -> None:
    member = tarfile.TarInfo("release/checkpoint.pt")
    with archive_with(member) as archive:
        safe = _safe_members(
            archive,
            tmp_path,
            tmp_path / "release",
            {"release/checkpoint.pt"},
        )
    assert [item.name for item in safe] == ["release/checkpoint.pt"]


@pytest.mark.parametrize("name", ["../escape", "release/../../escape", "/absolute"])
def test_safe_members_rejects_path_traversal(tmp_path: Path, name: str) -> None:
    with archive_with(tarfile.TarInfo(name)) as archive:
        with pytest.raises(RuntimeError):
            _safe_members(archive, tmp_path, tmp_path / "release", {name})


def test_safe_members_rejects_links_and_undeclared_members(tmp_path: Path) -> None:
    link = tarfile.TarInfo("release/link")
    link.type = tarfile.SYMTYPE
    link.linkname = "checkpoint.pt"
    with archive_with(link) as archive:
        with pytest.raises(RuntimeError):
            _safe_members(archive, tmp_path, tmp_path / "release", {"release/link"})

    with archive_with(tarfile.TarInfo("release/extra.pt")) as archive:
        with pytest.raises(RuntimeError):
            _safe_members(archive, tmp_path, tmp_path / "release", {"release/checkpoint.pt"})


def test_csv_structure_check_rejects_malformed_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "malformed.csv"
    csv_path.write_text("a,b\n1,2\n3,4,5\n", encoding="utf-8")
    failures = _csv_structure_failures(csv_path)
    assert len(failures) == 1
    assert "expected 2 columns, found 3" in failures[0]
