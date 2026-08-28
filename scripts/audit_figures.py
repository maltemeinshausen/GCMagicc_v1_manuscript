#!/usr/bin/env python3
"""Validate release figure files and render compact contact sheets for review."""

from __future__ import annotations

import argparse
import math
import re
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = ROOT / "figures"


def command(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def audit_pdf(path: Path, render_dir: Path) -> Path:
    info = command("pdfinfo", str(path))
    pages = re.search(r"^Pages:\s+(\d+)$", info, flags=re.MULTILINE)
    if not pages or pages.group(1) != "1":
        raise RuntimeError(f"{path}: expected one page")
    fonts = command("pdffonts", str(path)).splitlines()[2:]
    if not fonts:
        raise RuntimeError(f"{path}: no fonts reported")
    for line in fonts:
        columns = line.split()
        if len(columns) < 8 or columns[-5] != "yes":
            raise RuntimeError(f"{path}: non-embedded font: {line}")
    destination = render_dir / (path.relative_to(FIGURE_ROOT).as_posix().replace("/", "__") + ".png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-scale-to", "800", str(path), str(destination.with_suffix(""))],
        check=True,
        capture_output=True,
    )
    return destination


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(24)
    if signature[:8] != b"\x89PNG\r\n\x1a\n" or signature[12:16] != b"IHDR":
        raise RuntimeError(f"{path}: invalid PNG signature")
    return struct.unpack(">II", signature[16:24])


def write_contact_sheets(images: list[tuple[Path, Path]], output_dir: Path, prefix: str) -> list[Path]:
    sheets: list[Path] = []
    per_sheet = 12
    columns = 3
    for sheet_index, start in enumerate(range(0, len(images), per_sheet), start=1):
        batch = images[start:start + per_sheet]
        rows = math.ceil(len(batch) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(15, rows * 4.8), squeeze=False)
        for axis, item in zip(axes.flat, batch):
            source, raster = item
            axis.imshow(mpimg.imread(raster))
            axis.set_title(source.relative_to(ROOT).as_posix(), fontsize=7, wrap=True)
            axis.axis("off")
        for axis in axes.flat[len(batch):]:
            axis.axis("off")
        figure.tight_layout()
        destination = output_dir / f"{prefix}_{sheet_index:02d}.png"
        figure.savefig(destination, dpi=140, bbox_inches="tight")
        plt.close(figure)
        sheets.append(destination)
    return sheets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(tempfile.gettempdir()) / "gcmagicc-figure-audit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    render_dir = args.output / "pdf-renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(FIGURE_ROOT.rglob("*.pdf"))
    pngs = sorted(FIGURE_ROOT.rglob("*.png"))
    svgs = sorted(FIGURE_ROOT.rglob("*.svg"))
    rendered_pdfs = [(path, audit_pdf(path, render_dir)) for path in pdfs]
    for path in pngs:
        width, height = png_dimensions(path)
        if width < 300 or height < 300:
            raise RuntimeError(f"{path}: unexpectedly small PNG ({width}x{height})")
    for path in svgs:
        ET.parse(path)

    sheets = write_contact_sheets(rendered_pdfs, args.output, "pdf_contact_sheet")
    sheets += write_contact_sheets([(path, path) for path in pngs], args.output, "png_contact_sheet")
    print(f"validated {len(pdfs)} PDFs, {len(pngs)} PNGs, and {len(svgs)} SVGs")
    for sheet in sheets:
        print(sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
