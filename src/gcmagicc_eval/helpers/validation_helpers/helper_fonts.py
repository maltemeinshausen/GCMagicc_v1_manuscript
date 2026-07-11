"""
Matplotlib font helpers for manuscript figures.

These utilities resolve one installed sans-serif family up front so figure
scripts do not repeatedly ask Matplotlib for Arial/Helvetica on machines where
those fonts are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager


DEFAULT_SANS_FONT_CANDIDATES: tuple[str, ...] = (
    "Arial",
    "Helvetica",
    "Nimbus Sans",
    "Liberation Sans",
    "Carlito",
    "DejaVu Sans",
)


@dataclass(frozen=True)
class ResolvedFontFamily:
    family: str
    path: str


def resolve_sans_font_family(
    candidates: Sequence[str] = DEFAULT_SANS_FONT_CANDIDATES,
    *,
    fallback: str = "DejaVu Sans",
) -> ResolvedFontFamily:
    seen: set[str] = set()
    for raw_family in [*candidates, fallback]:
        family = str(raw_family).strip()
        if not family or family in seen:
            continue
        seen.add(family)
        try:
            path = font_manager.findfont(family, fallback_to_default=False)
        except Exception:
            continue
        if path:
            return ResolvedFontFamily(family=family, path=str(path))

    fallback_path = font_manager.findfont(fallback)
    return ResolvedFontFamily(family=str(fallback), path=str(fallback_path))


def apply_sans_font_rcparams(
    *,
    rc_updates: Mapping[str, Any] | None = None,
    candidates: Sequence[str] = DEFAULT_SANS_FONT_CANDIDATES,
    fallback: str = "DejaVu Sans",
) -> ResolvedFontFamily:
    resolved = resolve_sans_font_family(candidates=candidates, fallback=fallback)
    updates: dict[str, Any] = {
        "font.family": "sans-serif",
        "font.sans-serif": [resolved.family],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    if rc_updates:
        updates.update(dict(rc_updates))
    plt.rcParams.update(updates)
    return resolved
