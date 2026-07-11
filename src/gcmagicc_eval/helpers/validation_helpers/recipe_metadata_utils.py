# ruff: noqa: E402
from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Dict, Iterable, List

import numpy as np

try:  # pandas is optional in some environments
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas may be unavailable
    pd = None  # type: ignore


def _sanitize_value(value: Any) -> Any:
    """Convert common scientific Python objects to JSON-serialisable forms."""
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if pd is not None and isinstance(value, pd.Timestamp):  # pragma: no cover - depends on pandas
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_sanitize_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def build_metric_metadata(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a sanitized copy of metric records with per-record timestamps."""
    metadata: List[Dict[str, Any]] = []
    for rec in records:
        if not rec:
            continue
        cleaned = _sanitize_value(copy.deepcopy(rec))
        ts = datetime.utcnow().isoformat()
        if isinstance(cleaned, dict):
            cleaned.setdefault("timestamp", ts)
        metadata.append(cleaned)
    return metadata
