"""Shared schema registry for recipe JSON figure payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

ReplotFunc = Callable[[Dict[str, Any]], Any]


_REPLOT_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_replot_schema(*schema_names: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a payload -> figure renderer for one or more schemas."""

    if not schema_names:
        raise ValueError("At least one schema name must be provided")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        for name in schema_names:
            _REPLOT_REGISTRY[name] = func
        return func

    return decorator


def load_figure_from_json(
    path: str | Path,
    *,
    default_schema: str | None = None,
    **kwargs: Any,
) -> Any:
    """Load a JSON payload from *path* and dispatch to the registered renderer."""

    with open(Path(path), "r", encoding="utf-8") as f:
        payload = json.load(f)

    schema_name = payload.get("schema") or default_schema
    if not schema_name:
        raise ValueError("Figure JSON is missing a 'schema' field and no default was provided")

    try:
        renderer = _REPLOT_REGISTRY[schema_name]
    except KeyError as exc:  # pragma: no cover - guard for misconfiguration
        raise ValueError(f"No replot renderer registered for schema '{schema_name}'") from exc

    return renderer(payload, **kwargs)


def registered_schemas() -> Iterable[str]:
    """Return an iterable of all registered schema names."""

    return tuple(_REPLOT_REGISTRY.keys())
