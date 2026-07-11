"""Utility helpers for chunked GCMagicc inference."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

try:  # Torch is an optional dependency for these helpers.
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - torch always available in prod envs
    torch = None  # type: ignore


def _slice_output(output: Any, start: int):
    """Return the output truncated along the first axis."""

    if start <= 0:
        return output

    if torch is not None and isinstance(output, torch.Tensor):  # type: ignore[has-type]
        return output[start:]

    if isinstance(output, np.ndarray):
        return output[start:]

    raise TypeError(f"Unsupported chunk output type: {type(output)!r}")


def _concat_chunks(chunks: List[Any]):
    """Concatenate chunk outputs along the first axis."""

    if not chunks:
        raise ValueError("No chunks to concatenate")

    first = chunks[0]

    if torch is not None and isinstance(first, torch.Tensor):  # type: ignore[has-type]
        return torch.cat(chunks, dim=0)

    if isinstance(first, np.ndarray):
        return np.concatenate(chunks, axis=0)

    raise TypeError(f"Unsupported chunk output type: {type(first)!r}")


def run_chunked_sample(
    sample_fn: Callable[..., Any],
    *,
    x,
    base_kwargs: Dict[str, Any],
    chunk_size: int,
    dependence: bool,
    context_len: int,
) -> Any:
    """Run ``sample_fn`` over ``x`` in chunks while preserving dependence."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    total = int(x.shape[0])
    if total == 0:
        raise ValueError("Cannot chunk empty input")

    effective_context = context_len if dependence else 0
    outputs: List[Any] = []

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk_start = start if effective_context == 0 else max(0, start - effective_context)

        chunk_x = x[chunk_start:end]

        chunk_kwargs = base_kwargs.copy()
        chunk_kwargs["x"] = chunk_x

        chunk_output = sample_fn(**chunk_kwargs)

        if dependence and chunk_start < start:
            drop = start - chunk_start
            chunk_output = _slice_output(chunk_output, drop)

        outputs.append(chunk_output)

    return _concat_chunks(outputs)
