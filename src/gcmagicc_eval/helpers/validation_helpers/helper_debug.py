"""Debug and diagnostics helpers shared across the validation toolkit."""

from __future__ import annotations

import os
from typing import Optional


def _get_psutil_process():
    """Return the current psutil process instance if psutil is available."""

    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid())
    except Exception:
        return None


def get_memory_usage_mb() -> Optional[float]:
    """Return the resident set size in megabytes for the current process."""

    process = _get_psutil_process()
    if process is None:
        return None
    try:
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return None


def log_initial_memory(label: str, debug: bool) -> Optional[float]:
    """Log and return the initial memory usage for a debug section."""

    if not debug:
        return None

    memory = get_memory_usage_mb()
    if memory is None:
        return None

    print(f"🔍 [{label}] Initial memory: {memory:.1f} MB")
    return memory


def log_stage_memory(
    label: str,
    message: str,
    debug: bool,
    *,
    baseline: Optional[float] = None,
) -> Optional[float]:
    """Log an intermediate memory measurement, optionally showing delta."""

    if not debug:
        return None

    memory = get_memory_usage_mb()
    if memory is None:
        return None

    if baseline is None:
        print(f"🔍 [{label}] {message}: {memory:.1f} MB")
    else:
        delta = memory - baseline
        print(f"🔍 [{label}] {message}: {memory:.1f} MB (change: {delta:+.1f} MB)")

    return memory


def log_final_memory(
    label: str,
    debug: bool,
    initial: Optional[float],
    *,
    warn_threshold: Optional[float] = None,
) -> Optional[float]:
    """Log the final memory usage and optional warning if delta is large."""

    if not debug:
        return None

    memory = get_memory_usage_mb()
    if memory is None:
        return None

    if initial is None:
        print(f"🔍 [{label}] Final memory: {memory:.1f} MB")
        delta = None
    else:
        delta = memory - initial
        print(f"🔍 [{label}] Final memory: {memory:.1f} MB (total change: {delta:+.1f} MB)")

    if warn_threshold is not None and delta is not None and delta > warn_threshold:
        print(f"⚠️  [{label}] WARNING: Large memory increase detected!")

    return memory
