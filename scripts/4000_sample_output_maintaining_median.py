#!/usr/bin/env python3
"""
4000_sample_output_maintaining_median.py

Create reduced-ensemble versions of MAGICC probabilistic output parquet files while
preserving the median end-of-century warming.

Context
-------
The 2002_* workflow generates probabilistic MAGICC outputs (typically 600 ensemble
members) and writes one parquet file per scenario into::

    ./output/AR6/runmode_all/*.parquet
    ./output/AR6/runmode_natural/*.parquet
    ./output/AR6/runmode_aerosol/*.parquet
    ./output/AR7/...

This script creates *parallel* directories next to ``./output``:

    ./output_resampled_2/
    ./output_resampled_10/
    ./output_resampled_20/
    ./output_resampled_50/
    ./output_resampled_100/

For every input parquet file, it writes a corresponding parquet file into each of
those directories, but keeping only N ensemble members (run_id) out of the original
~600.

Definition of "end of century warming"
--------------------------------------
For each ensemble member (identified by ``run_id``), end-of-century warming is
computed from the temperature variable (``tas`` or ``tas_smoothed`` / "Surface Air
Temperature Change") as:

    mean(2081-2100) - mean(1995-2014)

The median is taken across the ensemble members.

Median-preserving sub-sampling method
-------------------------------------
The requested sample sizes (2, 10, 20, 50, 100) are *even*. The MAGICC ensemble size is
typically 600, also *even*. For even-sized samples, the median equals the average
of the two central order statistics.

Therefore, we can preserve the full-ensemble median **exactly** (not just
approximately) by:

1. Sorting ensemble members by end-of-century warming.
2. Taking the two ensemble members that define the full-ensemble median.
3. Selecting an equal number of members from below and above those two.

By default, the additional members are chosen in a *stratified* / evenly spaced
fashion across the lower and upper halves to better preserve the overall
distribution and its percentiles. A purely random choice is still available via
``--strategy random``.

This produces a subset whose median warming equals the original median warming
(provided the full ensemble size is even, as expected). If the script encounters an
odd number of available members (e.g., failed runs), it will fall back to a
"random search" approach that tries multiple random subsets and keeps the one with
the closest median.

Sampling is always done *without replacement* (no duplicate run_id within a subset).

Outputs
-------
- Resampled parquet files under ``output_resampled_{N}/...`` mirroring the input
  directory structure under ``output/...``.
- A CSV summary (default: ``./output_resampled_summary.csv``) with medians and
  selection diagnostics.

Typical usage
-------------
Run from the 2025magicc project directory (the directory containing ``output/``)::

    python 4000_sample_output_maintaining_median.py

To overwrite existing resampled outputs::

    python 4000_sample_output_maintaining_median.py --overwrite

To delete and recreate the ``output_resampled_*`` directories in one go::

    python 4000_sample_output_maintaining_median.py --force-resampling

To use the user-suggested random-search method everywhere::

    python 4000_sample_output_maintaining_median.py --method random_search --attempts 100

Notes
-----
- The script assumes the parquet files use a MultiIndex that includes at least the
  levels ``variable`` and ``run_id`` (as produced by the provided 2002_* scripts).
- Temperature variable detection is automatic but can be overridden via
  ``--tas-variable``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_indexing as pix

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration defaults
# -----------------------------------------------------------------------------

DEFAULT_SAMPLE_SIZES = (2, 10, 20, 50, 100)

DEFAULT_PERIOD1 = (1995, 2014)
DEFAULT_PERIOD2 = (2081, 2100)

DEFAULT_ATTEMPTS = 100
DEFAULT_SEED = 12345

# We prefer a smoothed tas if present, otherwise fall back to raw tas / SAT.
DEFAULT_PREFER_SMOOTHED = True

# Region choice for temperature, if multiple are present.
DEFAULT_REGION_PREFERENCE = "World"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    """Configure logging to stdout."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _coerce_year(col: Any) -> Optional[int]:
    """
    Try to interpret a dataframe column label as a year.

    Returns None if it cannot be interpreted as an integer year.
    """
    # Common case: int / np.int64
    if isinstance(col, (int, np.integer)):
        return int(col)

    # Float that is an integer (rare, but possible)
    if isinstance(col, (float, np.floating)):
        if float(col).is_integer():
            return int(col)
        return None

    # String with digits
    if isinstance(col, str):
        s = col.strip()
        if s.isdigit():
            return int(s)

    return None


def build_year_column_map(columns: Iterable[Any]) -> dict[int, Any]:
    """
    Build a map {year_int -> original_column_label} for all columns that look like years.

    If multiple columns map to the same year, the first occurrence is kept.
    """
    year_to_col: dict[int, Any] = {}
    for col in columns:
        year = _coerce_year(col)
        if year is None:
            continue
        year_to_col.setdefault(year, col)
    return year_to_col


def pick_period_columns(
    year_to_col: dict[int, Any],
    start_year: int,
    end_year: int,
) -> list[Any]:
    """Return the list of column labels for the inclusive year range [start_year, end_year]."""
    missing = [y for y in range(start_year, end_year + 1) if y not in year_to_col]
    if missing:
        raise ValueError(
            f"Missing years in dataframe columns for range {start_year}-{end_year}. "
            f"First missing years: {missing[:10]} (total missing: {len(missing)})."
        )
    return [year_to_col[y] for y in range(start_year, end_year + 1)]


def _score_temperature_variable(name: str, prefer_smoothed: bool) -> int:
    """
    Heuristic scoring for temperature variables.

    Higher score means 'more likely the correct tas / surface air temperature change' variable.
    """
    s = name.lower()
    score = 0

    # Strong temperature markers
    if "temperature" in s:
        score += 100
    if "surface air" in s:
        score += 100
    if "tas" in s:
        score += 80
    if "change" in s:
        score += 30

    # Smoothed preference
    if any(tok in s for tok in ("tas_smoothed", "smoothed", "smooth")):
        score += 60 if prefer_smoothed else 10

    # De-prioritise blended / ocean blended if present
    if "ocean blended" in s or "blended" in s:
        score -= 30

    return score


def detect_temperature_variable(
    variables: Sequence[str],
    *,
    prefer_smoothed: bool = DEFAULT_PREFER_SMOOTHED,
    explicit: Optional[str] = None,
) -> str:
    """
    Pick the most likely tas / surface-air-temperature-change variable from a list.

    Parameters
    ----------
    variables
        Available variable names (strings) from the dataframe index.
    prefer_smoothed
        If True, prefer variables that look like a smoothed temperature output.
    explicit
        If provided, require this variable to exist and return it.

    Returns
    -------
    str
        Selected variable name.

    Raises
    ------
    ValueError
        If no suitable variable is found (or explicit variable is missing).
    """
    if explicit is not None:
        if explicit not in variables:
            raise ValueError(
                f"Requested --tas-variable '{explicit}' not found. "
                f"Available variables include: {sorted(variables)[:20]}..."
            )
        return explicit

    # First pass: if prefer_smoothed, focus on smoothed-like candidates
    if prefer_smoothed:
        smoothed_like = [
            v
            for v in variables
            if any(tok in v.lower() for tok in ("tas_smoothed", "smoothed", "smooth"))
        ]
        if smoothed_like:
            return max(
                smoothed_like,
                key=lambda v: _score_temperature_variable(v, prefer_smoothed=True),
            )

    # Second pass: any temperature-like variable
    temp_like = [
        v for v in variables if ("temperature" in v.lower()) or ("tas" in v.lower())
    ]
    if temp_like:
        return max(
            temp_like,
            key=lambda v: _score_temperature_variable(
                v, prefer_smoothed=prefer_smoothed
            ),
        )

    # Last resort: try exact common names
    for candidate in ("Surface Air Temperature Change", "tas", "tas_smoothed"):
        if candidate in variables:
            return candidate

    raise ValueError(
        "Could not detect a tas / temperature variable. "
        f"Available variables include: {sorted(variables)[:30]}..."
    )


def pick_single_level_value(
    df: pd.DataFrame,
    level: str,
    *,
    preferred: Optional[str] = None,
) -> tuple[pd.DataFrame, Optional[str]]:
    """
    If a MultiIndex level exists with multiple values, pick a single one.

    Returns the (possibly filtered) dataframe and the chosen value (or None if the level is absent).

    Notes
    -----
    - If `preferred` is present, it will be chosen.
    - Otherwise the first value in sorted order is chosen.
    """
    if not isinstance(df.index, pd.MultiIndex):
        return df, None
    if level not in df.index.names:
        return df, None

    values = list(df.index.get_level_values(level).unique())
    if not values:
        return df, None

    chosen: Any
    if preferred is not None and preferred in values:
        chosen = preferred
    else:
        # Deterministic: choose first value in sorted order (string sort)
        try:
            chosen = sorted(values)[0]
        except TypeError:
            # If values are not directly sortable (mixed types), fall back to first
            chosen = values[0]

    filtered = df.loc[pix.isin(**{level: [chosen]})]
    return filtered, str(chosen)


# -----------------------------------------------------------------------------
# Warming metric
# -----------------------------------------------------------------------------


@dataclass
class WarmingComputationInfo:
    tas_variable: str
    region: Optional[str]
    unit: Optional[str]
    n_total_members: int


def compute_end_of_century_warming_by_run_id(
    df: pd.DataFrame,
    *,
    tas_variable: Optional[str] = None,
    prefer_smoothed: bool = DEFAULT_PREFER_SMOOTHED,
    region_preference: str = DEFAULT_REGION_PREFERENCE,
    period1: tuple[int, int] = DEFAULT_PERIOD1,
    period2: tuple[int, int] = DEFAULT_PERIOD2,
) -> tuple[pd.Series, WarmingComputationInfo]:
    """
    Compute end-of-century warming for each run_id in a parquet dataframe.

    Parameters
    ----------
    df
        Data loaded from parquet. Expected to have a MultiIndex including 'variable' and 'run_id'.
    tas_variable
        If provided, use this variable name instead of auto-detection.
    prefer_smoothed
        Prefer smoothed temperature variable if available.
    region_preference
        If multiple regions exist, prefer this (e.g., 'World').
    period1, period2
        Inclusive year ranges used in the warming definition.

    Returns
    -------
    warming_by_run_id, info
        warming_by_run_id is a Series indexed by run_id.
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError(
            "Expected parquet to load into a DataFrame with a MultiIndex index."
        )
    for required_level in ("variable", "run_id"):
        if required_level not in df.index.names:
            raise ValueError(
                f"Expected MultiIndex level '{required_level}' in parquet file. "
                f"Found index levels: {df.index.names}"
            )

    variables = list(df.index.get_level_values("variable").unique())
    chosen_tas_var = detect_temperature_variable(
        variables,
        prefer_smoothed=prefer_smoothed,
        explicit=tas_variable,
    )
    tas_df = df.loc[pix.isin(variable=[chosen_tas_var])]

    # Prefer World if available, otherwise choose deterministically.
    tas_df, chosen_region = pick_single_level_value(
        tas_df, "region", preferred=region_preference
    )
    tas_df, chosen_unit = pick_single_level_value(tas_df, "unit", preferred=None)

    year_to_col = build_year_column_map(tas_df.columns)
    p1_cols = pick_period_columns(year_to_col, period1[0], period1[1])
    p2_cols = pick_period_columns(year_to_col, period2[0], period2[1])

    # Mean over years for each row (each row typically corresponds to one run_id for tas).
    p1_mean = tas_df.loc[:, p1_cols].mean(axis=1, skipna=True)
    p2_mean = tas_df.loc[:, p2_cols].mean(axis=1, skipna=True)
    diff = p2_mean - p1_mean

    # Aggregate to run_id in case there are multiple rows per run_id (shouldn't happen, but safe).
    warming_by_run_id = diff.groupby(level="run_id").mean()

    # Drop NaNs defensively (cannot sample/compute median with NaNs)
    n_before = len(warming_by_run_id)
    warming_by_run_id = warming_by_run_id.dropna()
    n_after = len(warming_by_run_id)
    if n_after != n_before:
        LOGGER.warning(
            "Dropped %d run_id(s) with NaN warming (from %d to %d).",
            n_before - n_after,
            n_before,
            n_after,
        )

    info = WarmingComputationInfo(
        tas_variable=chosen_tas_var,
        region=chosen_region,
        unit=chosen_unit,
        n_total_members=len(warming_by_run_id),
    )
    return warming_by_run_id, info


# -----------------------------------------------------------------------------
# Sub-sampling algorithms
# -----------------------------------------------------------------------------


@dataclass
class SubsampleSelectionInfo:
    method: str
    sample_size: int
    n_available: int
    full_median: float
    subset_median: float
    abs_median_diff: float
    selected_run_ids: list[Any]


def select_run_ids_median_anchored(
    warming_by_run_id: pd.Series,
    *,
    sample_size: int,
    rng: np.random.Generator,
    strategy: str = "random",
) -> SubsampleSelectionInfo:
    """
    Select run_ids such that the subset median is preserved (exactly for even N + even sample_size).

    Method
    ------
    - Sort by warming.
    - Choose a "median-defining pair" (two run_ids) from the full ensemble.
      * If N is even: use the two central members (guarantees exact preservation for even sample_size).
      * If N is odd: choose an adjacent pair whose average is closest to the full median.
    - Select equal numbers from below and above this pair to reach the desired sample_size.

    Parameters
    ----------
    warming_by_run_id
        Series indexed by run_id.
    sample_size
        Desired number of run_ids to select.
    rng
        Numpy random generator.
    strategy
        How to pick the additional members from below/above:
        - "random": uniform random without replacement from each side
        - "stratified": deterministic quantile-like selection (evenly spaced) from each side

    Returns
    -------
    SubsampleSelectionInfo
    """
    if sample_size < 2:
        raise ValueError("sample_size must be >= 2.")
    if sample_size % 2 != 0:
        raise ValueError(
            "This implementation expects an even sample_size (10/20/50/100). "
            f"Got sample_size={sample_size}."
        )

    warming_sorted = warming_by_run_id.sort_values()
    run_ids_sorted = warming_sorted.index.to_list()
    warmings_sorted = warming_sorted.to_numpy()
    n_available = len(run_ids_sorted)

    if sample_size > n_available:
        raise ValueError(
            f"Requested sample_size={sample_size} but only {n_available} run_id(s) available."
        )

    full_median = float(np.median(warmings_sorted))

    # Choose anchor pair indices (lo, hi) that will define the subset median.
    if n_available % 2 == 0:
        # Exact median pair for even-sized full ensemble
        anchor_lo = n_available // 2 - 1
        anchor_hi = n_available // 2
    else:
        # Full median is a single element; choose an adjacent pair whose mean is closest.
        mid = n_available // 2
        candidates: list[tuple[int, int]] = []
        if mid - 1 >= 0:
            candidates.append((mid - 1, mid))
        if mid + 1 < n_available:
            candidates.append((mid, mid + 1))
        if not candidates:
            # n_available == 1 would be the only case, but sample_size>=2 prevents this.
            raise RuntimeError(
                "Internal error: no candidate anchor pairs for odd-sized ensemble."
            )

        best_pair = min(
            candidates,
            key=lambda ij: abs(
                ((warmings_sorted[ij[0]] + warmings_sorted[ij[1]]) / 2.0) - full_median
            ),
        )
        anchor_lo, anchor_hi = best_pair

    anchor_lo_run = run_ids_sorted[anchor_lo]
    anchor_hi_run = run_ids_sorted[anchor_hi]

    k_each_side = sample_size // 2 - 1  # number below + number above

    below_pool = run_ids_sorted[:anchor_lo]
    above_pool = run_ids_sorted[anchor_hi + 1 :]

    if len(below_pool) < k_each_side or len(above_pool) < k_each_side:
        raise ValueError(
            "Not enough members below/above the median anchor pair to construct the requested subset. "
            f"Need {k_each_side} from each side, have {len(below_pool)} below and {len(above_pool)} above."
        )

    if strategy not in {"random", "stratified"}:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Use 'random' or 'stratified'."
        )

    if k_each_side == 0:
        below_sel: list[Any] = []
        above_sel: list[Any] = []
    elif strategy == "random":
        below_sel = list(rng.choice(below_pool, size=k_each_side, replace=False))
        above_sel = list(rng.choice(above_pool, size=k_each_side, replace=False))
    else:
        # Stratified / evenly spaced selection
        def stratified(pool: Sequence[Any], k: int) -> list[Any]:
            if k == 0:
                return []
            if k >= len(pool):
                return list(pool)
            # Evenly spaced indices
            idx = np.linspace(0, len(pool) - 1, k)
            idx_int = np.unique(np.round(idx).astype(int))
            # If rounding produced too few unique indices, fill in the gaps.
            if len(idx_int) < k:
                missing = k - len(idx_int)
                # Add additional indices not yet used, starting from the middle outward.
                candidates_int = [i for i in range(len(pool)) if i not in set(idx_int)]
                # Deterministic order: by distance to center
                center = (len(pool) - 1) / 2.0
                candidates_int.sort(key=lambda i: (abs(i - center), i))
                idx_int = np.concatenate(
                    [idx_int, np.array(candidates_int[:missing], dtype=int)]
                )
                idx_int = np.unique(idx_int)
            # Ensure exactly k indices
            idx_int = np.sort(idx_int)[:k]
            return [pool[i] for i in idx_int]

        below_sel = stratified(below_pool, k_each_side)
        above_sel = stratified(above_pool, k_each_side)

    selected = [*below_sel, anchor_lo_run, anchor_hi_run, *above_sel]

    subset_median = float(np.median(warming_by_run_id.loc[selected].to_numpy()))
    abs_diff = float(abs(subset_median - full_median))

    return SubsampleSelectionInfo(
        method=f"median_anchored_{strategy}",
        sample_size=sample_size,
        n_available=n_available,
        full_median=full_median,
        subset_median=subset_median,
        abs_median_diff=abs_diff,
        selected_run_ids=selected,
    )


def select_run_ids_random_search(
    warming_by_run_id: pd.Series,
    *,
    sample_size: int,
    rng: np.random.Generator,
    attempts: int = DEFAULT_ATTEMPTS,
) -> SubsampleSelectionInfo:
    """
    User-suggested approach: try multiple random subsets and keep the one with the closest median.

    Sampling is done *without replacement* within each attempt.

    Parameters
    ----------
    warming_by_run_id
        Series indexed by run_id.
    sample_size
        Desired subset size.
    rng
        Numpy RNG.
    attempts
        Number of random subsets to try.

    Returns
    -------
    SubsampleSelectionInfo
    """
    if sample_size < 1:
        raise ValueError("sample_size must be >= 1.")
    if attempts < 1:
        raise ValueError("attempts must be >= 1.")

    warming_clean = warming_by_run_id.dropna()
    run_ids = warming_clean.index.to_numpy()
    warmings = warming_clean.to_numpy()
    n_available = len(run_ids)

    if sample_size > n_available:
        raise ValueError(
            f"Requested sample_size={sample_size} but only {n_available} run_id(s) available."
        )

    full_median = float(np.median(warmings))

    best_selected: Optional[list[Any]] = None
    best_subset_median: Optional[float] = None
    best_abs_diff = float("inf")

    # For speed, pre-create an array of indices.
    all_idx = np.arange(n_available)

    for _ in range(attempts):
        sel_idx = rng.choice(all_idx, size=sample_size, replace=False)
        subset_vals = warmings[sel_idx]
        subset_median = float(np.median(subset_vals))
        abs_diff = float(abs(subset_median - full_median))

        if abs_diff < best_abs_diff:
            best_abs_diff = abs_diff
            best_subset_median = subset_median
            best_selected = run_ids[sel_idx].tolist()

        # Early exit if we matched exactly (possible for even/odd combinations)
        if best_abs_diff == 0.0:
            break

    if best_selected is None or best_subset_median is None:
        raise RuntimeError("Failed to find a subset; check inputs and attempts.")

    return SubsampleSelectionInfo(
        method=f"random_search_{attempts}",
        sample_size=sample_size,
        n_available=n_available,
        full_median=full_median,
        subset_median=float(best_subset_median),
        abs_median_diff=float(best_abs_diff),
        selected_run_ids=list(best_selected),
    )


# -----------------------------------------------------------------------------
# IO + processing
# -----------------------------------------------------------------------------


def _normalize_file_globs(file_glob: Optional[Union[Sequence[str], str]]) -> list[str]:
    """
    Normalize file_glob input to a list of glob patterns.

    Supports:
    - None -> []
    - single string -> [string] (comma-separated values are split)
    - sequence of strings -> list of strings (empty entries removed)
    """
    if file_glob is None:
        return []
    if isinstance(file_glob, str):
        parts = [p.strip() for p in file_glob.split(",")]
        return [p for p in parts if p]
    return [str(p).strip() for p in file_glob if str(p).strip()]


def iter_parquet_files(
    input_dir: Path,
    file_glob: Optional[Union[Sequence[str], str]] = None,
) -> list[Path]:
    """
    Collect parquet files under input_dir.

    If file_glob is provided, it is applied to the *relative path* via Path.match.
    Multiple patterns are supported and matched with OR logic, e.g.
    ["ssp245*.parquet", "*runmode_all*"].
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(input_dir.rglob("*.parquet"))
    patterns = _normalize_file_globs(file_glob)
    if patterns:
        filtered: list[Path] = []
        for p in files:
            rel = p.relative_to(input_dir)
            rel_posix = rel.as_posix()
            if any(
                (
                    rel.match(f"**/{pattern}")
                    if any(ch in pattern for ch in "*?[]")
                    else pattern in rel_posix
                )
                for pattern in patterns
            ):
                filtered.append(p)
        files = filtered
    return files


def filter_df_to_run_ids(df: pd.DataFrame, run_ids: Sequence[Any]) -> pd.DataFrame:
    """Return a filtered dataframe keeping only the given run_id values."""
    if not isinstance(df.index, pd.MultiIndex) or "run_id" not in df.index.names:
        raise ValueError(
            "Cannot filter: dataframe index must be a MultiIndex that includes level 'run_id'."
        )
    return df.loc[pix.isin(run_id=list(run_ids))]


def write_summary_csv(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a CSV summary with stable column order."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Stable column ordering
    fieldnames = [
        "input_file",
        "output_file",
        "sample_size",
        "n_available",
        "full_median_warming",
        "subset_median_warming",
        "abs_median_diff",
        "tas_variable",
        "region",
        "unit",
        "method",
        "selected_run_ids_json",
    ]

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    LOGGER.info("Wrote resampling summary to %s", summary_path)


def plot_warming_ranges(
    warming_data: dict[int, pd.Series],
    output_path: Path,
    input_file_rel: str,
) -> None:
    """
    Plot percentiles (5, 10, 33, 50, 67, 90, 95) for end-of-century warming across different sample sizes.

    Parameters
    ----------
    warming_data
        Dictionary mapping sample_size -> Series of warming values.
        Use a special key (e.g., -1) for the original full ensemble.
    output_path
        Path where the plot will be saved.
    input_file_rel
        Relative path of the input file (for plot title).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Separate full ensemble from subsamples
    # Look for negative integer key (we use -1 for full ensemble)
    full_key = None
    for key in warming_data.keys():
        if isinstance(key, int) and key < 0:
            full_key = key
            break

    if full_key is None:
        LOGGER.warning("No full ensemble data found for plotting %s", input_file_rel)
        return

    full_warming = warming_data[full_key]
    subsample_sizes = sorted(
        [k for k in warming_data.keys() if k != full_key and isinstance(k, int)]
    )

    # Percentiles to compute and display
    percentiles = [5, 10, 33, 50, 67, 90, 95]

    # Compute percentiles for each sample size
    sample_sizes = []
    percentile_data = {p: [] for p in percentiles}

    # Add full ensemble first
    sample_sizes.append(len(full_warming))
    for p in percentiles:
        percentile_data[p].append(float(np.percentile(full_warming, p)))

    # Add subsamples
    for n in subsample_sizes:
        warming_vals = warming_data[n]
        sample_sizes.append(n)
        for p in percentiles:
            percentile_data[p].append(float(np.percentile(warming_vals, p)))

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot ranges as horizontal bars (5-95% range)
    y_pos = range(len(sample_sizes))
    y_labels = [
        f"Full\n({sample_sizes[0]})" if i == 0 else str(s)
        for i, s in enumerate(sample_sizes)
    ]

    p5_values = percentile_data[5]
    p95_values = percentile_data[95]

    # Plot 5-95% range as horizontal bars
    for i, (p5, p95) in enumerate(zip(p5_values, p95_values)):
        ax.barh(
            i,
            p95 - p5,
            left=p5,
            height=0.6,
            alpha=0.2,
            color="steelblue",
            edgecolor="black",
            linewidth=1.5,
        )

    # Define markers and colors for each percentile
    percentile_styles = {
        5: {
            "color": "darkblue",
            "marker": "|",
            "size": 80,
            "linewidth": 2.5,
            "label": "5th",
        },
        10: {
            "color": "blue",
            "marker": "|",
            "size": 70,
            "linewidth": 2,
            "label": "10th",
        },
        33: {
            "color": "teal",
            "marker": "|",
            "size": 60,
            "linewidth": 2,
            "label": "33rd",
        },
        50: {
            "color": "red",
            "marker": "o",
            "size": 120,
            "linewidth": 1.5,
            "label": "50th (median)",
        },
        67: {
            "color": "orange",
            "marker": "|",
            "size": 60,
            "linewidth": 2,
            "label": "67th",
        },
        90: {
            "color": "purple",
            "marker": "|",
            "size": 70,
            "linewidth": 2,
            "label": "90th",
        },
        95: {
            "color": "darkred",
            "marker": "|",
            "size": 80,
            "linewidth": 2.5,
            "label": "95th",
        },
    }

    # Plot all percentiles
    for p in percentiles:
        values = percentile_data[p]
        style = percentile_styles[p]

        if style["marker"] == "|":
            # Vertical lines for non-median percentiles
            for i, val in enumerate(values):
                ax.plot(
                    [val, val],
                    [i - 0.3, i + 0.3],
                    color=style["color"],
                    linewidth=style["linewidth"],
                    zorder=4,
                    label=style["label"] if i == 0 else "",
                )
        else:
            # Points for median
            ax.scatter(
                values,
                y_pos,
                color=style["color"],
                s=style["size"],
                zorder=5,
                marker=style["marker"],
                edgecolors="darkred" if p == 50 else style["color"],
                linewidths=style["linewidth"],
                label=style["label"],
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("End-of-century warming (°C)", fontsize=12, weight="bold")
    ax.set_ylabel("Sample size", fontsize=12, weight="bold")
    ax.set_title(
        f"Percentiles of End-of-Century Warming\n{input_file_rel}",
        fontsize=13,
        pad=15,
        weight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x", linestyle="--")

    # Create legend with unique entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="best",
        fontsize=9,
        framealpha=0.9,
        ncol=2,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    LOGGER.info("Saved warming range plot to %s", output_path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input_dir = script_dir / "output"
    default_summary = script_dir / "output_resampled_summary.csv"

    p = argparse.ArgumentParser(
        description=(
            "Create output_resampled_{N} directories with N-member subsets of the "
            "MAGICC probabilistic output parquet files, preserving median end-of-century warming."
        )
    )

    p.add_argument(
        "--input-dir",
        type=str,
        default=str(default_input_dir),
        help="Directory containing the original parquet files (default: ./output).",
    )
    p.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SAMPLE_SIZES),
        help="Subsample sizes to create (default: 2 10 20 50 100).",
    )
    p.add_argument(
        "--period1",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=list(DEFAULT_PERIOD1),
        help="Baseline period (inclusive) used in warming definition (default: 1995 2014).",
    )
    p.add_argument(
        "--period2",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=list(DEFAULT_PERIOD2),
        help="End-of-century period (inclusive) used in warming definition (default: 2081 2100).",
    )
    p.add_argument(
        "--tas-variable",
        type=str,
        default=None,
        help=(
            "Explicit temperature variable name to use. If omitted, the script will auto-detect "
            "a tas/tas_smoothed/surface-air-temperature-change variable."
        ),
    )
    p.add_argument(
        "--prefer-smoothed",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_PREFER_SMOOTHED,
        help="Prefer a smoothed tas variable if present (default: True).",
    )
    p.add_argument(
        "--region-preference",
        type=str,
        default=DEFAULT_REGION_PREFERENCE,
        help="Preferred region name for tas if multiple are present (default: World).",
    )
    p.add_argument(
        "--method",
        choices=("median_anchored", "random_search"),
        default="median_anchored",
        help=(
            "Subsampling method. 'median_anchored' preserves the median exactly for even-sized "
            "ensembles and even sample sizes. 'random_search' tries multiple random subsets and "
            "keeps the closest-median subset."
        ),
    )
    p.add_argument(
        "--strategy",
        choices=("random", "stratified"),
        default="stratified",
        help=(
            "When using method=median_anchored, how to choose members below/above the median pair. "
            "'stratified' picks evenly spaced members to preserve distribution shape (default)."
        ),
    )
    p.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_ATTEMPTS,
        help="Number of attempts for method=random_search (default: 100).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility (default: 12345).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing resampled parquet files (default: False, i.e. skip existing).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write any parquet files; only compute and write the summary CSV.",
    )
    p.add_argument(
        "--force-resampling",
        action="store_true",
        help=(
            "Delete existing output_resampled_{N} directories for the requested sample sizes "
            "before processing, ensuring no stale files remain."
        ),
    )
    p.add_argument(
        "--file-glob",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional glob(s) to restrict processed files (matched against the relative path). "
            "By default all *.parquet files under --input-dir are processed. "
            "Provide one or more patterns, e.g. 'ssp245*.parquet' '*runmode_all*'. "
            "Patterns without wildcard characters are treated as substring matches."
        ),
    )
    p.add_argument(
        "--summary-file",
        type=str,
        default=str(default_summary),
        help="Path for the summary CSV (default: ./output_resampled_summary.csv).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    script_dir = Path(__file__).resolve().parent

    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.is_absolute():
        input_dir = (script_dir / input_dir).resolve()

    sample_sizes = [int(x) for x in args.sample_sizes]
    if any(x <= 0 for x in sample_sizes):
        raise ValueError(f"All --sample-sizes must be positive. Got: {sample_sizes}")
    # Ensure unique sizes and stable order
    sample_sizes = sorted(set(sample_sizes))

    period1 = (int(args.period1[0]), int(args.period1[1]))
    period2 = (int(args.period2[0]), int(args.period2[1]))
    if period1[0] > period1[1] or period2[0] > period2[1]:
        raise ValueError(f"Invalid period ranges: period1={period1}, period2={period2}")

    # Output directories live next to the input_dir (parallel to ./output)
    output_dirs = {n: input_dir.parent / f"output_resampled_{n}" for n in sample_sizes}

    if args.force_resampling:
        for n, outdir in output_dirs.items():
            if outdir.exists():
                if not outdir.is_dir():
                    raise ValueError(
                        f"Expected output path to be a directory: {outdir}"
                    )
                LOGGER.warning(
                    "Removing existing resampled directory (--force-resampling): %s",
                    outdir,
                )
                shutil.rmtree(outdir)

    for n, outdir in output_dirs.items():
        outdir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Ensured output directory exists: %s", outdir)

    file_glob_patterns = _normalize_file_globs(args.file_glob)
    if file_glob_patterns:
        LOGGER.info("Restricting input files with glob(s): %s", file_glob_patterns)
    else:
        LOGGER.info("No --file-glob supplied; processing all parquet files.")

    parquet_files = iter_parquet_files(input_dir, file_glob=file_glob_patterns)
    if not parquet_files:
        LOGGER.warning("No parquet files found under %s", input_dir)
        return 0

    LOGGER.info("Found %d parquet file(s) under %s", len(parquet_files), input_dir)

    rng = np.random.default_rng(int(args.seed))

    summary_rows: list[dict[str, Any]] = []

    for i, in_file in enumerate(parquet_files, start=1):
        rel = in_file.relative_to(input_dir)
        LOGGER.info("[%d/%d] Processing %s", i, len(parquet_files), rel)

        try:
            df = pd.read_parquet(in_file)
        except Exception:
            LOGGER.exception("Failed to read %s", in_file)
            continue

        try:
            warming_by_run, warm_info = compute_end_of_century_warming_by_run_id(
                df,
                tas_variable=args.tas_variable,
                prefer_smoothed=bool(args.prefer_smoothed),
                region_preference=str(args.region_preference),
                period1=period1,
                period2=period2,
            )
        except Exception:
            LOGGER.exception("Failed to compute warming for %s", in_file)
            continue

        # Store warming data for plotting: full ensemble and subsamples
        warming_data_for_plot: dict[int, pd.Series] = {}
        warming_data_for_plot[-1] = warming_by_run  # Use -1 as key for full ensemble

        for n in sample_sizes:
            out_file = output_dirs[n] / rel
            out_file.parent.mkdir(parents=True, exist_ok=True)

            skip_writing = out_file.exists() and not args.overwrite
            if skip_writing:
                LOGGER.info(
                    "  - Skipping existing output (use --overwrite): %s", out_file
                )
                # Still need warming data for plotting, so read existing file to get run_ids
                try:
                    df_existing = pd.read_parquet(out_file)
                    existing_run_ids = list(
                        df_existing.index.get_level_values("run_id").unique()
                    )
                    if len(existing_run_ids) == n:
                        # Store warming values for this subsample for plotting
                        subsample_warming = warming_by_run.loc[existing_run_ids]
                        warming_data_for_plot[n] = subsample_warming
                        LOGGER.info(
                            "  - Extracted warming data from existing file for plotting (n=%d)",
                            n,
                        )
                    else:
                        LOGGER.warning(
                            "  - Existing file has %d run_ids, expected %d. Skipping plot data for n=%d",
                            len(existing_run_ids),
                            n,
                            n,
                        )
                except Exception as exc:
                    LOGGER.warning(
                        "  - Failed to read existing file for plotting data: %s", exc
                    )
                continue

            try:
                if args.method == "median_anchored":
                    try:
                        sel_info = select_run_ids_median_anchored(
                            warming_by_run,
                            sample_size=n,
                            rng=rng,
                            strategy=str(args.strategy),
                        )
                    except Exception as exc_anchor:
                        LOGGER.warning(
                            "  - median_anchored failed for %s (n=%d): %s. Falling back to random_search.",
                            rel,
                            n,
                            exc_anchor,
                        )
                        sel_info = select_run_ids_random_search(
                            warming_by_run,
                            sample_size=n,
                            rng=rng,
                            attempts=int(args.attempts),
                        )
                else:
                    sel_info = select_run_ids_random_search(
                        warming_by_run,
                        sample_size=n,
                        rng=rng,
                        attempts=int(args.attempts),
                    )
            except Exception:
                LOGGER.exception("  - Failed to select run_ids for %s (n=%d)", rel, n)
                continue

            LOGGER.info(
                "  - n=%d: full_median=%.4f, subset_median=%.4f, |Δ|=%.6f (%s)",
                n,
                sel_info.full_median,
                sel_info.subset_median,
                sel_info.abs_median_diff,
                sel_info.method,
            )

            if not args.dry_run:
                try:
                    df_sel = filter_df_to_run_ids(df, sel_info.selected_run_ids)
                    df_sel.to_parquet(out_file)
                except Exception:
                    LOGGER.exception("  - Failed to write %s", out_file)
                    continue

            summary_rows.append(
                {
                    "input_file": str(rel),
                    "output_file": str(out_file),
                    "sample_size": n,
                    "n_available": sel_info.n_available,
                    "full_median_warming": sel_info.full_median,
                    "subset_median_warming": sel_info.subset_median,
                    "abs_median_diff": sel_info.abs_median_diff,
                    "tas_variable": warm_info.tas_variable,
                    "region": warm_info.region,
                    "unit": warm_info.unit,
                    "method": sel_info.method,
                    "selected_run_ids_json": json.dumps(sel_info.selected_run_ids),
                }
            )

            # Store warming values for this subsample for plotting
            subsample_warming = warming_by_run.loc[sel_info.selected_run_ids]
            warming_data_for_plot[n] = subsample_warming

        # Create plot for this file showing 5-95% ranges
        plot_output_dir = input_dir.parent / "output_resampled_plots"
        plot_output_dir.mkdir(parents=True, exist_ok=True)

        # Create a safe filename from the relative path
        plot_filename = str(rel).replace("/", "_").replace("\\", "_")
        if not plot_filename.endswith(".png"):
            plot_filename = plot_filename.replace(".parquet", ".png")
            if not plot_filename.endswith(".png"):
                plot_filename += ".png"

        plot_path = plot_output_dir / plot_filename

        try:
            plot_warming_ranges(warming_data_for_plot, plot_path, str(rel))
        except Exception:
            LOGGER.exception("Failed to create plot for %s", rel)

    summary_path = Path(args.summary_file).expanduser()
    if not summary_path.is_absolute():
        summary_path = (script_dir / summary_path).resolve()
    write_summary_csv(summary_path, summary_rows)

    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
