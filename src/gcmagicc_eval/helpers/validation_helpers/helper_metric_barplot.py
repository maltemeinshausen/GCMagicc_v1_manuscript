# helper_metric_barplot.py - revised with x-tick indices, better highlight
from __future__ import annotations
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import to_hex
import re
from typing import Union, List, Optional, Dict

# ---------------------- paths & constants ------------------------
_DATA_DIR = "./data/metric_databases"
# Storage backend: 'csv' | 'sqlite' | 'auto'
_DATA_STORAGE = "auto"

# Known CSV filenames per metrickey. ZSCORE kept for backward-compat.
_DB_FILES = dict(
    GOFNC="gofnc_database.csv",
    GOFCC="gofcc_database.csv",
    GOFCZ="gofcz_database.csv",
    GOFOC="gofoc_database.csv",
    GOFON="gofon_database.csv",
    GOFZZ="gofzz_database.csv",
    GOFNN="gofnn_database.csv",
    GOFNN_SC="gofnn_database.csv",
    GOFNN_SO="gofnn_database.csv",
    ZSCORE="zscore_database.csv",  # alias; falls back to GOFZZ if missing
    EDISTNC="edistnc_database.csv",
    EDISTCZ="edistcz_database.csv",
    SCOREEDISTC="score_edist_database.csv",
    EDISTOC="edistoc_database.csv",
    EDISTON="ediston_database.csv",
    SCOREEDISTO="score_edisto_database.csv",
)

# Known SQLite table names per metrickey
_DB_TABLES = dict(
    GOFNC="gofnc",
    GOFCC="gofcc",
    GOFCZ="gofcz",
    GOFOC="gofoc",
    GOFON="gofon",
    GOFZZ="gofzz",
    GOFNN="gofnn",
    GOFNN_SC="gofnn",
    GOFNN_SO="gofnn",
    ZSCORE="gofzz",  # alias to GOFZZ in sqlite
    EDISTNC="edistnc",
    EDISTCZ="edistcz",
    SCOREEDISTC="score_edist",
    EDISTOC="edistoc",
    EDISTON="ediston",
    SCOREEDISTO="score_edisto",
)
_TAB20B = [cm.get_cmap("tab20b")(i) for i in range(20)]


def wildcard_match(pattern: str, text: str) -> bool:
    """
    Check if text matches a wildcard pattern.

    Parameters:
    -----------
    pattern : str
        Pattern with * wildcards (e.g., "Frequencies*RMSE")
    text : str
        Text to match against

    Returns:
    --------
    bool
        True if text matches pattern
    """
    # Convert wildcard pattern to regex
    # Escape special regex characters except *
    pattern_regex = re.escape(pattern)
    # Replace escaped \* with .* for wildcard matching
    pattern_regex = pattern_regex.replace(r"\*", ".*")
    # Add start and end anchors
    pattern_regex = f"^{pattern_regex}$"

    return bool(re.match(pattern_regex, text, re.IGNORECASE))


def create_row_key(row: pd.Series) -> str:
    """
    Create a row key by concatenating specific columns with '__' separator.

    Parameters:
    -----------
    row : pd.Series
        A row from the dataframe

    Returns:
    --------
    str
        Concatenated string of specific column values
    """
    # Define the columns to include in the row key
    key_columns = [
        "metrickey",
        "metricdomain",
        "metrictype",
        "variable",
        "source_id",
        "member_id",
        "experiment_id",
        "comparison_code",
    ]

    # Get values for each key column, handling missing columns gracefully
    key_values = []
    for col in key_columns:
        if col in row.index:
            key_values.append(str(row[col]))
        else:
            key_values.append("")  # Empty string for missing columns

    return "__".join(key_values)


def apply_advanced_filtering_with_row_keys(
    tidy: pd.DataFrame,
    filter4str_AND: Optional[Union[str, List[str]]] = None,
    filter4str_OR: Optional[Union[str, List[str]]] = None,
    filter4str_NOT: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Apply advanced filtering with AND, OR, and NOT logic using wildcard patterns
    on row keys created from specific columns.

    Parameters:
    -----------
    tidy : pd.DataFrame
        The dataframe to filter
    filter4str_AND : Optional[Union[str, List[str]]]
        Patterns that ALL must match (AND logic)
    filter4str_OR : Optional[Union[str, List[str]]]
        Patterns where ANY must match (OR logic)
    filter4str_NOT : Optional[Union[str, List[str]]]
        Patterns that NONE must match (NOT logic)

    Returns:
    --------
    pd.DataFrame
        Filtered dataframe
    """
    filtered_df = tidy.copy()

    # Convert single strings to lists for consistent processing
    if filter4str_AND is not None:
        if isinstance(filter4str_AND, str):
            filter4str_AND = [filter4str_AND]

    if filter4str_OR is not None:
        if isinstance(filter4str_OR, str):
            filter4str_OR = [filter4str_OR]

    if filter4str_NOT is not None:
        if isinstance(filter4str_NOT, str):
            filter4str_NOT = [filter4str_NOT]

    # Create row keys for filtering
    row_keys = filtered_df.apply(create_row_key, axis=1)

    # Apply AND filter (all patterns must match)
    if filter4str_AND:
        and_mask = pd.Series([True] * len(filtered_df), index=filtered_df.index)
        for pattern in filter4str_AND:
            pattern_mask = row_keys.apply(lambda x: wildcard_match(pattern, x))
            and_mask = and_mask & pattern_mask
        filtered_df = filtered_df[and_mask]

    # Apply OR filter (any pattern must match)
    if filter4str_OR:
        or_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
        for pattern in filter4str_OR:
            pattern_mask = row_keys.apply(lambda x: wildcard_match(pattern, x))
            or_mask = or_mask | pattern_mask
        filtered_df = filtered_df[or_mask]

    # Apply NOT filter (no pattern must match)
    if filter4str_NOT:
        not_mask = pd.Series([True] * len(filtered_df), index=filtered_df.index)
        for pattern in filter4str_NOT:
            pattern_mask = row_keys.apply(lambda x: wildcard_match(pattern, x))
            not_mask = not_mask & ~pattern_mask
        filtered_df = filtered_df[not_mask]

    return filtered_df


def apply_advanced_filtering(
    groups: List[str],
    filter4str_AND: Optional[Union[str, List[str]]] = None,
    filter4str_OR: Optional[Union[str, List[str]]] = None,
    filter4str_NOT: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    """
    Apply advanced filtering with AND, OR, and NOT logic using wildcard patterns.

    Parameters:
    -----------
    groups : List[str]
        List of group names to filter
    filter4str_AND : Optional[Union[str, List[str]]]
        Patterns that ALL must match (AND logic)
    filter4str_OR : Optional[Union[str, List[str]]]
        Patterns where ANY must match (OR logic)
    filter4str_NOT : Optional[Union[str, List[str]]]
        Patterns that NONE must match (NOT logic)

    Returns:
    --------
    List[str]
        Filtered list of groups
    """
    filtered_groups = groups.copy()

    # Convert single strings to lists for consistent processing
    if filter4str_AND is not None:
        if isinstance(filter4str_AND, str):
            filter4str_AND = [filter4str_AND]

    if filter4str_OR is not None:
        if isinstance(filter4str_OR, str):
            filter4str_OR = [filter4str_OR]

    if filter4str_NOT is not None:
        if isinstance(filter4str_NOT, str):
            filter4str_NOT = [filter4str_NOT]

    # Apply AND filter (all patterns must match)
    if filter4str_AND:
        and_matching = []
        for group in filtered_groups:
            all_match = True
            for pattern in filter4str_AND:
                if not wildcard_match(pattern, group):
                    all_match = False
                    break
            if all_match:
                and_matching.append(group)
        filtered_groups = and_matching

    # Apply OR filter (any pattern must match)
    if filter4str_OR:
        or_matching = []
        for group in filtered_groups:
            any_match = False
            for pattern in filter4str_OR:
                if wildcard_match(pattern, group):
                    any_match = True
                    break
            if any_match:
                or_matching.append(group)
        filtered_groups = or_matching

    # Apply NOT filter (no pattern must match)
    if filter4str_NOT:
        not_matching = []
        for group in filtered_groups:
            no_match = True
            for pattern in filter4str_NOT:
                if wildcard_match(pattern, group):
                    no_match = False
                    break
            if no_match:
                not_matching.append(group)
        filtered_groups = not_matching

    return filtered_groups


def _version_colour_map(tags: list[str]) -> dict[str, str]:
    """
    Returns a mapping: version_tag -> distinct color (hex code).

    Uses Set3 for ≤12 tags, otherwise falls back to tab20b.
    """
    tags_sorted = sorted(tags, reverse=True)
    n = len(tags_sorted)

    if n <= 8:
        cmap = cm.get_cmap("Set2", 8)
        hexes = [to_hex(cmap(i)) for i in range(n)]
    else:
        cmap = cm.get_cmap("tab20b", n)
        hexes = [to_hex(cmap(i)) for i in range(n)]

    return dict(zip(tags_sorted, hexes))


def _resolve_sqlite_file(data_dir: str) -> str | None:
    """Pick a sqlite file in the directory by convention or presence."""
    # Common names
    for name in ("metrics.sqlite", "edist.sqlite"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return p
    # Fallback: first *.sqlite
    candidates = [f for f in os.listdir(data_dir) if f.endswith(".sqlite")]
    if candidates:
        return os.path.join(data_dir, candidates[0])
    return None


def _load_metric_database(
    metrickey: str, data_dir: str | None = None, storage: str | None = None
) -> pd.DataFrame:
    metrickey = metrickey.upper()
    requested_mk = metrickey
    if metrickey not in _DB_FILES and metrickey not in _DB_TABLES:
        raise ValueError(f"Unknown metrickey '{metrickey}'")

    data_dir = data_dir or _DATA_DIR
    storage = (storage or _DATA_STORAGE or "auto").lower()

    # Auto-detect storage if requested
    if storage == "auto":
        sqlite_path = _resolve_sqlite_file(data_dir)
        storage = "sqlite" if sqlite_path else "csv"

    if storage == "csv":
        # Prefer exact mapping; allow ZSCORE->GOFZZ fallback if zscore file missing
        filename = _DB_FILES.get(metrickey)
        if metrickey == "ZSCORE":
            # if explicit zscore file not present, try gofzz
            zpath = os.path.join(data_dir, filename)
            if not os.path.exists(zpath):
                filename = _DB_FILES["GOFZZ"]
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Database not found at {path}")
        df = pd.read_csv(path)

    elif storage == "sqlite":
        import sqlite3

        sqlite_path = _resolve_sqlite_file(data_dir)
        if not sqlite_path:
            raise FileNotFoundError(f"No sqlite file found in {data_dir}")
        table = _DB_TABLES.get(metrickey)
        if not table:
            raise ValueError(f"No table mapping for metrickey '{metrickey}'")
        con = sqlite3.connect(sqlite_path)
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table}", con)
        finally:
            con.close()

    else:
        raise ValueError(f"Unsupported storage '{storage}' (expected 'csv'|'sqlite'|'auto')")

    # Ensure version_tag is present and normalized to strings ('' for empties)
    if "version_tag" not in df.columns:
        df["version_tag"] = ""
    df["version_tag"] = df["version_tag"].astype(str).fillna("")

    # Virtual GOFNN views for separate S(c) vs S(o) portal links.
    # - GOFNN_SC: conventional CMIP6-calibrated GOFNN (exclude v101gxe)
    # - GOFNN_SO: observation-side GOFNN (v101gxe only)
    if requested_mk in {"GOFNN_SC", "GOFNN_SO"}:
        if "metrickey" in df.columns:
            mk_mask = (
                df["metrickey"]
                .astype(str)
                .str.upper()
                .isin(["GOFNN", "GOFNN_SC", "GOFNN_SO"])
            )
            df = df[mk_mask].copy()
        vt = df["version_tag"].astype(str).str.lower()
        if requested_mk == "GOFNN_SO":
            df = df[vt.eq("v101gxe")].copy()
            df["metrickey"] = "GOFNN_SO"
        else:
            df = df[~vt.eq("v101gxe")].copy()
            df["metrickey"] = "GOFNN_SC"

    return df


def _filter_df_by_column_contains(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    """Filter rows by column->patterns with case-insensitive substring match.

    Example mapping: {"variable": ["pr", "tas"], "metrictype": ["RMSE"]}
    Unspecified columns are not filtered. Missing columns are ignored.
    """
    if not mapping:
        return df
    out = df
    for col, patterns in mapping.items():
        if not patterns:
            continue
        if col not in out.columns:
            # Skip silently if column not present
            continue
        ser = out[col].astype(str).str.lower()
        col_mask = False
        for pat in patterns:
            if pat is None:
                continue
            pat_s = str(pat).lower()
            col_mask = col_mask | ser.str.contains(pat_s, na=False)
        out = out[col_mask]
    return out


def _norm_filter_map(m) -> Dict[str, List[str]]:
    """
    Normalize filter mapping input to {column: [patterns...]},
    accepting either a dict mapping or a legacy list like ['col=pat','col2=pat2'].
    """
    if m is None:
        return {}
    out: Dict[str, List[str]] = {}
    # Legacy: list of "col=pattern" strings
    if isinstance(m, list):
        for item in m:
            if item is None:
                continue
            s = str(item)
            if "=" not in s:
                # fallback: treat as wildcard against "row key" columns not supported now; skip
                continue
            col, pat = s.split("=", 1)
            col = col.strip()
            pat = pat.strip()
            if not col:
                continue
            out.setdefault(col, []).append(pat)
        return out
    # Preferred: dict
    if isinstance(m, dict):
        for col, pats in m.items():
            if pats is None:
                continue
            if isinstance(pats, str):
                out[col] = [pats]
            else:
                out[col] = [str(p) for p in pats]
        return out
    # Unsupported type
    return {}


def apply_column_filters(
    df: pd.DataFrame,
    and_map: Optional[Dict[str, Union[str, List[str]]]] = None,
    or_map: Optional[Dict[str, Union[str, List[str]]]] = None,
    not_map: Optional[Dict[str, Union[str, List[str]]]] = None,
) -> pd.DataFrame:
    """
    Column-wise substring filters with AND / OR / NOT semantics.

    - and_map: all listed columns must match at least one of their patterns.
    - or_map: at least one listed column must match at least one of its patterns.
    - not_map: exclude rows where any listed column matches any of its patterns.

    Matching is case-insensitive substring; missing columns are skipped.
    """
    if df.empty:
        return df

    and_map = _norm_filter_map(and_map)
    or_map = _norm_filter_map(or_map)
    not_map = _norm_filter_map(not_map)

    mask = pd.Series(True, index=df.index)

    def col_mask_for(col: str, patterns: List[str]) -> Optional[pd.Series]:
        if col not in df.columns or not patterns:
            return None
        ser = df[col].astype(str).str.lower()
        m = pd.Series(False, index=df.index)
        for p in patterns:
            m = m | ser.str.contains(str(p).lower(), na=False)
        return m

    # AND
    if and_map:
        and_mask = pd.Series(True, index=df.index)
        for col, patterns in and_map.items():
            cm = col_mask_for(col, patterns)
            if cm is None:
                continue
            and_mask = and_mask & cm
        mask = mask & and_mask

    # OR
    if or_map:
        or_mask = pd.Series(False, index=df.index)
        for col, patterns in or_map.items():
            cm = col_mask_for(col, patterns)
            if cm is None:
                continue
            or_mask = or_mask | cm
        mask = mask & or_mask

    # NOT
    if not_map:
        bad_mask = pd.Series(False, index=df.index)
        for col, patterns in not_map.items():
            cm = col_mask_for(col, patterns)
            if cm is None:
                continue
            bad_mask = bad_mask | cm
        mask = mask & (~bad_mask)

    return df[mask]


def _normalize_version_tag_simple(tag: Union[str, float, int, None]) -> Union[str, None]:
    """Strip timestamp suffix after first underscore, e.g., 'Nextvers1_12Aug' -> 'Nextvers1'."""
    if tag is None or (isinstance(tag, float) and pd.isna(tag)):
        return None
    s = str(tag)
    return s.split("_")[0] if "_" in s else s


def _symlog_hierarchical_barplot(
    tidy: pd.DataFrame,
    value_col: str,
    hierarchy: list[str],
    highlight_tag: str | None,
    fig_title: str | None,
    out_dir: str | None,
    figsize=(14, 7),
    max_labels_for_readability: int = 15,
    tight_yaxis: bool = False,
    filter4str_AND=None,
    filter4str_OR=None,
    filter4str_NOT=None,
    figure_name: Optional[str] = None,
) -> pd.DataFrame:
    tidy = tidy.copy()

    # Create group ID based on hierarchy
    if len(hierarchy) == 1:
        tidy["_group_id"] = tidy[hierarchy[0]].astype(str)
    else:
        # Use string concatenation to avoid DataFrame issues
        group_parts = []
        for col in hierarchy:
            group_parts.append(tidy[col].astype(str))
        tidy["_group_id"] = group_parts[0]
        for part in group_parts[1:]:
            tidy["_group_id"] = tidy["_group_id"] + "." + part

    # Get unique groups and create x-position mapping
    unique_groups = sorted(tidy["_group_id"].unique())
    group_to_xpos = {group: i for i, group in enumerate(unique_groups)}

    # Assign x-positions to groups with spacing
    group_spacing = 1.0  # Space between groups
    tidy["_x_pos"] = tidy["_group_id"].map(group_to_xpos) * group_spacing

    # -- Version-tag handling (treat '' as a real "versionless" bucket) ---------
    vtags_raw = tidy.get("version_tag", pd.Series("", index=tidy.index)).astype(str).fillna("")
    has_any_nonempty = (vtags_raw != "").any()
    tidy["__plot_tag"] = vtags_raw.replace("", "(versionless)")
    highlight_plot = (
        None
        if highlight_tag is None
        else ("(versionless)" if str(highlight_tag) == "" else str(highlight_tag))
    )
    col_map = (
        _version_colour_map(list(tidy["__plot_tag"].unique()))
        if has_any_nonempty
        else {"(versionless)": "#1f77b4"}
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_yscale("symlog", linthresh=1, base=10)
    ax.axhline(0, color="k", lw=0.7, zorder=1)
    ax.set_ylabel(value_col)
    ax.set_title(fig_title or "", pad=20)

    # Set y-axis limits if tight_yaxis is True
    if tight_yaxis:
        ax.set_ylim(-100, 100)

    # Ensure aggregation columns exist (when no duplicates)
    if "min" not in tidy.columns:
        tidy["min"] = tidy[value_col]
    if "max" not in tidy.columns:
        tidy["max"] = tidy[value_col]
    if "n" not in tidy.columns:
        tidy["n"] = 1

    # Pre-compute per-group tag indices and number of versions (vectorized)
    uniq = (
        tidy[["_group_id", "__plot_tag"]]
        .drop_duplicates()
        .sort_values(["_group_id", "__plot_tag"])
    )
    uniq["tag_index"] = uniq.groupby("_group_id").cumcount()
    uniq["n_versions"] = uniq.groupby("_group_id")["__plot_tag"].transform("size")
    tidy = tidy.merge(uniq, on=["_group_id", "__plot_tag"], how="left")

    # Vectorized offsets (spread 0.6 width if >1 versions)
    denom = (tidy["n_versions"] - 1).replace(0, 1)
    offsets = np.where(
        tidy["n_versions"] > 1,
        (tidy["tag_index"] - (tidy["n_versions"] - 1) / 2.0) * (0.3 / denom),
        0.0,
    )
    tidy["_x_plot"] = tidy["_x_pos"] + offsets

    # Plot per tag (vectorized), with error bars only when n>1
    for tag_plot, sub in tidy.groupby("__plot_tag", sort=True):
        color = col_map[tag_plot]
        edge_color = (
            "black" if (highlight_plot is not None and tag_plot == highlight_plot) else "none"
        )
        edge_width = 2 if edge_color == "black" else 0

        # n>1 -> errorbar; n==1 -> scatter
        with_err = sub[sub["n"] > 1]
        no_err = sub[sub["n"] <= 1]
        if not with_err.empty:
            y = with_err[value_col].to_numpy()
            yerr = np.vstack([(y - with_err["min"].to_numpy()), (with_err["max"].to_numpy() - y)])
            ax.errorbar(
                with_err["_x_plot"].to_numpy(),
                y,
                yerr=yerr,
                fmt="o",
                color=color,
                elinewidth=1.5,
                capsize=4,
                markersize=7,
                markeredgewidth=edge_width,
                markeredgecolor=edge_color,
                zorder=5,
            )
        if not no_err.empty:
            ax.plot(
                no_err["_x_plot"].to_numpy(),
                no_err[value_col].to_numpy(),
                "o",
                color=color,
                markersize=9,
                markeredgewidth=edge_width,
                markeredgecolor=edge_color,
                zorder=5,
            )

    # Set x-axis ticks at group centers
    group_positions = [i * group_spacing for i in range(len(unique_groups))]
    ax.set_xticks(group_positions)

    # Create tick labels with index numbers
    xtick_labels = [f"**{i}**: {group}" for i, group in enumerate(unique_groups)]
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=10, ha="left")

    # Create legend
    handles = []
    labels = []
    # Build legend from available plot tags
    for tag_plot, color in col_map.items():
        rect = plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=color,
            edgecolor="black"
            if (highlight_plot is not None and tag_plot == highlight_plot)
            else "none",
            linewidth=2 if (highlight_plot is not None and tag_plot == highlight_plot) else 0,
            alpha=0.9,
        )
        handles.append(rect)
        lab = (
            f"{tag_plot} (highlight)"
            if (highlight_plot is not None and tag_plot == highlight_plot)
            else f"{tag_plot}"
        )
        labels.append(lab)
    ax.legend(handles, labels, title="version_tag", loc="upper left", bbox_to_anchor=(1.02, 1))

    # Set x-axis limits to show all groups with some padding
    ax.set_xlim(-0.5, len(unique_groups) * group_spacing - 0.5)
    ax.margins(y=0.1)

    # Add adaptive background patch: only if data spans positive values and y=0 is within the plot range
    y_min, y_max = ax.get_ylim()
    if y_min <= 0 <= y_max and y_max > 0:
        # Only add background if y=0 is visible and there are positive values
        background_max = min(1.0, y_max)  # Don't extend beyond y=1 or the data max
        if background_max > 0:
            ax.axhspan(0, background_max, color="lightgreen", alpha=0.2, zorder=0)

    plt.tight_layout()

    # Compose compact mapping (already aggregated rows)
    mapping_df = tidy.copy()
    mapping_df = mapping_df.rename(columns={"_x_plot": "x_pos"})
    # retain only useful columns in the mapping
    keep_map_cols = [
        c
        for c in (hierarchy + ["version_tag", "__plot_tag", "x_pos", value_col, "min", "max", "n"])
        if c in mapping_df.columns
    ]
    mapping_df = mapping_df[keep_map_cols]

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use custom figure name if provided, otherwise use default
        if figure_name:
            base_name = figure_name
        else:
            base_name = f"barplot_{stamp}"

        png_path = os.path.join(out_dir, f"{base_name}.png")
        pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
        csv_path = os.path.join(out_dir, f"mapping_{base_name}.csv")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        mapping_df.to_csv(csv_path, index=False)
        print(
            f"Saved:\n  * figure (PNG) -> {png_path}\n  * figure (PDF) -> {pdf_path}\n  * mapping -> {csv_path}"
        )

    return mapping_df


def create_overview_barplot(
    metrickey: str,
    highlight_tag: str | None = None,
    hierarchy_groups: list[str] | None = None,
    out_dir: str | None = None,
    figsize=(14, 7),  # ← accepts "auto"
    max_labels_for_readability: int = 15,
    tight_yaxis: bool = False,
    filter4str_AND: Optional[Dict[str, Union[str, List[str]]]] = None,
    filter4str_OR: Optional[Dict[str, Union[str, List[str]]]] = None,
    filter4str_NOT: Optional[Dict[str, Union[str, List[str]]]] = None,
    data_dir: str | None = None,
    storage: str | None = None,
    normalize_version_tags: bool = False,
    figure_name: Optional[str] = None,
) -> pd.DataFrame:
    if hierarchy_groups is None:
        hierarchy_groups = ["variable", "metricdomain", "metrictype"]

    df = _load_metric_database(metrickey, data_dir=data_dir, storage=storage)

    # Normalize version tags across the entire dataframe if requested
    if normalize_version_tags and "version_tag" in df.columns:
        df = df.copy()
        df["version_tag"] = df["version_tag"].apply(_normalize_version_tag_simple)
        if highlight_tag is not None:
            highlight_tag = _normalize_version_tag_simple(highlight_tag)

    # New-style column filters (AND / OR / NOT)
    before_rows = len(df)
    df = apply_column_filters(
        df, and_map=filter4str_AND, or_map=filter4str_OR, not_map=filter4str_NOT
    )
    if len(df) == 0:
        raise ValueError("No rows found after applying column filters")
    if len(df) != before_rows:
        print(f"Column filters reduced rows: {before_rows} -> {len(df)}")
    value_col = "value"  # All databases use 'value' column

    cat_cols = [
        "metricdomain",
        "metrictype",
        "variable",
        "source_id",
        "member_id",
        "experiment_id",
        "comparison_code",
    ]
    # Guarantee presence (harmless if missing)
    if "comparison_code" not in df.columns:
        df["comparison_code"] = ""
    residual = [c for c in cat_cols if c not in hierarchy_groups and c in df.columns]

    keep_cols = hierarchy_groups + ["version_tag", value_col] + residual
    print(f"DEBUG: create_overview_barplot - df shape after filtering: {df.shape}")
    print(f"DEBUG: create_overview_barplot - keep_cols: {keep_cols}")
    print(f"DEBUG: create_overview_barplot - available columns in df: {list(df.columns)}")

    tidy = df[keep_cols].copy()
    print(f"DEBUG: create_overview_barplot - tidy shape after column selection: {tidy.shape}")

    # Decide whether to include version_tag in grouping:
    # - For inter-model (GOFCC/GOFCZ/GOFZZ) with all-empty tags -> IGNORE version_tag (behaves as versionless)
    # - Otherwise include version_tag so different versions plot side-by-side.
    is_inter_model_db = metrickey.upper() in ["GOFCC", "GOFCZ", "GOFZZ"]
    has_any_nonempty = df["version_tag"].astype(str).fillna("").ne("").any()
    include_version_dim = (not is_inter_model_db) or has_any_nonempty

    grouping_cols = hierarchy_groups + (["version_tag"] if include_version_dim else [])

    if tidy.duplicated(subset=grouping_cols).any():
        print("DEBUG: create_overview_barplot - duplicates found, performing aggregation")
        print(f"DEBUG: create_overview_barplot - grouping columns: {grouping_cols}")

        # Check what we're grouping by
        for col in grouping_cols:
            unique_vals = tidy[col].unique()
            print(f"DEBUG: create_overview_barplot - {col}: {len(unique_vals)} unique values")
            if len(unique_vals) <= 5:
                print(f"DEBUG: create_overview_barplot - {col} values: {unique_vals}")

        grouped = tidy.groupby(grouping_cols, dropna=False)[value_col]
        print(f"DEBUG: create_overview_barplot - number of groups: {grouped.ngroups}")

        if grouped.ngroups > 0:
            tidy = grouped.agg(["median", "min", "max", "count"]).reset_index()
            tidy.rename(columns={"median": value_col, "count": "n"}, inplace=True)
            print(f"DEBUG: create_overview_barplot - tidy shape after aggregation: {tidy.shape}")
        else:
            print("DEBUG: create_overview_barplot - ❌ PROBLEM: No groups found!")
    else:
        print("DEBUG: create_overview_barplot - no duplicates found, no aggregation needed")

    # After aggregation ignoring version_tag, ensure column exists (versionless)
    if "version_tag" not in tidy.columns:
        tidy["version_tag"] = ""

    # --- AUTO-FIGSIZE logic ---------------------------------------------
    if figsize == "auto":
        n_bars = tidy.groupby(grouping_cols).ngroups
        base_h = 4
        extra_per_bar = 0.2  # adjust to taste
        height = base_h + extra_per_bar * n_bars
        figsize = (14, height)
    # --------------------------------------------------------------------

    # Title
    # Generate appropriate title
    if is_inter_model_db and not has_any_nonempty:
        fig_title = f"{metrickey} performance (versionless)"
    elif highlight_tag is not None:
        fig_title = f"{metrickey} performance (highlight: {highlight_tag})"
    else:
        fig_title = f"{metrickey} performance"

    print(
        f"DEBUG: create_overview_barplot - tidy shape before calling _symlog_hierarchical_barplot: {tidy.shape}"
    )
    print(f"DEBUG: create_overview_barplot - tidy columns: {list(tidy.columns)}")
    # Map highlight '' -> '(versionless)' for plotting function
    if highlight_tag is not None and str(highlight_tag) == "":
        hl_plot = "(versionless)"
    else:
        hl_plot = highlight_tag

    return _symlog_hierarchical_barplot(
        tidy=tidy,
        value_col=value_col,
        hierarchy=hierarchy_groups,
        highlight_tag=hl_plot,
        fig_title=fig_title,
        out_dir=out_dir,
        figsize=figsize,
        max_labels_for_readability=max_labels_for_readability,
        tight_yaxis=tight_yaxis,
        filter4str_AND=None,
        filter4str_OR=None,
        filter4str_NOT=None,
        figure_name=figure_name,
    )


if __name__ == "__main__":
    create_overview_barplot(
        metrickey="GOFNC",
        highlight_tag="VersionNfour2_28Jul2025_0516",
        hierarchy_groups=["variable", "metricdomain", "metrictype"],
        out_dir="./reports/test",
    )
