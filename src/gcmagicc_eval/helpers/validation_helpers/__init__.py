"""Validation helpers package for the GCMAGICC validation suite.

This package has a wide optional-dependency surface area. To keep light-weight
submodules importable in environments that do not have the full plotting /
benchmark stack installed, exports are resolved lazily on first access.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "compute_nino34_index": "helper_bench_metric",
    "dprint": "helper_bench_metric",
    "parse_filename": "helper_bench_metric",
    "check_for_existing_records_batch": "helper_bench_metric",
    "select_robust_region": "helper_bench_metric",
    "_apply_smoothing_and_detrending": "helper_bench_metric",
    "generate_pdf_filename": "helper_bench_plot",
    "get_standard_colors": "helper_bench_plot",
    "add_rmse_text_to_ax": "helper_bench_plot",
    "compose_gcmagicc_model_name": "helper_bench_plot",
    "convert_cftime_to_numeric": "helper_bench_plot",
    "extract_gcmagicc_code": "helper_bench_plot",
    "get_common_variables_from_pair": "helper_bench_plot",
    "get_gcmagicc_prefix_ts": "helper_bench_plot",
    "get_paired_files": "helper_bench_plot",
    "compute_rmse_score": "helper_bench_plot",
    "_generate_time_windows": "helper_benchmark",
    "discover_common_spatial_vars_from_files": "helper_benchmark",
    "setup_carlito_font": "helper_recipes",
    "add_bold_title": "helper_recipes",
    "get_segment_title": "helper_recipes",
    "get_variable_info": "helper_recipes",
    "log_initial_memory": "helper_debug",
    "log_stage_memory": "helper_debug",
    "log_final_memory": "helper_debug",
    "get_memory_usage_mb": "helper_debug",
    "get_variable_units": "helper_ipcc_colormaps",
    "create_overview_barplot": "helper_metric_barplot",
    "apply_column_filters": "helper_metric_barplot",
    "_load_metric_database": "helper_metric_barplot",
    "N_WORKERS": "parallel_config",
    "INTERNAL_THREADS": "parallel_config",
    "TOTAL_CPU_USAGE": "parallel_config",
    "TIMEOUT_SECONDS": "parallel_config",
    "set_low_priority": "parallelization_strategies",
    "set_joblib_low_priority": "parallelization_strategies",
    "get_strategy_config": "parallelization_strategies",
    "get_adaptive_config": "parallelization_strategies",
    "get_recommended_strategy": "parallelization_strategies",
    "get_joblib_strategies": "parallelization_strategies",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
