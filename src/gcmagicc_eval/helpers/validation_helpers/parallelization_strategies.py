#!/usr/bin/env python3
"""
Parallelization Strategies for the Validation Suite

This module provides joblib-based parallelization configurations for optimal
performance and resource utilization in the validation suite.

All strategies now use joblib for consistent and efficient parallelization.

It also exposes process priority utilities:
  - set_low_priority(enable=True, cpu_nice=15, io_class="idle", io_priority=7)
  - set_joblib_low_priority(cpu_nice=15, io_class="idle", io_priority=7)
Workers inherit these priorities on spawn.
"""

import os
import multiprocessing as mp

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

# =============================================================================
# PRIORITY / NICENESS HELPERS (formerly a separate priority_manager.py)
# =============================================================================


def _safe_nice(target_nice: int) -> None:
    try:
        cur = os.nice(0)
        delta = max(0, int(target_nice) - cur)
        if delta:
            os.nice(delta)
    except Exception:
        pass


def _safe_ionice(io_class: str = "idle", io_priority: int = 7) -> None:
    try:
        if psutil:
            p = psutil.Process()
            c = str(io_class).lower()
            if c in ("idle", "3"):
                p.ionice(psutil.IOPRIO_CLASS_IDLE)  # type: ignore[attr-defined]
            elif c in ("be", "best-effort", "best_effort", "2"):
                p.ionice(psutil.IOPRIO_CLASS_BE, value=int(io_priority))  # type: ignore[attr-defined]
            elif c in ("rt", "1"):
                p.ionice(psutil.IOPRIO_CLASS_RT, value=int(min(7, max(0, io_priority))))  # type: ignore[attr-defined]
            return
    except Exception:
        pass
    # Fallback to shell ionice if psutil path fails
    try:
        cls = {
            "idle": "3",
            "3": "3",
            "be": "2",
            "best-effort": "2",
            "best_effort": "2",
            "2": "2",
            "rt": "1",
            "1": "1",
        }.get(str(io_class).lower(), "3")
        os.system(f"ionice -c {cls} -p {os.getpid()} >/dev/null 2>&1")
    except Exception:
        pass


def set_low_priority(
    enable: bool = True, cpu_nice: int = 15, io_class: str = "idle", io_priority: int = 7
) -> None:
    """Lower CPU and I/O priority of the current process (safe no-op on failure)."""
    if not enable:
        return
    _safe_nice(cpu_nice)
    _safe_ionice(io_class, io_priority)


def set_joblib_low_priority(
    cpu_nice: int = 15, io_class: str = "idle", io_priority: int = 7
) -> None:
    """Lower priority for the launcher; child workers inherit on spawn."""
    set_low_priority(True, cpu_nice, io_class, io_priority)


# =============================================================================
# PARALLELIZATION STRATEGIES
# =============================================================================


def get_strategy_config(strategy_name: str = "joblib_flat80") -> dict:
    """
    Get configuration for different joblib-based parallelization strategies.

    Strategies:
    - "joblib_adaptive": Automatically choose based on system resources (default)
    - "joblib_conservative": Conservative joblib approach for stability
    - "joblib_balanced": Balanced joblib approach for most systems
    - "joblib_aggressive": Aggressive joblib approach for maximum performance
    - "joblib_high_performance": Optimized for high-performance servers (60+ CPUs, 800GB+ RAM)
    - "joblib_debug": Single-threaded joblib for debugging
    """

    cpu_count = mp.cpu_count()

    strategies = {
        "joblib_flat80": {
            # Aim to saturate large nodes: leave a small CPU cushion but allow up to 64 workers.
            "n_workers": max(1, min(64, cpu_count - 8)),
            "timeout_seconds": 10800,
            "debug": False,
            "use_joblib_parallelization": True,
            "description": "High-throughput mode (threads=1). Plays well with external autoscaler.",
        },
        "joblib_debug": {
            "n_workers": 1,
            "timeout_seconds": 1800,
            "debug": True,
            "use_joblib_parallelization": True,
            "description": "Single worker for troubleshooting",
        },
    }

    if strategy_name not in strategies:
        print(f"⚠️  Unknown strategy '{strategy_name}', using 'joblib_flat80'")
        strategy_name = "joblib_flat80"

    config = strategies[strategy_name].copy()
    print(f"📋 Using joblib parallelization strategy: {strategy_name}")
    print(f"   - {config['description']}")
    print(f"   - Workers: {config['n_workers']}")
    print(f"   - Joblib parallelization: {config['use_joblib_parallelization']}")

    return config


def get_adaptive_config() -> dict:
    """
    Get adaptive configuration based on system resources.
    Now uses joblib strategies exclusively for better nested parallelization.
    """
    cpu_count = mp.cpu_count()
    memory_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)

    print("🔍 System Analysis:")
    print(f"   - CPU Count: {cpu_count}")
    print(f"   - Available Memory: {memory_gb:.1f} GB")

    # Keep it simple: flat80 by default on high-resource nodes
    print("   - Using flat80 strategy by default")
    return get_strategy_config("joblib_flat80")


def print_system_diagnostics():
    """Print detailed system diagnostics."""
    print("🔍 System Diagnostics:")
    print(f"   - CPU Count: {mp.cpu_count()}")
    print(f"   - Process ID: {os.getpid()}")

    try:
        import psutil

        memory = psutil.virtual_memory()
        print(f"   - Total Memory: {memory.total / (1024**3):.1f} GB")
        print(f"   - Available Memory: {memory.available / (1024**3):.1f} GB")
        print(f"   - Memory Usage: {memory.percent}%")
        print(f"   - CPU Usage: {psutil.cpu_percent()}%")
    except ImportError:
        print("   - psutil not available for detailed memory info")

    # Test multiprocessing start method
    try:
        mp.set_start_method("spawn")
        print("   - Multiprocessing start method: spawn")
    except RuntimeError:
        print("   - Multiprocessing start method: already set")

    # Test joblib availability
    try:
        __import__("joblib")
        print("   - Joblib: Available")
    except ImportError:
        print("   - Joblib: Not available (pip install joblib)")


def get_recommended_strategy() -> str:
    """Get recommended (existing) strategy based on current system state."""
    try:
        import psutil

        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().available / (1024**3)
        # cpu_usage = psutil.cpu_percent()  # unused variable removed

        # Simple, robust rule: prefer flat80 unless the box is tiny
        return "joblib_flat80" if cpu_count >= 8 and memory_gb >= 16 else "joblib_debug"

    except ImportError:
        return "joblib_flat80"


def get_joblib_strategies() -> list:
    return ["joblib_flat80", "joblib_debug"]


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("🧪 Joblib Parallelization Strategy Tester")
    print("=" * 50)

    # Print system diagnostics
    print_system_diagnostics()

    # Test different strategies
    strategies = ["joblib_flat80", "joblib_debug"]

    for strategy in strategies:
        print(f"\n📋 Testing strategy: {strategy}")
        config = get_strategy_config(strategy)
        print(f"   Configuration: {config}")

    # Get recommended strategy
    recommended = get_recommended_strategy()
    print(f"\n💡 Recommended strategy: {recommended}")

    # Show available strategies
    print(f"\n🔧 Available joblib strategies: {get_joblib_strategies()}")

    print("\n✅ Strategy testing completed!")
