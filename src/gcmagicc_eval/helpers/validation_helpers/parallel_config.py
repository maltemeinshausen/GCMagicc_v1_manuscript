#!/usr/bin/env python3
"""
Parallel Configuration for dev-maps-turbo

This script provides easy configuration for parallelizing the dev-maps-turbo segment.
You can adjust the settings below to optimize for your available CPUs.

Current Configuration:
- Process-level parallelization: 20 workers (processes)
- Thread-level parallelization: 6 threads per process
- Total theoretical CPU usage: 20 × 6 = 120 CPUs

You have 120 CPUs available, so this configuration should utilize them efficiently.
"""

# =============================================================================
# PARALLELIZATION CONFIGURATION
# =============================================================================

# Process-level parallelization (how many pairs to process simultaneously)
# Start aggressively; helper_benchmark will throttle if memory watchdogs trigger.
N_WORKERS = 64  # High-throughput default for 128-core nodes

# Thread-level parallelization (how many threads per process for internal work)
INTERNAL_THREADS = 1  # 64 × 1 = 64 logical CPUs; OMP threads stay at 1 per worker

# Total theoretical CPU usage
TOTAL_CPU_USAGE = N_WORKERS * INTERNAL_THREADS

# Timeout settings for robust parallelization
TIMEOUT_SECONDS = 10000  # 2.7 hours timeout for individual tasks

# print("Parallel Configuration:")
# print(f"  Process workers: {N_WORKERS}")
# print(f"  Threads per process: {INTERNAL_THREADS}")
# print(f"  Total CPU usage: {TOTAL_CPU_USAGE}")
# print("  Available CPUs: 128")
# print(f"  CPU utilization: {TOTAL_CPU_USAGE/128*100:.1f}%")
# print(f"  Timeout per task: {TIMEOUT_SECONDS} seconds")

# =============================================================================
# ALTERNATIVE CONFIGURATIONS
# =============================================================================

# Option 1: Balanced high-throughput
# Uses many workers with single-threaded BLAS (ideal for wide nodes)
CONFIG_IO_BOUND = {
    "n_workers": 48,
    "internal_threads": 1,
    "description": "Balanced high-throughput (48×1=48 CPUs)",
}

# Option 2: Fewer processes, more threads per process
# Good for CPU bound tasks
CONFIG_CPU_BOUND = {
    "n_workers": 16,
    "internal_threads": 2,
    "description": "CPU bound configuration (16×2=32 CPUs)",
}

# Option 3: Conservative configuration (RECOMMENDED for memory-intensive workloads)
# Uses fewer CPUs to avoid memory issues and OOM kills
CONFIG_CONSERVATIVE = {
    "n_workers": 8,
    "internal_threads": 2,
    "description": "Conservative configuration (8×2=16 CPUs) - prevents OOM kills",
}

# Option 4: Ultra-conservative for high-memory recipes
# For recipes that consume 100+ GB (ENSOTeleconnections, BiasMaps, etc.)
CONFIG_ULTRA_CONSERVATIVE = {
    "n_workers": 4,
    "internal_threads": 1,
    "description": "Ultra-conservative (4×1=4 CPUs) - for very high memory workloads",
}

# =============================================================================
# RECOMMENDATIONS
# =============================================================================


def get_recommendation():
    """Get recommended configuration based on system characteristics."""

    print("\nConfiguration Recommendations:")
    print("=" * 50)

    # Default recommendation
    print(
        f"1. DEFAULT (Current): {N_WORKERS} processes × {INTERNAL_THREADS} threads = {TOTAL_CPU_USAGE} CPUs"
    )
    print("   - Good balance of process and thread parallelization")
    print("   - Suitable for most workloads")

    print(
        f"\n2. I/O BOUND: {CONFIG_IO_BOUND['n_workers']} processes × {CONFIG_IO_BOUND['internal_threads']} threads = {CONFIG_IO_BOUND['n_workers'] * CONFIG_IO_BOUND['internal_threads']} CPUs"
    )
    print("   - More processes, fewer threads per process")
    print("   - Good when reading many files or network I/O is the bottleneck")

    print(
        f"\n3. CPU BOUND: {CONFIG_CPU_BOUND['n_workers']} processes × {CONFIG_CPU_BOUND['internal_threads']} threads = {CONFIG_CPU_BOUND['n_workers'] * CONFIG_CPU_BOUND['internal_threads']} CPUs"
    )
    print("   - Fewer processes, more threads per process")
    print("   - Good when computation is the bottleneck")

    print(
        f"\n4. CONSERVATIVE: {CONFIG_CONSERVATIVE['n_workers']} processes × {CONFIG_CONSERVATIVE['internal_threads']} threads = {CONFIG_CONSERVATIVE['n_workers'] * CONFIG_CONSERVATIVE['internal_threads']} CPUs"
    )
    print("   - Uses fewer CPUs to avoid memory issues")
    print("   - Good for systems with limited RAM or when debugging")


if __name__ == "__main__":
    get_recommendation()

    print("\nTo use a different configuration:")
    print("1. Update N_WORKERS and INTERNAL_THREADS in this file")
    print("2. Update the same values in 405_validation_suite.py")
    print("3. Or modify the validation_config dictionary directly")
