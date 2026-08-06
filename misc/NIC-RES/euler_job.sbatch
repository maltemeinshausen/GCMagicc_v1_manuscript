#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=8 --mem-per-cpu=20G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1 --time=6:00:00
#SBATCH --output=nic_res_%j.slurm.log
# NIC-RES on ETH Euler: Part B (compute-scaling benchmark) + Part A (spatial render).
# GPU type pinned so the timing panel is measured on a single, named card.
set -euo pipefail
module load stack/2024-06 python_cuda/3.11.6 eth_proxy

# SCRATCH_BASE = data root (A5 checkpoints + normalized ERA5 H5; see EXTERNAL_MANIFEST.csv).
# BUNDLE_ROOT self-locates so the script runs from wherever the bundle is unpacked.
: "${SCRATCH_BASE:=/cluster/work/math/nicolai/gcm_release_data}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"

B="$BUNDLE_ROOT"
export MODEL_DIR="${SCRATCH_BASE}/modelsA/"
export DEVICE=cuda:0

cd "$B/code"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Part B — timing/memory vs nside. months=120 (10 yr) keeps peak memory well
# within a 24 GB card; the A5 sampler runs the full 2->64 cascade regardless of
# the target nside (<=64), so coarse levels are computed en route (near-flat
# cost 1..64) and cost grows only at 128/256 (documented in the README).
python -u bench_resolution.py --model-dir "$MODEL_DIR" --device "$DEVICE" \
    --months 120 --nside-max 64 --warmup 1 --reps 3 --outdir ../outputs

# Part A — spatial multi-resolution render (needs the ERA5 file)
export H5_PATH=$(ls ${SCRATCH_BASE}/normalized_era5/data_DAT_ERA5_historical-*.h5 | head -1)
python -u plotResolutions.py
echo "NIC-RES regeneration complete."
