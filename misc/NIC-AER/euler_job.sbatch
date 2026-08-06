#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=8 --mem-per-cpu=24G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1 --time=24:00:00
#SBATCH --output=nic_aer_%j.slurm.log
# NIC-AER on ETH Euler: aerosol intervention at nside=64, nsim=100, MC 0..19.
# Heaviest of the four jobs; GPU type pinned to avoid a straggler.
set -euo pipefail
module load stack/2024-06 python_cuda/3.11.6 eth_proxy

# SCRATCH_BASE = data root (A5 checkpoints + normalized ERA5 H5; see EXTERNAL_MANIFEST.csv).
# BUNDLE_ROOT self-locates so the script runs from wherever the bundle is unpacked.
: "${SCRATCH_BASE:=/cluster/work/math/nicolai/gcm_release_data}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"

B="$BUNDLE_ROOT"
export MODEL_DIR="${SCRATCH_BASE}/modelsA/"
export DEVICE=cuda:0
export H5_PATH=$(ls ${SCRATCH_BASE}/normalized_era5/data_DAT_ERA5_historical-*.h5 | head -1)

cd "$B/code"
python -u cfAllAerJson.py
# Collect the final delta_mean map(s) into outputs/
cp -p DAER3_tasmax_mean/*.json DAER3_tasmax_mean/*.pdf ../outputs/ 2>/dev/null || true
echo "NIC-AER regeneration complete."
