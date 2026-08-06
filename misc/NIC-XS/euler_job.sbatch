#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=8 --mem-per-cpu=16G
#SBATCH --gpus=1 --time=8:00:00
#SBATCH --output=nic_xs_%j.slurm.log
# NIC-XS regeneration on ETH Euler (canonical env). nside=2 -> light compute.
set -euo pipefail
module load stack/2024-06 python_cuda/3.11.6 eth_proxy

# SCRATCH_BASE = data root holding the normalized H5 inputs (see EXTERNAL_MANIFEST.csv).
# BUNDLE_ROOT self-locates so the script runs from wherever the bundle is unpacked.
: "${SCRATCH_BASE:=/cluster/work/math/nicolai/gcm_release_data}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"

B="$BUNDLE_ROOT"
export DATA_DIR="${SCRATCH_BASE}/normalized_xs/"
export MODEL_BASE="$B/checkpoints/modelsNfour_7Augext_miss585_nosmooth"

cd "$B/code"
python -u eval_nside2_nosmooth_miss585.py --tag _release --n-draws 200 --seed 0 --max-files 1
python -u 1080_Figure_GCMAGICC-XS_predictionskill.py \
    --csv plots_eval_nside2_nosmooth_miss585_release/full_monthly_results.csv \
    --outdir ../outputs
echo "NIC-XS regeneration complete."
