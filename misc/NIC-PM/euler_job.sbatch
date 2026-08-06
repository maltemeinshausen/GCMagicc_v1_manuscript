#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=2 --mem-per-cpu=8G
#SBATCH --time=00:20:00
#SBATCH --output=nic_pm_%j.slurm.log
# NIC-PM on ETH Euler: emergent-constraint figure + quantiles (CPU-only, no GPU,
# no external data). Reproduces byte-identically from the shipped trend tables.
set -euo pipefail
module load stack/2024-06 python_cuda/3.11.6

# BUNDLE_ROOT self-locates so the script runs from wherever the bundle is unpacked.
# (NIC-PM needs no external data root, so no SCRATCH_BASE.)
BUNDLE_ROOT="${BUNDLE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
B="$BUNDLE_ROOT"
cd "$B/data"
python -u ../code/sample_emergent2.py \
    --data_ssp figchangeeraSsmall_1/trends_ssp_True_6.csv \
    --seed 0 --nsim 2000 \
    --output_base ../outputs/warming_panels \
    --quantiles_csv ../outputs/quantiles_by_scenario.csv
echo "NIC-PM regeneration complete."
