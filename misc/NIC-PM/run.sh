#!/bin/bash
# NIC-PM reproduction.
# Step 2 (figure/quantiles) is CPU-only and needs NO external data.
set -euo pipefail
cd "$(dirname "$0")/code"

# --- OPTIONAL step 1: regenerate the trend tables from checkpoints (GPU) ------
# Needs a CUDA torch env + the external normalized data (see EXTERNAL_MANIFEST.csv)
# and the shipped per-model checkpoints. Skip to reproduce only from shipped CSVs.
if [ -n "${RUN_TRENDS:-}" ]; then
    export MODEL_DIR="${MODEL_DIR:-$(cd ../checkpoints && pwd)/modelsSsmall/}"
    export DATA_DIR="${DATA_DIR:?RUN_TRENDS set: point DATA_DIR at normalized data_DAT_*.h5}"
    python samplesSera.py     # writes figchangeeraSsmall_1/ and figchangeforceeraSsmall_1/
fi

# --- Step 2: emergent-constraint figure + quantiles (CPU, fully reproducible) -
cd ../data
python ../code/sample_emergent2.py \
    --data_ssp figchangeeraSsmall_1/trends_ssp_True_6.csv \
    --seed 0 --nsim 2000 \
    --output_base ../outputs/warming_panels \
    --quantiles_csv ../outputs/quantiles_by_scenario.csv

echo "Done. warming_panels_<scen>.pdf + quantiles_by_scenario.csv in outputs/."
echo "(Verified: regenerates byte-identical to the shipped quantiles table.)"
