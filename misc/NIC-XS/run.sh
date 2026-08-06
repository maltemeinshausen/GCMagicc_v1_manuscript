#!/bin/bash
# NIC-XS reproduction: inputs -> full_monthly_results.csv (+ member_draws.npz) -> figure.
# Requires a CUDA torch env for step 1 (see env/requirements.lock.txt); step 2 is CPU-only.
set -euo pipefail
cd "$(dirname "$0")/code"

# ---- Point these at the fetched inputs (see data/EXTERNAL_MANIFEST.csv) ----
export DATA_DIR="${DATA_DIR:?set DATA_DIR to the dir holding normalized data_DAT_*.h5}"
export MODEL_BASE="${MODEL_BASE:-$(cd ../checkpoints && pwd)/modelsNfour_7Augext_miss585_nosmooth}"
SEED="${SEED:-0}"
NDRAWS="${NDRAWS:-200}"
TAG="${TAG:-_release}"

# 1) GPU: regenerate results table + member draws (5/50/95 percentiles)
python eval_nside2_nosmooth_miss585.py \
    --tag "$TAG" --n-draws "$NDRAWS" --seed "$SEED" \
    --model-base "$MODEL_BASE" --data-dir "$DATA_DIR" --max-files 1

CSV="plots_eval_nside2_nosmooth_miss585${TAG}/full_monthly_results.csv"

# 2) CPU: release figure + metrics table
python 1080_Figure_GCMAGICC-XS_predictionskill.py \
    --csv "$CSV" --outdir ../outputs

echo "Done. Figure(s) + xs_prediction_skill_metrics.csv in outputs/; "
echo "intermediate CSV + member_draws.npz in code/plots_eval_nside2_nosmooth_miss585${TAG}/"
