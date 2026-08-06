#!/bin/bash
# NIC-RES reproduction. Requires a CUDA torch env (see env/requirements.lock.txt).
set -euo pipefail
cd "$(dirname "$0")/code"

export MODEL_DIR="${MODEL_DIR:-$(cd ../checkpoints && pwd)/modelsA/}"   # A5 checkpoints (external, see EXTERNAL_MANIFEST.csv)
export DEVICE="${DEVICE:-cuda:0}"

# Part B — compute-cost scaling (needs only checkpoints; synthesises x by default)
python bench_resolution.py --model-dir "$MODEL_DIR" --device "$DEVICE" \
    --months "${MONTHS:-1200}" --nside-max "${NSIDE_MAX:-64}" \
    --warmup "${WARMUP:-2}" --reps "${REPS:-5}" --outdir ../outputs

# Part A — spatial multi-resolution render (needs one normalized data_DAT_*.h5)
# Provide the input via DATA_DIR or H5_PATH (see data/EXTERNAL_MANIFEST.csv), then:
#   export H5_PATH=/path/to/data_DAT_ERA5_historical-...nc.h5
#   python plotResolutions.py
if [ -n "${RUN_RENDER:-}" ]; then
    python plotResolutions.py
fi

echo "Done. resolution_timing.{csv,pdf} in outputs/. Set RUN_RENDER=1 (+H5_PATH) for Part A."
