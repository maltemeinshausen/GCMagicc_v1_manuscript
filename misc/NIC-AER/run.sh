#!/bin/bash
# NIC-AER reproduction. Requires a CUDA torch env (see env/requirements.lock.txt).
set -euo pipefail
cd "$(dirname "$0")/code"

export MODEL_DIR="${MODEL_DIR:-$(cd ../checkpoints && pwd)/modelsA/}"   # A5 checkpoints (external)
export DEVICE="${DEVICE:-cuda:0}"
# One normalized data_DAT_*.h5 supplies X (col 13 = aer_ERF). Provide via H5_PATH
# or DATA_DIR (see data/EXTERNAL_MANIFEST.csv):
export H5_PATH="${H5_PATH:?set H5_PATH to a normalized data_DAT_*.h5 (contains X with aer_ERF at col 13)}"

python cfAllAerJson.py

echo "Done. aer_ERF_<mc>_Nxl_tasmax.json + *_delta_mean.pdf written under code/DAER3_tasmax_mean/."
echo "Copy the final map(s) into ../outputs/ for the release."
