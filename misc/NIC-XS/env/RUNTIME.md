# NIC-XS runtime

- **Step 1 (`eval_nside2_nosmooth_miss585.py`)** — needs CUDA torch + numpy + h5py.
  GPU: any single CUDA device (authors used an NVIDIA L40S / RTX-class card).
  Precision: fp32 (default). Memory: modest (nside=2 target, 48 pixels) — a few GB.
  Runtime scales with #models × #scenarios × `--n-draws`; ~minutes to a couple of
  hours for the full model set at n_draws=200.
- **Step 2 (`1080_Figure_GCMAGICC-XS_predictionskill.py`)** — CPU-only, numpy +
  pandas + matplotlib. Seconds.
- **Seeds:** `--seed 0` (global python/numpy/torch/cuda seeding). Deterministic
  draws on matched hardware + torch build.
- **Determinism caveat:** bit-exact member draws require the same Python/torch
  build and GPU model as recorded in `requirements.lock.txt`; different CUDA
  versions can differ at the float-noise level.
