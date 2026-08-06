# NIC-RES runtime

- **Part B (`bench_resolution.py`)** — CUDA torch + numpy + healpy (+ matplotlib
  for the figure). Report timings on a single, named GPU; the manuscript's
  efficiency numbers should cite the same card (confirm target: L40S / RTX-4090 /
  A100). fp32. Runs in minutes; the loader reloads checkpoints per call, so wall
  time includes checkpoint I/O (documented boundary).
- **Part A (`plotResolutions.py`)** — CUDA torch + numpy + healpy + h5py +
  matplotlib (cartopy optional for coastlines). Needs one `data_DAT_*.h5`.
- **Determinism:** `seed` is fixed in the sampler call; renderer time indices use
  `SEED_TIMEPOINTS` (recorded in `run_meta.json`). Float-noise reproducibility
  requires matching torch/GPU.
