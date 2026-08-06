# NIC-PM runtime

- **Step 2 (`sample_emergent2.py`, the figure/quantiles)** — CPU-only, numpy +
  pandas + matplotlib. Seconds. Seed 0, nsim 2000. **Reproduces byte-identically**
  to the shipped `quantiles_by_scenario.csv` (verified 2026-07-12). This is the
  bundle's primary deliverable and needs no external data.
- **Step 1 (`samplesSera.py`, optional trend regeneration)** — CUDA torch + numpy
  + healpy + h5py; needs the external normalized data + the shipped per-model
  checkpoints. niter=3. Only required to rebuild the trend tables from scratch.
- **Determinism:** the figure step uses `np.random.default_rng(seed)` → exact
  reproduction on any platform. The upstream GPU step is float-noise reproducible
  on matched torch/GPU.
