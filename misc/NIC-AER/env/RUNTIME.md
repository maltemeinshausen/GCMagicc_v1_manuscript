# NIC-AER runtime

- `cfAllAerJson.py` — CUDA torch + numpy + healpy + h5py + matplotlib
  (cartopy optional). nside=64, nsim=100, MC 0..19 → the heaviest of the four
  bundles per figure; budget GPU time accordingly (tens of minutes to hours
  depending on the number of MC members and the horizon length).
- Precision fp32. Seeds: global `seed` + per-sim offset (`seed + i`).
- Float-noise reproducibility requires matching torch/GPU build.
