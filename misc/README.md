# GCMagicc reproducibility bundles

Four public release bundles for the GCMagicc GMD manuscript. Each bundle carries code,
config, checkpoints (or an external manifest for large ones), an input manifest,
a pinned environment, a README explaining the science and every panel, SHA-256
hashes, and dual licenses. Local private provenance notes and Python caches are
explicitly excluded by `scripts/pack_release_bundles.py`.

| Bundle | Figure | Public model | Reproduction status |
|---|---|---|---|
| **NIC-XS** | GCMagicc-XS extrapolation skill | GCMagicc-XS | code ready + verified on synthetic CSV; **GPU re-run blocked on external data** |
| **NIC-RES** | resolution (spatial + compute scaling) | GCMagicc | benchmark **verified end-to-end** on real A5 model (CPU); render step needs external data |
| **NIC-AER** | aerosol-pattern intervention | GCMagicc | code ready + scrubbed; **GPU run blocked on external data** |
| **NIC-PM** | perfect-model / emergent constraint | GCMagicc-PM | **figure reproduces BYTE-IDENTICALLY** from shipped tables (verified) |

## Layout (per bundle)

`code/` `config/` `checkpoints/` `data/` `outputs/` `env/` + `run.sh`,
`README.md`, `PROVENANCE_INTERNAL.md`, `LICENSE-Apache-2.0.txt`,
`LICENSE-CC-BY-4.0.txt`, `SHA256SUMS.txt`, `data/EXTERNAL_MANIFEST.csv`.

## Cross-cutting conventions

- **No credentials, hostnames, or private paths** in shipped code. Data roots,
  model directories, and device pins are environment-configurable (`SCRATCH_BASE`,
  `DATA_DIR`, `H5_PATH`, `MODEL_DIR`, `MODEL_BASE`, `DEVICE`).
- **Environment:** `env/requirements.lock.txt` (Python 3.14.6; torch 2.10.0+cu130;
  or the Euler stack torch 2.3.1+cu121). `env/RUNTIME.md` per bundle.
- **Files ≤ 50 MB** are eligible for the Git repository. Larger bundle inputs and
  release archives are listed with byte size, SHA-256, and immutable Zenodo URL
  in each `data/EXTERNAL_MANIFEST.csv` and the top-level external-data manifest.
- **Licensing:** code © Nicolai Meinshausen, Apache-2.0; data/figures/docs CC BY 4.0.

## Raw-input boundary

The 4.6-TB normalized `data_DAT_*.h5` collection (`y64` plus the predictor
matrix) is not redistributed. It is required for the GPU regeneration of
NIC-XS, NIC-AER, the optional NIC-PM trend step, and the NIC-RES render step.
The published figures remain reproducible from the frozen prepared tables,
compact plotted data, and final artifacts shipped here. NIC-RES Part B and the
NIC-PM figure do not require the normalized collection.

## Regeneration (when data is available)

Follow `gcm_sequence/EULER.md`, set the paths documented in
`_shared/regenerate.sbatch.template`, and submit with the pinned environment.
The historical GPU stacks reproduce up to floating-point noise.

## Acceptance

Each bundle is validated in a clean env: fresh venv from `env/requirements.lock.txt`,
fetch `EXTERNAL_MANIFEST.csv` files and verify their size+SHA-256, run `run.sh`,
recompute SHA-256 of regenerated outputs and diff against `SHA256SUMS.txt`. NIC-PM
already passes this end-to-end from shipped files.
