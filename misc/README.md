# GCMagicc reproducibility bundles

Four self-contained, clean-room-reproducible bundles for the GCMagicc GMD
manuscript. Each bundle regenerates its intermediates + figure and carries code,
config, checkpoints (or an external manifest for large ones), an input manifest,
a pinned environment, a README explaining the science and every panel, SHA-256
hashes, dual licenses, and a **private** provenance note (internal id → public
name; strip `PROVENANCE_INTERNAL.md` before public release and re-hash).

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

- **No credentials / hostnames / private paths** in shipped code. All former
  `/scratch2/...`, model dirs and device pins are env-configurable
  (`SCRATCH_BASE`, `DATA_DIR`, `H5_PATH`, `MODEL_DIR`, `MODEL_BASE`, `DEVICE`).
- **Environment:** `env/requirements.lock.txt` (Python 3.14.6; torch 2.10.0+cu130;
  or the Euler stack torch 2.3.1+cu121). `env/RUNTIME.md` per bundle.
- **Files ≤ 50 MB** shipped directly (incl. the two NIC-XS checkpoints and the
  354 MB NIC-PM per-model checkpoint set). **Larger** artifacts (the ~8.9 GB A5
  checkpoint set for NIC-RES/NIC-AER; the normalized data cubes) are listed in
  each `data/EXTERNAL_MANIFEST.csv` for hosting at immutable gcmagicc.org URLs
  with byte size + SHA-256 (fields marked TBD until hosted).
- **Licensing:** code © Nicolai Meinshausen, Apache-2.0; data/figures/docs CC BY 4.0.

## ⚠ Outstanding blocker for full regeneration

The normalized input cube `/scratch2/userdata/nicolai/cmip6/normalized_7Augext/`
(the `data_DAT_*.h5` files holding `y64` + the `X` predictor matrix) and the raw
vetted prep sources are **no longer on the authors' host** (scratch purge). They
are required for the GPU regeneration of **NIC-XS**, **NIC-AER**, and the optional
**NIC-PM** trend step, and the **NIC-RES** render step. To finish the "regenerate"
half, point the pipelines at the current data location (ada / Euler `work/math`)
via `DATA_DIR`/`H5_PATH`, or rebuild via `prep_data_new.py`. NIC-RES Part B
(compute scaling) and the NIC-PM figure do **not** need this data.

## Regeneration (when data is available)

Follow `gcm_sequence/EULER.md`: scp `code/` to Euler, `module load`, submit with
pinned GPU + `--preload`, write to `/cluster/scratch/nicolai/gcm_runs`. A generic
sbatch template is in `_shared/regenerate.sbatch.template`. NIC-XS/AER used ada
(torch-2.10 build) historically; either stack reproduces up to float noise.

## Acceptance

Each bundle is validated in a clean env: fresh venv from `env/requirements.lock.txt`,
fetch `EXTERNAL_MANIFEST.csv` files and verify their size+SHA-256, run `run.sh`,
recompute SHA-256 of regenerated outputs and diff against `SHA256SUMS.txt`. NIC-PM
already passes this end-to-end from shipped files.
