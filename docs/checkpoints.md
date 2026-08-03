# Trained model checkpoints

GCMagicc inference requires trained network weights. They are far too large for a git repository
(16.94 GB across 61 files; the largest single file is 2.13 GB), so they are published as external
release objects with a persistent DOI and fetched on demand.

## The two public models

| Public model | Code | Internal snapshot of record | Files | Size |
|---|---|---|---|---|
| **GCMagicc** | `src/gcmagicc_model/gcmagicc/` | `model_NxlversA5` (`modelsNfour_7Augext_*`) | 37 | 8.52 GB |
| **GCMagicc-CE** | `src/gcmagicc_model/gcmagicc_ce/` | `model_NthreeversT1` (`modelsNthree_7Augext_*`) | 24 | 8.42 GB |

The internal snapshot names are recorded for provenance only. **The manifest, not a directory name,
is the authority for which checkpoint belongs to which public model** — see
`provenance/README.md`. Publishing these files with their SHA-256 values makes that mapping
independently verifiable rather than merely asserted.

## What is published, and what is not

Each internal snapshot contains a wider hyperparameter sweep than the paper uses (36 and 27 variant
directories respectively, ~22 GB with optimizer backups and `__pycache__`). Published here is
**only the file set the released inference code actually resolves**:

1. **Multiresolution cascade** — `best_model.pt` for each `_b<nside>` bias model and its paired
   generator (`_bdadde<nside>`, `_bS<nside>`), covering nside 1 through 256. These are loaded by
   `sample_from_combined_model()` in `run.py`.
2. **Dependence and lag models** — the `_bde*` / `_l*` variants used when `dependence=True`.
3. **Normalisation files** — `ranges_7Augext.pt` (variable min/max per nside) and
   `meta_7Augext.pkl` (transformation scalars and variable list). Both are small and are already
   committed in this repository; they are listed here so the manifest is complete.

Deliberately **not** published: optimizer state backups (`model_backup.pt`), `__pycache__`, the
`_m*` and `_stacked` experimental variants, and sweep variants no released code path loads. None
is required to reproduce any published result. This is a scope decision, not a restriction — the
excluded files carry no information needed to rerun the paper.

## Checkpoint inventory

### GCMagicc — multiresolution cascade

| Variant | Role | Size |
|---|---|---|
| `modelsNfour_7Augext_b1` | cascade | 0.2 MB |
| `modelsNfour_7Augext_bS2` | cascade | 0.9 MB |
| `modelsNfour_7Augext_b2` | cascade | 1.9 MB |
| `modelsNfour_7Augext_b4` | cascade | 8.5 MB |
| `modelsNfour_7Augext_b8` | cascade | 34.1 MB |
| `modelsNfour_7Augext_bdadde16` | cascade | 94.1 MB |
| `modelsNfour_7Augext_b16` | cascade | 136.4 MB |
| `modelsNfour_7Augext_b128` | cascade | 202.9 MB |
| `modelsNfour_7Augext_bdadde32` | cascade | 376.5 MB |
| `modelsNfour_7Augext_bdadde128` | cascade | 499.5 MB |
| `modelsNfour_7Augext_b32` | cascade | 545.7 MB |
| `modelsNfour_7Augext_b256` | cascade | 811.5 MB |
| `modelsNfour_7Augext_bdadde64` | cascade | 1,506.0 MB |
| `modelsNfour_7Augext_bdadde256` | cascade | 1,998.0 MB |
| `modelsNfour_7Augext_b64` | cascade | 2,182.9 MB |

Plus 20 dependence/lag models (323 MB total) and 2 normalisation files.

### GCMagicc-CE — multiresolution cascade

| Variant | Role | Size |
|---|---|---|
| `modelsNthree_7Augext_b1` | cascade | 0.2 MB |
| `modelsNthree_7Augext_b2` | cascade | 1.9 MB |
| `modelsNthree_7Augext_b4` | cascade | 8.5 MB |
| `modelsNthree_7Augext_b8` | cascade | 34.1 MB |
| `modelsNthree_7Augext_bdadde16` | cascade | 94.1 MB |
| `modelsNthree_7Augext_b16` | cascade | 136.4 MB |
| `modelsNthree_7Augext_b128` | cascade | 202.9 MB |
| `modelsNthree_7Augext_bdadde32` | cascade | 376.5 MB |
| `modelsNthree_7Augext_bdadde128` | cascade | 499.5 MB |
| `modelsNthree_7Augext_b32` | cascade | 545.7 MB |
| `modelsNthree_7Augext_b256` | cascade | 811.5 MB |
| `modelsNthree_7Augext_bdadde64` | cascade | 1,506.0 MB |
| `modelsNthree_7Augext_bdadde256` | cascade | 1,998.0 MB |
| `modelsNthree_7Augext_b64` | cascade | 2,182.9 MB |

Plus 8 dependence/lag models (221 MB total) and 2 normalisation files.

The complete list with a SHA-256 for every file is in `data/checkpoint_manifest.json`.

## Obtaining the checkpoints

```sh
python -m gcmagicc_repro fetch
```

Each public model is distributed as a single `.tar.gz`, so one URL, one size and one SHA-256
fully identify it. `fetch` downloads that archive to the `destination` recorded in
`data/external_data_manifest.json`, checks its size and SHA-256, then — because the entry carries
an `extract_to` field — unpacks it **at the repository root** and **re-verifies every extracted
file** against its own SHA-256 in `data/checkpoint_manifest.json`. The archive is deleted once all
files verify.

Member names inside the archive are repository-root-relative (they are exactly the `path` field of
each manifest entry), which is why extraction happens at the root rather than inside
`src/gcmagicc_model/<model>/`. The `extract_to` field declares the subtree the archive is permitted
to write into and is enforced: a member resolving outside it is refused.

Both checks are hard errors. A checkpoint that does not match its recorded hash is not the
checkpoint the paper used, so a mismatch aborts rather than leaving a partially trusted tree in
place. Path traversal, links, and members outside the declared subtree are all refused.

To place the files by hand instead, unpack the archive at the repository root; the paths inside it
are already relative to it and match the `path` field of each manifest entry. The layout the
inference code expects is:

```
src/gcmagicc_model/gcmagicc/
├── meta_7Augext.pkl
├── ranges_7Augext.pt
└── modelsA/
    ├── modelsNfour_7Augext_b1/best_model.pt
    ├── modelsNfour_7Augext_b2/best_model.pt
    ├── modelsNfour_7Augext_bdadde16/best_model.pt
    └── ...
```

## Running inference

```python
from gcmagicc_model.gcmagicc.run import sample_from_combined_model

yh = sample_from_combined_model(
    x_tensor,            # predictor tensor from the MAGICC predictor export
    device="cpu",        # or "cuda"
    dependence=False,    # True engages the lag/dependence models
    nside=64,            # 64 is the documented default; 128/256 need the high-nside cascade
    rectangular=True,    # False returns HEALPix ordering
    nlat=180,
    seed=None,           # set an integer for reproducible draws
)
```

Resolution note: `nside=64` is the default and is what the published figures use. `nside=128` and
`nside=256` require the correspondingly larger cascade entries, which are included in this release.

## Hardware

Inference at `nside=64` runs on CPU. Peak memory is governed by the largest cascade entry loaded
(2.13 GB for `_b64`). Note that peak memory during inference is dominated by the rollout, not by the checkpoints: see `docs/benchmark.md` for measured figures (about 140 GB for a full 100-year member in one call, about 23 GB when chunked). A GPU is optional and
is selected with `device="cuda"`.

## Licence and attribution

Model, training and inference code and the trained weights are copyright **Nicolai Meinshausen**
and released under **Apache-2.0**. See `docs/licensing_and_authorship.md` for the split between
model code and the evaluation/portal code in this release.
