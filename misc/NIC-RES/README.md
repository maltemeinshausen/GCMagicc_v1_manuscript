# NIC-RES — resolution experiment (spatial multi-resolution + compute scaling)

**Public model name:** **GCMagicc** (default production emulator, A5 lineage).

This bundle has **two complementary parts** (hybrid figure):

## Part A — spatial multi-resolution rendering (`plotResolutions.py`)

GCMagicc solves the sphere as a HEALPix cascade, coarsest first
(`nside = 1 → 2 → 4 → 8 → 16 → 32 → 64 → 128 → 256`; `npix = 12·nside²`), each
level refining the previous. This renderer runs one inference and saves, for a
chosen time index and variable, the field at **every** HEALPix level, so the same
climate state can be shown from 12 pixels up to 786 432 pixels.

Outputs per `(variable, time index)`: `RESOLUTIONPLOTS/<var>/json/t<t:06d>_h<hash>_levels.json`
and matching PDFs, plus one `RESOLUTIONPLOTS/run_meta.json`.

`*_levels.json` schema:
- `meta`: `DATE, MODEL, MODEL_DIR, H5_PATH, DEVICE, DEPENDENCE, NSIDE_MAX,
  MC (Monte-Carlo draws), t_index_global, years_tail, seed_timepoints, variable,
  channel_index_in_vars_order, vars_order[10], x_row[15] (the forcing predictors
  used), europe_extent, colormap, vmin_vmax, healpix_sampling`.
- `levels`: keyed by nside `1..256`, each `{nside, npix, map: <zlib+base64 float32
  HEALPix RING array>}`.

**Grids / regridding:** targets live on HEALPix RING; the comparison lat/lon grid
is `GLOBAL_NLAT` (see `run_meta.json`); `healpix_to_latlon_grid` /
`hp.get_interp_val` performs the interpolation for display. Training resolution is
nside=64 (the default trained output); nside=128/256 use the `*mini` refinement
heads. Document each rendered nside, variable, period and sample count from the
JSON `meta`.

## Part B — compute-cost scaling (`bench_resolution.py`, new for this release)

Measures **end-to-end inference wall-time and peak memory as a function of output
nside**. Wraps the multi-resolution sampler (`run_helpers.sample_from_combined_model`).

- Inputs: only model checkpoints + a forcing matrix `x` (T × 15). It does **not**
  need target data — pass a representative `x` via `--x-file`, or let it
  synthesise a smooth `x` of `--months` length (the numbers only drive compute
  cost, not scientific output).
- Measurement boundaries (matched across nside): the HEALPix→lat/lon reprojection
  is excluded (`rectangular=False`); each timed call includes per-level checkpoint
  load + forward cascade to the target nside (this is the production invocation —
  it is **not** a forward-only microbenchmark). `--warmup` untimed iterations warm
  the page/CUDA caches; `--reps` timed iterations report **median + IQR**.
  Precision fp32; GPU pinned via `--device`.
- Outputs: `outputs/resolution_timing.csv` (`nside, npix, months, wall_s_median,
  wall_s_iqr, peak_mem_bytes, reps, warmup, device, dtype`) and
  `outputs/resolution_timing.pdf` (time-vs-nside and memory-vs-nside panels).

**Figure interpretation:** left panel — median inference time per nside (log axes),
error bars = IQR over reps; right panel — peak memory per nside. The plotted metric
is wall-clock median; uncertainty is the inter-quartile range across `--reps`
repetitions after `--warmup` warm-ups.

**Key empirical caveat (cascade floor).** The A5 sampler
(`run_helpers.sample_from_combined_model`) always executes the full HEALPix
cascade through nside=64 irrespective of the requested target nside ≤ 64, so the
coarse levels (1..32) are produced *en route* at no extra cost. Consequently the
measured time and peak memory are **essentially constant for nside 1..64** (e.g.
nside=1 and nside=2 both ≈ 5.3 s at ~18.7 GB peak for a 10-yr rollout on an
RTX-4090), and inference cost grows only when the nside=128/256 refinement heads
are added. The near-flat 1..64 segment is therefore a genuine property of the
multi-resolution design (coarse fields are a free by-product of computing the
fine field), not a measurement artefact — read the panel accordingly.

## Model / checkpoints / config

- Sampler: `run_helpers.py::sample_from_combined_model` (A5). Per-level bias
  (`LB2`/`LB2mini`) + generator (`L4de`/`L4adde`/`L4addemini`).
- Checkpoints: the full A5 set `modelsNfour_7Augext_*` (36 dirs, ~8.9 GB) —
  hosted externally (see `EXTERNAL_MANIFEST.csv`); point `MODEL_DIR` at it.
- Config: `config/meta_7Augext.pkl`, `config/ranges_7Augext.pt`.

## Reproduce

See `run.sh`. Part B is runnable with just the checkpoints; Part A additionally
needs one normalized `data_DAT_*.h5` input (external).

## Licensing

Code © Nicolai Meinshausen, Apache-2.0. Data/figures/docs: CC BY 4.0.
