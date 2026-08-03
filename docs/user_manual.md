# User manual

## What is included

`src/gcmagicc_model/` contains frozen model code for the two verified public configurations (GCMagicc and GCMagicc-CE). The trained network weights are too large to commit (16.94 GB across 61 files) and are published as external release objects with a persistent DOI; see [`checkpoints.md`](checkpoints.md) for the full inventory, the SHA-256 of every file, the expected directory layout, and how to run inference once they are fetched. `src/gcmagicc_eval/workflows/` contains frozen predictor, ensemble, drought, regional-scenario, and figure workflows. `src/gcmagicc_repro/` provides the stable release interface.

## Installation

Requires Python 3.11 or later. From the repository root:

```sh
python -m venv .venv && . .venv/bin/activate
python -m pip install -e .
python -m pip install -e '.[analysis]'   # xarray, torch and friends, for full inference
python -m pip install -e '.[test]'       # pytest, to run the test suite
```

Inference at the default `nside=64` runs on CPU. Peak resident memory scales with the length of the rollout rather than with checkpoint size: a full 100-year (1200-month) member generated in a single call peaks at about 140 GB, while emitting it in ten-year chunks peaks at about 23 GB for roughly 20% more wall-clock time -- the recommended configuration on memory-limited hardware. See `docs/benchmark.md` for the measured figures. A GPU is optional
(`device="cuda"`). Model, training and inference code and the trained weights are licensed
Apache-2.0; data, figures and documentation are licensed CC BY 4.0. See
[`licensing_and_authorship.md`](licensing_and_authorship.md).

## Commands

- `python -m gcmagicc_repro fetch`: download every published external object into its declared destination and verify byte count and SHA-256. Pending entries are reported and cause a non-zero result.
- `python -m gcmagicc_repro smoke`: run deterministic scientific-kernel and release-layout checks.
- `python -m gcmagicc_repro reproduce --figure ID`: run a registered figure workflow after checking required inputs.
- `python -m gcmagicc_repro verify`: verify file-size policy, provenance hashes, license files, external manifest schema, and natural-forcing metadata.

## Full model inference

Install the `analysis` optional dependencies, run `fetch`, and use the frozen full-ensemble workflow with a scenario predictor file. The public release manifest is the authority for model/checkpoint mapping. Do not substitute an internal development directory based on its name.

## Drought protocol

The revision protocol uses simplified unadjusted Thornthwaite, modified Hargreaves with the 0.408 energy-to-water conversion, and FAO-56 Penman–Monteith. The event is the Iranian area-weighted December 2025 SPEI-48 value; uncertainty uses 10,000 fixed-seed hierarchical bootstrap replicates over members and moving five-year blocks.

The common-protocol rerun is registered as `drought-common-protocol`. Configure the three large local or downloaded inputs without editing the script:

```sh
export GCMAGICC_CMIP6_ROOT=/path/to/vetted/cmip6
export GCMAGICC_ENSEMBLE_ROOT=/path/to/gcmagicc/ssp245/all-and-natural
export GCMAGICC_ERA5_FILE=/path/to/era5-monthly.nc
python -m gcmagicc_repro reproduce --figure drought-common-protocol
```

The direct CMIP6 comparison uses CanESM5, MIROC6, and GISS-E2-1-G. Its common factual/natural-only probability window is 1995--2014 because the GISS-E2-1-G natural-only archive ends in 2014. GCMagicc retains 2021--2025 and 2041--2060. Input roots can also be supplied through explicit CLI options; the output manifest records source filenames, byte sizes, timestamps, protocol settings, and hashes of every derived table and figure.

The corrected seven-panel main figure can be regenerated without the large source NetCDF files:

```sh
python -m gcmagicc_repro reproduce --figure drought-main-figure
```

It restores the bordered ERA5 event map, separate GCMagicc `ssp245` and `ssp245-nat` ensemble series, recent/future histograms, and thin red CMIP6 sidecar context from the former 1040 design. Its final two panels use only the corrected common-protocol event probabilities and three-SMILE probability ratios. The frozen map artifact records the exact ERA5 source hash, 1991--2010 baseline, Penman–Monteith calculation, area-weighted series, and clipped Natural Earth boundaries. A second compact artifact records the former sidecar hash and the 54 factual and nine natural-only CMIP6 regional lines; those lines are visual context only and are not used in the corrected attribution calculations.

## Türkiye regional application

The release includes annual Türkiye projections for all 25 frozen pathways and for `tas`, `pr`, and `hurs`. The 3-by-3 figure separates CMIP6 SSPs, NDC/SSP2-com/current-policy pathways, and CMIP7 scenarios by column and overlays ERA5 in black through 2025. Reproduce it without any sibling repository:

```sh
python -m gcmagicc_repro reproduce --figure turkiye
```

The Türkiye workflow reports medians and 5--95% ranges, uses 1995--2014 as its baseline and 2081--2100 as its future period, and writes a JSON summary containing the 20-member median changes and SHA-256 hashes.

## Workflow schematic

Regenerate the vector-native training and inference schematic with:

```sh
python -m gcmagicc_repro reproduce --figure workflow
```

The figure separates the one-time CMIP6/ERA5 training and held-out evaluation from scenario inference. It also records the fixed two-pass correction used by the full-predictor variants: only `tas_smoothed` changes before a single same-seed rerun; GCMagicc-PM and GCMagicc-XS use reduced predictors and one emulator pass.

## Validation database audit

The 11.9 GB metrics database is not committed. `1110_metrics_database_audit.py` opens it with SQLite read-only and immutable flags, groups `gofnc` records by public version, recipe domain, and experiment, and writes a small auditable JSON artifact containing the exact SQL, source byte size, and SHA-256. The frozen audit underlying the manuscript contains 4,539,079 records, of which 1,265,222 are SSP2-4.5 hold-out records.

## Emergent constraints

The release-native command reads the frozen prepared trend tables, reconstructs the four-panel GCMagicc-PM figure, and writes its plotted-data composite:

```sh
python -m gcmagicc_repro reproduce --figure emergent
```

Warming is `(2081--2100 mean - 1995--2014 mean) + 0.85 degC`, expressed relative to 1850--1900. The calibrated scenario distributions use 2,000 replacement draws with seed 0. The release fully covers prepared-data-to-figure reproduction. Raw CMIP6, ERA5, and GCMagicc-PM checkpoint processing into the 799-row trend table remains outside the release and is recorded under `gcmagicc-pm-bundle`.

## GCMagicc-XS compact replot

The ten prediction-skill plots can be regenerated from the deterministic compact plotted-point table:

```sh
python -m gcmagicc_repro reproduce --figure xs
```

The external 1.53-GB monthly CSV remains checksum-locked for full raw recomputation under `gcmagicc-xs-bundle`.

## GCMagicc-CE sensitivity figures

The selected resolution and aerosol artifacts are frozen with normalized metadata. Recomputing their raw maps requires `gcmagicc-ce-checkpoints`. The aerosol diagnostic is the full-forcing minus zero-aerosol-ERF change in `tasmax - tasmin` in K at `nside=64`, based on 100 stochastic samples; 2015--2024 is explicitly provisional.

## Reproducibility boundary

The frozen workflows retain scientific source history, while the release CLI provides stable discovery, fetch, verification, and dispatch. A figure whose manifest status is pending cannot be represented as reproducible and must be omitted at manuscript freeze.
