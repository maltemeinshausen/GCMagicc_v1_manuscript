# User manual

## What is included

`src/gcmagicc_model/` contains frozen model code for the two verified public configurations. Large learned checkpoints are external release objects. `src/gcmagicc_eval/workflows/` contains frozen predictor, ensemble, drought, regional-scenario, and figure workflows. `src/gcmagicc_repro/` provides the stable release interface.

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

## Reproducibility boundary

The frozen workflows retain scientific source history, while the release CLI provides stable discovery, fetch, verification, and dispatch. A figure whose manifest status is pending cannot be represented as reproducible and must be omitted at manuscript freeze.
