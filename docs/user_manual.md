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

## Reproducibility boundary

The frozen workflows retain scientific source history, while the release CLI provides stable discovery, fetch, verification, and dispatch. A figure whose manifest status is pending cannot be represented as reproducible and must be omitted at manuscript freeze.
