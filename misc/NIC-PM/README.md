# NIC-PM — perfect-model & emergent-constraint experiment

**Public model name:** **GCMagicc-PM** (perfect-model; per-CMIP6-model
historical-only retraining) with a **leave-one-family-out (LOFO)** cross-validation
producing the emergent-constraint prediction intervals.

> **Reproduction status:** the figure/quantile step reproduces **byte-identically**
> from the shipped trend tables (verified clean-room, seed 0, nsim 2000). The
> upstream trend regeneration (optional) needs the external normalized data.

## What this bundle computes (the science)

Two linked results:

1. **Perfect-model structural uncertainty** — GCMagicc-PM is retrained on each
   CMIP6 model's historical run alone (reduced "nosmooth" predictor set). The
   spread of per-model end-of-century deviations gives a structural-uncertainty
   range that, combined with the ERA5-informed central estimate, produces the
   small **black uncertainty bars** in the main-text era5-splicing figure.

2. **Emergent constraint via LOFO** — for each held-out model family, the
   emulator's warming prediction is compared against the model's own warming, and
   the residual `(trend_y_true − trend_y_hat)` is collected across folds. A
   prediction distribution is then built as **ERA5-informed central value
   (`trend_yera_hat`) + a resampled cross-validated residual**, giving calibrated
   5–95% (and 2.5/97.5%) warming intervals per scenario.

## Families / splits

- **Family = first 4 characters of the CMIP6 `source_id`** (e.g. `ACCE`, `CanE`,
  `CESM`, `EC-E`, `FGOA`, `GFDL`, `GISS`, `HadG`, `INM-`, `IPSL`, `MIRO`, `SAM0`,
  `UKES`) → **13 families** partitioning the 33 ids (32 CMIP6 + ERA5). Mapping
  driven by `config/UniqueModels_10Jun.csv` (`source_id, model_index`). The
  release and manuscript use this 13-family definition consistently.
- LOFO: hold out one family, train on the rest, predict the held-out family;
  repeat for all families.

## Warming metric

Per series: baseline **1995–2014** (`samplesSera.py`: `i0,i1 = 1740,1980`
month indices), target **2081–2100** (last 240 months). Global means use
**HEALPix equal-area pixels** (nside=1, 12 pixels) — not cosine-latitude weights.
`trend_y_true` = target-ESM warming; `trend_y_hat` = emulator warming;
`trend_yera_hat` = ERA5-forced emulator warming.

## Pipeline & outputs

```
per-model/ERA5 checkpoints (modelsSsmall/, shipped)   [OPTIONAL upstream, GPU + external data]
        │  samplesSera.py  (niter=3, forceEraModel toggles ERA-forced vs SSP)
        ▼
figchangeeraSsmall_1/trends_ssp_True_6.csv            [SHIPPED in data/]
figchangeforceeraSsmall_1/trends_ssp<NNN>_True_6.csv  [SHIPPED in data/]
        │  sample_emergent2.py  (seed 0, nsim 2000)    [CPU-only, fully reproducible]
        ▼
outputs/warming_panels_<scen>.pdf (+ _left/_right)  +  quantiles_by_scenario.csv
```

Columns of the trend CSVs: `version, iter, trend_y_true, trend_y_truesecond,
trend_y_hat, trend_yera_hat`. `quantiles_by_scenario.csv`: rows = 2.5/5/10/90/95/
97.5%, columns = ssp119/126/370/434/460/585.

Also shipped: `sample_emergent.py` (single-scenario variant) and
`sample_era_plus_residuals.py` (minimal reference implementation of the
ERA-central + residual prediction interval).

## Figure interpretation

`warming_panels_<scen>.pdf` is a two-panel scatter: **left** — target-ESM warming
vs a second realisation (internal-variability reference); **right** — target-ESM
warming (`trend_y_true`, y) vs emulator warming (`trend_y_hat`, x), with an ERA
"rug" of `trend_yera_hat` values along the axis. Points = CMIP6 models; the ERA
rug + residual spread define the emergent-constraint prediction interval. The
`quantiles_by_scenario.csv` rows are that interval (the black-bar / whisker values).

## Model / checkpoints / config

- Figure step imports no model modules (pure CSV→figure).
- Upstream `samplesSera.py` imports `help_functions.DownsampleWithNoise`,
  `train_functions.BiasSum`, `models.{LB2,L4de,L4adde,LB2mini,L4addemini,L4des}`.
- Checkpoints: `checkpoints/modelsSsmall/modelsS_<source_id>/` (per-model + ERA5;
  ~354 MB, **shipped in the bundle**).
- Config: `config/meta_7Augext.pkl`, `config/ranges_7Augext.pt`,
  `config/UniqueModels_10Jun.csv`.

## Reproduce / Licensing

See `run.sh` (figure step needs no external data). Code © Nicolai Meinshausen,
Apache-2.0; data/figures/docs CC BY 4.0.
