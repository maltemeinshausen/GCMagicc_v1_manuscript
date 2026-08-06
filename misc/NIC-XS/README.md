# NIC-XS — GCMagicc-XS extrapolation-skill evaluation

**Public model name:** **GCMagicc-XS** (the reduced-predictor, "forcing-only"
emulator; the deprecated alias *GCMagicc-F* appears in an older methods draft —
they denote the same model).

## What this bundle computes (the science)

GCMagicc-XS is a variant of the GCMagicc generative climate emulator that omits
the two MAGICC-derived smoothed global indicators (`tas_smoothed`,
`rtmt_smoothed`) from its predictor set and is trained with the high-warming /
idealised experiments held out. Running it forward on those held-out experiments
is a genuine **out-of-distribution extrapolation test**: can the emulator
reproduce a driving ESM's climate under forcings stronger than anything it saw
in training?

For each driving CMIP6 model and each of the 10 target variables, the emulator
produces a stochastic ensemble of global-mean monthly timeseries. We compare
these against the driving ESM's own output ("observed" here = the target ESM),
for both **training** scenarios and the **held-out** scenarios. The held-out set
is the extrapolation test; the training set is a fidelity check.

## Held-out (test) vs training experiments

- **Held out from training (the extrapolation test):** `ssp585`, `ssp534-over`,
  `abrupt-4xCO2`. Historical is always retained. `ssp245` is additionally an
  unseen middle-of-the-road fidelity case.
- The eval script labels each row `is_test = True/False` accordingly. In the
  figure, held-out scenarios are drawn solid/emphasised; training scenarios muted.

## Pipeline

```
normalized data_DAT_*.h5  ──►  eval_nside2_nosmooth_miss585.py  ──►  full_monthly_results.csv
   (per model × scenario)          (GPU: bias LB2 + generator L4de,        member_draws.npz
                                     n_draws stochastic realisations)
                                          │
                                          ▼
                              1080_Figure_GCMAGICC-XS_predictionskill.py
                                          │
                    Figure_GCMAGICC-XS_predictionskill_<var>.pdf  +  xs_prediction_skill_metrics.csv
```

`eval_nside2_nosmooth_miss585.py` (modified for release) loads the two
GCMagicc-XS checkpoints, generates `--n-draws` (default **200**) independent
stochastic draws per model×scenario file, takes the HEALPix global mean, inverts
the affine normalization to physical units, and writes:

- `full_monthly_results.csv` — columns `model, scenario, is_test, variable,
  month_idx, obs, pred_mean, pred_p05, pred_p10, pred_p50, pred_p90, pred_p95,
  n_draws`. **The p05/p95 columns give the agreed 5–95% prediction interval.**
- `member_draws.npz` — raw global-mean member draws (physical units) for the
  held-out scenarios, keyed `"<scenario>|<model>|<variable>" -> array (n_draws, T)`,
  so any quantile is recomputable offline without re-running the GPU.

`1080_Figure_GCMAGICC-XS_predictionskill.py` is a pure CSV→figure transform
(no model, CPU-only).

## Figure interpretation (panel / line / band / metric)

Per variable, one small panel per driving CMIP6 model (annual global means):
- **Black line** — the driving ESM ("observed"); solid for held-out scenarios,
  faded for training scenarios.
- **Red line** — GCMagicc-XS ensemble mean for **held-out** scenarios (the
  extrapolation cases). **Blue line** — ensemble mean for training scenarios.
- **Shaded band** — the **5–95%** prediction interval across the stochastic
  draws (falls back to 10–90% and is relabelled if an older CSV lacks p05/p95).
- **Panel title in red + "*OUTLIER*"** — a model flagged by the outlier rule.

`xs_prediction_skill_metrics.csv` (per model×scenario×variable): `obs_mean,
pred_mean, interval_width_mean, band, coverage` (fraction of months with obs
inside the band; well-calibrated ≈ 0.90 for a 5–95% band), `rmse, bias, r2, n,
rmse_outlier`.

**Outlier-naming rule (deterministic):** per variable, over held-out rows, a
model is a *material outlier* if its RMSE exceeds
`median(RMSE) + 1.5 · IQR(RMSE)` across models.

**Units / transforms:** metrics are computed in physical units after the inverse
affine transform (Table `tab:affine` in the manuscript); no further transform.

## Model / checkpoints / config

- Architecture (nside=2): bias `LB2(nside_lo=0, nside_hi=2)`; stochastic
  generator `L4de(nside=2, x_features=13, variab=10, maxlag=72,
  lags=[1,2,3,4,5,6,8,10,12,14,16,18,20,24,27,30,33,36,42,48,54,60,66,72],
  add_latent_dim=50)`. `x_features=13` because the two smoothed global indicators
  are dropped (`X_COLS = [0,1,2,3,6,7,8,9,10,11,12,13,14]`).
- Checkpoints (shipped in `checkpoints/`, both ≤ 50 MB):
  `modelsNfour_7Augext_miss585_nosmooth_bS2/best_model.pt` (bias) and
  `modelsNfour_7Augext_miss585_nosmooth_bdeSxlsp2_72_10/best_model.pt` (generator).
- Config: `config/meta_7Augext.pkl` (transformation scalars, variable order),
  `config/ranges_7Augext.pt` (clamp ranges).

## Inputs (see `data/EXTERNAL_MANIFEST.csv`)

The normalized `data_DAT_<model>_<scenario>_*.h5` inputs are large and hosted
externally. Each file holds `y64` (target, HEALPix nside=64, 10 vars) and `X`
(predictor matrix, 15 columns). Source: CMIP6 ESGF model output + ERA5, regridded
to 1°×1° then HEALPix, affine-normalized. See `data/INPUT_MANIFEST.csv` for the
per-file variable/unit/period/member details and `PROVENANCE_INTERNAL.md` for the
internal id mapping.

## Reproduce

See `run.sh`. Seeds and the software environment are pinned; on matched hardware
the draws and quantiles are bit-reproducible.

## Licensing

Code © Nicolai Meinshausen, Apache-2.0 (`LICENSE-Apache-2.0.txt`). Data, figures,
documentation: CC BY 4.0 (`LICENSE-CC-BY-4.0.txt`).
