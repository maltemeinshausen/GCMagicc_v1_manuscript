# NIC-AER — aerosol-pattern intervention experiment

**Public model name:** **GCMagicc** (default production emulator, A5 lineage).

## What this bundle computes (the science)

An **aerosol-forcing intervention**: run the emulator with the actual (factual)
forcings, then re-run with the aerosol effective radiative forcing switched off,
and map the difference. This isolates the spatial pattern of the climate response
attributable to anthropogenic aerosol forcing under an otherwise-identical world.

## Factual / counterfactual / reference definitions

For a fixed forcing epoch (predictors frozen to a chosen time step so the
comparison is a clean single-forcing contrast):

- **Factual `x_actual`** — real predictor row, all forcings present. Forcings held
  constant over the analysed window (`x[:, 4:] = x[<epoch>, 4:]`).
- **Aerosol-only counterfactual `x_cf`** — identical to factual **except the
  aerosol ERF channel is zeroed**: `x_cf[:, 13] = 0` (column 13 = `aer_ERF`).
- **Reference/baseline** — the factual run is the reference against which the
  aerosol-off run is differenced.

The emulator is sampled `nsim = 100` times for each of factual and counterfactual,
over `MC_RANGE = 0..19` model-index Monte-Carlo settings, at `nside = 64`. Target
variable is **`tasmax`** with `subtract_tasmin = True`, i.e. the output field is
`tasmax − tasmin` (diurnal temperature range) unless configured otherwise.

## Quantities (units, sign)

Per pixel (HEALPix nside=64), inverse-affine to physical units:
- `yh_mean`  = mean over draws of the **factual** field.
- `ycf_mean` = mean over draws of the **aerosol-off** counterfactual field.
- **`delta_mean = yh_mean − ycf_mean`** = the aerosol-attributable response
  (factual minus aerosol-off). Sign: positive where aerosol forcing raises the
  field, negative where it lowers it. Units follow the target variable (K for the
  `tasmax`/`tasmin`-derived field).
- Uncertainty is the spread across the `MC_RANGE` model-index members (and the
  `nsim` draws). Note: the shipped generator writes the **mean** map only
  (`delta_std` is not currently emitted — documented limitation; add a std map in
  a re-run if required).

## Aerosol ERF source

`aer_ERF` is **column 13** of the predictor matrix `X` inside the normalized
`data_DAT_*.h5` cube (produced upstream by the data-prep pipeline). There is no
separate aerosol file and **no extra pattern scaling or normalization** is applied
beyond the standard predictor affine normalization; the intervention is simply
zeroing that column.

## Pipeline & outputs

```
data_DAT_*.h5 (X col 13 = aer_ERF)  ─►  cfAllAerJson.py  ─►  DAER3_tasmax_mean/
   + A5 checkpoints                        (factual vs           aer_ERF_<mc>_Nxl_tasmax.json   (zlib+b64 delta_mean)
                                            aerosol-off,          aer_ERF_<mc>_Nxl_tasmax_delta_mean.pdf
                                            nsim=100, MC 0..19)
```

Each JSON records `meta` (nsim, subtract_tasmin, v_absmax, …) and the zlib+base64
float32 `delta_mean` HEALPix map.

## Figure interpretation

A global (or Europe-zoom) map of `delta_mean`: the aerosol-attributable change in
the target field. Diverging colormap centred at zero; blue/red = aerosol cooling /
warming of the field. `v_absmax` in the JSON meta sets the symmetric colour range.

## Model / checkpoints / config

- Sampler `sample_Nxl` (inline in `cfAllAerJson.py`), A5 lineage.
- Checkpoints: A5 set `modelsNfour_7Augext_*` (~8.9 GB, external — see
  `EXTERNAL_MANIFEST.csv`); point `MODEL_DIR` at it.
- Config: `config/meta_7Augext.pkl`, `config/ranges_7Augext.pt`.

## Reproduce / Licensing

See `run.sh`. Code © Nicolai Meinshausen, Apache-2.0; data/figures/docs CC BY 4.0.
