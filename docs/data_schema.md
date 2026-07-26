# Data schemas

## External object manifest

Each object has `id`, final public model name, `status`, immutable `url`, exact `sha256`, exact byte count, and repository-relative `destination`. Published objects require all fields. Pending objects intentionally use null URL/hash/size.

## Natural forcing CSV

`natural_forcing_ssp245_ar6_run0_1850-2100.csv` contains one row per year and columns `solar_erf_w_m-2` and `volcanic_erf_w_m-2`. It is derived from the exact AR6, runmode-natural, SSP2-4.5, resampled-100 artifact for run ID 0.

## Figure workflow contract

Every registered workflow declares the frozen script, required external object IDs, output directory, configuration/arguments, and scientific status. Successful release records include command, environment, input SHA-256 values, and output SHA-256 values.

## Emergent-constraint prepared data

`data/derived/emergent_constraints/model_trends.csv` contains 799 rows and 19 models. The six `era5_conditioned_<scenario>.csv` files contain the ERA5-conditioned estimates. `quantiles_by_scenario.csv` contains the seven required probability levels. The 50th percentiles are deterministically reconstructed with 2,000 replacement draws and seed 0 by adding a randomly selected ERA5-conditioned estimate to a randomly selected pooled cross-validated residual.

## Compact GCMagicc-XS plotted points

`data/derived/gcmagicc_xs/compact_plotted_points.csv` uses the eight-column input schema consumed by the prediction-skill workflow: `model`, `scenario`, `is_test`, `variable`, `obs`, `pred_mean`, `pred_p10`, and `pred_p90`. It is a deterministic 1-in-64 row-hash sample of the external monthly CSV after excluding the two abrupt-CO2 scenarios, with at least one row retained for every model-scenario-test-variable group.

## Semantic provenance

`provenance/figure_registry.csv` links each manuscript role to its selected artifact, prepared-data directory, raw external dependency, and scientific status. `provenance/ada_data_audit.csv` records source host, normalized source path, byte count, checksum, model variant, licence, disposition, and remaining blocker. Neither file encodes manuscript figure numbers.
