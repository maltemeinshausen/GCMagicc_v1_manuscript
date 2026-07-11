# Data schemas

## External object manifest

Each object has `id`, final public model name, `status`, immutable `url`, exact `sha256`, exact byte count, and repository-relative `destination`. Published objects require all fields. Pending objects intentionally use null URL/hash/size.

## Natural forcing CSV

`natural_forcing_ssp245_ar6_run0_1850-2100.csv` contains one row per year and columns `solar_erf_w_m-2` and `volcanic_erf_w_m-2`. It is derived from the exact AR6, runmode-natural, SSP2-4.5, resampled-100 artifact for run ID 0.

## Figure workflow contract

Every registered workflow declares the frozen script, required external object IDs, output directory, configuration/arguments, and scientific status. Successful release records include command, environment, input SHA-256 values, and output SHA-256 values.
