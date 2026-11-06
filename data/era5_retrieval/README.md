# ERA5 retrieval recipe

GCMagicc v1.0.1 is trained and evaluated against the ERA5 reanalysis (Hersbach et al., 2020), cited in the
manuscript as Copernicus Climate Change Service (C3S): *ERA5 monthly averaged data on single levels from 1940 to
present*, https://doi.org/10.24381/cds.f17050d7. The ERA5 fields are not redistributed in this release. They are
retrieved from the Copernicus Climate Data Store (CDS) with the script in this folder.

## Script

`040_ERA5_APIdownload.py` is a verbatim copy of `notebooks/040_ERA5_APIdownload.py` from the
`cmipcruncher_firefly` data-preparation repository at revision `44b57e27eca37d8309d4705b8007a6a2394fe41d`
(SHA-256 `f3743eb5dc4e27bdba5e6038a537dd71ff67fa2677dc5c04e56ebd9b720235e9`). It uses the official `cdsapi`
client. Credentials are read by `cdsapi` from the user's own `~/.cdsapirc` and are not part of this release.

It retrieves, in netCDF format on the native 0.25° × 0.25° grid (1440 × 721 points):

- **Monthly means** (`reanalysis-era5-single-levels-monthly-means`, product type `monthly_averaged_reanalysis`),
  1940 to present, one file per variable and decade, for 13 variables:
  `10m_u_component_of_wind`, `10m_v_component_of_wind`, `10m_wind_speed`, `2m_dewpoint_temperature`,
  `2m_temperature`, `mean_sea_level_pressure`, `skin_temperature`, `surface_solar_radiation_downwards`,
  `toa_incident_solar_radiation`, `top_net_solar_radiation`, `top_net_thermal_radiation`, `total_cloud_cover`,
  `total_precipitation`.
- **Daily maximum and minimum 2 m temperature** computed by the CDS from hourly ERA5 (UTC days), 1940 to present,
  one file per statistic and year. These are the basis of the monthly `tasmax` and `tasmin` fields.

The manuscript uses 1940–2024. Section 3.1 of the manuscript describes how these fields are converted
to the ten target variables.

## Volume

| Part | Volume |
|---|---|
| Monthly means, 13 variables, 1940–2025 | 18.6 GB as retrieved (compressed netCDF); ≈ 56 GB as uncompressed float32 |
| Daily max/min 2 m temperature, 1940–2024 | ≈ 258 GB as uncompressed float32 (2 × 31,047 daily global fields) |
| **Total ERA5 input** | **≈ 314 GB uncompressed** |

## Licence

ERA5 is distributed through the CDS under the CC BY 4.0 licence. The script in this folder is licensed under
Apache-2.0 (see `.reuse/dep5`).
