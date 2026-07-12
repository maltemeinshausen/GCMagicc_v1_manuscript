<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Corrected Iran drought common-protocol outputs

These are the release outputs of
`src/gcmagicc_eval/workflows/1090_drought_common_protocol.py` for the December
2025 Iranian SPEI-48 event.

- `*_GCMagicc_v100_and_SMILEs.*` contains the 100-member GCMagicc v1.0.1
  factual and natural-only ensembles and the eligible conventional CMIP6
  ensembles: CanESM5 (50 factual, 50 natural-only), MIROC6 (50, 50), and
  GISS-E2-1-G (21, 20). The comparison window is 1995--2014, the common
  period available across those inputs.
- `*_GCMagicc_CE_v101.*` contains the 100-member GCMagicc-CE factual and
  natural-only sensitivity run.
- `summary` files contain probabilities, probability ratios, percentile
  intervals, and one-sided bounds.
- `series` files contain the annual December regional series used by the
  bootstrap.
- `manifest` files record the locked protocol, source-file inventory, output
  hashes, and portable input-root environment contract.
- `era5_irn_penman_monteith_spei48_map.json` is the small standalone map input
  for the main synthesis figure. It freezes the corrected December 2025 map,
  area-weighted ERA5 series, clipped Natural Earth boundaries, and exact source
  filename, byte count, and SHA-256.

The primary lane uses the 1991--2010 fitting baseline, ERA5-adjusted `rsds`,
and an area-weighted mean of grid-cell-standardized SPEI. The files also
contain the 1981--2010 baseline, unadjusted-`rsds`, and aggregated regional
water-balance sensitivities. All uncertainty calculations use 10,000
fixed-seed hierarchical bootstrap replicates over ensemble members and moving
five-year blocks. When no natural-only events are observed, the finite
one-sided probability bound is based on the effective member-by-five-year-
block trial count; the infinite empirical point ratio is retained separately.

Copyright Malte Meinshausen and GCMagicc evaluation suite contributors.
Licensed under CC BY 4.0.
