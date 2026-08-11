# Ada data provenance audit

This audit records the publication-relevant material found in the Ada `gcmmagicc/data` tree and the authoritative selections made from the current manuscript. Paths are normalized repository-relative paths; large archives, checkpoints, databases, caches, and raw model output are not copied into this repository.

## Promoted material

- **Emergent constraints:** `plots.tar` and `plotsT1.tar` are byte-identical 559,450,108-byte archives with SHA-256 `43923f2cf9563ec19273ad98a7422737448749ed7f8d6bddb92e5e590f8ef36c`. The release contains the 799-row trend table covering 19 models, the six ERA5-conditioned tables with 30/39/48/15/12/57 rows, Nicolai Meinshausen's original sampler, the completed seven-quantile table, a normalized composite, and the canonical PDF/PNG. Prepared-data-to-figure reproduction is complete. Generation of the prepared trends from raw CMIP6, ERA5, and GCMagicc-PM checkpoints is unavailable and unverified.
- **Resolution sensitivity:** the T1 material is identified as GCMagicc-CE. Normalized metadata and ten time-index-1022 synthesis PDFs are frozen. Large per-map JSON and checkpoints remain external under `gcmagicc-ce-checkpoints`.
- **Aerosol sensitivity:** the T1 material is identified as GCMagicc-CE. The diagnostic is the full-forcing minus zero-aerosol-ERF change in diurnal temperature range (`tasmax - tasmin`) in K at `nside=64`, using 100 stochastic samples. Eleven compact maps, the combined matrix, and normalized source scripts are frozen. The period is provisionally 2015--2024.
- **GCMagicc-XS:** the ten manuscript PNGs are frozen. The 1,528,623,568-byte monthly CSV remains external with its checksum. A deterministic 205,407-row plotted-point sample makes compact replotting release-native; full raw recomputation continues to require `gcmagicc-xs-bundle`.
- **Validation and observational alignment:** the release freezes the current main validation artifact `validation_diagnostics_main_v100_20260810_0845` and supplementary alternate draws `s01`--`s10` from the same August publication set, plus three SSP2-4.5 SCOREEDISTC panels and three observation-referenced EDISTO panels. They are linked to the existing 11.9-GB metrics database audit. The observational-alignment source is the authoritative Gus August bundle `observational_alignment_v100_20260807_110344`, replacing the July bundle and the older Ada output. Its panel-c model index is unchanged, and its panel diagnostics and row-5 match-map inputs are unchanged, so the existing three-part supplementary EDISTO figure remains current. Because both current main figures were generated from a modified `main-gus` working tree, their provenance records anchor the Git base revision and exact generator-script hashes as well as the final artifact hashes.
- **SPEI sensitivity:** provenance is anchored to Ada `main` revision `16c9d72`, which integrates Gus revisions `8a3bb76` and `34581af` while retaining the Ada pathway default and plotting semantics. The semantic Iran PDF and normalized compact tables are frozen.

## Recorded without promotion

- **Crop failure:** excluded because the authoritative manuscript no longer uses it.
- **Separate ten-variable scenario projections:** audit-only because no corresponding figure is active in the authoritative manuscript.
- **Older A5 archive:** inventory-only because its model identity conflicts with the authoritative T1 GCMagicc-CE selection.
- **Duplicate archives, caches, database copies, and operational intermediates:** inventory-only. They are mutable or redundant and are not suitable release artifacts.

## Attribution and licensing

The archived emergent-constraint data and original sampling script are attributed to Nicolai Meinshausen under CC BY 4.0. The assembled evaluation figure and release workflow are attributed to Malte Meinshausen and the GCMagicc evaluation suite contributors. Code remains Apache-2.0; data, figures, provenance records, and documentation remain CC BY 4.0.
