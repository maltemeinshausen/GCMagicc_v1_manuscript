# Provenance

`manifest.csv` is generated from the frozen release tree. Each row records the original repository or unversioned source root, original path, source revision, destination, SHA-256, byte count, copyright attribution, license, and role.

The two core model directories came from an unversioned data checkout. Their exact file hashes are therefore the immutable source identity. The verified mappings are:

- source snapshot `model_NxlversA5` → public GCMagicc;
- source snapshot `model_NthreeversT1` → public GCMagicc-CE.

GCMagicc-PM and GCMagicc-XS are distributed as separately checksummed model-author
bundles. GCMagicc-PM uses the documented first-four-character grouping into 13
model families; GCMagicc-XS includes its two reduced-predictor checkpoints.

Publication figures generated from ignored data trees and a modified source working tree are identified by the last Git base revision together with SHA-256 snapshots of the relevant generator scripts, compact plotted data, and final artifacts. This avoids attributing uncommitted generated output to a clean commit that did not contain the plotting changes.

The validation selection freezes the main artifact and ten supplementary alternate draws from the same `publication_set_v100/v100` bundle generated on 2026-08-21. The observational-alignment record also states which supplementary panel-index and EDISTO inputs were checked against the refreshed 2026-08-27 bundle.
