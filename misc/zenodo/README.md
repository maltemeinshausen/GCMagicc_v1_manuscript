# Zenodo external release objects

The ignored ZIPs in this directory and checkpoint archives under `dist/` are
build products. `scripts/stage_zenodo_release.py` verifies and stages the complete
13-file web upload, including `SHA256SUMS.txt`, without duplicating large files
when hard links are supported.

The pre-DOI candidate contains 12 release objects totalling 46,421,771,982
bytes, plus the small checksum ledger. This remains more than 3.5 GB below
Zenodo's 50,000,000,000-byte limit and uses 13 of the permitted 100 files.
The staging script recomputes both limits after the reserved DOI is inserted
and the four bundle ZIPs are rebuilt.

Create one Zenodo **Dataset** record with:

- title: **GCMagicc v1.0.1 manuscript companion — external release objects**;
- version: **1.0.1**;
- creators: **Nicolai Meinshausen** and **Malte Meinshausen**;
- licences: **Apache-2.0** and **CC-BY-4.0**;
- related identifier: `https://github.com/maltemeinshausen/GCMagicc_v1_manuscript`;
- public visibility.

Reserve the DOI before the final archive build. The reserved DOI and Zenodo
record number are inserted into the repository and bundle manifests, after which
the deterministic ZIPs and global checksum file are rebuilt for upload. The
record must remain unpublished until all staged sizes and SHA-256 values have
been checked against `data/external_data_manifest.json`.

Zenodo documentation: [create an upload](https://help.zenodo.org/docs/deposit/create-new-upload/),
[reserve a DOI](https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/), and
[manage files](https://help.zenodo.org/docs/deposit/manage-files/).
