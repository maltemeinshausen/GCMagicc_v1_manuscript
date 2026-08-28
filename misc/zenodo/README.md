# Zenodo external release objects

The ignored ZIPs in this directory and checkpoint archives under `dist/` are
build products. `scripts/stage_zenodo_release.py` verifies and stages the complete
12-file web upload, including `SHA256SUMS.txt`, without duplicating large files
when hard links are supported.

The candidate for reserved DOI `10.5281/zenodo.22139683` contains 11 deposited
release objects totalling 44,893,148,613 bytes, plus the small checksum ledger.
This remains more than 5.1 GB below Zenodo's 50,000,000,000-byte limit and uses
12 of the permitted 100 files. The checksum-recorded historical 1.528-GB XS
figure-source CSV is explicitly not redistributed in v1.0.1; the smaller July 13
regenerated CSV and its member-draw archive remain part of the deposit.
The staging script recomputes both limits after the reserved DOI is inserted
and the four bundle ZIPs are rebuilt.

Create one Zenodo **Dataset** record with:

- title: **GCMagicc v1.0.1 manuscript companion — external release objects**;
- publication date: **2026-07-11**;
- version: **1.0.1**;
- creators: **Nicolai Meinshausen** and **Malte Meinshausen**;
- licences: **Apache-2.0** and **CC-BY-4.0**;
- language: **English**;
- related identifier: `https://github.com/maltemeinshausen/GCMagicc_v1_manuscript`
  with relation **Is supplement to**;
- public visibility.

Use this description:

> External release objects for the GCMagicc v1.0.1 manuscript companion. The
> deposit contains deterministic archives of the trained GCMagicc and
> GCMagicc-CE checkpoints; sanitized NIC-AER, NIC-PM, NIC-RES, and NIC-XS
> reproduction bundles; three validation-database snapshots; and the July 13
> GCMagicc-XS regenerated monthly summary and member draws. Repository-sized
> code, figures, prepared data, provenance, and per-file manifests are maintained
> at the related GitHub repository. The checksum-recorded historical 1.528-GB
> GCMagicc-XS figure-source CSV and the 4.6-TB normalized HDF5 input collection
> are not redistributed in v1.0.1. `SHA256SUMS.txt` records the exact identity of
> every deposited file.

Suggested keywords: `climate emulator`, `GCMagicc`, `CMIP6`, `climate model
emulation`, `machine learning`, `validation`, `reproducibility`.

Upload every file in `dist/zenodo/` without renaming it. The expected upload is
12 files and 44,893,149,702 bytes including `SHA256SUMS.txt`. Wait until no file
is marked pending, make `SHA256SUMS.txt` the default preview, save the draft, and
preview it before publishing.

The reserved DOI and Zenodo record number are inserted into the repository and
bundle manifests before the deterministic ZIPs and global checksum file are
rebuilt for upload. The
record must remain unpublished until all staged sizes and SHA-256 values have
been checked against `data/external_data_manifest.json`.

Zenodo documentation: [create an upload](https://help.zenodo.org/docs/deposit/create-new-upload/),
[reserve a DOI](https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/), and
[manage files](https://help.zenodo.org/docs/deposit/manage-files/).
