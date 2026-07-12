# GCMagicc v1.0.1 manuscript companion

This repository is the standalone, publishable code and small-data companion for the GCMagicc v1.0.1 manuscript. It freezes the model inference code, evaluation workflows, figure generators, configuration examples, and provenance needed to audit the paper. It does not import code from neighbouring development repositories at runtime.

## Licensing and attribution

- Code under `src/`, `tests/`, and executable scripts is licensed under Apache License 2.0.
- Data, figures, and documentation are licensed under Creative Commons Attribution 4.0.
- The GCMagicc model, training, and inference implementation is copyright Nicolai Meinshausen.
- Evaluation, analysis, figure, and portal code is copyright Malte Meinshausen and GCMagicc evaluation suite contributors.
- Mixed-origin files and source snapshots are identified in `provenance/manifest.csv` and `docs/licensing_and_authorship.md`.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m gcmagicc_repro smoke
python -m gcmagicc_repro verify
```

Download released external objects, once the immutable release URLs are populated:

```bash
python -m gcmagicc_repro fetch
```

List or reproduce a manuscript figure:

```bash
python -m gcmagicc_repro reproduce --figure turkiye --dry-run
python -m gcmagicc_repro reproduce --figure turkiye
```

The Türkiye workflow is fully standalone: it reads 18 frozen annual-percentile files for `tas`, `pr`, and `hurs`, writes PDF/PNG outputs, and records every input and output checksum in a JSON sidecar. The validation-count audit is also release-native; point it at a local or fetched `metrics.sqlite` without modifying the database:

```bash
python src/gcmagicc_eval/workflows/1110_metrics_database_audit.py \
  /path/to/metrics.sqlite \
  --output data/derived/validation_metrics/metrics_audit.json
```

## Release boundaries

Files larger than 50 MB are never committed. `data/external_data_manifest.json` records their immutable URL, size, SHA-256, public model name, and destination. Entries marked `pending-publication` are release blockers, not guessed URLs. The PM and XS internal-to-public mappings remain unassigned until supported by model-author provenance.

The frozen scientific scripts are preserved under `src/gcmagicc_eval/workflows/`. Some complete figure workflows require the external data listed in the manifest. The small `smoke` command tests deterministic release plumbing and the audited Hargreaves, area-weighting, December-event, bootstrap-seed, and two-pass-correction kernels without downloading multi-gigabyte checkpoints.

See `docs/user_manual.md`, `docs/data_schema.md`, and `provenance/manifest.csv` for details.
