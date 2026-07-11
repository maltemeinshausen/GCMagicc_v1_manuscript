#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the frozen-source provenance manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def describe(path: Path) -> dict[str, str]:
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith("data/derived/drought_common_protocol/"):
        return {
            "source_repository": "GCMagicc_v1_manuscript",
            "source_revision": "release-generated common protocol v1",
            "source_path": "src/gcmagicc_eval/workflows/1090_drought_common_protocol.py",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "drought common-protocol derived data and run manifest",
        }
    if rel.startswith("figures/drought_common_protocol/"):
        return {
            "source_repository": "GCMagicc_v1_manuscript",
            "source_revision": "release-generated common protocol v1",
            "source_path": "src/gcmagicc_eval/workflows/1090_drought_common_protocol.py",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "CC-BY-4.0",
            "role": "drought common-protocol manuscript figure",
        }
    if rel.startswith("src/gcmagicc_model/gcmagicc/"):
        name = rel.rsplit("/", 1)[-1]
        return {
            "source_repository": "unversioned:gcm_firefly_data",
            "source_revision": "content-hash snapshot 2026-07-11",
            "source_path": f"model_NxlversA5/{name}",
            "copyright": "Nicolai Meinshausen",
            "license": "Apache-2.0",
            "role": "GCMagicc core model/inference",
        }
    if rel.startswith("src/gcmagicc_model/gcmagicc_ce/"):
        name = rel.rsplit("/", 1)[-1]
        return {
            "source_repository": "unversioned:gcm_firefly_data",
            "source_revision": "content-hash snapshot 2026-07-11",
            "source_path": f"model_NthreeversT1/{name}",
            "copyright": "Nicolai Meinshausen",
            "license": "Apache-2.0",
            "role": "GCMagicc-CE core model/inference",
        }
    if rel.startswith("src/gcmagicc_eval/helpers/"):
        helper_rel = rel.removeprefix("src/gcmagicc_eval/helpers/")
        if helper_rel == "cmip6_operations.py":
            return {
                "source_repository": "cmipcruncher_firefly",
                "source_revision": "e310d238536fa136eacb5e7ace118b1ec2cc0837",
                "source_path": "src/cmip6cruncher/operations.py",
                "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
                "license": "Apache-2.0",
                "role": "training-data smoothing helper",
            }
        return {
            "source_repository": "gcmmagicc",
            "source_revision": "bc0a782e019d4d04bf60fad676ac46758145fae4",
            "source_path": f"scr/{helper_rel}",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "Apache-2.0",
            "role": "evaluation helper",
        }
    if rel.startswith("src/gcmagicc_eval/recipes/"):
        name = rel.rsplit("/", 1)[-1]
        return {
            "source_repository": "gcmmagicc",
            "source_revision": "bc0a782e019d4d04bf60fad676ac46758145fae4",
            "source_path": f"notebooks/recipes/{name}",
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "Apache-2.0",
            "role": "evaluation PET and SPEI recipe adapted for standalone import",
        }
    if rel.startswith("src/gcmagicc_eval/workflows/"):
        name = rel.rsplit("/", 1)[-1]
        if name == "1090_drought_common_protocol.py":
            return {
                "source_repository": "GCMagicc_v1_manuscript",
                "source_revision": "release-native implementation 2026-07-11",
                "source_path": rel,
                "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
                "license": "Apache-2.0",
                "role": "corrected drought and three-SMILE common-protocol workflow",
            }
        if name.startswith(("320_", "321_", "331_")):
            repo, revision = "gcmagicc_ensemble_runner", "fabeb92623a82d7adc0527ded00177ba09f1d2a8"
            source = f"notebooks/{name}"
        elif name == "220_data_processing.py":
            repo, revision = "cmipcruncher_firefly", "e310d238536fa136eacb5e7ace118b1ec2cc0837"
            source = f"notebooks/{name}"
        else:
            repo, revision = "gcmmagicc", "bc0a782e019d4d04bf60fad676ac46758145fae4"
            source = f"notebooks/{name}"
        return {
            "source_repository": repo,
            "source_revision": revision,
            "source_path": source,
            "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
            "license": "Apache-2.0",
            "role": "evaluation/figure workflow",
        }
    return {
        "source_repository": "2025magicc",
        "source_revision": "67e3609782c400bff42a04e384fb7fd0ec12bf9a",
        "source_path": "derived from output_resampled_100/AR6/runmode_natural/ssp245_ar6_natural.parquet run_id=0",
        "copyright": "Malte Meinshausen and GCMagicc evaluation suite contributors",
        "license": "CC-BY-4.0",
        "role": "natural-forcing predictor data",
    }


def main() -> None:
    files = sorted((ROOT / "src/gcmagicc_model").rglob("*.py"))
    files += sorted((ROOT / "src/gcmagicc_model").rglob("*.pkl"))
    files += sorted((ROOT / "src/gcmagicc_model").rglob("*.pt"))
    files += sorted((ROOT / "src/gcmagicc_model").rglob("*.csv"))
    files += sorted((ROOT / "src/gcmagicc_eval").rglob("*.py"))
    files += [ROOT / "data/natural_forcing_ssp245_ar6_run0_1850-2100.csv"]
    files += sorted((ROOT / "data/derived/drought_common_protocol").glob("*"))
    files += sorted((ROOT / "figures/drought_common_protocol").glob("*"))
    rows = []
    for path in sorted(set(files)):
        meta = describe(path)
        rows.append(
            {
                **meta,
                "destination": path.relative_to(ROOT).as_posix(),
                "sha256": digest(path),
                "bytes": str(path.stat().st_size),
            }
        )
    fields = ["source_repository", "source_revision", "source_path", "destination", "sha256", "bytes", "copyright", "license", "role"]
    with (ROOT / "provenance/manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} provenance rows")


if __name__ == "__main__":
    main()
