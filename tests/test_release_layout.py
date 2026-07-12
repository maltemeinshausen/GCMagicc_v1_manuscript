# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_external_manifest_never_guesses_pm_or_xs_mapping() -> None:
    objects = {item["public_model"]: item for item in json.loads((ROOT / "data/external_data_manifest.json").read_text())["objects"]}
    assert objects["GCMagicc-PM"]["status"] == "pending-model-author-provenance"
    assert objects["GCMagicc-XS"]["status"] == "pending-model-author-provenance"


def test_repository_has_no_file_over_50_mb() -> None:
    too_large = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts and path.stat().st_size > 50 * 1024 * 1024]
    assert too_large == []


def test_turkiye_release_bundle_is_complete() -> None:
    summary = json.loads(
        (ROOT / "figures/turkiye_regional_application/Figure3_Turkiye_tas_pr_hurs.json").read_text()
    )
    assert summary["baseline"] == [1995, 2014]
    assert summary["future"] == [2081, 2100]
    assert set(summary["variables"]) == {"tas", "pr", "hurs"}
    assert len(summary["inputs"]) == 18
    assert all(item["members"] == 20 for values in summary["variables"].values() for item in values.values())


def test_validation_audit_matches_frozen_database_snapshot() -> None:
    audit = json.loads((ROOT / "data/derived/validation_metrics/metrics_audit.json").read_text())
    assert audit["database"]["sha256"] == "34feda39f369142300224773efde1c5e80c32133cc4c324d20de83403239fb03"
    assert audit["all_versions"] == {
        "gofnc_records": 4_539_079,
        "ssp245_holdout_records": 1_265_222,
    }
