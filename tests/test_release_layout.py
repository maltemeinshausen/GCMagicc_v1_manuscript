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
    assert summary["schema"] == "gcmagicc-turkiye-application/v2"
    assert set(summary["variables"]) == {"tas", "pr", "hurs"}
    assert {key: len(value) for key, value in summary["scenario_groups"].items()} == {
        "cmip6_ssps": 8,
        "ndc_ssp2com_current_policy": 10,
        "cmip7": 7,
    }
    assert len(summary["inputs"]) == 75
    assert all(item["plotted"] for item in summary["era5"].values())
    assert all(item["end_year"] == 2025 for item in summary["era5"].values())
    assert all(item["members"] == 20 for values in summary["variables"].values() for item in values.values())


def test_workflow_schematic_records_the_locked_method() -> None:
    summary = json.loads((ROOT / "figures/gcmagicc_workflow/Figure4_GCMagicc_workflow.json").read_text())
    assert summary["schema"] == "gcmagicc-workflow-schematic/v1"
    assert summary["inference"]["requires_gridded_esm_or_reanalysis_fields"] is False
    assert summary["inference"]["requires_prescribed_sst_or_ocean_state"] is False
    assert summary["inference"]["full_predictor_variants"] == ["GCMagicc", "GCMagicc-CE"]
    assert summary["inference"]["reduced_predictor_variants"] == ["GCMagicc-PM", "GCMagicc-XS"]
    assert summary["inference"]["correction"] == {
        "changed_predictors": ["tas_smoothed"],
        "iterative_convergence_loop": False,
        "passes": 2,
        "unchanged_seed": True,
    }
    assert set(summary["outputs"]) == {
        "Figure4_GCMagicc_workflow.pdf",
        "Figure4_GCMagicc_workflow.png",
        "Figure4_GCMagicc_workflow.svg",
    }


def test_validation_audit_matches_frozen_database_snapshot() -> None:
    audit = json.loads((ROOT / "data/derived/validation_metrics/metrics_audit.json").read_text())
    assert audit["database"]["sha256"] == "34feda39f369142300224773efde1c5e80c32133cc4c324d20de83403239fb03"
    assert audit["all_versions"] == {
        "gofnc_records": 4_539_079,
        "ssp245_holdout_records": 1_265_222,
    }
