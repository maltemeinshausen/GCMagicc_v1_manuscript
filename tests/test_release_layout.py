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


def test_drought_hybrid_figure_restores_context_without_old_results() -> None:
    sidecar = json.loads(
        (ROOT / "figures/drought_common_protocol/Figure5_DroughtAttribution_IRN_hybrid_common_protocol.json").read_text()
    )
    assert sidecar["schema"] == "gcmagicc-drought-hybrid-figure/v1"
    assert list(sidecar["panels"]) == list("abcdefg")
    assert sidecar["method"]["baseline"] == [1991, 2010]
    assert sidecar["method"]["aggregation"] == "area-weighted mean of grid-cell-standardized December SPEI-48"
    assert sidecar["design_provenance"]["scientific_results"].startswith("corrected common-protocol")
    assert len(sidecar["inputs"]) == 4
    assert set(sidecar["outputs"]) == {
        "Figure5_DroughtAttribution_IRN_hybrid_common_protocol.pdf",
        "Figure5_DroughtAttribution_IRN_hybrid_common_protocol.png",
    }

    map_artifact = json.loads(
        (ROOT / "data/derived/drought_common_protocol/era5_irn_penman_monteith_spei48_map.json").read_text()
    )
    assert map_artifact["schema"] == "gcmagicc-era5-irn-event-map/v1"
    assert map_artifact["event"] == "December 2025"
    assert map_artifact["baseline"] == [1991, 2010]
    assert any(boundary["iso3"] == "IRN" for boundary in map_artifact["boundaries"])
    assert map_artifact["source"]["sha256"] == "2c6fd00ebd257794dd1bbe17be6c5a0e24b4caa7e8234949108e1ac01dcbcef0"

    cmip6_sidecar = json.loads(
        (ROOT / "data/derived/drought_common_protocol/cmip6_irn_penman_monteith_spei48_sidecar.json").read_text()
    )
    assert cmip6_sidecar["schema"] == "gcmagicc-cmip6-drought-sidecar/v1"
    assert len(cmip6_sidecar["factual"]) == 54
    assert len(cmip6_sidecar["natural"]) == 9
    assert "not used in corrected attribution statistics" in cmip6_sidecar["purpose"]
    assert cmip6_sidecar["source"]["sha256"] == "c9e76841922b6b552bc82c3fae70787269121d704fef09cbf47e7ae992a72b3b"


def test_validation_audit_matches_frozen_database_snapshot() -> None:
    audit = json.loads((ROOT / "data/derived/validation_metrics/metrics_audit.json").read_text())
    assert audit["database"]["sha256"] == "34feda39f369142300224773efde1c5e80c32133cc4c324d20de83403239fb03"
    assert audit["all_versions"] == {
        "gofnc_records": 4_539_079,
        "ssp245_holdout_records": 1_265_222,
    }
